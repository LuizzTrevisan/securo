"""Tests for ingesting provider-reported trades into the asset ledger.

Storage behaviour: insert once, never duplicate, refresh restated rows, and
never delete a row just because it aged out of the provider's window.
"""

from datetime import date
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import select

from app.models.asset import Asset
from app.models.asset_transaction import AssetTransaction
from app.models.user import User
from app.providers.base import HoldingData, HoldingTransactionData
from app.services import asset_transaction_sync_service


def _holding(transactions, quantity="53") -> HoldingData:
    return HoldingData(
        external_id="inv-1",
        name="MXRF11",
        currency="BRL",
        current_value=Decimal("400"),
        quantity=Decimal(quantity) if quantity is not None else None,
        transactions=transactions,
    )


def _tx(external_id="tx-1", kind="buy", quantity="5", price="8.22",
        fee="0.13", d=date(2026, 7, 27)) -> HoldingTransactionData:
    return HoldingTransactionData(
        external_id=external_id, kind=kind, quantity=Decimal(quantity),
        price=Decimal(price), fee=Decimal(fee), date=d,
    )


@pytest_asyncio.fixture
async def synced_asset(session, test_user: User, test_workspace) -> Asset:
    asset = Asset(
        user_id=test_user.id,
        workspace_id=test_workspace.id,
        name="MXRF11",
        type="investment",
        currency="BRL",
        source="pluggy",
        external_id="inv-1",
        units=Decimal("53"),
        valuation_method="manual",
    )
    session.add(asset)
    await session.flush()
    return asset


async def _ledger(session, asset: Asset) -> list[AssetTransaction]:
    result = await session.execute(
        select(AssetTransaction)
        .where(AssetTransaction.asset_id == asset.id)
        .order_by(AssetTransaction.date)
    )
    return list(result.scalars().all())


@pytest.mark.asyncio
async def test_first_sync_inserts_rows(session, synced_asset):
    await asset_transaction_sync_service.sync_holding_ledger(
        session, synced_asset, _holding([_tx()])
    )
    rows = await _ledger(session, synced_asset)
    assert len(rows) == 1
    assert rows[0].source == "pluggy"
    assert rows[0].external_id == "tx-1"
    assert rows[0].kind == "buy"
    assert rows[0].quantity == Decimal("5")
    assert rows[0].price == Decimal("8.22")
    assert rows[0].fee == Decimal("0.13")
    assert rows[0].date == date(2026, 7, 27)
    assert rows[0].workspace_id == synced_asset.workspace_id
    assert rows[0].import_id is None


@pytest.mark.asyncio
async def test_second_sync_does_not_duplicate(session, synced_asset):
    holding = _holding([_tx()])
    await asset_transaction_sync_service.sync_holding_ledger(session, synced_asset, holding)
    await asset_transaction_sync_service.sync_holding_ledger(session, synced_asset, holding)
    assert len(await _ledger(session, synced_asset)) == 1


@pytest.mark.asyncio
async def test_restated_row_is_refreshed_in_place(session, synced_asset):
    await asset_transaction_sync_service.sync_holding_ledger(
        session, synced_asset, _holding([_tx(price="8.22", fee="0.13")])
    )
    await asset_transaction_sync_service.sync_holding_ledger(
        session, synced_asset,
        _holding([_tx(price="8.30", fee="0.15", quantity="6", d=date(2026, 7, 28))]),
    )
    rows = await _ledger(session, synced_asset)
    assert len(rows) == 1
    assert rows[0].price == Decimal("8.30")
    assert rows[0].fee == Decimal("0.15")
    assert rows[0].quantity == Decimal("6")
    assert rows[0].date == date(2026, 7, 28)


@pytest.mark.asyncio
async def test_row_absent_from_payload_survives(session, synced_asset):
    """Brokerages return a moving window: absence means aged out, not reversed."""
    await asset_transaction_sync_service.sync_holding_ledger(
        session, synced_asset, _holding([_tx("tx-old", d=date(2025, 1, 5))])
    )
    await asset_transaction_sync_service.sync_holding_ledger(
        session, synced_asset, _holding([_tx("tx-new")])
    )
    rows = await _ledger(session, synced_asset)
    assert {r.external_id for r in rows} == {"tx-old", "tx-new"}


@pytest.mark.asyncio
async def test_no_transactions_writes_nothing(session, synced_asset):
    await asset_transaction_sync_service.sync_holding_ledger(
        session, synced_asset, _holding([])
    )
    assert await _ledger(session, synced_asset) == []


@pytest.mark.asyncio
async def test_manual_rows_on_the_same_asset_are_left_alone(session, synced_asset):
    """A row this service didn't write is never touched by external_id
    collision — matching is scoped to source="pluggy"."""
    session.add(AssetTransaction(
        asset_id=synced_asset.id, workspace_id=synced_asset.workspace_id,
        kind="buy", quantity=Decimal("10"), price=Decimal("7.00"),
        fee=Decimal("0"), date=date(2026, 1, 5), source="manual",
        external_id="tx-1",
    ))
    await session.flush()
    await asset_transaction_sync_service.sync_holding_ledger(
        session, synced_asset, _holding([_tx("tx-1")])
    )
    rows = await _ledger(session, synced_asset)
    assert len(rows) == 2
    manual = next(r for r in rows if r.source == "manual")
    assert manual.price == Decimal("7.00")


# ---------------------------------------------------------------------------
# Promotion: when the ledger becomes the source of truth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_ledger_drives_the_position(session, synced_asset):
    """35 units bought across two trades, and the provider agrees → the
    position is derived: preço médio, cost basis and purchase date all come
    from the ledger."""
    synced_asset.units = Decimal("35")
    holding = _holding(
        [
            _tx("tx-1", quantity="5", price="8.22", fee="0.13", d=date(2026, 7, 27)),
            _tx("tx-2", quantity="30", price="8.22", fee="0.13", d=date(2026, 7, 27)),
        ],
        quantity="35",
    )
    await asset_transaction_sync_service.sync_holding_ledger(session, synced_asset, holding)

    assert synced_asset.units == Decimal("35")
    # (5*8.22 + 0.13) + (30*8.22 + 0.13) = 287.96
    assert synced_asset.purchase_price == Decimal("287.96")
    assert synced_asset.average_price.quantize(Decimal("0.0001")) == Decimal("8.2274")
    assert synced_asset.purchase_date == date(2026, 7, 27)


@pytest.mark.asyncio
async def test_partial_history_leaves_provider_numbers_alone(session, synced_asset):
    """The broker returned one recent buy but the position is 53 units — the
    ledger is stored, the numbers are not touched."""
    synced_asset.units = Decimal("53")
    synced_asset.purchase_price = Decimal("380.00")
    await asset_transaction_sync_service.sync_holding_ledger(
        session, synced_asset, _holding([_tx(quantity="5")], quantity="53")
    )
    assert len(await _ledger(session, synced_asset)) == 1
    assert synced_asset.units == Decimal("53")
    assert synced_asset.purchase_price == Decimal("380.00")
    assert synced_asset.average_price is None


@pytest.mark.asyncio
async def test_unknown_provider_quantity_blocks_promotion(session, synced_asset):
    """Fixed income often reports no quantity — with no reference figure,
    completeness can't be established."""
    synced_asset.units = Decimal("53")
    await asset_transaction_sync_service.sync_holding_ledger(
        session, synced_asset, _holding([_tx(quantity="5")], quantity=None)
    )
    assert len(await _ledger(session, synced_asset)) == 1
    assert synced_asset.average_price is None
    assert synced_asset.units == Decimal("53")


@pytest.mark.asyncio
async def test_withdrawn_holding_is_not_promoted(session, synced_asset):
    """recompute_and_cache clears sell_date whenever derived units are
    positive, which would undo the redemption marker the sync sets."""
    synced_asset.units = Decimal("5")
    synced_asset.sell_date = date(2026, 8, 1)
    holding = _holding([_tx(quantity="5")], quantity="5")
    holding.is_withdrawn = True
    await asset_transaction_sync_service.sync_holding_ledger(session, synced_asset, holding)
    assert len(await _ledger(session, synced_asset)) == 1
    assert synced_asset.sell_date == date(2026, 8, 1)
    assert synced_asset.average_price is None


@pytest.mark.asyncio
async def test_ledger_netting_to_zero_against_a_live_position_is_not_promoted(
    session, synced_asset
):
    """A buy and an offsetting sell inside the window, against a position the
    provider still reports, means history is missing — promoting would archive
    a live holding."""
    synced_asset.units = Decimal("53")
    await asset_transaction_sync_service.sync_holding_ledger(
        session, synced_asset,
        _holding([
            _tx("tx-1", kind="buy", quantity="5", d=date(2026, 7, 27)),
            _tx("tx-2", kind="sell", quantity="5", d=date(2026, 8, 3)),
        ], quantity="53"),
    )
    assert synced_asset.units == Decimal("53")
    assert synced_asset.sell_date is None


@pytest.mark.asyncio
async def test_fractional_quantities_within_tolerance_promote(session, synced_asset):
    """Fund quantities carry many decimals; compare at the ledger's scale
    rather than demanding bit-exact equality."""
    synced_asset.units = Decimal("10.5")
    await asset_transaction_sync_service.sync_holding_ledger(
        session, synced_asset,
        _holding([_tx(quantity="10.5000004", price="8.00")], quantity="10.5"),
    )
    assert synced_asset.average_price is not None


@pytest.mark.asyncio
async def test_manual_rows_count_toward_completeness(session, synced_asset):
    """The test is over the whole ledger, not just the provider's slice."""
    synced_asset.units = Decimal("15")
    session.add(AssetTransaction(
        asset_id=synced_asset.id, workspace_id=synced_asset.workspace_id,
        kind="buy", quantity=Decimal("10"), price=Decimal("7.00"),
        fee=Decimal("0"), date=date(2026, 1, 5), source="manual",
    ))
    await session.flush()
    await asset_transaction_sync_service.sync_holding_ledger(
        session, synced_asset, _holding([_tx(quantity="5")], quantity="15")
    )
    assert synced_asset.units == Decimal("15")
    assert synced_asset.average_price is not None
