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
