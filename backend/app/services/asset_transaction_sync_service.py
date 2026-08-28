"""Ingest provider-reported trades into an asset's ledger.

Pluggy embeds a holding's buy/sell history in its /investments payload, and
other providers may do the same. Rows go into `asset_transactions` tagged
with the asset's own `source` so a synced brokerage position gets the same
ledger a manual or imported one has — preço médio, realized gains, and
something to export at tax time.

Kept out of `connection_service` on purpose: that module is already the most
intricate file in the backend, and this logic is worth testing without a
provider and a bank connection in the way.
"""

import uuid
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.asset import Asset
from app.models.asset_transaction import AssetTransaction
from app.providers.base import HoldingData
from app.services import asset_transaction_service

# `asset_transactions.quantity` is Numeric(18, 6): anything below this is
# rounding noise, not a missing trade.
_QUANTITY_TOLERANCE = Decimal("0.000001")


async def sync_holding_ledger(
    session: AsyncSession, asset: Asset, holding: HoldingData
) -> None:
    """Upsert the provider's trades onto this asset's ledger.

    Does not commit: the caller (`connection_service._sync_holdings`) owns the
    transaction boundary for the whole sync.
    """
    if not holding.transactions:
        # The common case by far — no query, no flush.
        return

    existing = await _existing_by_external_id(session, asset)

    for row in holding.transactions:
        current = existing.get(row.external_id)
        if current is None:
            session.add(
                AssetTransaction(
                    asset_id=asset.id,
                    workspace_id=asset.workspace_id,
                    kind=row.kind,
                    quantity=row.quantity,
                    price=row.price,
                    fee=row.fee,
                    date=row.date,
                    source=asset.source,
                    external_id=row.external_id,
                )
            )
            continue
        # The provider restated a trade it already reported. Correct it in
        # place rather than adding a second row for the same event.
        current.kind = row.kind
        current.quantity = row.quantity
        current.price = row.price
        current.fee = row.fee
        current.date = row.date

    # Rows that vanished from the payload are deliberately left alone.
    # Brokerages return a moving window of history, so absence means the trade
    # aged out, not that it was reversed — deleting on absence would drain the
    # ledger over time.
    await session.flush()

    if await _should_promote(session, asset, holding):
        # The ledger accounts for the whole position, so derive it: units,
        # preço médio, cost basis, purchase date and realized gain all come
        # from the trades, exactly as they do for a manual holding.
        await asset_transaction_service.recompute_and_cache(session, asset)
    elif asset.sell_date is None:
        # Non-promotion means the provider's numbers stand, and
        # average_price/realized_gain are supposed to be the observability
        # marker for "the ledger is authoritative" — so clear them here.
        # Skipped when the user has marked the asset sold: that asset may
        # have been legitimately promoted before the sell_date was set, and
        # its historical average_price/realized_gain belong to the user's
        # own record, not to this sync.
        asset.average_price = None
        asset.realized_gain = None


async def _existing_by_external_id(
    session: AsyncSession, asset: Asset
) -> dict[str, AssetTransaction]:
    """This service's own rows, keyed by provider id.

    Scoped to this asset's own source so a manual or imported row that
    happens to carry the same `external_id` is never overwritten.
    """
    result = await session.execute(
        select(AssetTransaction).where(
            AssetTransaction.asset_id == asset.id,
            AssetTransaction.source == asset.source,
        )
    )
    return {
        tx.external_id: tx
        for tx in result.scalars().all()
        if tx.external_id
    }


async def _should_promote(
    session: AsyncSession, asset: Asset, holding: HoldingData
) -> bool:
    """Is this ledger complete enough to own the position?

    Only when its derived quantity matches the one the provider reports.
    Brokerages return a moving window of history, so a ledger that covers the
    last six months of a five-year position would understate the holding —
    better to keep the provider's numbers and show the trades as history.
    """
    if asset.sell_date is not None:
        # Covers both a provider-reported closure and a sell_date the user
        # set themselves. recompute_and_cache clears sell_date/sell_price
        # whenever derived units are positive, which would silently re-open
        # a position the user (or the provider) marked closed.
        return False
    if holding.is_withdrawn:
        # Belt and suspenders: the real sync path (_sync_holdings) already
        # `continue`s past a withdrawn holding before this is ever called,
        # and it sets sell_date first, which the check above now also
        # catches. Kept so this function is safe to call directly.
        return False
    if holding.quantity is None:
        # No reference figure (typical of fixed income) — completeness can't
        # be established, so it isn't claimed.
        return False

    rows = await _load_ledger(session, asset.id)
    derived = _derived_units(rows)
    if derived <= 0:
        # Trades that net to nothing against a position the provider still
        # reports means history is missing; promoting would archive a live
        # holding.
        return False
    return abs(derived - holding.quantity) <= _QUANTITY_TOLERANCE


def _derived_units(rows: list[AssetTransaction]) -> Decimal:
    """Net quantity the ledger accounts for, across every source.

    Manual and imported rows count: the question is whether the asset's whole
    ledger explains the position, not whether this provider's slice does.
    """
    units = Decimal("0")
    for row in rows:
        quantity = Decimal(str(row.quantity or 0))
        units += quantity if row.kind == "buy" else -quantity
    return units


async def _load_ledger(
    session: AsyncSession, asset_id: uuid.UUID
) -> list[AssetTransaction]:
    result = await session.execute(
        select(AssetTransaction).where(AssetTransaction.asset_id == asset_id)
    )
    return list(result.scalars().all())
