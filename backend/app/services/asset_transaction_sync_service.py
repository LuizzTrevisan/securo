"""Ingest provider-reported trades into an asset's ledger.

Pluggy embeds a holding's buy/sell history in its /investments payload. Those
rows go into `asset_transactions` with source="pluggy" so a synced brokerage
position gets the same ledger a manual or imported one has — preço médio,
realized gains, and something to export at tax time.

Kept out of `connection_service` on purpose: that module is already the most
intricate file in the backend, and this logic is worth testing without a
provider and a bank connection in the way.
"""

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.asset import Asset
from app.models.asset_transaction import AssetTransaction
from app.providers.base import HoldingData

logger = logging.getLogger(__name__)

_SOURCE = "pluggy"


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

    existing = await _existing_by_external_id(session, asset.id)

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
                    source=_SOURCE,
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


async def _existing_by_external_id(
    session: AsyncSession, asset_id: uuid.UUID
) -> dict[str, AssetTransaction]:
    """This service's own rows, keyed by provider id.

    Scoped to source="pluggy" so a manual or imported row that happens to
    carry the same `external_id` is never overwritten.
    """
    result = await session.execute(
        select(AssetTransaction).where(
            AssetTransaction.asset_id == asset_id,
            AssetTransaction.source == _SOURCE,
        )
    )
    return {
        tx.external_id: tx
        for tx in result.scalars().all()
        if tx.external_id
    }
