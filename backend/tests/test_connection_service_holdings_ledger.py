"""End-to-end: a holdings sync populates the asset ledger from the payload
Pluggy already returns, and a second sync is idempotent.
"""

from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from app.models.asset import Asset
from app.models.asset_transaction import AssetTransaction
from app.providers.base import HoldingData, HoldingTransactionData
from app.services import connection_service


def _holding_with_trades() -> HoldingData:
    return HoldingData(
        external_id="inv-1",
        name="MXRF11",
        currency="BRL",
        current_value=Decimal("287.00"),
        quantity=Decimal("35"),
        unit_price=Decimal("8.20"),
        transactions=[
            HoldingTransactionData(
                external_id="tx-1", kind="buy", quantity=Decimal("5"),
                price=Decimal("8.22"), fee=Decimal("0.13"), date=date(2026, 7, 27),
            ),
            HoldingTransactionData(
                external_id="tx-2", kind="buy", quantity=Decimal("30"),
                price=Decimal("8.22"), fee=Decimal("0.13"), date=date(2026, 7, 27),
            ),
        ],
    )


@pytest.mark.asyncio
async def test_sync_populates_the_ledger_and_derives_the_position(
    session, test_user, test_connection
):
    mock_provider = AsyncMock()
    mock_provider.get_holdings = AsyncMock(return_value=[_holding_with_trades()])
    with patch("app.services.connection_service.get_provider", return_value=mock_provider):
        await connection_service._sync_holdings(
            session, test_user.id, test_connection, {"item_id": "i"}
        )
    await session.flush()

    asset = (await session.execute(
        select(Asset).where(Asset.external_id == "inv-1")
    )).scalar_one()
    rows = (await session.execute(
        select(AssetTransaction).where(AssetTransaction.asset_id == asset.id)
    )).scalars().all()

    assert {r.external_id for r in rows} == {"tx-1", "tx-2"}
    # Rows are tagged with the asset's own source, not a hardcoded provider
    # name — this test's connection uses provider="test".
    assert all(r.source == asset.source for r in rows)
    assert asset.units == Decimal("35")
    assert asset.average_price is not None


@pytest.mark.asyncio
async def test_second_sync_does_not_duplicate_the_ledger(
    session, test_user, test_connection
):
    mock_provider = AsyncMock()
    mock_provider.get_holdings = AsyncMock(return_value=[_holding_with_trades()])
    with patch("app.services.connection_service.get_provider", return_value=mock_provider):
        for _ in range(2):
            await connection_service._sync_holdings(
                session, test_user.id, test_connection, {"item_id": "i"}
            )
    await session.flush()

    asset = (await session.execute(
        select(Asset).where(Asset.external_id == "inv-1")
    )).scalar_one()
    rows = (await session.execute(
        select(AssetTransaction).where(AssetTransaction.asset_id == asset.id)
    )).scalars().all()
    assert len(rows) == 2
