"""Mapping tests for the investment-transaction array Pluggy embeds in
`GET /investments`, which feeds the asset ledger.

Pure parser tests — no network, no database.
"""

from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.providers.pluggy import PluggyProvider, _build_holding_data, _sum_expenses


def _investment(transactions: list[dict] | None) -> dict:
    """A minimal /investments result carrying the given transactions."""
    payload = {
        "id": "inv-1",
        "name": "MXRF11",
        "currencyCode": "BRL",
        "balance": 400.0,
        "quantity": 53,
        "value": 7.49,
        "type": "EQUITY",
    }
    if transactions is not None:
        payload["transactions"] = transactions
    return payload


def test_buy_maps_to_ledger_row():
    holding = _build_holding_data(_investment([
        {
            "id": "tx-1",
            "type": "BUY",
            "quantity": 5,
            "value": 8.22,
            "tradeDate": "2026-07-27T00:00:00.000Z",
            "date": "2026-07-30T00:00:00.000Z",
            "expenses": None,
        },
    ]))
    assert len(holding.transactions) == 1
    tx = holding.transactions[0]
    assert tx.external_id == "tx-1"
    assert tx.kind == "buy"
    assert tx.quantity == Decimal("5")
    assert tx.price == Decimal("8.22")
    assert tx.fee == Decimal("0")


def test_trade_date_wins_over_date():
    """tradeDate is the date tax reporting uses; `date` is the settlement."""
    holding = _build_holding_data(_investment([
        {"id": "tx-1", "type": "BUY", "quantity": 5, "value": 8.22,
         "tradeDate": "2026-07-27T00:00:00.000Z", "date": "2026-07-30T00:00:00.000Z"},
    ]))
    assert holding.transactions[0].date == date(2026, 7, 27)


def test_date_used_when_trade_date_missing():
    holding = _build_holding_data(_investment([
        {"id": "tx-1", "type": "BUY", "quantity": 5, "value": 8.22,
         "date": "2026-07-30T00:00:00.000Z"},
    ]))
    assert holding.transactions[0].date == date(2026, 7, 30)


def test_sell_maps_to_sell_kind():
    holding = _build_holding_data(_investment([
        {"id": "tx-1", "type": "SELL", "quantity": 5, "value": 9.10,
         "tradeDate": "2026-07-27T00:00:00.000Z"},
    ]))
    assert holding.transactions[0].kind == "sell"


def test_negative_quantity_is_normalized_positive():
    """The ledger stores magnitude; direction lives in `kind`."""
    holding = _build_holding_data(_investment([
        {"id": "tx-1", "type": "SELL", "quantity": -5, "value": 9.10,
         "tradeDate": "2026-07-27T00:00:00.000Z"},
    ]))
    assert holding.transactions[0].quantity == Decimal("5")


def test_interest_row_is_dropped():
    """INTEREST carries no quantity — there is no position to derive."""
    holding = _build_holding_data(_investment([
        {"id": "tx-1", "type": "INTEREST", "amount": 76.6, "quantity": None,
         "value": None, "tradeDate": "2026-08-14T00:00:00.000Z"},
    ]))
    assert holding.transactions == []


def test_buy_without_quantity_or_price_is_dropped():
    holding = _build_holding_data(_investment([
        {"id": "tx-1", "type": "BUY", "quantity": None, "value": 8.22,
         "tradeDate": "2026-07-27T00:00:00.000Z"},
        {"id": "tx-2", "type": "BUY", "quantity": 5, "value": None,
         "tradeDate": "2026-07-27T00:00:00.000Z"},
        {"id": "tx-3", "type": "BUY", "quantity": 0, "value": 8.22,
         "tradeDate": "2026-07-27T00:00:00.000Z"},
    ]))
    assert holding.transactions == []


def test_row_without_id_is_dropped():
    """The provider id is the dedupe key; without it we would duplicate on
    every sync."""
    holding = _build_holding_data(_investment([
        {"type": "BUY", "quantity": 5, "value": 8.22,
         "tradeDate": "2026-07-27T00:00:00.000Z"},
    ]))
    assert holding.transactions == []


def test_expenses_are_summed_with_nulls_as_zero():
    assert _sum_expenses({
        "brokerageFee": 0, "settlementFee": 0.11, "clearingFee": 0,
        "stockExchangeFee": 0.02, "custodyFee": 0, "incomeTax": 0,
        "serviceTax": None, "other": None, "tradingAssetsNoticeFee": 0,
        "maintenanceFee": None, "operatingFee": None, "iof": None,
        "id": "exp-1", "transactionId": "tx-1",
    }) == Decimal("0.13")


def test_expenses_none_is_zero_fee():
    assert _sum_expenses(None) == Decimal("0")


def test_expenses_flow_into_the_row_fee():
    holding = _build_holding_data(_investment([
        {"id": "tx-1", "type": "BUY", "quantity": 5, "value": 8.22,
         "tradeDate": "2026-07-27T00:00:00.000Z",
         "expenses": {"settlementFee": 0.11, "stockExchangeFee": 0.02}},
    ]))
    assert holding.transactions[0].fee == Decimal("0.13")


def test_missing_transactions_key_yields_empty_list():
    assert _build_holding_data(_investment(None)).transactions == []


def test_transactions_no_longer_duplicated_into_metadata():
    """Promoted fields must leave `metadata`, which is stored verbatim on
    `assets.external_metadata` — keeping both would store the same array twice."""
    holding = _build_holding_data(_investment([
        {"id": "tx-1", "type": "BUY", "quantity": 5, "value": 8.22,
         "tradeDate": "2026-07-27T00:00:00.000Z"},
    ]))
    assert "transactions" not in (holding.metadata or {})


# ---- Async get_holdings tests fetching from /investments/{id}/transactions ---


@pytest.mark.asyncio
async def test_get_holdings_fetches_transactions_per_investment():
    """get_holdings calls /investments and then /investments/{id}/transactions."""
    investments_response = MagicMock()
    investments_response.raise_for_status = MagicMock()
    investments_response.json = MagicMock(return_value={
        "results": [
            {
                "id": "inv-1",
                "name": "MXRF11",
                "currencyCode": "BRL",
                "balance": 400.0,
                "quantity": 53,
                "value": 7.49,
                "type": "EQUITY",
            }
        ],
        "totalPages": 1,
    })

    txns_response = MagicMock()
    txns_response.status_code = 200
    txns_response.raise_for_status = MagicMock()
    txns_response.json = MagicMock(return_value={
        "results": [
            {
                "id": "tx-1",
                "type": "BUY",
                "quantity": 53,
                "value": 7.49,
                "tradeDate": "2026-07-27T00:00:00.000Z",
            }
        ],
        "totalPages": 1,
    })

    async def fake_get(url, headers=None, params=None):
        if url.endswith("/investments"):
            return investments_response
        if "/investments/inv-1/transactions" in url:
            return txns_response
        raise AssertionError(f"Unexpected url: {url}")

    client = MagicMock()
    client.get = AsyncMock(side_effect=fake_get)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    provider = PluggyProvider()
    with patch.object(
        PluggyProvider, "_ensure_api_key", new=AsyncMock(return_value="fake-key")
    ), patch("app.providers.pluggy.httpx.AsyncClient", return_value=client):
        holdings = await provider.get_holdings({"item_id": "item-1"})

    assert len(holdings) == 1
    assert holdings[0].external_id == "inv-1"
    assert len(holdings[0].transactions) == 1
    assert holdings[0].transactions[0].external_id == "tx-1"
    assert holdings[0].transactions[0].quantity == Decimal("53")


@pytest.mark.asyncio
async def test_get_holdings_handles_404_on_transactions_gracefully():
    """If /investments/{id}/transactions returns 404, holding still loads with empty transactions."""
    investments_response = MagicMock()
    investments_response.raise_for_status = MagicMock()
    investments_response.json = MagicMock(return_value={
        "results": [
            {
                "id": "inv-2",
                "name": "CDB Prefixado",
                "currencyCode": "BRL",
                "balance": 1000.0,
                "type": "FIXED_INCOME",
            }
        ],
        "totalPages": 1,
    })

    txns_response = MagicMock()
    txns_response.status_code = 404

    async def fake_get(url, headers=None, params=None):
        if url.endswith("/investments"):
            return investments_response
        if "/investments/inv-2/transactions" in url:
            return txns_response
        raise AssertionError(f"Unexpected url: {url}")

    client = MagicMock()
    client.get = AsyncMock(side_effect=fake_get)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    provider = PluggyProvider()
    with patch.object(
        PluggyProvider, "_ensure_api_key", new=AsyncMock(return_value="fake-key")
    ), patch("app.providers.pluggy.httpx.AsyncClient", return_value=client):
        holdings = await provider.get_holdings({"item_id": "item-1"})

    assert len(holdings) == 1
    assert holdings[0].external_id == "inv-2"
    assert holdings[0].transactions == []


@pytest.mark.asyncio
async def test_get_holdings_handles_http_error_on_transactions_gracefully():
    """If /investments/{id}/transactions encounters a network or server error, holding still loads."""
    investments_response = MagicMock()
    investments_response.raise_for_status = MagicMock()
    investments_response.json = MagicMock(return_value={
        "results": [
            {
                "id": "inv-3",
                "name": "Fundo Multimercado",
                "currencyCode": "BRL",
                "balance": 500.0,
                "type": "MUTUAL_FUND",
            }
        ],
        "totalPages": 1,
    })

    async def fake_get(url, headers=None, params=None):
        if url.endswith("/investments"):
            return investments_response
        if "/investments/inv-3/transactions" in url:
            raise httpx.ConnectTimeout("timeout")
        raise AssertionError(f"Unexpected url: {url}")

    client = MagicMock()
    client.get = AsyncMock(side_effect=fake_get)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    provider = PluggyProvider()
    with patch.object(
        PluggyProvider, "_ensure_api_key", new=AsyncMock(return_value="fake-key")
    ), patch("app.providers.pluggy.httpx.AsyncClient", return_value=client):
        holdings = await provider.get_holdings({"item_id": "item-1"})

    assert len(holdings) == 1
    assert holdings[0].external_id == "inv-3"
    assert holdings[0].transactions == []

