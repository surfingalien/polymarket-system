"""
Tests for PolymarketOrderSigner.

These verify the SAFE paths without ever touching the live CLOB:
- mock_mode never signs or posts
- live mode without a key fails loudly instead of silently doing nothing
- can_sign gate reflects key presence + mock state
"""
from __future__ import annotations

import asyncio

import pytest

from polymarket_bot.order_signer import (
    PolymarketOrderSigner,
    SignedOrderResult,
    SigningUnavailable,
)


class TestMockMode:
    def test_mock_mode_returns_mock_result_without_key(self):
        signer = PolymarketOrderSigner(mock_mode=True)
        res = signer.place_order(token_id="t1", price=0.55, size=10, side="BUY")
        assert isinstance(res, SignedOrderResult)
        assert res.status == "MOCK"
        assert res.order_id.startswith("mock_")

    def test_mock_mode_ignores_real_key(self):
        # Even with a (fake) key, mock mode must not attempt real signing.
        signer = PolymarketOrderSigner(private_key="0xabc", mock_mode=True)
        res = signer.place_order(token_id="t1", price=0.4, size=5, side="SELL")
        assert res.status == "MOCK"

    def test_can_sign_false_in_mock_mode(self):
        signer = PolymarketOrderSigner(private_key="0xabc", mock_mode=True)
        assert signer.can_sign is False


class TestLiveGuards:
    def test_live_mode_without_key_raises(self):
        signer = PolymarketOrderSigner(mock_mode=False)  # no key
        with pytest.raises(SigningUnavailable):
            signer.place_order(token_id="t1", price=0.5, size=10, side="BUY")

    def test_can_sign_true_when_key_present_and_live(self):
        signer = PolymarketOrderSigner(private_key="0xabc", mock_mode=False)
        assert signer.can_sign is True


class TestAsyncWrapper:
    def test_async_mock_path(self):
        signer = PolymarketOrderSigner(mock_mode=True)
        res = asyncio.run(
            signer.place_order_async(token_id="t1", price=0.5, size=1, side="BUY")
        )
        assert res.status == "MOCK"
