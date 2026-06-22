"""
Polymarket CLOB V2 order signer.

Real orders on the CLOB must be EIP-712 signed by the wallet that owns the
funds — an HMAC API-key header alone is not enough; the exchange rejects
unsigned orders. We delegate the signing + posting to the official
`py-clob-client` library (lazily imported) rather than hand-rolling EIP-712,
because a signing bug here loses real money.

SAFETY: mock_mode=True is the hard default. No network call that could move
real funds happens unless mock_mode is explicitly False AND a wallet private
key is present.

Wallet types (signature_type):
    0  EOA — a normal wallet you hold the private key for (default)
    1  POLY_PROXY — Polymarket email/Magic wallet (set `funder`)
    2  POLY_GNOSIS_SAFE — Polymarket browser wallet / Safe (set `funder`)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import structlog

log = structlog.get_logger(__name__)


class SigningUnavailable(RuntimeError):
    """Raised when live signing is requested but the toolchain/key is missing."""


@dataclass
class SignedOrderResult:
    order_id: str
    status: str
    raw: dict


class PolymarketOrderSigner:
    """
    Wraps the official py-clob-client to EIP-712-sign and post CLOB orders.

    The py-clob-client is synchronous; async callers get `place_order_async`,
    which offloads to a thread so the event loop is not blocked.
    """

    def __init__(
        self,
        private_key: str = "",
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        host: str = "https://clob.polymarket.com",
        chain_id: int = 137,
        signature_type: int = 0,
        funder: str = "",
        mock_mode: bool = True,
    ) -> None:
        self._private_key = self._normalize_private_key(private_key)
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_passphrase = api_passphrase
        self._host = host.rstrip("/")
        self._chain_id = chain_id
        self._signature_type = signature_type
        self._funder = funder
        self.mock_mode = mock_mode
        self._client = None  # lazily built py-clob-client instance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_private_key(key: str) -> str:
        """
        Light cleanup of a wallet private key read from secrets/env so signing
        doesn't fail with 'Non-hexadecimal digit found'. NEVER raises — strict
        validation happens at sign time in _validated_signing_key().

        Handles the common copy-paste / TOML-secret problems:
          - surrounding single/double quotes
          - leading/trailing whitespace and newlines
          - inner whitespace (spaces/newlines pasted mid-key)

        Keeps the original 0x prefix convention if present; eth-account / the
        CLOB client accept the key with or without it.
        """
        if not key:
            return ""
        k = key.strip().strip('"').strip("'").strip()
        # Drop any internal whitespace (spaces, tabs, newlines pasted mid-key)
        k = "".join(k.split())
        return k

    def _validated_signing_key(self) -> str:
        """
        Return a canonical 0x-prefixed 64-hex-char key for live signing, or
        raise SigningUnavailable with an actionable message. Called only on the
        real signing path so mock/test construction never trips over it.
        """
        k = self._private_key
        if not k:
            raise SigningUnavailable(
                "No wallet private key configured — cannot sign a real order. "
                "Set POLYMARKET_PRIVATE_KEY (never paste it into chat)."
            )
        if k[:2] in ("0x", "0X"):
            k = k[2:]
        try:
            int(k, 16)
        except ValueError as exc:
            raise SigningUnavailable(
                "POLYMARKET_PRIVATE_KEY is not valid hexadecimal (check for stray "
                "characters or quotes, and that you pasted the raw private key — "
                "not a seed phrase or wallet address)."
            ) from exc
        if len(k) != 64:
            raise SigningUnavailable(
                f"POLYMARKET_PRIVATE_KEY should be 64 hex chars (32 bytes); got "
                f"{len(k)}. Make sure you pasted the full wallet private key."
            )
        return "0x" + k.lower()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def can_sign(self) -> bool:
        """True if a real (non-mock) signed order could be attempted."""
        return bool(self._private_key) and not self.mock_mode

    def place_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,            # "BUY" | "SELL"
        order_type: str = "GTC",
    ) -> SignedOrderResult:
        """Sign + post a single order. Synchronous."""
        if self.mock_mode:
            log.info(
                "MOCK_SIGNED_ORDER",
                token_id=token_id, side=side, price=price, size=size,
            )
            return SignedOrderResult(
                order_id=f"mock_{int(time.time())}",
                status="MOCK",
                raw={"mock": True, "token_id": token_id, "side": side,
                     "price": price, "size": size},
            )

        if not self._private_key:
            raise SigningUnavailable(
                "No wallet private key configured — cannot sign a real order. "
                "Set POLYMARKET_PRIVATE_KEY (never paste it into chat)."
            )

        client = self._get_client()
        OrderArgs, OrderType, BUY, SELL = self._clob_types()

        order_args = OrderArgs(
            price=round(float(price), 4),
            size=round(float(size), 4),
            side=BUY if side.upper() == "BUY" else SELL,
            token_id=str(token_id),
        )
        signed = client.create_order(order_args)
        ot = getattr(OrderType, order_type.upper(), OrderType.GTC)
        resp = client.post_order(signed, ot)

        order_id = (resp or {}).get("orderID") or (resp or {}).get("orderId", "")
        status = (resp or {}).get("status", "unknown")
        log.info("signed_order_posted", order_id=order_id, status=status)
        return SignedOrderResult(order_id=order_id, status=status, raw=resp or {})

    async def place_order_async(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
        order_type: str = "GTC",
    ) -> SignedOrderResult:
        """Async wrapper — runs the synchronous signer in a worker thread."""
        return await asyncio.to_thread(
            self.place_order, token_id, price, size, side, order_type
        )

    def cancel_order_sync(self, order_id: str) -> dict:
        """Cancel an order using py-clob-client's L2 auth (correct HMAC)."""
        client = self._get_client()
        resp = client.cancel(order_id)
        return resp if isinstance(resp, dict) else {}

    async def cancel_order_async(self, order_id: str) -> dict:
        return await asyncio.to_thread(self.cancel_order_sync, order_id)

    def get_positions_sync(self) -> list[dict]:
        """Fetch open orders/positions via py-clob-client L2 auth."""
        from py_clob_client.clob_types import OpenOrderParams
        client = self._get_client()
        resp = client.get_orders(params=OpenOrderParams()) or []
        if isinstance(resp, dict):
            resp = resp.get("data", []) or []
        return resp if isinstance(resp, list) else []

    async def get_positions_async(self) -> list[dict]:
        return await asyncio.to_thread(self.get_positions_sync)

    def get_balance_sync(self) -> float:
        """Fetch USDC collateral balance via py-clob-client L2 auth."""
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
        client = self._get_client()
        resp = client.get_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        ) or {}
        return float(resp.get("balance", 0.0))

    async def get_balance_async(self) -> float:
        return await asyncio.to_thread(self.get_balance_sync)

    # ------------------------------------------------------------------
    # Lazy py-clob-client construction
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds
        except ImportError as exc:
            raise SigningUnavailable(
                "py-clob-client is not installed in this environment. "
                "The Streamlit Cloud deployment intentionally excludes it "
                "to avoid C-extension build failures on Python 3.14. "
                "To execute live orders, either: (a) run the local bot "
                "(`MOCK_MODE=false python main.py`) which uses the full "
                "requirements.txt, or (b) add py-clob-client to a custom "
                "Streamlit environment with a matching Python version."
            ) from exc

        kwargs = {"key": self._validated_signing_key(), "chain_id": self._chain_id}
        if self._signature_type:
            kwargs["signature_type"] = self._signature_type
        if self._funder:
            kwargs["funder"] = self._funder

        client = ClobClient(self._host, **kwargs)

        # L2 API credentials: use provided creds, else derive from the key.
        if self._api_key and self._api_secret and self._api_passphrase:
            creds = ApiCreds(
                api_key=self._api_key,
                api_secret=self._api_secret,
                api_passphrase=self._api_passphrase,
            )
        else:
            creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)

        self._client = client
        return client

    @staticmethod
    def _clob_types():
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY, SELL
        return OrderArgs, OrderType, BUY, SELL
