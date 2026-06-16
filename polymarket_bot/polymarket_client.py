"""
Polymarket CLOB V2 client.

Auth: L1 (wallet signature) for account actions, L2 (API key) for trading.
API key credentials are obtained by signing a nonce with your private key.
CLOB V2 orders no longer include feeRateBps (deprecated April 28, 2026).
"""
from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from .order_signer import PolymarketOrderSigner

log = structlog.get_logger(__name__)

CLOB_HOST = "https://clob.polymarket.com"
GAMMA_HOST = "https://gamma-api.polymarket.com"


@dataclass
class ClobOrder:
    condition_id: str
    token_id: str          # YES token ID
    side: str              # "BUY" | "SELL"
    size: float            # number of shares
    price: float           # limit price (0–1)
    order_type: str = "GTC"  # GTC | FOK | GTD
    expiration: int = 0    # unix ts, 0 = no expiry

    def to_dict(self) -> dict:
        d = {
            "tokenID": self.token_id,
            "side": self.side,
            "size": str(self.size),
            "price": str(self.price),
            "orderType": self.order_type,
        }
        if self.expiration:
            d["expiration"] = str(self.expiration)
        return d


@dataclass
class Market:
    condition_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    best_bid: float = 0.0
    best_ask: float = 0.0
    last_price: float = 0.0
    volume_24h: float = 0.0
    liquidity: float = 0.0
    active: bool = True
    end_date_iso: Optional[str] = None
    outcomes: list[str] = field(default_factory=lambda: ["YES", "NO"])

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    def to_analysis_dict(self) -> dict:
        return {
            "id": self.condition_id,
            "question": self.question,
            "price": self.mid_price,
            "volume": self.volume_24h,
            "liquidity": self.liquidity,
            "platform": "polymarket",
            "yes_ask": self.best_ask,
            "yes_bid": self.best_bid,
        }


class PolymarketClient:
    """
    Polymarket CLOB V2 HTTP client.

    In mock_mode: no real orders are placed; all reads go to live API.
    Authentication: HMAC-SHA256 on the request timestamp + method + path + body.
    """

    def __init__(
        self,
        private_key: str = "",
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        clob_host: str = CLOB_HOST,
        gamma_host: str = GAMMA_HOST,
        chain_id: int = 137,
        mock_mode: bool = True,
        signature_type: int = 0,
        funder: str = "",
    ) -> None:
        self._pk = private_key
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_passphrase = api_passphrase
        self._clob = clob_host.rstrip("/")
        self._gamma = gamma_host.rstrip("/")
        self._chain_id = chain_id
        self.mock_mode = mock_mode
        self._http = httpx.AsyncClient(timeout=20.0)
        # EIP-712 order signer (lazily uses py-clob-client only when going live)
        self._signer = PolymarketOrderSigner(
            private_key=private_key,
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
            host=self._clob,
            chain_id=chain_id,
            signature_type=signature_type,
            funder=funder,
            mock_mode=mock_mode,
        )

    # ------------------------------------------------------------------
    # Market data (public)
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def get_markets(
        self, limit: int = 20, active_only: bool = True
    ) -> list[Market]:
        try:
            params: dict = {"limit": limit}
            if active_only:
                params["active"] = "true"
            resp = await self._http.get(f"{self._gamma}/markets", params=params)
            resp.raise_for_status()
            data = resp.json()
            markets = []
            for m in data if isinstance(data, list) else data.get("markets", []):
                parsed = self._parse_market(m)
                if parsed:
                    markets.append(parsed)
            return markets
        except Exception as exc:
            log.error("get_markets_failed", error=str(exc))
            return self._mock_markets() if self.mock_mode else []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def get_order_book(self, token_id: str) -> dict:
        """Returns raw order book {bids: [...], asks: [...]}."""
        try:
            resp = await self._http.get(
                f"{self._clob}/book", params={"token_id": token_id}
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("get_order_book_failed", token_id=token_id, error=str(exc))
            return {"bids": [], "asks": []}

    async def get_market(self, condition_id: str) -> Optional[Market]:
        try:
            resp = await self._http.get(f"{self._gamma}/markets/{condition_id}")
            resp.raise_for_status()
            return self._parse_market(resp.json())
        except Exception as exc:
            log.warning("get_market_failed", condition_id=condition_id, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Trading (requires auth)
    # ------------------------------------------------------------------

    async def place_order(self, order: ClobOrder) -> dict:
        if self.mock_mode:
            log.info(
                "MOCK_ORDER",
                condition_id=order.condition_id,
                side=order.side,
                price=order.price,
                size=order.size,
            )
            return {"id": f"mock_{int(time.time())}", "status": "MOCK"}

        # Live order: must be EIP-712 signed by the funding wallet. The CLOB
        # rejects orders that carry only an HMAC API-key header.
        try:
            result = await self._signer.place_order_async(
                token_id=order.token_id,
                price=order.price,
                size=order.size,
                side=order.side,
                order_type=order.order_type,
            )
            log.info("order_placed", order_id=result.order_id, status=result.status)
            return {"id": result.order_id, "status": result.status, **result.raw}
        except Exception as exc:
            log.error("place_order_failed", error=str(exc))
            raise

    async def cancel_order(self, order_id: str) -> dict:
        if self.mock_mode:
            return {"status": "MOCK_CANCELLED"}

        headers = self._auth_headers("DELETE", f"/order/{order_id}", "")
        try:
            resp = await self._http.delete(
                f"{self._clob}/order/{order_id}", headers=headers
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.error("cancel_order_failed", order_id=order_id, error=str(exc))
            raise

    async def get_positions(self) -> list[dict]:
        if not self._api_key:
            return []
        headers = self._auth_headers("GET", "/positions", "")
        try:
            resp = await self._http.get(f"{self._clob}/positions", headers=headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("get_positions_failed", error=str(exc))
            return []

    async def get_balance(self) -> float:
        if not self._api_key:
            return 0.0
        headers = self._auth_headers("GET", "/balance", "")
        try:
            resp = await self._http.get(f"{self._clob}/balance", headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("balance", 0.0))
        except Exception as exc:
            log.warning("get_balance_failed", error=str(exc))
            return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _auth_headers(self, method: str, path: str, body: str) -> dict:
        timestamp = str(int(time.time() * 1000))
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            self._api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        return {
            "POLY-API-KEY": self._api_key,
            "POLY-SIGNATURE": signature,
            "POLY-TIMESTAMP": timestamp,
            "POLY-PASSPHRASE": self._api_passphrase,
            "Content-Type": "application/json",
        }

    def _parse_market(self, m: dict) -> Optional[Market]:
        if not isinstance(m, dict):
            return None
        try:
            cid = m.get("conditionId") or m.get("condition_id") or m.get("id", "")
            if not cid:
                return None
            tokens = m.get("tokens") or []
            yes_token = ""
            no_token = ""
            for t in tokens:
                if not isinstance(t, dict):
                    continue
                outcome = (t.get("outcome") or "").lower()
                if outcome == "yes":
                    yes_token = t.get("token_id", "")
                elif outcome == "no":
                    no_token = t.get("token_id", "")
            # Fallback: clobTokenIds is a flat list [yes_id, no_id]
            if not yes_token:
                clob_ids = m.get("clobTokenIds") or []
                if len(clob_ids) >= 1:
                    yes_token = clob_ids[0]
                if len(clob_ids) >= 2:
                    no_token = clob_ids[1]
            prices = m.get("outcomePrices") or []
            try:
                yes_price = float(prices[0]) if prices else 0.5
            except (ValueError, TypeError):
                yes_price = 0.5
            return Market(
                condition_id=cid,
                question=m.get("question", ""),
                yes_token_id=yes_token,
                no_token_id=no_token,
                best_bid=max(0.0, yes_price - 0.005),
                best_ask=min(1.0, yes_price + 0.005),
                last_price=yes_price,
                volume_24h=float(m.get("volumeNum") or m.get("volume") or 0),
                liquidity=float(m.get("liquidityNum") or m.get("liquidity") or 0),
                active=bool(m.get("active", True)),
                end_date_iso=m.get("endDate") or m.get("end_date_iso"),
            )
        except Exception as exc:
            log.debug("market_parse_error", error=str(exc))
            return None

    def _mock_markets(self) -> list[Market]:
        return [
            Market(
                condition_id=f"mock_cid_{i}",
                question=q,
                yes_token_id=f"mock_yes_{i}",
                no_token_id=f"mock_no_{i}",
                best_bid=p - 0.01,
                best_ask=p + 0.01,
                last_price=p,
                volume_24h=50000.0,
                liquidity=20000.0,
            )
            for i, (q, p) in enumerate([
                ("Will the Fed cut rates in Q3 2026?", 0.38),
                ("Will BTC exceed $120k by end of 2026?", 0.45),
                ("Will SpaceX land on Mars by 2027?", 0.12),
                ("Will GPT-5 be released before Claude 4?", 0.25),
            ])
        ]

    async def close(self) -> None:
        await self._http.aclose()
