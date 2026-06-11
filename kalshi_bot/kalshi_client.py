"""
Kalshi V2 API client with RSA-PSS SHA-256 authentication.

All requests to trading-api.kalshi.com (or demo-api.kalshi.co) must
include an RSA-PSS signature over: timestamp + method + path.
Key generation: run scripts/setup_kalshi_keys.py and upload the public key
to https://kalshi.com/profile/api.

V2 endpoint reference: /trade-api/v2/
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger(__name__)


@dataclass
class KalshiMarket:
    ticker: str
    event_ticker: str
    title: str
    yes_ask: float = 0.5
    yes_bid: float = 0.5
    no_ask: float = 0.5
    no_bid: float = 0.5
    volume: float = 0.0
    open_interest: float = 0.0
    status: str = "open"
    close_time: Optional[str] = None

    @property
    def mid_price(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2.0

    def to_analysis_dict(self) -> dict:
        return {
            "id": self.ticker,
            "question": self.title,
            "price": self.mid_price,
            "volume": self.volume,
            "liquidity": self.open_interest * self.mid_price,
            "platform": "kalshi",
            "yes_ask": self.yes_ask,
            "yes_bid": self.yes_bid,
            "ticker": self.ticker,
            "event_ticker": self.event_ticker,
        }


@dataclass
class KalshiOrder:
    ticker: str
    action: str          # "buy" | "sell"
    side: str            # "yes" | "no"
    count: int           # number of contracts
    limit_price: int     # in cents (0–100)
    order_type: str = "limit"
    client_order_id: str = ""

    def to_dict(self) -> dict:
        d = {
            "ticker": self.ticker,
            "action": self.action,
            "side": self.side,
            "type": self.order_type,
            "count": self.count,
        }
        if self.order_type == "limit":
            d["limit_price"] = self.limit_price
        if self.client_order_id:
            d["client_order_id"] = self.client_order_id
        return d


class KalshiClient:
    """
    Kalshi V2 trading API client.
    mock_mode=True → logs orders but never sends them to the API.
    """

    def __init__(
        self,
        api_key_id: str = "",
        private_key_path: str = "keys/kalshi_private.pem",
        base_url: str = "https://trading-api.kalshi.com/trade-api/v2",
        mock_mode: bool = True,
    ) -> None:
        self._key_id = api_key_id
        self._key_path = private_key_path
        self._base = base_url.rstrip("/")
        self.mock_mode = mock_mode
        self._private_key = self._load_private_key(private_key_path)
        self._http = httpx.AsyncClient(timeout=20.0)

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def get_markets(
        self, limit: int = 20, status: str = "open"
    ) -> list[KalshiMarket]:
        try:
            resp = await self._http.get(
                f"{self._base}/markets",
                params={"limit": limit, "status": status},
                headers=self._sign("GET", "/trade-api/v2/markets"),
            )
            resp.raise_for_status()
            data = resp.json()
            markets = []
            for m in data.get("markets", []):
                parsed = self._parse_market(m)
                if parsed:
                    markets.append(parsed)
            return markets
        except Exception as exc:
            log.error("kalshi_get_markets_failed", error=str(exc))
            return self._mock_markets() if self.mock_mode else []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def get_market(self, ticker: str) -> Optional[KalshiMarket]:
        try:
            path = f"/trade-api/v2/markets/{ticker}"
            resp = await self._http.get(
                f"{self._base}/markets/{ticker}",
                headers=self._sign("GET", path),
            )
            resp.raise_for_status()
            return self._parse_market(resp.json().get("market", {}))
        except Exception as exc:
            log.warning("kalshi_get_market_failed", ticker=ticker, error=str(exc))
            return None

    async def get_order_book(self, ticker: str) -> dict:
        try:
            path = f"/trade-api/v2/markets/{ticker}/orderbook"
            resp = await self._http.get(
                f"{self._base}/markets/{ticker}/orderbook",
                headers=self._sign("GET", path),
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("kalshi_ob_failed", ticker=ticker, error=str(exc))
            return {}

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_balance(self) -> float:
        if not self._key_id:
            return 0.0
        try:
            path = "/trade-api/v2/portfolio/balance"
            resp = await self._http.get(
                f"{self._base}/portfolio/balance",
                headers=self._sign("GET", path),
            )
            resp.raise_for_status()
            data = resp.json()
            # balance in cents
            return float(data.get("balance", 0)) / 100.0
        except Exception as exc:
            log.warning("kalshi_balance_failed", error=str(exc))
            return 0.0

    async def get_positions(self) -> list[dict]:
        if not self._key_id:
            return []
        try:
            path = "/trade-api/v2/portfolio/positions"
            resp = await self._http.get(
                f"{self._base}/portfolio/positions",
                headers=self._sign("GET", path),
            )
            resp.raise_for_status()
            return resp.json().get("market_positions", [])
        except Exception as exc:
            log.warning("kalshi_positions_failed", error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------

    async def place_order(self, order: KalshiOrder) -> dict:
        if self.mock_mode:
            log.info(
                "MOCK_KALSHI_ORDER",
                ticker=order.ticker,
                action=order.action,
                side=order.side,
                count=order.count,
                limit_price=order.limit_price,
            )
            return {"order_id": f"mock_{int(time.time())}", "status": "MOCK"}

        path = "/trade-api/v2/portfolio/orders"
        body = order.to_dict()
        headers = self._sign("POST", path, body)
        try:
            resp = await self._http.post(
                f"{self._base}/portfolio/orders",
                json=body,
                headers=headers,
            )
            resp.raise_for_status()
            result = resp.json()
            log.info(
                "kalshi_order_placed",
                order_id=result.get("order", {}).get("order_id"),
                status=result.get("order", {}).get("status"),
            )
            return result
        except Exception as exc:
            log.error("kalshi_place_order_failed", error=str(exc))
            raise

    async def cancel_order(self, order_id: str) -> dict:
        if self.mock_mode:
            return {"status": "MOCK_CANCELLED"}
        path = f"/trade-api/v2/portfolio/orders/{order_id}"
        headers = self._sign("DELETE", path)
        try:
            resp = await self._http.delete(
                f"{self._base}/portfolio/orders/{order_id}",
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.error("kalshi_cancel_order_failed", order_id=order_id, error=str(exc))
            raise

    async def amend_order(
        self, order_id: str, count: int, limit_price: int
    ) -> dict:
        if self.mock_mode:
            return {"status": "MOCK_AMENDED"}
        path = f"/trade-api/v2/portfolio/orders/{order_id}/amend"
        body = {"count": count, "limit_price": limit_price}
        headers = self._sign("POST", path, body)
        try:
            resp = await self._http.post(
                f"{self._base}/portfolio/orders/{order_id}/amend",
                json=body,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.error("kalshi_amend_failed", order_id=order_id, error=str(exc))
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sign(
        self, method: str, path: str, body: Optional[dict] = None
    ) -> dict:
        if not self._private_key or not self._key_id:
            return {"Content-Type": "application/json"}

        ts_ms = str(int(time.time() * 1000))
        msg = ts_ms + method.upper() + path

        signature = self._private_key.sign(
            msg.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        sig_b64 = base64.b64encode(signature).decode()

        headers = {
            "KALSHI-ACCESS-KEY": self._key_id,
            "KALSHI-ACCESS-SIGNATURE": sig_b64,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "Content-Type": "application/json",
        }
        return headers

    def _load_private_key(self, path: str):
        p = Path(path)
        if not p.exists():
            log.warning("kalshi_key_not_found", path=path)
            return None
        try:
            with open(p, "rb") as f:
                return serialization.load_pem_private_key(f.read(), password=None)
        except Exception as exc:
            log.warning("kalshi_key_load_failed", error=str(exc))
            return None

    def _parse_market(self, m: dict) -> Optional[KalshiMarket]:
        try:
            ticker = m.get("ticker", "")
            if not ticker:
                return None
            yes_ask = m.get("yes_ask", 50) / 100.0
            yes_bid = m.get("yes_bid", 50) / 100.0
            return KalshiMarket(
                ticker=ticker,
                event_ticker=m.get("event_ticker", ""),
                title=m.get("title", ""),
                yes_ask=yes_ask,
                yes_bid=yes_bid,
                no_ask=1.0 - yes_bid,
                no_bid=1.0 - yes_ask,
                volume=float(m.get("volume", 0)),
                open_interest=float(m.get("open_interest", 0)),
                status=m.get("status", "open"),
                close_time=m.get("close_time"),
            )
        except Exception as exc:
            log.debug("kalshi_parse_error", error=str(exc))
            return None

    def _mock_markets(self) -> list[KalshiMarket]:
        return [
            KalshiMarket(
                ticker=f"MOCK-{i}",
                event_ticker=f"MOCK-EVENT-{i}",
                title=t,
                yes_ask=p + 0.01,
                yes_bid=p - 0.01,
                no_ask=1.0 - p + 0.01,
                no_bid=1.0 - p - 0.01,
                volume=25000.0,
                open_interest=10000.0,
            )
            for i, (t, p) in enumerate([
                ("Federal Reserve rate cut Q3 2026", 0.40),
                ("Bitcoin above $120k end 2026", 0.43),
                ("US recession by Q1 2027", 0.29),
                ("SpaceX Starship orbit 2026", 0.65),
            ])
        ]

    async def close(self) -> None:
        await self._http.aclose()
