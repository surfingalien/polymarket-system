"""
Kalshi V2 API client with RSA-PSS SHA-256 authentication.

All requests to api.elections.kalshi.com (or demo-api.kalshi.co) must
include an RSA-PSS signature over: timestamp + method + path.
Key generation: run scripts/setup_kalshi_keys.py and upload the public key
to https://kalshi.com/profile/api.

V2 endpoint reference: /trade-api/v2/
"""
from __future__ import annotations

import base64
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger(__name__)


def _fp(v) -> float:
    """Parse a Kalshi V2 fixed-point string (e.g. '10.00', '0.5600') to float.
    Returns 0.0 for None/blank/unparseable."""
    try:
        return float(v) if v not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


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
        private_key_content: str = "",
        base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
        mock_mode: bool = True,
    ) -> None:
        self._key_id = api_key_id
        self._key_path = private_key_path
        self._base = base_url.rstrip("/")
        self.mock_mode = mock_mode
        if private_key_content:
            self._private_key = self._load_private_key_from_string(private_key_content)
        else:
            self._private_key = self._load_private_key(private_key_path)
        self._http = httpx.AsyncClient(timeout=20.0)

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
    async def get_markets(
        self, limit: int = 20, status: str = "open"
    ) -> list[KalshiMarket]:
        if self.mock_mode:
            return self._mock_markets()
        if not self._private_key:
            raise RuntimeError(
                "Kalshi private key failed to parse — check KALSHI_PRIVATE_KEY_PEM format"
            )
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
        log.info("kalshi_markets_fetched", count=len(markets))
        return markets

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

    async def auth_check(self) -> tuple[bool, str]:
        """Make a real AUTHENTICATED call (balance) to verify trading credentials.
        Returns (ok, detail). Unlike GET /markets (public), this endpoint requires
        a valid signature, so it definitively tells whether the key ID + PEM are
        accepted by Kalshi for trading."""
        headers = self._sign("GET", "/trade-api/v2/portfolio/balance")
        if "KALSHI-ACCESS-SIGNATURE" not in headers:
            return False, "no signature built — missing key ID or unparseable PEM"
        try:
            resp = await self._http.get(
                f"{self._base}/portfolio/balance", headers=headers,
            )
            if resp.status_code == 200:
                bal = float(resp.json().get("balance", 0)) / 100.0
                return True, f"balance ${bal:,.2f}"
            _body = ""
            try:
                _body = resp.text[:200]
            except Exception:
                pass
            return False, f"HTTP {resp.status_code}: {_body}"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

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

        # V2 order endpoint. Kalshi deprecated the legacy /portfolio/orders
        # endpoint (returns deprecated_v1_order_endpoint / 410); the current
        # surface is /portfolio/events/orders with a single YES-denominated book:
        #   side="bid" → buy YES   |   side="ask" → sell YES (= buy NO)
        #   price is the YES price in dollars; client_order_id is required.
        # We translate our (action, side, contract-cents) order into that model.
        path = "/trade-api/v2/portfolio/events/orders"
        # Reconstruct the YES price (cents) from the stored contract price.
        yes_cents = order.limit_price if order.side == "yes" else (100 - order.limit_price)
        yes_cents = max(1, min(99, int(yes_cents)))
        v2_side = "bid" if (
            (order.action == "buy" and order.side == "yes")
            or (order.action == "sell" and order.side == "no")
        ) else "ask"
        body = {
            "ticker": order.ticker,
            "client_order_id": order.client_order_id or str(uuid.uuid4()),
            "side": v2_side,
            "count": str(int(order.count)),
            "price": f"{yes_cents / 100:.4f}",
            "time_in_force": "good_till_canceled",
            # Required by the V2 schema. taker_at_cross = if this order would
            # cross our own resting order, execute it as the taker (a safe default
            # for a bot that isn't market-making against itself).
            "self_trade_prevention_type": "taker_at_cross",
        }
        # A "sell" is closing an existing position. In the single-book model the
        # opposite-side order would otherwise read as OPENING a new (collateralized)
        # position and fail with insufficient_balance. reduce_only caps the order
        # at the held position so it only ever closes — no balance required.
        if order.action == "sell":
            body["reduce_only"] = True
        headers = self._sign("POST", path, body)
        # Fail loudly if we couldn't build authentication. Without a key ID +
        # parseable private key, _sign omits the KALSHI-ACCESS-* headers and the
        # order would go out unauthenticated — Kalshi then treats it as a browser
        # request and returns the confusing INVALID_CSRF_TOKEN / 410 instead of a
        # clear auth error. Surface the real cause instead.
        if "KALSHI-ACCESS-SIGNATURE" not in headers:
            raise RuntimeError(
                "Kalshi order not authenticated: missing API key ID or unparseable "
                "private key. Check KALSHI_API_KEY (the key ID from the dashboard) "
                "and KALSHI_PRIVATE_KEY_PEM."
            )
        try:
            resp = await self._http.post(
                f"{self._base}/portfolio/events/orders",
                json=body,
                headers=headers,
            )
            if resp.status_code >= 400:
                # Include Kalshi's response body — the status line alone hides the
                # actual cause (e.g. INVALID_CSRF_TOKEN = signature not accepted,
                # which means the key ID and PEM don't match the uploaded public key).
                _body = ""
                try:
                    _body = resp.text[:300]
                except Exception:
                    pass
                raise RuntimeError(f"HTTP {resp.status_code} placing order: {_body}")
            result = resp.json()
            if not isinstance(result, dict):
                result = {}
            # V2 CreateOrderV2Response is FLAT (no "order" wrapper) and has no
            # "status" field — only order_id, fill_count, remaining_count, ts_ms.
            # Derive a status from the fill so callers have something meaningful.
            _fill = _fp(result.get("fill_count"))
            _rem  = _fp(result.get("remaining_count"))
            _status = ("filled" if _rem == 0 and _fill > 0
                       else "partially_filled" if _fill > 0
                       else "resting")
            log.info(
                "kalshi_order_placed",
                order_id=result.get("order_id"),
                fill_count=_fill, remaining_count=_rem, status=_status,
            )
            # Normalize to a shape callers already expect.
            return {"order": {"order_id": result.get("order_id"), "status": _status,
                              "fill_count": _fill, "remaining_count": _rem}, **result}
        except Exception as exc:
            log.error("kalshi_place_order_failed", error=str(exc))
            raise

    async def cancel_order(self, order_id: str) -> dict:
        if self.mock_mode:
            return {"status": "MOCK_CANCELLED"}
        # V2 cancel lives on the events/orders surface, matching create.
        path = f"/trade-api/v2/portfolio/events/orders/{order_id}"
        headers = self._sign("DELETE", path)
        try:
            resp = await self._http.delete(
                f"{self._base}/portfolio/events/orders/{order_id}",
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
        # V2 amend on the events/orders surface; price is fixed-point YES dollars.
        path = f"/trade-api/v2/portfolio/events/orders/{order_id}/amend"
        yes_cents = max(1, min(99, int(limit_price)))
        body = {
            "count": str(int(count)),
            "price": f"{yes_cents / 100:.4f}",
        }
        headers = self._sign("POST", path, body)
        try:
            resp = await self._http.post(
                f"{self._base}/portfolio/events/orders/{order_id}/amend",
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
        # Kalshi's gateway rejects authenticated requests that carry a session
        # cookie (set by a prior public GET) with INVALID_CSRF_TOKEN / 410 — it
        # treats the request as browser-session auth and ignores our signature
        # headers. Clearing the jar before every signed request guarantees each
        # one is a clean API-key call. Public GETs don't need cookies either, so
        # this is safe across the board.
        try:
            self._http.cookies.clear()
        except Exception:
            pass

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

    def _load_private_key_from_string(self, pem_content: str):
        try:
            return serialization.load_pem_private_key(
                pem_content.encode(), password=None
            )
        except Exception as exc:
            log.warning("kalshi_key_parse_failed", error=str(exc))
            return None

    def _parse_market(self, m: dict) -> Optional[KalshiMarket]:
        try:
            ticker = m.get("ticker", "")
            if not ticker:
                return None
            # yes_ask/yes_bid are in cents (0–100); 0 means no order in book
            raw_ask = m.get("yes_ask") or 0
            raw_bid = m.get("yes_bid") or 0
            # Fall back to 50 cents if both sides have no orders
            if raw_ask == 0 and raw_bid == 0:
                raw_ask, raw_bid = 51, 49
            elif raw_ask == 0:
                raw_ask = raw_bid + 2
            elif raw_bid == 0:
                raw_bid = raw_ask - 2
            yes_ask = raw_ask / 100.0
            yes_bid = raw_bid / 100.0
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
