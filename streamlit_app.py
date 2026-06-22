"""
Polymarket / Kalshi AI Trading Bot — Streamlit Demo Dashboard

Runs the full predictive algorithm stack on mock market data.
No API keys required for demo. Set ANTHROPIC_API_KEY to enable live Claude analysis.

Uses ONLY Streamlit-native components (no Plotly) to guarantee compatibility
across Python 3.10–3.14 and all Streamlit Cloud versions.
"""
from __future__ import annotations

import asyncio
import math
import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as _stcomp

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Polymarket AI Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── path ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── imports ───────────────────────────────────────────────────────────────────
try:
    from shared.advanced_signals import (
        CategoryEdgeModel, LongshotBiasDetector,
        MarketQualityScorer, TemporalPatternSignal,
    )
    from shared.claude_agent import ClaudeAgent
    from shared.learning_engine import LearningEngine
    from shared.predictive_models import (
        BayesianEstimator, CrossMarketCorrelator,
        EnsemblePredictor, KellyCriterion,
        MomentumAnalyzer, OrderBookAnalyzer,
        OrderBookSnapshot, PricePoint,
        ResolutionDecayModel,
    )
    from shared.intra_market_arb import IntraMarketArbitrage
    from shared.monte_carlo import MonteCarloPortfolio, position_from_signal
    from polymarket_bot.polymarket_client import PolymarketClient
    from polymarket_bot.order_signer import PolymarketOrderSigner, SigningUnavailable
    from kalshi_bot.kalshi_client import KalshiClient, KalshiOrder
except ImportError as _e:
    st.error(f"Missing dependency — check requirements: {_e}")
    st.stop()

# ── API key resolution (Streamlit secrets → env var → None) ──────────────────
import os

def _get_secret(name: str) -> str:
    """Read from st.secrets first, fall back to environment variable."""
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, "")


_UNSET = object()  # sentinel: distinguishes "never set" from a real None/[] result


def _run_coro_sync(coro, timeout: int = 30):
    """Run an async coroutine safely from a synchronous Streamlit context."""
    result_box: list = [_UNSET]
    exc_box: list = [None]

    def _target():
        try:
            result_box[0] = asyncio.run(coro)
        except Exception as e:
            exc_box[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError(f"API request timed out after {timeout} s — the host may be unreachable")
    if exc_box[0]:
        raise exc_box[0]
    if result_box[0] is _UNSET:
        # Thread finished without setting a result or raising — should be
        # unreachable, but fail loudly rather than returning the sentinel.
        raise RuntimeError("API call returned no data")
    return result_box[0]  # empty list / None are valid results


def _live_ai_analysis(api_key: str, markets: list[dict]) -> dict[str, dict]:
    """Batch-call Claude AI; returns {market_id: plain_dict}."""
    agent = ClaudeAgent(api_key=api_key, model="claude-opus-4-8")
    # Cap at 8 markets (single batch) to keep latency under 90 s timeout
    batch_input = [
        {"id": m["id"], "question": m["question"], "price": m["price"]}
        for m in markets[:8]
    ]
    analyses = _run_coro_sync(agent.analyze_batch(batch_input), timeout=90)
    return {
        a.market_id: {
            "prob":      float(a.estimated_probability),
            "conf":      float(a.confidence),
            "reasoning": str(a.reasoning),
            "factors":   [str(f) for f in a.key_factors],
            "flags":     [str(f) for f in a.uncertainty_flags],
        }
        for a in analyses
    }


# ── Polymarket live market fetching ───────────────────────────────────────────

_CAT_KEYWORDS: dict[str, list[str]] = {
    "crypto":   ["bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain",
                 "defi", "nft", "token", "solana", "doge", "xrp"],
    "finance":  ["fed", "interest rate", "inflation", "cpi", "gdp", "recession",
                 "s&p", "nasdaq", "stock", "economy", "fomc", "treasury",
                 "rate cut", "rate hike", "tariff"],
    "politics": ["president", "election", "congress", "senate", "trump", "biden",
                 "democrat", "republican", "vote", "law", "regulation",
                 "executive order", "supreme court", "house", "legislation"],
    "sports":   ["championship", "nba", "nfl", "mlb", "nhl", "world cup",
                 "super bowl", "playoffs", "league", "win", "team", "fifa",
                 "wimbledon", "olympic"],
}


def _guess_category(question: str) -> str:
    q = question.lower()
    for cat, kws in _CAT_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return cat
    return "finance"


def _days_until(iso_str: str | None) -> int:
    if not iso_str:
        return 30
    try:
        end = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return max(1, (end - datetime.now(timezone.utc)).days)
    except Exception:
        return 30


def _fetch_polymarket_markets_sync(api_key: str, limit: int = 25) -> list[dict]:
    """Fetch live Polymarket markets in a background thread (async-safe)."""
    async def _inner():
        client = PolymarketClient(api_key=api_key)
        try:
            raw = await client.get_markets(limit=limit, active_only=True)
            result = []
            for m in raw:
                price = float(m.mid_price)
                if not (0.03 <= price <= 0.97):
                    continue
                if m.volume_24h < 500:
                    continue
                result.append({
                    "id":           m.condition_id,
                    "platform":     "Polymarket",
                    "question":     m.question,
                    "price":        price,
                    "volume":       float(m.volume_24h),
                    "liquidity":    float(m.liquidity),
                    "days":         _days_until(m.end_date_iso),
                    "category":     _guess_category(m.question),
                    "trend":        0.0,
                    "yes_token_id": m.yes_token_id,
                    "no_token_id":  m.no_token_id,
                })
            return result
        finally:
            await client.close()

    return _run_coro_sync(_inner())


def _best_ask(book: dict) -> float | None:
    """Lowest ask price from a raw CLOB order book {asks: [{price,size}|[price,size]]}."""
    asks = (book or {}).get("asks") or []
    prices = []
    for a in asks:
        try:
            p = float(a["price"]) if isinstance(a, dict) else float(a[0])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if math.isfinite(p) and 0.0 < p < 1.0:
            prices.append(p)
    return min(prices) if prices else None


def _fetch_polymarket_arb_sync(api_key: str, markets: list[dict]) -> list[dict]:
    """For each market, fetch REAL YES and NO order-book asks (intra-market arb input).

    markets: live Polymarket dicts carrying yes_token_id / no_token_id.
    Returns dicts with yes_ask / no_ask populated from live books.
    """
    async def _inner():
        client = PolymarketClient(api_key=api_key)
        try:
            enriched = []
            for m in markets:
                yes_tid = m.get("yes_token_id", "")
                no_tid = m.get("no_token_id", "")
                if not yes_tid or not no_tid:
                    continue
                yes_book = await client.get_order_book(yes_tid)
                no_book = await client.get_order_book(no_tid)
                yes_ask = _best_ask(yes_book)
                no_ask = _best_ask(no_book)
                if yes_ask is None or no_ask is None:
                    continue
                enriched.append({
                    "id":           m["id"],
                    "condition_id": m["id"],
                    "question":     m["question"],
                    "yes_ask":      yes_ask,
                    "no_ask":       no_ask,
                    "yes_token_id": yes_tid,
                    "no_token_id":  no_tid,
                })
            return enriched
        finally:
            await client.close()

    return _run_coro_sync(_inner(), timeout=60)


def _fetch_kalshi_markets_sync(api_key_id: str, pem_content: str, limit: int = 20) -> list[dict]:
    """Fetch live Kalshi markets using RSA-PSS signed requests."""
    async def _inner():
        client = KalshiClient(
            api_key_id=api_key_id,
            private_key_content=pem_content,
            mock_mode=False,
        )
        try:
            raw = await client.get_markets(limit=limit, status="open")
            result = []
            drop_price = 0
            for m in raw:
                mid = m.mid_price
                if not (0.02 <= mid <= 0.98):
                    drop_price += 1
                    continue
                result.append({
                    "id":        m.ticker,
                    "platform":  "Kalshi",
                    "question":  m.title,
                    "price":     float(mid),
                    "volume":    float(m.volume),
                    "liquidity": float(m.open_interest) * float(mid),
                    "days":      _days_until(m.close_time),
                    "category":  _guess_category(m.title),
                    "trend":     0.0,
                })
            # If the API returned markets but filters dropped them all, surface why
            if raw and not result:
                raise RuntimeError(
                    f"API returned {len(raw)} markets but all filtered out on price "
                    f"({drop_price} dropped). "
                    f"Sample: {[(round(m.mid_price, 3), m.volume) for m in raw[:5]]}"
                )
            if not raw:
                raise RuntimeError(
                    "API returned 0 markets — check status param / response shape"
                )
            return result
        finally:
            await client.close()
    return _run_coro_sync(_inner())


@st.cache_data(ttl=60, show_spinner=False)
def _get_markets(
    poly_key: str = "",
    kalshi_key: str = "",
    kalshi_pem: str = "",
) -> tuple[list[dict], bool, bool, str, str]:
    """Return (markets, poly_live, kalshi_live, poly_err, kalshi_err)."""
    poly_live, kalshi_live = False, False
    poly_mkts: list[dict] = []
    kalshi_mkts: list[dict] = []
    poly_err = ""
    kalshi_err = ""

    # ── Polymarket ────────────────────────────────────────────────────────────
    if poly_key:
        try:
            fetched = _fetch_polymarket_markets_sync(poly_key, limit=50)
            if len(fetched) >= 3:
                poly_mkts, poly_live = fetched, True
            else:
                poly_err = f"Only {len(fetched)} markets returned (need ≥ 3, check filters)"
        except Exception as e:
            poly_err = f"{type(e).__name__}: {e}"
    if not poly_live:
        poly_mkts = [m for m in MOCK_MARKETS if m["platform"] == "Polymarket"]

    # ── Kalshi ────────────────────────────────────────────────────────────────
    if kalshi_key and kalshi_pem:
        try:
            fetched = _fetch_kalshi_markets_sync(kalshi_key, kalshi_pem, limit=50)
            if len(fetched) >= 2:
                kalshi_mkts, kalshi_live = fetched, True
            else:
                kalshi_err = f"Only {len(fetched)} markets returned (need ≥ 2)"
        except Exception as e:
            # Unwrap tenacity RetryError to show the real HTTP status
            real_exc = e
            try:
                from tenacity import RetryError
                if isinstance(e, RetryError):
                    real_exc = e.last_attempt.exception() or e
            except Exception:
                pass
            # Extract HTTP status + body snippet for 4xx errors
            detail = str(real_exc)
            try:
                if hasattr(real_exc, "response"):
                    r = real_exc.response
                    detail = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception:
                pass
            kalshi_err = f"{type(real_exc).__name__}: {detail}"
    elif kalshi_key and not kalshi_pem:
        kalshi_err = "KALSHI_PRIVATE_KEY_PEM not set — RSA auth required for all Kalshi endpoints"
    if not kalshi_live:
        kalshi_mkts = [m for m in MOCK_MARKETS if m["platform"] == "Kalshi"]

    return poly_mkts + kalshi_mkts, poly_live, kalshi_live, poly_err, kalshi_err


def _execute_live_polymarket_order(
    token_id: str, price: float, size_usd: float, side: str,
    private_key: str, sig_type: int, funder: str,
) -> dict:
    """Place a REAL EIP-712-signed Polymarket order. Only ever called from the
    gated live-trading path. Never runs in mock mode."""
    shares = size_usd / price if price > 0 else 0.0
    signer = PolymarketOrderSigner(
        private_key=private_key,
        signature_type=sig_type,
        funder=funder,
        mock_mode=False,
    )
    res = _run_coro_sync(signer.place_order_async(
        token_id=token_id,
        price=round(float(price), 4),
        size=round(float(shares), 4),
        side=side,
    ))
    return {"order_id": res.order_id, "status": res.status, "shares": round(shares, 4)}


def _execute_live_kalshi_order(
    ticker: str, price: float, size_usd: float, side: str,
    api_key_id: str, pem_content: str,
) -> dict:
    """Place a REAL RSA-PSS-signed Kalshi order. Only ever called from the gated
    live-trading path. Kalshi is US-regulated and US-accessible, so unlike
    Polymarket it works from a US account/IP. Never runs in mock mode.

    `side` is "yes" or "no". `price` is the YES mid (0–1); we convert to the
    contract price for the chosen side and a cents limit price."""
    contract_price = price if side == "yes" else (1.0 - price)
    contract_price = max(0.01, min(0.99, contract_price))
    count = max(1, int(size_usd / contract_price))
    limit_cents = int(round(contract_price * 100))

    async def _inner():
        client = KalshiClient(
            api_key_id=api_key_id,
            private_key_content=pem_content,
            mock_mode=False,
        )
        try:
            order = KalshiOrder(
                ticker=ticker,
                action="buy",
                side=side,
                count=count,
                limit_price=limit_cents,
            )
            resp = await client.place_order(order)
            o = (resp or {}).get("order", {}) if isinstance(resp, dict) else {}
            return {
                "order_id": o.get("order_id", f"kalshi_{int(time.time())}"),
                "status":   o.get("status", "submitted"),
                "count":    count,
            }
        finally:
            await client.close()

    return _run_coro_sync(_inner(), timeout=60)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Tighter top padding so content starts higher (feels faster/cleaner) */
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
/* Hide Streamlit chrome for a cleaner app-like feel */
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }

/* Gradient hero banner */
.hero {
    background: linear-gradient(110deg,#6366f1 0%,#8b5cf6 45%,#ec4899 100%);
    border-radius: 16px; padding: 22px 28px; margin-bottom: 18px;
    box-shadow: 0 8px 28px rgba(99,102,241,.28);
}
.hero h1 { color:#fff; font-size:1.65rem; font-weight:800; margin:0; letter-spacing:-.02em; }
.hero p  { color:#eef; font-size:.92rem; margin:.3rem 0 0; opacity:.92; }

.card { background:#1e1e2e; border-radius:12px; padding:16px 20px;
        margin-bottom:10px; border-left:4px solid #6366f1;
        box-shadow: 0 2px 10px rgba(0,0,0,.18); }
.prob-big { font-size:3.4rem; font-weight:900; line-height:1.05; letter-spacing:-.02em; }
.signal-yes { color:#00d26a; font-weight:700; }
.signal-no  { color:#ff4b4b; font-weight:700; }
.signal-hold{ color:#888888; font-weight:600; }

/* Metric cards: subtle panel + hover lift */
div[data-testid="stMetric"] {
    background:#1a1a28; border:1px solid #2a2a3c; border-radius:12px;
    padding:14px 16px; transition: transform .12s ease, border-color .12s ease;
}
div[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color:#6366f1; }

/* Pill-style tabs */
button[data-baseweb="tab"] { font-weight:600; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }

/* Rounder dataframes */
div[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }

/* ── Remove sidebar collapse/expand entirely ──────────────────────────────────
   Hide every variant of the collapse toggle across Streamlit versions so the
   sidebar is permanently visible and can never be accidentally hidden. */
div[data-testid="stSidebarCollapsedControl"],
button[data-testid="stSidebarCollapsedControl"],
section[data-testid="stSidebarCollapsedControl"],
div[data-testid="collapsedControl"],
button[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCloseButton"] {
    display: none !important;
}
""", unsafe_allow_html=True)

SIGNAL_ICONS = {"BUY_YES": "🟢 BUY YES", "BUY_NO": "🔴 BUY NO", "HOLD": "⚪ HOLD"}
SIG_CSS      = {"BUY_YES": "signal-yes",  "BUY_NO": "signal-no",  "HOLD": "signal-hold"}
SIG_COLOR    = {"BUY_YES": "#00d26a",     "BUY_NO": "#ff4b4b",    "HOLD": "#888888"}

# ── mock market data ──────────────────────────────────────────────────────────
MOCK_MARKETS = [
    {"id": "poly_fed_jul",   "platform": "Polymarket",
     "question": "Will the Fed cut rates at the July 2026 FOMC?",
     "price": 0.38, "volume": 2_450_000, "liquidity": 180_000,
     "days": 25,  "category": "finance", "trend": +0.003},
    {"id": "poly_btc_120k",  "platform": "Polymarket",
     "question": "Will Bitcoin exceed $120k before end of 2026?",
     "price": 0.52, "volume": 5_100_000, "liquidity": 420_000,
     "days": 202, "category": "crypto",  "trend": +0.006},
    {"id": "poly_ai_exec",   "platform": "Polymarket",
     "question": "Will Trump sign a new AI regulation executive order before July?",
     "price": 0.21, "volume": 890_000,   "liquidity": 75_000,
     "days": 18,  "category": "politics","trend": -0.002},
    {"id": "poly_nba",       "platform": "Polymarket",
     "question": "Will the Boston Celtics win the 2026 NBA Championship?",
     "price": 0.34, "volume": 3_200_000, "liquidity": 260_000,
     "days": 12,  "category": "sports",  "trend": +0.008},
    {"id": "poly_recession", "platform": "Polymarket",
     "question": "Will the US enter a recession in 2026?",
     "price": 0.29, "volume": 1_800_000, "liquidity": 145_000,
     "days": 202, "category": "finance", "trend": -0.001},
    {"id": "poly_eth_5k",    "platform": "Polymarket",
     "question": "Will Ethereum exceed $5,000 before September 2026?",
     "price": 0.61, "volume": 2_700_000, "liquidity": 210_000,
     "days": 90,  "category": "crypto",  "trend": +0.004},
    {"id": "kalshi_fed",     "platform": "Kalshi",
     "question": "Federal Reserve rate cut at July 2026 FOMC?",
     "price": 0.41, "volume": 980_000,   "liquidity": 85_000,
     "days": 25,  "category": "finance", "trend": +0.002},
    {"id": "poly_brazil",    "platform": "Polymarket",
     "question": "Will Brazil win the 2026 FIFA World Cup?",
     "price": 0.18, "volume": 1_200_000, "liquidity": 95_000,
     "days": 45,  "category": "sports",  "trend": -0.003},
    {"id": "poly_sp500",     "platform": "Polymarket",
     "question": "Will the S&P 500 set a new all-time high before September?",
     "price": 0.67, "volume": 980_000,   "liquidity": 88_000,
     "days": 78,  "category": "finance", "trend": +0.002},
    {"id": "kalshi_btc100k", "platform": "Kalshi",
     "question": "Bitcoin above $100,000 on August 1 2026?",
     "price": 0.55, "volume": 1_100_000, "liquidity": 96_000,
     "days": 50,  "category": "crypto",  "trend": +0.005},
]

_REASONING = {
    "finance":  "Fed dot-plot and PCE trajectory suggest markets are under-pricing a cut. Employment data remains solid but leading indicators are softening, giving the Fed political cover to ease.",
    "crypto":   "On-chain accumulation by large wallets is at a 6-month high. Spot ETF inflow trend is accelerating and not yet reflected in current pricing.",
    "politics": "Historical base rate for this type of executive action is ~28%. Current congressional dynamics and the administration's packed legislative calendar make a near-term signing unlikely.",
    "sports":   "Advanced metrics (RAPTOR, EPM) favour this team. Recent form shows a 4-win streak with improving defensive efficiency and key opponent injury concerns.",
}
_FACTORS = {
    "finance":  ["Fed dot-plot signals", "PCE inflation trend", "Primary-dealer consensus"],
    "crypto":   ["Whale accumulation pace", "Options IV skew", "ETF inflow acceleration"],
    "politics": ["Historical base rate 28%", "Legislative calendar congestion", "Timeline risk"],
    "sports":   ["RAPTOR rating differential", "4-game win streak", "Defensive efficiency rank"],
}
_FLAGS = {
    "finance":  ["Surprise CPI print", "FOMC dissent risk"],
    "crypto":   ["Regulatory shock", "Large-holder sell pressure"],
    "politics": ["Legal challenge", "Policy reversal"],
    "sports":   ["Key player injury", "Bracket luck"],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _mock_ai(mkt: dict) -> dict:
    rng  = random.Random(mkt["id"])
    bias = {"finance": 0.05, "crypto": 0.07, "politics": -0.04, "sports": 0.04}
    cat  = mkt["category"]
    est  = float(max(0.05, min(0.95, mkt["price"] + bias.get(cat, 0) + rng.gauss(0, 0.04))))
    return {
        "prob":      est,
        "conf":      float(rng.uniform(0.58, 0.88)),
        "reasoning": _REASONING.get(cat, "Analysis in progress."),
        "factors":   _FACTORS.get(cat, []),
        "flags":     _FLAGS.get(cat, []),
    }


def _history(mkt: dict, n: int = 30) -> list[float]:
    rng   = random.Random(mkt["id"] + "h")
    price = mkt["price"] - mkt.get("trend", 0) * n
    out   = []
    for _ in range(n):
        price = max(0.03, min(0.97, price + mkt.get("trend", 0) + rng.gauss(0, 0.007)))
        out.append(float(price))
    return out


# ── full analysis pipeline — returns only plain primitives ────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def _run_analysis(markets: list[dict], anthropic_key: str = "") -> tuple[list[dict], bool, str]:
    # Fetch AI probability estimates — live Claude when key is present, else mock
    live_ai: dict[str, dict] = {}
    using_live = False
    ai_err = ""
    if anthropic_key:
        try:
            live_ai = _live_ai_analysis(anthropic_key, markets)
            using_live = bool(live_ai)
        except Exception as _live_err:
            live_ai = {}
            using_live = False
            ai_err = f"{type(_live_err).__name__}: {_live_err}"

    bay   = BayesianEstimator()
    kelly = KellyCriterion(max_fraction=0.15)
    ob    = OrderBookAnalyzer()
    mom   = MomentumAnalyzer()
    ens   = EnsemblePredictor(kelly)
    decay = ResolutionDecayModel()
    ls    = LongshotBiasDetector()
    cat   = CategoryEdgeModel()
    qsc   = MarketQualityScorer()
    tmp   = TemporalPatternSignal()

    out = []
    for mkt in markets:
        ai_d = live_ai.get(mkt["id"]) or _mock_ai(mkt)
        hist = _history(mkt)

        ob_snap = OrderBookSnapshot(
            bids=[(mkt["price"] - 0.01 * i, float(random.randint(15_000, 80_000)))
                  for i in range(1, 4)],
            asks=[(mkt["price"] + 0.01 * i, float(random.randint(12_000, 65_000)))
                  for i in range(1, 4)],
        )
        pp = [PricePoint(price=p, volume=50_000.0, timestamp=float(i))
              for i, p in enumerate(hist)]

        ev = [((ai_d["prob"] - mkt["price"]) * 2, ai_d["conf"])]
        bay_m, bay_lo, bay_hi = bay.estimate(mkt["price"], ev)

        ob_sigs  = ob.analyze(ob_snap)
        mom_sigs = mom.analyze(pp)

        raw = ens.predict(
            market_price=mkt["price"],
            ai_probability=ai_d["prob"],
            ai_confidence=ai_d["conf"],
            bayesian_estimate=float(bay_m),
            microstructure_signals=ob_sigs,
            momentum_signals=mom_sigs,
            sentiment_signal=None,
        )

        adj_conf, adj_prob = decay.adjust_for_time(
            mkt["days"], raw.confidence,
            raw.estimated_probability, market_price=mkt["price"],
        )
        adj_prob  = ls.adjust_probability(mkt["price"], float(adj_prob))
        q_score   = qsc.score(mkt["volume"], mkt["liquidity"], 0.0, mkt["price"])
        adj_conf  = float(adj_conf) * (0.5 + 0.5 * float(q_score))
        min_edge  = tmp.adjusted_min_edge(cat.adjusted_min_edge(mkt["question"], 0.03))
        edge      = float(adj_prob) - mkt["price"]
        # Confidence-scaled Kelly: shrink the bet by model confidence so
        # uncertain edges deploy less capital (matches market_analyzer).
        kf        = float(kelly.compute(float(adj_prob), mkt["price"])) * float(adj_conf)

        signal = ("HOLD" if abs(edge) < min_edge or adj_conf < 0.50
                  else ("BUY_YES" if edge > 0 else "BUY_NO"))

        sigs = [
            {"name": str(s.name), "value": float(s.value),
             "confidence": float(s.confidence),
             "weight": float(s.weight if s.weight is not None else 1.0)}
            for s in raw.signals if s.value != 0.0
        ]

        out.append({
            "id": mkt["id"], "platform": mkt["platform"],
            "question": mkt["question"], "price": mkt["price"],
            "volume": mkt["volume"], "liquidity": mkt["liquidity"],
            "days": mkt["days"], "category": mkt["category"],
            "ai_prob": ai_d["prob"], "ai_conf": ai_d["conf"],
            "ai_reasoning": ai_d["reasoning"],
            "ai_factors": ai_d["factors"], "ai_flags": ai_d["flags"],
            "bayesian": float(bay_m), "bay_lo": float(bay_lo), "bay_hi": float(bay_hi),
            "prob": float(adj_prob), "conf": float(adj_conf),
            "edge": float(edge), "kelly": float(kf),
            "quality": float(q_score), "signal": signal,
            "min_edge": float(min_edge), "history": hist, "signals": sigs,
            "yes_token_id": str(mkt.get("yes_token_id", "")),
            "no_token_id":  str(mkt.get("no_token_id", "")),
        })
    return out, using_live, ai_err


# ── sidebar ───────────────────────────────────────────────────────────────────
_anthropic_key   = _get_secret("ANTHROPIC_API_KEY")
_poly_key        = _get_secret("POLYMARKET_API_KEY")
_kalshi_key      = _get_secret("KALSHI_API_KEY")
_kalshi_pem      = _get_secret("KALSHI_PRIVATE_KEY_PEM")
_poly_pk         = _get_secret("POLYMARKET_PRIVATE_KEY")
_poly_sig_type   = int(_get_secret("POLYMARKET_SIGNATURE_TYPE") or 0)
_poly_funder     = _get_secret("POLYMARKET_FUNDER")
_supabase_url    = _get_secret("SUPABASE_URL")
_supabase_key    = _get_secret("SUPABASE_KEY")

# Live execution is possible if EITHER platform has its credentials:
#   Polymarket → wallet private key (US accounts get 403 from the CLOB)
#   Kalshi     → API key id + RSA PEM (US-regulated, US-accessible)
_kalshi_live_creds = bool(_kalshi_key and _kalshi_pem)
_has_live_creds    = bool(_poly_pk or _kalshi_live_creds)

# Inject Supabase creds as env vars so shared/cloud_store.py (no streamlit import) can reach them
import os as _os
if _supabase_url: _os.environ["SUPABASE_URL"] = _supabase_url
if _supabase_key: _os.environ["SUPABASE_KEY"] = _supabase_key

if "paper_ledger" not in st.session_state:
    st.session_state.paper_ledger = []
if "live_ledger" not in st.session_state:
    st.session_state.live_ledger = []

# ── Persist user settings across browser refreshes via URL query params ────────
# On first load: read saved values from URL and pre-populate session_state so
# widgets pick them up. On every run: write current values back into the URL.
_qp = st.query_params
def _qp_float(k: str, default: float) -> float:
    try: return float(_qp.get(k, default))
    except: return default
def _qp_bool(k: str, default: bool) -> bool:
    v = _qp.get(k)
    return (v == "1") if v is not None else default

_VALID_RL = ["Off", "Every 1 min", "Every 5 min", "Every 15 min", "Every 30 min"]

if "poly_budget"       not in st.session_state:
    st.session_state.poly_budget       = _qp_float("pb",  10.0)
if "kalshi_budget"     not in st.session_state:
    st.session_state.kalshi_budget     = _qp_float("kb",  10.0)
if "sidebar_min_conf"  not in st.session_state:
    st.session_state.sidebar_min_conf  = _qp_float("mc",  0.55)
if "auto_paper"        not in st.session_state:
    st.session_state.auto_paper        = _qp_bool("ap",   False)
if "refresh_label"     not in st.session_state:
    _rl = _qp.get("rl", "Off")
    st.session_state.refresh_label     = _rl if _rl in _VALID_RL else "Off"
# Live toggles persist across refreshes (user opted in). Restored only if live
# credentials are present — never auto-arm live trading without them.
if "live_trading"      not in st.session_state:
    st.session_state.live_trading      = _qp_bool("lt",   False) and _has_live_creds
if "auto_live"         not in st.session_state:
    st.session_state.auto_live         = _qp_bool("al",   False) and _has_live_creds

# ── AI brain singleton — one instance per browser session, survives reruns ────
import types as _types
if "_brain" not in st.session_state:
    _brain_ens = EnsemblePredictor()
    _brain_inst = LearningEngine(_brain_ens, state_file=Path("data/learning_state.json"))
    _brain_inst.load()
    st.session_state._brain = _brain_inst

def _to_brain_input(a: dict):
    """Wrap an analysis dict into the duck-typed object LearningEngine expects."""
    sigs = [
        _types.SimpleNamespace(
            name=s["name"], value=float(s["value"]),
            confidence=float(s["confidence"]),
            weight=float(s.get("weight", 1.0)),
        )
        for s in a.get("signals", [])
    ]
    return _types.SimpleNamespace(
        market_id=a["id"],
        signal=a["signal"],
        market_price=float(a["price"]),
        ensemble=_types.SimpleNamespace(signals=sigs) if sigs else None,
    )

# ── demo_mode must be initialized before data loading (used for _eff_* keys) ──
_any_key = bool(
    _get_secret("ANTHROPIC_API_KEY") or
    _get_secret("POLYMARKET_API_KEY") or
    _get_secret("KALSHI_API_KEY")
)
if "demo_mode" not in st.session_state:
    if "demo" in _qp:
        st.session_state.demo_mode = _qp["demo"] == "1"
    else:
        st.session_state.demo_mode = not _any_key

# Stamp the start of every full script run so the fragment can detect timer-fires
import time as _time_mod
st.session_state._last_full_run = _time_mod.time()

with st.sidebar:
    st.title("🤖 Polymarket AI Bot")
    st.markdown("**API Keys**")
    st.markdown("✅ Anthropic key set" if _anthropic_key  else "⚪ No Anthropic key → mock AI")
    st.markdown("✅ Polymarket key set" if _poly_key       else "⚪ No Polymarket key → mock data")
    st.markdown("✅ Kalshi keys set"    if (_kalshi_key and _kalshi_pem) else "⚪ No Kalshi keys → mock data")
    st.markdown("✅ Supabase connected" if (_supabase_url and _supabase_key) else "⚪ No Supabase → brain resets on restart")
    if not _supabase_url:
        _raw_url = ""
        try:
            _raw_url = str(st.secrets.get("SUPABASE_URL", ""))
        except Exception:
            pass
        if _raw_url:
            st.caption(f"⚠️ Secret found but under wrong name. URL starts: `{_raw_url[:12]}…`")
        else:
            st.caption("ℹ️ Reboot the app after saving secrets (Manage app → Reboot)")
    st.caption("Live connection status is shown in the main panel →")
    st.divider()
    st.caption("💰 Budget, confidence & refresh controls are in the main panel ↗")
    st.divider()
    if st.button("🔄 Re-run Analysis", use_container_width=True, key="rerun_sidebar"):
        st.cache_data.clear()
        st.rerun()

    # ── live trading gate (DANGER) ──────────────────────────────────────────
    st.divider()
    st.markdown("### ⚠️ Live Trading")
    if not _poly_pk:
        st.caption("📝 Paper mode only — add `POLYMARKET_PRIVATE_KEY` to secrets "
                   "to unlock live execution (never paste keys into chat).")
        st.caption("Toggle visible in the main panel once wallet key is set.")
    else:
        st.caption("Wallet key detected. Toggle Live trading in the main panel ↗")
        if st.session_state.get("live_trading", False):
            st.error("⚡ LIVE MODE ACTIVE — real money at risk!")
            st.caption(f"Wallet sig type: {_poly_sig_type} "
                       f"({'EOA' if _poly_sig_type == 0 else 'proxy/safe'})")

    _missing = [k for k, v in {
        "ANTHROPIC_API_KEY": _anthropic_key,
        "POLYMARKET_API_KEY": _poly_key,
        "KALSHI_API_KEY": _kalshi_key,
        "KALSHI_PRIVATE_KEY_PEM": _kalshi_pem,
        "SUPABASE_URL": _supabase_url,
        "SUPABASE_KEY": _supabase_key,
    }.items() if not v]
    if _missing:
        st.divider()
        st.caption("Missing secrets (add via Manage app → Secrets):")
        for _k in _missing:
            st.caption(f"  • {_k}")

# ── static hero banner (outside fragment — never disrupted by timer refresh) ───
_dm_static = st.session_state.get("demo_mode", True)
_hero_sub_static = (
    "🔵 Demo mode — mock data, no keys required"
    if _dm_static else
    "🟢 Live mode — real API connections active"
)
st.markdown(f"""
<div class="hero">
  <h1>📊 Polymarket + Kalshi AI Trading Dashboard</h1>
  <p>Bayesian × Kelly × momentum × arbitrage, fused by a self-learning signal brain. &nbsp;|&nbsp; {_hero_sub_static}</p>
</div>
""", unsafe_allow_html=True)

# ── demo toggle + run button (outside fragment — must not be disrupted) ────────
# IMPORTANT: every control below binds DIRECTLY to its session_state key with no
# conflicting `value=`/default. Passing both a `value=` and a key that is already
# in session_state makes Streamlit re-seed the widget on some reruns, which is
# what caused "▶ Run" to silently reset Demo / Live / Auto-execute. Likewise,
# auto_live is ALWAYS rendered (disabled when unavailable) — a hidden keyed
# widget has its state dropped by Streamlit, which broke auto-execution.
def _on_demo_change():
    # Demo flips force a fresh data fetch under the new mode.
    st.cache_data.clear()

_bar1, _bar2, _bar3 = st.columns([3, 2, 1])
with _bar1:
    # Budget controls here so they're always visible (sidebar may be collapsed)
    _bc1, _bc2, _bc3 = st.columns(3)
    with _bc1:
        st.number_input("Polymarket budget ($)", min_value=1.0, step=5.0,
                        key="poly_budget", help="Max $ per Polymarket trade")
    with _bc2:
        st.number_input("Kalshi budget ($)", min_value=1.0, step=5.0,
                        key="kalshi_budget", help="Max $ per Kalshi trade")
    with _bc3:
        st.slider("Min confidence", 0.40, 0.90, step=0.05, key="sidebar_min_conf")
with _bar2:
    st.toggle(
        "🔵 Demo mode",
        key="demo_mode",
        on_change=_on_demo_change,
        help="ON = safe simulated data. OFF = live API connections.",
    )
with _bar3:
    if st.button("▶ Run", type="primary", use_container_width=True, key="run_main"):
        # Only refresh data — never touch widget state.
        st.cache_data.clear()
        st.rerun()

# ── bot status row (outside fragment — toggles must survive timer refresh) ─────
_bs1, _bs2, _bs3, _bs4, _bs5 = st.columns(5)
with _bs1:
    st.toggle(
        "📝 Auto paper trades",
        key="auto_paper",
        help="Bot automatically logs a paper trade for every Buy signal "
             "that passes your confidence filter on each analysis run.",
    )
with _bs2:
    if _has_live_creds:
        st.toggle(
            "🚨 Live trading (real $)",
            key="live_trading",
            help="Switch from Paper to real-money execution. Polymarket needs "
                 "POLYMARKET_PRIVATE_KEY (US accounts are geo-blocked → 403); "
                 "Kalshi needs KALSHI_API_KEY + KALSHI_PRIVATE_KEY_PEM (US-OK). "
                 "Start with $10. Both bots default mock_mode=true — never "
                 "enable without weeks of mock testing and verified API connections.",
        )
        if st.session_state.get("live_trading", False):
            st.error("LIVE MODE ON", icon="🚨")
    else:
        st.caption("📝 Paper only\n(add keys for live)")
with _bs3:
    # Always render so Streamlit never drops the auto_live state. Disable it
    # (rather than hide) when live trading is off or no live creds are present.
    _live_on = st.session_state.get("live_trading", False)
    _al_ready = bool(_has_live_creds and _live_on)
    st.toggle(
        "🤖 Auto-execute LIVE",
        key="auto_live",
        disabled=not _al_ready,
        help="Bot auto-places REAL orders on every refresh cycle — Polymarket "
             "and/or Kalshi, whichever has credentials. Uses the per-platform "
             "budgets above. Stays ON across browser refreshes — disable it "
             "manually to stop the bot.",
    )
    if _al_ready and st.session_state.get("auto_live", False):
        st.error("Auto-LIVE ON", icon="⚡")
    elif not _al_ready:
        st.caption("Enable Live trading\nto unlock auto-live")
with _bs4:
    _n_p_outer = len(st.session_state.paper_ledger)
    _n_l_outer = len(st.session_state.live_ledger)
    _trade_txt = f"📋 Paper: {_n_p_outer}"
    if _n_l_outer:
        _trade_txt += f" · 🚨 Live: {_n_l_outer}"
    st.info(f"**Trades:** {_trade_txt}")
with _bs5:
    st.selectbox(
        "⏱ Auto-refresh",
        ["Off", "Every 1 min", "Every 5 min", "Every 15 min", "Every 30 min"],
        key="refresh_label",
        help="Run the bot 24/7 — fetches fresh data and auto-executes trades "
             "on the chosen interval.",
    )

# ── refresh interval computation (outside fragment — drives run_every param) ──
_REFRESH_MAP = {"Off": None, "Every 1 min": 60, "Every 5 min": 300,
                "Every 15 min": 900, "Every 30 min": 1800}
_refresh_secs = _REFRESH_MAP.get(st.session_state.get("refresh_label", "Off"))


# ══════════════════════════════════════════════════════════════════════════════
# DATA + DISPLAY FRAGMENT
# Only this fragment re-executes on timer ticks — outer controls are untouched.
# ══════════════════════════════════════════════════════════════════════════════
@st.fragment(run_every=_refresh_secs)
def _live_dashboard():
    # Read control values from session_state (set by widgets rendered above)
    _dm    = st.session_state.get("demo_mode", True)
    _ap    = st.session_state.get("auto_paper", False)
    _lt    = st.session_state.get("live_trading", False)
    _al    = st.session_state.get("auto_live", False)
    poly_budget   = st.session_state.get("poly_budget",   10.0)
    kalshi_budget = st.session_state.get("kalshi_budget", 10.0)
    budget        = poly_budget + kalshi_budget
    min_conf = st.session_state.get("sidebar_min_conf", 0.55)

    # Clear cache when this is a timer-fired re-execution (not the initial load)
    _is_timer = time.time() - st.session_state.get("_last_full_run", 0) > 5
    if _is_timer and _refresh_secs:
        st.cache_data.clear()

    # Compute effective API keys (demo mode forces mock regardless of key presence)
    _eff_anthropic  = "" if _dm else _anthropic_key
    _eff_poly       = "" if _dm else _poly_key
    _eff_kalshi     = "" if _dm else _kalshi_key
    _eff_kalshi_pem = "" if _dm else _kalshi_pem

    # ── load data ─────────────────────────────────────────────────────────────
    try:
        with st.spinner("Fetching markets and running predictive models…"):
            _source_markets, _poly_live, _kalshi_live, _poly_err, _kalshi_err = \
                _get_markets(_eff_poly, _eff_kalshi, _eff_kalshi_pem)
            all_analyses, _live_ai_mode, _ai_err = _run_analysis(_source_markets, _eff_anthropic)
    except Exception as exc:
        st.error("Analysis pipeline failed — see details below.")
        st.exception(exc)
        return

    # ── connection status chips ────────────────────────────────────────────────
    if _dm:
        st.info(
            "🔵 **Demo mode** — all APIs intentionally bypassed, using simulated data. "
            "Turn off the **Demo mode** toggle above to connect real APIs.",
            icon="ℹ️",
        )
    else:
        _CHIP = ("display:inline-block;border-radius:20px;padding:4px 16px;"
                 "font-size:.82rem;font-weight:600;margin:2px 6px 2px 0;"
                 "border:1px solid;white-space:nowrap;")

        def _conn_chip(connected: bool, has_key: bool, label: str) -> str:
            if connected:
                s = "background:#052e16;color:#00d26a;border-color:#00d26a;"
                return f'<span style="{_CHIP}{s}">🟢 {label} live</span>'
            if has_key:
                s = "background:#422006;color:#fb923c;border-color:#fb923c;"
                return f'<span style="{_CHIP}{s}">🟡 {label} — key set, connection failed</span>'
            s = "background:#0f172a;color:#64748b;border-color:#334155;"
            return f'<span style="{_CHIP}{s}">⚫ {label} — no API key</span>'

        _chips = "".join([
            _conn_chip(_live_ai_mode, bool(_anthropic_key), "Claude AI"),
            _conn_chip(_poly_live,    bool(_poly_key),       "Polymarket"),
            _conn_chip(_kalshi_live,  bool(_kalshi_key and _kalshi_pem), "Kalshi"),
        ])
        st.markdown(f'<div style="padding:4px 0 10px">{_chips}</div>', unsafe_allow_html=True)

    # ── API / connection warnings ──────────────────────────────────────────────
    if _eff_poly and not _poly_live:
        st.warning(
            f"**Polymarket live fetch failed** — using mock markets.  \n"
            f"**Error:** `{_poly_err}`  \n"
            "Common causes: API endpoint changed, network restriction, or invalid key format."
        )
    if _eff_kalshi and not _kalshi_live:
        st.warning(
            f"**Kalshi live fetch failed** — using mock markets.  \n"
            f"**Error:** `{_kalshi_err}`"
        )
    if _eff_anthropic and not _live_ai_mode:
        _err_low = _ai_err.lower()
        if "usage limit" in _err_low or "api usage" in _err_low:
            import re as _re
            _reset = (_re.search(r'(\d{4}-\d{2}-\d{2})', _ai_err) or type("", (), {"group": lambda *a: "July 1"})()).group(1)
            st.warning(
                f"**Anthropic API monthly limit reached** — Claude AI unavailable until **{_reset}**.  \n"
                "Analysis is running on Bayesian + momentum signals (no AI). "
                "Switch to **Demo mode** (toggle above) to silence this warning.",
                icon="💳",
            )
        else:
            _ai_err_msg = f"  \n**Error:** `{_ai_err}`" if _ai_err else ""
            st.warning(
                "**ANTHROPIC_API_KEY set but Claude AI call failed** — using mock analysis.  \n"
                f"Check key validity at console.anthropic.com.{_ai_err_msg}"
            )

    analyses = [a for a in all_analyses if a["conf"] >= min_conf or a["signal"] == "HOLD"]

    # ── P&L helpers ───────────────────────────────────────────────────────────
    _price_map = {a["question"][:50]: a["price"] for a in analyses}

    def _trade_pnl(row: dict) -> float:
        """Unrealised P&L in dollars for a paper trade row."""
        ep_yes = float(row.get("entry_num") or 0)
        if ep_yes <= 0:
            return 0.0
        size  = float(row.get("Size $") or 0)
        dir_  = str(row.get("Dir", "YES"))
        ep    = ep_yes if dir_ == "YES" else (1.0 - ep_yes)
        if ep <= 0:
            return 0.0
        q       = str(row.get("Question", ""))
        yes_now = _price_map.get(q, ep_yes)
        cp      = yes_now if dir_ == "YES" else (1.0 - yes_now)
        return round((size / ep) * cp - size, 2)

    def _render_order_book(key_prefix: str = "") -> None:
        """Render unified order book: paper + live positions with live P&L."""
        paper = st.session_state.paper_ledger
        live  = st.session_state.live_ledger
        if not paper and not live:
            st.caption("No trades recorded yet.")
            return

        # ── summary metrics ───────────────────────────────────────────────────
        _p_invested = sum(float(r.get("Size $", 0)) for r in paper)
        _p_pnl      = sum(_trade_pnl(r) for r in paper)
        _l_invested = sum(float(r.get("Size $", 0)) for r in live)
        _l_pnl      = sum(_trade_pnl(r) for r in live)
        _tot_inv    = _p_invested + _l_invested
        _tot_pnl    = _p_pnl + _l_pnl
        _roi        = _tot_pnl / _tot_inv if _tot_inv else 0.0

        _m1, _m2, _m3, _m4, _m5 = st.columns(5)
        _m1.metric("Paper trades",  len(paper))
        _m2.metric("Live orders",   len(live))
        _m3.metric("Total invested", f"${_tot_inv:.2f}")
        _m4.metric("Total P&L",     f"${_tot_pnl:+.2f}",
                   delta_color="normal" if _tot_pnl >= 0 else "inverse")
        _m5.metric("ROI",           f"{_roi:+.1%}",
                   delta_color="normal" if _roi >= 0 else "inverse")

        # ── paper positions ───────────────────────────────────────────────────
        if paper:
            st.markdown("**📝 Paper Positions**")
            df_p = pd.DataFrame(paper)
            df_p["P&L $"]    = [_trade_pnl(r) for r in paper]
            df_p["Current %"] = df_p.apply(
                lambda r: f"{_price_map.get(str(r.get('Question','')), float(r.get('entry_num',0))):.0%}",
                axis=1,
            )
            df_p["ROI"] = df_p.apply(
                lambda r: f"{r['P&L $'] / r['Size $']:+.1%}" if r["Size $"] else "—", axis=1
            )
            _show_p = [c for c in ["Platform","Question","Dir","Size $","Entry",
                                    "Current %","AI Est.","Edge","P&L $","ROI","Time","Mode"]
                       if c in df_p.columns]
            st.dataframe(
                df_p[_show_p], use_container_width=True, hide_index=True,
                column_config={
                    "P&L $":  st.column_config.NumberColumn("P&L $", format="%+.2f"),
                    "Size $": st.column_config.NumberColumn("Size $", format="$%.2f"),
                },
            )
            if st.button("Clear paper trades", key=f"clear_paper_{key_prefix}"):
                st.session_state.paper_ledger.clear()
                st.rerun()

        # ── live positions ────────────────────────────────────────────────────
        if live:
            st.markdown("**🚨 Live Orders (real money)**")
            df_l = pd.DataFrame(live)
            if "entry_num" in df_l.columns:
                df_l["P&L $"] = [_trade_pnl(r) for r in live]
                df_l["Current %"] = df_l.apply(
                    lambda r: f"{_price_map.get(str(r.get('Question','')), float(r.get('entry_num',0))):.0%}",
                    axis=1,
                )
            _show_l = [c for c in ["Platform","Question","Dir","Size $","Entry",
                                    "Current %","P&L $","Order ID","Status","Time","Mode"]
                       if c in df_l.columns]
            st.dataframe(
                df_l[_show_l], use_container_width=True, hide_index=True,
                column_config={
                    "P&L $":  st.column_config.NumberColumn("P&L $", format="%+.2f"),
                    "Size $": st.column_config.NumberColumn("Size $", format="$%.2f"),
                },
            )
            if st.button("Clear live ledger", key=f"clear_live_{key_prefix}"):
                st.session_state.live_ledger.clear()
                st.rerun()

    def _render_paper_ledger(key_prefix: str = "") -> None:
        """Backwards-compat wrapper — renders the full order book."""
        _render_order_book(key_prefix)

    # ── auto paper trades ─────────────────────────────────────────────────────
    if _ap:
        _auto_added = 0
        for _a in analyses:
            if _a["signal"] == "HOLD" or _a["kelly"] <= 0:
                continue
            _plat_budget = poly_budget if _a["platform"] == "Polymarket" else kalshi_budget
            _size = round(min(float(_plat_budget) * _a["kelly"], float(_plat_budget) * 0.15), 2)
            if _size < 1.0:
                continue
            _dir = "YES" if _a["signal"] == "BUY_YES" else "NO"
            if any(p.get("Question", "")[:40] == _a["question"][:40] and p.get("Dir") == _dir
                   for p in st.session_state.paper_ledger):
                continue
            st.session_state.paper_ledger.append({
                "Platform":  _a["platform"],
                "Question":  _a["question"][:50],
                "Dir":       _dir,
                "Size $":    _size,
                "entry_num": _a["price"],
                "Entry":     f"{_a['price']:.0%}",
                "AI Est.":   f"{_a['prob']:.0%}",
                "Edge":      f"{_a['edge']:+.1%}",
                "Conf":      f"{_a['conf']:.0%}",
                "Time":      time.strftime("%H:%M:%S"),
                "Mode":      "🤖 Auto",
            })
            st.session_state._brain.on_trade_placed(_to_brain_input(_a))
            _auto_added += 1
        if _auto_added:
            st.toast(
                f"🤖 Bot auto-executed {_auto_added} paper trade{'s' if _auto_added != 1 else ''}",
                icon="📝",
            )

    # ── auto LIVE trades (Polymarket only — gated behind live_trading toggle) ──
    if _al and _lt and _poly_pk and not _dm:
        _live_added = 0
        for _a in analyses:
            if _a["signal"] == "HOLD" or _a["kelly"] <= 0 or _a["platform"] != "Polymarket":
                continue
            _tok = _a.get("yes_token_id") if _a["signal"] == "BUY_YES" else _a.get("no_token_id")
            if not _tok:
                continue
            _size = round(min(float(poly_budget) * _a["kelly"], float(poly_budget) * 0.15), 2)
            if _size < 1.0:
                continue
            _dir = "YES" if _a["signal"] == "BUY_YES" else "NO"
            if any(l.get("Question", "")[:40] == _a["question"][:40] and l.get("Dir") == _dir
                   for l in st.session_state.live_ledger):
                continue
            try:
                _res = _execute_live_polymarket_order(
                    token_id=_tok, price=_a["price"], size_usd=_size,
                    side="BUY", private_key=_poly_pk,
                    sig_type=_poly_sig_type, funder=_poly_funder,
                )
                st.session_state.live_ledger.append({
                    "Platform": "Polymarket",
                    "Question": _a["question"][:50],
                    "Dir":      _dir,
                    "Size $":   round(_size, 2),
                    "entry_num": _a["price"],
                    "Entry":    f"{_a['price']:.0%}",
                    "AI Est.":  f"{_a['prob']:.0%}",
                    "Edge":     f"{_a['edge']:+.1%}",
                    "Order ID": _res["order_id"],
                    "Status":   _res["status"],
                    "Time":     time.strftime("%H:%M:%S"),
                    "Mode":     "🤖 Auto-Live",
                })
                st.session_state._brain.on_trade_placed(_to_brain_input(_a))
                _live_added += 1
            except Exception as _exc:
                # A Polymarket failure here is almost always systemic for the
                # whole cycle (US geo-block → 403, or an invalid wallet key) —
                # every other signal will fail identically. Record one clear
                # message and STOP retrying so the log isn't spammed and the
                # Kalshi block below still runs. Kalshi is the US-accessible path.
                _msg = str(_exc)
                if "403" in _msg or "restricted" in _msg.lower():
                    _hint = ("Polymarket blocks US accounts/IPs (403). This is a "
                             "jurisdiction rule, not a bug — trade Kalshi instead.")
                elif "hex" in _msg.lower():
                    _hint = ("POLYMARKET_PRIVATE_KEY isn't valid hex. Polymarket is "
                             "US-blocked anyway — remove the key from secrets to "
                             "silence this and trade Kalshi only.")
                else:
                    _hint = "Polymarket live unavailable — trading Kalshi only."
                st.session_state._last_live_error = (
                    f"{time.strftime('%H:%M:%S')} — Polymarket disabled this cycle: {_hint} "
                    f"(raw: {_msg[:60]})"
                )
                st.toast(f"Polymarket skipped: {_hint}", icon="⚠️")
                break  # stop hammering Polymarket; fall through to Kalshi
        if _live_added:
            st.toast(
                f"🚨 Bot auto-placed {_live_added} LIVE order{'s' if _live_added != 1 else ''} on Polymarket",
                icon="⚡",
            )

    # ── auto LIVE trades (Kalshi — US-accessible, gated behind live_trading) ───
    # Kalshi is US-regulated so live orders work from a US account/IP, unlike
    # Polymarket which 403s US users. Same gates: auto_live + live_trading + not demo.
    if _al and _lt and _kalshi_key and _kalshi_pem and not _dm:
        _kal_added = 0
        _kal_candidates = 0   # Kalshi BUY signals seen this cycle
        _kal_toosmall = 0     # skipped because Kelly-sized order < $1
        for _a in analyses:
            if _a["signal"] == "HOLD" or _a["kelly"] <= 0 or _a["platform"] != "Kalshi":
                continue
            _kal_candidates += 1
            _size = round(min(float(kalshi_budget) * _a["kelly"], float(kalshi_budget) * 0.15), 2)
            if _size < 1.0:
                _kal_toosmall += 1
                continue
            _dir = "YES" if _a["signal"] == "BUY_YES" else "NO"
            _kside = "yes" if _a["signal"] == "BUY_YES" else "no"
            if any(l.get("Question", "")[:40] == _a["question"][:40] and l.get("Dir") == _dir
                   for l in st.session_state.live_ledger):
                continue
            try:
                _res = _execute_live_kalshi_order(
                    ticker=_a["id"], price=_a["price"], size_usd=_size,
                    side=_kside, api_key_id=_kalshi_key, pem_content=_kalshi_pem,
                )
                st.session_state.live_ledger.append({
                    "Platform": "Kalshi",
                    "Question": _a["question"][:50],
                    "Dir":      _dir,
                    "Size $":   round(_size, 2),
                    "entry_num": _a["price"],
                    "Entry":    f"{_a['price']:.0%}",
                    "AI Est.":  f"{_a['prob']:.0%}",
                    "Edge":     f"{_a['edge']:+.1%}",
                    "Order ID": _res["order_id"],
                    "Status":   _res["status"],
                    "Time":     time.strftime("%H:%M:%S"),
                    "Mode":     "🤖 Auto-Live",
                })
                st.session_state._brain.on_trade_placed(_to_brain_input(_a))
                _kal_added += 1
            except Exception as _exc:
                st.session_state._last_live_error = (
                    f"{time.strftime('%H:%M:%S')} — {_a['question'][:40]}: {_exc}"
                )
                st.toast(f"Auto-live Kalshi order failed: {_exc}", icon="❌")
        if _kal_added:
            st.toast(
                f"🚨 Bot auto-placed {_kal_added} LIVE order{'s' if _kal_added != 1 else ''} on Kalshi",
                icon="⚡",
            )
        else:
            # No Kalshi order fired — record why so it's visible in diagnostics.
            if _kal_candidates == 0:
                _kal_why = ("no actionable Kalshi signals this cycle (all HOLD or no "
                            "edge). Lower Min confidence or wait for Kalshi markets "
                            "with an edge.")
            elif _kal_toosmall == _kal_candidates:
                _kal_why = (f"{_kal_candidates} Kalshi signal(s) found but each Kelly-sized "
                            f"order is < $1 — raise the Kalshi budget (now ${kalshi_budget:.0f}).")
            else:
                _kal_why = (f"{_kal_candidates} Kalshi signal(s) found but already held "
                            "(deduped against the live ledger).")
            st.session_state._last_kalshi_status = (
                f"{time.strftime('%H:%M:%S')} — Kalshi placed 0 orders: {_kal_why}"
            )

    # ── brain: detect resolutions + decay + save ──────────────────────────────
    _brain = st.session_state._brain
    _brain_dirty = False
    for _ma in all_analyses:
        _mid = _ma["id"]
        if _mid in _brain._memories and _brain._memories[_mid].resolved_outcome is None:
            if _ma["price"] >= 0.99:
                _brain.on_market_resolved(_mid, 1.0)
                _brain_dirty = True
            elif _ma["price"] <= 0.01:
                _brain.on_market_resolved(_mid, 0.0)
                _brain_dirty = True
    _brain.apply_decay()
    if _brain_dirty or _is_timer:
        _brain.save()

    # ── markets filter guard ───────────────────────────────────────────────────
    if not analyses:
        st.warning(
            "No markets pass the current **Min confidence** filter. "
            "Lower the slider in the sidebar or click **▶ Run**."
        )
        return

    actionable = [a for a in analyses if a["signal"] != "HOLD"]
    avg_edge   = float(np.mean([abs(a["edge"]) for a in actionable])) if actionable else 0.0
    best       = max(analyses, key=lambda a: abs(a["edge"]))

    # ── metrics row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Markets Scanned",    len(analyses))
    c2.metric("Actionable Signals", len(actionable))
    c3.metric("Avg Edge",           f"{avg_edge:.1%}")
    c4.metric("Best Edge",          f"{abs(best['edge']):.1%}", best["platform"])
    _any_live = _live_ai_mode or _poly_live or _kalshi_live
    _mode_str = "🟢 LIVE" if _any_live else "🔵 MOCK"
    c5.metric("Mode", _mode_str)

    # ── live trading warning (inside fragment so it reflects current toggle) ──
    live_trading = _lt

    # ── session trades strip / order book ─────────────────────────────────────
    _n_paper = len(st.session_state.paper_ledger)
    _n_live  = len(st.session_state.live_ledger)

    if _n_paper + _n_live > 0:
        _strip_pnl_p = sum(_trade_pnl(r) for r in st.session_state.paper_ledger)
        _strip_pnl_l = sum(_trade_pnl(r) for r in st.session_state.live_ledger)
        _strip_total_pnl = _strip_pnl_p + _strip_pnl_l
        _badge = f"📋 Order Book & P&L — {_n_paper} paper"
        if _n_live:
            _badge += f" · {_n_live} live 🚨"
        _badge += f" · Total P&L ${_strip_total_pnl:+.2f}"
        with st.expander(_badge, expanded=_n_live > 0):
            _render_order_book("strip")
    else:
        st.info("No trades recorded yet. Enable **📝 Auto paper** above to start, or use **🚀 Execute** to place trades manually.")

    st.divider()

    # ── tabs ──────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📡 Market Scanner", "🔬 Deep Analysis",
        "🧠 Learning Brain", "⚡ Arbitrage", "💼 Portfolio", "🚀 Execute",
    ])


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Market Scanner
    # ══════════════════════════════════════════════════════════════════════════
    with t1:
        st.subheader("Live Market Scan")
        st.caption("Sorted by absolute edge. The **30-Day Trend** sparkline is the live "
                   "price history — one fast render instead of a chart per market.")

        rows = []
        for a in sorted(analyses, key=lambda x: -abs(x["edge"])):
            rows.append({
                "Platform":  a["platform"],
                "Question":  a["question"][:65],
                "30-Day Trend": a["history"],          # inline sparkline
                "Market":    a["price"] * 100,         # NumberColumn formats raw value
                "AI Est.":   a["prob"] * 100,
                "Edge":      a["edge"] * 100,
                "Conf":      a["conf"] * 100,
                "Kelly":     a["kelly"] * 100,
                "Quality":   round(a["quality"], 2),
                "Signal":    SIGNAL_ICONS[a["signal"]],
                "Days":      a["days"],
            })

        st.dataframe(
            pd.DataFrame(rows), use_container_width=True, hide_index=True,
            column_config={
                "30-Day Trend": st.column_config.LineChartColumn(
                    "30-Day Trend", width="small",
                ),
                "Market":  st.column_config.NumberColumn("Market", format="%.0f%%",
                                                         help="Current market price"),
                "AI Est.": st.column_config.NumberColumn("AI Est.", format="%.0f%%"),
                "Edge":    st.column_config.NumberColumn("Edge", format="%+.1f%%"),
                "Conf":    st.column_config.NumberColumn("Conf", format="%.0f%%"),
                "Kelly":   st.column_config.NumberColumn("Kelly", format="%.1f%%"),
                "Quality": st.column_config.ProgressColumn(
                    "Quality", min_value=0.0, max_value=1.0, format="%.2f",
                ),
            },
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Deep Analysis
    # ══════════════════════════════════════════════════════════════════════════
    with t2:
        st.subheader("Deep Market Analysis")

        options = {f"{a['platform']} | {a['question'][:70]}": i for i, a in enumerate(analyses)}
        chosen  = analyses[options[st.selectbox("Select a market", list(options.keys()))]]

        col_l, col_r = st.columns([3, 2])

        with col_l:
            # Probability card
            clr = SIG_COLOR[chosen["signal"]]
            st.markdown(f"""
<div class="card" style="text-align:center;">
  <div style="font-size:.8rem;color:#aaa;letter-spacing:.08em;">AI ESTIMATED PROBABILITY (YES)</div>
  <div class="prob-big" style="color:{clr};">{chosen['prob']:.1%}</div>
  <div style="color:#ccc;margin-top:4px;">
    Market: <b>{chosen['price']:.1%}</b> &nbsp;·&nbsp;
    Edge: <b style="color:{clr};">{chosen['edge']:+.1%}</b> &nbsp;·&nbsp;
    Bayesian: <b>{chosen['bayesian']:.1%}</b>
  </div>
</div>
""", unsafe_allow_html=True)

            # Probability comparison — native bar chart
            prob_df = pd.DataFrame({
                "Probability": [chosen["price"], chosen["prob"], chosen["bayesian"]],
            }, index=["Market Price", "AI Estimate", "Bayesian Est."])
            st.markdown("**Probability comparison**")
            st.bar_chart(prob_df, height=180, use_container_width=True)

            # Price history — native line chart
            hist_df = pd.DataFrame(
                {"Market Price": chosen["history"],
                 "AI Estimate":  [chosen["prob"]] * len(chosen["history"])},
            )
            st.markdown("**30-Day price history**")
            st.line_chart(hist_df, height=180, use_container_width=True)

        with col_r:
            st.markdown(f"""
<div class="card">
<b>Signal:</b> <span class="{SIG_CSS[chosen['signal']]}">{SIGNAL_ICONS[chosen['signal']]}</span><br>
<b>Edge:</b> {chosen['edge']:+.2%} &nbsp;(min ±{chosen['min_edge']:.2%})<br>
<b>Kelly:</b> {chosen['kelly']:.2%} → <b>${budget * chosen['kelly']:.2f}</b><br>
<b>Confidence:</b> {chosen['conf']:.0%}<br>
<b>Quality score:</b> {chosen['quality']:.2f}<br>
<b>Days to resolve:</b> {chosen['days']}<br>
<b>Bayesian CI:</b> [{chosen['bay_lo']:.1%} – {chosen['bay_hi']:.1%}]
</div>
""", unsafe_allow_html=True)

            st.markdown("**🤖 Claude AI Reasoning**")
            st.info(chosen["ai_reasoning"])

            st.markdown("**✅ Key Factors**")
            for f in chosen["ai_factors"]:
                st.markdown(f"- {f}")

            st.markdown("**⚠️ Uncertainty Flags**")
            for f in chosen["ai_flags"]:
                st.markdown(f"- {f}")

        # Signal breakdown table + bar chart
        st.markdown("#### Signal Breakdown")
        sigs = chosen["signals"]
        if sigs:
            sig_df = pd.DataFrame([{
                "Signal":     s["name"].replace("_", " ").title(),
                "Value":      round(s["value"], 3),
                "Confidence": round(s["confidence"], 2),
                "Weight":     round(s["weight"], 2),
                "Direction":  "🟢 Bullish" if s["value"] > 0 else "🔴 Bearish",
            } for s in sigs])
            st.dataframe(sig_df, use_container_width=True, hide_index=True)

            chart_df = pd.DataFrame(
                {"Signal Value": [s["value"] for s in sigs]},
                index=[s["name"].replace("_", " ").title() for s in sigs],
            )
            st.bar_chart(chart_df, height=200, use_container_width=True)
        else:
            st.info("No directional signals — all signals are neutral for this market.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Learning Brain
    # ══════════════════════════════════════════════════════════════════════════
    with t3:
        st.subheader("🧠 Self-Learning Signal Weight Brain")
        st.markdown(
            "Weights update automatically as real markets resolve using Brier-score "
            "credit assignment. After ~50 resolved markets the weights will meaningfully "
            "diverge from defaults. State saved to `data/learning_state.json`."
        )

        try:
            summ = st.session_state._brain.performance_summary()
        except Exception:
            summ = {
                "resolved_markets": 0, "win_rate": 0.0,
                "avg_brier_improvement": 0.0, "weight_drift": {},
                "current_weights": dict(EnsemblePredictor.DEFAULT_WEIGHTS),
                "pending_markets": 0,
            }

        _resolved = summ["resolved_markets"]
        _pending  = summ.get("pending_markets", 0)
        if _resolved == 0:
            _lvl, _lvl_desc = "🌱 Fresh", "At default weights — no resolved markets yet"
            _next_milestone, _progress = 1, 0.0
        elif _resolved < 10:
            _lvl, _lvl_desc = "📚 Learning", "Weights beginning to drift from early outcomes"
            _next_milestone, _progress = 10, _resolved / 10
        elif _resolved < 50:
            _lvl, _lvl_desc = "🧠 Trained", "Reliable signal discrimination from resolved history"
            _next_milestone, _progress = 50, _resolved / 50
        else:
            _lvl, _lvl_desc = "⚡ Expert", "Highly calibrated — deep resolved market history"
            _next_milestone, _progress = _resolved, 1.0
        st.markdown(f"### Brain State: {_lvl}")
        st.caption(_lvl_desc)
        if _resolved < 50:
            st.progress(_progress, text=f"{_resolved} / {_next_milestone} resolved markets to next level")
        st.markdown("")

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Resolved Markets", _resolved)
        b2.metric("Win Rate",  f"{summ['win_rate']:.0%}" if _resolved else "N/A")
        b3.metric("Pending",   _pending,
                  help="Markets with open trades not yet resolved — learning queued")
        b4.metric("Avg Brier Δ",
                  f"{summ['avg_brier_improvement']:+.4f}" if _resolved else "N/A",
                  help="Positive = bot's predictions beat the naïve 50/50 baseline")

        if _pending:
            st.info(f"📬 **{_pending} trade(s) pending resolution** — weights will update "
                    "automatically when those markets resolve.")

        st.markdown("#### Signal Weights")
        defaults = EnsemblePredictor.DEFAULT_WEIGHTS
        cur_w    = summ.get("current_weights", defaults)
        drift    = summ.get("weight_drift", {})

        w_rows = [{
            "Signal":   n.replace("_", " ").title(),
            "Default":  round(float(defaults[n]), 3),
            "Current":  round(float(cur_w.get(n, defaults[n])), 3),
            "Drift":    f"{float(drift.get(n, 0.0)):+.3f}",
            "Weight":   float(cur_w.get(n, defaults[n])),
        } for n in defaults]

        st.dataframe(
            pd.DataFrame(w_rows),
            use_container_width=True, hide_index=True,
            column_config={
                "Weight": st.column_config.ProgressColumn(
                    "Current Weight", min_value=0.0, max_value=8.0, format="%.2f",
                ),
            },
        )

        drift_vals = {n.replace("_", " ").title(): float(drift.get(n, 0.0)) for n in defaults}
        st.markdown("#### Weight Drift from Default")
        st.bar_chart(
            pd.DataFrame.from_dict({"Drift": drift_vals}, orient="columns"),
            height=200, use_container_width=True,
        )

        if not _resolved:
            st.info(
                "🌱 **Brain is fresh** — no resolved markets yet, signal weights are at defaults.  \n"
                "**How to start learning:** enable Auto-execute paper trades above. As those markets "
                "resolve on Polymarket/Kalshi, the bot will score each signal and adjust weights "
                "automatically. After ~10 resolved markets you'll see meaningful drift."
            )

        # ── cloud persistence status ──────────────────────────────────────────
        from shared.cloud_store import is_available as _cloud_ok
        if _cloud_ok():
            st.success("☁️ **Supabase connected** — brain state persists across restarts and deploys.",
                       icon="✅")
        else:
            _has_url = bool(_supabase_url)
            _has_key = bool(_supabase_key)
            if _has_url or _has_key:
                st.error(
                    f"☁️ **Supabase credentials partially received** — URL: {'✅' if _has_url else '❌'}, "
                    f"Key: {'✅' if _has_key else '❌'}.  \n"
                    "If both are ✅ but still not connected, try **Manage app → Reboot** in Streamlit Cloud.",
                    icon="⚠️",
                )
            else:
                st.warning(
                    "☁️ **No Supabase** — brain resets to defaults every time Streamlit Cloud restarts.  \n"
                    "Add `SUPABASE_URL` and `SUPABASE_KEY` to your secrets to enable persistent learning.",
                    icon="💾",
                )
            with st.expander("🛠 How to set up Supabase persistence (free, 5 min)"):
                st.markdown("""
**Step 1 — Create a free Supabase project**
1. Go to [supabase.com](https://supabase.com) → New project (free tier is fine)
2. Choose a region close to you, set a DB password

**Step 2 — Create the `bot_state` table**

Open **SQL Editor** in your Supabase dashboard and run each statement separately:

*Statement 1 — create table:*
```sql
CREATE TABLE bot_state (
    key        TEXT PRIMARY KEY,
    value      JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```
*Statement 2 — disable RLS (run after statement 1 succeeds):*
```sql
ALTER TABLE bot_state DISABLE ROW LEVEL SECURITY;
```

**Step 3 — Copy your credentials**

In Supabase → **Project Settings → API**:
- `SUPABASE_URL` = **Project URL** (e.g. `https://xxxx.supabase.co`)
- `SUPABASE_KEY` = **service_role** secret key (not the anon key)

**Step 4 — Add to Streamlit secrets**

In your Streamlit Cloud app → **Settings → Secrets**:
```toml
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "your-service-role-key"
```
Then **Reboot app**. The brain will immediately start persisting after the next resolved market.
""")

        with st.expander("How the learning loop works"):
            st.markdown("""
| Step | Action |
|---|---|
| Trade placed | Snapshot signals (value, confidence, weight) at decision time |
| Market resolves | Compute Brier improvement per signal vs naïve baseline |
| Weight update | `delta = 0.08 × improvement × confidence` |
| Regularisation | Nudge weights 2% back toward defaults each cycle |
| Persist | Saved to `data/learning_state.json` |
""")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — Arbitrage
    # ══════════════════════════════════════════════════════════════════════════
    with t4:
        st.subheader("⚡ Cross-Platform Arbitrage Scanner")
        st.markdown("Jaccard word-set similarity matching between Polymarket and Kalshi markets.")

        poly_raw   = [{"condition_id": a["id"], "question": a["question"],
                       "best_ask": a["price"]} for a in analyses if a["platform"] == "Polymarket"]
        kalshi_raw = [{"ticker": a["id"], "title": a["question"],
                       "yes_ask": a["price"]} for a in analyses if a["platform"] == "Kalshi"]

        try:
            # Only surface pairs whose spread actually clears fees (matches the
            # explanatory text below); min_spread=0 would list non-actionable rows.
            arb = CrossMarketCorrelator().find_arbitrage(poly_raw, kalshi_raw, min_spread=0.01)
        except Exception:
            arb = []

        if arb:
            arb_rows = [{
                "Polymarket":    o.poly_question[:55],
                "Kalshi":        o.kalshi_question[:55],
                "Poly":          f"{o.polymarket_yes_price:.1%}",
                "Kalshi Price":  f"{o.kalshi_yes_price:.1%}",
                "Gross Spread":  f"{o.spread:+.1%}",
                "Net Spread":    f"{o.net_spread:+.1%}",
                "Direction":     o.direction,
            } for o in arb]
            st.dataframe(pd.DataFrame(arb_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No arb opportunities in the current mock dataset.")
            st.markdown("""
**How it works:**
- Compare all Poly × Kalshi market pairs using Jaccard word similarity
- Flag pairs > 40% similar as covering the same event
- Show only pairs where net spread (after fees) exceeds the threshold
""")

        # ── Intra-market arbitrage (single-platform, risk-free) ──────────────
        st.divider()
        st.subheader("🎯 Intra-Market Arbitrage (Polymarket)")
        st.markdown(
            "Risk-free **binary bundle** arb: if `YES_ask + NO_ask < $1.00`, buying "
            "both sides locks in profit no matter how the market resolves. Needs "
            "**live order books** (real separate YES/NO asks), so this scans on demand."
        )

        _live_poly = [m for m in _source_markets
                      if m["platform"] == "Polymarket" and m.get("yes_token_id")]

        col_a, col_b = st.columns([1, 2])
        with col_a:
            fee_bps = st.number_input(
                "Fee/gas buffer per share (¢)", value=0.0, min_value=0.0,
                max_value=5.0, step=0.5,
                help="Subtracted per leg. Polymarket CLOB currently has no taker fee.",
            )
            scan_n = st.slider("Markets to scan", 3, 25, 12,
                               help="Each market = 2 order-book calls.")

        if not _poly_live:
            st.info("Connect a live Polymarket feed (POLYMARKET_API_KEY) to scan real "
                    "order books. Mock mids always sum to ~$1.00, so no arb appears.")
        elif st.button("🔍 Scan live order books for arb", use_container_width=True):
            with st.spinner(f"Fetching YES/NO books for {min(scan_n, len(_live_poly))} markets…"):
                try:
                    enriched = _fetch_polymarket_arb_sync(_poly_key, _live_poly[:scan_n])
                    detector = IntraMarketArbitrage(
                        fee_per_share=fee_bps / 100.0, min_net_profit=0.0
                    )
                    arbs = detector.find_binary_bundle(enriched)
                except Exception as e:
                    arbs = []
                    st.error(f"Order-book scan failed: {type(e).__name__}: {e}")
                    enriched = []

            if arbs:
                st.success(f"Found {len(arbs)} binary-bundle arbitrage opportunit"
                           f"{'y' if len(arbs) == 1 else 'ies'}!")
                ia_rows = [{
                    "Question":   o.question[:60],
                    "YES ask":    f"{o.legs[0].ask:.1%}",
                    "NO ask":     f"{o.legs[1].ask:.1%}",
                    "Total cost": f"${o.total_cost:.4f}",
                    "Net profit": f"${o.net_profit:+.4f}",
                    "ROI":        f"{o.roi:+.2%}",
                    "Conf.":      f"{o.confidence:.0%}",
                } for o in arbs]
                st.dataframe(pd.DataFrame(ia_rows), use_container_width=True, hide_index=True)
            elif enriched:
                st.info(f"Scanned {len(enriched)} live markets — no bundle priced under "
                        "$1.00 right now. These windows are brief; re-scan during volatility.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — Portfolio
    # ══════════════════════════════════════════════════════════════════════════
    with t5:
        st.subheader("💼 Monte Carlo Portfolio Simulator")
        st.info("**MOCK MODE** — outcomes are simulated, no real orders placed.")
        st.markdown(
            "Each prediction-market position is a binary bet that wins (resolves "
            "your way) or loses its full stake. Rather than one coin-flip per "
            "position, this runs the **whole basket thousands of times** to show "
            "the *distribution* of outcomes — including the probability of profit "
            "and the **risk of ruin**."
        )

        # ── build positions from actionable signals ──────────────────────────
        # Stake floor is a small fraction of the budget (not a hard $1) so the
        # simulator still works with small per-platform budgets ($10–$20). This
        # is a simulation, not live execution, so we include any positive stake.
        _mc_min_stake = max(0.05, budget * 0.01)   # e.g. $0.20 at a $20 budget
        mc_positions = []
        pos_rows = []
        _n_actionable = 0
        for a in analyses:
            if a["signal"] == "HOLD":
                continue
            _n_actionable += 1
            size = min(budget * a["kelly"], budget * 0.15)
            if size < _mc_min_stake:
                continue
            mcp = position_from_signal(
                label=a["question"][:30],
                signal=a["signal"],
                market_price=a["price"],
                true_prob=a["prob"],
                stake=size,
            )
            if mcp is None:
                continue
            mc_positions.append(mcp)
            ev = mcp.win_prob * mcp.stake * mcp.win_payoff_mult - (1 - mcp.win_prob) * mcp.stake
            pos_rows.append({
                "Platform":  a["platform"],
                "Question":  a["question"][:50],
                "Direction": a["signal"].replace("BUY_", ""),
                "Stake $":   round(size, 2),
                "Entry":     f"{a['price']:.0%}",
                "AI Est.":   f"{a['prob']:.0%}",
                "Win Prob":  f"{mcp.win_prob:.0%}",
                "Edge":      f"{a['edge']:+.1%}",
                "EV $":      round(ev, 2),
            })

        if not mc_positions:
            if _n_actionable == 0:
                st.info(
                    "No actionable BUY signals at the current **Min confidence** "
                    f"({min_conf:.0%}). Lower the Min confidence slider above, or "
                    "click **▶ Run** to re-scan. In Demo mode the simulated edges "
                    "may all fall below your threshold."
                )
            else:
                st.info(
                    f"{_n_actionable} actionable signal(s) found, but each suggested "
                    f"stake is below the simulator's floor of ${_mc_min_stake:.2f} "
                    "(Kelly sizing × your budget). Increase the Polymarket/Kalshi "
                    "budgets above to size positions large enough to simulate."
                )
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                n_sims = st.select_slider(
                    "Simulations", options=[1000, 5000, 10000, 25000, 50000],
                    value=10000,
                )
            with c2:
                shock = st.slider(
                    "Market correlation", 0.0, 0.6, 0.0, 0.05,
                    help="0 = positions independent. Higher = outcomes move "
                         "together (macro shock), widening the tails.",
                )

            mc = MonteCarloPortfolio(n_sims=n_sims, seed=42, market_shock=shock)
            res = mc.simulate(mc_positions, bankroll=budget, ruin_fraction=0.5)

            # headline metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Positions", res.n_positions)
            m2.metric("Total Staked", f"${res.total_staked:.0f}")
            m3.metric("Expected P&L", f"${res.mean_pnl:+.2f}",
                      f"{res.expected_roi:+.1%} ROI")
            m4.metric("Prob. of Profit", f"{res.prob_profit:.0%}")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Median P&L", f"${res.median_pnl:+.2f}")
            m6.metric("P5 (downside)", f"${res.p5:+.2f}")
            m7.metric("P95 (upside)", f"${res.p95:+.2f}")
            m8.metric("Risk of Ruin", f"{res.risk_of_ruin:.1%}",
                      help="Chance of losing ≥ 50% of the budget across the basket.")

            if res.risk_of_ruin >= 0.10:
                st.warning(
                    f"⚠️ **{res.risk_of_ruin:.0%} risk of ruin** — Kelly sizing may be "
                    "too aggressive for this basket. Consider lowering the budget or "
                    "raising the confidence threshold."
                )

            # fan chart: cumulative P&L percentile bands as bets are added
            st.markdown("**Cumulative P&L fan chart** (P5 / Median / P95 across paths)")
            fan_df = pd.DataFrame(
                {"P5": res.band_p5, "Median": res.band_p50, "P95": res.band_p95},
                index=res.step_labels,
            )
            st.line_chart(fan_df, height=260, use_container_width=True)

            # final P&L distribution
            st.markdown("**Final P&L distribution** (all simulated outcomes)")
            hist_df = pd.DataFrame(
                {"Frequency": res.hist_counts},
                index=[f"${c:+.0f}" for c in res.hist_centers],
            )
            st.bar_chart(hist_df, height=220, use_container_width=True)

            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
            st.caption(
                "Win probability uses the AI/ensemble estimate; payoff is the binary "
                "market's mechanical $1 resolution. Outcomes assumed independent unless "
                "you raise the market-correlation slider."
            )

        st.divider()
        st.markdown("""
#### Pre-live checklist
- [ ] Run in mock mode for **≥ 2 weeks**, review each decision manually
- [ ] `python scripts/quickstart.py` — verify all API connections
- [ ] Risk limits reviewed for your bankroll size
- [ ] Start with **< $5 per trade** when switching to live
- [ ] Only then: set `MOCK_MODE=false` in `.env`
""")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — Execute
    # ══════════════════════════════════════════════════════════════════════════
    with t6:
        st.subheader("🚀 Execute Trades")

        if live_trading:
            st.error("🚨 **LIVE TRADING ENABLED** — buttons below place REAL orders with "
                     "REAL money on Polygon. Each requires an explicit confirmation.")
            st.caption("Note: live execution requires `py-clob-client` to be installed in "
                       "this environment. If you see a 'SigningUnavailable' error, use the "
                       "local bot (`MOCK_MODE=false python main.py`) instead — it includes "
                       "the full signing toolchain via requirements.txt.")
        else:
            st.info("🔵 **Paper mode** — buttons below simulate fills only. "
                    "No real orders are placed.")

        # ── auto-execute diagnostics — why are (or aren't) trades firing? ──────
        st.markdown("### 🤖 Auto-Execute Status")
        _gate_demo  = not _dm
        _gate_live  = bool(_lt)
        _gate_key   = bool(_poly_pk or _kalshi_live_creds)
        _gate_auto  = bool(_al)
        _gate_loop  = bool(_refresh_secs)   # auto-refresh drives repeated cycles
        _all_gates  = _gate_demo and _gate_live and _gate_key and _gate_auto

        _g1, _g2, _g3, _g4, _g5 = st.columns(5)
        _g1.metric("Demo OFF",        "✅" if _gate_demo else "❌ ON")
        _g2.metric("Live trading",    "✅ ON" if _gate_live else "❌ OFF")
        _g3.metric("Live creds",      "✅" if _gate_key  else "❌ missing")
        _g4.metric("Auto-execute",    "✅ ON" if _gate_auto else "❌ OFF")
        _g5.metric("Auto-refresh",    "✅" if _gate_loop else "⚠️ Off")

        # Which platforms can actually execute live right now?
        _live_plats = []
        if _poly_pk:            _live_plats.append("Polymarket")
        if _kalshi_live_creds:  _live_plats.append("Kalshi")
        st.caption(
            "Live execution enabled for: **" + (", ".join(_live_plats) or "none") + "**. "
            "Polymarket 403s US accounts; Kalshi is US-accessible."
        )

        if not _all_gates:
            _need = []
            if not _gate_demo: _need.append("turn **Demo mode OFF**")
            if not _gate_live: _need.append("turn **Live trading ON**")
            if not _gate_key:  _need.append("add **POLYMARKET_PRIVATE_KEY** or "
                                            "**KALSHI_API_KEY + KALSHI_PRIVATE_KEY_PEM** to secrets")
            if not _gate_auto: _need.append("turn **🤖 Auto-execute LIVE ON**")
            st.warning("The AI brain will NOT auto-trade until you: " + "; ".join(_need) + ".")
        elif not _gate_loop:
            st.warning("All gates are ON, but **Auto-refresh is Off** — the bot only "
                       "evaluates once per page load / ▶ Run. Set **⏱ Auto-refresh** "
                       "to an interval (e.g. *Every 5 min*) for continuous trading.")
        else:
            st.success("All systems GO — the brain auto-places live orders on "
                       + (" + ".join(_live_plats) or "(no platform)") + " each refresh cycle.")

        _last_err = st.session_state.get("_last_live_error")
        if _last_err:
            st.error(f"⚠️ Last live-order error: {_last_err}")

        _kal_status = st.session_state.get("_last_kalshi_status")
        if _kal_status:
            st.info(f"📊 Kalshi: {_kal_status}")

        # Per-candidate breakdown: show why each live signal did/didn't trade.
        _live_sigs = [a for a in analyses
                      if a["signal"] != "HOLD"
                      and ((a["platform"] == "Polymarket" and _poly_pk)
                           or (a["platform"] == "Kalshi" and _kalshi_live_creds))]
        if _live_sigs:
            _diag_rows = []
            for _a in _live_sigs:
                _dir = "YES" if _a["signal"] == "BUY_YES" else "NO"
                _is_poly = _a["platform"] == "Polymarket"
                _pb  = float(poly_budget) if _is_poly else float(kalshi_budget)
                _tok = (_a.get("yes_token_id") if _a["signal"] == "BUY_YES"
                        else _a.get("no_token_id")) if _is_poly else "n/a"
                _sz  = round(min(_pb * _a["kelly"], _pb * 0.15), 2)
                _dup = any(l.get("Question", "")[:40] == _a["question"][:40] and l.get("Dir") == _dir
                           for l in st.session_state.live_ledger)
                if _a["kelly"] <= 0:                _why = "❌ Kelly = 0 (no edge)"
                elif _is_poly and not _tok:         _why = "❌ no token id (mock/illiquid)"
                elif _sz < 1.0:                     _why = f"❌ size ${_sz:.2f} < $1 min (raise budget)"
                elif _dup:                          _why = "⏸ already held"
                elif not _all_gates:                _why = "⏸ gates off (see above)"
                else:                               _why = "✅ eligible — will trade next cycle"
                _diag_rows.append({
                    "Platform": _a["platform"],
                    "Question": _a["question"][:40],
                    "Dir":      _dir,
                    "Kelly":    f"{_a['kelly']:.1%}",
                    "Size $":   f"${_sz:.2f}",
                    "Edge":     f"{_a['edge']:+.1%}",
                    "Status":   _why,
                })
            with st.expander(f"🔍 Why these {len(_live_sigs)} live signal(s) did/didn't trade", expanded=not _all_gates):
                st.dataframe(pd.DataFrame(_diag_rows), use_container_width=True, hide_index=True)
                st.caption("A trade needs Kelly × budget ≥ $1 to clear the minimum order. "
                           "Raise the per-platform **budget** above to size more signals over the minimum.")
        else:
            st.caption("No actionable live signals at the current Min confidence. "
                       "Lower **Min confidence** or click **▶ Run** to re-scan.")

        # ── order book with live P&L ──────────────────────────────────────────
        st.markdown("### 📊 Order Book & P&L")
        _render_order_book("exec")
        st.divider()

        st.markdown("### 🛒 Place New Trades")
        exec_signals = [a for a in actionable if a["kelly"] > 0]
        if not exec_signals:
            st.info("No actionable BUY signals at the current confidence threshold.")
        else:
            st.caption(f"{len(exec_signals)} actionable signal(s). "
                       "Size = Kelly fraction × platform budget (capped at 15%).")

            for a in sorted(exec_signals, key=lambda x: -abs(x["edge"])):
                _pb = poly_budget if a["platform"] == "Polymarket" else kalshi_budget
                size = float(min(_pb * a["kelly"], _pb * 0.15))
                direction = "YES" if a["signal"] == "BUY_YES" else "NO"
                is_poly = a["platform"] == "Polymarket"
                token_id = a["yes_token_id"] if direction == "YES" else a["no_token_id"]
                can_live_poly   = bool(live_trading and is_poly and token_id and _poly_pk)
                can_live_kalshi = bool(live_trading and not is_poly and _kalshi_live_creds)
                can_live = can_live_poly or can_live_kalshi

                with st.container(border=True):
                    cL, cR = st.columns([3, 2])
                    with cL:
                        st.markdown(
                            f"**{a['platform']}** · {a['question'][:70]}  \n"
                            f"{SIGNAL_ICONS[a['signal']]} · entry **{a['price']:.0%}** · "
                            f"AI **{a['prob']:.0%}** · edge **{a['edge']:+.1%}** · "
                            f"conf **{a['conf']:.0%}**"
                        )
                        st.caption(f"Suggested size: **${size:.2f}** "
                                   f"(≈{(size / a['price']) if a['price'] else 0:.1f} shares @ {a['price']:.0%})")
                    with cR:
                        # Paper trade — always available
                        if st.button("📝 Paper Buy", key=f"paper_{a['id']}",
                                     use_container_width=True):
                            st.session_state.paper_ledger.append({
                                "Platform":  a["platform"],
                                "Question":  a["question"][:50],
                                "Dir":       direction,
                                "Size $":    round(size, 2),
                                "entry_num": a["price"],
                                "Entry":     f"{a['price']:.0%}",
                                "AI Est.":   f"{a['prob']:.0%}",
                                "Edge":      f"{a['edge']:+.1%}",
                                "Conf":      f"{a['conf']:.0%}",
                                "Time":      time.strftime("%H:%M:%S"),
                                "Mode":      "👆 Manual",
                            })
                            st.session_state._brain.on_trade_placed(_to_brain_input(a))
                            st.toast(f"Paper buy recorded: {a['question'][:30]}")

                        # Live execution — heavily gated
                        if live_trading:
                            if is_poly and not _poly_pk:
                                st.caption("⚠️ No POLYMARKET_PRIVATE_KEY in secrets — paper only.")
                            elif is_poly and not token_id:
                                st.caption("⚠️ No token id (mock market) — paper only.")
                            elif not is_poly and not _kalshi_live_creds:
                                st.caption("⚠️ No Kalshi API key + PEM in secrets — paper only.")
                            elif can_live:
                                confirm = st.checkbox(
                                    f"Confirm REAL ${size:.2f} order",
                                    key=f"confirm_{a['id']}",
                                )
                                if st.button("🚨 EXECUTE LIVE", key=f"live_{a['id']}",
                                             type="primary", use_container_width=True,
                                             disabled=not confirm):
                                    try:
                                        with st.spinner("Signing & posting order…"):
                                            if is_poly:
                                                res = _execute_live_polymarket_order(
                                                    token_id=token_id, price=a["price"],
                                                    size_usd=size, side="BUY",
                                                    private_key=_poly_pk,
                                                    sig_type=_poly_sig_type,
                                                    funder=_poly_funder,
                                                )
                                            else:
                                                res = _execute_live_kalshi_order(
                                                    ticker=a["id"], price=a["price"],
                                                    size_usd=size,
                                                    side="yes" if direction == "YES" else "no",
                                                    api_key_id=_kalshi_key,
                                                    pem_content=_kalshi_pem,
                                                )
                                        st.session_state.live_ledger.append({
                                            "Platform":  a["platform"],
                                            "Question":  a["question"][:50],
                                            "Dir":       direction,
                                            "Size $":    round(size, 2),
                                            "entry_num": a["price"],
                                            "Entry":     f"{a['price']:.0%}",
                                            "AI Est.":   f"{a['prob']:.0%}",
                                            "Edge":      f"{a['edge']:+.1%}",
                                            "Order ID":  res["order_id"],
                                            "Status":    res["status"],
                                            "Time":      time.strftime("%H:%M:%S"),
                                            "Mode":      "👆 Manual-Live",
                                        })
                                        st.session_state._brain.on_trade_placed(_to_brain_input(a))
                                        st.success(f"Order posted: {res['order_id']} "
                                                   f"({res['status']})")
                                    except SigningUnavailable as exc:
                                        st.error(f"Live signing unavailable: {exc}")
                                    except Exception as exc:
                                        st.error(f"Order failed: {exc}")

        # ── ledger link ──────────────────────────────────────────────────────
        _tot = len(st.session_state.paper_ledger) + len(st.session_state.live_ledger)
        if _tot:
            st.info(f"📋 **{_tot} trade(s) recorded** this session — see the "
                    "**Today's Session** strip above the tabs for the full ledger.")

    # ── footer (inside fragment so mode indicators stay current) ─────────────
    st.divider()
    _parts = []
    if _poly_live:    _parts.append("Live Polymarket")
    if _kalshi_live:  _parts.append("Live Kalshi")
    if _live_ai_mode: _parts.append("Live Claude AI")
    if not _parts:    _parts.append("Mock mode")
    _parts.append("🚨 LIVE TRADING" if live_trading else "Paper trades only")
    st.caption(f"🤖 Polymarket AI Bot | {' · '.join(_parts)} | {time.strftime('%H:%M UTC', time.gmtime())}")


# ── call the fragment ──────────────────────────────────────────────────────────
_live_dashboard()

# ── Save current settings to URL so they survive a browser refresh ─────────────
st.query_params.update({
    "demo": "1" if st.session_state.get("demo_mode", True)   else "0",
    "pb":   str(st.session_state.get("poly_budget",    10.0)),
    "kb":   str(st.session_state.get("kalshi_budget",  10.0)),
    "mc":   str(st.session_state.get("sidebar_min_conf", 0.55)),
    "rl":   st.session_state.get("refresh_label", "Off"),
    "ap":   "1" if st.session_state.get("auto_paper", False) else "0",
    "lt":   "1" if st.session_state.get("live_trading", False) else "0",
    "al":   "1" if st.session_state.get("auto_live", False) else "0",
})
