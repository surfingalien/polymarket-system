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
    from kalshi_bot.kalshi_client import KalshiClient
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


def _run_coro_sync(coro, timeout: int = 30):
    """Run an async coroutine safely from a synchronous Streamlit context."""
    result_box: list = [None]
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
    if result_box[0] is None:
        raise RuntimeError("API call returned no data")
    return result_box[0]


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
        if p > 0:
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

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.card { background:#1e1e2e; border-radius:10px; padding:14px 18px;
        margin-bottom:10px; border-left:4px solid #6366f1; }
.prob-big { font-size:3.2rem; font-weight:900; line-height:1.1; }
.signal-yes { color:#00d26a; font-weight:700; }
.signal-no  { color:#ff4b4b; font-weight:700; }
.signal-hold{ color:#888888; font-weight:600; }
</style>
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
        kf        = float(kelly.compute(float(adj_prob), mkt["price"]))

        signal = ("HOLD" if abs(edge) < min_edge or adj_conf < 0.50
                  else ("BUY_YES" if edge > 0 else "BUY_NO"))

        sigs = [
            {"name": str(s.name), "value": float(s.value),
             "confidence": float(s.confidence), "weight": float(s.weight or 1.0)}
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

if "paper_ledger" not in st.session_state:
    st.session_state.paper_ledger = []
if "live_ledger" not in st.session_state:
    st.session_state.live_ledger = []

with st.sidebar:
    st.title("🤖 Polymarket AI Bot")
    st.markdown(("🟢 **LIVE CLAUDE AI**" if _anthropic_key else "🟡 Mock AI analysis"))
    st.markdown(("🟢 **LIVE POLYMARKET**" if _poly_key      else "🟡 Mock Polymarket data"))
    st.markdown(("🟢 **LIVE KALSHI**"     if (_kalshi_key and _kalshi_pem)    else "🟡 Mock Kalshi data"))
    st.divider()
    budget   = st.number_input("Paper budget ($)", value=100.0, min_value=10.0, step=10.0)
    min_conf = st.slider("Min confidence", 0.40, 0.90, 0.55, 0.05)
    st.divider()
    if st.button("🔄 Re-run Analysis", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # ── live trading gate (DANGER) ──────────────────────────────────────────
    st.divider()
    st.markdown("### ⚠️ Live Trading")
    live_trading = False
    if not _poly_pk:
        st.caption("🔵 Paper mode only. Add `POLYMARKET_PRIVATE_KEY` to secrets "
                   "to unlock live execution (never paste keys into chat).")
    else:
        st.caption("Wallet key detected. Live execution is OFF until you enable it.")
        live_trading = st.toggle("🚨 Enable LIVE trading (real money)", value=False)
        if live_trading:
            st.error("LIVE MODE — orders placed here spend REAL funds on Polygon. "
                     "Start with < $5 per trade.")
            st.caption(f"Wallet sig type: {_poly_sig_type} "
                       f"({'EOA' if _poly_sig_type == 0 else 'proxy/safe'})")

    _missing = [k for k, v in {
        "ANTHROPIC_API_KEY": _anthropic_key,
        "POLYMARKET_API_KEY": _poly_key,
        "KALSHI_API_KEY": _kalshi_key,
        "KALSHI_PRIVATE_KEY_PEM": _kalshi_pem,
    }.items() if not v]
    if _missing:
        st.divider()
        st.caption("Missing secrets (add via Manage app → Secrets):")
        for _k in _missing:
            st.caption(f"  • {_k}")

# ── load data ─────────────────────────────────────────────────────────────────
try:
    with st.spinner("Fetching markets and running predictive models…"):
        _source_markets, _poly_live, _kalshi_live, _poly_err, _kalshi_err = \
            _get_markets(_poly_key, _kalshi_key, _kalshi_pem)
        all_analyses, _live_ai_mode, _ai_err = _run_analysis(_source_markets, _anthropic_key)
except Exception as exc:
    st.error("Analysis pipeline failed — see details below.")
    st.exception(exc)
    st.stop()

if _poly_key and not _poly_live:
    st.warning(
        f"**Polymarket live fetch failed** — using mock markets.  \n"
        f"**Error:** `{_poly_err}`  \n"
        "Common causes: API endpoint changed, network restriction, or invalid key format."
    )
if _kalshi_key and not _kalshi_live:
    st.warning(
        f"**Kalshi live fetch failed** — using mock markets.  \n"
        f"**Error:** `{_kalshi_err}`"
    )
if _anthropic_key and not _live_ai_mode:
    _ai_err_msg = f"  \n**Error:** `{_ai_err}`" if _ai_err else ""
    st.warning(
        "**ANTHROPIC_API_KEY set but Claude AI call failed** — using mock analysis.  \n"
        f"Check key validity at console.anthropic.com.{_ai_err_msg}"
    )

analyses = [a for a in all_analyses if a["conf"] >= min_conf or a["signal"] == "HOLD"]

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("## 📊 Polymarket + Kalshi AI Trading Dashboard")
actionable = [a for a in analyses if a["signal"] != "HOLD"]
avg_edge   = float(np.mean([abs(a["edge"]) for a in actionable])) if actionable else 0.0
best       = max(analyses, key=lambda a: abs(a["edge"]))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Markets Scanned",    len(analyses))
c2.metric("Actionable Signals", len(actionable))
c3.metric("Avg Edge",           f"{avg_edge:.1%}")
c4.metric("Best Edge",          f"{abs(best['edge']):.1%}", best["platform"])
_any_live = _live_ai_mode or _poly_live or _kalshi_live
_mode_str = "🟢 LIVE" if _any_live else "🔵 MOCK"
c5.metric("Mode", _mode_str)
st.divider()

# ── tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📡 Market Scanner", "🔬 Deep Analysis",
    "🧠 Learning Brain", "⚡ Arbitrage", "💼 Portfolio", "🚀 Execute",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Market Scanner
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    st.subheader("Live Market Scan")

    rows = []
    for a in sorted(analyses, key=lambda x: -abs(x["edge"])):
        rows.append({
            "Platform":  a["platform"],
            "Question":  a["question"][:65],
            "Market":    f"{a['price']:.0%}",
            "AI Est.":   f"{a['prob']:.0%}",
            "Edge":      f"{a['edge']:+.1%}",
            "Conf":      f"{a['conf']:.0%}",
            "Kelly":     f"{a['kelly']:.1%}",
            "Quality":   round(a["quality"], 2),
            "Signal":    SIGNAL_ICONS[a["signal"]],
            "Days":      a["days"],
        })

    st.dataframe(
        pd.DataFrame(rows), use_container_width=True, hide_index=True,
        column_config={
            "Quality": st.column_config.ProgressColumn(
                "Quality", min_value=0.0, max_value=1.0, format="%.2f",
            ),
        },
    )

    st.markdown("#### 30-Day Price Trend")
    cols = st.columns(4)
    for i, a in enumerate(analyses):
        with cols[i % 4]:
            lbl = f'<span class="{SIG_CSS[a["signal"]]}">{SIGNAL_ICONS[a["signal"]]}</span>'
            st.markdown(
                f"**{a['question'][:40]}**  \n"
                f"mkt {a['price']:.0%} → AI {a['prob']:.0%} "
                f"({a['edge']:+.1%}) {lbl}",
                unsafe_allow_html=True,
            )
            # Pure Streamlit line chart — no Plotly
            hist_df = pd.DataFrame({"price": a["history"]})
            st.line_chart(hist_df, height=80, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Deep Analysis
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Learning Brain
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    st.subheader("🧠 Self-Learning Signal Weight Brain")
    st.markdown(
        "Weights update automatically as real markets resolve using Brier-score "
        "credit assignment. After ~50 resolved markets the weights will meaningfully "
        "diverge from defaults. State saved to `data/learning_state.json`."
    )

    try:
        _ens = EnsemblePredictor()
        _eng = LearningEngine(_ens, state_file=Path("data/learning_state.json"))
        _eng.load()
        summ = _eng.performance_summary()
    except Exception:
        summ = {
            "resolved_markets": 0, "win_rate": 0.0,
            "avg_brier_improvement": 0.0, "weight_drift": {},
            "current_weights": dict(EnsemblePredictor.DEFAULT_WEIGHTS),
            "pending_markets": 0,
        }

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Resolved Markets", summ["resolved_markets"])
    b2.metric("Win Rate",  f"{summ['win_rate']:.0%}" if summ["resolved_markets"] else "N/A")
    b3.metric("Pending",   summ.get("pending_markets", 0))
    b4.metric("Avg Brier Δ",
              f"{summ['avg_brier_improvement']:+.4f}" if summ["resolved_markets"] else "N/A")

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

    if not summ["resolved_markets"]:
        st.info("No resolved markets yet — brain is at defaults. Weights evolve as trades resolve.")

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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Arbitrage
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    st.subheader("⚡ Cross-Platform Arbitrage Scanner")
    st.markdown("Jaccard word-set similarity matching between Polymarket and Kalshi markets.")

    poly_raw   = [{"condition_id": a["id"], "question": a["question"],
                   "best_ask": a["price"]} for a in analyses if a["platform"] == "Polymarket"]
    kalshi_raw = [{"ticker": a["id"], "title": a["question"],
                   "yes_ask": a["price"]} for a in analyses if a["platform"] == "Kalshi"]

    try:
        arb = CrossMarketCorrelator().find_arbitrage(poly_raw, kalshi_raw, min_spread=0.0)
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

    # ── Intra-market arbitrage (single-platform, risk-free) ──────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Portfolio
# ══════════════════════════════════════════════════════════════════════════════
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

    # ── build positions from actionable signals ──────────────────────────────
    mc_positions = []
    pos_rows = []
    for a in analyses:
        if a["signal"] == "HOLD":
            continue
        size = min(budget * a["kelly"], budget * 0.15)
        if size < 1.0:
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
        st.info("No positions meet the current thresholds.")
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Execute
# ══════════════════════════════════════════════════════════════════════════════
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

    exec_signals = [a for a in actionable if a["kelly"] > 0]
    if not exec_signals:
        st.info("No actionable BUY signals at the current confidence threshold.")
    else:
        st.caption(f"{len(exec_signals)} actionable signal(s). "
                   "Suggested size = Kelly fraction × budget (capped at 15%).")

        for a in sorted(exec_signals, key=lambda x: -abs(x["edge"])):
            size = float(min(budget * a["kelly"], budget * 0.15))
            direction = "YES" if a["signal"] == "BUY_YES" else "NO"
            is_poly = a["platform"] == "Polymarket"
            token_id = a["yes_token_id"] if direction == "YES" else a["no_token_id"]
            can_live = bool(live_trading and is_poly and token_id and _poly_pk)

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
                            "Platform": a["platform"], "Question": a["question"][:50],
                            "Dir": direction, "Size $": round(size, 2),
                            "Entry": f"{a['price']:.0%}", "Edge": f"{a['edge']:+.1%}",
                            "Time": time.strftime("%H:%M:%S"),
                        })
                        st.toast(f"Paper buy recorded: {a['question'][:30]}")

                    # Live execution — heavily gated
                    if live_trading:
                        if not is_poly:
                            st.caption("⚠️ Live execution from the dashboard supports "
                                       "Polymarket only. Use the bot for Kalshi.")
                        elif not token_id:
                            st.caption("⚠️ No token id (mock market) — paper only.")
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
                                        res = _execute_live_polymarket_order(
                                            token_id=token_id, price=a["price"],
                                            size_usd=size, side="BUY",
                                            private_key=_poly_pk,
                                            sig_type=_poly_sig_type,
                                            funder=_poly_funder,
                                        )
                                    st.session_state.live_ledger.append({
                                        "Platform": a["platform"],
                                        "Question": a["question"][:50],
                                        "Dir": direction, "Size $": round(size, 2),
                                        "Order ID": res["order_id"],
                                        "Status": res["status"],
                                        "Time": time.strftime("%H:%M:%S"),
                                    })
                                    st.success(f"Order posted: {res['order_id']} "
                                               f"({res['status']})")
                                except SigningUnavailable as exc:
                                    st.error(f"Live signing unavailable: {exc}")
                                except Exception as exc:
                                    st.error(f"Order failed: {exc}")

    # ── ledgers ─────────────────────────────────────────────────────────────
    if st.session_state.live_ledger:
        st.markdown("#### 🚨 Live Orders This Session")
        st.dataframe(pd.DataFrame(st.session_state.live_ledger),
                     use_container_width=True, hide_index=True)
    if st.session_state.paper_ledger:
        st.markdown("#### 📝 Paper Orders This Session")
        st.dataframe(pd.DataFrame(st.session_state.paper_ledger),
                     use_container_width=True, hide_index=True)
        if st.button("Clear paper ledger"):
            st.session_state.paper_ledger = []
            st.rerun()


# ── footer ────────────────────────────────────────────────────────────────────
st.divider()
_parts = []
if _poly_live:    _parts.append("Live Polymarket")
if _kalshi_live:  _parts.append("Live Kalshi")
if _live_ai_mode: _parts.append("Live Claude AI")
if not _parts:    _parts.append("Mock mode")
_parts.append("🚨 LIVE TRADING" if live_trading else "Paper trades only")
st.caption(f"🤖 Polymarket AI Bot | {' · '.join(_parts)} | {time.strftime('%H:%M UTC', time.gmtime())}")
