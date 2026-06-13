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
from pathlib import Path

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


def _run_coro_sync(coro):
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
    t.join(timeout=90)
    if exc_box[0]:
        raise exc_box[0]
    return result_box[0]


def _live_ai_analysis(api_key: str, markets: list[dict]) -> dict[str, dict]:
    """Batch-call Claude AI; returns {market_id: plain_dict}."""
    agent = ClaudeAgent(api_key=api_key, model="claude-opus-4-8")
    batch_input = [
        {"id": m["id"], "question": m["question"], "price": m["price"]}
        for m in markets
    ]
    analyses = _run_coro_sync(agent.analyze_batch(batch_input))
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
def _run_analysis(anthropic_key: str = "") -> tuple[list[dict], bool]:
    # Fetch AI probability estimates — live Claude when key is present, else mock
    live_ai: dict[str, dict] = {}
    using_live = False
    if anthropic_key:
        try:
            live_ai = _live_ai_analysis(anthropic_key, MOCK_MARKETS)
            using_live = True
        except Exception as _live_err:
            live_ai = {}
            using_live = False

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
    for mkt in MOCK_MARKETS:
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
        })
    return out, using_live


# ── sidebar ───────────────────────────────────────────────────────────────────
_anthropic_key = _get_secret("ANTHROPIC_API_KEY")

with st.sidebar:
    st.title("🤖 Polymarket AI Bot")
    if _anthropic_key:
        st.markdown("🟢 **LIVE CLAUDE AI** · 🔵 Mock trades")
    else:
        st.markdown("🔵 **MOCK MODE** — no real trades placed")
    st.divider()
    budget   = st.number_input("Paper budget ($)", value=100.0, min_value=10.0, step=10.0)
    min_conf = st.slider("Min confidence", 0.40, 0.90, 0.55, 0.05)
    st.divider()
    if st.button("🔄 Re-run Analysis", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    if not _anthropic_key:
        st.divider()
        st.caption("Add API keys to Streamlit secrets to go live:")
        st.code("ANTHROPIC_API_KEY = 'sk-ant-...'\nPOLYMARKET_API_KEY = '...'", language="toml")

# ── load data ─────────────────────────────────────────────────────────────────
try:
    with st.spinner("Running predictive models…"):
        all_analyses, _live_ai_mode = _run_analysis(_anthropic_key)
except Exception as exc:
    st.error("Analysis pipeline failed — see details below.")
    st.exception(exc)
    st.stop()

if _anthropic_key and not _live_ai_mode:
    st.warning("ANTHROPIC_API_KEY found but Claude AI call failed — showing mock analysis. Check key validity.")

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
c5.metric("Mode",               "🟢 LIVE AI" if _live_ai_mode else "🔵 MOCK")
st.divider()

# ── tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "📡 Market Scanner", "🔬 Deep Analysis",
    "🧠 Learning Brain", "⚡ Arbitrage", "💼 Portfolio",
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Portfolio
# ══════════════════════════════════════════════════════════════════════════════
with t5:
    st.subheader("💼 Paper Portfolio Simulator")
    st.info("**MOCK MODE** — all P&L is simulated, no real orders placed.")

    rng_p = random.Random(42)
    pos   = []
    total = 0.0

    for a in analyses:
        if a["signal"] == "HOLD":
            continue
        size = min(budget * a["kelly"], budget * 0.15)
        if size < 1.0:
            continue
        bias = 0.62 if (
            (a["signal"] == "BUY_YES" and a["prob"] > a["price"]) or
            (a["signal"] == "BUY_NO"  and a["prob"] < a["price"])
        ) else 0.38
        won  = rng_p.random() < bias
        pnl  = size * (1.0 / a["price"] - 1.0) if won else -size
        total += pnl
        pos.append({
            "Platform":  a["platform"],
            "Question":  a["question"][:55],
            "Direction": a["signal"].replace("BUY_", ""),
            "Size $":    round(size, 2),
            "Entry":     f"{a['price']:.0%}",
            "AI Est.":   f"{a['prob']:.0%}",
            "Edge":      f"{a['edge']:+.1%}",
            "P&L $":     round(pnl, 2),
            "Result":    "✅ WIN" if won else "❌ LOSS",
        })

    if pos:
        p1, p2, p3 = st.columns(3)
        p1.metric("Simulated Trades", len(pos))
        p2.metric("Net Mock P&L",     f"${total:+.2f}")
        p3.metric("ROI",              f"{total / budget:.1%}")

        pnl_df = pd.DataFrame(
            {"P&L ($)": [p["P&L $"] for p in pos]},
            index=[p["Question"][:30] for p in pos],
        )
        st.bar_chart(pnl_df, height=220, use_container_width=True)
        st.dataframe(pd.DataFrame(pos), use_container_width=True, hide_index=True)
    else:
        st.info("No positions meet the current thresholds.")

    st.divider()
    st.markdown("""
#### Pre-live checklist
- [ ] Run in mock mode for **≥ 2 weeks**, review each decision manually
- [ ] `python scripts/quickstart.py` — verify all API connections
- [ ] Risk limits reviewed for your bankroll size
- [ ] Start with **< $5 per trade** when switching to live
- [ ] Only then: set `MOCK_MODE=false` in `.env`
""")

# ── footer ────────────────────────────────────────────────────────────────────
st.divider()
_mode_label = "LIVE CLAUDE AI · Mock trades" if _live_ai_mode else "MOCK MODE · No real trades"
st.caption(
    f"🤖 Polymarket AI Bot | {_mode_label} | "
    f"{time.strftime('%H:%M UTC', time.gmtime())}"
)
