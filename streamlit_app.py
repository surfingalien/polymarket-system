"""
Polymarket / Kalshi AI Trading Bot — Streamlit Demo Dashboard

Runs the full predictive algorithm stack on mock market data.
No API keys required. Set ANTHROPIC_API_KEY to enable live Claude analysis.
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Polymarket AI Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── imports (wrapped so Streamlit shows a friendly error on missing package) ──
try:
    from shared.advanced_signals import (
        CategoryEdgeModel,
        LongshotBiasDetector,
        MarketQualityScorer,
        TemporalPatternSignal,
    )
    from shared.learning_engine import LearningEngine
    from shared.predictive_models import (
        BayesianEstimator,
        CrossMarketCorrelator,
        EnsemblePredictor,
        KellyCriterion,
        MomentumAnalyzer,
        OrderBookAnalyzer,
        OrderBookSnapshot,
        PricePoint,
        ResolutionDecayModel,
        Signal,
    )
except ImportError as _e:
    st.error(f"Import error — check requirements: {_e}")
    st.stop()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .signal-yes  { color:#00d26a; font-weight:700; }
  .signal-no   { color:#ff4b4b; font-weight:700; }
  .signal-hold { color:#aaaaaa; font-weight:600; }
  .metric-card { background:#1e1e2e; border-radius:8px; padding:12px 16px;
                 margin-bottom:8px; border-left:3px solid #6366f1; }
</style>
""", unsafe_allow_html=True)

# ── signal label lookup (top-level so all tabs can access it) ─────────────────
SIGNAL_ICONS = {"BUY_YES": "🟢 BUY YES", "BUY_NO": "🔴 BUY NO", "HOLD": "⚪ HOLD"}

# ── mock market catalogue ─────────────────────────────────────────────────────
MOCK_MARKETS = [
    {"id": "poly_fed_rate_jul", "platform": "Polymarket",
     "question": "Will the Fed cut rates at the July 2026 FOMC?",
     "price": 0.38, "volume": 2_450_000, "liquidity": 180_000,
     "days": 25, "category": "finance", "trend": +0.003},
    {"id": "poly_btc_120k",     "platform": "Polymarket",
     "question": "Will Bitcoin exceed $120k before end of 2026?",
     "price": 0.52, "volume": 5_100_000, "liquidity": 420_000,
     "days": 202, "category": "crypto",  "trend": +0.006},
    {"id": "poly_ai_exec_order","platform": "Polymarket",
     "question": "Will Trump sign a new AI regulation executive order before July?",
     "price": 0.21, "volume": 890_000,   "liquidity": 75_000,
     "days": 18,  "category": "politics","trend": -0.002},
    {"id": "poly_nba_celtics",  "platform": "Polymarket",
     "question": "Will the Boston Celtics win the 2026 NBA Championship?",
     "price": 0.34, "volume": 3_200_000, "liquidity": 260_000,
     "days": 12,  "category": "sports",  "trend": +0.008},
    {"id": "poly_recession_26", "platform": "Polymarket",
     "question": "Will the US enter a recession in 2026?",
     "price": 0.29, "volume": 1_800_000, "liquidity": 145_000,
     "days": 202, "category": "finance", "trend": -0.001},
    {"id": "poly_eth_5k",       "platform": "Polymarket",
     "question": "Will Ethereum exceed $5,000 before September 2026?",
     "price": 0.61, "volume": 2_700_000, "liquidity": 210_000,
     "days": 90,  "category": "crypto",  "trend": +0.004},
    {"id": "kalshi_fed_jul",    "platform": "Kalshi",
     "question": "Federal Reserve rate cut at July 2026 FOMC?",
     "price": 0.41, "volume": 980_000,   "liquidity": 85_000,
     "days": 25,  "category": "finance", "trend": +0.002},
    {"id": "poly_brazil_wc",    "platform": "Polymarket",
     "question": "Will Brazil win the 2026 FIFA World Cup?",
     "price": 0.18, "volume": 1_200_000, "liquidity": 95_000,
     "days": 45,  "category": "sports",  "trend": -0.003},
    {"id": "poly_sp500_high",   "platform": "Polymarket",
     "question": "Will the S&P 500 set a new all-time high before September?",
     "price": 0.67, "volume": 980_000,   "liquidity": 88_000,
     "days": 78,  "category": "finance", "trend": +0.002},
    {"id": "kalshi_btc_100k",   "platform": "Kalshi",
     "question": "Bitcoin above $100,000 on August 1 2026?",
     "price": 0.55, "volume": 1_100_000, "liquidity": 96_000,
     "days": 50,  "category": "crypto",  "trend": +0.005},
]

_REASONINGS = {
    "finance": (
        "Fed dot-plot and PCE trajectory suggest markets are under-pricing a cut. "
        "Employment data remains solid but leading indicators are softening, giving "
        "the Fed political cover to ease. Consensus among primary dealers shifted "
        "toward a July move in the past 10 days."
    ),
    "crypto": (
        "On-chain accumulation by large wallets is at a 6-month high. Options market "
        "implied volatility shows institutional hedging consistent with an anticipated "
        "move higher. Spot ETF inflow trend is accelerating and not yet reflected in pricing."
    ),
    "politics": (
        "Historical base rate for this type of executive action is ~28%. Current "
        "congressional dynamics and the administration's packed legislative calendar "
        "make a near-term signing unlikely. Timeline constraints are significant."
    ),
    "sports": (
        "Advanced metrics (RAPTOR, EPM) favour this team by +4.2 points. Recent form "
        "shows a 4-win streak with improving defensive efficiency. Home-court advantage "
        "and opponent injury reports support a slight upside vs. the current market."
    ),
}
_KEY_FACTORS = {
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

def _mock_ai_values(mkt: dict) -> dict:
    """Return mock AI analysis as a plain dict — no custom dataclasses."""
    rng  = random.Random(mkt["id"])
    bias = {"finance": 0.05, "crypto": 0.07, "politics": -0.04, "sports": 0.04}
    edge = bias.get(mkt["category"], 0.0) + rng.gauss(0, 0.04)
    cat  = mkt["category"]
    return {
        "estimated_prob": float(max(0.05, min(0.95, mkt["price"] + edge))),
        "confidence":     float(rng.uniform(0.58, 0.88)),
        "reasoning":      _REASONINGS.get(cat, "Analysis in progress."),
        "key_factors":    _KEY_FACTORS.get(cat, []),
        "uncertainty":    _FLAGS.get(cat, []),
    }


def _price_history_values(mkt: dict, n: int = 30) -> list[float]:
    rng   = random.Random(mkt["id"] + "h")
    trend = mkt.get("trend", 0.0)
    price = mkt["price"] - trend * n
    out   = []
    for _ in range(n):
        price = max(0.03, min(0.97, price + trend + rng.gauss(0, 0.007)))
        out.append(float(price))
    return out


# ── core analysis — returns ONLY plain Python primitives ─────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def _run_analysis() -> list[dict]:
    """
    Full predictive stack on mock markets.
    Returns plain dicts/lists with float/str/bool values only — no custom
    dataclass objects — so Streamlit's cache serialiser never trips.
    """
    bay    = BayesianEstimator()
    kelly  = KellyCriterion(max_fraction=0.15)
    ob_ana = OrderBookAnalyzer()
    mom    = MomentumAnalyzer()
    ens    = EnsemblePredictor(kelly)
    decay  = ResolutionDecayModel()
    ls     = LongshotBiasDetector()
    cat    = CategoryEdgeModel()
    qsc    = MarketQualityScorer()
    tmp    = TemporalPatternSignal()

    results = []
    for mkt in MOCK_MARKETS:
        ai_d    = _mock_ai_values(mkt)
        history = _price_history_values(mkt)

        # Build dataclass objects locally — they are NOT stored in the cache return
        ob = OrderBookSnapshot(
            bids=[(mkt["price"] - 0.01 * i, random.uniform(15_000, 80_000)) for i in range(1, 4)],
            asks=[(mkt["price"] + 0.01 * i, random.uniform(12_000, 65_000)) for i in range(1, 4)],
        )
        pp_list = [
            PricePoint(price=p, volume=50_000.0, timestamp=float(i))
            for i, p in enumerate(history)
        ]

        evidence = [((ai_d["estimated_prob"] - mkt["price"]) * 2, ai_d["confidence"])]
        bay_mean, bay_lo, bay_hi = bay.estimate(mkt["price"], evidence)

        ob_sigs  = ob_ana.analyze(ob)
        mom_sigs = mom.analyze(pp_list)

        raw_ens = ens.predict(
            market_price=mkt["price"],
            ai_probability=ai_d["estimated_prob"],
            ai_confidence=ai_d["confidence"],
            bayesian_estimate=float(bay_mean),
            microstructure_signals=ob_sigs,
            momentum_signals=mom_sigs,
            sentiment_signal=None,
        )

        adj_conf, adj_prob = decay.adjust_for_time(
            mkt["days"], raw_ens.confidence,
            raw_ens.estimated_probability, market_price=mkt["price"],
        )
        adj_prob = ls.adjust_probability(mkt["price"], float(adj_prob))
        q_score  = qsc.score(mkt["volume"], mkt["liquidity"], 0.0, mkt["price"])
        adj_conf  = float(adj_conf) * (0.5 + 0.5 * float(q_score))
        min_edge = tmp.adjusted_min_edge(cat.adjusted_min_edge(mkt["question"], 0.03))

        edge = float(adj_prob) - mkt["price"]
        kf   = float(kelly.compute(float(adj_prob), mkt["price"]))

        if abs(edge) < min_edge or adj_conf < 0.50:
            sig = "HOLD"
        elif edge > 0:
            sig = "BUY_YES"
        else:
            sig = "BUY_NO"

        # Serialise Signal objects to plain dicts immediately
        signal_dicts = [
            {
                "name":       str(s.name),
                "value":      float(s.value),
                "confidence": float(s.confidence),
                "weight":     float(s.weight) if s.weight else 1.0,
            }
            for s in raw_ens.signals
            if s.value != 0.0
        ]

        results.append({
            # market metadata (already primitives)
            "id":       mkt["id"],
            "platform": mkt["platform"],
            "question": mkt["question"],
            "price":    mkt["price"],
            "volume":   mkt["volume"],
            "liquidity":mkt["liquidity"],
            "days":     mkt["days"],
            "category": mkt["category"],
            # analysis outputs — all plain floats/strings/lists
            "ai_prob":      float(ai_d["estimated_prob"]),
            "ai_conf":      float(ai_d["confidence"]),
            "ai_reasoning": ai_d["reasoning"],
            "ai_factors":   list(ai_d["key_factors"]),
            "ai_flags":     list(ai_d["uncertainty"]),
            "bayesian":     float(bay_mean),
            "bay_lo":       float(bay_lo),
            "bay_hi":       float(bay_hi),
            "prob":         float(adj_prob),
            "conf":         float(adj_conf),
            "edge":         float(edge),
            "kelly":        float(kf),
            "quality":      float(q_score),
            "signal":       sig,
            "min_edge":     float(min_edge),
            "history":      history,   # list[float]
            "signals":      signal_dicts,
        })
    return results


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🤖 Polymarket AI Bot")
    st.markdown("Status: **🔵 MOCK MODE** — no real trades")
    st.divider()

    budget       = st.number_input("Paper budget (USD)", value=100.0, min_value=10.0, step=10.0)
    min_conf     = st.slider("Min confidence", 0.40, 0.90, 0.55, 0.05)
    st.divider()

    if st.button("🔄 Re-run Analysis", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("**Live mode** — add keys to Streamlit secrets:")
    st.code("ANTHROPIC_API_KEY = 'sk-ant-...'\nPOLYMARKET_API_KEY = '...'\nKALSHI_API_KEY = '...'", language="toml")


# ── load data ─────────────────────────────────────────────────────────────────

try:
    with st.spinner("Running predictive models…"):
        analyses = _run_analysis()
except Exception as _exc:
    st.error(f"Analysis failed: {_exc}")
    st.exception(_exc)
    st.stop()

analyses = [a for a in analyses if a["conf"] >= min_conf or a["signal"] == "HOLD"]

# ── header metrics ────────────────────────────────────────────────────────────

st.markdown("## 📊 Polymarket + Kalshi AI Trading Dashboard")

actionable = [a for a in analyses if a["signal"] != "HOLD"]
avg_edge   = float(np.mean([abs(a["edge"]) for a in actionable])) if actionable else 0.0
best       = max(analyses, key=lambda a: abs(a["edge"]))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Markets Scanned",    len(analyses))
c2.metric("Actionable Signals", len(actionable))
c3.metric("Avg Edge",           f"{avg_edge:.1%}")
c4.metric("Best Edge",          f"{abs(best['edge']):.1%}", best["platform"])
c5.metric("Mode",               "MOCK 🔵")
st.divider()

# ── tabs ──────────────────────────────────────────────────────────────────────

tab_scan, tab_deep, tab_brain, tab_arb, tab_port = st.tabs([
    "📡 Market Scanner", "🔬 Deep Analysis", "🧠 Learning Brain",
    "⚡ Arbitrage", "💼 Portfolio",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Market Scanner
# ═══════════════════════════════════════════════════════════════════════════════

with tab_scan:
    st.subheader("Live Market Scan — All Platforms")

    rows = []
    for a in sorted(analyses, key=lambda x: -abs(x["edge"])):
        rows.append({
            "Platform":  a["platform"],
            "Question":  a["question"][:62] + ("…" if len(a["question"]) > 62 else ""),
            "Market %":  f"{a['price']:.0%}",
            "AI Est.":   f"{a['prob']:.0%}",
            "Edge":      f"{a['edge']:+.1%}",
            "Conf.":     f"{a['conf']:.0%}",
            "Kelly":     f"{a['kelly']:.1%}",
            "Quality":   a["quality"],
            "Signal":    SIGNAL_ICONS[a["signal"]],
            "Days Left": a["days"],
        })

    df_scan = pd.DataFrame(rows)
    st.dataframe(
        df_scan,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Quality": st.column_config.ProgressColumn(
                label="Quality", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )

    # Sparkline grid
    st.markdown("#### 30-Day Price Trend")
    n_cols = 4
    cols   = st.columns(n_cols)
    _fill  = {"BUY_YES": "rgba(0,210,106,0.12)",
               "BUY_NO":  "rgba(255,75,75,0.12)",
               "HOLD":    "rgba(136,136,136,0.10)"}
    _line  = {"BUY_YES": "#00d26a", "BUY_NO": "#ff4b4b", "HOLD": "#888888"}

    for i, a in enumerate(analyses):
        with cols[i % n_cols]:
            fig = go.Figure(go.Scatter(
                y=a["history"], mode="lines",
                line=dict(color=_line[a["signal"]], width=2),
                fill="tozeroy", fillcolor=_fill[a["signal"]],
            ))
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0), height=70,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False, range=[0, 1]),
                showlegend=False,
            )
            lbl = (
                f'<span class="signal-yes">{SIGNAL_ICONS[a["signal"]]}</span>'
                if a["signal"] == "BUY_YES" else
                f'<span class="signal-no">{SIGNAL_ICONS[a["signal"]]}</span>'
                if a["signal"] == "BUY_NO" else
                f'<span class="signal-hold">{SIGNAL_ICONS[a["signal"]]}</span>'
            )
            st.markdown(
                f"**{a['question'][:38]}…**  "
                f"mkt {a['price']:.0%}→{a['prob']:.0%} ({a['edge']:+.1%}) {lbl}",
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Deep Analysis
# ═══════════════════════════════════════════════════════════════════════════════

with tab_deep:
    st.subheader("Deep Market Analysis")

    options = {
        f"{a['platform']} | {a['question'][:70]}": i for i, a in enumerate(analyses)
    }
    chosen = analyses[options[st.selectbox("Select a market", list(options.keys()))]]

    col_l, col_r = st.columns([3, 2])

    with col_l:
        # Probability comparison — native Streamlit components (no Plotly Indicator)
        sig_color = {"BUY_YES": "#00d26a", "BUY_NO": "#ff4b4b", "HOLD": "#888888"}
        st.markdown(f"""
<div style="text-align:center; padding:12px; background:#1e1e2e; border-radius:10px; margin-bottom:12px;">
  <div style="font-size:0.85rem; color:#aaa; margin-bottom:4px;">AI ESTIMATED PROBABILITY (YES)</div>
  <div style="font-size:3rem; font-weight:800; color:{sig_color[chosen['signal']]};">
    {chosen['prob']:.1%}
  </div>
  <div style="font-size:1rem; color:#ccc;">
    Market price: {chosen['price']:.1%} &nbsp;|&nbsp;
    Edge: <span style="color:{sig_color[chosen['signal']]};font-weight:700;">{chosen['edge']:+.1%}</span>
  </div>
</div>
""", unsafe_allow_html=True)

        # Probability comparison bar
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            x=["Market Price", "AI Estimate", "Bayesian Est."],
            y=[chosen["price"] * 100, chosen["prob"] * 100, chosen["bayesian"] * 100],
            marker_color=["#6366f1", sig_color[chosen["signal"]], "#f59e0b"],
            text=[f"{chosen['price']:.1%}", f"{chosen['prob']:.1%}", f"{chosen['bayesian']:.1%}"],
            textposition="outside",
        ))
        fig_cmp.update_layout(
            height=220, title="Probability Comparison",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, 110], gridcolor="#333", ticksuffix="%"),
            xaxis=dict(gridcolor="#333"),
            margin=dict(l=0, r=0, t=40, b=0), font_color="#ccc",
            showlegend=False,
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Price history
        dates = pd.date_range(end=pd.Timestamp.today(),
                              periods=len(chosen["history"]), freq="D")
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            x=list(dates), y=chosen["history"],
            mode="lines+markers",
            line=dict(color="#6366f1", width=2),
            marker=dict(size=3),
            name="Market Price",
        ))
        fig_h.add_hline(y=chosen["prob"], line_dash="dash",
                        line_color="#00d26a",
                        annotation_text=f"AI Est {chosen['prob']:.0%}",
                        annotation_position="right")
        fig_h.update_layout(
            title="30-Day Price History", height=210,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#333"), font_color="#ccc",
            yaxis=dict(gridcolor="#333", range=[0, 1]),
            margin=dict(l=0, r=70, t=40, b=0),
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with col_r:
        sig_color = {"BUY_YES": "#00d26a", "BUY_NO": "#ff4b4b", "HOLD": "#888888"}
        st.markdown(f"""
<div class="metric-card">
<b>Signal</b>: <span style="color:{sig_color[chosen['signal']]};font-size:1.1rem;font-weight:700">
{SIGNAL_ICONS[chosen['signal']]}</span><br>
<b>Edge</b>: {chosen['edge']:+.2%} &nbsp;|&nbsp; <b>Min required</b>: ±{chosen['min_edge']:.2%}<br>
<b>Kelly fraction</b>: {chosen['kelly']:.2%} of bankroll<br>
<b>Suggested size</b>: ${budget * chosen['kelly']:.2f}<br>
<b>Confidence</b>: {chosen['conf']:.0%} &nbsp;|&nbsp; <b>Quality</b>: {chosen['quality']:.2f}<br>
<b>Days to resolve</b>: {chosen['days']}<br>
<b>Bayesian est.</b>: {chosen['bayesian']:.2%} [{chosen['bay_lo']:.2%}–{chosen['bay_hi']:.2%}]
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

    # Signal breakdown
    st.markdown("#### Signal Breakdown")
    sigs = chosen["signals"]
    if sigs:
        names  = [s["name"].replace("_", " ").title() for s in sigs]
        vals   = [s["value"] for s in sigs]
        confs  = [s["confidence"] for s in sigs]
        colors = ["#00d26a" if v > 0 else "#ff4b4b" for v in vals]

        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(
            x=names, y=vals, marker_color=colors,
            text=[f"{v:+.2f}" for v in vals], textposition="outside",
            name="Signal Value",
        ))
        fig_s.add_trace(go.Scatter(
            x=names, y=confs, mode="markers",
            marker=dict(size=10, color="#f59e0b", symbol="diamond"),
            name="Confidence", yaxis="y2",
        ))
        fig_s.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Signal value", gridcolor="#333",
                       range=[-1.1, 1.1], zeroline=True, zerolinecolor="#555"),
            yaxis2=dict(title="Confidence", overlaying="y", side="right",
                        range=[0, 1.2], gridcolor="#333"),
            xaxis=dict(gridcolor="#333", tickangle=-20),
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=0, r=60, t=40, b=60), font_color="#ccc",
        )
        st.plotly_chart(fig_s, use_container_width=True)
    else:
        st.info("No directional signals for this market — all signals are neutral.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Learning Brain
# ═══════════════════════════════════════════════════════════════════════════════

with tab_brain:
    st.subheader("🧠 Self-Learning Signal Weight Brain")
    st.markdown(
        "Signal weights auto-update as markets resolve. After ~50 resolved markets, "
        "the bot weights reflect which signals are genuinely predictive for your "
        "market mix. State persists in `data/learning_state.json`."
    )

    try:
        _ens_b = EnsemblePredictor()
        _brain = LearningEngine(_ens_b, state_file=Path("data/learning_state.json"))
        _brain.load()
        summary = _brain.performance_summary()
    except Exception as _be:
        summary = {"resolved_markets": 0, "win_rate": 0.0,
                   "avg_brier_improvement": 0.0, "weight_drift": {},
                   "current_weights": dict(EnsemblePredictor.DEFAULT_WEIGHTS),
                   "pending_markets": 0}

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Resolved Markets",  summary["resolved_markets"])
    b2.metric("Win Rate",
              f"{summary['win_rate']:.0%}" if summary["resolved_markets"] else "N/A")
    b3.metric("Pending",           summary.get("pending_markets", 0))
    b4.metric("Avg Brier Δ",
              f"{summary['avg_brier_improvement']:+.4f}"
              if summary["resolved_markets"] else "N/A")

    st.markdown("#### Signal Weights: Current vs Default")
    defaults = EnsemblePredictor.DEFAULT_WEIGHTS
    current  = summary.get("current_weights", defaults)
    drift    = summary.get("weight_drift", {})

    w_rows = [
        {
            "Signal":   name.replace("_", " ").title(),
            "Default":  round(default_w, 3),
            "Current":  round(float(current.get(name, default_w)), 3),
            "Drift":    f"{float(drift.get(name, 0.0)):+.3f}",
            "Bar":      float(current.get(name, default_w)),
        }
        for name, default_w in defaults.items()
    ]
    st.dataframe(
        pd.DataFrame(w_rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Bar": st.column_config.ProgressColumn(
                label="Weight", min_value=0.0, max_value=8.0, format="%.2f"
            ),
        },
    )

    # Drift chart
    _dnames = [r["Signal"] for r in w_rows]
    _dvals  = [float(r["Drift"]) for r in w_rows]
    fig_d = go.Figure(go.Bar(
        x=_dnames, y=_dvals,
        marker_color=["#00d26a" if v >= 0 else "#ff4b4b" for v in _dvals],
        text=[f"{v:+.3f}" for v in _dvals], textposition="outside",
    ))
    fig_d.update_layout(
        title="Weight Drift from Default",
        height=250,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#333", tickangle=-20),
        yaxis=dict(gridcolor="#333", zeroline=True, zerolinecolor="#555"),
        margin=dict(l=0, r=0, t=40, b=80), font_color="#ccc",
    )
    st.plotly_chart(fig_d, use_container_width=True)

    if summary["resolved_markets"] == 0:
        st.info("No resolved markets yet — weights are at defaults. "
                "The brain starts learning as real markets resolve.")

    with st.expander("How the learning loop works"):
        st.markdown("""
| Step | Action |
|------|--------|
| Trade placed | Snapshot which signals fired, with values + confidence |
| Market resolves | Compute Brier improvement per signal vs naïve baseline |
| Weight update | `delta = 0.08 × improvement × confidence` |
| Regularisation | Each cycle: weights nudge 2% back toward defaults |
| Persist | Saved to `data/learning_state.json` on shutdown |
""")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Arbitrage
# ═══════════════════════════════════════════════════════════════════════════════

with tab_arb:
    st.subheader("⚡ Cross-Platform Arbitrage Scanner")
    st.markdown("Jaccard word-set similarity matching between Polymarket and Kalshi.")

    poly_raw   = [
        {"condition_id": a["id"], "question": a["question"], "best_ask": a["price"]}
        for a in analyses if a["platform"] == "Polymarket"
    ]
    kalshi_raw = [
        {"ticker": a["id"], "title": a["question"], "yes_ask": a["price"]}
        for a in analyses if a["platform"] == "Kalshi"
    ]

    try:
        arb = CrossMarketCorrelator().find_arbitrage(poly_raw, kalshi_raw, min_spread=0.0)
    except Exception:
        arb = []

    if arb:
        arb_rows = [
            {
                "Polymarket":   o.poly_question[:50] + "…",
                "Kalshi":       o.kalshi_question[:50] + "…",
                "Poly Price":   f"{o.polymarket_yes_price:.1%}",
                "Kalshi Price": f"{o.kalshi_yes_price:.1%}",
                "Gross Spread": f"{o.spread:+.1%}",
                "Net Spread":   f"{o.net_spread:+.1%}",
                "Direction":    o.direction,
            }
            for o in arb
        ]
        st.dataframe(pd.DataFrame(arb_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No arb opportunities above the spread threshold in this mock dataset. "
                "With real live market data, overlapping Poly/Kalshi markets would surface here.")

    with st.expander("How arb detection works"):
        st.markdown("""
1. Every Polymarket market is compared to every Kalshi market using Jaccard word-set similarity
2. Pairs above 40% similarity are flagged as covering the same event
3. Gross spread (price difference) − platform fees = **net spread**
4. Only net spreads above the minimum threshold are shown
""")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Portfolio
# ═══════════════════════════════════════════════════════════════════════════════

with tab_port:
    st.subheader("💼 Paper Portfolio Simulator")
    st.info("**MOCK MODE** — All P&L is simulated. No real trades are placed.")

    rng_p    = random.Random(42)
    positions = []
    pnl_total = 0.0

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
        won = rng_p.random() < bias
        pnl = size * (1.0 / a["price"] - 1.0) if won else -size
        pnl_total += pnl
        positions.append({
            "Platform":  a["platform"],
            "Question":  a["question"][:52] + "…",
            "Direction": a["signal"].replace("BUY_", ""),
            "Size":      f"${size:.2f}",
            "Entry":     f"{a['price']:.0%}",
            "AI Est.":   f"{a['prob']:.0%}",
            "Edge":      f"{a['edge']:+.1%}",
            "Mock P&L":  f"${pnl:+.2f}",
            "Result":    "✅ WIN" if won else "❌ LOSS",
        })

    if positions:
        p1, p2, p3 = st.columns(3)
        p1.metric("Simulated Trades", len(positions))
        p2.metric("Net Mock P&L",     f"${pnl_total:+.2f}")
        p3.metric("ROI",              f"{pnl_total / budget:.1%}")

        _qs   = [p["Question"][:30] for p in positions]
        _pnls = [float(p["Mock P&L"].replace("$", "").replace("+", "")) for p in positions]
        fig_pnl = go.Figure(go.Bar(
            x=_qs, y=_pnls,
            marker_color=["#00d26a" if v >= 0 else "#ff4b4b" for v in _pnls],
            text=[f"${v:+.2f}" for v in _pnls], textposition="outside",
        ))
        fig_pnl.update_layout(
            title="Simulated Trade P&L", height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#333", tickangle=-30),
            yaxis=dict(gridcolor="#333", zeroline=True, zerolinecolor="#555"),
            margin=dict(l=0, r=0, t=40, b=100), font_color="#ccc",
        )
        st.plotly_chart(fig_pnl, use_container_width=True)
        st.dataframe(pd.DataFrame(positions), use_container_width=True, hide_index=True)
    else:
        st.info("No positions meet the current confidence/edge thresholds.")

    st.divider()
    st.markdown("""
#### Pre-live checklist
- [ ] Run in mock mode for **≥ 2 weeks** and review decisions manually
- [ ] Run `python scripts/quickstart.py` to verify all API connections
- [ ] Confirm risk limits are appropriate for your bankroll
- [ ] Start with **< $5 per trade** when going live
- [ ] Only then set `MOCK_MODE=false` in `.env`
""")

# ── footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"🤖 Polymarket AI Bot | **MOCK MODE** | "
    f"No real trades | Last refreshed: {time.strftime('%H:%M UTC', time.gmtime())}"
)
