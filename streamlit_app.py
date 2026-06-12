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

# ── path ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from shared.advanced_signals import (
    CategoryEdgeModel,
    LongshotBiasDetector,
    MarketQualityScorer,
    TemporalPatternSignal,
)
from shared.claude_agent import MarketAnalysis
from shared.learning_engine import LearningEngine
from shared.predictive_models import (
    BayesianEstimator,
    CalibrationTracker,
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

# ── page config (must be first) ───────────────────────────────────────────────
st.set_page_config(
    page_title="Polymarket AI Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .signal-yes  { color:#00d26a; font-weight:700; font-size:1.05rem; }
  .signal-no   { color:#ff4b4b; font-weight:700; font-size:1.05rem; }
  .signal-hold { color:#aaaaaa; font-weight:600; font-size:1.05rem; }
  .metric-card { background:#1e1e2e; border-radius:8px; padding:12px 16px;
                 margin-bottom:8px; border-left:3px solid #6366f1; }
  .badge-mock  { background:#2563eb; color:#fff; border-radius:4px;
                 padding:2px 8px; font-size:0.8rem; font-weight:700; }
  .badge-live  { background:#16a34a; color:#fff; border-radius:4px;
                 padding:2px 8px; font-size:0.8rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

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
        "the Fed political cover to ease. Consensus among primary dealers has shifted "
        "toward a July move in the past 10 days."
    ),
    "crypto": (
        "On-chain accumulation by large wallets is at a 6-month high. Options market "
        "implied volatility shows institutional hedging consistent with an anticipated "
        "move higher. The spot ETF inflow trend is accelerating and not yet reflected "
        "in current pricing."
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

def _mock_ai(mkt: dict) -> MarketAnalysis:
    rng = random.Random(mkt["id"])
    bias = {"finance": 0.05, "crypto": 0.07, "politics": -0.04, "sports": 0.04}
    edge = bias.get(mkt["category"], 0.0) + rng.gauss(0, 0.04)
    est  = max(0.05, min(0.95, mkt["price"] + edge))
    cat  = mkt["category"]
    return MarketAnalysis(
        market_id=mkt["id"], question=mkt["question"],
        market_price=mkt["price"], estimated_probability=est,
        confidence=rng.uniform(0.58, 0.88),
        reasoning=_REASONINGS.get(cat, "Analysis in progress."),
        key_factors=_KEY_FACTORS.get(cat, []),
        uncertainty_flags=_FLAGS.get(cat, []),
    )


def _price_history(mkt: dict, n: int = 30) -> list[PricePoint]:
    rng   = random.Random(mkt["id"] + "h")
    trend = mkt.get("trend", 0.0)
    price = mkt["price"] - trend * n
    out   = []
    for i in range(n):
        price = max(0.03, min(0.97, price + trend + rng.gauss(0, 0.007)))
        out.append(PricePoint(price=price, volume=rng.uniform(40_000, 180_000), timestamp=float(i)))
    return out


def _order_book(mkt: dict) -> OrderBookSnapshot:
    rng = random.Random(mkt["id"] + "ob")
    p   = mkt["price"]
    bids = [(p - 0.01 * i, rng.uniform(15_000, 80_000)) for i in range(1, 4)]
    asks = [(p + 0.01 * i, rng.uniform(12_000, 65_000)) for i in range(1, 4)]
    return OrderBookSnapshot(bids=bids, asks=asks)


# ── core analysis (cached) ────────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def _run_analysis() -> list[dict]:
    """Full predictive stack on mock markets. Result cached 2 min."""
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
        ai      = _mock_ai(mkt)
        history = _price_history(mkt)
        ob      = _order_book(mkt)

        evidence = [((ai.estimated_probability - mkt["price"]) * 2, ai.confidence)]
        bay_mean, bay_lo, bay_hi = bay.estimate(mkt["price"], evidence)

        ob_sigs  = ob_ana.analyze(ob)
        mom_sigs = mom.analyze(history)

        raw_ens = ens.predict(
            market_price=mkt["price"],
            ai_probability=ai.estimated_probability,
            ai_confidence=ai.confidence,
            bayesian_estimate=bay_mean,
            microstructure_signals=ob_sigs,
            momentum_signals=mom_sigs,
            sentiment_signal=None,
        )

        adj_conf, adj_prob = decay.adjust_for_time(
            mkt["days"], raw_ens.confidence,
            raw_ens.estimated_probability, market_price=mkt["price"],
        )
        adj_prob = ls.adjust_probability(mkt["price"], adj_prob)
        q_score  = qsc.score(mkt["volume"], mkt["liquidity"], 0.0, mkt["price"])
        adj_conf *= (0.5 + 0.5 * q_score)
        min_edge = tmp.adjusted_min_edge(cat.adjusted_min_edge(mkt["question"], 0.03))

        edge = adj_prob - mkt["price"]
        kf   = kelly.compute(adj_prob, mkt["price"])

        if abs(edge) < min_edge or adj_conf < 0.50:
            sig = "HOLD"
        elif edge > 0:
            sig = "BUY_YES"
        else:
            sig = "BUY_NO"

        results.append({
            "mkt": mkt,
            "ai": ai,
            "bayesian": bay_mean,
            "bayesian_ci": (bay_lo, bay_hi),
            "ob_signals": ob_sigs,
            "mom_signals": mom_sigs,
            "all_signals": raw_ens.signals,
            "prob": adj_prob,
            "conf": adj_conf,
            "edge": edge,
            "kelly": kf,
            "quality": q_score,
            "signal": sig,
            "min_edge": min_edge,
            "history": [pp.price for pp in history],
        })
    return results


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🤖 Polymarket AI Bot")
    mode_label = '<span class="badge-mock">MOCK MODE</span>' if True else '<span class="badge-live">LIVE</span>'
    st.markdown(f"Status: {mode_label} &nbsp;✅ Safe", unsafe_allow_html=True)
    st.divider()

    budget = st.number_input("Paper budget (USD)", value=100.0, min_value=10.0, step=10.0)
    min_conf = st.slider("Min confidence threshold", 0.40, 0.90, 0.55, 0.05)
    min_edge_override = st.slider("Min edge threshold", 0.02, 0.12, 0.04, 0.01)
    st.divider()

    auto_refresh = st.checkbox("Auto-refresh (90s)", value=False)
    if auto_refresh:
        st.info("Page will refresh every 90 seconds.")
        time.sleep(0.1)
        st.rerun()

    if st.button("🔄 Re-run Analysis", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("**API keys** — set in `.env` to enable live data:")
    st.code("ANTHROPIC_API_KEY=sk-...\nPOLYMARKET_API_KEY=...\nKALSHI_API_KEY=...", language="bash")
    st.caption("Run `python scripts/quickstart.py` for guided setup.")


# ── load data ─────────────────────────────────────────────────────────────────

with st.spinner("Running predictive models…"):
    analyses = _run_analysis()

# apply sidebar filters
analyses = [a for a in analyses if a["conf"] >= min_conf or a["signal"] == "HOLD"]

# ── top header ────────────────────────────────────────────────────────────────

st.markdown("## 📊 Polymarket + Kalshi AI Trading Dashboard")
col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns(5)

actionable    = [a for a in analyses if a["signal"] != "HOLD"]
avg_edge      = np.mean([abs(a["edge"]) for a in actionable]) if actionable else 0.0
best          = max(analyses, key=lambda a: abs(a["edge"]))
arb_pairs     = [
    (a, b) for a in analyses for b in analyses
    if a["mkt"]["id"] != b["mkt"]["id"]
    and a["mkt"]["question"][:25].lower() == b["mkt"]["question"][:25].lower()
    and abs(a["mkt"]["price"] - b["mkt"]["price"]) > 0.03
]

col_h1.metric("Markets Scanned",   len(analyses))
col_h2.metric("Actionable Signals", len(actionable))
col_h3.metric("Avg Edge",          f"{avg_edge:.1%}")
col_h4.metric("Best Edge",         f"{abs(best['edge']):.1%}", best["mkt"]["platform"])
col_h5.metric("Arb Opportunities", len(arb_pairs))

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

    SIGNAL_ICONS = {"BUY_YES": "🟢 BUY YES", "BUY_NO": "🔴 BUY NO", "HOLD": "⚪ HOLD"}

    rows = []
    for a in sorted(analyses, key=lambda x: -abs(x["edge"])):
        m   = a["mkt"]
        bar = "█" * int(a["prob"] * 10) + "░" * (10 - int(a["prob"] * 10))
        rows.append({
            "Platform":  m["platform"],
            "Question":  m["question"][:62] + ("…" if len(m["question"]) > 62 else ""),
            "Market":    f"{m['price']:.0%}",
            "AI Est.":   f"{a['prob']:.0%}",
            "Edge":      f"{a['edge']:+.1%}",
            "Conf.":     f"{a['conf']:.0%}",
            "Kelly":     f"{a['kelly']:.1%}",
            "Quality":   f"{a['quality']:.2f}",
            "Signal":    SIGNAL_ICONS[a["signal"]],
            "Days Left": m["days"],
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Edge":    st.column_config.TextColumn(width="small"),
            "Signal":  st.column_config.TextColumn(width="medium"),
            "Quality": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
        },
    )

    # Sparkline grid
    st.markdown("#### Price History (30 days)")
    cols = st.columns(4)
    for i, a in enumerate(analyses):
        with cols[i % 4]:
            hist = a["history"]
            color    = "#00d26a" if a["signal"] == "BUY_YES" else ("#ff4b4b" if a["signal"] == "BUY_NO" else "#888888")
            fillclr  = {"#00d26a": "rgba(0,210,106,0.12)",
                        "#ff4b4b": "rgba(255,75,75,0.12)",
                        "#888888": "rgba(136,136,136,0.10)"}[color]
            fig = go.Figure(go.Scatter(
                y=hist, mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy", fillcolor=fillclr,
            ))
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0), height=80,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False), yaxis=dict(visible=False, range=[0, 1]),
                showlegend=False,
            )
            sig_html = (
                f'<span class="signal-yes">{SIGNAL_ICONS[a["signal"]]}</span>'
                if a["signal"] == "BUY_YES" else
                f'<span class="signal-no">{SIGNAL_ICONS[a["signal"]]}</span>'
                if a["signal"] == "BUY_NO" else
                f'<span class="signal-hold">{SIGNAL_ICONS[a["signal"]]}</span>'
            )
            st.markdown(
                f"**{a['mkt']['question'][:38]}…**<br>"
                f"mkt {a['mkt']['price']:.0%} → est {a['prob']:.0%} "
                f"({a['edge']:+.1%}) {sig_html}",
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Deep Analysis
# ═══════════════════════════════════════════════════════════════════════════════

with tab_deep:
    st.subheader("Deep Market Analysis")

    market_options = {
        f"{a['mkt']['platform']} | {a['mkt']['question'][:70]}": i
        for i, a in enumerate(analyses)
    }
    chosen_label = st.selectbox("Select a market", list(market_options.keys()))
    chosen_idx   = market_options[chosen_label]
    a = analyses[chosen_idx]
    m = a["mkt"]

    col_l, col_r = st.columns([3, 2])

    with col_l:
        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=a["prob"] * 100,
            delta={"reference": m["price"] * 100, "valueformat": ".1f",
                   "suffix": "pp vs market"},
            number={"suffix": "%", "valueformat": ".1f"},
            gauge={
                "axis":  {"range": [0, 100]},
                "bar":   {"color": "#6366f1"},
                "steps": [
                    {"range": [0, 30],  "color": "#ff4b4b40"},
                    {"range": [30, 70], "color": "#ffffff10"},
                    {"range": [70, 100],"color": "#00d26a40"},
                ],
                "threshold": {"line": {"color": "#ffffff", "width": 2},
                              "thickness": 0.8, "value": m["price"] * 100},
            },
            title={"text": "AI Estimated Probability (YES)"},
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20),
                                paper_bgcolor="rgba(0,0,0,0)", font_color="#ffffff")
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Price history chart
        hist_dates = pd.date_range(end=pd.Timestamp.today(), periods=len(a["history"]), freq="D")
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            x=hist_dates, y=a["history"], mode="lines+markers",
            line=dict(color="#6366f1", width=2), marker=dict(size=4),
            name="Market Price",
        ))
        fig_h.add_hline(y=a["prob"], line_dash="dash", line_color="#00d26a",
                        annotation_text=f"AI Est. {a['prob']:.0%}", annotation_position="right")
        fig_h.update_layout(
            title="30-Day Price History", height=220,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333", range=[0, 1]),
            margin=dict(l=0, r=60, t=40, b=0), font_color="#ccc",
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with col_r:
        # Key metrics
        signal_color = {"BUY_YES": "#00d26a", "BUY_NO": "#ff4b4b", "HOLD": "#888888"}
        st.markdown(f"""
<div class="metric-card">
<b>Signal</b>: <span style="color:{signal_color[a['signal']]}; font-size:1.2rem; font-weight:700">{SIGNAL_ICONS[a['signal']]}</span><br>
<b>Edge</b>: {a['edge']:+.2%} &nbsp;|&nbsp; <b>Min required</b>: ±{a['min_edge']:.2%}<br>
<b>Kelly fraction</b>: {a['kelly']:.2%} of bankroll<br>
<b>Suggested size</b>: ${budget * a['kelly']:.2f}<br>
<b>Confidence</b>: {a['conf']:.0%} &nbsp;|&nbsp; <b>Quality</b>: {a['quality']:.2f}<br>
<b>Days to resolution</b>: {m['days']}<br>
<b>Category</b>: {m['category'].title()}<br>
<b>Bayesian est.</b>: {a['bayesian']:.2%} &nbsp;[{a['bayesian_ci'][0]:.2%}–{a['bayesian_ci'][1]:.2%}]
</div>
""", unsafe_allow_html=True)

        # Claude AI reasoning
        st.markdown("**🤖 Claude AI Reasoning**")
        st.info(a["ai"].reasoning)

        st.markdown("**✅ Key Factors**")
        for f in a["ai"].key_factors:
            st.markdown(f"- {f}")

        st.markdown("**⚠️ Uncertainty Flags**")
        for f in a["ai"].uncertainty_flags:
            st.markdown(f"- {f}")

    # Signal breakdown bar chart
    st.markdown("#### Signal Breakdown")
    sigs = [(s.name, s.value, s.confidence, s.weight)
            for s in a["all_signals"] if s.value != 0.0]
    if sigs:
        names, vals, confs, weights = zip(*sigs)
        colors = ["#00d26a" if v > 0 else "#ff4b4b" for v in vals]
        fig_sig = go.Figure()
        fig_sig.add_trace(go.Bar(
            x=list(names), y=list(vals), marker_color=colors,
            text=[f"{v:+.2f}" for v in vals], textposition="outside",
            name="Signal Value",
        ))
        fig_sig.add_trace(go.Scatter(
            x=list(names), y=list(confs), mode="markers",
            marker=dict(size=10, color="#f59e0b", symbol="diamond"),
            name="Confidence", yaxis="y2",
        ))
        fig_sig.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Signal value [-1,+1]", gridcolor="#333",
                       range=[-1.1, 1.1], zeroline=True, zerolinecolor="#555"),
            yaxis2=dict(title="Confidence", overlaying="y", side="right",
                        range=[0, 1.1], gridcolor="#333"),
            xaxis=dict(gridcolor="#333"), legend=dict(orientation="h", y=1.1),
            margin=dict(l=0, r=60, t=40, b=60), font_color="#ccc",
        )
        st.plotly_chart(fig_sig, use_container_width=True)
    else:
        st.info("All signals are neutral — no directional edge detected for this market.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Learning Brain
# ═══════════════════════════════════════════════════════════════════════════════

with tab_brain:
    st.subheader("🧠 Self-Learning Signal Weight Brain")
    st.markdown(
        "The bot's **LearningEngine** adjusts these weights automatically as markets resolve. "
        "After ~50 resolved markets the weights reflect which signals are genuinely predictive "
        "for your specific market mix. Weights are persisted across restarts in `data/learning_state.json`."
    )

    ens_for_brain = EnsemblePredictor()
    brain         = LearningEngine(ens_for_brain, state_file=Path("data/learning_state.json"))
    brain.load()
    summary       = brain.performance_summary()

    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    col_b1.metric("Resolved Markets",    summary["resolved_markets"])
    col_b2.metric("Win Rate",            f"{summary['win_rate']:.0%}"
                  if summary["resolved_markets"] else "N/A")
    col_b3.metric("Pending Signals",     summary.get("pending_markets", 0))
    col_b4.metric("Avg Brier Δ",         f"{summary['avg_brier_improvement']:+.4f}"
                  if summary["resolved_markets"] else "N/A")

    st.markdown("#### Current Signal Weights vs Defaults")
    defaults = EnsemblePredictor.DEFAULT_WEIGHTS
    current  = summary["current_weights"]
    drift    = summary["weight_drift"]

    weight_rows = []
    for name, default_w in defaults.items():
        cur_w    = current.get(name, default_w)
        drift_w  = drift.get(name, 0.0)
        weight_rows.append({
            "Signal":       name.replace("_", " ").title(),
            "Default":      round(default_w, 3),
            "Current":      round(cur_w, 3),
            "Drift":        f"{drift_w:+.3f}",
            "Learned":      cur_w,
        })

    df_w = pd.DataFrame(weight_rows)
    st.dataframe(
        df_w[["Signal", "Default", "Current", "Drift", "Learned"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Learned": st.column_config.ProgressColumn(
                label="Current Weight",
                min_value=0.0, max_value=8.0, format="%.2f"
            ),
        },
    )

    # Weight drift bar chart
    sig_names = [r["Signal"] for r in weight_rows]
    drift_vals = [float(r["Drift"]) for r in weight_rows]
    drift_colors = ["#00d26a" if v >= 0 else "#ff4b4b" for v in drift_vals]

    fig_drift = go.Figure(go.Bar(
        x=sig_names, y=drift_vals, marker_color=drift_colors,
        text=[f"{v:+.3f}" for v in drift_vals], textposition="outside",
    ))
    fig_drift.update_layout(
        title="Weight Drift from Default (learning signal)",
        height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#333"), font_color="#ccc",
        yaxis=dict(title="Δ weight", gridcolor="#333", zeroline=True, zerolinecolor="#555"),
        margin=dict(l=0, r=0, t=40, b=80),
    )
    st.plotly_chart(fig_drift, use_container_width=True)

    if summary["resolved_markets"] == 0:
        st.info(
            "No resolved markets yet — weights are at defaults. "
            "As markets resolve, the brain will automatically shift weight toward "
            "the signals that proved most predictive."
        )

    st.markdown("#### How it learns")
    st.markdown("""
| Step | What happens |
|---|---|
| 1. Trade fires | Signal snapshot (value, confidence, weight) saved in memory |
| 2. Market resolves | For each signal: compute Brier improvement vs naïve market-price baseline |
| 3. Weight update | `delta = 0.08 × improvement × confidence` applied to `EnsemblePredictor` |
| 4. Regularisation | Every cycle weights drift 2% back toward defaults (prevents overfitting) |
| 5. Persist | Saved to `data/learning_state.json` — survives restarts |
""")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Arbitrage
# ═══════════════════════════════════════════════════════════════════════════════

with tab_arb:
    st.subheader("⚡ Cross-Platform Arbitrage Scanner")
    st.markdown("Jaccard similarity matching between Polymarket and Kalshi markets. Net spread after fees.")

    poly_raw   = [
        {"condition_id": a["mkt"]["id"], "question": a["mkt"]["question"], "best_ask": a["mkt"]["price"]}
        for a in analyses if a["mkt"]["platform"] == "Polymarket"
    ]
    kalshi_raw = [
        {"ticker": a["mkt"]["id"], "title": a["mkt"]["question"], "yes_ask": a["mkt"]["price"]}
        for a in analyses if a["mkt"]["platform"] == "Kalshi"
    ]

    correlator = CrossMarketCorrelator()
    arb_opps   = correlator.find_arbitrage(poly_raw, kalshi_raw, min_spread=0.0)

    if arb_opps:
        arb_rows = []
        for opp in arb_opps:
            arb_rows.append({
                "Polymarket Question": opp.poly_question[:55] + "…",
                "Kalshi Question":     opp.kalshi_question[:55] + "…",
                "Poly Price":          f"{opp.polymarket_yes_price:.1%}",
                "Kalshi Price":        f"{opp.kalshi_yes_price:.1%}",
                "Gross Spread":        f"{opp.spread:+.1%}",
                "Net Spread":          f"{opp.net_spread:+.1%}",
                "Direction":           opp.direction,
            })
        st.dataframe(pd.DataFrame(arb_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No arbitrage opportunities found above the minimum spread threshold.")
        st.markdown("""
**How arb detection works:**
1. Every Polymarket market is compared against every Kalshi market using Jaccard word-set similarity
2. Matches above 40% similarity are compared on price
3. Gross spread − platform fees = net spread
4. Only net spreads > threshold are flagged

Add more Kalshi markets with overlapping questions to surface real opportunities.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Portfolio
# ═══════════════════════════════════════════════════════════════════════════════

with tab_port:
    st.subheader("💼 Portfolio Simulator (Mock Mode)")
    st.info("Running in **MOCK MODE** — no real trades are placed. P&L is simulated from signals.")

    # Simulate a small paper portfolio
    rng_port = random.Random(42)
    positions = []
    pnl_total = 0.0

    for a in analyses:
        if a["signal"] == "HOLD":
            continue
        size = min(budget * a["kelly"], budget * 0.15)
        if size < 1.0:
            continue
        entry_price = a["mkt"]["price"]
        # Random mock outcome: favour the edge direction
        bias = 0.6 if (
            (a["signal"] == "BUY_YES" and a["prob"] > entry_price) or
            (a["signal"] == "BUY_NO"  and a["prob"] < entry_price)
        ) else 0.4
        won   = rng_port.random() < bias
        pnl   = size * (1 / entry_price - 1) if won else -size
        pnl_total += pnl
        positions.append({
            "Platform":   a["mkt"]["platform"],
            "Question":   a["mkt"]["question"][:55] + "…",
            "Direction":  a["signal"].replace("BUY_", ""),
            "Size ($)":   f"${size:.2f}",
            "Entry":      f"{entry_price:.1%}",
            "Est.":       f"{a['prob']:.1%}",
            "Edge":       f"{a['edge']:+.1%}",
            "Mock P&L":   f"${pnl:+.2f}",
            "Status":     "✅ WIN" if won else "❌ LOSS",
        })

    if positions:
        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("Simulated Trades",    len(positions))
        col_p2.metric("Net Mock P&L",        f"${pnl_total:+.2f}",
                      delta_color="normal" if pnl_total >= 0 else "inverse")
        col_p3.metric("ROI",                 f"{pnl_total / budget:.1%}")

        # P&L bar chart
        qs   = [p["Question"][:35] + "…" for p in positions]
        pnls = [float(p["Mock P&L"].replace("$", "").replace("+", "")) for p in positions]
        fig_pnl = go.Figure(go.Bar(
            x=qs, y=pnls,
            marker_color=["#00d26a" if v >= 0 else "#ff4b4b" for v in pnls],
            text=[f"${v:+.2f}" for v in pnls], textposition="outside",
        ))
        fig_pnl.update_layout(
            title="Simulated Trade P&L", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#333", tickangle=-30), font_color="#ccc",
            yaxis=dict(gridcolor="#333", zeroline=True, zerolinecolor="#555"),
            margin=dict(l=0, r=0, t=40, b=120),
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

        st.dataframe(pd.DataFrame(positions), use_container_width=True, hide_index=True)
    else:
        st.info("No positions meet the current confidence/edge thresholds.")

    st.divider()
    st.markdown("""
#### Before going live
- [ ] Run in mock mode for **at least 2 weeks**
- [ ] Verify all API connections (`python scripts/quickstart.py`)
- [ ] Confirm risk limits are appropriate for your bankroll
- [ ] Start with **tiny position sizes** (< $5 per trade)
- [ ] Only then set `MOCK_MODE=false` in `.env`
""")

# ── footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🤖 Polymarket AI Bot — **MOCK MODE** | "
    "Data is simulated. No real trades executed. | "
    f"Analysis refreshes every 2 min | Last run: {time.strftime('%H:%M:%S UTC', time.gmtime())}"
)
