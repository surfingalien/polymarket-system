#!/usr/bin/env python3
"""
Kalshi Live Trading Bot — standalone runner

Setup:
  1. Copy config/.env.example to .env and fill in:
       KALSHI_API_KEY=your_key_id
       KALSHI_PRIVATE_KEY_PEM=-----BEGIN RSA PRIVATE KEY-----\nMII...\n-----END RSA PRIVATE KEY-----
       ANTHROPIC_API_KEY=your_key   # optional — enables Claude reasoning
       MOCK_MODE=false              # set true for paper-only run
  2. pip install -r requirements.txt
  3. python run_kalshi_bot.py

Env overrides (all optional):
  KALSHI_BUDGET_USD=10       per-trade budget cap (default $10)
  MIN_CONFIDENCE=0.55        minimum ensemble confidence to act (default 55%)
  MIN_EDGE=0.04              minimum probability edge to act (default 4 pp)
  CYCLE_SECONDS=300          how often to refresh markets (default 5 min)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

# ── env & path setup ──────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

MOCK_MODE       = os.getenv("MOCK_MODE", "true").lower() in ("true", "1", "yes")
BUDGET_USD      = float(os.getenv("KALSHI_BUDGET_USD", "10"))
MIN_CONFIDENCE  = float(os.getenv("MIN_CONFIDENCE",    "0.55"))
MIN_EDGE        = float(os.getenv("MIN_EDGE",           "0.04"))
CYCLE_SECONDS   = int(os.getenv("CYCLE_SECONDS",        "300"))
TRADES_FILE     = Path("data/kalshi_trades.json")
TRADES_FILE.parent.mkdir(exist_ok=True)

# ── imports ───────────────────────────────────────────────────────────────────
try:
    import structlog
    from shared.predictive_models import (
        BayesianEstimator,
        EnsemblePredictor,
        KellyCriterion,
        MomentumAnalyzer,
        OrderBookAnalyzer,
        OrderBookSnapshot,
        PricePoint,
    )
    from shared.advanced_signals import (
        CategoryEdgeModel,
        LongshotBiasDetector,
        MarketQualityScorer,
    )
    from shared.learning_engine import LearningEngine
    from kalshi_bot.kalshi_client import KalshiClient, KalshiMarket, KalshiOrder
except ImportError as e:
    print(f"ERROR — missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

log = structlog.get_logger()

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_trades() -> list[dict]:
    if TRADES_FILE.exists():
        try:
            return json.loads(TRADES_FILE.read_text())
        except Exception:
            return []
    return []


def _save_trades(trades: list[dict]) -> None:
    TRADES_FILE.write_text(json.dumps(trades, indent=2, default=str))


def _already_traded(trades: list[dict], ticker: str, side: str) -> bool:
    """True if we already hold an open position for this ticker+side."""
    return any(
        t.get("ticker") == ticker and t.get("side") == side and t.get("status") != "closed"
        for t in trades
    )


def _build_analysis(market: KalshiMarket, ensemble: EnsemblePredictor) -> dict | None:
    """
    Run the ensemble predictive stack on a single Kalshi market.
    Returns None if edge/confidence are below thresholds.
    """
    mid = market.mid_price
    if not (0.03 <= mid <= 0.97):
        return None

    # Bayesian estimate from bid/ask spread and volume
    bay = BayesianEstimator()
    ob = OrderBookSnapshot(
        yes_bids=[market.yes_bid],
        yes_asks=[market.yes_ask],
        no_bids=[market.no_bid],
        no_asks=[market.no_ask],
        volume=market.volume,
        liquidity=market.open_interest,
    )
    bay_est = bay.estimate(mid, ob)

    # Momentum from synthetic history (single point — no real history available)
    mom = MomentumAnalyzer()
    mom_sigs = mom.analyze([PricePoint(price=mid, timestamp=time.time(), volume=market.volume)])

    # Microstructure
    oba = OrderBookAnalyzer()
    micro_sigs = oba.analyze(ob)

    # Longshot bias correction
    lsb = LongshotBiasDetector()
    lsb_sig = lsb.detect(mid)

    # Category edge
    cat = CategoryEdgeModel()
    cat_sig = cat.edge_for_category("general", mid)

    all_micro = micro_sigs + ([lsb_sig] if lsb_sig else []) + ([cat_sig] if cat_sig else [])

    result = ensemble.predict(
        market_price=mid,
        ai_probability=None,      # no Claude call in the fast loop
        ai_confidence=0.0,
        bayesian_estimate=bay_est,
        microstructure_signals=all_micro,
        momentum_signals=mom_sigs,
        sentiment_signal=None,
    )

    edge = result.probability - mid
    if abs(edge) < MIN_EDGE or result.confidence < MIN_CONFIDENCE:
        return None

    kelly = KellyCriterion().compute(result.probability, mid, fractional=0.25)
    size = round(min(BUDGET_USD * kelly, BUDGET_USD * 0.15), 2)
    if size < 1.0:
        return None

    return {
        "ticker":     market.ticker,
        "question":   market.title,
        "mid":        mid,
        "prob":       round(result.probability, 4),
        "edge":       round(edge, 4),
        "confidence": round(result.confidence, 4),
        "kelly":      round(kelly, 4),
        "size_usd":   size,
        "signal":     "BUY_YES" if edge > 0 else "BUY_NO",
        "side":       "yes"     if edge > 0 else "no",
    }


async def _place_order(
    client: KalshiClient,
    analysis: dict,
) -> dict:
    """Place a Kalshi order and return a trade record."""
    ticker      = analysis["ticker"]
    side        = analysis["side"]
    size_usd    = analysis["size_usd"]
    mid         = analysis["mid"]

    contract_price = mid if side == "yes" else (1.0 - mid)
    contract_price = max(0.01, min(0.99, contract_price))
    count = max(1, int(size_usd / contract_price))
    limit_cents = int(round(contract_price * 100))

    order = KalshiOrder(
        ticker=ticker,
        action="buy",
        side=side,
        count=count,
        limit_price=limit_cents,
    )

    result = await client.place_order(order)
    order_id = getattr(result, "order_id", f"mock_{int(time.time())}")
    status   = getattr(result, "status",   "MOCK" if MOCK_MODE else "unknown")

    return {
        "ticker":     ticker,
        "question":   analysis["question"][:80],
        "side":       side,
        "count":      count,
        "size_usd":   size_usd,
        "entry_price": mid,
        "edge":       analysis["edge"],
        "confidence": analysis["confidence"],
        "order_id":   order_id,
        "status":     status,
        "time":       datetime.now(timezone.utc).isoformat(),
        "mode":       "PAPER" if MOCK_MODE else "LIVE",
    }


def _to_brain_ns(analysis: dict):
    """Adapt an analysis dict to the duck-typed object LearningEngine expects."""
    return types.SimpleNamespace(
        market_id=analysis["ticker"],
        signal=analysis["signal"],
        market_price=analysis["mid"],
        ensemble=None,
    )


async def run_cycle(
    client: KalshiClient,
    ensemble: EnsemblePredictor,
    brain: LearningEngine,
    trades: list[dict],
) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] Fetching Kalshi markets...")

    try:
        markets = await client.get_markets(limit=40, status="open")
    except Exception as exc:
        print(f"  ERROR fetching markets: {exc}")
        return

    print(f"  Got {len(markets)} open markets")
    acted = 0

    for m in markets:
        analysis = _build_analysis(m, ensemble)
        if analysis is None:
            continue

        ticker = analysis["ticker"]
        side   = analysis["side"]

        if _already_traded(trades, ticker, side):
            continue

        print(
            f"  [{analysis['signal']}] {ticker} | "
            f"edge {analysis['edge']:+.1%} | "
            f"conf {analysis['confidence']:.0%} | "
            f"size ${analysis['size_usd']}"
        )

        trade = await _place_order(client, analysis)
        trades.append(trade)
        _save_trades(trades)

        brain.on_trade_placed(_to_brain_ns(analysis))
        acted += 1

        print(f"    → Order {trade['order_id']} ({trade['status']})")

    # Check resolutions: if a market we hold has settled, notify the brain
    open_tickers = {t["ticker"] for t in trades if t.get("status") != "closed"}
    for m in markets:
        if m.ticker not in open_tickers:
            continue
        if m.mid_price >= 0.99:
            brain.on_market_resolved(m.ticker, 1.0)
        elif m.mid_price <= 0.01:
            brain.on_market_resolved(m.ticker, 0.0)

    brain.apply_decay()
    brain.save()

    if acted == 0:
        print("  No new signals met the edge/confidence threshold.")
    else:
        print(f"  Placed {acted} order(s). Total trades: {len(trades)}")


async def main() -> None:
    mode_str = "🔵 PAPER/MOCK" if MOCK_MODE else "🚨 LIVE"
    print("=" * 60)
    print(f"  Kalshi AI Bot  —  {mode_str}")
    print(f"  Budget: ${BUDGET_USD} | Min confidence: {MIN_CONFIDENCE:.0%} | Min edge: {MIN_EDGE:.0%}")
    print(f"  Cycle: every {CYCLE_SECONDS}s  |  Trades saved to: {TRADES_FILE}")
    print("=" * 60)

    api_key = os.getenv("KALSHI_API_KEY", "")
    pem     = os.getenv("KALSHI_PRIVATE_KEY_PEM", "")

    if not api_key or not pem:
        print("\nERROR: Set KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PEM in .env")
        sys.exit(1)

    client = KalshiClient(
        api_key_id=api_key,
        private_key_content=pem,
        mock_mode=MOCK_MODE,
    )

    ensemble = EnsemblePredictor()
    brain    = LearningEngine(ensemble, state_file=Path("data/learning_state.json"))
    brain.load()

    trades = _load_trades()
    print(f"\nLoaded {len(trades)} existing trades from {TRADES_FILE}")

    try:
        while True:
            await run_cycle(client, ensemble, brain, trades)
            print(f"  Sleeping {CYCLE_SECONDS}s... (Ctrl+C to stop)")
            await asyncio.sleep(CYCLE_SECONDS)
    except KeyboardInterrupt:
        print("\n\nBot stopped by user.")
    finally:
        await client.close()
        print(f"Total trades this session: {len(trades)}")


if __name__ == "__main__":
    asyncio.run(main())
