#!/usr/bin/env python3
# Ensure project root is on sys.path when run as a script
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Strategy backtester using historical resolved markets.

Usage:
    python scripts/run_backtest.py --source mock --cycles 50
    python scripts/run_backtest.py --source file --data data/resolved_markets.json

Metrics produced:
  - ROI, Sharpe ratio, max drawdown
  - Win rate, average edge
  - Calibration (Brier score, ECE)
  - Kelly vs flat-bet comparison
"""
import argparse
import asyncio
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from rich.console import Console
from rich.table import Table

from shared.predictive_models import (
    BayesianEstimator,
    CalibrationTracker,
    EnsemblePredictor,
    KellyCriterion,
    MomentumAnalyzer,
    PricePoint,
)

console = Console()
random.seed(42)


@dataclass
class BacktestMarket:
    market_id: str
    question: str
    implied_prob: float        # market price at time of signal
    true_prob: float           # estimated true probability (simulated AI output)
    outcome: float             # 1.0 = YES won, 0.0 = NO won
    days_to_resolution: float = 14.0
    volume_usd: float = 10000.0
    price_history: list[float] = field(default_factory=list)


@dataclass
class BacktestResult:
    initial_bankroll: float
    final_bankroll: float
    flat_final: float         # flat $10 bet comparison
    n_trades: int
    n_wins: int
    pnl_history: list[float] = field(default_factory=list)
    brier_score: float = 0.25
    ece: float = 0.0

    @property
    def roi(self) -> float:
        return (self.final_bankroll / self.initial_bankroll - 1.0) * 100

    @property
    def flat_roi(self) -> float:
        return (self.flat_final / self.initial_bankroll - 1.0) * 100

    @property
    def win_rate(self) -> float:
        return self.n_wins / self.n_trades if self.n_trades else 0.0

    @property
    def sharpe(self) -> float:
        if len(self.pnl_history) < 2:
            return 0.0
        returns = np.diff(self.pnl_history) / np.array(self.pnl_history[:-1])
        returns = returns[np.isfinite(returns)]
        if returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * math.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        if not self.pnl_history:
            return 0.0
        peak = self.pnl_history[0]
        max_dd = 0.0
        for v in self.pnl_history:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd


def generate_mock_markets(n: int = 100) -> list[BacktestMarket]:
    """Generate synthetic markets with known ground truth for testing."""
    markets = []
    for i in range(n):
        # Simulate a market where AI has +5% to +20% edge on average
        implied = random.uniform(0.1, 0.9)
        ai_skill = random.uniform(0.03, 0.12)  # AI edge
        true_prob = max(0.01, min(0.99, implied + random.gauss(ai_skill, 0.05)))
        outcome = 1.0 if random.random() < true_prob else 0.0

        # Simulate price history
        history = [implied + random.gauss(0, 0.02) for _ in range(24)]
        history = [max(0.01, min(0.99, p)) for p in history]

        markets.append(BacktestMarket(
            market_id=f"bt_{i:04d}",
            question=f"Mock market #{i}",
            implied_prob=implied,
            true_prob=true_prob,
            outcome=outcome,
            days_to_resolution=random.uniform(1, 60),
            volume_usd=random.uniform(5000, 500000),
            price_history=history,
        ))
    return markets


def load_markets_from_file(path: str) -> list[BacktestMarket]:
    data = json.loads(Path(path).read_text())
    markets = []
    for m in data:
        markets.append(BacktestMarket(
            market_id=str(m.get("id", "")),
            question=str(m.get("question", "")),
            implied_prob=float(m.get("market_price", 0.5)),
            true_prob=float(m.get("ai_estimate", m.get("market_price", 0.5))),
            outcome=float(m.get("outcome", 0)),
            days_to_resolution=float(m.get("days_to_resolution", 14)),
            volume_usd=float(m.get("volume", 10000)),
        ))
    return markets


def run_backtest(
    markets: list[BacktestMarket],
    initial_bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.04,
    min_confidence: float = 0.60,
    flat_bet: float = 10.0,
) -> BacktestResult:
    kelly = KellyCriterion(max_fraction=kelly_fraction)
    calibration = CalibrationTracker()

    bankroll = initial_bankroll
    flat_bankroll = initial_bankroll
    n_trades = 0
    n_wins = 0
    pnl_history = [bankroll]

    for m in markets:
        # Simulate confidence from edge magnitude + volume
        edge = m.true_prob - m.implied_prob
        confidence = min(0.95, 0.5 + abs(edge) * 2 + (m.volume_usd / 1_000_000) * 0.1)

        if abs(edge) < min_edge or confidence < min_confidence:
            continue

        # Kelly sizing
        fraction = kelly.compute(m.true_prob, m.implied_prob, min_edge=min_edge)
        if fraction == 0.0:
            continue

        bet_kelly = min(bankroll * fraction, bankroll * kelly_fraction)
        n_trades += 1

        if edge > 0:  # Bet YES
            if m.outcome == 1.0:
                payout = bet_kelly * (1.0 / m.implied_prob - 1.0)
                bankroll += payout
                flat_bankroll += flat_bet * (1.0 / m.implied_prob - 1.0)
                n_wins += 1
            else:
                bankroll -= bet_kelly
                flat_bankroll -= flat_bet
        else:  # Bet NO
            no_price = 1.0 - m.implied_prob
            if m.outcome == 0.0:
                payout = bet_kelly * (1.0 / no_price - 1.0)
                bankroll += payout
                flat_bankroll += flat_bet * (1.0 / no_price - 1.0)
                n_wins += 1
            else:
                bankroll -= bet_kelly
                flat_bankroll -= flat_bet

        bankroll = max(0.01, bankroll)
        flat_bankroll = max(0.01, flat_bankroll)
        pnl_history.append(bankroll)

        calibration.record(m.market_id, m.true_prob)
        calibration.resolve(m.market_id, m.outcome)

    stats = calibration.accuracy_stats()
    return BacktestResult(
        initial_bankroll=initial_bankroll,
        final_bankroll=bankroll,
        flat_final=flat_bankroll,
        n_trades=n_trades,
        n_wins=n_wins,
        pnl_history=pnl_history,
        brier_score=stats.get("brier", 0.25),
        ece=stats.get("ece", 0.0) or 0.0,
    )


def print_results(result: BacktestResult, n_markets: int) -> None:
    table = Table(title="Backtest Results", show_lines=True)
    table.add_column("Metric")
    table.add_column("Kelly Strategy", justify="right")
    table.add_column("Flat $10 Bet", justify="right")

    color_roi = "green" if result.roi > 0 else "red"
    color_flat = "green" if result.flat_roi > 0 else "red"

    rows = [
        ("Markets analyzed", str(n_markets), "—"),
        ("Trades executed", str(result.n_trades), str(result.n_trades)),
        ("Win rate", f"{result.win_rate:.1%}", f"{result.win_rate:.1%}"),
        ("Final bankroll", f"${result.final_bankroll:,.2f}", f"${result.flat_final:,.2f}"),
        ("ROI", f"[{color_roi}]{result.roi:+.1f}%[/{color_roi}]",
                f"[{color_flat}]{result.flat_roi:+.1f}%[/{color_flat}]"),
        ("Sharpe ratio", f"{result.sharpe:.2f}", "—"),
        ("Max drawdown", f"{result.max_drawdown:.1%}", "—"),
        ("Brier score", f"{result.brier_score:.4f}", "—"),
        ("ECE", f"{result.ece:.4f}", "—"),
    ]

    for row in rows:
        table.add_row(*row)

    console.print(table)

    if result.sharpe > 1.0:
        console.print("  [green]✓ Sharpe > 1.0 — strategy has strong risk-adjusted returns[/green]")
    if result.brier_score < 0.20:
        console.print("  [green]✓ Brier < 0.20 — model is well-calibrated[/green]")
    if result.max_drawdown > 0.40:
        console.print("  [yellow]⚠ High max drawdown — consider reducing Kelly fraction[/yellow]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the AI prediction strategy")
    parser.add_argument("--source", choices=["mock", "file"], default="mock")
    parser.add_argument("--data", default="data/resolved_markets.json")
    parser.add_argument("--n", type=int, default=200, help="Number of mock markets")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction (0-1)")
    parser.add_argument("--min-edge", type=float, default=0.04)
    args = parser.parse_args()

    console.print(f"[bold]Loading markets ({args.source})...[/bold]")
    if args.source == "mock":
        markets = generate_mock_markets(args.n)
    else:
        markets = load_markets_from_file(args.data)

    console.print(f"  Loaded {len(markets)} markets")
    result = run_backtest(
        markets,
        initial_bankroll=args.bankroll,
        kelly_fraction=args.kelly,
        min_edge=args.min_edge,
    )
    print_results(result, len(markets))


if __name__ == "__main__":
    main()
