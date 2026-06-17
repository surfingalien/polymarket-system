"""
Tests for MonteCarloPortfolio — binary prediction-market outcome simulation.

Verify the payoff math, distribution statistics, edge cases, and that the
risk metrics move in the expected direction.
"""
from __future__ import annotations

import numpy as np
import pytest

from shared.monte_carlo import (
    MonteCarloPortfolio,
    MonteCarloResult,
    MCPosition,
    position_from_signal,
)


class TestPositionFromSignal:
    def test_buy_yes_payoff(self):
        # YES at 0.25 → win pays 1/0.25 - 1 = 3x stake; win prob = true prob
        p = position_from_signal("m", "BUY_YES", market_price=0.25,
                                 true_prob=0.40, stake=10)
        assert p is not None
        assert abs(p.win_payoff_mult - 3.0) < 1e-9
        assert abs(p.win_prob - 0.40) < 1e-9

    def test_buy_no_payoff(self):
        # NO at price 0.25 means NO costs 0.75 → win pays 1/0.75 - 1 ≈ 0.333x
        # win prob = 1 - true_prob
        p = position_from_signal("m", "BUY_NO", market_price=0.25,
                                 true_prob=0.40, stake=10)
        assert p is not None
        assert abs(p.win_payoff_mult - (1 / 0.75 - 1)) < 1e-9
        assert abs(p.win_prob - 0.60) < 1e-9

    def test_rejects_degenerate_price(self):
        assert position_from_signal("m", "BUY_YES", 0.0, 0.5, 10) is None
        assert position_from_signal("m", "BUY_YES", 1.0, 0.5, 10) is None

    def test_rejects_zero_stake(self):
        assert position_from_signal("m", "BUY_YES", 0.5, 0.5, 0) is None

    def test_rejects_unknown_signal(self):
        assert position_from_signal("m", "HOLD", 0.5, 0.5, 10) is None

    def test_clamps_probability(self):
        p = position_from_signal("m", "BUY_YES", 0.5, 1.7, 10)
        assert p.win_prob == 1.0


class TestSimulationBasics:
    def test_empty_portfolio(self):
        res = MonteCarloPortfolio(n_sims=1000).simulate([], bankroll=100)
        assert res.n_positions == 0
        assert res.mean_pnl == 0.0
        assert res.prob_profit == 0.0

    def test_zero_bankroll(self):
        pos = [MCPosition("a", 10, 0.6, 1.0)]
        res = MonteCarloPortfolio(n_sims=1000).simulate(pos, bankroll=0)
        assert res.n_positions == 0

    def test_certain_win_always_profits(self):
        # win_prob = 1.0 → every path wins
        pos = [MCPosition("a", 10, 1.0, 1.0)]  # +10 every time
        res = MonteCarloPortfolio(n_sims=2000, seed=1).simulate(pos, bankroll=100)
        assert res.prob_profit == 1.0
        assert abs(res.mean_pnl - 10.0) < 1e-6
        assert res.risk_of_ruin == 0.0

    def test_certain_loss_always_ruins_when_large(self):
        # win_prob = 0 → always lose the full stake
        pos = [MCPosition("a", 60, 0.0, 1.0)]  # -60 on a 100 bankroll > 50%
        res = MonteCarloPortfolio(n_sims=1000, seed=1).simulate(
            pos, bankroll=100, ruin_fraction=0.5
        )
        assert res.prob_profit == 0.0
        assert res.risk_of_ruin == 1.0

    def test_mean_pnl_matches_expected_value(self):
        # Single position: EV = q*stake*payoff - (1-q)*stake
        q, stake, payoff = 0.6, 10.0, 1.0
        pos = [MCPosition("a", stake, q, payoff)]
        res = MonteCarloPortfolio(n_sims=200_000, seed=7).simulate(pos, bankroll=100)
        ev = q * stake * payoff - (1 - q) * stake  # = 0.2*10 = 2.0
        assert abs(res.mean_pnl - ev) < 0.1


class TestDistributionShape:
    def test_percentiles_ordered(self):
        pos = [MCPosition(f"m{i}", 10, 0.5, 1.0) for i in range(10)]
        res = MonteCarloPortfolio(n_sims=10000, seed=3).simulate(pos, bankroll=200)
        assert res.p5 <= res.median_pnl <= res.p95

    def test_fan_bands_length_matches_positions(self):
        pos = [MCPosition(f"m{i}", 5, 0.55, 1.0) for i in range(6)]
        res = MonteCarloPortfolio(n_sims=2000, seed=2).simulate(pos, bankroll=100)
        assert len(res.band_p5) == 6
        assert len(res.band_p50) == 6
        assert len(res.band_p95) == 6
        assert len(res.step_labels) == 6

    def test_histogram_populated(self):
        pos = [MCPosition(f"m{i}", 10, 0.5, 1.0) for i in range(8)]
        res = MonteCarloPortfolio(n_sims=5000, seed=4).simulate(
            pos, bankroll=200, hist_bins=20
        )
        assert len(res.hist_counts) == 20
        assert len(res.hist_centers) == 20
        assert sum(res.hist_counts) == 5000

    def test_reproducible_with_seed(self):
        pos = [MCPosition(f"m{i}", 10, 0.5, 1.0) for i in range(5)]
        r1 = MonteCarloPortfolio(n_sims=3000, seed=99).simulate(pos, bankroll=100)
        r2 = MonteCarloPortfolio(n_sims=3000, seed=99).simulate(pos, bankroll=100)
        assert r1.mean_pnl == r2.mean_pnl
        assert r1.p5 == r2.p5


class TestMarketShock:
    def test_correlation_widens_tails(self):
        # Same positions, with vs without market shock. Shock should increase
        # dispersion (wider P95-P5 spread) because outcomes co-move.
        pos = [MCPosition(f"m{i}", 10, 0.5, 1.0) for i in range(20)]
        indep = MonteCarloPortfolio(n_sims=20000, seed=5, market_shock=0.0).simulate(
            pos, bankroll=400
        )
        corr = MonteCarloPortfolio(n_sims=20000, seed=5, market_shock=0.5).simulate(
            pos, bankroll=400
        )
        spread_indep = indep.p95 - indep.p5
        spread_corr = corr.p95 - corr.p5
        assert spread_corr > spread_indep

    def test_shock_clamped(self):
        mc = MonteCarloPortfolio(market_shock=5.0)
        assert mc.market_shock <= 0.9
        mc2 = MonteCarloPortfolio(market_shock=-1.0)
        assert mc2.market_shock == 0.0
