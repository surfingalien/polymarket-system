"""
Monte Carlo portfolio simulation for binary prediction markets.

A prediction-market portfolio is a basket of independent Bernoulli bets:
each position either wins (the market resolves your way) or loses its full
stake. A single simulated outcome — one coin-flip per position — tells you
almost nothing about risk. Monte Carlo runs the whole basket thousands of
times and reports the *distribution* of outcomes:

    - P&L percentiles (P5 / P50 / P95) and a fan chart
    - probability of finishing profitable
    - risk of ruin (chance of losing more than a set fraction of bankroll)
    - expected ROI vs. dispersion (exposes over-aggressive Kelly sizing)

Binary payoff model (prices in dollars, 0–1):
    BUY_YES at market price p:
        cost/share = p; if resolves YES each share pays $1
        profit-if-win = stake * (1/p - 1);  win prob = true P(YES)
    BUY_NO at price p (NO costs 1-p):
        profit-if-win = stake * (1/(1-p) - 1);  win prob = 1 - true P(YES)

Outcomes are assumed independent. Real markets have some correlation
(a macro shock moves many at once); an optional market-wide shock factor
models a crude common component without pretending to a full covariance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.special import ndtr  # standard-normal CDF (vectorised)


@dataclass
class MCPosition:
    """One binary bet in the simulated portfolio."""
    label: str
    stake: float            # dollars at risk
    win_prob: float         # true probability this position wins (0–1)
    win_payoff_mult: float  # profit multiple on stake if it wins (e.g. 1/p - 1)


@dataclass
class MonteCarloResult:
    n_sims: int
    bankroll: float
    n_positions: int
    total_staked: float
    mean_pnl: float
    median_pnl: float
    p5: float
    p95: float
    std_pnl: float
    prob_profit: float
    risk_of_ruin: float
    ruin_fraction: float
    expected_roi: float
    # cumulative-P&L percentile bands per position step (for a fan chart)
    step_labels: list[str] = field(default_factory=list)
    band_p5: list[float] = field(default_factory=list)
    band_p50: list[float] = field(default_factory=list)
    band_p95: list[float] = field(default_factory=list)
    # binned final-P&L histogram (for a distribution chart)
    hist_centers: list[float] = field(default_factory=list)
    hist_counts: list[int] = field(default_factory=list)


def position_from_signal(
    label: str,
    signal: str,          # "BUY_YES" | "BUY_NO"
    market_price: float,  # YES price (0–1)
    true_prob: float,     # AI/ensemble estimate of P(YES)
    stake: float,
) -> Optional[MCPosition]:
    """Build an MCPosition from a directional signal. Returns None if unbettable."""
    p = float(market_price)
    if not (0.0 < p < 1.0) or stake <= 0:
        return None
    q = float(true_prob)
    # Clamp away from 0/1 so a stray "certain" estimate can't fabricate a
    # risk-free bet (risk_of_ruin=0, infinite-looking EV) in the simulation.
    q = min(0.999, max(0.001, q))
    if signal == "BUY_YES":
        return MCPosition(label, stake, win_prob=q, win_payoff_mult=(1.0 / p - 1.0))
    if signal == "BUY_NO":
        no_price = 1.0 - p
        if no_price <= 0:
            return None
        return MCPosition(label, stake, win_prob=1.0 - q,
                          win_payoff_mult=(1.0 / no_price - 1.0))
    return None


class MonteCarloPortfolio:
    """
    Vectorised Monte Carlo over a basket of binary positions.

    n_sims: number of simulated portfolio outcomes.
    seed: fix for reproducible results (dashboard uses a fixed seed so the
        chart is stable between reruns).
    market_shock: 0.0 = fully independent. >0 introduces a shared latent
        factor so a fraction of positions tend to win/lose together,
        widening the tails realistically. Kept simple and clearly bounded.
    """

    def __init__(
        self,
        n_sims: int = 10_000,
        seed: Optional[int] = 42,
        market_shock: float = 0.0,
    ) -> None:
        self.n_sims = int(n_sims)
        self.market_shock = max(0.0, min(0.9, float(market_shock)))
        self._rng = np.random.default_rng(seed)

    def simulate(
        self,
        positions: list[MCPosition],
        bankroll: float,
        ruin_fraction: float = 0.5,
        hist_bins: int = 30,
    ) -> MonteCarloResult:
        n = len(positions)
        if n == 0 or bankroll <= 0:
            return MonteCarloResult(
                n_sims=self.n_sims, bankroll=float(bankroll), n_positions=0,
                total_staked=0.0, mean_pnl=0.0, median_pnl=0.0, p5=0.0, p95=0.0,
                std_pnl=0.0, prob_profit=0.0, risk_of_ruin=0.0,
                ruin_fraction=ruin_fraction, expected_roi=0.0,
            )

        stakes = np.array([p.stake for p in positions], dtype=float)
        probs = np.array([p.win_prob for p in positions], dtype=float)
        payoffs = np.array([p.win_payoff_mult for p in positions], dtype=float)

        # Per-position uniform draws, shape (n_sims, n_positions).
        if self.market_shock > 0:
            # Gaussian copula: correlate outcomes via a shared latent factor
            # WITHOUT distorting each position's marginal win probability.
            #   z_i = sqrt(1-rho)*eps_i + sqrt(rho)*Z_common   (each ~ N(0,1))
            #   u_i = Phi(z_i)  is exactly Uniform(0,1), so P(u_i < p) == p.
            # rho>0 makes positions tend to win/lose together → wider tails,
            # but mean P&L and per-position win rates are unchanged.
            rho = self.market_shock
            eps = self._rng.standard_normal((self.n_sims, n))
            common = self._rng.standard_normal((self.n_sims, 1))
            z = np.sqrt(1.0 - rho) * eps + np.sqrt(rho) * common
            u = ndtr(z)
        else:
            u = self._rng.random((self.n_sims, n))

        wins = u < probs  # broadcast over columns
        # P&L per position: win → stake*payoff, loss → -stake
        pnl = np.where(wins, stakes * payoffs, -stakes)  # (n_sims, n)

        cum = np.cumsum(pnl, axis=1)         # cumulative P&L as bets are added
        final = cum[:, -1]

        # Fan-chart percentile bands at each step
        p5_band, p50_band, p95_band = np.percentile(cum, [5, 50, 95], axis=0)

        # Final-P&L histogram
        counts, edges = np.histogram(final, bins=hist_bins)
        centers = (edges[:-1] + edges[1:]) / 2.0

        ruin_threshold = -ruin_fraction * bankroll

        return MonteCarloResult(
            n_sims=self.n_sims,
            bankroll=float(bankroll),
            n_positions=n,
            total_staked=float(stakes.sum()),
            mean_pnl=float(final.mean()),
            median_pnl=float(np.median(final)),
            p5=float(np.percentile(final, 5)),
            p95=float(np.percentile(final, 95)),
            std_pnl=float(final.std()),
            prob_profit=float((final > 0).mean()),
            risk_of_ruin=float((final <= ruin_threshold).mean()),
            ruin_fraction=float(ruin_fraction),
            expected_roi=float(final.mean() / bankroll),
            step_labels=[p.label for p in positions],
            band_p5=[float(x) for x in p5_band],
            band_p50=[float(x) for x in p50_band],
            band_p95=[float(x) for x in p95_band],
            hist_centers=[float(x) for x in centers],
            hist_counts=[int(x) for x in counts],
        )
