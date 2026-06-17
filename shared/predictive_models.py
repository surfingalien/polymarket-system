"""
Enhanced predictive algorithms for prediction market trading.

Implements: Bayesian probability estimation, Kelly Criterion sizing,
order-book microstructure analysis, momentum signals, cross-market
correlation, ensemble scoring, and calibration tracking.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import stats
from scipy.special import betaln


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OrderBookSnapshot:
    bids: list[tuple[float, float]]  # [(price, size), …] descending
    asks: list[tuple[float, float]]  # [(price, size), …] ascending
    timestamp: float = field(default_factory=time.time)

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 1.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid


@dataclass
class PricePoint:
    price: float
    volume: float
    timestamp: float


@dataclass
class Signal:
    name: str
    value: float        # -1 bearish … +1 bullish
    confidence: float   # 0–1
    weight: float = 1.0

    @property
    def weighted_value(self) -> float:
        return self.value * self.confidence * self.weight


@dataclass
class EnsembleResult:
    raw_probability: float          # market implied
    estimated_probability: float    # our estimate
    edge: float                     # estimated_prob - raw_prob
    kelly_fraction: float           # optimal bet size fraction
    confidence: float               # overall confidence 0–1
    signals: list[Signal] = field(default_factory=list)
    reasoning: str = ""

    @property
    def has_edge(self) -> bool:
        return abs(self.edge) >= 0.03

    @property
    def direction(self) -> str:
        return "YES" if self.edge > 0 else "NO"


@dataclass
class CalibrationRecord:
    predicted_prob: float
    actual_outcome: Optional[float]  # 1.0 YES, 0.0 NO, None pending
    market_id: str
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# 1. Bayesian Probability Estimator
# ---------------------------------------------------------------------------

class BayesianEstimator:
    """
    Beta-Binomial Bayesian estimator for binary market probabilities.

    Prior: Beta(alpha, beta) — represents prior belief about YES probability.
    Update: each evidence item shifts alpha/beta based on strength.
    Posterior mean: alpha / (alpha + beta)
    """

    def __init__(self, prior_alpha: float = 2.0, prior_beta: float = 2.0) -> None:
        # Weakly informative prior: slightly pulls toward 0.5
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta

    def estimate(
        self,
        market_price: float,
        evidence_items: list[tuple[float, float]],  # (yes_strength, weight) each in [-1, 1]
    ) -> tuple[float, float, float]:
        """
        Returns (posterior_mean, lower_95ci, upper_95ci).

        market_price initializes the prior mass so the market's wisdom is
        respected; evidence then pushes the posterior away from it.
        """
        # Convert market price to pseudo-counts
        market_strength = 20.0  # how much weight to give market price
        alpha = self._prior_alpha + market_price * market_strength
        beta = self._prior_beta + (1.0 - market_price) * market_strength

        # Incorporate evidence
        for strength, weight in evidence_items:
            # strength > 0 → evidence for YES
            effective = max(-1.0, min(1.0, strength))
            obs_weight = abs(weight) * 5.0  # scale evidence into pseudo-counts
            if effective > 0:
                alpha += effective * obs_weight
            else:
                beta += abs(effective) * obs_weight

        # Posterior mean and credible interval
        posterior_mean = alpha / (alpha + beta)
        lower, upper = stats.beta.ppf([0.025, 0.975], alpha, beta)

        return float(posterior_mean), float(lower), float(upper)

    def log_bayes_factor(
        self, p_hypothesis: float, p_alternative: float, evidence_strength: float
    ) -> float:
        """Log Bayes factor for a single piece of evidence."""
        if p_hypothesis <= 0 or p_alternative <= 0:
            return 0.0
        return math.log(p_hypothesis / p_alternative) * evidence_strength


# ---------------------------------------------------------------------------
# 2. Kelly Criterion Position Sizer
# ---------------------------------------------------------------------------

class KellyCriterion:
    """
    Computes optimal bet fraction using the Kelly formula with safety caps.

    Full Kelly: f* = (bp - q) / b  where b = net odds, p = win prob, q = 1-p
    For prediction markets (binary, price = implied prob):
      b = (1 - price) / price  (profit per unit staked on YES)
    Fractional Kelly applied to cap volatility.
    """

    def __init__(self, max_fraction: float = 0.25) -> None:
        self._max_fraction = max_fraction

    def compute(
        self,
        estimated_prob: float,
        market_price: float,
        min_edge: float = 0.02,
        fractional: float = 0.25,
    ) -> float:
        """
        Returns recommended fraction of bankroll to wager (0–max_fraction).
        Negative edge → returns 0 (no bet).
        """
        estimated_prob = max(0.01, min(0.99, estimated_prob))
        market_price = max(0.01, min(0.99, market_price))

        edge = estimated_prob - market_price
        if abs(edge) < min_edge:
            return 0.0

        if edge > 0:
            # Betting YES
            b = (1.0 - market_price) / market_price
            p = estimated_prob
        else:
            # Betting NO (flip perspective)
            b = market_price / (1.0 - market_price)
            p = 1.0 - estimated_prob

        q = 1.0 - p
        if b <= 0:
            return 0.0

        full_kelly = (b * p - q) / b
        full_kelly = max(0.0, full_kelly)

        # Apply fractional Kelly and cap
        fraction = full_kelly * fractional
        return min(fraction, self._max_fraction)

    def position_size_usd(
        self,
        bankroll: float,
        estimated_prob: float,
        market_price: float,
        max_position_usd: float,
    ) -> float:
        fraction = self.compute(estimated_prob, market_price)
        raw = bankroll * fraction
        return min(raw, max_position_usd)


# ---------------------------------------------------------------------------
# 3. Order Book Microstructure Analyzer
# ---------------------------------------------------------------------------

class OrderBookAnalyzer:
    """
    Extracts microstructure signals from limit-order-book snapshots.

    Signals:
    - Order book imbalance (OBI): buying vs selling pressure
    - Effective spread: cost of a round-trip trade
    - Depth asymmetry: imbalance at multiple price levels
    - Price pressure: estimated short-term price direction
    """

    def __init__(self, depth_levels: int = 5) -> None:
        self._depth = depth_levels

    def analyze(self, ob: OrderBookSnapshot) -> list[Signal]:
        signals = []

        if not ob.bids or not ob.asks:
            return signals

        # Order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
        bid_vol = sum(s for _, s in ob.bids[: self._depth])
        ask_vol = sum(s for _, s in ob.asks[: self._depth])
        total_vol = bid_vol + ask_vol
        if total_vol > 0:
            obi = (bid_vol - ask_vol) / total_vol
            signals.append(Signal(
                name="order_book_imbalance",
                value=obi,
                confidence=min(1.0, total_vol / 1000.0),
            ))

        # Spread signal: tighter spread = more confidence in mid-price
        spread_pct = ob.spread / ob.mid_price if ob.mid_price > 0 else 1.0
        spread_conf = max(0.0, 1.0 - spread_pct * 10)
        # Positive spread_signal: market price is trustworthy
        signals.append(Signal(
            name="spread_quality",
            value=0.0,       # neutral directional signal
            confidence=spread_conf,
        ))

        # Depth asymmetry at 2nd and 3rd levels
        if len(ob.bids) >= 3 and len(ob.asks) >= 3:
            deep_bid = sum(s for _, s in ob.bids[1:3])
            deep_ask = sum(s for _, s in ob.asks[1:3])
            deep_total = deep_bid + deep_ask
            if deep_total > 0:
                depth_asym = (deep_bid - deep_ask) / deep_total
                signals.append(Signal(
                    name="depth_asymmetry",
                    value=depth_asym,
                    confidence=0.6,
                    weight=0.5,
                ))

        return signals

    def estimate_price_impact(self, ob: OrderBookSnapshot, size_usd: float) -> float:
        """Estimate slippage for a YES buy of size_usd dollars."""
        if not ob.asks:
            return 1.0
        remaining = size_usd
        total_cost = 0.0
        shares = 0.0
        for price, volume in ob.asks:
            fill = min(remaining, volume * price)
            shares_filled = fill / price
            total_cost += fill
            shares += shares_filled
            remaining -= fill
            if remaining <= 0:
                break
        if shares == 0:
            return ob.best_ask
        avg_price = total_cost / shares
        return avg_price - ob.best_ask  # slippage above best ask


# ---------------------------------------------------------------------------
# 4. Momentum & Mean-Reversion Analyzer
# ---------------------------------------------------------------------------

class MomentumAnalyzer:
    """
    Computes momentum, RSI, and VWAP deviation signals from price history.
    """

    def __init__(self, short_window: int = 6, long_window: int = 24) -> None:
        self._short = short_window
        self._long = long_window

    def analyze(self, history: Sequence[PricePoint]) -> list[Signal]:
        if len(history) < 3:
            return []

        prices = np.array([p.price for p in history])
        volumes = np.array([p.volume for p in history])
        signals = []

        # --- Price momentum: short EMA / long EMA ratio ---
        if len(prices) >= self._long:
            short_ema = self._ema(prices, self._short)
            long_ema = self._ema(prices, self._long)
            ratio = short_ema / long_ema - 1.0 if long_ema > 0 else 0.0
            signals.append(Signal(
                name="ema_momentum",
                value=max(-1.0, min(1.0, ratio * 10)),
                confidence=0.7,
            ))

        # --- RSI-style overbought/oversold ---
        if len(prices) >= 7:
            rsi = self._rsi(prices, 7)
            # RSI > 70 → price overbought → lean toward mean-reversion (negative for YES)
            # RSI < 30 → oversold → positive for YES
            rsi_signal = -((rsi - 50) / 50)  # maps [0,100] to [-1,+1] inverted
            signals.append(Signal(
                name="rsi_mean_reversion",
                value=float(rsi_signal),
                confidence=0.55,
                weight=0.6,
            ))

        # --- VWAP deviation ---
        if volumes.sum() > 0:
            vwap = float(np.average(prices, weights=volumes))
            current = float(prices[-1])
            deviation = (current - vwap) / vwap if vwap > 0 else 0.0
            # Price above VWAP suggests upward pressure
            signals.append(Signal(
                name="vwap_deviation",
                value=max(-1.0, min(1.0, deviation * 5)),
                confidence=0.5,
                weight=0.4,
            ))

        # --- Velocity (recent rate of change) ---
        if len(prices) >= 4:
            velocity = (prices[-1] - prices[-4]) / prices[-4] if prices[-4] > 0 else 0.0
            signals.append(Signal(
                name="price_velocity",
                value=max(-1.0, min(1.0, velocity * 20)),
                confidence=0.65,
            ))

        # --- Mean reversion: deviation from the historical average ---
        # Distinct from RSI: compares the latest price to the mean of the
        # *prior* history. A price stretched far above its mean is expected to
        # revert down (bearish for YES), and vice-versa.
        if len(prices) >= 5:
            hist = prices[:-1]
            avg_price = float(hist.mean())
            if avg_price > 0:
                dev_pct = (float(prices[-1]) - avg_price) / avg_price
                if abs(dev_pct) > 0.10:
                    signals.append(Signal(
                        name="price_mean_reversion",
                        # price above mean → negative (fade up-move), below → positive
                        value=max(-1.0, min(1.0, -dev_pct * 3.0)),
                        confidence=0.5,
                        weight=0.6,
                    ))

        # --- Volume-price divergence: a price move on declining volume lacks
        # conviction and is more likely to reverse. We fade the unconfirmed move.
        if len(prices) >= 5 and volumes.sum() > 0:
            window = min(5, len(prices))
            recent_prices = prices[-window:]
            recent_volumes = volumes[-window:]
            price_trend = float(recent_prices[-1] - recent_prices[0])
            avg_vol = float(recent_volumes.mean())
            if avg_vol > 0 and abs(price_trend) > 0.02:
                vol_ratio = float(recent_volumes[-1]) / avg_vol
                if vol_ratio < 0.7:
                    strength = min((1.0 - vol_ratio) * 2.0, 1.0)
                    direction = -1.0 if price_trend > 0 else 1.0  # fade the move
                    signals.append(Signal(
                        name="volume_divergence",
                        value=max(-1.0, min(1.0, direction * strength)),
                        confidence=0.45,
                        weight=0.4,
                    ))

        return signals

    @staticmethod
    def _ema(prices: np.ndarray, window: int) -> float:
        k = 2.0 / (window + 1)
        ema = float(prices[0])
        for p in prices[1:]:
            ema = p * k + ema * (1 - k)
        return ema

    @staticmethod
    def _rsi(prices: np.ndarray, period: int) -> float:
        deltas = np.diff(prices[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1 + rs))


# ---------------------------------------------------------------------------
# 5. Sentiment-to-Probability Converter
# ---------------------------------------------------------------------------

class SentimentConverter:
    """
    Converts news sentiment scores into probability adjustments.
    Uses a sigmoid transformation so extreme sentiment produces bounded shifts.
    """

    def __init__(self, sensitivity: float = 0.15) -> None:
        self._sensitivity = sensitivity

    def adjust_probability(
        self,
        base_prob: float,
        sentiment: float,  # -1 … +1
        confidence: float = 0.5,
    ) -> float:
        shift = self._sensitivity * sentiment * confidence
        adjusted = base_prob + shift
        return max(0.01, min(0.99, adjusted))

    def to_signal(self, sentiment: float, num_articles: int) -> Signal:
        conf = min(1.0, num_articles / 5.0) * 0.7
        return Signal(
            name="news_sentiment",
            value=sentiment,
            confidence=conf,
            weight=0.8,
        )


# ---------------------------------------------------------------------------
# 6. Cross-Market Correlation Detector
# ---------------------------------------------------------------------------

@dataclass
class ArbitrageOpportunity:
    poly_market_id: str
    kalshi_market_id: str
    poly_yes_price: float
    kalshi_yes_price: float
    spread: float
    net_spread: float   # after fees
    direction: str      # "BUY_POLY_YES" | "BUY_KALSHI_YES"
    confidence: float


class CrossMarketCorrelator:
    """
    Detects arbitrage opportunities between Polymarket and Kalshi
    by matching semantically equivalent markets and comparing prices.
    """

    POLY_FEE = 0.02   # 2% CLOB taker fee
    KALSHI_FEE = 0.07  # ~7% round-trip

    def find_arbitrage(
        self,
        poly_markets: list[dict],
        kalshi_markets: list[dict],
        min_spread: float = 0.03,
    ) -> list[ArbitrageOpportunity]:
        opportunities = []
        for pm in poly_markets:
            match = self._find_match(pm, kalshi_markets)
            if not match:
                continue
            opp = self._compute_opportunity(pm, match, min_spread)
            if opp:
                opportunities.append(opp)
        return sorted(opportunities, key=lambda o: o.net_spread, reverse=True)

    def _find_match(self, poly: dict, kalshi_markets: list[dict]) -> Optional[dict]:
        poly_title = poly.get("question", "").lower()
        best_score = 0.0
        best_match = None
        for km in kalshi_markets:
            kalshi_title = km.get("title", "").lower()
            score = self._similarity(poly_title, kalshi_title)
            if score > best_score:
                best_score = score
                best_match = km
        if best_score >= 0.45:
            return best_match
        return None

    def _compute_opportunity(
        self, poly: dict, kalshi: dict, min_spread: float
    ) -> Optional[ArbitrageOpportunity]:
        poly_yes = float(poly.get("best_ask", poly.get("yes_price", 0.5)))
        kalshi_yes = float(kalshi.get("yes_ask", kalshi.get("yes_price", 0.5)))

        spread = abs(poly_yes - kalshi_yes)
        net_spread = spread - self.POLY_FEE - self.KALSHI_FEE

        if net_spread < min_spread:
            return None

        direction = "BUY_POLY_YES" if poly_yes < kalshi_yes else "BUY_KALSHI_YES"
        confidence = min(1.0, net_spread / 0.15)

        return ArbitrageOpportunity(
            poly_market_id=str(poly.get("condition_id", poly.get("id", ""))),
            kalshi_market_id=str(kalshi.get("ticker", kalshi.get("id", ""))),
            poly_yes_price=poly_yes,
            kalshi_yes_price=kalshi_yes,
            spread=spread,
            net_spread=net_spread,
            direction=direction,
            confidence=confidence,
        )

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Jaccard similarity on word sets with stopword removal."""
        stops = {"will", "the", "a", "an", "of", "in", "to", "be", "by", "at",
                 "on", "for", "or", "and", "is", "are", "was", "were"}
        words_a = {w for w in a.split() if w not in stops and len(w) > 2}
        words_b = {w for w in b.split() if w not in stops and len(w) > 2}
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union


# ---------------------------------------------------------------------------
# 7. Ensemble Predictor
# ---------------------------------------------------------------------------

class EnsemblePredictor:
    """
    Combines AI, Bayesian, momentum, microstructure, and sentiment signals
    into a single probability estimate with Kelly-sized position recommendation.
    """

    DEFAULT_WEIGHTS = {
        "ai_analysis": 3.0,
        # Bayesian posterior is largely derived from the same AI evidence,
        # so keep its weight below ai_analysis to limit double-counting.
        "bayesian": 1.5,
        "order_book_imbalance": 1.5,
        "depth_asymmetry": 0.5,
        "spread_quality": 0.2,
        "ema_momentum": 1.0,
        "rsi_mean_reversion": 0.6,
        "vwap_deviation": 0.4,
        "price_velocity": 0.8,
        "price_mean_reversion": 0.6,
        "volume_divergence": 0.4,
        "news_sentiment": 1.2,
    }

    def __init__(self, kelly: Optional[KellyCriterion] = None) -> None:
        self._kelly = kelly or KellyCriterion()
        self._weights = dict(self.DEFAULT_WEIGHTS)

    def predict(
        self,
        market_price: float,
        ai_probability: Optional[float],
        ai_confidence: float,
        bayesian_estimate: Optional[float],
        microstructure_signals: list[Signal],
        momentum_signals: list[Signal],
        sentiment_signal: Optional[Signal],
    ) -> EnsembleResult:
        all_signals: list[Signal] = []

        # AI signal — highest weight
        if ai_probability is not None:
            ai_edge = ai_probability - market_price
            ai_value = max(-1.0, min(1.0, ai_edge * 5))
            all_signals.append(Signal(
                name="ai_analysis",
                value=ai_value,
                confidence=ai_confidence,
                weight=self._weights["ai_analysis"],
            ))

        # Bayesian signal
        if bayesian_estimate is not None:
            bay_edge = bayesian_estimate - market_price
            bay_value = max(-1.0, min(1.0, bay_edge * 5))
            all_signals.append(Signal(
                name="bayesian",
                value=bay_value,
                confidence=0.75,
                weight=self._weights["bayesian"],
            ))

        # Microstructure + momentum signals
        for sig in microstructure_signals + momentum_signals:
            sig.weight = self._weights.get(sig.name, 0.5)
            all_signals.append(sig)

        # Sentiment
        if sentiment_signal is not None:
            sentiment_signal.weight = self._weights.get("news_sentiment", 1.2)
            all_signals.append(sentiment_signal)

        # Weighted probability estimate.
        # Only directional signals (value != 0) move the estimate; neutral
        # meta-signals (value == 0, e.g. spread_quality) modulate confidence
        # but must not dilute the weighted mean, otherwise adding more
        # meta-signals arbitrarily shrinks every real edge toward zero.
        directional = [s for s in all_signals if s.value != 0.0]
        dir_weight = sum(s.weight * s.confidence for s in directional)
        if dir_weight == 0:
            estimated = market_price
            confidence = 0.3
        else:
            weighted_sum = sum(s.weighted_value for s in directional)
            mean_value = weighted_sum / dir_weight
            # Convert signal [-1,+1] back to probability shift from market_price
            raw_shift = mean_value / 5.0
            estimated = max(0.01, min(0.99, market_price + raw_shift))

            # Confidence: weighted average of all signal confidences,
            # discounted when directional signals disagree with each other.
            total_w = max(sum(s.weight for s in all_signals), 1e-9)  # guard /0
            avg_conf = sum(s.weight * s.confidence for s in all_signals) / total_w
            dispersion = sum(
                s.weight * s.confidence * abs(s.value - mean_value)
                for s in directional
            ) / dir_weight
            agreement = max(0.0, 1.0 - dispersion / 2.0)   # dispersion ∈ [0, 2]
            confidence = min(1.0, avg_conf * (0.7 + 0.3 * agreement))

        edge = estimated - market_price
        kelly_f = self._kelly.compute(estimated, market_price)

        return EnsembleResult(
            raw_probability=market_price,
            estimated_probability=estimated,
            edge=edge,
            kelly_fraction=kelly_f,
            confidence=confidence,
            signals=all_signals,
        )

    def update_weights(self, signal_name: str, performance_delta: float) -> None:
        """Adjust signal weight based on recent performance (online learning)."""
        if signal_name in self._weights:
            lr = 0.05
            # Bounds MUST match LearningEngine's [_MIN_WEIGHT, _MAX_WEIGHT]
            # (0.1, 8.0); otherwise a weight learned/persisted above 5.0 gets
            # silently re-clamped here on the next online update.
            self._weights[signal_name] = max(
                0.1,
                min(8.0, self._weights[signal_name] + lr * performance_delta),
            )


# ---------------------------------------------------------------------------
# 8. Calibration Tracker
# ---------------------------------------------------------------------------

class CalibrationTracker:
    """
    Tracks predicted vs actual outcomes to measure model accuracy.
    Computes Brier score and Expected Calibration Error (ECE).
    """

    def __init__(self, max_records: int = 1000) -> None:
        self._records: deque[CalibrationRecord] = deque(maxlen=max_records)

    def record(self, market_id: str, predicted_prob: float) -> None:
        # The analyzer re-predicts the same market every cycle; update the
        # pending record in place so one market contributes one data point.
        for rec in self._records:
            if rec.market_id == market_id and rec.actual_outcome is None:
                rec.predicted_prob = predicted_prob
                rec.timestamp = time.time()
                return
        self._records.append(
            CalibrationRecord(
                market_id=market_id,
                predicted_prob=predicted_prob,
                actual_outcome=None,
            )
        )

    def resolve(self, market_id: str, outcome: float) -> None:
        for rec in self._records:
            if rec.market_id == market_id and rec.actual_outcome is None:
                rec.actual_outcome = outcome
                break

    def brier_score(self) -> float:
        resolved = [r for r in self._records if r.actual_outcome is not None]
        if not resolved:
            return 0.25  # baseline for uninformative model
        return float(
            np.mean([(r.predicted_prob - r.actual_outcome) ** 2 for r in resolved])
        )

    def ece(self, n_bins: int = 10) -> float:
        """Expected Calibration Error across probability buckets."""
        resolved = [r for r in self._records if r.actual_outcome is not None]
        if len(resolved) < 10:
            return float("nan")

        bins = np.linspace(0, 1, n_bins + 1)
        ece_val = 0.0
        n = len(resolved)

        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            bucket = [r for r in resolved if lo <= r.predicted_prob < hi]
            if not bucket:
                continue
            avg_conf = np.mean([r.predicted_prob for r in bucket])
            avg_acc = np.mean([r.actual_outcome for r in bucket])
            ece_val += (len(bucket) / n) * abs(avg_conf - avg_acc)

        return float(ece_val)

    def accuracy_stats(self) -> dict:
        resolved = [r for r in self._records if r.actual_outcome is not None]
        if not resolved:
            return {"n": 0, "brier": 0.25, "ece": float("nan")}
        correct = sum(
            1 for r in resolved
            if (r.predicted_prob >= 0.5 and r.actual_outcome == 1.0)
            or (r.predicted_prob < 0.5 and r.actual_outcome == 0.0)
        )
        return {
            "n": len(resolved),
            "accuracy": correct / len(resolved),
            "brier": self.brier_score(),
            "ece": self.ece(),
            "avg_confidence": float(np.mean([r.predicted_prob for r in resolved])),
        }


# ---------------------------------------------------------------------------
# 9. Time-to-Resolution Decay Model
# ---------------------------------------------------------------------------

class ResolutionDecayModel:
    """
    Models how market dynamics change as resolution approaches.

    Key insight: markets near resolution have:
    - Higher autocorrelation (prices drift toward truth)
    - Lower mean-reversion opportunity
    - Tighter bid-ask spreads
    - Asymmetric liquidity (winners hard to buy cheap)

    Returns a confidence multiplier and signal adjustments.
    """

    def adjust_for_time(
        self,
        days_to_resolution: float,
        base_confidence: float,
        estimated_prob: float,
        market_price: Optional[float] = None,
    ) -> tuple[float, float]:
        """
        Returns (adjusted_confidence, adjusted_probability).

        Near resolution: if market is already extreme (>0.85 or <0.15),
        reduce edge hunting since the signal is likely priced in.
        """
        if days_to_resolution <= 0:
            return 0.0, estimated_prob

        # Confidence is higher when there's time for the edge to materialize
        time_factor = min(1.0, math.log1p(days_to_resolution) / math.log1p(30))

        # Near-resolution discount: extreme probs are well-discovered
        extremity = abs(estimated_prob - 0.5) * 2  # 0 at 0.5, 1 at extremes
        if days_to_resolution < 3:
            discovery_discount = 0.3 + 0.7 * (1.0 - extremity * 0.8)
        elif days_to_resolution < 14:
            discovery_discount = 0.7 + 0.3 * (1.0 - extremity * 0.4)
        else:
            discovery_discount = 1.0

        adjusted_conf = base_confidence * time_factor * discovery_discount

        # Shrink estimate toward the market price when very close to resolution
        # (the market has discovered most information by then). Shrinking toward
        # anything other than market price would manufacture edge out of thin air.
        if days_to_resolution < 2 and market_price is not None:
            pull = 0.3
            adjusted_prob = estimated_prob * (1 - pull) + market_price * pull
        else:
            adjusted_prob = estimated_prob

        return float(adjusted_conf), float(adjusted_prob)
