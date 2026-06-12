"""
Advanced trading signals derived from prediction market microstructure research.

Sources:
- Jon-Becker/prediction-market-analysis: longshot bias, maker/taker gap,
  trade-size informativeness, calibration MAD, hour-of-day patterns
- MiroFish: time-based activity multipliers, dual-source sentiment divergence

All signals return a Signal object compatible with EnsemblePredictor.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np

from .predictive_models import Signal


# ---------------------------------------------------------------------------
# 1. Longshot Bias Detector
# ---------------------------------------------------------------------------
# Well-documented phenomenon: bettors systematically overpay for YES at low
# prices and overpay for NO at high prices.
# -> NO contracts outperform YES at prices 1–25¢
# -> YES contracts outperform NO at prices 75–99¢
# -> Near 50¢: no systematic bias
#
# Source: Polymarket 36GB dataset analysis (Jon-Becker), confirmed in
# academic literature (Thaler & Ziemba 1988, Snowberg & Wolfers 2010).

class LongshotBiasDetector:
    """
    Adjusts the probability estimate for the well-documented longshot bias.

    At prices below `low_threshold`:  market overestimates YES → lean NO
    At prices above `high_threshold`: market underestimates YES → lean YES
    """

    def __init__(
        self,
        low_threshold: float = 0.20,
        high_threshold: float = 0.80,
        max_adjustment: float = 0.06,   # cap at 6 percentage points
    ) -> None:
        self._low = low_threshold
        self._high = high_threshold
        self._max_adj = max_adjustment

    def signal(self, market_price: float) -> Signal:
        """
        Returns a signal representing the longshot bias correction.

        value < 0: lean NO (market overpriced YES)
        value > 0: lean YES (market underpriced YES)
        """
        if market_price < self._low:
            # Longshot: YES overpriced → lean NO
            # Stronger bias closer to 0
            strength = (self._low - market_price) / self._low
            adjustment = -min(strength * self._max_adj * 5, 1.0)
            confidence = 0.55 + strength * 0.20
        elif market_price > self._high:
            # Favourite reversal: YES underpriced → lean YES
            strength = (market_price - self._high) / (1.0 - self._high)
            adjustment = min(strength * self._max_adj * 5, 1.0)
            confidence = 0.50 + strength * 0.20
        else:
            adjustment = 0.0
            confidence = 0.0

        return Signal(
            name="longshot_bias",
            value=adjustment,
            confidence=confidence,
            weight=0.9,
        )

    def adjust_probability(self, market_price: float, estimated_prob: float) -> float:
        """Direct probability adjustment, keeping within [0.01, 0.99]."""
        sig = self.signal(market_price)
        delta = sig.value * self._max_adj
        return max(0.01, min(0.99, estimated_prob + delta))


# ---------------------------------------------------------------------------
# 2. Order Flow Signal (trade-size informativeness)
# ---------------------------------------------------------------------------
# Research finding: larger trades correlate with higher win rates.
# Slope: ~+2.3 percentage points per log10 increase in trade size.
# Informed traders place large orders; retail places small ones.

@dataclass
class TradeEvent:
    price: float        # 0–1 (YES price)
    size_usd: float
    side: str           # "YES" | "NO"
    timestamp: float = field(default_factory=time.time)


class OrderFlowAnalyzer:
    """
    Tracks recent trades and produces a signal based on large-order direction.

    Large trades (top quartile by size) are treated as informed and carry
    more weight than small trades.
    """

    # Regression slope from research: log10(size) → +2.3pp excess return
    SIZE_SLOPE_PP = 2.3

    def __init__(
        self,
        window_seconds: int = 3600,
        large_trade_threshold_usd: float = 500.0,
        max_trades: int = 200,
    ) -> None:
        self._window = window_seconds
        self._large_threshold = large_trade_threshold_usd
        self._trades: deque[TradeEvent] = deque(maxlen=max_trades)

    def record_trade(self, price: float, size_usd: float, side: str) -> None:
        self._trades.append(TradeEvent(price=price, size_usd=size_usd, side=side))

    def signal(self) -> Optional[Signal]:
        now = time.time()
        recent = [t for t in self._trades if now - t.timestamp < self._window]
        if not recent:
            return None

        # Separate large (informed) vs small (retail) trades
        large = [t for t in recent if t.size_usd >= self._large_threshold]
        if not large:
            return None

        # Weighted directional signal: YES=+1, NO=−1, weighted by log(size)
        weights = [math.log10(max(t.size_usd, 1.0)) for t in large]
        directions = [1.0 if t.side == "YES" else -1.0 for t in large]
        total_weight = sum(weights)
        if total_weight == 0:
            return None

        raw = sum(d * w for d, w in zip(directions, weights)) / total_weight
        # Confidence based on number of large trades and recency
        conf = min(0.85, 0.40 + len(large) * 0.05)

        return Signal(
            name="order_flow",
            value=max(-1.0, min(1.0, raw)),
            confidence=conf,
            weight=1.4,
        )

    def large_trade_fraction(self) -> float:
        now = time.time()
        recent = [t for t in self._trades if now - t.timestamp < self._window]
        if not recent:
            return 0.0
        large = [t for t in recent if t.size_usd >= self._large_threshold]
        return len(large) / len(recent)


# ---------------------------------------------------------------------------
# 3. Hour-of-Day Temporal Signal
# ---------------------------------------------------------------------------
# Research finding: mispricing opportunities are highest during low-activity
# hours. VWAP analysis shows retail dominance at off-peak times.
# US prediction markets peak 12:00–20:00 ET (16:00–00:00 UTC).

class TemporalPatternSignal:
    """
    Adjusts confidence based on time of day.

    During peak hours (institutional activity high) → markets more efficient
    → require stronger edge.
    During off-peak hours → markets less efficient → lower edge threshold OK.
    """

    # UTC hours of high institutional activity (US afternoon + EU morning)
    PEAK_HOURS_UTC = set(range(13, 23))   # 13:00–22:59 UTC
    DEAD_HOURS_UTC = set(range(2, 8))     # 02:00–07:59 UTC

    def __init__(self) -> None:
        pass

    def efficiency_multiplier(self, utc_hour: Optional[int] = None) -> float:
        """
        Returns 0.5–1.5 multiplier for confidence/edge thresholds.
        1.0 = normal; >1.0 = markets more efficient (need higher edge);
        <1.0 = markets less efficient (edge more exploitable).
        """
        if utc_hour is None:
            utc_hour = datetime.now(timezone.utc).hour

        if utc_hour in self.PEAK_HOURS_UTC:
            return 1.30   # more efficient — tighten thresholds
        if utc_hour in self.DEAD_HOURS_UTC:
            return 0.75   # less efficient — relax thresholds
        return 1.00

    def signal(self, utc_hour: Optional[int] = None) -> Signal:
        """
        Returns a meta-signal encoding whether the current window is
        favourable for finding mispricings.
        value=+1 means good hunting window; value=−1 means efficient.
        """
        mult = self.efficiency_multiplier(utc_hour)
        # Map multiplier to [-1, +1]: 0.75→+1, 1.30→-1
        value = -(mult - 1.0) / 0.30
        value = max(-1.0, min(1.0, value))
        return Signal(
            name="temporal_efficiency",
            value=value,
            confidence=0.40,   # weak signal alone; amplifies other signals
            weight=0.4,
        )

    def adjusted_min_edge(self, base_min_edge: float, utc_hour: Optional[int] = None) -> float:
        """Scale min_edge requirement up during efficient hours."""
        return base_min_edge * self.efficiency_multiplier(utc_hour)


# ---------------------------------------------------------------------------
# 4. Category Edge Model
# ---------------------------------------------------------------------------
# Research finding: Sports and Crypto markets have significantly larger
# maker-taker gaps than Finance/Politics markets, implying more retail
# participation and larger exploitable edges.
# Category effect size (Cohen's d): Sports > Crypto > Politics > Finance

@dataclass
class CategoryProfile:
    name: str
    min_edge_multiplier: float     # scale base min_edge
    confidence_threshold: float    # override for this category
    longshot_bias_strength: float  # how pronounced the longshot bias is
    typical_maker_taker_gap: float # percentage points


CATEGORY_PROFILES: dict[str, CategoryProfile] = {
    "sports":    CategoryProfile("sports",    0.7,  0.60, 1.5, 3.2),
    "crypto":    CategoryProfile("crypto",    0.8,  0.62, 1.3, 2.8),
    "politics":  CategoryProfile("politics",  1.0,  0.65, 1.0, 2.1),
    "finance":   CategoryProfile("finance",   1.2,  0.70, 0.8, 1.4),
    "weather":   CategoryProfile("weather",   0.9,  0.63, 1.1, 2.5),
    "science":   CategoryProfile("science",   1.0,  0.65, 1.0, 2.0),
    "default":   CategoryProfile("default",   1.0,  0.65, 1.0, 2.0),
}


class CategoryEdgeModel:
    """
    Adjusts edge requirements and confidence thresholds per market category.
    Categories with more retail participation (Sports) have lower min_edge.
    """

    def __init__(self) -> None:
        self._profiles = CATEGORY_PROFILES

    @staticmethod
    def _has_word(word: str, text: str) -> bool:
        """Word-boundary check so 'eth' doesn't match 'something'."""
        import re
        return bool(re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text))

    def detect_category(self, question: str) -> str:
        """Keyword-based category detection from market question."""
        q = question.lower()
        if any(self._has_word(w, q) for w in ["nba", "nfl", "mlb", "soccer", "football",
                                                "basketball", "tennis", "golf", "ufc",
                                                "super bowl", "world cup", "sport"]):
            return "sports"
        if any(self._has_word(w, q) for w in ["bitcoin", "btc", "eth", "crypto", "solana",
                                                "coinbase", "blockchain", "defi", "nft",
                                                "altcoin"]):
            return "crypto"
        if any(w in q for w in ["election", "president", "congress", "senate", "vote",
                                  "democrat", "republican", "parliament", "prime minister"]):
            return "politics"
        if any(w in q for w in ["fed", "rate", "gdp", "cpi", "inflation", "stock", "nasdaq",
                                  "s&p", "recession", "unemployment", "earnings", "ipo"]):
            return "finance"
        if any(w in q for w in ["temperature", "hurricane", "rain", "storm", "weather",
                                  "earthquake", "flood", "drought"]):
            return "weather"
        if any(w in q for w in ["ai ", "gpt", "claude", "gemini", "robot", "nasa", "rocket",
                                  "space", "science", "research"]):
            return "science"
        return "default"

    def get_profile(self, question: str) -> CategoryProfile:
        cat = self.detect_category(question)
        return self._profiles.get(cat, self._profiles["default"])

    def adjusted_min_edge(self, question: str, base_min_edge: float) -> float:
        profile = self.get_profile(question)
        return base_min_edge * profile.min_edge_multiplier

    def adjusted_confidence_threshold(self, question: str, base_conf: float) -> float:
        profile = self.get_profile(question)
        return profile.confidence_threshold

    def signal(self, question: str, edge: float) -> Signal:
        """
        Returns a signal encoding whether the category supports the detected edge.
        Higher edge multiplier categories → higher signal confidence.
        """
        profile = self.get_profile(question)
        # Larger maker-taker gap → more exploitable → higher confidence
        gap_factor = profile.typical_maker_taker_gap / 2.0  # normalize to ~1.0
        conf = min(0.75, 0.35 + gap_factor * 0.10)
        return Signal(
            name="category_edge",
            value=0.0,  # directionally neutral — amplifies confidence only
            confidence=conf,
            weight=0.5,
        )


# ---------------------------------------------------------------------------
# 5. Excess Return Tracker (enhanced calibration with MAD)
# ---------------------------------------------------------------------------
# Excess return = actual win_rate − implied_probability
# Mean Absolute Deviation (MAD) across time-windows measures calibration drift.
# Spikes in MAD → market mispricing → increased trading opportunity.

@dataclass
class ExcessReturnRecord:
    market_id: str
    predicted_prob: float
    outcome: float          # 1.0 YES, 0.0 NO
    timestamp: float = field(default_factory=time.time)

    @property
    def excess_return(self) -> float:
        return self.outcome - self.predicted_prob


class ExcessReturnTracker:
    """
    Tracks excess return = win_rate − implied_probability over time.
    Computes running MAD (Mean Absolute Deviation) to detect calibration drift.

    High MAD → markets currently mis-calibrated → higher edge opportunities.
    Low MAD → markets well-calibrated → need larger raw edge to profit.
    """

    def __init__(self, window: int = 100) -> None:
        self._records: deque[ExcessReturnRecord] = deque(maxlen=window)

    def record(self, market_id: str, predicted_prob: float) -> None:
        self._records.append(
            ExcessReturnRecord(
                market_id=market_id,
                predicted_prob=predicted_prob,
                outcome=-1.0,   # pending
            )
        )

    def resolve(self, market_id: str, outcome: float) -> None:
        for r in self._records:
            if r.market_id == market_id and r.outcome == -1.0:
                r.outcome = outcome
                break

    def mad(self) -> float:
        """Mean Absolute Deviation of excess returns over resolved records."""
        resolved = [r for r in self._records if r.outcome >= 0]
        if not resolved:
            return 0.25   # baseline uninformative
        return float(np.mean([abs(r.excess_return) for r in resolved]))

    def mean_excess_return(self) -> float:
        resolved = [r for r in self._records if r.outcome >= 0]
        if not resolved:
            return 0.0
        return float(np.mean([r.excess_return for r in resolved]))

    def calibration_signal(self) -> Signal:
        """
        When MAD is high (markets mis-calibrated), return a bullish signal
        for trading (more opportunity), with confidence proportional to MAD.
        """
        mad = self.mad()
        baseline_mad = 0.20   # well-calibrated prediction market
        excess_mad = mad - baseline_mad

        # High MAD = mis-calibrated = more opportunity
        if excess_mad > 0.02:
            value = min(1.0, excess_mad / 0.10)
            confidence = min(0.70, 0.40 + len([r for r in self._records if r.outcome >= 0]) * 0.005)
        else:
            value = 0.0
            confidence = 0.0

        return Signal(
            name="calibration_drift",
            value=value,
            confidence=confidence,
            weight=0.5,
        )

    def stats(self) -> dict:
        resolved = [r for r in self._records if r.outcome >= 0]
        return {
            "n_resolved": len(resolved),
            "n_pending": sum(1 for r in self._records if r.outcome < 0),
            "mad": round(self.mad(), 4),
            "mean_excess_return": round(self.mean_excess_return(), 4),
        }


# ---------------------------------------------------------------------------
# 6. Dual-Source Sentiment Divergence
# ---------------------------------------------------------------------------
# Inspired by MiroFish: Twitter (fast/emotional) vs Reddit (analytical) diverge
# during uncertain events. When two independent sentiment sources strongly
# disagree, reduce overall confidence (market is contested).

class SentimentDivergenceSignal:
    """
    Computes a confidence penalty when two sentiment sources disagree.

    High divergence = contested market = lower confidence in any signal.
    Low divergence (both agree) = boosted confidence.
    """

    def __init__(self, divergence_threshold: float = 0.3) -> None:
        self._threshold = divergence_threshold

    def signal(
        self,
        sentiment_a: float,   # source A: -1..+1
        sentiment_b: float,   # source B: -1..+1
        weight_a: float = 1.0,
        weight_b: float = 1.0,
    ) -> Signal:
        divergence = abs(sentiment_a - sentiment_b)
        total_w = weight_a + weight_b
        consensus = (sentiment_a * weight_a + sentiment_b * weight_b) / total_w

        if divergence > self._threshold:
            # Sources disagree — reduce confidence, signal is weak
            penalty = min(1.0, (divergence - self._threshold) / 0.4)
            conf = max(0.1, 0.5 - penalty * 0.4)
        else:
            # Sources agree — boost confidence
            conf = 0.5 + (1.0 - divergence / self._threshold) * 0.2

        return Signal(
            name="sentiment_divergence",
            value=consensus,
            confidence=conf,
            weight=0.6,
        )


# ---------------------------------------------------------------------------
# 7. Market Quality Scorer
# ---------------------------------------------------------------------------
# Source: last30days-skill / polymarket.py market quality formula.
# Composite score: volume (50%) + liquidity (25%) + price movement (15%)
#                 + competitive ratio (10%).
# High-quality markets (liquid, active, near 50/50) are better hunting grounds.

class MarketQualityScorer:
    """
    Scores a market 0–1 based on volume, liquidity, recent price movement,
    and how competitive (near 50/50) it is.

    Used to:
    - Weight down signals on thin/illiquid markets
    - Boost confidence on high-volume, competitive markets
    - Filter out markets not worth analyzing
    """

    # Log-scale denominators: ~$9M volume → 1.0, ~$1.2M liquidity → 1.0
    _VOL_DENOM = math.log1p(9_000_000)
    _LIQ_DENOM = math.log1p(1_200_000)

    def score(
        self,
        volume_usd: float,
        liquidity_usd: float,
        price_change_pct: float = 0.0,   # recent % price change (0–1 scale)
        market_price: float = 0.5,
    ) -> float:
        """Returns quality score 0–1."""
        vol_score = min(1.0, math.log1p(max(0, volume_usd)) / self._VOL_DENOM)
        liq_score = min(1.0, math.log1p(max(0, liquidity_usd)) / self._LIQ_DENOM)
        movement_score = min(1.0, abs(price_change_pct) * 5.0)  # 20% move → 1.0
        # Competitive: closer to 0.5 → higher score
        competitive = 1.0 - abs(market_price - 0.5) * 2.0
        competitive = max(0.0, competitive)

        return (
            0.50 * vol_score
            + 0.25 * liq_score
            + 0.15 * movement_score
            + 0.10 * competitive
        )

    def signal(
        self,
        volume_usd: float,
        liquidity_usd: float,
        price_change_pct: float = 0.0,
        market_price: float = 0.5,
    ) -> Signal:
        """Confidence-boosting meta-signal based on market quality."""
        quality = self.score(volume_usd, liquidity_usd, price_change_pct, market_price)
        # High-quality markets → higher confidence in all signals
        # Low-quality markets → reduce overall confidence
        return Signal(
            name="market_quality",
            value=0.0,           # directionally neutral
            confidence=quality,
            weight=0.8,
        )

    def is_tradeable(
        self,
        volume_usd: float,
        liquidity_usd: float,
        min_quality: float = 0.20,
        min_volume_usd: float = 5_000.0,
        min_liquidity_usd: float = 1_000.0,
    ) -> bool:
        if volume_usd < min_volume_usd or liquidity_usd < min_liquidity_usd:
            return False
        return self.score(volume_usd, liquidity_usd) >= min_quality


# ---------------------------------------------------------------------------
# 8. Two-Pass Query Expander (for news fetching)
# ---------------------------------------------------------------------------
# Source: last30days-skill / polymarket.py two-pass expansion strategy.
# Pass 1: Direct queries from market question.
# Pass 2: Extract key domain terms from pass-1 results, re-query those.

class QueryExpander:
    """
    Expands a market question into multiple targeted search queries.
    Strips noise prefixes and adds individual key terms.
    """

    _NOISE_PREFIXES = [
        "will ", "who will ", "what will ", "when will ", "how will ",
        "does ", "is ", "are ", "was ", "were ", "did ", "do ",
        "can ", "could ", "would ", "should ",
    ]

    _GENERIC_WORDS = {
        "market", "prediction", "odds", "chance", "probability", "forecast",
        "outcome", "result", "win", "lose", "happen", "occur", "event",
        "the", "a", "an", "of", "in", "to", "be", "by", "at", "on", "for",
        "or", "and", "is", "are", "was", "were", "will", "would", "could",
        "this", "that", "than", "then", "its", "it", "as",
    }

    def expand(self, question: str, max_queries: int = 5) -> list[str]:
        """Returns list of search queries ordered by expected relevance."""
        queries = []
        q = question.strip()

        # Core question (cleaned)
        core = q
        for prefix in self._NOISE_PREFIXES:
            if core.lower().startswith(prefix):
                core = core[len(prefix):]
                break
        core = core.rstrip("?").strip()
        if core:
            queries.append(core)

        # Add individual key terms (non-generic words of length > 3)
        words = [
            w.strip(".,;:!?\"'()[]") for w in q.split()
            if len(w) > 3 and w.lower() not in self._GENERIC_WORDS
        ]
        for w in words:
            if w not in queries and len(queries) < max_queries:
                queries.append(w)

        # Full question as final fallback
        if q != core and q not in queries and len(queries) < max_queries:
            queries.append(q)

        return list(dict.fromkeys(queries))[:max_queries]   # dedupe, preserve order
