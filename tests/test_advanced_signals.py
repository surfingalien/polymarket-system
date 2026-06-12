"""Tests for all advanced signal modules."""
from __future__ import annotations

import math
import time
import pytest

from shared.advanced_signals import (
    CategoryEdgeModel,
    ExcessReturnTracker,
    LongshotBiasDetector,
    MarketQualityScorer,
    OrderFlowAnalyzer,
    QueryExpander,
    SentimentDivergenceSignal,
    TemporalPatternSignal,
)


# ---------------------------------------------------------------------------
# LongshotBiasDetector
# ---------------------------------------------------------------------------

class TestLongshotBiasDetector:
    def test_longshot_leans_no_at_low_price(self):
        det = LongshotBiasDetector(low_threshold=0.20)
        sig = det.signal(market_price=0.05)
        assert sig.value < 0          # lean NO
        assert sig.confidence > 0.5

    def test_favourite_leans_yes_at_high_price(self):
        det = LongshotBiasDetector(high_threshold=0.80)
        sig = det.signal(market_price=0.95)
        assert sig.value > 0          # lean YES
        assert sig.confidence > 0.5

    def test_neutral_at_midpoint(self):
        det = LongshotBiasDetector()
        sig = det.signal(market_price=0.50)
        assert sig.value == 0.0
        assert sig.confidence == 0.0

    def test_adjust_probability_pulls_down_at_low_price(self):
        det = LongshotBiasDetector()
        adjusted = det.adjust_probability(market_price=0.10, estimated_prob=0.12)
        assert adjusted < 0.12        # pulled down toward NO

    def test_adjust_probability_bounded(self):
        det = LongshotBiasDetector()
        adjusted = det.adjust_probability(market_price=0.01, estimated_prob=0.01)
        assert 0.01 <= adjusted <= 0.99


# ---------------------------------------------------------------------------
# OrderFlowAnalyzer
# ---------------------------------------------------------------------------

class TestOrderFlowAnalyzer:
    def test_no_trades_returns_none(self):
        ofa = OrderFlowAnalyzer()
        assert ofa.signal() is None

    def test_large_yes_trades_produce_positive_signal(self):
        ofa = OrderFlowAnalyzer(large_trade_threshold_usd=100.0)
        for _ in range(5):
            ofa.record_trade(price=0.6, size_usd=500.0, side="YES")
        sig = ofa.signal()
        assert sig is not None
        assert sig.value > 0

    def test_large_no_trades_produce_negative_signal(self):
        ofa = OrderFlowAnalyzer(large_trade_threshold_usd=100.0)
        for _ in range(5):
            ofa.record_trade(price=0.4, size_usd=500.0, side="NO")
        sig = ofa.signal()
        assert sig is not None
        assert sig.value < 0

    def test_small_trades_below_threshold_ignored(self):
        ofa = OrderFlowAnalyzer(large_trade_threshold_usd=1000.0)
        for _ in range(10):
            ofa.record_trade(price=0.5, size_usd=50.0, side="YES")
        assert ofa.signal() is None   # all trades below threshold

    def test_large_trade_fraction(self):
        ofa = OrderFlowAnalyzer(large_trade_threshold_usd=100.0)
        ofa.record_trade(0.5, 500.0, "YES")  # large
        ofa.record_trade(0.5, 10.0, "YES")   # small
        frac = ofa.large_trade_fraction()
        assert abs(frac - 0.5) < 0.01


# ---------------------------------------------------------------------------
# TemporalPatternSignal
# ---------------------------------------------------------------------------

class TestTemporalPatternSignal:
    def test_peak_hours_high_multiplier(self):
        temporal = TemporalPatternSignal()
        mult = temporal.efficiency_multiplier(utc_hour=16)  # peak
        assert mult > 1.0

    def test_dead_hours_low_multiplier(self):
        temporal = TemporalPatternSignal()
        mult = temporal.efficiency_multiplier(utc_hour=4)   # dead
        assert mult < 1.0

    def test_normal_hours_near_one(self):
        temporal = TemporalPatternSignal()
        mult = temporal.efficiency_multiplier(utc_hour=10)
        assert abs(mult - 1.0) < 0.05

    def test_min_edge_higher_during_peak(self):
        temporal = TemporalPatternSignal()
        base = 0.05
        peak_edge = temporal.adjusted_min_edge(base, utc_hour=16)
        dead_edge = temporal.adjusted_min_edge(base, utc_hour=4)
        assert peak_edge > dead_edge

    def test_signal_returns_signal_object(self):
        temporal = TemporalPatternSignal()
        sig = temporal.signal(utc_hour=16)
        assert sig.name == "temporal_efficiency"
        assert -1.0 <= sig.value <= 1.0


# ---------------------------------------------------------------------------
# CategoryEdgeModel
# ---------------------------------------------------------------------------

class TestCategoryEdgeModel:
    def test_detects_sports(self):
        model = CategoryEdgeModel()
        assert model.detect_category("Will the NBA Finals go to 7 games?") == "sports"

    def test_detects_crypto(self):
        model = CategoryEdgeModel()
        assert model.detect_category("Will BTC exceed $120k by end of 2026?") == "crypto"

    def test_detects_politics(self):
        model = CategoryEdgeModel()
        assert model.detect_category("Who will win the 2026 Senate election?") == "politics"

    def test_detects_finance(self):
        model = CategoryEdgeModel()
        assert model.detect_category("Will the Fed cut rates in Q3 2026?") == "finance"

    def test_sports_has_lower_min_edge_than_finance(self):
        model = CategoryEdgeModel()
        base = 0.05
        sports_edge = model.adjusted_min_edge("NBA Finals game 7?", base)
        finance_edge = model.adjusted_min_edge("Fed rate decision?", base)
        assert sports_edge < finance_edge

    def test_finance_has_higher_confidence_threshold(self):
        model = CategoryEdgeModel()
        sports_conf = model.adjusted_confidence_threshold("NBA Finals?", 0.65)
        finance_conf = model.adjusted_confidence_threshold("Fed rate cut?", 0.65)
        assert finance_conf >= sports_conf

    def test_unknown_returns_default(self):
        model = CategoryEdgeModel()
        assert model.detect_category("Something completely unrelated") == "default"


# ---------------------------------------------------------------------------
# ExcessReturnTracker
# ---------------------------------------------------------------------------

class TestExcessReturnTracker:
    def test_mad_baseline_when_no_records(self):
        tracker = ExcessReturnTracker()
        assert tracker.mad() == pytest.approx(0.25)

    def test_zero_mad_on_perfect_calibration(self):
        tracker = ExcessReturnTracker()
        for i in range(20):
            tracker.record(f"m{i}", 0.7)
            tracker.resolve(f"m{i}", 1.0)
        # perfect: predicted 0.7, outcome 1.0 → excess = 0.3 each
        assert abs(tracker.mad() - 0.30) < 0.01

    def test_calibration_signal_high_when_mad_elevated(self):
        tracker = ExcessReturnTracker()
        for i in range(20):
            tracker.record(f"m{i}", 0.5)
            tracker.resolve(f"m{i}", 1.0)  # systematically wrong → high MAD
        sig = tracker.calibration_signal()
        assert sig.confidence > 0  # some signal detected

    def test_stats_structure(self):
        tracker = ExcessReturnTracker()
        stats = tracker.stats()
        assert "n_resolved" in stats
        assert "mad" in stats
        assert "mean_excess_return" in stats


# ---------------------------------------------------------------------------
# MarketQualityScorer
# ---------------------------------------------------------------------------

class TestMarketQualityScorer:
    def test_zero_volume_and_liquidity_low_score(self):
        scorer = MarketQualityScorer()
        score = scorer.score(0, 0, 0.0, 0.5)
        assert score < 0.15

    def test_high_volume_liquidity_high_score(self):
        scorer = MarketQualityScorer()
        score = scorer.score(5_000_000, 800_000, 0.10, 0.5)
        assert score > 0.70

    def test_competitive_market_higher_than_extreme(self):
        scorer = MarketQualityScorer()
        comp = scorer.score(100_000, 50_000, 0.0, 0.50)   # near 50/50
        extreme = scorer.score(100_000, 50_000, 0.0, 0.95)  # near 100%
        assert comp > extreme

    def test_is_tradeable_passes_with_decent_quality(self):
        scorer = MarketQualityScorer()
        assert scorer.is_tradeable(500_000, 100_000) is True

    def test_is_tradeable_fails_thin_market(self):
        scorer = MarketQualityScorer()
        assert scorer.is_tradeable(100, 50) is False

    def test_signal_returns_signal_object(self):
        scorer = MarketQualityScorer()
        sig = scorer.signal(1_000_000, 200_000, 0.05, 0.5)
        assert sig.name == "market_quality"
        assert 0.0 <= sig.confidence <= 1.0


# ---------------------------------------------------------------------------
# QueryExpander
# ---------------------------------------------------------------------------

class TestQueryExpander:
    def test_returns_list(self):
        expander = QueryExpander()
        queries = expander.expand("Will BTC exceed $120k by end of 2026?")
        assert isinstance(queries, list)
        assert len(queries) >= 1

    def test_strips_will_prefix(self):
        expander = QueryExpander()
        queries = expander.expand("Will the Fed cut rates in Q3?")
        assert queries[0] == "the Fed cut rates in Q3"

    def test_respects_max_queries(self):
        expander = QueryExpander()
        queries = expander.expand("A very long question with many words and topics", max_queries=3)
        assert len(queries) <= 3

    def test_deduplicates_results(self):
        expander = QueryExpander()
        queries = expander.expand("Bitcoin Bitcoin Bitcoin")
        assert len(queries) == len(set(queries))

    def test_short_generic_words_excluded(self):
        expander = QueryExpander()
        queries = expander.expand("Will this market prediction odds resolve yes?")
        # "this", "odds", "market", "prediction" are generic — should not dominate
        all_text = " ".join(queries)
        assert "BTC" not in all_text  # unrelated word not injected


# ---------------------------------------------------------------------------
# SentimentDivergenceSignal
# ---------------------------------------------------------------------------

class TestSentimentDivergenceSignal:
    def test_low_divergence_high_confidence(self):
        sig_obj = SentimentDivergenceSignal(divergence_threshold=0.3)
        sig = sig_obj.signal(0.6, 0.65)  # nearly identical
        assert sig.confidence > 0.55

    def test_high_divergence_low_confidence(self):
        sig_obj = SentimentDivergenceSignal(divergence_threshold=0.3)
        sig = sig_obj.signal(0.9, -0.9)  # total disagreement
        assert sig.confidence < 0.25

    def test_consensus_is_weighted_average(self):
        sig_obj = SentimentDivergenceSignal()
        sig = sig_obj.signal(1.0, 0.0, weight_a=1.0, weight_b=1.0)
        assert abs(sig.value - 0.5) < 0.01

    def test_signal_value_bounded(self):
        sig_obj = SentimentDivergenceSignal()
        sig = sig_obj.signal(-1.0, 1.0)
        assert -1.0 <= sig.value <= 1.0
