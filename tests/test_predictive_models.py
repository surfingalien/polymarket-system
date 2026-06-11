"""Tests for all predictive model components."""
from __future__ import annotations

import math
import pytest

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
    SentimentConverter,
    Signal,
)


# --- BayesianEstimator ---

class TestBayesianEstimator:
    def test_no_evidence_returns_near_market_price(self):
        est = BayesianEstimator()
        mean, lo, hi = est.estimate(0.6, [])
        assert abs(mean - 0.6) < 0.1  # stays close to market
        assert lo < mean < hi

    def test_strong_yes_evidence_raises_probability(self):
        est = BayesianEstimator()
        mean_base, _, _ = est.estimate(0.5, [])
        mean_yes, _, _ = est.estimate(0.5, [(0.9, 1.0), (0.8, 0.9)])
        assert mean_yes > mean_base + 0.05

    def test_strong_no_evidence_lowers_probability(self):
        est = BayesianEstimator()
        mean, _, _ = est.estimate(0.5, [(-0.9, 1.0), (-0.8, 0.9)])
        assert mean < 0.45

    def test_returns_valid_credible_interval(self):
        est = BayesianEstimator()
        mean, lo, hi = est.estimate(0.3, [(0.5, 0.8)])
        assert 0.0 <= lo < mean < hi <= 1.0


# --- KellyCriterion ---

class TestKellyCriterion:
    def test_positive_edge_returns_nonzero_fraction(self):
        k = KellyCriterion(max_fraction=0.25)
        f = k.compute(estimated_prob=0.70, market_price=0.50)
        assert f > 0.0
        assert f <= 0.25

    def test_zero_edge_returns_zero(self):
        k = KellyCriterion()
        f = k.compute(estimated_prob=0.50, market_price=0.50)
        assert f == 0.0

    def test_negative_edge_returns_positive_no_fraction(self):
        # Negative edge → bet NO; compute() still returns a positive Kelly fraction
        k = KellyCriterion()
        f = k.compute(estimated_prob=0.40, market_price=0.55)
        assert f > 0.0   # caller uses sign of edge to determine YES/NO direction

    def test_small_edge_below_min_returns_zero(self):
        k = KellyCriterion()
        f = k.compute(estimated_prob=0.52, market_price=0.50, min_edge=0.05)
        assert f == 0.0

    def test_caps_at_max_fraction(self):
        k = KellyCriterion(max_fraction=0.10)
        f = k.compute(estimated_prob=0.95, market_price=0.10)
        assert f <= 0.10

    def test_position_size_respects_max(self):
        k = KellyCriterion()
        size = k.position_size_usd(1000.0, 0.70, 0.50, max_position_usd=25.0)
        assert size <= 25.0


# --- OrderBookAnalyzer ---

class TestOrderBookAnalyzer:
    def _make_ob(self, bid_vol=100, ask_vol=50):
        return OrderBookSnapshot(
            bids=[(0.50, bid_vol), (0.49, 80), (0.48, 60)],
            asks=[(0.51, ask_vol), (0.52, 60), (0.53, 40)],
        )

    def test_imbalance_positive_when_bids_dominate(self):
        ob = self._make_ob(bid_vol=200, ask_vol=50)
        analyzer = OrderBookAnalyzer()
        signals = analyzer.analyze(ob)
        obi = next((s for s in signals if s.name == "order_book_imbalance"), None)
        assert obi is not None
        assert obi.value > 0

    def test_imbalance_negative_when_asks_dominate(self):
        ob = self._make_ob(bid_vol=20, ask_vol=200)
        analyzer = OrderBookAnalyzer()
        signals = analyzer.analyze(ob)
        obi = next((s for s in signals if s.name == "order_book_imbalance"), None)
        assert obi is not None
        assert obi.value < 0

    def test_empty_book_returns_no_signals(self):
        ob = OrderBookSnapshot(bids=[], asks=[])
        signals = OrderBookAnalyzer().analyze(ob)
        assert signals == []

    def test_spread_quality_signal_exists(self):
        ob = self._make_ob()
        signals = OrderBookAnalyzer().analyze(ob)
        names = {s.name for s in signals}
        assert "spread_quality" in names


# --- MomentumAnalyzer ---

class TestMomentumAnalyzer:
    def _history(self, n: int, trend: float = 0.005):
        return [PricePoint(price=0.5 + i * trend, volume=1000.0, timestamp=float(i))
                for i in range(n)]

    def test_empty_history_returns_no_signals(self):
        signals = MomentumAnalyzer().analyze([])
        assert signals == []

    def test_uptrend_produces_positive_velocity(self):
        history = self._history(30, trend=0.005)
        signals = MomentumAnalyzer().analyze(history)
        velocity = next((s for s in signals if s.name == "price_velocity"), None)
        assert velocity is not None
        assert velocity.value > 0

    def test_downtrend_produces_negative_velocity(self):
        history = self._history(30, trend=-0.005)
        signals = MomentumAnalyzer().analyze(history)
        velocity = next((s for s in signals if s.name == "price_velocity"), None)
        assert velocity is not None
        assert velocity.value < 0

    def test_produces_ema_signal_for_long_history(self):
        history = self._history(30)
        signals = MomentumAnalyzer(long_window=24).analyze(history)
        names = {s.name for s in signals}
        assert "ema_momentum" in names


# --- EnsemblePredictor ---

class TestEnsemblePredictor:
    def test_returns_ensemble_result(self):
        pred = EnsemblePredictor()
        result = pred.predict(
            market_price=0.5,
            ai_probability=0.65,
            ai_confidence=0.80,
            bayesian_estimate=0.62,
            microstructure_signals=[],
            momentum_signals=[],
            sentiment_signal=None,
        )
        assert 0.0 <= result.estimated_probability <= 1.0
        assert result.estimated_probability > 0.5  # AI says higher
        assert result.edge > 0

    def test_no_signals_returns_market_price(self):
        pred = EnsemblePredictor()
        result = pred.predict(
            market_price=0.4,
            ai_probability=None,
            ai_confidence=0.0,
            bayesian_estimate=None,
            microstructure_signals=[],
            momentum_signals=[],
            sentiment_signal=None,
        )
        assert abs(result.estimated_probability - 0.4) < 0.05

    def test_has_edge_when_large_deviation(self):
        pred = EnsemblePredictor()
        result = pred.predict(
            market_price=0.5,
            ai_probability=0.75,
            ai_confidence=0.9,
            bayesian_estimate=0.72,
            microstructure_signals=[],
            momentum_signals=[],
            sentiment_signal=None,
        )
        assert result.has_edge
        assert result.direction == "YES"


# --- CalibrationTracker ---

class TestCalibrationTracker:
    def test_brier_score_perfect_prediction(self):
        tracker = CalibrationTracker()
        for i in range(20):
            tracker.record(f"m{i}", 0.9)
            tracker.resolve(f"m{i}", 1.0)
        score = tracker.brier_score()
        assert score < 0.02  # 0.9 predicting 1.0 → (0.9-1)^2 = 0.01

    def test_brier_score_worst_prediction(self):
        tracker = CalibrationTracker()
        for i in range(10):
            tracker.record(f"m{i}", 0.0)
            tracker.resolve(f"m{i}", 1.0)
        score = tracker.brier_score()
        assert score == pytest.approx(1.0)

    def test_accuracy_stats_structure(self):
        tracker = CalibrationTracker()
        stats = tracker.accuracy_stats()
        assert "n" in stats
        assert "brier" in stats


# --- ResolutionDecayModel ---

class TestResolutionDecayModel:
    def test_zero_days_returns_zero_confidence(self):
        model = ResolutionDecayModel()
        conf, _ = model.adjust_for_time(0, 0.8, 0.7)
        assert conf == 0.0

    def test_far_future_preserves_confidence(self):
        model = ResolutionDecayModel()
        conf, prob = model.adjust_for_time(60, 0.8, 0.7)
        assert conf > 0.7
        assert abs(prob - 0.7) < 0.01

    def test_near_resolution_reduces_confidence(self):
        model = ResolutionDecayModel()
        conf_far, _ = model.adjust_for_time(30, 0.8, 0.7)
        conf_near, _ = model.adjust_for_time(1, 0.8, 0.7)
        assert conf_near < conf_far


# --- CrossMarketCorrelator ---

class TestCrossMarketCorrelator:
    def test_finds_matching_markets(self):
        correlator = CrossMarketCorrelator()
        poly = [{"condition_id": "p1", "question": "federal reserve rate cut 2026", "best_ask": 0.40}]
        kalshi = [{"ticker": "k1", "title": "Federal Reserve rate cut Q3 2026", "yes_ask": 0.55}]
        arb = correlator.find_arbitrage(poly, kalshi, min_spread=0.0)
        assert len(arb) > 0

    def test_no_arb_when_prices_match(self):
        correlator = CrossMarketCorrelator()
        poly = [{"condition_id": "p1", "question": "bitcoin above 100k", "best_ask": 0.50}]
        kalshi = [{"ticker": "k1", "title": "bitcoin price above 100k", "yes_ask": 0.50}]
        # Same price, net_spread after fees will be negative
        arb = correlator.find_arbitrage(poly, kalshi, min_spread=0.05)
        assert len(arb) == 0
