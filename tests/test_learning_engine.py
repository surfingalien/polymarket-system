"""Tests for the self-learning AI brain (LearningEngine)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from shared.learning_engine import LearningEngine, SignalSnapshot, TradeMemory
from shared.predictive_models import EnsemblePredictor, EnsembleResult, Signal


def _make_ensemble() -> EnsemblePredictor:
    return EnsemblePredictor()


def _make_analysis(
    market_id: str = "m1",
    signal: str = "BUY_YES",
    market_price: float = 0.40,
    estimated_prob: float = 0.60,
):
    """Minimal FullMarketAnalysis-like object for testing."""
    sig = Signal(name="ai_analysis", value=0.5, confidence=0.8, weight=3.0)
    ensemble = EnsembleResult(
        raw_probability=market_price,
        estimated_probability=estimated_prob,
        edge=estimated_prob - market_price,
        kelly_fraction=0.05,
        confidence=0.8,
        signals=[sig],
    )
    obj = MagicMock()
    obj.market_id = market_id
    obj.signal = signal
    obj.market_price = market_price
    obj.ensemble = ensemble
    return obj


# ---------------------------------------------------------------------------
# trade_placed recording
# ---------------------------------------------------------------------------

class TestOnTradePlaced:
    def test_records_memory_for_buy_trade(self):
        engine = LearningEngine(_make_ensemble())
        engine.on_trade_placed(_make_analysis("m1", "BUY_YES"))
        assert "m1" in engine._memories

    def test_ignores_hold(self):
        engine = LearningEngine(_make_ensemble())
        engine.on_trade_placed(_make_analysis("m1", "HOLD"))
        assert "m1" not in engine._memories

    def test_direction_mapped_correctly(self):
        engine = LearningEngine(_make_ensemble())
        engine.on_trade_placed(_make_analysis("m1", "BUY_NO"))
        assert engine._memories["m1"].direction == "NO"

    def test_no_ensemble_is_ignored(self):
        engine = LearningEngine(_make_ensemble())
        obj = _make_analysis("m1", "BUY_YES")
        obj.ensemble = None
        engine.on_trade_placed(obj)
        assert "m1" not in engine._memories


# ---------------------------------------------------------------------------
# market_resolved weight updates
# ---------------------------------------------------------------------------

class TestOnMarketResolved:
    def test_correct_prediction_raises_ai_weight(self):
        ens = _make_ensemble()
        initial_w = ens._weights["ai_analysis"]
        engine = LearningEngine(ens)
        # ai_analysis value=0.5 → implies 75% YES; outcome=1.0 → correct direction
        engine.on_trade_placed(_make_analysis("m1", "BUY_YES", market_price=0.40))
        engine.on_market_resolved("m1", 1.0)
        assert ens._weights["ai_analysis"] > initial_w

    def test_wrong_prediction_lowers_ai_weight(self):
        ens = _make_ensemble()
        initial_w = ens._weights["ai_analysis"]
        engine = LearningEngine(ens)
        # Signal says YES (value=+0.5) but market resolved NO (0.0)
        engine.on_trade_placed(_make_analysis("m1", "BUY_YES", market_price=0.40))
        engine.on_market_resolved("m1", 0.0)
        assert ens._weights["ai_analysis"] < initial_w

    def test_no_memory_is_safe(self):
        engine = LearningEngine(_make_ensemble())
        # Should not raise even when market_id is unknown
        engine.on_market_resolved("unknown_market", 1.0)

    def test_double_resolution_is_ignored(self):
        ens = _make_ensemble()
        engine = LearningEngine(ens)
        engine.on_trade_placed(_make_analysis("m1", "BUY_YES"))
        engine.on_market_resolved("m1", 1.0)
        w_after_first = ens._weights["ai_analysis"]
        engine.on_market_resolved("m1", 0.0)  # second call should be no-op
        assert ens._weights["ai_analysis"] == w_after_first

    def test_resolved_count_increments(self):
        engine = LearningEngine(_make_ensemble())
        engine.on_trade_placed(_make_analysis("m1", "BUY_YES"))
        engine.on_trade_placed(_make_analysis("m2", "BUY_NO"))
        engine.on_market_resolved("m1", 1.0)
        engine.on_market_resolved("m2", 0.0)
        assert engine._resolved_count == 2

    def test_weight_snapshot_recorded(self):
        engine = LearningEngine(_make_ensemble())
        engine.on_trade_placed(_make_analysis("m1", "BUY_YES"))
        engine.on_market_resolved("m1", 1.0)
        assert len(engine._weight_snapshots) == 1


# ---------------------------------------------------------------------------
# apply_decay — regularisation
# ---------------------------------------------------------------------------

class TestApplyDecay:
    def test_decay_pulls_toward_default(self):
        ens = _make_ensemble()
        # Push ai_analysis weight to extreme
        ens._weights["ai_analysis"] = 8.0
        engine = LearningEngine(ens)
        # After many decay steps it should converge back toward 3.0
        for _ in range(100):
            engine.apply_decay()
        assert ens._weights["ai_analysis"] < 6.0

    def test_decay_does_not_go_below_min(self):
        ens = _make_ensemble()
        ens._weights["ai_analysis"] = 0.05   # below min
        engine = LearningEngine(ens)
        engine.apply_decay()
        assert ens._weights["ai_analysis"] >= 0.1


# ---------------------------------------------------------------------------
# persistence — save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_and_load_restores_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            ens1 = _make_ensemble()
            engine1 = LearningEngine(ens1, state_file=state_file)
            engine1.on_trade_placed(_make_analysis("m1", "BUY_YES"))
            engine1.on_market_resolved("m1", 1.0)
            w_after = ens1._weights["ai_analysis"]
            engine1.save()

            ens2 = _make_ensemble()
            engine2 = LearningEngine(ens2, state_file=state_file)
            loaded = engine2.load()
            assert loaded is True
            assert ens2._weights["ai_analysis"] == pytest.approx(w_after)

    def test_load_missing_file_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "nonexistent.json"
            engine = LearningEngine(_make_ensemble(), state_file=state_file)
            assert engine.load() is False

    def test_save_and_load_restores_memories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            ens = _make_ensemble()
            engine = LearningEngine(ens, state_file=state_file)
            engine.on_trade_placed(_make_analysis("m1", "BUY_YES"))
            engine.save()

            ens2 = _make_ensemble()
            engine2 = LearningEngine(ens2, state_file=state_file)
            engine2.load()
            assert "m1" in engine2._memories

    def test_load_corrupt_file_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state_file.write_text("not json {{{{")
            engine = LearningEngine(_make_ensemble(), state_file=state_file)
            assert engine.load() is False

    def test_weights_clamped_on_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state_file.write_text(json.dumps({
                "weights": {"ai_analysis": 999.0},
                "resolved_count": 0,
                "cumulative_brier_improvement": 0.0,
                "memories": {},
            }))
            ens = _make_ensemble()
            LearningEngine(ens, state_file=state_file).load()
            assert ens._weights["ai_analysis"] <= 8.0


# ---------------------------------------------------------------------------
# performance_summary
# ---------------------------------------------------------------------------

class TestPerformanceSummary:
    def test_empty_state_returns_zeros(self):
        engine = LearningEngine(_make_ensemble())
        summary = engine.performance_summary()
        assert summary["resolved_markets"] == 0
        assert summary["win_rate"] == 0.0

    def test_win_rate_correct_after_trades(self):
        engine = LearningEngine(_make_ensemble())
        for i in range(4):
            engine.on_trade_placed(_make_analysis(f"m{i}", "BUY_YES"))
        engine.on_market_resolved("m0", 1.0)   # win
        engine.on_market_resolved("m1", 1.0)   # win
        engine.on_market_resolved("m2", 0.0)   # loss
        engine.on_market_resolved("m3", 0.0)   # loss
        summary = engine.performance_summary()
        assert summary["win_rate"] == pytest.approx(0.5)
        assert summary["resolved_markets"] == 4


# ---------------------------------------------------------------------------
# weight_trend
# ---------------------------------------------------------------------------

class TestWeightTrend:
    def test_trend_empty_before_resolutions(self):
        engine = LearningEngine(_make_ensemble())
        assert engine.weight_trend("ai_analysis") == []

    def test_trend_grows_with_resolutions(self):
        engine = LearningEngine(_make_ensemble())
        for i in range(3):
            engine.on_trade_placed(_make_analysis(f"m{i}", "BUY_YES"))
            engine.on_market_resolved(f"m{i}", 1.0)
        trend = engine.weight_trend("ai_analysis")
        assert len(trend) == 3
