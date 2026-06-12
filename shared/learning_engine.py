"""
LearningEngine — self-improving AI brain that updates ensemble signal weights
based on resolved market outcomes via Brier-score credit assignment.

How it works
------------
1. When the bot places a trade, `on_trade_placed()` snapshots which signals
   fired and at what values.
2. When a market resolves, `on_market_resolved()` asks: for each signal,
   did its implied direction help or hurt vs the naïve baseline (market price)?
   - Improvement > 0  → signal added value → raise its weight (reward)
   - Improvement < 0  → signal was wrong    → lower its weight (penalize)
3. Weights are bounded in [0.1, 8.0] and gently regularised back toward
   DEFAULT_WEIGHTS each cycle (prevents runaway drift on small samples).
4. Everything is persisted to `data/learning_state.json` so the brain
   survives restarts and accumulates knowledge over days/weeks.

Brier credit assignment
-----------------------
For a signal with value v ∈ [-1, +1]:
  implied_p = 0.5 + v * 0.5          # map to [0, 1] probability
  signal_brier  = (implied_p - outcome)²
  baseline_brier = (market_price - outcome)²
  improvement   = baseline_brier - signal_brier   # positive = helped

  weight_delta  = LR × improvement × signal.confidence
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

from .predictive_models import EnsemblePredictor

log = structlog.get_logger(__name__)

_DEFAULT_STATE_FILE = Path("data/learning_state.json")
_LR = 0.08          # learning rate per resolved market
_DECAY = 0.02       # fraction to pull back toward DEFAULT_WEIGHTS each cycle
_MIN_WEIGHT = 0.1
_MAX_WEIGHT = 8.0
_MEMORY_TTL_DAYS = 45


@dataclass
class SignalSnapshot:
    """Signal state captured at trade placement time."""
    name: str
    value: float        # [-1, +1]
    confidence: float
    weight_at_trade: float


@dataclass
class TradeMemory:
    """Full context of a trade, retained until the market resolves."""
    market_id: str
    direction: str              # "YES" or "NO"
    market_price_at_trade: float
    signals: list[SignalSnapshot] = field(default_factory=list)
    placed_at: float = field(default_factory=time.time)
    resolved_outcome: Optional[float] = None   # 1.0 = YES resolved, 0.0 = NO
    resolved_at: Optional[float] = None
    pnl_sign: float = 0.0       # +1 = trade was correct, -1 = wrong


class LearningEngine:
    """
    Self-improving signal weight manager.

    Quick start::

        engine = LearningEngine(analyzer.ensemble)
        engine.load()                               # restore persisted state

        # After each BUY trade:
        engine.on_trade_placed(analysis)

        # After a market resolves (outcome 1.0 = YES, 0.0 = NO):
        engine.on_market_resolved(market_id, outcome)

        # Once per trading cycle (regularise weights):
        engine.apply_decay()

        # Periodically write to disk:
        engine.save()
    """

    def __init__(
        self,
        ensemble: EnsemblePredictor,
        state_file: Path = _DEFAULT_STATE_FILE,
    ) -> None:
        self._ensemble = ensemble
        self._state_file = state_file
        self._memories: dict[str, TradeMemory] = {}
        self._resolved_count = 0
        self._cumulative_brier_improvement = 0.0
        # Rolling list of (timestamp, weight_snapshot) for dashboard sparklines
        self._weight_snapshots: list[tuple[float, dict[str, float]]] = []

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def on_trade_placed(self, analysis) -> None:
        """
        Record a snapshot of the signals that drove a BUY_YES / BUY_NO decision.
        Called by the signal router after a trade executes successfully.
        """
        if not analysis.ensemble or analysis.signal == "HOLD":
            return

        direction = "YES" if analysis.signal == "BUY_YES" else "NO"
        signals = [
            SignalSnapshot(
                name=s.name,
                value=s.value,
                confidence=s.confidence,
                weight_at_trade=s.weight if s.weight else 0.5,
            )
            for s in analysis.ensemble.signals
            if s.value != 0.0       # only directional signals can earn credit
        ]

        self._memories[analysis.market_id] = TradeMemory(
            market_id=analysis.market_id,
            direction=direction,
            market_price_at_trade=analysis.market_price,
            signals=signals,
        )
        log.debug(
            "trade_memory_saved",
            market_id=analysis.market_id,
            direction=direction,
            n_signals=len(signals),
        )

    def on_market_resolved(self, market_id: str, outcome: float) -> None:
        """
        Attribute performance credit to signals and update ensemble weights.

        Args:
            market_id:  ID of the resolved market.
            outcome:    1.0 if the YES token won, 0.0 if NO token won.
        """
        memory = self._memories.get(market_id)
        if memory is None or memory.resolved_outcome is not None:
            return

        memory.resolved_outcome = outcome
        memory.resolved_at = time.time()

        correct = (
            (memory.direction == "YES" and outcome >= 0.5) or
            (memory.direction == "NO"  and outcome < 0.5)
        )
        memory.pnl_sign = 1.0 if correct else -1.0

        total_improvement = 0.0
        for sig in memory.signals:
            implied_yes = max(0.01, min(0.99, 0.5 + sig.value * 0.5))
            sig_brier   = (implied_yes - outcome) ** 2
            base_brier  = (memory.market_price_at_trade - outcome) ** 2
            improvement = base_brier - sig_brier          # + means signal helped
            delta       = _LR * improvement * sig.confidence
            self._ensemble.update_weights(sig.name, delta)
            total_improvement += improvement

        self._resolved_count += 1
        self._cumulative_brier_improvement += total_improvement

        # Snapshot current weights for trend display
        self._weight_snapshots.append((time.time(), dict(self._ensemble._weights)))
        if len(self._weight_snapshots) > 500:
            self._weight_snapshots = self._weight_snapshots[-500:]

        log.info(
            "weights_updated_from_resolution",
            market_id=market_id,
            outcome=outcome,
            correct=correct,
            brier_improvement=round(total_improvement, 4),
            resolved_total=self._resolved_count,
        )

    def apply_decay(self) -> None:
        """
        Gentle regularisation: nudge each weight 2% back toward its default.
        Call once per trading cycle to prevent over-fitting on small samples.
        """
        defaults = EnsemblePredictor.DEFAULT_WEIGHTS
        for name, default_w in defaults.items():
            current = self._ensemble._weights.get(name, default_w)
            self._ensemble._weights[name] = max(
                _MIN_WEIGHT,
                min(_MAX_WEIGHT, current * (1 - _DECAY) + default_w * _DECAY),
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write current weights and trade memories to disk."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        cutoff = time.time() - _MEMORY_TTL_DAYS * 86400
        state = {
            "saved_at": time.time(),
            "resolved_count": self._resolved_count,
            "cumulative_brier_improvement": self._cumulative_brier_improvement,
            "weights": dict(self._ensemble._weights),
            "memories": {
                mid: {
                    "market_id": m.market_id,
                    "direction": m.direction,
                    "market_price_at_trade": m.market_price_at_trade,
                    "placed_at": m.placed_at,
                    "resolved_outcome": m.resolved_outcome,
                    "pnl_sign": m.pnl_sign,
                    "signals": [
                        {
                            "name": s.name,
                            "value": s.value,
                            "confidence": s.confidence,
                            "weight_at_trade": s.weight_at_trade,
                        }
                        for s in m.signals
                    ],
                }
                for mid, m in self._memories.items()
                if m.placed_at >= cutoff
            },
        }
        self._state_file.write_text(json.dumps(state, indent=2))
        log.info("learning_state_saved", file=str(self._state_file),
                 resolved=self._resolved_count)

    def load(self) -> bool:
        """
        Restore weights and memories from disk.

        Returns True if a state file was found and loaded, False otherwise.
        """
        if not self._state_file.exists():
            log.info("no_learning_state_found", file=str(self._state_file))
            return False
        try:
            state = json.loads(self._state_file.read_text())

            for name, w in state.get("weights", {}).items():
                if name in self._ensemble._weights:
                    self._ensemble._weights[name] = max(
                        _MIN_WEIGHT, min(_MAX_WEIGHT, float(w))
                    )

            self._resolved_count = int(state.get("resolved_count", 0))
            self._cumulative_brier_improvement = float(
                state.get("cumulative_brier_improvement", 0.0)
            )

            for mid, m_data in state.get("memories", {}).items():
                signals = [
                    SignalSnapshot(
                        name=s["name"],
                        value=s["value"],
                        confidence=s["confidence"],
                        weight_at_trade=s["weight_at_trade"],
                    )
                    for s in m_data.get("signals", [])
                ]
                self._memories[mid] = TradeMemory(
                    market_id=m_data["market_id"],
                    direction=m_data["direction"],
                    market_price_at_trade=m_data["market_price_at_trade"],
                    placed_at=m_data["placed_at"],
                    signals=signals,
                    resolved_outcome=m_data.get("resolved_outcome"),
                    pnl_sign=m_data.get("pnl_sign", 0.0),
                )

            log.info(
                "learning_state_loaded",
                resolved=self._resolved_count,
                memories=len(self._memories),
                weights=dict(self._ensemble._weights),
            )
            return True
        except Exception as exc:
            log.warning("learning_state_load_failed", error=str(exc))
            return False

    # ------------------------------------------------------------------
    # Dashboard / reporting
    # ------------------------------------------------------------------

    def performance_summary(self) -> dict:
        """Returns a dict suitable for dashboard display."""
        resolved = [
            m for m in self._memories.values()
            if m.resolved_outcome is not None
        ]
        defaults = EnsemblePredictor.DEFAULT_WEIGHTS

        if not resolved:
            return {
                "resolved_markets": 0,
                "win_rate": 0.0,
                "avg_brier_improvement": 0.0,
                "weight_drift": {n: 0.0 for n in defaults},
                "current_weights": dict(self._ensemble._weights),
            }

        wins = sum(1 for m in resolved if m.pnl_sign > 0)
        drift = {
            name: round(self._ensemble._weights.get(name, w) - w, 3)
            for name, w in defaults.items()
        }
        return {
            "resolved_markets": len(resolved),
            "pending_markets": sum(
                1 for m in self._memories.values() if m.resolved_outcome is None
            ),
            "win_rate": wins / len(resolved),
            "avg_brier_improvement": (
                self._cumulative_brier_improvement / max(1, self._resolved_count)
            ),
            "weight_drift": drift,
            "current_weights": dict(self._ensemble._weights),
        }

    def weight_trend(self, signal_name: str, last_n: int = 20) -> list[float]:
        """Returns the last N recorded weights for a named signal (for sparklines)."""
        return [
            snap[signal_name]
            for _, snap in self._weight_snapshots[-last_n:]
            if signal_name in snap
        ]
