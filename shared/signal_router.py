"""
Routes trading signals to the correct platform bot with size calculation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Coroutine, Optional

import structlog

from .learning_engine import LearningEngine
from .market_analyzer import FullMarketAnalysis
from .risk_manager import RiskDecision, RiskManager

log = structlog.get_logger(__name__)

TradeExecutor = Callable[
    [str, str, float, float], Coroutine  # (market_id, direction, size_usd, price)
]


@dataclass
class RoutedSignal:
    analysis: FullMarketAnalysis
    risk_decision: RiskDecision
    executed: bool = False
    execution_error: str = ""


class SignalRouter:
    """Validates signals through risk manager and dispatches to executors."""

    def __init__(
        self,
        risk_manager: RiskManager,
        poly_executor: Optional[TradeExecutor] = None,
        kalshi_executor: Optional[TradeExecutor] = None,
        bankroll_usd: float = 1000.0,
        learning_engine: Optional[LearningEngine] = None,
    ) -> None:
        self._risk = risk_manager
        self._executors: dict[str, TradeExecutor] = {}
        if poly_executor:
            self._executors["polymarket"] = poly_executor
        if kalshi_executor:
            self._executors["kalshi"] = kalshi_executor
        self._bankroll = bankroll_usd
        self._learning = learning_engine

    async def route(self, analysis: FullMarketAnalysis) -> RoutedSignal:
        if analysis.signal == "HOLD":
            return RoutedSignal(
                analysis=analysis,
                risk_decision=self._risk.approve_trade(
                    market_id=analysis.market_id,
                    platform=analysis.platform,
                    direction="HOLD",
                    proposed_size_usd=0.0,
                    entry_price=analysis.market_price,
                    edge_pct=analysis.edge,
                    confidence=analysis.final_confidence,
                    liquidity_usd=analysis.liquidity_usd,
                ),
            )

        direction = "YES" if analysis.signal == "BUY_YES" else "NO"
        entry_price = analysis.market_price

        # Kelly-based position sizing
        kelly_size = self._bankroll * analysis.kelly_fraction
        proposed_size = max(1.0, kelly_size)

        risk_decision = self._risk.approve_trade(
            market_id=analysis.market_id,
            platform=analysis.platform,
            direction=direction,
            proposed_size_usd=proposed_size,
            entry_price=entry_price,
            edge_pct=analysis.edge,
            confidence=analysis.final_confidence,
            liquidity_usd=analysis.liquidity_usd,
        )

        routed = RoutedSignal(analysis=analysis, risk_decision=risk_decision)

        if not risk_decision.approved:
            log.info(
                "trade_rejected",
                market_id=analysis.market_id,
                reason=risk_decision.reason,
                message=risk_decision.message,
            )
            return routed

        executor = self._executors.get(analysis.platform.lower())
        if not executor:
            log.warning("no_executor", platform=analysis.platform)
            return routed

        try:
            await executor(
                analysis.market_id,
                direction,
                risk_decision.approved_size_usd,
                entry_price,
            )
            self._risk.open_position(
                market_id=analysis.market_id,
                platform=analysis.platform,
                direction=direction,
                size_usd=risk_decision.approved_size_usd,
                entry_price=entry_price,
            )
            routed.executed = True
            # Notify learning engine so signals can earn credit when this market resolves
            if self._learning is not None:
                self._learning.on_trade_placed(analysis)
            log.info(
                "trade_executed",
                market_id=analysis.market_id,
                direction=direction,
                size=round(risk_decision.approved_size_usd, 2),
                price=entry_price,
            )
        except Exception as exc:
            routed.execution_error = str(exc)
            log.error("trade_execution_failed", market_id=analysis.market_id, error=str(exc))

        return routed

    async def route_all(
        self, analyses: list[FullMarketAnalysis]
    ) -> list[RoutedSignal]:
        results = []
        for analysis in analyses:
            signal = await self.route(analysis)
            results.append(signal)
        return results

    def update_bankroll(self, bankroll_usd: float) -> None:
        self._bankroll = bankroll_usd
