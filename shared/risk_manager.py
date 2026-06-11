"""
Risk management engine: position sizing, daily loss limits, stop-loss,
portfolio exposure, and market cooldowns.

All monetary values in USD. Thread-safe via asyncio; not multi-process safe
unless backed by a shared store (Redis/DB).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import structlog

log = structlog.get_logger(__name__)


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RejectionReason(str, Enum):
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    MAX_POSITION_SIZE = "MAX_POSITION_SIZE"
    MAX_PORTFOLIO_EXPOSURE = "MAX_PORTFOLIO_EXPOSURE"
    STOP_LOSS_TRIGGERED = "STOP_LOSS_TRIGGERED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    INSUFFICIENT_EDGE = "INSUFFICIENT_EDGE"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    MARKET_COOLDOWN = "MARKET_COOLDOWN"
    MAX_POSITIONS = "MAX_POSITIONS"
    OK = "OK"


@dataclass
class Position:
    market_id: str
    platform: str
    direction: str          # YES | NO
    size_usd: float
    entry_price: float
    current_price: float = 0.0
    opened_at: float = field(default_factory=time.time)
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0

    @property
    def pnl_usd(self) -> float:
        if self.current_price == 0:
            return 0.0
        if self.direction == "YES":
            return self.size_usd * (self.current_price / self.entry_price - 1.0)
        return self.size_usd * (self.entry_price / self.current_price - 1.0)

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.direction == "YES":
            return (self.current_price - self.entry_price) / self.entry_price
        return (self.entry_price - self.current_price) / self.entry_price


@dataclass
class RiskDecision:
    approved: bool
    reason: RejectionReason
    approved_size_usd: float
    risk_level: RiskLevel
    message: str = ""


class RiskManager:
    """Central risk engine for both Polymarket and Kalshi positions."""

    def __init__(
        self,
        max_position_usd: float = 50.0,
        max_portfolio_usd: float = 500.0,
        daily_loss_limit_usd: float = 100.0,
        stop_loss_pct: float = 0.30,
        take_profit_pct: float = 0.80,
        min_edge_pct: float = 0.05,
        min_confidence: float = 0.65,
        min_liquidity_usd: float = 1000.0,
        cooldown_hours: float = 2.0,
        max_concurrent: int = 10,
        kelly_max_fraction: float = 0.25,
    ) -> None:
        self.max_position_usd = max_position_usd
        self.max_portfolio_usd = max_portfolio_usd
        self.daily_loss_limit_usd = daily_loss_limit_usd
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_edge_pct = min_edge_pct
        self.min_confidence = min_confidence
        self.min_liquidity_usd = min_liquidity_usd
        self.cooldown_hours = cooldown_hours
        self.max_concurrent = max_concurrent
        self.kelly_max_fraction = kelly_max_fraction

        self._positions: dict[str, Position] = {}  # market_id → Position
        self._daily_pnl: float = 0.0
        self._daily_reset_time: float = self._next_midnight()
        self._cooldown_markets: dict[str, float] = {}  # market_id → expiry timestamp

    # ------------------------------------------------------------------
    # Trade approval
    # ------------------------------------------------------------------

    def approve_trade(
        self,
        market_id: str,
        platform: str,
        direction: str,
        proposed_size_usd: float,
        entry_price: float,
        edge_pct: float,
        confidence: float,
        liquidity_usd: float,
    ) -> RiskDecision:
        self._maybe_reset_daily()

        # 1. Daily loss limit
        if self._daily_pnl <= -self.daily_loss_limit_usd:
            return RiskDecision(
                approved=False,
                reason=RejectionReason.DAILY_LOSS_LIMIT,
                approved_size_usd=0.0,
                risk_level=RiskLevel.CRITICAL,
                message=f"Daily loss limit hit: ${self._daily_pnl:.2f}",
            )

        # 2. Confidence gate
        if confidence < self.min_confidence:
            return RiskDecision(
                approved=False,
                reason=RejectionReason.LOW_CONFIDENCE,
                approved_size_usd=0.0,
                risk_level=RiskLevel.HIGH,
                message=f"Confidence {confidence:.2f} < threshold {self.min_confidence:.2f}",
            )

        # 3. Edge gate
        if abs(edge_pct) < self.min_edge_pct:
            return RiskDecision(
                approved=False,
                reason=RejectionReason.INSUFFICIENT_EDGE,
                approved_size_usd=0.0,
                risk_level=RiskLevel.MEDIUM,
                message=f"Edge {edge_pct:.1%} < minimum {self.min_edge_pct:.1%}",
            )

        # 4. Liquidity
        if liquidity_usd < self.min_liquidity_usd:
            return RiskDecision(
                approved=False,
                reason=RejectionReason.LOW_LIQUIDITY,
                approved_size_usd=0.0,
                risk_level=RiskLevel.MEDIUM,
                message=f"Liquidity ${liquidity_usd:.0f} < minimum ${self.min_liquidity_usd:.0f}",
            )

        # 5. Cooldown
        cooldown_expiry = self._cooldown_markets.get(market_id, 0.0)
        if time.time() < cooldown_expiry:
            remaining = (cooldown_expiry - time.time()) / 3600
            return RiskDecision(
                approved=False,
                reason=RejectionReason.MARKET_COOLDOWN,
                approved_size_usd=0.0,
                risk_level=RiskLevel.LOW,
                message=f"Market on cooldown for {remaining:.1f}h more",
            )

        # 6. Concurrent positions cap
        if len(self._positions) >= self.max_concurrent:
            return RiskDecision(
                approved=False,
                reason=RejectionReason.MAX_POSITIONS,
                approved_size_usd=0.0,
                risk_level=RiskLevel.MEDIUM,
                message=f"Max concurrent positions ({self.max_concurrent}) reached",
            )

        # 7. Position size cap
        approved_size = min(proposed_size_usd, self.max_position_usd)

        # 8. Portfolio exposure cap
        current_exposure = sum(p.size_usd for p in self._positions.values())
        remaining_capacity = self.max_portfolio_usd - current_exposure
        if remaining_capacity <= 0:
            return RiskDecision(
                approved=False,
                reason=RejectionReason.MAX_PORTFOLIO_EXPOSURE,
                approved_size_usd=0.0,
                risk_level=RiskLevel.HIGH,
                message=f"Portfolio at max exposure ${current_exposure:.0f}",
            )
        approved_size = min(approved_size, remaining_capacity)

        if approved_size < 1.0:
            return RiskDecision(
                approved=False,
                reason=RejectionReason.MAX_PORTFOLIO_EXPOSURE,
                approved_size_usd=0.0,
                risk_level=RiskLevel.HIGH,
                message="Remaining capacity < $1.00",
            )

        risk_level = self._classify_risk(approved_size, edge_pct, confidence)

        log.info(
            "trade_approved",
            market_id=market_id,
            platform=platform,
            direction=direction,
            size=round(approved_size, 2),
            edge=round(edge_pct, 3),
            risk_level=risk_level,
        )
        return RiskDecision(
            approved=True,
            reason=RejectionReason.OK,
            approved_size_usd=approved_size,
            risk_level=risk_level,
        )

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------

    def open_position(
        self,
        market_id: str,
        platform: str,
        direction: str,
        size_usd: float,
        entry_price: float,
    ) -> Position:
        pos = Position(
            market_id=market_id,
            platform=platform,
            direction=direction,
            size_usd=size_usd,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss_price=entry_price * (1.0 - self.stop_loss_pct)
            if direction == "YES"
            else entry_price * (1.0 + self.stop_loss_pct),
            take_profit_price=entry_price * (1.0 + self.take_profit_pct)
            if direction == "YES"
            else entry_price * (1.0 - self.take_profit_pct),
        )
        self._positions[market_id] = pos
        log.info(
            "position_opened",
            market_id=market_id,
            direction=direction,
            size=size_usd,
            stop_loss=round(pos.stop_loss_price, 3),
            take_profit=round(pos.take_profit_price, 3),
        )
        return pos

    def update_position_price(self, market_id: str, current_price: float) -> Optional[str]:
        """
        Updates position price and returns "STOP_LOSS" | "TAKE_PROFIT" | None.
        """
        pos = self._positions.get(market_id)
        if not pos:
            return None
        pos.current_price = current_price

        if pos.direction == "YES":
            if current_price <= pos.stop_loss_price:
                return "STOP_LOSS"
            if current_price >= pos.take_profit_price:
                return "TAKE_PROFIT"
        else:  # NO
            if current_price >= pos.stop_loss_price:
                return "STOP_LOSS"
            if current_price <= pos.take_profit_price:
                return "TAKE_PROFIT"
        return None

    def close_position(self, market_id: str, exit_price: float) -> Optional[float]:
        """Returns realized PnL in USD."""
        pos = self._positions.pop(market_id, None)
        if not pos:
            return None
        pos.current_price = exit_price
        pnl = pos.pnl_usd
        self._daily_pnl += pnl

        if pnl < 0:
            self._set_cooldown(market_id)

        log.info(
            "position_closed",
            market_id=market_id,
            direction=pos.direction,
            pnl=round(pnl, 2),
            pct=round(pos.pnl_pct * 100, 1),
        )
        return pnl

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def scan_stop_losses(self, price_map: dict[str, float]) -> list[str]:
        """Returns list of market_ids that should be closed immediately."""
        triggers = []
        for market_id, price in price_map.items():
            action = self.update_position_price(market_id, price)
            if action in ("STOP_LOSS", "TAKE_PROFIT"):
                log.warning(
                    "exit_triggered",
                    market_id=market_id,
                    action=action,
                    price=price,
                )
                triggers.append(market_id)
        return triggers

    def portfolio_summary(self) -> dict:
        self._maybe_reset_daily()
        total_exposure = sum(p.size_usd for p in self._positions.values())
        unrealized_pnl = sum(p.pnl_usd for p in self._positions.values())
        return {
            "open_positions": len(self._positions),
            "total_exposure_usd": round(total_exposure, 2),
            "daily_pnl_usd": round(self._daily_pnl, 2),
            "unrealized_pnl_usd": round(unrealized_pnl, 2),
            "daily_loss_remaining_usd": round(
                self.daily_loss_limit_usd + self._daily_pnl, 2
            ),
            "capacity_remaining_usd": round(
                self.max_portfolio_usd - total_exposure, 2
            ),
            "trading_halted": self._daily_pnl <= -self.daily_loss_limit_usd,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _classify_risk(self, size: float, edge: float, confidence: float) -> RiskLevel:
        score = 0
        if size > self.max_position_usd * 0.8:
            score += 2
        if abs(edge) < 0.08:
            score += 1
        if confidence < 0.75:
            score += 1
        if self._daily_pnl < -self.daily_loss_limit_usd * 0.5:
            score += 2
        if score >= 4:
            return RiskLevel.HIGH
        if score >= 2:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _set_cooldown(self, market_id: str) -> None:
        self._cooldown_markets[market_id] = time.time() + self.cooldown_hours * 3600

    def _maybe_reset_daily(self) -> None:
        if time.time() >= self._daily_reset_time:
            log.info("daily_pnl_reset", previous_pnl=self._daily_pnl)
            self._daily_pnl = 0.0
            self._daily_reset_time = self._next_midnight()

    @staticmethod
    def _next_midnight() -> float:
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        tomorrow = (now + datetime.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return tomorrow.timestamp()
