"""Tests for RiskManager: position sizing, stop-loss, daily limits."""
from __future__ import annotations

import time
import pytest

from shared.risk_manager import (
    RejectionReason,
    RiskDecision,
    RiskLevel,
    RiskManager,
)


@pytest.fixture
def risk():
    return RiskManager(
        max_position_usd=100.0,
        max_portfolio_usd=500.0,
        daily_loss_limit_usd=200.0,
        stop_loss_pct=0.30,
        take_profit_pct=0.80,
        min_edge_pct=0.05,
        min_confidence=0.65,
        min_liquidity_usd=1000.0,
        cooldown_hours=0.001,  # tiny for tests
        max_concurrent=5,
    )


def _approve(risk, **kwargs) -> RiskDecision:
    defaults = dict(
        market_id="test_market",
        platform="poly",
        direction="YES",
        proposed_size_usd=50.0,
        entry_price=0.5,
        edge_pct=0.10,
        confidence=0.75,
        liquidity_usd=10000.0,
    )
    defaults.update(kwargs)
    return risk.approve_trade(**defaults)


# --- Approval ---

def test_approve_valid_trade(risk):
    d = _approve(risk)
    assert d.approved
    assert d.reason == RejectionReason.OK
    assert d.approved_size_usd == 50.0


def test_caps_to_max_position(risk):
    d = _approve(risk, proposed_size_usd=999.0)
    assert d.approved
    assert d.approved_size_usd == 100.0  # capped at max_position_usd


# --- Rejection ---

def test_rejects_low_confidence(risk):
    d = _approve(risk, confidence=0.40)
    assert not d.approved
    assert d.reason == RejectionReason.LOW_CONFIDENCE


def test_rejects_insufficient_edge(risk):
    d = _approve(risk, edge_pct=0.02)
    assert not d.approved
    assert d.reason == RejectionReason.INSUFFICIENT_EDGE


def test_rejects_low_liquidity(risk):
    d = _approve(risk, liquidity_usd=500.0)
    assert not d.approved
    assert d.reason == RejectionReason.LOW_LIQUIDITY


def test_rejects_after_daily_loss_limit(risk):
    # Simulate hitting the daily loss limit
    risk._daily_pnl = -201.0
    d = _approve(risk)
    assert not d.approved
    assert d.reason == RejectionReason.DAILY_LOSS_LIMIT


def test_rejects_over_max_positions(risk):
    for i in range(5):
        risk.open_position(f"m{i}", "poly", "YES", 10.0, 0.5)
    d = _approve(risk, market_id="overflow")
    assert not d.approved
    assert d.reason == RejectionReason.MAX_POSITIONS


def test_rejects_over_portfolio_exposure(risk):
    for i in range(5):
        risk.open_position(f"m{i}", "poly", "YES", 99.0, 0.5)
    # Keep only 1 position (stay under max_concurrent) but fill portfolio to ~$499.50
    # so remaining capacity is $0.50 < minimum $1.00 → rejected
    risk._positions = {"only": risk._positions.get("m0")}
    risk._positions["only"].size_usd = 499.5
    d = _approve(risk, market_id="overflow2")
    assert not d.approved
    assert d.reason == RejectionReason.MAX_PORTFOLIO_EXPOSURE


# --- Cooldown ---

def test_cooldown_after_loss(risk):
    risk.open_position("loser", "poly", "YES", 50.0, 0.5)
    risk.close_position("loser", 0.2)  # loss triggers cooldown
    d = _approve(risk, market_id="loser")
    assert not d.approved
    assert d.reason == RejectionReason.MARKET_COOLDOWN


# --- Position lifecycle ---

def test_open_close_position(risk):
    risk.open_position("mkt", "poly", "YES", 50.0, 0.5)
    assert "mkt" in risk._positions
    pnl = risk.close_position("mkt", 0.75)
    assert pnl is not None
    assert pnl > 0
    assert "mkt" not in risk._positions
    assert risk._daily_pnl > 0


def test_stop_loss_trigger(risk):
    risk.open_position("sl_test", "poly", "YES", 50.0, entry_price=0.50)
    pos = risk._positions["sl_test"]
    # Stop loss at 0.5 * (1 - 0.30) = 0.35
    action = risk.update_position_price("sl_test", 0.34)
    assert action == "STOP_LOSS"


def test_take_profit_trigger(risk):
    risk.open_position("tp_test", "poly", "YES", 50.0, entry_price=0.50)
    pos = risk._positions["tp_test"]
    # Take profit at 0.5 * (1 + 0.80) = 0.90
    action = risk.update_position_price("tp_test", 0.91)
    assert action == "TAKE_PROFIT"


def test_no_trigger_at_mid_price(risk):
    risk.open_position("safe", "poly", "YES", 50.0, entry_price=0.50)
    action = risk.update_position_price("safe", 0.60)
    assert action is None


# --- Portfolio summary ---

def test_portfolio_summary_structure(risk):
    summary = risk.portfolio_summary()
    assert "open_positions" in summary
    assert "total_exposure_usd" in summary
    assert "daily_pnl_usd" in summary
    assert "trading_halted" in summary
    assert summary["trading_halted"] is False


def test_scan_stop_losses(risk):
    risk.open_position("watch1", "poly", "YES", 50.0, entry_price=0.50)
    risk.open_position("watch2", "poly", "YES", 50.0, entry_price=0.50)
    # watch1 at stop-loss level, watch2 safe
    exits = risk.scan_stop_losses({"watch1": 0.30, "watch2": 0.60})
    assert "watch1" in exits
    assert "watch2" not in exits
