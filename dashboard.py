"""
Live trading dashboard with market trend visualization.

Run:  python dashboard.py
      python dashboard.py --mock          (safe, no real orders)
      python dashboard.py --budget 100    (set bankroll)

Shows:
  - Live price trend sparklines per market
  - Active positions with live P&L
  - Buy/Sell signals as they fire
  - Portfolio summary + daily P&L bar
  - Stop-loss / take-profit status
  - Calibration metrics
"""
from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

from config.settings import get_settings
from kalshi_bot.kalshi_client import KalshiClient, KalshiOrder
from polymarket_bot.polymarket_client import ClobOrder, PolymarketClient
from shared.claude_agent import ClaudeAgent
from shared.market_analyzer import FullMarketAnalysis, MarketAnalyzer
from shared.news_fetcher import NewsFetcher
from shared.risk_manager import RiskManager
from shared.signal_router import SignalRouter

console = Console()

# ── Sparkline renderer ────────────────────────────────────────────────────────

_SPARK_CHARS = " ▁▂▃▄▅▆▇█"

def sparkline(values: list[float], width: int = 12) -> str:
    """Render a list of 0–1 floats as a unicode sparkline."""
    if not values:
        return "─" * width
    sample = values[-width:]
    lo, hi = min(sample), max(sample)
    span = hi - lo or 1e-9
    out = []
    for v in sample:
        idx = int((v - lo) / span * (len(_SPARK_CHARS) - 1))
        out.append(_SPARK_CHARS[idx])
    # Pad left if shorter than width
    return "─" * (width - len(out)) + "".join(out)

def trend_arrow(values: list[float]) -> str:
    if len(values) < 2:
        return "→"
    delta = values[-1] - values[-2]
    if delta > 0.01:
        return "[green]▲[/green]"
    if delta < -0.01:
        return "[red]▼[/red]"
    return "[dim]→[/dim]"

def pnl_color(pnl: float) -> str:
    if pnl > 0:
        return f"[green]+${pnl:.2f}[/green]"
    if pnl < 0:
        return f"[red]-${abs(pnl):.2f}[/red]"
    return f"[dim]$0.00[/dim]"


# ── Trade log entry ───────────────────────────────────────────────────────────

@dataclass
class TradeLog:
    timestamp: str
    platform: str
    market: str
    direction: str
    size: float
    price: float
    signal: str
    status: str   # EXECUTED | REJECTED | MOCK


# ── Main dashboard bot ────────────────────────────────────────────────────────

class DashboardBot:
    def __init__(self, budget: float = 100.0, mock: bool = True) -> None:
        cfg = get_settings()
        self._budget = budget
        self._mock = mock
        self._running = True

        # Price history: market_id → deque of prices
        self._price_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=24))
        self._trade_log: deque[TradeLog] = deque(maxlen=50)
        self._cycle = 0
        self._start_time = time.time()

        # Shared AI
        self._news = NewsFetcher(
            tavily_key=cfg.news.tavily_api_key,
            newsapi_key=cfg.news.newsapi_key,
        )
        self._claude = ClaudeAgent(
            api_key=cfg.claude.api_key,
            model=cfg.claude.model,
            cache_ttl=cfg.claude.cache_ttl_seconds,
        )
        self._analyzer = MarketAnalyzer(
            claude=self._claude,
            news=self._news,
            kelly_max_fraction=cfg.risk.max_kelly_fraction,
        )

        # Risk — scaled to budget
        max_pos = budget * 0.10
        daily_limit = budget * 0.20
        self._risk = RiskManager(
            max_position_usd=max_pos,
            max_portfolio_usd=budget * 0.80,
            daily_loss_limit_usd=daily_limit,
            stop_loss_pct=cfg.risk.stop_loss_pct,
            take_profit_pct=cfg.risk.take_profit_pct,
            min_edge_pct=cfg.risk.min_edge_pct,
            min_confidence=cfg.risk.min_confidence_score,
        )

        # Platform clients
        self._poly = PolymarketClient(
            private_key=cfg.polymarket.private_key,
            api_key=cfg.polymarket.api_key,
            api_secret=cfg.polymarket.api_secret,
            api_passphrase=cfg.polymarket.api_passphrase,
            clob_host=cfg.polymarket.clob_host,
            gamma_host=cfg.polymarket.gamma_host,
            chain_id=cfg.polymarket.chain_id,
            mock_mode=mock or cfg.polymarket.mock_mode,
        )
        self._kalshi = KalshiClient(
            api_key_id=cfg.kalshi.api_key_id,
            private_key_path=cfg.kalshi.private_key_path,
            base_url=cfg.kalshi.base_url,
            mock_mode=mock or cfg.kalshi.mock_mode,
        )
        self._router = SignalRouter(
            risk_manager=self._risk,
            poly_executor=self._execute_poly,
            kalshi_executor=self._execute_kalshi,
            bankroll_usd=budget,
        )

        # Latest analyses for display
        self._last_analyses: list[FullMarketAnalysis] = []
        self._last_arb: list = []

    # ── Execution ─────────────────────────────────────────────────────────────

    async def _execute_poly(self, market_id: str, direction: str, size: float, price: float) -> None:
        market = await self._poly.get_market(market_id)
        if not market:
            return
        token = market.yes_token_id if direction == "YES" else market.no_token_id
        order = ClobOrder(condition_id=market_id, token_id=token,
                          side="BUY", size=round(size / price, 4), price=round(price, 4))
        result = await self._poly.place_order(order)
        self._trade_log.appendleft(TradeLog(
            timestamp=time.strftime("%H:%M:%S"),
            platform="POLY",
            market=market_id[:16],
            direction=direction,
            size=size,
            price=price,
            signal="TRADE",
            status="MOCK" if self._mock else "EXECUTED",
        ))

    async def _execute_kalshi(self, market_id: str, direction: str, size: float, price: float) -> None:
        order = KalshiOrder(ticker=market_id, action="buy", side=direction.lower(),
                            count=max(1, int(size)), limit_price=int(round(price * 100)),
                            client_order_id=f"dash_{int(time.time())}")
        await self._kalshi.place_order(order)
        self._trade_log.appendleft(TradeLog(
            timestamp=time.strftime("%H:%M:%S"),
            platform="KALSHI",
            market=market_id[:16],
            direction=direction,
            size=size,
            price=price,
            signal="TRADE",
            status="MOCK" if self._mock else "EXECUTED",
        ))

    # ── Cycle ─────────────────────────────────────────────────────────────────

    async def run_cycle(self) -> None:
        self._cycle += 1

        poly_dicts, kalshi_dicts = await asyncio.gather(
            self._fetch_poly(),
            self._fetch_kalshi(),
        )
        all_markets = poly_dicts + kalshi_dicts
        if not all_markets:
            return

        # Update price history
        for m in all_markets:
            mid = m["id"]
            self._price_history[mid].append(float(m.get("price", 0.5)))

        # Batch AI analysis
        analyses = await self._analyzer.analyze_batch(all_markets, batch_size=5)
        self._last_analyses = analyses

        # Arbitrage
        if poly_dicts and kalshi_dicts:
            self._last_arb = self._analyzer.find_arbitrage(poly_dicts, kalshi_dicts)

        # Route actionable signals
        actionable = [a for a in analyses if a.signal != "HOLD"]
        for analysis in actionable:
            result = await self._router.route(analysis)
            if not result.risk_decision.approved:
                self._trade_log.appendleft(TradeLog(
                    timestamp=time.strftime("%H:%M:%S"),
                    platform=analysis.platform.upper()[:6],
                    market=analysis.market_id[:16],
                    direction=analysis.signal,
                    size=0,
                    price=analysis.market_price,
                    signal=analysis.signal,
                    status=f"REJECT:{result.risk_decision.reason.value[:10]}",
                ))

        # Stop-loss monitoring
        price_map = {m["id"]: float(m.get("price", 0.5)) for m in all_markets}
        self._risk.scan_stop_losses(price_map)

    async def _fetch_poly(self) -> list[dict]:
        try:
            markets = await self._poly.get_markets(limit=15)
            return [m.to_analysis_dict() for m in markets]
        except Exception:
            return []

    async def _fetch_kalshi(self) -> list[dict]:
        try:
            markets = await self._kalshi.get_markets(limit=15)
            return [m.to_analysis_dict() for m in markets]
        except Exception:
            return []

    # ── Dashboard rendering ───────────────────────────────────────────────────

    def render(self) -> Panel:
        layout = Layout()
        layout.split_column(
            Layout(self._render_header(), size=4),
            Layout(name="middle"),
            Layout(self._render_trade_log(), size=14),
            Layout(self._render_portfolio(), size=6),
        )
        layout["middle"].split_row(
            Layout(self._render_signals(), ratio=3),
            Layout(self._render_arb(), ratio=2),
        )
        return Panel(layout, title="🤖 AI Prediction Market Bot", border_style="cyan")

    def _render_header(self) -> Table:
        t = Table.grid(expand=True)
        t.add_column(ratio=1)
        t.add_column(ratio=1)
        t.add_column(ratio=1)
        uptime = (time.time() - self._start_time) / 60
        summary = self._risk.portfolio_summary()
        daily_pnl = summary["daily_pnl_usd"]
        mode_str = "[red]LIVE[/red]" if not self._mock else "[yellow]MOCK[/yellow]"
        t.add_row(
            f"  Mode: {mode_str}  Budget: [cyan]${self._budget:.0f}[/cyan]  "
            f"Cycle: [dim]#{self._cycle}[/dim]  Uptime: [dim]{uptime:.0f}m[/dim]",
            f"[center]Daily P&L: {pnl_color(daily_pnl)}  "
            f"Positions: [cyan]{summary['open_positions']}[/cyan]  "
            f"Exposure: [cyan]${summary['total_exposure_usd']:.0f}[/cyan][/center]",
            f"  [right]Halt: {'[red]YES[/red]' if summary['trading_halted'] else '[green]NO[/green]'}  "
            f"Capacity: [cyan]${summary['capacity_remaining_usd']:.0f}[/cyan][/right]",
        )
        return t

    def _render_signals(self) -> Panel:
        t = Table(show_lines=False, expand=True, show_header=True, box=None,
                  header_style="bold dim")
        t.add_column("Platform", width=8)
        t.add_column("Category", width=8)
        t.add_column("Market", max_width=30)
        t.add_column("Trend", width=14)
        t.add_column("Mkt%", width=5, justify="right")
        t.add_column("Est%", width=5, justify="right")
        t.add_column("Edge", width=6, justify="right")
        t.add_column("Q", width=4, justify="right")
        t.add_column("Signal", width=10, justify="center")

        top = sorted(self._last_analyses, key=lambda a: abs(a.edge), reverse=True)[:14]
        for a in top:
            hist = list(self._price_history.get(a.market_id, [a.market_price]))
            spark = sparkline(hist)
            arrow = trend_arrow(hist)

            if a.signal == "BUY_YES":
                sig_str = "[bold green]▲ BUY YES[/bold green]"
                edge_str = f"[green]{a.edge:+.1%}[/green]"
            elif a.signal == "BUY_NO":
                sig_str = "[bold red]▼ BUY NO[/bold red]"
                edge_str = f"[red]{a.edge:+.1%}[/red]"
            else:
                sig_str = "[dim]  HOLD[/dim]"
                edge_str = f"[dim]{a.edge:+.1%}[/dim]"

            t.add_row(
                f"[dim]{a.platform[:7]}[/dim]",
                f"[dim]{a.category[:7]}[/dim]",
                a.question[:30],
                f"[dim]{spark}[/dim] {arrow}",
                f"{a.market_price:.0%}",
                f"{a.final_probability:.0%}",
                edge_str,
                f"{a.market_quality:.2f}",
                sig_str,
            )

        return Panel(t, title="📊 Market Signals + Trends", border_style="dim")

    def _render_arb(self) -> Panel:
        t = Table(show_lines=False, expand=True, box=None, header_style="bold dim")
        t.add_column("Direction", width=16)
        t.add_column("Poly%", width=6, justify="right")
        t.add_column("Kalshi%", width=7, justify="right")
        t.add_column("Net", width=6, justify="right")
        t.add_column("Conf", width=5, justify="right")

        for opp in self._last_arb[:8]:
            t.add_row(
                f"[yellow]{opp.direction[:16]}[/yellow]",
                f"{opp.poly_yes_price:.0%}",
                f"{opp.kalshi_yes_price:.0%}",
                f"[yellow]{opp.net_spread:.1%}[/yellow]",
                f"{opp.confidence:.0%}",
            )
        if not self._last_arb:
            t.add_row("[dim]No arb found[/dim]", "", "", "", "")

        calib = self._analyzer.calibration_stats()
        n = calib.get("n", 0)
        brier = calib.get("brier", 0.25)
        er = calib.get("excess_return", {})
        mad = er.get("mad", 0.25) if isinstance(er, dict) else 0.25

        footer = (
            f"\n[dim]Calibration: n={n}  Brier={brier:.3f}  MAD={mad:.3f}[/dim]"
        )
        return Panel(
            t.__rich_console__(console, console.options) and t or t,
            title="⚡ Arbitrage",
            subtitle=f"[dim]n={n} Brier={brier:.3f} MAD={mad:.3f}[/dim]",
            border_style="dim",
        )

    def _render_trade_log(self) -> Panel:
        t = Table(show_lines=False, expand=True, box=None, header_style="bold dim")
        t.add_column("Time", width=9)
        t.add_column("Platform", width=8)
        t.add_column("Market", max_width=22)
        t.add_column("Dir", width=10)
        t.add_column("Size", width=8, justify="right")
        t.add_column("Price", width=6, justify="right")
        t.add_column("Status", width=22)

        for log in list(self._trade_log)[:10]:
            status_color = (
                "green" if log.status == "EXECUTED"
                else "yellow" if log.status == "MOCK"
                else "dim"
            )
            t.add_row(
                f"[dim]{log.timestamp}[/dim]",
                f"[dim]{log.platform}[/dim]",
                log.market,
                f"[cyan]{log.direction}[/cyan]",
                f"${log.size:.2f}" if log.size > 0 else "[dim]—[/dim]",
                f"{log.price:.2%}",
                f"[{status_color}]{log.status}[/{status_color}]",
            )

        if not self._trade_log:
            t.add_row("[dim]No trades yet — waiting for signals...[/dim]",
                      "", "", "", "", "", "")

        return Panel(t, title="📋 Trade Log", border_style="dim")

    def _render_portfolio(self) -> Panel:
        summary = self._risk.portfolio_summary()
        positions = self._risk._positions

        t = Table.grid(expand=True)
        t.add_column(ratio=1)
        t.add_column(ratio=2)

        # Daily P&L progress bar (20% daily limit)
        daily_pnl = summary["daily_pnl_usd"]
        daily_limit = self._risk.daily_loss_limit_usd
        pnl_pct = abs(daily_pnl) / daily_limit if daily_limit > 0 else 0
        bar_filled = int(pnl_pct * 20)
        bar_color = "red" if daily_pnl < 0 else "green"
        bar = f"[{bar_color}]{'█' * bar_filled}[/{bar_color}]{'░' * (20 - bar_filled)}"

        t.add_row(
            f"  Daily P&L: {pnl_color(daily_pnl)} / limit ${daily_limit:.0f}\n"
            f"  {bar}  {pnl_pct:.0%} of limit used",
            "  " + "  ".join(
                f"[cyan]{pid[:12]}[/cyan] {p.direction} "
                f"${p.size_usd:.0f} @ {p.entry_price:.0%} "
                f"{pnl_color(p.pnl_usd)}"
                for pid, p in list(positions.items())[:4]
            ) or "[dim]  No open positions[/dim]",
        )

        return Panel(t, title="💼 Portfolio", border_style="dim")

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self, interval: int = 60) -> None:
        signal.signal(signal.SIGINT, lambda *_: self._stop())
        signal.signal(signal.SIGTERM, lambda *_: self._stop())

        with Live(self.render(), console=console, refresh_per_second=2,
                  screen=True) as live:
            while self._running:
                cycle_start = time.time()
                try:
                    await self.run_cycle()
                except Exception as exc:
                    self._trade_log.appendleft(TradeLog(
                        timestamp=time.strftime("%H:%M:%S"),
                        platform="SYS", market="error",
                        direction="—", size=0, price=0,
                        signal="ERROR", status=str(exc)[:30],
                    ))
                live.update(self.render())
                elapsed = time.time() - cycle_start
                sleep_for = max(1.0, interval - elapsed)
                # Refresh display every second while sleeping
                for _ in range(int(sleep_for)):
                    if not self._running:
                        break
                    await asyncio.sleep(1.0)
                    live.update(self.render())

    def _stop(self) -> None:
        self._running = False

    async def close(self) -> None:
        await self._poly.close()
        await self._kalshi.close()
        await self._news.close()


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="AI Prediction Market Trading Dashboard")
    parser.add_argument("--mock", action="store_true", default=False,
                        help="Force mock mode (no real orders)")
    parser.add_argument("--budget", type=float, default=None,
                        help="Trading budget in USD (default: from settings)")
    parser.add_argument("--interval", type=int, default=None,
                        help="Cycle interval in seconds (default: from settings)")
    args = parser.parse_args()

    cfg = get_settings()
    mock = args.mock or cfg.bot.mock_mode
    budget = args.budget or float(os.getenv("TRADING_BUDGET_USD", "100"))
    interval = args.interval or cfg.bot.cycle_interval_seconds

    console.print(Panel.fit(
        f"[bold green]AI Prediction Market Bot[/bold green]\n"
        f"Budget: [cyan]${budget:.0f}[/cyan]  "
        f"Mode: {'[yellow]MOCK[/yellow]' if mock else '[red]LIVE[/red]'}  "
        f"Interval: [dim]{interval}s[/dim]\n\n"
        f"[dim]Starting up... Ctrl+C to exit[/dim]",
    ))
    await asyncio.sleep(1)

    bot = DashboardBot(budget=budget, mock=mock)
    try:
        await bot.run(interval=interval)
    finally:
        await bot.close()
        console.print("\n[green]Bot stopped cleanly.[/green]")
        summary = bot._risk.portfolio_summary()
        console.print(f"Final daily P&L: {pnl_color(summary['daily_pnl_usd'])}")


if __name__ == "__main__":
    asyncio.run(main())
