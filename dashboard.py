"""
Live trading dashboard — prediction analysis, research, and execution.

Run:  python dashboard.py
      python dashboard.py --mock          (safe, no real orders)
      python dashboard.py --budget 100    (set bankroll)

Panels:
  1. Header bar       — mode, budget, daily P&L, cycle count
  2. Market Signals   — all markets: sparkline, edge, quality, signal
  3. AI Research      — Claude's reasoning + key factors for top opportunity
  4. Signal Breakdown — per-signal view: Bayesian / AI / momentum / order-flow etc.
  5. News Feed        — headlines that drove each AI analysis
  6. Calibration      — rolling Brier score, MAD, category performance
  7. Arbitrage        — cross-platform spread opportunities
  8. Trade Log        — every signal with MOCK/EXECUTED/REJECTED status
  9. Portfolio bar    — daily P&L + open positions with live unrealised P&L
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

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
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

# ── Sparkline renderer ─────────────────────────────────────────────────────────

_SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 12) -> str:
    """Render 0-1 floats as a unicode block-character trend line."""
    if not values:
        return "─" * width
    sample = values[-width:]
    lo, hi = min(sample), max(sample)
    span = hi - lo or 1e-9
    out = [_SPARK_CHARS[int((v - lo) / span * (len(_SPARK_CHARS) - 1))] for v in sample]
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
    return "[dim]$0.00[/dim]"


def conf_bar(c: float, width: int = 10) -> str:
    """Render a 0-1 confidence as a coloured block bar."""
    filled = int(c * width)
    col = "green" if c >= 0.7 else "yellow" if c >= 0.5 else "red"
    return f"[{col}]{'█' * filled}[/{col}][dim]{'░' * (width - filled)}[/dim]"


# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class TradeLog:
    timestamp: str
    platform: str
    market: str
    direction: str
    size: float
    price: float
    status: str   # EXECUTED | REJECTED | MOCK
    reason: str = ""


@dataclass
class NewsItem:
    market_id: str
    market_question: str
    headline: str
    sentiment: float   # -1 .. +1
    fetched_at: float = field(default_factory=time.time)


# ── Main dashboard bot ─────────────────────────────────────────────────────────

class DashboardBot:
    def __init__(self, budget: float = 100.0, mock: bool = True) -> None:
        cfg = get_settings()
        self._budget = budget
        self._mock = mock
        self._running = True

        # Price / news history
        self._price_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=60))
        self._news_cache: dict[str, list[str]] = {}   # market_id → headlines list
        self._trade_log: deque[TradeLog] = deque(maxlen=50)
        self._news_feed: deque[NewsItem] = deque(maxlen=30)

        # Calibration trend: rolling brier score history
        self._brier_history: deque[float] = deque(maxlen=30)
        self._mad_history: deque[float] = deque(maxlen=30)

        # Category win tracking
        self._cat_stats: dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0, "pnl": 0.0})

        self._cycle = 0
        self._start_time = time.time()
        self._last_analyses: list[FullMarketAnalysis] = []
        self._last_arb: list = []
        self._selected_market: Optional[FullMarketAnalysis] = None  # drilled-down view

        # AI / fetchers
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

        # Risk manager — scaled to budget
        self._risk = RiskManager(
            max_position_usd=budget * 0.10,
            max_portfolio_usd=budget * 0.80,
            daily_loss_limit_usd=budget * 0.20,
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

    # ── Execution ──────────────────────────────────────────────────────────────

    async def _execute_poly(self, market_id: str, direction: str, size: float, price: float) -> None:
        market = await self._poly.get_market(market_id)
        if not market:
            return
        token = market.yes_token_id if direction == "YES" else market.no_token_id
        order = ClobOrder(condition_id=market_id, token_id=token,
                          side="BUY", size=round(size / price, 4), price=round(price, 4))
        await self._poly.place_order(order)
        self._trade_log.appendleft(TradeLog(
            timestamp=time.strftime("%H:%M:%S"), platform="POLY",
            market=market_id[:18], direction=direction,
            size=size, price=price,
            status="MOCK" if self._mock else "EXECUTED",
        ))

    async def _execute_kalshi(self, market_id: str, direction: str, size: float, price: float) -> None:
        order = KalshiOrder(ticker=market_id, action="buy", side=direction.lower(),
                            count=max(1, int(size)), limit_price=int(round(price * 100)),
                            client_order_id=f"dash_{int(time.time())}")
        await self._kalshi.place_order(order)
        self._trade_log.appendleft(TradeLog(
            timestamp=time.strftime("%H:%M:%S"), platform="KALSHI",
            market=market_id[:18], direction=direction,
            size=size, price=price,
            status="MOCK" if self._mock else "EXECUTED",
        ))

    # ── Main analysis cycle ────────────────────────────────────────────────────

    async def run_cycle(self) -> None:
        self._cycle += 1

        poly_dicts, kalshi_dicts = await asyncio.gather(
            self._fetch_poly(), self._fetch_kalshi(),
        )
        all_markets = poly_dicts + kalshi_dicts
        if not all_markets:
            return

        # Update price history
        for m in all_markets:
            mid = m["id"]
            self._price_history[mid].append(float(m.get("price", 0.5)))
            # Cache news summary per market for the research panel
            if m.get("news_summary"):
                self._news_cache[mid] = m["news_summary"]

        # Batch AI analysis
        analyses = await self._analyzer.analyze_batch(all_markets, batch_size=5)
        self._last_analyses = analyses

        # Update news feed from analyses that have AI results
        for a in analyses:
            if a.ai_analysis and a.ai_analysis.key_factors:
                for kf in a.ai_analysis.key_factors[:2]:
                    self._news_feed.appendleft(NewsItem(
                        market_id=a.market_id,
                        market_question=a.question[:35],
                        headline=kf,
                        sentiment=a.edge,
                    ))

        # Arbitrage detection
        if poly_dicts and kalshi_dicts:
            self._last_arb = self._analyzer.find_arbitrage(poly_dicts, kalshi_dicts)

        # Auto-select the highest-edge market for the research panel
        if analyses:
            self._selected_market = max(analyses, key=lambda a: abs(a.edge))

        # Route actionable signals through risk manager
        for analysis in [a for a in analyses if a.signal != "HOLD"]:
            result = await self._router.route(analysis)
            if not result.risk_decision.approved:
                reason = result.risk_decision.reason.value if result.risk_decision.reason else "?"
                self._trade_log.appendleft(TradeLog(
                    timestamp=time.strftime("%H:%M:%S"),
                    platform=analysis.platform.upper()[:6],
                    market=analysis.market_id[:18],
                    direction=analysis.signal,
                    size=0, price=analysis.market_price,
                    status="REJECT", reason=reason[:16],
                ))
            else:
                self._cat_stats[analysis.category]["total"] += 1

        # Stop-loss scan
        price_map = {m["id"]: float(m.get("price", 0.5)) for m in all_markets}
        self._risk.scan_stop_losses(price_map)

        # Update calibration trend
        calib = self._analyzer.calibration_stats()
        if calib.get("n", 0) > 0:
            self._brier_history.append(calib.get("brier", 0.25))
            er = calib.get("excess_return", {})
            if isinstance(er, dict):
                self._mad_history.append(er.get("mad", 0.25))

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

    # ── Rendering ──────────────────────────────────────────────────────────────

    def render(self) -> Panel:
        layout = Layout()
        layout.split_column(
            Layout(self._render_header(), size=3),
            Layout(name="body"),
            Layout(name="bottom", size=16),
        )
        layout["body"].split_row(
            Layout(name="left", ratio=5),
            Layout(name="right", ratio=4),
        )
        layout["left"].split_column(
            Layout(self._render_signals(), ratio=3),
            Layout(self._render_news_feed(), ratio=2),
        )
        layout["right"].split_column(
            Layout(self._render_ai_research(), ratio=3),
            Layout(self._render_signal_breakdown(), ratio=2),
        )
        layout["bottom"].split_row(
            Layout(self._render_trade_log(), ratio=3),
            Layout(name="br", ratio=2),
        )
        layout["br"].split_column(
            Layout(self._render_portfolio(), ratio=2),
            Layout(self._render_calibration(), ratio=3),
        )
        return Panel(layout, title="🤖 AI Prediction Market Bot — Research & Analysis", border_style="cyan")

    # Header bar ───────────────────────────────────────────────────────────────

    def _render_header(self) -> Table:
        t = Table.grid(expand=True)
        t.add_column(ratio=1); t.add_column(ratio=1); t.add_column(ratio=1)
        uptime = (time.time() - self._start_time) / 60
        s = self._risk.portfolio_summary()
        mode_str = "[red]LIVE[/red]" if not self._mock else "[yellow]MOCK[/yellow]"
        t.add_row(
            f"  Mode: {mode_str}  Budget: [cyan]${self._budget:.0f}[/cyan]"
            f"  Cycle: [dim]#{self._cycle}[/dim]  Up: [dim]{uptime:.0f}m[/dim]",
            f"[center]P&L: {pnl_color(s['daily_pnl_usd'])}  "
            f"Positions: [cyan]{s['open_positions']}[/cyan]  "
            f"Exposure: [cyan]${s['total_exposure_usd']:.0f}[/cyan][/center]",
            f"  [right]Halt: {'[red]YES[/red]' if s['trading_halted'] else '[green]NO[/green]'}  "
            f"Markets: [cyan]{len(self._last_analyses)}[/cyan]  "
            f"Capacity: [cyan]${s['capacity_remaining_usd']:.0f}[/cyan][/right]",
        )
        return t

    # Market signals with sparklines ───────────────────────────────────────────

    def _render_signals(self) -> Panel:
        t = Table(show_lines=False, expand=True, box=None, header_style="bold dim")
        t.add_column("Plt", width=5)
        t.add_column("Cat", width=7)
        t.add_column("Question", max_width=28)
        t.add_column("Trend", width=15)
        t.add_column("Mkt%", width=5, justify="right")
        t.add_column("Est%", width=5, justify="right")
        t.add_column("Edge", width=6, justify="right")
        t.add_column("Conf", width=5, justify="right")
        t.add_column("Q", width=4, justify="right")
        t.add_column("Signal", width=11, justify="center")

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
                sig_str = "[dim]  HOLD  [/dim]"
                edge_str = f"[dim]{a.edge:+.1%}[/dim]"

            conf = a.final_confidence
            conf_col = "green" if conf >= 0.7 else "yellow" if conf >= 0.5 else "dim"

            t.add_row(
                f"[dim]{a.platform[:4]}[/dim]",
                f"[dim]{a.category[:6]}[/dim]",
                a.question[:28],
                f"[dim]{spark}[/dim] {arrow}",
                f"{a.market_price:.0%}",
                f"{a.final_probability:.0%}",
                edge_str,
                f"[{conf_col}]{conf:.0%}[/{conf_col}]",
                f"{a.market_quality:.2f}",
                sig_str,
            )

        if not top:
            t.add_row("[dim]Fetching markets...[/dim]", "", "", "", "", "", "", "", "", "")

        return Panel(t, title="📊 Live Market Signals + Price Trends", border_style="dim")

    # AI Research panel (Claude reasoning for top opportunity) ─────────────────

    def _render_ai_research(self) -> Panel:
        a = self._selected_market
        if a is None or a.ai_analysis is None:
            return Panel(
                "[dim]Waiting for AI analysis...\n\nClaude will analyse the top opportunity\nand show its reasoning here.[/dim]",
                title="🧠 Claude AI Research", border_style="blue",
            )

        ai = a.ai_analysis
        lines: list[str] = []

        # Market header
        sig_col = "green" if a.signal == "BUY_YES" else "red" if a.signal == "BUY_NO" else "dim"
        lines.append(f"[bold]{a.question[:70]}[/bold]")
        lines.append(
            f"Platform: [cyan]{a.platform}[/cyan]  "
            f"Category: [cyan]{a.category}[/cyan]  "
            f"Quality: [cyan]{a.market_quality:.2f}[/cyan]"
        )
        lines.append("")

        # Probability comparison
        lines.append("[bold dim]─── Probability Estimates ─────────────────────────────[/bold dim]")
        lines.append(
            f"  Market price:     [yellow]{a.market_price:.1%}[/yellow]   "
            f"({a.market_price * 100:.1f}¢ per share)"
        )
        lines.append(
            f"  Claude estimate:  [cyan]{ai.estimated_probability:.1%}[/cyan]   "
            f"({'cached' if ai.cached else 'live'})"
        )
        if a.ensemble:
            lines.append(
                f"  Ensemble (final): [bold]{a.final_probability:.1%}[/bold]   "
                f"Kelly: [cyan]{a.kelly_fraction:.1%}[/cyan]"
            )
        lines.append(
            f"  Edge:  [{sig_col}]{a.edge:+.1%}[/{sig_col}]   "
            f"Confidence: {conf_bar(a.final_confidence)} {a.final_confidence:.0%}"
        )
        lines.append("")

        # Claude's reasoning
        lines.append("[bold dim]─── AI Reasoning ──────────────────────────────────────[/bold dim]")
        reasoning = ai.reasoning or "No reasoning available."
        for chunk in [reasoning[i:i+65] for i in range(0, min(len(reasoning), 260), 65)]:
            lines.append(f"  [dim]{chunk}[/dim]")
        lines.append("")

        # Key factors
        if ai.key_factors:
            lines.append("[bold dim]─── Key Factors ───────────────────────────────────────[/bold dim]")
            for i, factor in enumerate(ai.key_factors[:5], 1):
                col = "green" if a.edge > 0 else "red"
                lines.append(f"  [{col}]{i}.[/{col}] {factor[:62]}")
            lines.append("")

        # Uncertainty flags
        if ai.uncertainty_flags:
            lines.append("[bold dim]─── Uncertainty Flags ─────────────────────────────────[/bold dim]")
            for flag in ai.uncertainty_flags[:3]:
                lines.append(f"  [yellow]⚠[/yellow]  {flag[:62]}")

        # Signal recommendation
        lines.append("")
        if a.signal == "BUY_YES":
            lines.append(
                f"  [bold green]▲ RECOMMENDATION: BUY YES @ {a.market_price:.0%}[/bold green]  "
                f"Size: [cyan]${a.kelly_fraction * self._budget:.2f}[/cyan]"
            )
        elif a.signal == "BUY_NO":
            lines.append(
                f"  [bold red]▼ RECOMMENDATION: BUY NO @ {1 - a.market_price:.0%}[/bold red]  "
                f"Size: [cyan]${a.kelly_fraction * self._budget:.2f}[/cyan]"
            )
        else:
            lines.append("  [dim]  HOLD — insufficient edge or confidence[/dim]")

        return Panel("\n".join(lines), title="🧠 Claude AI Research", border_style="blue")

    # Signal breakdown panel ───────────────────────────────────────────────────

    def _render_signal_breakdown(self) -> Panel:
        a = self._selected_market
        if a is None or a.ensemble is None:
            return Panel(
                "[dim]Select a market to see signal breakdown.[/dim]",
                title="⚡ Prediction Signal Breakdown", border_style="dim",
            )

        t = Table(show_lines=False, expand=True, box=None, header_style="bold dim")
        t.add_column("Signal", max_width=22)
        t.add_column("Value", width=8, justify="right")
        t.add_column("Conf", width=8, justify="right")
        t.add_column("Bar", width=12)
        t.add_column("Weight", width=7, justify="right")

        for sig in a.ensemble.signals:
            v = sig.value
            v_col = "green" if v > 0.05 else "red" if v < -0.05 else "dim"
            t.add_row(
                sig.name[:22],
                f"[{v_col}]{v:+.3f}[/{v_col}]",
                f"{sig.confidence:.0%}",
                conf_bar(sig.confidence, width=8),
                f"[dim]{getattr(sig, 'weight', 1.0):.2f}[/dim]",
            )

        if not a.ensemble.signals:
            t.add_row("[dim]No signals computed[/dim]", "", "", "", "")

        return Panel(t, title="⚡ Prediction Signal Breakdown", border_style="dim")

    # News feed panel ──────────────────────────────────────────────────────────

    def _render_news_feed(self) -> Panel:
        t = Table(show_lines=False, expand=True, box=None, header_style="bold dim")
        t.add_column("Market", max_width=22)
        t.add_column("Research Finding / Key Factor", max_width=52)
        t.add_column("Sent", width=5, justify="right")

        shown = set()
        for item in list(self._news_feed)[:14]:
            key = item.headline[:20]
            if key in shown:
                continue
            shown.add(key)
            col = "green" if item.sentiment > 0.03 else "red" if item.sentiment < -0.03 else "dim"
            t.add_row(
                f"[dim]{item.market_question[:22]}[/dim]",
                item.headline[:52],
                f"[{col}]{item.sentiment:+.2f}[/{col}]",
            )

        # Also show raw news cache for selected market
        if self._selected_market:
            ns = self._news_cache.get(self._selected_market.market_id, "")
            if ns:
                for line in str(ns).split("\n")[:3]:
                    line = line.strip()
                    if line and len(line) > 10:
                        t.add_row("[cyan]news[/cyan]", line[:52], "")

        if not self._news_feed and not (self._selected_market and self._news_cache.get(getattr(self._selected_market, "market_id", ""))):
            t.add_row("[dim]News headlines will appear here after first analysis cycle...[/dim]", "", "")

        return Panel(t, title="📰 Research News Feed", border_style="dim")

    # Calibration & model performance panel ───────────────────────────────────

    def _render_calibration(self) -> Panel:
        calib = self._analyzer.calibration_stats()
        n = calib.get("n", 0)
        brier = calib.get("brier", 0.0)
        ece = calib.get("ece", 0.0)
        er = calib.get("excess_return", {})
        mad = er.get("mad", 0.0) if isinstance(er, dict) else 0.0
        mean_er = er.get("mean_excess_return", 0.0) if isinstance(er, dict) else 0.0

        brier_spark = sparkline(list(self._brier_history), width=16) if self._brier_history else "─" * 16
        mad_spark = sparkline(list(self._mad_history), width=16) if self._mad_history else "─" * 16

        t = Table(show_lines=False, expand=True, box=None, header_style="bold dim")
        t.add_column("Metric", max_width=22)
        t.add_column("Value", width=8, justify="right")
        t.add_column("Trend", width=18)
        t.add_column("Rating", width=10)

        brier_col = "green" if brier < 0.20 else "yellow" if brier < 0.25 else "red"
        mad_col = "green" if mad < 0.15 else "yellow" if mad < 0.25 else "red"
        er_col = "green" if mean_er > 0 else "red"

        t.add_row("Resolved n", str(n), "", "")
        t.add_row(
            "Brier score",
            f"[{brier_col}]{brier:.3f}[/{brier_col}]",
            f"[dim]{brier_spark}[/dim]",
            "[green]Good[/green]" if brier < 0.20 else "[yellow]Fair[/yellow]" if brier < 0.25 else "[red]Poor[/red]",
        )
        t.add_row(
            "Calibration ECE",
            f"{ece:.3f}",
            "",
            "[green]Good[/green]" if ece < 0.05 else "[yellow]Fair[/yellow]",
        )
        t.add_row(
            "MAD drift",
            f"[{mad_col}]{mad:.3f}[/{mad_col}]",
            f"[dim]{mad_spark}[/dim]",
            "[green]Stable[/green]" if mad < 0.15 else "[yellow]Drifting[/yellow]",
        )
        t.add_row(
            "Mean excess return",
            f"[{er_col}]{mean_er:+.3f}[/{er_col}]",
            "",
            "[green]Positive edge[/green]" if mean_er > 0 else "[dim]Neutral[/dim]",
        )

        # Category stats
        if self._cat_stats:
            t.add_row("", "", "", "")
            t.add_row("[bold dim]Category[/bold dim]", "[bold dim]Trades[/bold dim]", "", "")
            for cat, stats in list(self._cat_stats.items())[:4]:
                t.add_row(f"  {cat[:12]}", str(stats["total"]), "", "")

        return Panel(t, title="📈 Calibration & Model Performance", border_style="dim")

    # Arbitrage panel ──────────────────────────────────────────────────────────

    def _render_arb(self) -> Panel:
        t = Table(show_lines=False, expand=True, box=None, header_style="bold dim")
        t.add_column("Market", max_width=22)
        t.add_column("Poly%", width=6, justify="right")
        t.add_column("Kalshi%", width=7, justify="right")
        t.add_column("Net spread", width=10, justify="right")
        t.add_column("Conf", width=5, justify="right")

        for opp in self._last_arb[:8]:
            t.add_row(
                f"[yellow]{opp.direction[:22]}[/yellow]",
                f"{opp.poly_yes_price:.0%}",
                f"{opp.kalshi_yes_price:.0%}",
                f"[bold yellow]{opp.net_spread:.1%}[/bold yellow]",
                f"{opp.confidence:.0%}",
            )
        if not self._last_arb:
            t.add_row("[dim]No arb opportunities found this cycle[/dim]", "", "", "", "")

        return Panel(t, title="⚡ Cross-Platform Arbitrage", border_style="yellow")

    # Trade log panel ──────────────────────────────────────────────────────────

    def _render_trade_log(self) -> Panel:
        t = Table(show_lines=False, expand=True, box=None, header_style="bold dim")
        t.add_column("Time", width=9)
        t.add_column("Plt", width=6)
        t.add_column("Market", max_width=20)
        t.add_column("Dir", width=10)
        t.add_column("Size", width=7, justify="right")
        t.add_column("Price", width=6, justify="right")
        t.add_column("Status", width=22)

        for log in list(self._trade_log)[:10]:
            status_col = (
                "green" if log.status == "EXECUTED"
                else "yellow" if log.status == "MOCK"
                else "dim"
            )
            status_str = log.status if not log.reason else f"{log.status}:{log.reason}"
            t.add_row(
                f"[dim]{log.timestamp}[/dim]",
                f"[dim]{log.platform}[/dim]",
                log.market,
                f"[cyan]{log.direction}[/cyan]",
                f"${log.size:.2f}" if log.size > 0 else "[dim]—[/dim]",
                f"{log.price:.1%}",
                f"[{status_col}]{status_str[:22]}[/{status_col}]",
            )

        if not self._trade_log:
            t.add_row("[dim]No trades yet — waiting for signals...[/dim]", "", "", "", "", "", "")

        return Panel(t, title="📋 Trade Log", border_style="dim")

    # Portfolio panel ──────────────────────────────────────────────────────────

    def _render_portfolio(self) -> Panel:
        s = self._risk.portfolio_summary()
        positions = self._risk._positions

        daily_pnl = s["daily_pnl_usd"]
        daily_limit = self._risk.daily_loss_limit_usd
        pnl_pct = abs(daily_pnl) / daily_limit if daily_limit > 0 else 0
        bar_filled = int(pnl_pct * 20)
        bar_col = "red" if daily_pnl < 0 else "green"
        bar = f"[{bar_col}]{'█' * bar_filled}[/{bar_col}][dim]{'░' * (20 - bar_filled)}[/dim]"

        t = Table.grid(expand=True)
        t.add_column(ratio=1); t.add_column(ratio=2)
        pos_text = "  ".join(
            f"[cyan]{pid[:10]}[/cyan] {p.direction} ${p.size_usd:.0f}@{p.entry_price:.0%} {pnl_color(p.pnl_usd)}"
            for pid, p in list(positions.items())[:3]
        ) or "[dim]No open positions[/dim]"
        t.add_row(
            f"  P&L: {pnl_color(daily_pnl)} / limit ${daily_limit:.0f}\n  {bar} {pnl_pct:.0%}",
            f"  {pos_text}",
        )
        return Panel(t, title="💼 Portfolio", border_style="dim")

    # ── Main loop ──────────────────────────────────────────────────────────────

    async def run(self, interval: int = 60) -> None:
        signal.signal(signal.SIGINT, lambda *_: self._stop())
        signal.signal(signal.SIGTERM, lambda *_: self._stop())

        with Live(self.render(), console=console, refresh_per_second=2, screen=True) as live:
            while self._running:
                cycle_start = time.time()
                try:
                    await self.run_cycle()
                except Exception as exc:
                    self._trade_log.appendleft(TradeLog(
                        timestamp=time.strftime("%H:%M:%S"), platform="SYS",
                        market="error", direction="—", size=0, price=0,
                        status="ERROR", reason=str(exc)[:20],
                    ))
                live.update(self.render())
                elapsed = time.time() - cycle_start
                for _ in range(max(1, int(interval - elapsed))):
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


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="AI Prediction Market Trading Dashboard")
    parser.add_argument("--mock", action="store_true", default=False)
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--interval", type=int, default=None)
    args = parser.parse_args()

    cfg = get_settings()
    mock = args.mock or cfg.bot.mock_mode
    budget = args.budget or float(os.getenv("TRADING_BUDGET_USD", "100"))
    interval = args.interval or cfg.bot.cycle_interval_seconds

    console.print(Panel.fit(
        f"[bold green]AI Prediction Market Bot — Research Dashboard[/bold green]\n"
        f"Budget: [cyan]${budget:.0f}[/cyan]  "
        f"Mode: {'[yellow]MOCK[/yellow]' if mock else '[red]LIVE[/red]'}  "
        f"Interval: [dim]{interval}s[/dim]\n\n"
        f"Panels: Market Signals · AI Research · Signal Breakdown\n"
        f"        News Feed · Calibration · Arbitrage · Trade Log\n\n"
        f"[dim]Starting up... Ctrl+C to exit[/dim]",
    ))
    await asyncio.sleep(1)

    bot = DashboardBot(budget=budget, mock=mock)
    try:
        await bot.run(interval=interval)
    finally:
        await bot.close()
        console.print("\n[green]Bot stopped cleanly.[/green]")
        s = bot._risk.portfolio_summary()
        console.print(f"Final daily P&L: {pnl_color(s['daily_pnl_usd'])}")


if __name__ == "__main__":
    asyncio.run(main())
