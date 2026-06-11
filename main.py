"""
Unified bot: runs Polymarket + Kalshi simultaneously with a shared AI brain.

Architecture:
  - Single ClaudeAgent + MarketAnalyzer shared between both bots
  - Separate RiskManagers per platform (independent loss limits)
  - Parallel market fetching via asyncio.gather
  - Cross-platform arbitrage detection on each cycle
  - Rich terminal dashboard
"""
from __future__ import annotations

import asyncio
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import structlog
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
from shared.predictive_models import ArbitrageOpportunity
from shared.risk_manager import RiskManager
from shared.signal_router import SignalRouter

log = structlog.get_logger(__name__)
console = Console()


@dataclass
class BotStats:
    cycle_count: int = 0
    total_signals: int = 0
    executed_trades: int = 0
    arbitrage_found: int = 0
    poly_pnl: float = 0.0
    kalshi_pnl: float = 0.0
    start_time: float = field(default_factory=time.time)

    @property
    def uptime_hours(self) -> float:
        return (time.time() - self.start_time) / 3600


class UnifiedBot:
    """Runs both platforms with a single AI brain and cross-platform arbitrage."""

    def __init__(self) -> None:
        cfg = get_settings()
        self._cfg = cfg
        self._stats = BotStats()
        self._running = True

        # Shared AI layer
        self._news = NewsFetcher(
            tavily_key=cfg.news.tavily_api_key,
            newsapi_key=cfg.news.newsapi_key,
            max_articles=cfg.news.max_articles_per_query,
            cache_ttl=cfg.news.cache_ttl_seconds,
        )
        self._claude = ClaudeAgent(
            api_key=cfg.claude.api_key,
            model=cfg.claude.model,
            max_tokens=cfg.claude.max_tokens,
            cache_ttl=cfg.claude.cache_ttl_seconds,
        )
        self._analyzer = MarketAnalyzer(
            claude=self._claude,
            news=self._news,
            kelly_max_fraction=cfg.risk.max_kelly_fraction,
        )

        # Polymarket
        if cfg.bot.enable_polymarket:
            self._poly = PolymarketClient(
                private_key=cfg.polymarket.private_key,
                api_key=cfg.polymarket.api_key,
                api_secret=cfg.polymarket.api_secret,
                api_passphrase=cfg.polymarket.api_passphrase,
                clob_host=cfg.polymarket.clob_host,
                gamma_host=cfg.polymarket.gamma_host,
                chain_id=cfg.polymarket.chain_id,
                mock_mode=cfg.polymarket.mock_mode,
            )
            self._poly_risk = RiskManager(
                max_position_usd=cfg.risk.max_position_size_usd,
                max_portfolio_usd=cfg.risk.max_portfolio_exposure_usd,
                daily_loss_limit_usd=cfg.risk.daily_loss_limit_usd,
                stop_loss_pct=cfg.risk.stop_loss_pct,
                take_profit_pct=cfg.risk.take_profit_pct,
                min_edge_pct=cfg.risk.min_edge_pct,
                min_confidence=cfg.risk.min_confidence_score,
            )
            self._poly_router = SignalRouter(
                risk_manager=self._poly_risk,
                poly_executor=self._execute_poly,
                bankroll_usd=500.0,
            )
        else:
            self._poly = None
            self._poly_risk = None
            self._poly_router = None

        # Kalshi
        if cfg.bot.enable_kalshi:
            self._kalshi = KalshiClient(
                api_key_id=cfg.kalshi.api_key_id,
                private_key_path=cfg.kalshi.private_key_path,
                base_url=cfg.kalshi.base_url,
                mock_mode=cfg.kalshi.mock_mode,
            )
            self._kalshi_risk = RiskManager(
                max_position_usd=cfg.risk.max_position_size_usd,
                max_portfolio_usd=cfg.risk.max_portfolio_exposure_usd,
                daily_loss_limit_usd=cfg.risk.daily_loss_limit_usd,
                stop_loss_pct=cfg.risk.stop_loss_pct,
                take_profit_pct=cfg.risk.take_profit_pct,
                min_edge_pct=cfg.risk.min_edge_pct,
                min_confidence=cfg.risk.min_confidence_score,
            )
            self._kalshi_router = SignalRouter(
                risk_manager=self._kalshi_risk,
                kalshi_executor=self._execute_kalshi,
                bankroll_usd=500.0,
            )
        else:
            self._kalshi = None
            self._kalshi_risk = None
            self._kalshi_router = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._setup_signals()
        console.print(Panel.fit(
            "[bold green]Polymarket + Kalshi Unified AI Bot[/bold green]\n"
            f"Polymarket: {'[green]ON[/green]' if self._poly else '[red]OFF[/red]'}  "
            f"Kalshi: {'[green]ON[/green]' if self._kalshi else '[red]OFF[/red]'}  "
            f"Arbitrage: {'[green]ON[/green]' if self._cfg.bot.enable_arbitrage else '[red]OFF[/red]'}\n"
            f"Mock mode: [yellow]{self._cfg.bot.mock_mode}[/yellow]",
            title="🤖 AI Prediction Market Bot",
        ))

        interval = self._cfg.bot.cycle_interval_seconds
        while self._running:
            cycle_start = time.time()
            try:
                await self._run_cycle()
            except Exception as exc:
                log.error("main_cycle_error", error=str(exc), exc_info=True)

            elapsed = time.time() - cycle_start
            sleep_time = max(0.0, interval - elapsed)
            if sleep_time > 0 and self._running:
                await asyncio.sleep(sleep_time)

    async def _run_cycle(self) -> None:
        self._stats.cycle_count += 1
        log.info("cycle_start", n=self._stats.cycle_count)

        # Fetch markets from both platforms in parallel
        poly_markets_raw, kalshi_markets_raw = await asyncio.gather(
            self._fetch_poly_markets(),
            self._fetch_kalshi_markets(),
        )

        # Combine for batch AI analysis
        all_market_dicts = poly_markets_raw + kalshi_markets_raw

        if not all_market_dicts:
            log.warning("no_markets_fetched")
            return

        # Batch analysis (single AI call for efficiency)
        analyses = await self._analyzer.analyze_batch(
            all_market_dicts,
            batch_size=self._cfg.claude.batch_size,
        )

        poly_analyses = [a for a in analyses if a.platform == "polymarket"]
        kalshi_analyses = [a for a in analyses if a.platform == "kalshi"]

        # Route signals
        poly_results = []
        if self._poly_router and poly_analyses:
            poly_results = await self._poly_router.route_all(
                [a for a in poly_analyses if a.signal != "HOLD"]
            )

        kalshi_results = []
        if self._kalshi_router and kalshi_analyses:
            kalshi_results = await self._kalshi_router.route_all(
                [a for a in kalshi_analyses if a.signal != "HOLD"]
            )

        executed = sum(1 for r in poly_results + kalshi_results if r.executed)
        self._stats.total_signals += len(poly_results) + len(kalshi_results)
        self._stats.executed_trades += executed

        # Cross-platform arbitrage
        arb_opportunities: list[ArbitrageOpportunity] = []
        if self._cfg.bot.enable_arbitrage and poly_markets_raw and kalshi_markets_raw:
            arb_opportunities = self._analyzer.find_arbitrage(
                poly_markets_raw,
                kalshi_markets_raw,
                min_spread=self._cfg.bot.arbitrage_min_spread_pct,
            )
            if arb_opportunities:
                self._stats.arbitrage_found += len(arb_opportunities)
                await self._execute_arbitrage(arb_opportunities[:3])

        # Print dashboard
        self._print_cycle(analyses, arb_opportunities, executed)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _execute_poly(
        self, market_id: str, direction: str, size_usd: float, price: float
    ) -> None:
        if not self._poly:
            return
        market = await self._poly.get_market(market_id)
        if not market:
            raise ValueError(f"Poly market not found: {market_id}")
        token_id = market.yes_token_id if direction == "YES" else market.no_token_id
        shares = size_usd / price if price > 0 else 0
        order = ClobOrder(
            condition_id=market_id,
            token_id=token_id,
            side="BUY",
            size=round(shares, 4),
            price=round(price, 4),
        )
        await self._poly.place_order(order)

    async def _execute_kalshi(
        self, market_id: str, direction: str, size_usd: float, price: float
    ) -> None:
        if not self._kalshi:
            return
        order = KalshiOrder(
            ticker=market_id,
            action="buy",
            side=direction.lower(),
            count=max(1, int(size_usd)),
            limit_price=int(round(price * 100)),
            client_order_id=f"bot_{int(time.time())}",
        )
        await self._kalshi.place_order(order)

    async def _execute_arbitrage(
        self, opportunities: list[ArbitrageOpportunity]
    ) -> None:
        for opp in opportunities:
            log.info(
                "arbitrage_opportunity",
                poly_id=opp.poly_market_id,
                kalshi_id=opp.kalshi_market_id,
                spread=round(opp.spread, 3),
                net_spread=round(opp.net_spread, 3),
                direction=opp.direction,
                confidence=round(opp.confidence, 2),
            )
            # Research with AI before executing
            if self._cfg.bot.mock_mode:
                console.print(
                    f"  [yellow]MOCK ARB[/yellow]: {opp.direction} "
                    f"spread={opp.net_spread:.1%} conf={opp.confidence:.0%}"
                )

    # ------------------------------------------------------------------
    # Market fetchers
    # ------------------------------------------------------------------

    async def _fetch_poly_markets(self) -> list[dict]:
        if not self._poly:
            return []
        try:
            markets = await self._poly.get_markets(
                limit=self._cfg.polymarket.max_markets_per_cycle
            )
            return [m.to_analysis_dict() for m in markets]
        except Exception as exc:
            log.error("poly_fetch_error", error=str(exc))
            return []

    async def _fetch_kalshi_markets(self) -> list[dict]:
        if not self._kalshi:
            return []
        try:
            markets = await self._kalshi.get_markets(
                limit=self._cfg.kalshi.max_markets_per_cycle
            )
            return [m.to_analysis_dict() for m in markets]
        except Exception as exc:
            log.error("kalshi_fetch_error", error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def _print_cycle(
        self,
        analyses: list[FullMarketAnalysis],
        arb: list[ArbitrageOpportunity],
        executed: int,
    ) -> None:
        table = Table(
            title=f"Cycle #{self._stats.cycle_count} — "
                  f"{time.strftime('%H:%M:%S')} — "
                  f"Uptime {self._stats.uptime_hours:.1f}h",
            show_lines=True,
        )
        table.add_column("Platform", max_width=10)
        table.add_column("Question", max_width=40)
        table.add_column("Mkt%", justify="right")
        table.add_column("Est%", justify="right")
        table.add_column("Edge", justify="right")
        table.add_column("Conf", justify="right")
        table.add_column("Kelly", justify="right")
        table.add_column("Signal", justify="center")

        top = sorted(analyses, key=lambda x: abs(x.edge), reverse=True)[:12]
        for a in top:
            color = "green" if a.signal == "BUY_YES" else ("red" if a.signal == "BUY_NO" else "dim white")
            table.add_row(
                a.platform[:10],
                a.question[:40],
                f"{a.market_price:.0%}",
                f"{a.final_probability:.0%}",
                f"[{color}]{a.edge:+.1%}[/{color}]",
                f"{a.final_confidence:.0%}",
                f"{a.kelly_fraction:.1%}",
                f"[{color}]{a.signal}[/{color}]",
            )
        console.print(table)

        if arb:
            arb_table = Table(title="Arbitrage Opportunities", show_lines=True)
            arb_table.add_column("Poly Market")
            arb_table.add_column("Kalshi Market")
            arb_table.add_column("Poly%", justify="right")
            arb_table.add_column("Kalshi%", justify="right")
            arb_table.add_column("Net Spread", justify="right")
            arb_table.add_column("Direction")
            for o in arb[:5]:
                arb_table.add_row(
                    o.poly_market_id[:20],
                    o.kalshi_market_id[:20],
                    f"{o.poly_yes_price:.0%}",
                    f"{o.kalshi_yes_price:.0%}",
                    f"[yellow]{o.net_spread:.1%}[/yellow]",
                    o.direction,
                )
            console.print(arb_table)

        poly_summary = self._poly_risk.portfolio_summary() if self._poly_risk else {}
        kalshi_summary = self._kalshi_risk.portfolio_summary() if self._kalshi_risk else {}
        calib = self._analyzer.calibration_stats()

        console.print(
            f"  Executed this cycle: [bold]{executed}[/bold] | "
            f"Total: [bold]{self._stats.executed_trades}[/bold] | "
            f"Arb found: [bold]{self._stats.arbitrage_found}[/bold]\n"
            f"  Poly portfolio: {poly_summary} | "
            f"Kalshi portfolio: {kalshi_summary}\n"
            f"  Calibration: n={calib.get('n', 0)} "
            f"acc={calib.get('accuracy', 0):.0%} "
            f"brier={calib.get('brier', 0.25):.3f} "
            f"ece={calib.get('ece', float('nan'))}\n"
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _setup_signals(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, lambda *_: self._shutdown())
            except (OSError, ValueError):
                pass

    def _shutdown(self) -> None:
        console.print("\n[bold red]Shutting down...[/bold red]")
        self._running = False

    async def close(self) -> None:
        if self._poly:
            await self._poly.close()
        if self._kalshi:
            await self._kalshi.close()
        await self._news.close()


async def main() -> None:
    bot = UnifiedBot()
    try:
        await bot.run()
    finally:
        await bot.close()
        console.print("[green]Bot stopped cleanly.[/green]")


if __name__ == "__main__":
    asyncio.run(main())
