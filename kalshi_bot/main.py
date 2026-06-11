"""Kalshi-only bot runner."""
from __future__ import annotations

import asyncio
import time

import structlog
from rich.console import Console
from rich.table import Table

from config.settings import get_settings
from shared.claude_agent import ClaudeAgent
from shared.market_analyzer import MarketAnalyzer
from shared.news_fetcher import NewsFetcher
from shared.risk_manager import RiskManager
from shared.signal_router import SignalRouter
from .kalshi_client import KalshiClient, KalshiOrder

log = structlog.get_logger(__name__)
console = Console()


class KalshiBot:
    def __init__(self) -> None:
        cfg = get_settings()
        self._cfg = cfg

        self._client = KalshiClient(
            api_key_id=cfg.kalshi.api_key_id,
            private_key_path=cfg.kalshi.private_key_path,
            base_url=cfg.kalshi.base_url,
            mock_mode=cfg.kalshi.mock_mode,
        )

        self._news = NewsFetcher(
            tavily_key=cfg.news.tavily_api_key,
            newsapi_key=cfg.news.newsapi_key,
            max_articles=cfg.news.max_articles_per_query,
            cache_ttl=cfg.news.cache_ttl_seconds,
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

        self._risk = RiskManager(
            max_position_usd=cfg.risk.max_position_size_usd,
            max_portfolio_usd=cfg.risk.max_portfolio_exposure_usd,
            daily_loss_limit_usd=cfg.risk.daily_loss_limit_usd,
            stop_loss_pct=cfg.risk.stop_loss_pct,
            take_profit_pct=cfg.risk.take_profit_pct,
            min_edge_pct=cfg.risk.min_edge_pct,
            min_confidence=cfg.risk.min_confidence_score,
        )

        self._router = SignalRouter(
            risk_manager=self._risk,
            kalshi_executor=self._execute_trade,
            bankroll_usd=1000.0,
        )

    async def _execute_trade(
        self, market_id: str, direction: str, size_usd: float, price: float
    ) -> None:
        # Kalshi counts contracts ($1 each); price in cents
        count = max(1, int(size_usd))
        limit_price_cents = int(round(price * 100))

        order = KalshiOrder(
            ticker=market_id,
            action="buy",
            side=direction.lower(),  # "yes" | "no"
            count=count,
            limit_price=limit_price_cents,
            client_order_id=f"bot_{int(time.time())}",
        )
        await self._client.place_order(order)

    async def run_cycle(self) -> None:
        balance = await self._client.get_balance()
        if balance > 10:
            self._router.update_bankroll(balance)

        markets = await self._client.get_markets(
            limit=self._cfg.kalshi.max_markets_per_cycle
        )

        batch_input = [m.to_analysis_dict() for m in markets]
        analyses = await self._analyzer.analyze_batch(batch_input)

        actionable = [a for a in analyses if a.signal != "HOLD"]
        self._print_summary(analyses, actionable)

        for analysis in actionable:
            await self._router.route(analysis)

        # Monitor stop-losses
        price_map = {m.ticker: m.mid_price for m in markets}
        exits = self._risk.scan_stop_losses(price_map)
        for ticker in exits:
            market = next((m for m in markets if m.ticker == ticker), None)
            if market:
                await self._execute_exit(ticker, market.mid_price)

    async def _execute_exit(self, ticker: str, price: float) -> None:
        pos = self._risk._positions.get(ticker)
        if not pos:
            return
        count = max(1, int(pos.size_usd))
        limit_cents = int(round(price * 100))
        sell_order = KalshiOrder(
            ticker=ticker,
            action="sell",
            side=pos.direction.lower(),
            count=count,
            limit_price=limit_cents,
        )
        await self._client.place_order(sell_order)
        self._risk.close_position(ticker, price)

    def _print_summary(self, all_analyses, actionable) -> None:
        table = Table(title="Kalshi Analysis", show_lines=True)
        table.add_column("Title", max_width=45)
        table.add_column("Mkt%", justify="right")
        table.add_column("Est%", justify="right")
        table.add_column("Edge", justify="right")
        table.add_column("Conf", justify="right")
        table.add_column("Signal", justify="center")

        for a in sorted(all_analyses, key=lambda x: abs(x.edge), reverse=True)[:10]:
            color = "green" if a.signal == "BUY_YES" else ("red" if a.signal == "BUY_NO" else "white")
            table.add_row(
                a.question[:45],
                f"{a.market_price:.0%}",
                f"{a.final_probability:.0%}",
                f"{a.edge:+.1%}",
                f"{a.final_confidence:.0%}",
                f"[{color}]{a.signal}[/{color}]",
            )
        console.print(table)
        console.print(f"  Actionable signals: {len(actionable)} / {len(all_analyses)}")
        console.print(f"  Portfolio: {self._risk.portfolio_summary()}")

    async def run(self, interval_seconds: int = 60) -> None:
        console.print("[bold blue]Kalshi Bot starting...[/bold blue]")
        console.print(f"  Mock mode: {self._client.mock_mode}")
        console.print(f"  Base URL: {self._client._base}")
        while True:
            try:
                await self.run_cycle()
            except Exception as exc:
                log.error("kalshi_cycle_error", error=str(exc))
            await asyncio.sleep(interval_seconds)

    async def close(self) -> None:
        await self._client.close()
        await self._news.close()


async def main() -> None:
    bot = KalshiBot()
    try:
        await bot.run(interval_seconds=get_settings().bot.cycle_interval_seconds)
    finally:
        await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
