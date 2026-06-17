"""Polymarket-only bot runner."""
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
from shared.predictive_models import OrderBookSnapshot
from shared.risk_manager import RiskManager
from shared.signal_router import SignalRouter
from .polymarket_client import ClobOrder, PolymarketClient

log = structlog.get_logger(__name__)
console = Console()


async def execute_poly_trade(
    market_id: str, direction: str, size_usd: float, price: float
) -> None:
    # This is injected at runtime; here as type reference only
    raise NotImplementedError


class PolymarketBot:
    def __init__(self) -> None:
        cfg = get_settings()
        self._cfg = cfg

        self._client = PolymarketClient(
            private_key=cfg.polymarket.private_key,
            api_key=cfg.polymarket.api_key,
            api_secret=cfg.polymarket.api_secret,
            api_passphrase=cfg.polymarket.api_passphrase,
            clob_host=cfg.polymarket.clob_host,
            gamma_host=cfg.polymarket.gamma_host,
            chain_id=cfg.polymarket.chain_id,
            mock_mode=cfg.polymarket.mock_mode,
            signature_type=cfg.polymarket.signature_type,
            funder=cfg.polymarket.funder,
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
            poly_executor=self._execute_trade,
            bankroll_usd=1000.0,
        )

    # Max slippage above mid price before rejecting the order
    _MAX_SLIPPAGE = 0.03  # 3%

    async def _execute_trade(
        self, market_id: str, direction: str, size_usd: float, price: float
    ) -> None:
        market = await self._client.get_market(market_id)
        if not market:
            raise ValueError(f"Market {market_id} not found")

        token_id = market.yes_token_id if direction == "YES" else market.no_token_id

        # `price` is the YES mid price. For a NO buy the held token is (1 - YES).
        mid_token_price = price if direction == "YES" else (1.0 - price)
        mid_token_price = min(0.99, max(0.01, mid_token_price))

        # Fetch real order book to get the true best ask (BUY) limit price.
        limit_price = mid_token_price  # fallback if book is unavailable
        if not self._client.mock_mode:
            book = await self._client.get_order_book(token_id)
            asks = book.get("asks", [])
            bids = book.get("bids", [])
            if direction == "YES" or direction == "BUY":
                best_side = asks
            else:
                best_side = bids

            # asks are sorted ascending, bids descending; best is first entry.
            if best_side:
                try:
                    best_px = float(best_side[0].get("price", mid_token_price))
                    slippage = abs(best_px - mid_token_price) / mid_token_price
                    if slippage > self._MAX_SLIPPAGE:
                        raise ValueError(
                            f"Order book slippage {slippage:.1%} exceeds cap "
                            f"{self._MAX_SLIPPAGE:.0%} for {market_id} "
                            f"(best={best_px:.3f}, mid={mid_token_price:.3f})"
                        )
                    limit_price = best_px
                except (TypeError, KeyError):
                    pass  # malformed book entry — keep mid fallback

        limit_price = min(0.99, max(0.01, limit_price))
        shares = size_usd / limit_price if limit_price > 0 else 0

        order = ClobOrder(
            condition_id=market_id,
            token_id=token_id,
            side="BUY",
            size=round(shares, 4),
            price=round(limit_price, 4),
        )
        await self._client.place_order(order)

    async def run_cycle(self) -> None:
        balance = await self._client.get_balance()
        if balance > 0:
            self._router.update_bankroll(balance)

        markets = await self._client.get_markets(
            limit=self._cfg.polymarket.max_markets_per_cycle
        )

        # Prepare batch input
        batch_input = [m.to_analysis_dict() for m in markets]
        analyses = await self._analyzer.analyze_batch(batch_input)

        actionable = [a for a in analyses if a.signal != "HOLD"]
        self._print_summary(analyses, actionable)

        for analysis in actionable:
            await self._router.route(analysis)

        # Monitor stop-losses
        price_map = {
            m.condition_id: m.mid_price
            for m in markets
        }
        exits = self._risk.scan_stop_losses(price_map)
        for market_id in exits:
            market = next((m for m in markets if m.condition_id == market_id), None)
            if market:
                await self._execute_exit(market_id, market.mid_price)

    async def _execute_exit(self, market_id: str, price: float) -> None:
        pos = self._risk.get_position(market_id)
        if not pos:
            return
        market = await self._client.get_market(market_id)
        if not market:
            return
        token_id = market.yes_token_id if pos.direction == "YES" else market.no_token_id
        # Sell the contracts we hold: sized as size_usd / entry-token-price.
        entry_token_price = pos.entry_price if pos.direction == "YES" else (1.0 - pos.entry_price)
        entry_token_price = min(0.99, max(0.01, entry_token_price))
        shares = pos.size_usd / entry_token_price if entry_token_price > 0 else 0
        exit_token_price = price if pos.direction == "YES" else (1.0 - price)
        exit_token_price = min(0.99, max(0.01, exit_token_price))
        order = ClobOrder(
            condition_id=market_id,
            token_id=token_id,
            side="SELL",
            size=round(shares, 4),
            price=round(exit_token_price, 4),
        )
        await self._client.place_order(order)
        self._risk.close_position(market_id, price)

    def _print_summary(self, all_analyses, actionable) -> None:
        table = Table(title="Polymarket Analysis", show_lines=True)
        table.add_column("Question", max_width=45)
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
        console.print("[bold green]Polymarket Bot starting...[/bold green]")
        console.print(f"  Mock mode: {self._client.mock_mode}")
        while True:
            try:
                await self.run_cycle()
            except Exception as exc:
                log.error("cycle_error", error=str(exc))
            await asyncio.sleep(interval_seconds)

    async def close(self) -> None:
        await self._client.close()
        await self._news.close()


async def main() -> None:
    bot = PolymarketBot()
    try:
        await bot.run(interval_seconds=get_settings().bot.cycle_interval_seconds)
    finally:
        await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
