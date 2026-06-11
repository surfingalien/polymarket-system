"""Typed configuration for Polymarket-Kalshi AI bot."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PolymarketSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POLY_", env_file=".env", extra="ignore")

    private_key: str = Field(default="", description="EOA private key (0x...)")
    api_key: str = Field(default="", description="CLOB API key")
    api_secret: str = Field(default="", description="CLOB API secret")
    api_passphrase: str = Field(default="", description="CLOB API passphrase")
    clob_host: str = Field(default="https://clob.polymarket.com")
    gamma_host: str = Field(default="https://gamma-api.polymarket.com")
    chain_id: int = Field(default=137, description="137=Polygon mainnet, 80002=Amoy testnet")
    mock_mode: bool = Field(default=True)
    max_markets_per_cycle: int = Field(default=20)


class KalshiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KALSHI_", env_file=".env", extra="ignore")

    api_key_id: str = Field(default="", description="Key ID from Kalshi dashboard")
    private_key_path: str = Field(default="keys/kalshi_private.pem")
    api_host: str = Field(default="https://trading-api.kalshi.com")
    demo_mode: bool = Field(default=True)
    mock_mode: bool = Field(default=True)
    max_markets_per_cycle: int = Field(default=20)

    @property
    def base_url(self) -> str:
        if self.demo_mode:
            return "https://demo-api.kalshi.co/trade-api/v2"
        return f"{self.api_host}/trade-api/v2"


class ClaudeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLAUDE_", env_file=".env", extra="ignore")

    api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    model: str = Field(default="claude-opus-4-8")
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.2)
    batch_size: int = Field(default=5, description="Markets to analyze per AI call")
    cache_ttl_seconds: int = Field(default=300)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("api_key", mode="before")
    @classmethod
    def load_anthropic_key(cls, v: str) -> str:
        return v or os.getenv("ANTHROPIC_API_KEY", "")


class RiskSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RISK_", env_file=".env", extra="ignore")

    max_position_size_usd: float = Field(default=50.0)
    max_portfolio_exposure_usd: float = Field(default=500.0)
    daily_loss_limit_usd: float = Field(default=100.0)
    stop_loss_pct: float = Field(default=0.30, description="30% loss triggers stop")
    take_profit_pct: float = Field(default=0.80, description="80% gain triggers take-profit")
    min_edge_pct: float = Field(default=0.05, description="Min 5% edge required")
    max_kelly_fraction: float = Field(default=0.25, description="Never bet more than 25% Kelly")
    min_market_liquidity_usd: float = Field(default=1000.0)
    min_market_volume_usd: float = Field(default=5000.0)
    cooldown_after_loss_hours: float = Field(default=2.0)
    max_concurrent_positions: int = Field(default=10)
    min_confidence_score: float = Field(default=0.65, description="0-1 AI confidence threshold")


class NewsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEWS_", env_file=".env", extra="ignore")

    tavily_api_key: str = Field(default="")
    newsapi_key: str = Field(default="")
    max_articles_per_query: int = Field(default=5)
    cache_ttl_seconds: int = Field(default=600)


class BotSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mock_mode: bool = Field(default=True, description="Global mock override")
    cycle_interval_seconds: int = Field(default=60)
    log_level: str = Field(default="INFO")
    db_path: str = Field(default="data/bot.db")
    enable_arbitrage: bool = Field(default=True)
    enable_polymarket: bool = Field(default=True)
    enable_kalshi: bool = Field(default=True)
    arbitrage_min_spread_pct: float = Field(default=0.03)


class Settings:
    """Aggregate settings container."""

    def __init__(self) -> None:
        self.polymarket = PolymarketSettings()
        self.kalshi = KalshiSettings()
        self.claude = ClaudeSettings()
        self.risk = RiskSettings()
        self.news = NewsSettings()
        self.bot = BotSettings()

        # Global mock mode override
        global_mock = os.getenv("MOCK_MODE", "true").lower() in ("true", "1", "yes")
        if global_mock:
            self.polymarket.mock_mode = True
            self.kalshi.mock_mode = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
