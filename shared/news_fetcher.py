"""Multi-source news aggregation with caching and deduplication."""
from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
import structlog

log = structlog.get_logger(__name__)


@dataclass
class Article:
    title: str
    description: str
    url: str
    published_at: str
    source: str
    sentiment_score: float = 0.0  # -1 bearish … +1 bullish


@dataclass
class NewsResult:
    query: str
    articles: list[Article] = field(default_factory=list)
    summary: str = ""
    fetched_at: float = field(default_factory=time.time)
    sentiment_score: float = 0.0  # aggregate

    @property
    def is_fresh(self, ttl: int = 600) -> bool:
        return (time.time() - self.fetched_at) < ttl


class NewsFetcher:
    """Fetches news from Tavily and NewsAPI, caches results."""

    def __init__(
        self,
        tavily_key: str = "",
        newsapi_key: str = "",
        max_articles: int = 5,
        cache_ttl: int = 600,
    ) -> None:
        self._tavily_key = tavily_key
        self._newsapi_key = newsapi_key
        self._max_articles = max_articles
        self._cache_ttl = cache_ttl
        self._cache: dict[str, NewsResult] = {}
        self._client = httpx.AsyncClient(timeout=15.0)

    def _cache_key(self, query: str) -> str:
        return hashlib.md5(query.lower().encode()).hexdigest()[:12]

    def _get_cached(self, query: str) -> Optional[NewsResult]:
        key = self._cache_key(query)
        result = self._cache.get(key)
        if result and (time.time() - result.fetched_at) < self._cache_ttl:
            return result
        return None

    async def fetch(self, query: str) -> NewsResult:
        cached = self._get_cached(query)
        if cached:
            return cached

        articles: list[Article] = []
        tasks = []
        if self._tavily_key:
            tasks.append(self._fetch_tavily(query))
        if self._newsapi_key:
            tasks.append(self._fetch_newsapi(query))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, list):
                    articles.extend(r)
        else:
            articles = self._mock_articles(query)

        # Deduplicate by title similarity
        articles = self._deduplicate(articles)[: self._max_articles]
        sentiment = self._aggregate_sentiment(articles)

        result = NewsResult(
            query=query,
            articles=articles,
            sentiment_score=sentiment,
            summary=self._build_summary(articles),
        )
        self._cache[self._cache_key(query)] = result
        return result

    async def _fetch_tavily(self, query: str) -> list[Article]:
        try:
            resp = await self._client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self._tavily_key,
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": self._max_articles,
                    "include_answer": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            articles = []
            for r in data.get("results", []):
                articles.append(
                    Article(
                        title=r.get("title", ""),
                        description=r.get("content", "")[:500],
                        url=r.get("url", ""),
                        published_at=r.get("published_date", ""),
                        source="tavily",
                        sentiment_score=self._score_text(r.get("content", "")),
                    )
                )
            return articles
        except Exception as exc:
            log.warning("tavily_fetch_failed", error=str(exc))
            return []

    async def _fetch_newsapi(self, query: str) -> list[Article]:
        try:
            resp = await self._client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "apiKey": self._newsapi_key,
                    "pageSize": self._max_articles,
                    "sortBy": "publishedAt",
                    "language": "en",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            articles = []
            for a in data.get("articles", []):
                text = f"{a.get('title', '')} {a.get('description', '')}"
                articles.append(
                    Article(
                        title=a.get("title", ""),
                        description=a.get("description", "") or "",
                        url=a.get("url", ""),
                        published_at=a.get("publishedAt", ""),
                        source=a.get("source", {}).get("name", "newsapi"),
                        sentiment_score=self._score_text(text),
                    )
                )
            return articles
        except Exception as exc:
            log.warning("newsapi_fetch_failed", error=str(exc))
            return []

    def _mock_articles(self, query: str) -> list[Article]:
        return [
            Article(
                title=f"[MOCK] Latest news on: {query}",
                description="Mock article — configure NEWS_TAVILY_API_KEY for real data.",
                url="https://example.com",
                published_at="",
                source="mock",
                sentiment_score=0.0,
            )
        ]

    def _score_text(self, text: str) -> float:
        """Simple lexicon-based sentiment without heavy NLP dependencies."""
        text = text.lower()
        positive = ["win", "rise", "surge", "lead", "ahead", "victory", "likely",
                    "confirm", "approve", "pass", "increase", "beat", "record"]
        negative = ["loss", "fall", "drop", "behind", "defeat", "unlikely", "fail",
                    "reject", "deny", "decrease", "miss", "collapse", "crisis"]
        score = sum(1 for w in positive if w in text) - sum(1 for w in negative if w in text)
        total = sum(1 for w in positive + negative if w in text) or 1
        return max(-1.0, min(1.0, score / total))

    def _aggregate_sentiment(self, articles: list[Article]) -> float:
        if not articles:
            return 0.0
        return sum(a.sentiment_score for a in articles) / len(articles)

    def _deduplicate(self, articles: list[Article]) -> list[Article]:
        seen: set[str] = set()
        out = []
        for a in articles:
            key = a.title[:60].lower()
            if key not in seen:
                seen.add(key)
                out.append(a)
        return out

    def _build_summary(self, articles: list[Article]) -> str:
        if not articles:
            return "No recent news found."
        lines = [f"- {a.title} ({a.source})" for a in articles[:3]]
        return "\n".join(lines)

    async def close(self) -> None:
        await self._client.aclose()
