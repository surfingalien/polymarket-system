"""
Claude AI agent — probability estimation, batch analysis, and arbitrage research.

Uses claude-opus-4-8 for highest accuracy on prediction market analysis.
Implements response caching, structured JSON output, and confidence scoring.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic
import structlog

log = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are an expert prediction market analyst specializing in probability estimation.

Your job is to estimate the TRUE probability that a market resolves YES, given:
- The market question
- Current market price (implied probability)
- Recent news headlines and sentiment
- Any additional context

Guidelines:
- Base your estimate on evidence, not gut feeling
- Consider base rates, current trends, and information asymmetry
- Markets often under-react to strong evidence; sometimes over-react to noise
- Account for resolution criteria carefully (exact wording matters)
- Be calibrated: 70% confidence means it should happen 70% of the time

Output ONLY valid JSON. No markdown, no explanation outside the JSON.

Required format:
{
  "estimated_probability": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<2-3 sentence summary>",
  "key_factors": ["<factor1>", "<factor2>"],
  "uncertainty_flags": ["<flag1>"] // risks that could flip the outcome
}"""

_BATCH_SYSTEM_PROMPT = """You are an expert prediction market analyst.

Analyze each market and output a JSON array with one object per market in the same order.

Required format for each object:
{
  "market_id": "<id>",
  "estimated_probability": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<1-2 sentence summary>",
  "signal": "BUY_YES" | "BUY_NO" | "HOLD"
}

Output ONLY the JSON array. No markdown fences."""


@dataclass
class MarketAnalysis:
    market_id: str
    question: str
    market_price: float
    estimated_probability: float
    confidence: float
    reasoning: str
    key_factors: list[str] = field(default_factory=list)
    uncertainty_flags: list[str] = field(default_factory=list)
    signal: str = "HOLD"   # BUY_YES | BUY_NO | HOLD
    edge: float = 0.0
    cached: bool = False
    analyzed_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.edge = self.estimated_probability - self.market_price
        if abs(self.edge) < 0.04:
            self.signal = "HOLD"
        elif self.edge > 0:
            self.signal = "BUY_YES"
        else:
            self.signal = "BUY_NO"


class ClaudeAgent:
    """AI brain that calls Claude for market probability estimation."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-8",
        max_tokens: int = 1024,
        cache_ttl: int = 300,
        min_edge_for_signal: float = 0.04,
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._cache_ttl = cache_ttl
        self._min_edge = min_edge_for_signal
        self._cache: dict[str, tuple[MarketAnalysis, float]] = {}

    # ------------------------------------------------------------------
    # Single-market analysis
    # ------------------------------------------------------------------

    async def analyze_market(
        self,
        market_id: str,
        question: str,
        market_price: float,
        news_summary: str = "",
        extra_context: str = "",
    ) -> MarketAnalysis:
        cache_key = self._cache_key(market_id, market_price, news_summary)
        cached = self._get_cached(cache_key)
        if cached:
            cached.cached = True
            return cached

        prompt = self._build_single_prompt(question, market_price, news_summary, extra_context)

        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            data = self._parse_json(raw)

            analysis = MarketAnalysis(
                market_id=market_id,
                question=question,
                market_price=market_price,
                estimated_probability=float(data.get("estimated_probability", market_price)),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=str(data.get("reasoning", "")),
                key_factors=list(data.get("key_factors", [])),
                uncertainty_flags=list(data.get("uncertainty_flags", [])),
            )
            self._cache[cache_key] = (analysis, time.time())
            log.info(
                "market_analyzed",
                market_id=market_id,
                edge=round(analysis.edge, 3),
                signal=analysis.signal,
                confidence=round(analysis.confidence, 2),
            )
            return analysis

        except Exception as exc:
            log.error("claude_analysis_failed", market_id=market_id, error=str(exc))
            return self._fallback(market_id, question, market_price)

    # ------------------------------------------------------------------
    # Batch analysis (cheaper per-market)
    # ------------------------------------------------------------------

    async def analyze_batch(
        self,
        markets: list[dict],  # each: {id, question, price, news_summary?}
        batch_size: int = 5,
    ) -> list[MarketAnalysis]:
        results: list[MarketAnalysis] = []
        for i in range(0, len(markets), batch_size):
            batch = markets[i : i + batch_size]
            batch_results = await self._analyze_batch_chunk(batch)
            results.extend(batch_results)
            if i + batch_size < len(markets):
                await asyncio.sleep(0.5)  # respect rate limits
        return results

    async def _analyze_batch_chunk(self, markets: list[dict]) -> list[MarketAnalysis]:
        # Return cached items without API call
        uncached_indices: list[int] = []
        results: list[Optional[MarketAnalysis]] = [None] * len(markets)

        for idx, m in enumerate(markets):
            ck = self._cache_key(m["id"], m.get("price", 0.5), m.get("news_summary", ""))
            cached = self._get_cached(ck)
            if cached:
                cached.cached = True
                results[idx] = cached
            else:
                uncached_indices.append(idx)

        if not uncached_indices:
            return [r for r in results if r is not None]

        uncached = [markets[i] for i in uncached_indices]
        prompt = self._build_batch_prompt(uncached)

        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens * 2,
                system=_BATCH_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            parsed = self._parse_json(raw)
            if not isinstance(parsed, list):
                parsed = [parsed]

            for idx_offset, (orig_idx, m) in enumerate(
                zip(uncached_indices, uncached)
            ):
                data = parsed[idx_offset] if idx_offset < len(parsed) else {}
                analysis = MarketAnalysis(
                    market_id=m["id"],
                    question=m.get("question", ""),
                    market_price=float(m.get("price", 0.5)),
                    estimated_probability=float(
                        data.get("estimated_probability", m.get("price", 0.5))
                    ),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=str(data.get("reasoning", "")),
                )
                ck = self._cache_key(m["id"], float(m.get("price", 0.5)), m.get("news_summary", ""))
                self._cache[ck] = (analysis, time.time())
                results[orig_idx] = analysis

        except Exception as exc:
            log.error("batch_analysis_failed", error=str(exc))
            for orig_idx, m in zip(uncached_indices, uncached):
                results[orig_idx] = self._fallback(
                    m["id"], m.get("question", ""), float(m.get("price", 0.5))
                )

        return [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # Arbitrage research
    # ------------------------------------------------------------------

    async def research_arbitrage(
        self,
        poly_question: str,
        kalshi_question: str,
        poly_price: float,
        kalshi_price: float,
    ) -> dict:
        prompt = (
            f"Two prediction markets appear to cover the same event:\n\n"
            f"Polymarket: '{poly_question}' — current YES price: {poly_price:.0%}\n"
            f"Kalshi: '{kalshi_question}' — current YES price: {kalshi_price:.0%}\n\n"
            f"Price difference: {abs(poly_price - kalshi_price):.1%}\n\n"
            f"Analyze:\n"
            f"1. Are these markets truly equivalent (same resolution criteria)?\n"
            f"2. If yes, which side is mispriced and why?\n"
            f"3. Estimate the true probability.\n"
            f"4. Rate your confidence in the arbitrage (0-1).\n\n"
            f"Output JSON: {{\"equivalent\": bool, \"true_probability\": float, "
            f"\"mispriced_side\": \"poly\"|\"kalshi\"|\"neither\", "
            f"\"confidence\": float, \"reasoning\": str}}"
        )
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return self._parse_json(resp.content[0].text.strip())
        except Exception as exc:
            log.warning("arb_research_failed", error=str(exc))
            return {"equivalent": False, "confidence": 0.0, "reasoning": str(exc)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_single_prompt(
        self,
        question: str,
        price: float,
        news: str,
        extra: str,
    ) -> str:
        parts = [
            f"Market question: {question}",
            f"Current market price (implied YES probability): {price:.1%}",
        ]
        if news:
            parts.append(f"Recent news:\n{news}")
        if extra:
            parts.append(f"Additional context:\n{extra}")
        parts.append("What is the true probability this resolves YES?")
        return "\n\n".join(parts)

    def _build_batch_prompt(self, markets: list[dict]) -> str:
        items = []
        for m in markets:
            items.append(
                f"ID: {m['id']}\n"
                f"Question: {m.get('question', '')}\n"
                f"Market price: {float(m.get('price', 0.5)):.1%}\n"
                f"News: {m.get('news_summary', 'None')}"
            )
        return "Analyze these markets:\n\n" + "\n\n---\n\n".join(items)

    def _parse_json(self, text: str) -> dict | list:
        text = text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                l for l in lines if not l.startswith("```")
            ).strip()
        return json.loads(text)

    def _cache_key(self, market_id: str, price: float, news: str) -> str:
        raw = f"{market_id}:{price:.3f}:{news[:100]}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _get_cached(self, key: str) -> Optional[MarketAnalysis]:
        entry = self._cache.get(key)
        if entry and (time.time() - entry[1]) < self._cache_ttl:
            return entry[0]
        if entry:
            del self._cache[key]
        return None

    def _fallback(self, market_id: str, question: str, price: float) -> MarketAnalysis:
        return MarketAnalysis(
            market_id=market_id,
            question=question,
            market_price=price,
            estimated_probability=price,
            confidence=0.1,
            reasoning="Analysis unavailable — using market price as estimate.",
        )

    def cache_stats(self) -> dict:
        now = time.time()
        live = sum(1 for _, ts in self._cache.values() if now - ts < self._cache_ttl)
        return {"total": len(self._cache), "live": live}
