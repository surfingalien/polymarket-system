"""
MarketAnalyzer — orchestrates all predictive models into a single
analysis pipeline for a given market.

Pipeline:
  1. Fetch news
  2. Call Claude AI
  3. Run Bayesian estimator
  4. Analyze order book microstructure
  5. Compute momentum signals from price history
  6. Apply time-to-resolution decay
  7. Ensemble all signals → final probability + Kelly fraction
  8. Record for calibration
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import structlog

from .claude_agent import ClaudeAgent, MarketAnalysis
from .news_fetcher import NewsFetcher
from .predictive_models import (
    BayesianEstimator,
    CalibrationTracker,
    CrossMarketCorrelator,
    EnsemblePredictor,
    EnsembleResult,
    KellyCriterion,
    MomentumAnalyzer,
    OrderBookAnalyzer,
    OrderBookSnapshot,
    PricePoint,
    ResolutionDecayModel,
    SentimentConverter,
)

log = structlog.get_logger(__name__)


@dataclass
class FullMarketAnalysis:
    market_id: str
    question: str
    platform: str

    # Raw inputs
    market_price: float
    volume_usd: float = 0.0
    liquidity_usd: float = 0.0
    days_to_resolution: float = 30.0

    # AI layer
    ai_analysis: Optional[MarketAnalysis] = None

    # Ensemble result
    ensemble: Optional[EnsembleResult] = None

    # Final recommendation
    signal: str = "HOLD"              # BUY_YES | BUY_NO | HOLD
    final_probability: float = 0.0
    final_confidence: float = 0.0
    kelly_fraction: float = 0.0
    edge: float = 0.0

    analyzed_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.ensemble:
            self.final_probability = self.ensemble.estimated_probability
            self.final_confidence = self.ensemble.confidence
            self.kelly_fraction = self.ensemble.kelly_fraction
            self.edge = self.ensemble.edge
            if not self.ensemble.has_edge or self.final_confidence < 0.5:
                self.signal = "HOLD"
            elif self.edge > 0:
                self.signal = "BUY_YES"
            else:
                self.signal = "BUY_NO"
        elif self.ai_analysis:
            self.final_probability = self.ai_analysis.estimated_probability
            self.final_confidence = self.ai_analysis.confidence
            self.edge = self.ai_analysis.edge
            self.signal = self.ai_analysis.signal

    def summary(self) -> str:
        return (
            f"[{self.platform}] {self.question[:60]} | "
            f"mkt={self.market_price:.0%} est={self.final_probability:.0%} "
            f"edge={self.edge:+.1%} conf={self.final_confidence:.0%} → {self.signal}"
        )


class MarketAnalyzer:
    """Full analysis pipeline combining AI + statistical models."""

    def __init__(
        self,
        claude: ClaudeAgent,
        news: NewsFetcher,
        kelly_max_fraction: float = 0.25,
    ) -> None:
        self._claude = claude
        self._news = news
        self._bayesian = BayesianEstimator()
        self._kelly = KellyCriterion(max_fraction=kelly_max_fraction)
        self._ob_analyzer = OrderBookAnalyzer()
        self._momentum = MomentumAnalyzer()
        self._sentiment_conv = SentimentConverter()
        self._ensemble = EnsemblePredictor(self._kelly)
        self._decay = ResolutionDecayModel()
        self._calibration = CalibrationTracker()
        self._correlator = CrossMarketCorrelator()

    async def analyze(
        self,
        market_id: str,
        question: str,
        platform: str,
        market_price: float,
        volume_usd: float = 0.0,
        liquidity_usd: float = 0.0,
        days_to_resolution: float = 30.0,
        order_book: Optional[OrderBookSnapshot] = None,
        price_history: Optional[Sequence[PricePoint]] = None,
        news_query: Optional[str] = None,
    ) -> FullMarketAnalysis:

        result = FullMarketAnalysis(
            market_id=market_id,
            question=question,
            platform=platform,
            market_price=market_price,
            volume_usd=volume_usd,
            liquidity_usd=liquidity_usd,
            days_to_resolution=days_to_resolution,
        )

        # 1. News
        query = news_query or question[:80]
        news_result = await self._news.fetch(query)

        # 2. Claude AI analysis
        ai = await self._claude.analyze_market(
            market_id=market_id,
            question=question,
            market_price=market_price,
            news_summary=news_result.summary,
        )
        result.ai_analysis = ai

        # 3. Bayesian estimate — use news sentiment + AI as evidence
        evidence: list[tuple[float, float]] = []
        if ai.estimated_probability != market_price:
            ai_strength = (ai.estimated_probability - market_price) * 2
            evidence.append((ai_strength, ai.confidence))
        if news_result.articles:
            evidence.append((news_result.sentiment_score, 0.5))

        bay_mean, bay_lo, bay_hi = self._bayesian.estimate(market_price, evidence)

        # 4. Microstructure signals
        ob_signals = self._ob_analyzer.analyze(order_book) if order_book else []

        # 5. Momentum signals
        mom_signals = self._momentum.analyze(price_history) if price_history else []

        # 6. Sentiment signal
        sentiment_sig = self._sentiment_conv.to_signal(
            news_result.sentiment_score, len(news_result.articles)
        )

        # 7. Ensemble
        raw_ensemble = self._ensemble.predict(
            market_price=market_price,
            ai_probability=ai.estimated_probability,
            ai_confidence=ai.confidence,
            bayesian_estimate=bay_mean,
            microstructure_signals=ob_signals,
            momentum_signals=mom_signals,
            sentiment_signal=sentiment_sig,
        )

        # 8. Time-to-resolution decay
        adj_conf, adj_prob = self._decay.adjust_for_time(
            days_to_resolution,
            raw_ensemble.confidence,
            raw_ensemble.estimated_probability,
        )

        from .predictive_models import EnsembleResult as ER
        final_ensemble = ER(
            raw_probability=raw_ensemble.raw_probability,
            estimated_probability=adj_prob,
            edge=adj_prob - market_price,
            kelly_fraction=self._kelly.compute(adj_prob, market_price),
            confidence=adj_conf,
            signals=raw_ensemble.signals,
        )

        result.ensemble = final_ensemble
        result.__post_init__()

        # Record for calibration
        self._calibration.record(market_id, adj_prob)

        log.info(
            "full_analysis_complete",
            market_id=market_id,
            signal=result.signal,
            edge=round(result.edge, 3),
            confidence=round(result.final_confidence, 2),
            kelly=round(result.kelly_fraction, 3),
            bayesian_mean=round(bay_mean, 3),
        )
        return result

    async def analyze_batch(
        self,
        markets: list[dict],
        batch_size: int = 5,
    ) -> list[FullMarketAnalysis]:
        """
        Lightweight batch analysis — uses Claude batch endpoint, skips
        order book and price history for speed.
        """
        # Prepare Claude batch input
        claude_input = [
            {
                "id": m["id"],
                "question": m.get("question", ""),
                "price": float(m.get("price", 0.5)),
                "news_summary": m.get("news_summary", ""),
            }
            for m in markets
        ]
        ai_analyses = await self._claude.analyze_batch(claude_input, batch_size)
        ai_map = {a.market_id: a for a in ai_analyses}

        results = []
        for m in markets:
            ai = ai_map.get(m["id"])
            price = float(m.get("price", 0.5))

            if ai:
                bay_mean, _, _ = self._bayesian.estimate(
                    price, [(ai.estimated_probability - price, ai.confidence)]
                )
                ens = self._ensemble.predict(
                    market_price=price,
                    ai_probability=ai.estimated_probability,
                    ai_confidence=ai.confidence,
                    bayesian_estimate=bay_mean,
                    microstructure_signals=[],
                    momentum_signals=[],
                    sentiment_signal=None,
                )
            else:
                ens = None

            r = FullMarketAnalysis(
                market_id=m["id"],
                question=m.get("question", ""),
                platform=m.get("platform", "unknown"),
                market_price=price,
                volume_usd=float(m.get("volume", 0)),
                liquidity_usd=float(m.get("liquidity", 0)),
                days_to_resolution=float(m.get("days_to_resolution", 30)),
                ai_analysis=ai,
                ensemble=ens,
            )
            r.__post_init__()
            results.append(r)

        return results

    def find_arbitrage(
        self,
        poly_markets: list[dict],
        kalshi_markets: list[dict],
        min_spread: float = 0.03,
    ) -> list:
        return self._correlator.find_arbitrage(poly_markets, kalshi_markets, min_spread)

    def calibration_stats(self) -> dict:
        return self._calibration.accuracy_stats()

    def resolve_market(self, market_id: str, outcome: float) -> None:
        self._calibration.resolve(market_id, outcome)

    def update_signal_weight(self, signal_name: str, performance_delta: float) -> None:
        self._ensemble.update_weights(signal_name, performance_delta)
