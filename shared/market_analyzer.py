"""
MarketAnalyzer — orchestrates all predictive models into a single
analysis pipeline for a given market.

Pipeline:
  1. Expand query + fetch news (two-pass)
  2. Call Claude AI
  3. Run Bayesian estimator
  4. Analyze order book microstructure
  5. Compute momentum signals from price history
  6. Apply longshot bias correction
  7. Apply category-specific edge adjustments
  8. Apply hour-of-day temporal efficiency adjustment
  9. Market quality scoring (volume + liquidity + movement)
  10. Ensemble all signals → final probability + Kelly fraction
  11. Time-to-resolution decay
  12. Record for calibration + excess-return tracking
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import structlog

from .advanced_signals import (
    CategoryEdgeModel,
    ExcessReturnTracker,
    LongshotBiasDetector,
    MarketQualityScorer,
    OrderFlowAnalyzer,
    QueryExpander,
    SentimentDivergenceSignal,
    TemporalPatternSignal,
)
from .claude_agent import ClaudeAgent, MarketAnalysis
from .learning_engine import LearningEngine
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
    Signal,
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
    category: str = "default"
    market_quality: float = 0.0

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
    # Category- and hour-adjusted edge requirement (sports markets need less
    # edge than finance; efficient hours need more).
    min_edge_threshold: float = 0.03

    analyzed_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.ensemble:
            self.final_probability = self.ensemble.estimated_probability
            self.final_confidence = self.ensemble.confidence
            # Confidence-scaled Kelly: the point-estimate Kelly fraction ignores
            # model uncertainty, so a barely-confident edge and a high-confidence
            # edge of equal size would bet the same. Shrink the bet by the
            # model's own confidence so uncertain signals deploy less capital.
            self.kelly_fraction = self.ensemble.kelly_fraction * self.ensemble.confidence
            self.edge = self.ensemble.edge
            if abs(self.edge) < self.min_edge_threshold or self.final_confidence < 0.5:
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
            f"[{self.platform}/{self.category}] {self.question[:55]} | "
            f"mkt={self.market_price:.0%} est={self.final_probability:.0%} "
            f"edge={self.edge:+.1%} conf={self.final_confidence:.0%} "
            f"q={self.market_quality:.2f} → {self.signal}"
        )


class MarketAnalyzer:
    """Full analysis pipeline combining AI + statistical + microstructure models."""

    def __init__(
        self,
        claude: ClaudeAgent,
        news: NewsFetcher,
        kelly_max_fraction: float = 0.25,
        learning_state_file: str = "data/learning_state.json",
    ) -> None:
        self._claude = claude
        self._news = news

        # Core predictive models
        self._bayesian = BayesianEstimator()
        self._kelly = KellyCriterion(max_fraction=kelly_max_fraction)
        self._ob_analyzer = OrderBookAnalyzer()
        self._momentum = MomentumAnalyzer()
        self._sentiment_conv = SentimentConverter()
        self._ensemble = EnsemblePredictor(self._kelly)
        self._decay = ResolutionDecayModel()
        self._calibration = CalibrationTracker()
        self._correlator = CrossMarketCorrelator()

        # Self-learning brain — loads persisted weights on startup
        from pathlib import Path
        self.learning = LearningEngine(
            self._ensemble, state_file=Path(learning_state_file)
        )
        self.learning.load()

        # Advanced signals (research-derived)
        self._longshot = LongshotBiasDetector()
        # One order-flow analyzer per market: trades in one market must not
        # generate signals for unrelated markets.
        self._order_flow: dict[str, OrderFlowAnalyzer] = {}
        self._temporal = TemporalPatternSignal()
        self._category = CategoryEdgeModel()
        self._quality = MarketQualityScorer()
        self._excess_return = ExcessReturnTracker()
        self._divergence = SentimentDivergenceSignal()
        self._query_expander = QueryExpander()

    # ------------------------------------------------------------------
    # Full single-market analysis
    # ------------------------------------------------------------------

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
        price_change_pct: float = 0.0,
    ) -> FullMarketAnalysis:

        category = self._category.detect_category(question)
        quality_score = self._quality.score(
            volume_usd, liquidity_usd, price_change_pct, market_price
        )

        result = FullMarketAnalysis(
            market_id=market_id,
            question=question,
            platform=platform,
            market_price=market_price,
            volume_usd=volume_usd,
            liquidity_usd=liquidity_usd,
            days_to_resolution=days_to_resolution,
            category=category,
            market_quality=quality_score,
        )

        # 1. Expand query + fetch news (two-pass: primary query + key terms)
        queries = self._query_expander.expand(news_query or question, max_queries=3)
        primary_news = await self._news.fetch(queries[0])
        secondary_sentiment = 0.0
        if len(queries) > 1:
            secondary_news = await self._news.fetch(queries[1])
            secondary_sentiment = secondary_news.sentiment_score

        # 2. Claude AI analysis
        ai = await self._claude.analyze_market(
            market_id=market_id,
            question=question,
            market_price=market_price,
            news_summary=primary_news.summary,
        )
        result.ai_analysis = ai

        # 3. Bayesian estimate
        evidence: list[tuple[float, float]] = []
        if ai.estimated_probability != market_price:
            ai_strength = (ai.estimated_probability - market_price) * 2
            evidence.append((ai_strength, ai.confidence))
        if primary_news.articles:
            evidence.append((primary_news.sentiment_score, 0.5))
        bay_mean, _, _ = self._bayesian.estimate(market_price, evidence)

        # 4. Microstructure signals
        ob_signals = self._ob_analyzer.analyze(order_book) if order_book else []

        # 5. Momentum signals
        mom_signals = self._momentum.analyze(price_history) if price_history else []

        # 6. Longshot bias signal (applied ONCE, directly in step 14 —
        #    feeding it into the ensemble too would double-count it)
        longshot_sig = self._longshot.signal(market_price)

        # 7. Order flow signal (per-market, if trades have been recorded)
        of_analyzer = self._order_flow.get(market_id)
        of_sig = of_analyzer.signal() if of_analyzer else None

        # 8–10. Neutral meta-signals — kept for display/diagnostics; they
        #     act through thresholds (step 15/16), not through the ensemble.
        temporal_sig = self._temporal.signal()
        quality_sig = self._quality.signal(volume_usd, liquidity_usd, price_change_pct, market_price)
        category_sig = self._category.signal(question, ai.edge)
        meta_signals: list[Signal] = [longshot_sig, temporal_sig, quality_sig, category_sig]

        # 11. Sentiment (primary) + dual-source divergence
        primary_sentiment_sig = self._sentiment_conv.to_signal(
            primary_news.sentiment_score, len(primary_news.articles)
        )
        directional_extras: list[Signal] = []
        if secondary_sentiment != 0.0:
            directional_extras.append(self._divergence.signal(
                primary_news.sentiment_score,
                secondary_sentiment,
                weight_a=1.0,
                weight_b=0.7,
            ))
        if of_sig:
            directional_extras.append(of_sig)

        # 12. Ensemble — only genuinely directional inputs
        raw_ensemble = self._ensemble.predict(
            market_price=market_price,
            ai_probability=ai.estimated_probability,
            ai_confidence=ai.confidence,
            bayesian_estimate=bay_mean,
            microstructure_signals=ob_signals + directional_extras,
            momentum_signals=mom_signals,
            sentiment_signal=primary_sentiment_sig,
        )

        # 13. Time-to-resolution decay (shrinks toward market price near expiry)
        adj_conf, adj_prob = self._decay.adjust_for_time(
            days_to_resolution,
            raw_ensemble.confidence,
            raw_ensemble.estimated_probability,
            market_price=market_price,
        )

        # 14. Apply longshot bias as direct probability adjustment
        adj_prob = self._longshot.adjust_probability(market_price, adj_prob)

        # 15. Scale confidence by market quality
        adj_conf = adj_conf * (0.5 + 0.5 * quality_score)

        # 16. Category- and hour-adjusted edge requirement
        min_edge = self._temporal.adjusted_min_edge(
            self._category.adjusted_min_edge(question, 0.03)
        )

        final_ensemble = EnsembleResult(
            raw_probability=raw_ensemble.raw_probability,
            estimated_probability=adj_prob,
            edge=adj_prob - market_price,
            kelly_fraction=self._kelly.compute(adj_prob, market_price),
            confidence=adj_conf,
            signals=raw_ensemble.signals + meta_signals,
        )

        result.min_edge_threshold = min_edge
        result.ensemble = final_ensemble
        result.__post_init__()

        # 17. Record for calibration + excess return
        self._calibration.record(market_id, adj_prob)
        self._excess_return.record(market_id, adj_prob)

        log.info(
            "full_analysis_complete",
            market_id=market_id,
            category=category,
            quality=round(quality_score, 2),
            signal=result.signal,
            edge=round(result.edge, 3),
            confidence=round(result.final_confidence, 2),
            kelly=round(result.kelly_fraction, 3),
            bayesian_mean=round(bay_mean, 3),
        )
        return result

    # ------------------------------------------------------------------
    # Batch analysis (AI-only, no order book/price history for speed)
    # ------------------------------------------------------------------

    async def analyze_batch(
        self,
        markets: list[dict],
        batch_size: int = 5,
    ) -> list[FullMarketAnalysis]:
        """
        Lightweight batch analysis — single Claude call per batch,
        applies longshot bias, category, quality, and temporal signals.
        """
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
            volume = float(m.get("volume", 0))
            liquidity = float(m.get("liquidity", 0))
            question = m.get("question", "")

            category = self._category.detect_category(question)
            quality_score = self._quality.score(volume, liquidity, 0.0, price)

            if ai:
                # Same evidence scaling as the full analyze() path
                bay_mean, _, _ = self._bayesian.estimate(
                    price, [((ai.estimated_probability - price) * 2, ai.confidence)]
                )

                # Meta-signals for display only — longshot is applied directly
                # below, the others act through thresholds, not the ensemble.
                meta: list[Signal] = [
                    self._longshot.signal(price),
                    self._temporal.signal(),
                    self._quality.signal(volume, liquidity, 0.0, price),
                    self._category.signal(question, ai.edge),
                ]

                ens = self._ensemble.predict(
                    market_price=price,
                    ai_probability=ai.estimated_probability,
                    ai_confidence=ai.confidence,
                    bayesian_estimate=bay_mean,
                    microstructure_signals=[],
                    momentum_signals=[],
                    sentiment_signal=None,
                )

                # Apply longshot direct adjustment + quality confidence scaling
                adj_prob = self._longshot.adjust_probability(price, ens.estimated_probability)
                adj_conf = ens.confidence * (0.5 + 0.5 * quality_score)

                ens = EnsembleResult(
                    raw_probability=ens.raw_probability,
                    estimated_probability=adj_prob,
                    edge=adj_prob - price,
                    kelly_fraction=self._kelly.compute(adj_prob, price),
                    confidence=adj_conf,
                    signals=ens.signals + meta,
                )
            else:
                ens = None

            min_edge = self._temporal.adjusted_min_edge(
                self._category.adjusted_min_edge(question, 0.03)
            )
            r = FullMarketAnalysis(
                market_id=m["id"],
                question=question,
                platform=m.get("platform", "unknown"),
                market_price=price,
                volume_usd=volume,
                liquidity_usd=liquidity,
                days_to_resolution=float(m.get("days_to_resolution", 30)),
                category=category,
                market_quality=quality_score,
                ai_analysis=ai,
                ensemble=ens,
                min_edge_threshold=min_edge,
            )
            r.__post_init__()
            if ens:
                self._calibration.record(m["id"], ens.estimated_probability)
                self._excess_return.record(m["id"], ens.estimated_probability)
            results.append(r)

        return results

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def record_trade(self, market_id: str, price: float, size_usd: float, side: str) -> None:
        """Feed live trade data into the per-market order flow analyzer."""
        if market_id not in self._order_flow:
            self._order_flow[market_id] = OrderFlowAnalyzer()
        self._order_flow[market_id].record_trade(price, size_usd, side)

    def find_arbitrage(
        self,
        poly_markets: list[dict],
        kalshi_markets: list[dict],
        min_spread: float = 0.03,
    ) -> list:
        return self._correlator.find_arbitrage(poly_markets, kalshi_markets, min_spread)

    def resolve_market(self, market_id: str, outcome: float) -> None:
        """Feed a resolved outcome back into calibration, excess-return, and the learning engine."""
        self._calibration.resolve(market_id, outcome)
        self._excess_return.resolve(market_id, outcome)
        self.learning.on_market_resolved(market_id, outcome)

    def update_signal_weight(self, signal_name: str, performance_delta: float) -> None:
        self._ensemble.update_weights(signal_name, performance_delta)

    def apply_learning_decay(self) -> None:
        """Call once per cycle to regularise learned weights back toward defaults."""
        self.learning.apply_decay()

    def save_learning_state(self) -> None:
        """Persist the evolved weights and trade memories to disk."""
        self.learning.save()

    def calibration_stats(self) -> dict:
        cal = self._calibration.accuracy_stats()
        er = self._excess_return.stats()
        learning = self.learning.performance_summary()
        return {**cal, "excess_return": er, "learning": learning}

    def market_quality(self, volume_usd: float, liquidity_usd: float) -> float:
        return self._quality.score(volume_usd, liquidity_usd)

    def is_market_tradeable(self, volume_usd: float, liquidity_usd: float) -> bool:
        return self._quality.is_tradeable(volume_usd, liquidity_usd)
