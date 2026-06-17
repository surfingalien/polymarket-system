"""
Intra-market arbitrage detection for Polymarket.

Unlike CrossMarketCorrelator (which matches *equivalent* markets across
Polymarket and Kalshi), this finds risk-free arbitrage *within a single
Polymarket event* — pure arithmetic on live order-book asks, no model and
no AI required.

Two mechanically risk-free patterns:

1. Binary bundle
   A YES/NO market resolves to exactly $1.00 for the winning side. If you
   can buy BOTH sides for less than $1.00 total, you lock in the difference
   regardless of outcome:
       profit_per_set = 1.0 - (yes_ask + no_ask)

2. Neg-risk multi-outcome
   In an N-way event ("Who wins the World Cup?") exactly one outcome
   resolves YES and pays $1.00. If the sum of every outcome's YES ask is
   below $1.00, buying one share of each outcome guarantees a $1.00 payout
   for less than $1.00 of cost:
       profit_per_set = 1.0 - sum(yes_ask_i for i in outcomes)

Both use *ask* prices because you are a taker buying into the book. Prices
are in dollars (0.0–1.0). A configurable per-share fee buffer is subtracted
so only genuinely profitable opportunities survive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ArbLeg:
    """One side of an arbitrage trade."""
    outcome: str            # "YES" | "NO" | the outcome label for neg-risk
    ask: float              # taker price you pay (0–1)
    token_id: str = ""      # CLOB token id, if known


@dataclass
class IntraMarketArb:
    arb_type: str           # "binary_bundle" | "neg_risk"
    event_id: str           # condition_id (binary) or event/group id (neg-risk)
    question: str
    total_cost: float       # sum of asks paid to acquire one full set
    gross_profit: float     # 1.0 - total_cost  (per $1 guaranteed payout)
    net_profit: float       # gross_profit minus fee buffer
    legs: list[ArbLeg] = field(default_factory=list)
    confidence: float = 1.0  # mechanical arb → high, scaled by margin size

    @property
    def roi(self) -> float:
        """Return on capital deployed for one full set."""
        return self.net_profit / self.total_cost if self.total_cost > 0 else 0.0


class IntraMarketArbitrage:
    """
    Detects binary-bundle and neg-risk arbitrage on Polymarket order books.

    fee_per_share: flat cost (in price units) subtracted per share bought,
        covering gas / any taker fee. Polymarket's CLOB currently charges no
        explicit taker fee, so the default is a small gas buffer. Tune to your
        actual execution cost.
    min_net_profit: minimum net profit per $1 set to report (filters noise).
    """

    def __init__(
        self,
        fee_per_share: float = 0.0,
        min_net_profit: float = 0.005,
    ) -> None:
        self.fee_per_share = fee_per_share
        self.min_net_profit = min_net_profit

    # ------------------------------------------------------------------
    # Binary bundle  (YES_ask + NO_ask < 1)
    # ------------------------------------------------------------------

    def find_binary_bundle(self, markets: list[dict]) -> list[IntraMarketArb]:
        """
        markets: list of dicts, each with at least:
            id / condition_id, question, yes_ask, no_ask
            (optional) yes_token_id, no_token_id
        """
        out: list[IntraMarketArb] = []
        for m in markets:
            yes_ask = _coerce_price(m.get("yes_ask"))
            no_ask = _coerce_price(m.get("no_ask"))
            if yes_ask is None or no_ask is None:
                continue
            # asks must be live, tradeable prices
            if yes_ask <= 0 or no_ask <= 0:
                continue
            total = yes_ask + no_ask
            gross = 1.0 - total
            net = gross - 2 * self.fee_per_share  # two legs
            if net < self.min_net_profit:
                continue
            out.append(IntraMarketArb(
                arb_type="binary_bundle",
                event_id=str(m.get("condition_id") or m.get("id", "")),
                question=str(m.get("question", "")),
                total_cost=round(total, 4),
                gross_profit=round(gross, 4),
                net_profit=round(net, 4),
                legs=[
                    ArbLeg("YES", yes_ask, str(m.get("yes_token_id", ""))),
                    ArbLeg("NO", no_ask, str(m.get("no_token_id", ""))),
                ],
                confidence=_margin_confidence(net),
            ))
        return sorted(out, key=lambda o: o.net_profit, reverse=True)

    # ------------------------------------------------------------------
    # Neg-risk multi-outcome  (sum of YES_ask across outcomes < 1)
    # ------------------------------------------------------------------

    def find_neg_risk(self, events: list[dict]) -> list[IntraMarketArb]:
        """
        events: list of dicts, each representing one multi-outcome event:
            event_id, question, outcomes: [
                {label, yes_ask, token_id?}, ...
            ]
        Only events with >= 2 outcomes that all have live asks are considered.
        """
        out: list[IntraMarketArb] = []
        for ev in events:
            outcomes = ev.get("outcomes") or []
            legs: list[ArbLeg] = []
            total = 0.0
            valid = True
            for o in outcomes:
                ask = _coerce_price(o.get("yes_ask"))
                if ask is None or ask <= 0:
                    valid = False
                    break
                total += ask
                legs.append(ArbLeg(
                    str(o.get("label", "?")), ask, str(o.get("token_id", ""))
                ))
            if not valid or len(legs) < 2:
                continue
            gross = 1.0 - total
            net = gross - len(legs) * self.fee_per_share
            if net < self.min_net_profit:
                continue
            out.append(IntraMarketArb(
                arb_type="neg_risk",
                event_id=str(ev.get("event_id") or ev.get("id", "")),
                question=str(ev.get("question", "")),
                total_cost=round(total, 4),
                gross_profit=round(gross, 4),
                net_profit=round(net, 4),
                legs=legs,
                confidence=_margin_confidence(net),
            ))
        return sorted(out, key=lambda o: o.net_profit, reverse=True)

    # ------------------------------------------------------------------
    # Convenience: scan everything we can from a flat market list
    # ------------------------------------------------------------------

    def scan(
        self,
        binary_markets: Optional[list[dict]] = None,
        neg_risk_events: Optional[list[dict]] = None,
    ) -> list[IntraMarketArb]:
        results: list[IntraMarketArb] = []
        if binary_markets:
            results.extend(self.find_binary_bundle(binary_markets))
        if neg_risk_events:
            results.extend(self.find_neg_risk(neg_risk_events))
        return sorted(results, key=lambda o: o.net_profit, reverse=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _coerce_price(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    import math
    if not math.isfinite(f):  # reject NaN and ±inf
        return None
    return f


def _margin_confidence(net_profit: float) -> float:
    """
    Mechanical arb is always 'correct', but a razor-thin margin can be eaten
    by slippage / partial fills, so confidence scales with the margin size.
    5c+ net profit per $1 set → full confidence.
    """
    return max(0.0, min(1.0, net_profit / 0.05))
