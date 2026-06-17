"""
Tests for IntraMarketArbitrage — binary-bundle and neg-risk detection.

These verify the arithmetic is correct and that non-opportunities (the
common case where prices sum to ~$1.00) are correctly rejected.
"""
from __future__ import annotations

from shared.intra_market_arb import (
    IntraMarketArbitrage,
    IntraMarketArb,
    ArbLeg,
)


class TestBinaryBundle:
    def test_detects_underpriced_bundle(self):
        det = IntraMarketArbitrage(fee_per_share=0.0, min_net_profit=0.005)
        markets = [{
            "id": "c1", "question": "Will X happen?",
            "yes_ask": 0.45, "no_ask": 0.50,   # sum 0.95 → 0.05 profit
            "yes_token_id": "y1", "no_token_id": "n1",
        }]
        arbs = det.find_binary_bundle(markets)
        assert len(arbs) == 1
        a = arbs[0]
        assert a.arb_type == "binary_bundle"
        assert a.total_cost == 0.95
        assert abs(a.gross_profit - 0.05) < 1e-9
        assert abs(a.net_profit - 0.05) < 1e-9
        assert a.legs[0].outcome == "YES" and a.legs[0].token_id == "y1"
        assert a.legs[1].outcome == "NO" and a.legs[1].token_id == "n1"

    def test_rejects_fairly_priced_bundle(self):
        det = IntraMarketArbitrage()
        markets = [{"id": "c1", "question": "q", "yes_ask": 0.52, "no_ask": 0.50}]
        # sum 1.02 → no arb
        assert det.find_binary_bundle(markets) == []

    def test_rejects_exact_dollar_bundle(self):
        det = IntraMarketArbitrage(min_net_profit=0.005)
        markets = [{"id": "c1", "question": "q", "yes_ask": 0.50, "no_ask": 0.50}]
        assert det.find_binary_bundle(markets) == []

    def test_fee_eats_thin_margin(self):
        # 0.99 sum → 0.01 gross, but 2×0.01 fee = 0.02 → net negative
        det = IntraMarketArbitrage(fee_per_share=0.01, min_net_profit=0.005)
        markets = [{"id": "c1", "question": "q", "yes_ask": 0.49, "no_ask": 0.50}]
        assert det.find_binary_bundle(markets) == []

    def test_skips_missing_or_zero_asks(self):
        det = IntraMarketArbitrage()
        markets = [
            {"id": "c1", "question": "q", "yes_ask": 0.4},            # no_ask missing
            {"id": "c2", "question": "q", "yes_ask": 0.0, "no_ask": 0.4},  # zero ask
            {"id": "c3", "question": "q", "yes_ask": None, "no_ask": 0.4},
        ]
        assert det.find_binary_bundle(markets) == []

    def test_sorted_by_net_profit(self):
        det = IntraMarketArbitrage()
        markets = [
            {"id": "a", "question": "q", "yes_ask": 0.48, "no_ask": 0.50},  # 0.02
            {"id": "b", "question": "q", "yes_ask": 0.40, "no_ask": 0.50},  # 0.10
            {"id": "c", "question": "q", "yes_ask": 0.45, "no_ask": 0.50},  # 0.05
        ]
        arbs = det.find_binary_bundle(markets)
        nets = [a.net_profit for a in arbs]
        assert nets == sorted(nets, reverse=True)
        assert arbs[0].event_id == "b"


class TestNegRisk:
    def test_detects_underpriced_multi_outcome(self):
        det = IntraMarketArbitrage(fee_per_share=0.0, min_net_profit=0.005)
        events = [{
            "event_id": "wc", "question": "Who wins?",
            "outcomes": [
                {"label": "A", "yes_ask": 0.30, "token_id": "tA"},
                {"label": "B", "yes_ask": 0.30, "token_id": "tB"},
                {"label": "C", "yes_ask": 0.30, "token_id": "tC"},
            ],  # sum 0.90 → 0.10 profit
        }]
        arbs = det.find_neg_risk(events)
        assert len(arbs) == 1
        a = arbs[0]
        assert a.arb_type == "neg_risk"
        assert abs(a.total_cost - 0.90) < 1e-9
        assert abs(a.net_profit - 0.10) < 1e-9
        assert len(a.legs) == 3

    def test_rejects_overpriced_event(self):
        det = IntraMarketArbitrage()
        events = [{
            "event_id": "e", "question": "q",
            "outcomes": [
                {"label": "A", "yes_ask": 0.50},
                {"label": "B", "yes_ask": 0.55},
            ],  # sum 1.05
        }]
        assert det.find_neg_risk(events) == []

    def test_skips_event_with_missing_ask(self):
        det = IntraMarketArbitrage()
        events = [{
            "event_id": "e", "question": "q",
            "outcomes": [
                {"label": "A", "yes_ask": 0.30},
                {"label": "B"},  # no ask → whole event invalid
            ],
        }]
        assert det.find_neg_risk(events) == []

    def test_requires_two_outcomes(self):
        det = IntraMarketArbitrage()
        events = [{
            "event_id": "e", "question": "q",
            "outcomes": [{"label": "A", "yes_ask": 0.10}],
        }]
        assert det.find_neg_risk(events) == []


class TestScanAndRoi:
    def test_scan_combines_both(self):
        det = IntraMarketArbitrage(min_net_profit=0.005)
        binary = [{"id": "c1", "question": "q", "yes_ask": 0.45, "no_ask": 0.50}]
        events = [{
            "event_id": "e", "question": "q",
            "outcomes": [
                {"label": "A", "yes_ask": 0.30},
                {"label": "B", "yes_ask": 0.30},
                {"label": "C", "yes_ask": 0.30},
            ],
        }]
        results = det.scan(binary_markets=binary, neg_risk_events=events)
        assert len(results) == 2
        types = {r.arb_type for r in results}
        assert types == {"binary_bundle", "neg_risk"}
        # sorted by net profit desc → neg_risk (0.10) before binary (0.05)
        assert results[0].arb_type == "neg_risk"

    def test_roi_computation(self):
        a = IntraMarketArb(
            arb_type="binary_bundle", event_id="x", question="q",
            total_cost=0.95, gross_profit=0.05, net_profit=0.05,
            legs=[ArbLeg("YES", 0.45), ArbLeg("NO", 0.50)],
        )
        assert abs(a.roi - (0.05 / 0.95)) < 1e-9

    def test_confidence_scales_with_margin(self):
        det = IntraMarketArbitrage(min_net_profit=0.0)
        thin = det.find_binary_bundle(
            [{"id": "a", "question": "q", "yes_ask": 0.495, "no_ask": 0.50}]
        )[0]
        fat = det.find_binary_bundle(
            [{"id": "b", "question": "q", "yes_ask": 0.40, "no_ask": 0.50}]
        )[0]
        assert thin.confidence < fat.confidence
        assert fat.confidence == 1.0  # 0.10 margin >> 0.05 → capped
