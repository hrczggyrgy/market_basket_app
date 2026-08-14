"""Unit tests for the cross-domain Insight and Opportunity engines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.decision_center import run_decision_center
from src.analytics.insights import (
    generate_assortment_insights,
    generate_customer_insights,
    generate_overview_insights,
    generate_product_insights,
    generate_promotion_insights,
    generate_switching_insights,
)
from src.analytics.insights.promotion import classify_promo_score
from src.analytics.opportunities import (
    generate_assortment_opportunities,
    generate_cross_sell_opportunities,
    generate_promotion_opportunities,
    generate_retention_opportunities,
    generate_switching_opportunities,
)
from src.analytics.schemas import OPPORTUNITY_LIST, PRICING_INSIGHTS

_CONTRACT = PRICING_INSIGHTS


def _dt_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "from_product": ["A", "A", "B", "B"],
            "to_product": ["B", "C", "A", "C"],
            "switch_rate": [0.6, 0.4, 0.5, 0.5],
            "revenue_share_from": [0.5, 0.5, 0.3, 0.3],
            "observed_switching_transference": [0.3, 0.2, 0.15, 0.15],
            "observed_switching_recovery_proxy": [600.0, 400.0, 150.0, 150.0],
        }
    )


def _sdp_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stockcode": ["A", "B", "C", "D"],
            "sdp": [0.1, 0.9, 0.5, 0.05],
        }
    )


# ---------------------------------------------------------------------------
# Overview insights
# ---------------------------------------------------------------------------


def test_overview_insights_contract(sample_df: pd.DataFrame) -> None:
    insights = generate_overview_insights(sample_df)
    _CONTRACT.validate(insights, allow_empty=True)
    assert set(insights["domain"]) == {"overview"}


def test_overview_insights_empty() -> None:
    empty = pd.DataFrame(
        columns=["date", "customer_id", "transaction_id", "stockcode", "price", "quantity"]
    )
    insights = generate_overview_insights(empty)
    _CONTRACT.validate(insights, allow_empty=True)


# ---------------------------------------------------------------------------
# Switching insights / opportunities
# ---------------------------------------------------------------------------


def test_switching_insights_contract() -> None:
    sdp = _sdp_df()
    insights = generate_switching_insights(_dt_df(), sdp)
    _CONTRACT.validate(insights, allow_empty=True)
    kinds = set(insights["kind"])
    assert {"risk", "leakage", "efficiency", "growth"} <= kinds


def test_switching_insights_empty() -> None:
    insights = generate_switching_insights(pd.DataFrame(), pd.DataFrame())
    assert insights.empty
    _CONTRACT.validate(insights, allow_empty=True)


def test_switching_opportunities() -> None:
    sdp = _sdp_df()
    opps = generate_switching_opportunities(
        sdp, revenue_by_product=pd.Series({"A": 100.0, "B": 200.0, "C": 50.0, "D": 40.0})
    )
    OPPORTUNITY_LIST.validate(opps)
    assert not opps.empty
    assert set(opps["domain"]) == {"switching"}


# ---------------------------------------------------------------------------
# Promotion insights / opportunities
# ---------------------------------------------------------------------------


def _waterfall_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stockcode": ["P1", "P2", "P3", "P4"],
            "baseline_revenue": [1000.0, 1000.0, 1000.0, 1000.0],
            "incremental_revenue_qty": [400.0, 300.0, 0.0, -500.0],
            "incremental_revenue_price": [-100.0, -400.0, -50.0, -100.0],
            "incremental_revenue": [300.0, -100.0, -50.0, -600.0],
            "halo_revenue": [0.0, 0.0, 0.0, 0.0],
            "cannibalization_revenue": [0.0, 0.0, 0.0, 0.0],
            "stockpiling_revenue": [0.0, 0.0, 0.0, 0.0],
            "net_incremental_revenue": [300.0, -100.0, -50.0, -600.0],
            "roi": [0.3, -0.1, -0.05, -0.6],
        }
    )


def test_classify_promo_score_buckets() -> None:
    rows = pd.DataFrame(
        {
            "net_incremental_revenue": [300.0, -100.0, -50.0, -600.0],
            "incremental_revenue_qty": [400.0, 300.0, 0.0, -500.0],
            "incremental_revenue_price": [-100.0, -400.0, -50.0, -100.0],
            "roi": [0.3, -0.1, -0.05, -0.6],
        }
    )
    assert rows.apply(classify_promo_score, axis=1).tolist() == [
        "WIN",
        "MIXED",
        "INEFFECTIVE",
        "DESTROYS_VALUE",
    ]


def test_promotion_insights_contract() -> None:
    insights = generate_promotion_insights(_waterfall_df())
    _CONTRACT.validate(insights, allow_empty=True)
    kinds = set(insights["kind"])
    assert "opportunity" in kinds  # WIN
    assert "risk" in kinds  # DESTROYS VALUE


def test_promotion_opportunities() -> None:
    opps = generate_promotion_opportunities(_waterfall_df())
    OPPORTUNITY_LIST.validate(opps)
    assert not opps.empty
    assert set(opps["domain"]) == {"promotion"}


# ---------------------------------------------------------------------------
# Product insights
# ---------------------------------------------------------------------------


def _rationalization_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stockcode": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "revenue": [1000.0, 800.0, 600.0, 400.0, 200.0, 50.0],
            "abc_class": ["A", "A", "B", "B", "C", "C"],
            "xyz_class": ["X", "Y", "Z", "Z", "Z", "Z"],
            "velocity": [10.0, 5.0, 3.0, 2.0, 1.0, 0.5],
            "repeat_rate": [0.6, 0.4, 0.3, 0.2, 0.1, 0.05],
            "action": ["keep", "keep", "review", "review", "delist", "delist"],
        }
    )


def test_product_insights_contract() -> None:
    insights = generate_product_insights(_rationalization_df())
    _CONTRACT.validate(insights, allow_empty=True)
    titles = " ".join(insights["title"].astype(str))
    assert "delist" in titles
    assert "review" in titles


# ---------------------------------------------------------------------------
# Customer insights / retention opportunities
# ---------------------------------------------------------------------------


def _clv_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3", "c4", "c5"],
            "frequency": [10, 8, 3, 2, 1],
            "recency_days": [5, 10, 120, 200, 300],
            "customer_lifetime_days": [300, 250, 100, 80, 40],
            "total_revenue": [900.0, 700.0, 300.0, 200.0, 100.0],
            "avg_order_value": [90.0, 87.5, 100.0, 100.0, 100.0],
            "p_alive": [0.9, 0.8, 0.2, 0.1, 0.05],
            "predicted_purchases": [5.0, 3.0, 0.5, 0.2, 0.1],
            "expected_avg_value": [90.0, 87.5, 100.0, 100.0, 100.0],
            "predicted_clv": [450.0, 262.5, 50.0, 20.0, 10.0],
            "clv_12m": [450.0, 262.5, 50.0, 20.0, 10.0],
            "clv_12m_discounted": [450.0, 262.5, 50.0, 20.0, 10.0],
            "clv_segment": ["High", "High", "Medium", "Low", "Low"],
            "entropy": [0.5, 0.5, 0.5, 0.5, 0.5],
            "normalized_entropy": [0.5, 0.5, 0.5, 0.5, 0.5],
        }
    )


def test_customer_insights_contract() -> None:
    insights = generate_customer_insights(_clv_df())
    _CONTRACT.validate(insights, allow_empty=True)
    kinds = set(insights["kind"])
    assert "opportunity" in kinds  # at-risk high-value
    assert "risk" in kinds or "efficiency" in kinds  # concentration signal


def test_retention_opportunities() -> None:
    opps = generate_retention_opportunities(_clv_df())
    OPPORTUNITY_LIST.validate(opps)
    assert not opps.empty
    entities = set(opps["entity"])
    assert "c3" in entities  # low p_alive, medium CLV
    assert "c1" not in entities  # high p_alive -> not at risk


# ---------------------------------------------------------------------------
# Assortment insights / opportunities
# ---------------------------------------------------------------------------


def _scenario_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario_id": [1, 2, 3],
            "method": ["greedy", "random", "milp"],
            "n_skus": [50, 50, 50],
            "kept_revenue": [8000.0, 7000.0, 8500.0],
            "recovered_revenue": [1500.0, 1200.0, 1000.0],
            "lost_revenue": [2000.0, 3000.0, 1500.0],
            "unmet_demand": [500.0, 1800.0, 500.0],
            "expected_revenue": [9500.0, 8200.0, 9500.0],
            "coverage": [0.95, 0.82, 0.95],
            "recovery_rate": [0.75, 0.4, 0.67],
        }
    )


def test_assortment_insights_contract() -> None:
    insights = generate_assortment_insights(_scenario_df())
    _CONTRACT.validate(insights, allow_empty=True)
    assert not insights.empty


def test_assortment_opportunities() -> None:
    opps = generate_assortment_opportunities(_scenario_df())
    OPPORTUNITY_LIST.validate(opps)
    assert not opps.empty


# ---------------------------------------------------------------------------
# Cross-sell opportunities
# ---------------------------------------------------------------------------


def test_cross_sell_opportunities() -> None:
    addon = pd.DataFrame(
        {
            "anchor": ["A", "A", "B"],
            "addon": ["B", "C", "C"],
            "support": [0.1, 0.05, 0.08],
            "confidence": [0.4, 0.3, 0.5],
            "lift": [3.0, 1.2, 2.5],
            "cooccurrence": [20, 5, 15],
        }
    )
    rev = pd.Series({"A": 1000.0, "B": 800.0})
    opps = generate_cross_sell_opportunities(addon, revenue_by_product=rev)
    OPPORTUNITY_LIST.validate(opps)
    # low lift + low cooccurrence rows filtered out
    assert set(opps["entity"]) == {"B", "C"}


def test_cross_sell_empty() -> None:
    opps = generate_cross_sell_opportunities(pd.DataFrame())
    assert opps.empty
    OPPORTUNITY_LIST.validate(opps, allow_empty=True)


# ---------------------------------------------------------------------------
# Decision Center aggregation
# ---------------------------------------------------------------------------


def test_decision_center_contract(sample_df: pd.DataFrame) -> None:
    analysis = run_decision_center(sample_df)
    PRICING_INSIGHTS.validate(analysis.insights, allow_empty=True)
    OPPORTUNITY_LIST.validate(analysis.opportunities, allow_empty=True)
    assert analysis.n_signals == len(analysis.insights)
    assert analysis.n_opportunities == len(analysis.opportunities)
    assert set(analysis.insights["domain"]) <= {
        "overview",
        "pricing",
        "product",
        "switching",
        "promotion",
        "customer",
        "assortment",
    }
    assert analysis.domains_covered  # at least one engine fired


def test_decision_center_insights_ranked(sample_df: pd.DataFrame) -> None:
    analysis = run_decision_center(sample_df)
    if analysis.insights.empty:
        pytest.skip("no insights on sample data")
    values = analysis.insights["impact_value"].fillna(-1.0).tolist()
    assert values == sorted(values, reverse=True) or all(v < 0 for v in values)
