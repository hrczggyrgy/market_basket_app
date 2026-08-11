"""Unit tests for the Pricing insight generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.insights import generate_pricing_insights
from src.analytics.pricing import compute_pricing_decision_matrix, run_pricing_analysis
from src.analytics.schemas import PRICING_INSIGHTS
from tests.unit.pricing_fixtures import build_kvi_fixture, build_pricing_df


def _status_df(n_skus: int, n_estimated: int, n_weak: int = 0) -> pd.DataFrame:
    """Status frame with a given mix of estimated/weak SKUs."""
    statuses = (
        ["estimated"] * n_estimated
        + ["weak"] * n_weak
        + ["insufficient_variation"] * (n_skus - n_estimated - n_weak)
    )
    n_insuf_var = n_skus - n_estimated - n_weak
    return pd.DataFrame(
        {
            "stockcode": [f"SKU{i}" for i in range(n_skus)],
            "elasticity_status": statuses,
            "elasticity": [np.nan] * n_estimated + [-1.5] * n_weak + [np.nan] * n_insuf_var,
            "confidence": ["high"] * n_estimated + ["low"] * n_weak + [np.nan] * n_insuf_var,
            "n_obs": [30] * n_estimated + [10] * n_weak + [6] * n_insuf_var,
            "price_cv": [0.2] * n_skus,
            "r_squared": [0.6] * n_skus,
        }
    )


def _kvi_for_status(status_df: pd.DataFrame) -> pd.DataFrame:
    kvi = build_kvi_fixture()
    if kvi.empty or status_df.empty:
        return kvi
    skus = status_df["stockcode"].tolist()
    rev = pd.Series([1000.0 * (i + 1) for i in range(len(skus))], index=skus)
    return pd.DataFrame(
        {
            "stockcode": skus,
            "category": ["cat"] * len(skus),
            "kvi_score": [0.5 + 0.05 * (i % 5) for i in range(len(skus))],
            "total_revenue": rev.tolist(),
            "basket_penetration": [0.2] * len(skus),
            "trip_incidence": [0.1] * len(skus),
            "abs_elasticity": [np.nan] * len(skus),
            "elasticity_status": status_df["elasticity_status"].tolist(),
        }
    )


def test_insights_coverage_risk_below_half() -> None:
    status = _status_df(n_skus=10, n_estimated=3)
    insights = generate_pricing_insights(
        pd.DataFrame(), status, _kvi_for_status(status), pd.DataFrame()
    )
    PRICING_INSIGHTS.validate(insights, allow_empty=True)
    kinds = insights["kind"].tolist()
    assert "risk" in kinds
    risk = insights[insights["kind"] == "risk"].iloc[0]
    assert "30%" in risk["title"]
    assert risk["impact_value"] is not None


def test_insights_coverage_solid() -> None:
    status = _status_df(n_skus=10, n_estimated=8)
    insights = generate_pricing_insights(
        pd.DataFrame(), status, _kvi_for_status(status), pd.DataFrame()
    )
    kinds = insights["kind"].tolist()
    assert "efficiency" in kinds
    eff = insights[insights["kind"] == "efficiency"].iloc[0]
    assert "80%" in eff["title"]


def test_insights_weak_watch() -> None:
    status = _status_df(n_skus=10, n_estimated=5, n_weak=2)
    insights = generate_pricing_insights(
        pd.DataFrame(), status, _kvi_for_status(status), pd.DataFrame()
    )
    assert "watch" in insights["kind"].tolist()
    watch = insights[insights["kind"] == "watch"]
    assert any("unreliable" in str(t) for t in watch["title"])


def test_insights_decision_groups(sample_df: pd.DataFrame) -> None:
    analysis = run_pricing_analysis(sample_df, min_periods=5)
    insights = generate_pricing_insights(
        analysis.elasticity,
        analysis.elasticity_status,
        analysis.kvi,
        analysis.decision_matrix,
    )
    PRICING_INSIGHTS.validate(insights, allow_empty=True)
    if analysis.decision_matrix.empty:
        pytest.skip("sample data produced no decision matrix")
    titles = " ".join(insights["title"].astype(str))
    assert "insufficient_evidence" not in titles


def test_insights_decision_groups_synthetic() -> None:
    analysis = run_pricing_analysis(build_pricing_df(), min_periods=5)
    insights = generate_pricing_insights(
        analysis.elasticity,
        analysis.elasticity_status,
        analysis.kvi,
        analysis.decision_matrix,
    )
    PRICING_INSIGHTS.validate(insights, allow_empty=True)
    kinds = set(insights["kind"].tolist())
    assert {"opportunity", "growth"} <= kinds
    for decision in ("invest", "protect", "price_lever", "review"):
        assert f"{decision}" in " ".join(insights["title"].astype(str))


def test_insights_extreme_elasticity_watch() -> None:
    status = _status_df(n_skus=6, n_estimated=6)
    elast = pd.DataFrame(
        {
            "stockcode": ["SKU0", "SKU1", "SKU2", "SKU3", "SKU4", "SKU5"],
            "elasticity": [-9.5, -1.2, -0.8, -1.5, -2.0, -1.1],
            "r_squared": [0.3, 0.7, 0.8, 0.7, 0.6, 0.7],
            "p_value": [0.01, 0.001, 0.001, 0.001, 0.001, 0.001],
            "std_err": [0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
            "ci_lower": [-11.0, -1.4, -1.0, -1.7, -2.2, -1.3],
            "ci_upper": [-8.0, -1.0, -0.6, -1.3, -1.8, -0.9],
            "n_obs": [30] * 6,
            "avg_price": [5.0] * 6,
            "avg_weekly_qty": [50.0] * 6,
            "price_cv": [0.2] * 6,
        }
    )
    insights = generate_pricing_insights(
        elast, status, _kvi_for_status(status), pd.DataFrame()
    )
    PRICING_INSIGHTS.validate(insights, allow_empty=True)
    watch = insights[insights["kind"] == "watch"]
    assert any("extreme" in str(t).lower() for t in watch["title"])


def test_insights_empty_inputs() -> None:
    insights = generate_pricing_insights(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert insights.empty
    PRICING_INSIGHTS.validate(insights, allow_empty=True)


def test_insights_decision_matrix_confidence_aggregated() -> None:
    """Decision-group insights aggregate elasticity confidence sensibly."""
    kvi = build_kvi_fixture()
    dm = compute_pricing_decision_matrix(kvi)
    insights = generate_pricing_insights(pd.DataFrame(), kvi[["stockcode", "elasticity_status"]], kvi, dm)
    PRICING_INSIGHTS.validate(insights, allow_empty=True)
    invest = insights[insights["title"].str.startswith("1 SKUs: invest")]
    assert not invest.empty
    assert invest["confidence"].iloc[0] in {"high", "medium", "low"}
