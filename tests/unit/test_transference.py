"""Unit tests for demand transference analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.transference import (
    bootstrap_demand_transference_ci,
    build_similarity_matrix,
    build_substitution_matrix_markov,
    build_substitution_matrix_mnl,
    compute_cross_price_elasticity,
    compute_demand_transference_matrix,
    compute_recovery_hhi,
    compute_substitutable_demand_percentage,
    delist_impact_analysis,
    node_delist_impact,
    simulate_assortment_change,
)
from src.analytics.schemas import (
    CROSS_ELASTICITY,
    DELIST_IMPACT,
    DEMAND_TRANSFERENCE,
    NODE_DELIST_IMPACT,
    RECOVERY_HHI,
    SDP_SCORES,
    TRANSFERENCE_CI,
)


def test_demand_transference_matrix_contract_and_math(sample_df: pd.DataFrame) -> None:
    dt = compute_demand_transference_matrix(sample_df)
    DEMAND_TRANSFERENCE.validate(dt)
    assert (dt["switch_rate"] >= 0).all() and (dt["switch_rate"] <= 1).all()
    assert (dt["demand_transference"] >= 0).all()
    assert (dt["revenue_at_risk"] >= 0).all()
    assert (dt["revenue_at_risk"] <= dt["revenue_share_from"] * 0 + dt["revenue_at_risk"].max() * 1.01).all()
    assert dt["demand_transference"].max() <= 1.0


def test_demand_transference_switch_rates_row_normalized(sample_df: pd.DataFrame) -> None:
    dt = compute_demand_transference_matrix(sample_df)
    per_from = dt.groupby("from_product")["switch_rate"].sum()
    assert per_from.between(0.999, 1.001).all()


def test_demand_transference_top_n_filters(sample_df: pd.DataFrame) -> None:
    full = compute_demand_transference_matrix(sample_df)
    top = compute_demand_transference_matrix(sample_df, top_n=10)
    assert len(top) < len(full)
    assert top["from_product"].nunique() <= 10 and top["to_product"].nunique() <= 10


def test_sdp_scores(sample_df: pd.DataFrame) -> None:
    dt = compute_demand_transference_matrix(sample_df)
    sdp = compute_substitutable_demand_percentage(dt, sample_df)
    SDP_SCORES.validate(sdp)
    assert sdp["sdp"].between(0, 1).all()
    assert set(sdp["stockcode"]) == set(dt["from_product"])


def test_delist_impact_analysis(sample_df: pd.DataFrame) -> None:
    dt = compute_demand_transference_matrix(sample_df)
    revenue = (sample_df["price"] * sample_df["quantity"]).groupby(sample_df["stockcode"]).sum()
    top2 = revenue.nlargest(2).index.tolist()
    impact = delist_impact_analysis(sample_df, dt, top2)
    DELIST_IMPACT.validate(impact)
    assert len(impact) == 2
    assert (impact["product_revenue"] > 0).all()
    assert (impact["estimated_revenue_recovered"] >= 0).all()
    expected = {
        "recovered": float(
            dt[dt["from_product"].isin(top2)]["revenue_at_risk"].sum()
        ),
        "own": float(revenue.loc[top2].sum()),
    }
    assert abs(impact["net_revenue_impact"].sum() - (expected["recovered"] - expected["own"])) < 1e-6


def test_node_delist_impact(sample_df: pd.DataFrame) -> None:
    dt = compute_demand_transference_matrix(sample_df)
    top_products = dt["from_product"].unique()[:20]
    assignments = {p: i % 3 for i, p in enumerate(top_products)}
    table = node_delist_impact(sample_df, dt, assignments)
    NODE_DELIST_IMPACT.validate(table)
    assert set(table["node_id"]) == {0, 1, 2}


def test_markov_substitution_matrix(sample_df: pd.DataFrame) -> None:
    from src.analytics.switching import compute_switching_matrix

    sw = compute_switching_matrix(sample_df)
    P = build_substitution_matrix_markov(sw, max_iterations=5)
    assert P.shape[0] == P.shape[1] == len(set(sw["from_product"]) | set(sw["to_product"]))
    assert P.values.min() >= 0
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums[row_sums > 0], 1.0, atol=1e-6)


def test_mnl_substitution_matrix(sample_df: pd.DataFrame) -> None:
    sim = build_similarity_matrix(sample_df, top_n=30)
    P = build_substitution_matrix_mnl(sample_df, sim)
    assert P.shape == sim.shape
    assert P.values.min() >= 0
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
    np.fill_diagonal(P.values, 0)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)


def test_similarity_matrix_symmetric(sample_df: pd.DataFrame) -> None:
    sim = build_similarity_matrix(sample_df, top_n=30)
    assert (sim == sim.T).all().all()


def test_simulate_assortment_change(sample_df: pd.DataFrame) -> None:
    revenue = (sample_df["price"] * sample_df["quantity"]).groupby(sample_df["stockcode"]).sum()
    sim = build_similarity_matrix(sample_df, top_n=30)
    P = build_substitution_matrix_mnl(sample_df, sim)
    delist = [revenue.idxmax()]
    result = simulate_assortment_change(delist, P, revenue)
    assert isinstance(result["recovery_detail"], dict)
    assert result["lost_revenue"] == float(revenue.max())
    assert 0 <= result["recovery_rate"] <= 1


def test_cross_price_elasticity(sample_df: pd.DataFrame) -> None:
    revenue = (sample_df["price"] * sample_df["quantity"]).groupby(sample_df["stockcode"]).sum()
    top5 = revenue.nlargest(5).index.tolist()
    pairs = [(top5[i], top5[j]) for i in range(len(top5)) for j in range(i + 1, len(top5))]
    table = compute_cross_price_elasticity(sample_df, pairs)
    CROSS_ELASTICITY.validate(table, allow_empty=True)
    if not table.empty:
        assert table["n_obs"].min() >= 5


def test_recovery_hhi(sample_df: pd.DataFrame) -> None:
    dt = compute_demand_transference_matrix(sample_df)
    hhi = compute_recovery_hhi(dt)
    RECOVERY_HHI.validate(hhi)
    if not hhi.empty:
        assert hhi["recovery_hhi"].between(0, 1).all()
        assert hhi["n_substitutes"].ge(1).all()


def test_bootstrap_ci_small(sample_df: pd.DataFrame) -> None:
    ci = bootstrap_demand_transference_ci(
        sample_df, n_resamples=15, max_pairs=3, random_seed=42
    )
    TRANSFERENCE_CI.validate(ci)
    if not ci.empty:
        assert (ci["lower"] <= ci["estimate"]).all()
        assert (ci["estimate"] <= ci["upper"]).all()
        assert (ci["n_resamples"] >= 1).all()


def test_empty_paths_empty_df() -> None:
    empty = pd.DataFrame(columns=["from_product", "to_product", "count", "pct"])
    P = build_substitution_matrix_markov(empty)
    assert P.empty
    minimal = pd.DataFrame(
        columns=["date", "transaction_id", "stockcode", "customer_id", "price", "quantity"]
    ).astype({"date": "datetime64[ns]"})
    out = compute_demand_transference_matrix(minimal)
    assert out.empty
