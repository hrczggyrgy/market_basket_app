"""Tests for basket metrics and cohort analytics."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.basket_metrics import (
    basket_penetration_over_time,
    compute_basket_composition,
    compute_basket_penetration,
    compute_customer_entropy,
    compute_ipt_cv,
)
from src.analytics.cohort import (
    compute_cohort_decay_rate,
    compute_cohort_ltv_curve,
    compute_cohort_sizes,
    compute_cohorts,
    compute_role_retention,
    period_over_period_comparison,
    year_over_year_comparison,
)
from src.analytics.schemas import (
    BASKET_COMPOSITION,
    BASKET_OVER_TIME,
    BASKET_PENETRATION,
    COHORT_DECAY,
    COHORT_LTV,
    COHORT_RETENTION,
    COHORT_SIZES,
    CUSTOMER_ENTROPY,
    IPT_CV,
    POP_COMPARISON,
    ROLE_RETENTION,
    YOY_COMPARISON,
)

# --- basket metrics ---


def test_basket_penetration_contract(sample_df: pd.DataFrame) -> None:
    table = compute_basket_penetration(sample_df)
    BASKET_PENETRATION.validate(table)
    assert table["penetration"].between(0, 1).all()
    assert np.isclose(table["penetration"].sum() / len(table), table["penetration"].mean())
    assert table["basket_count"].min() >= 1
    assert np.isclose(table["revenue_share"].sum(), 1.0, atol=1e-9)


def test_basket_penetration_monotonic(sample_df: pd.DataFrame) -> None:
    table = compute_basket_penetration(sample_df)
    assert table["penetration"].is_monotonic_decreasing


def test_basket_penetration_over_time(sample_df: pd.DataFrame) -> None:
    table = basket_penetration_over_time(sample_df, period="W")
    BASKET_OVER_TIME.validate(table)
    assert len(table) >= 10
    assert table["avg_basket_size"].min() >= 1
    assert table["avg_basket_value"].min() > 0


def test_basket_composition_contract(sample_df: pd.DataFrame) -> None:
    table = compute_basket_composition(sample_df)
    BASKET_COMPOSITION.validate(table)
    assert np.isclose(table["pct"].sum(), 1.0, atol=1e-9)
    assert table["n_baskets"].sum() == sample_df["transaction_id"].nunique()


def test_customer_entropy_contract(sample_df: pd.DataFrame) -> None:
    table = compute_customer_entropy(sample_df)
    CUSTOMER_ENTROPY.validate(table)
    assert table["customer_id"].is_unique
    assert table["entropy"].min() >= 0
    assert table["normalized_entropy"].between(0, 1).all()


def test_customer_entropy_single_product_is_zero() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "transaction_id": ["T1", "T2", "T3"],
            "stockcode": ["A", "A", "A"],
            "product": ["p"] * 3,
            "customer_id": ["C1"] * 3,
            "price": [1.0] * 3,
            "quantity": [1] * 3,
        }
    )
    table = compute_customer_entropy(df)
    assert table.loc[0, "entropy"] == 0.0
    assert table.loc[0, "normalized_entropy"] == 0.0


def test_ipt_cv_contract(sample_df: pd.DataFrame) -> None:
    table = compute_ipt_cv(sample_df)
    IPT_CV.validate(table)
    assert table["stockcode"].is_unique
    assert (table["cv_ipt"] >= 0).all()
    assert (table["n_transactions"] >= 1).all()


# --- cohort ---


def test_compute_cohorts_contract(sample_df: pd.DataFrame) -> None:
    table = compute_cohorts(sample_df, cohort_period="M")
    COHORT_RETENTION.validate(table)
    assert table["retention_rate"].between(0, 1).all()
    assert table["period_index"].min() == 0
    assert (table["retention_rate"][table["period_index"] == 0] == 1.0).all()


def test_compute_cohort_sizes_contract(sample_df: pd.DataFrame) -> None:
    table = compute_cohort_sizes(sample_df, cohort_period="M")
    COHORT_SIZES.validate(table)
    assert table["n_customers"].sum() == sample_df["customer_id"].nunique()


def test_pop_comparison_contract(sample_df: pd.DataFrame) -> None:
    table = period_over_period_comparison(sample_df, period="W")
    POP_COMPARISON.validate(table)
    assert table["aov"].min() > 0


def test_yoy_comparison_contract(sample_df: pd.DataFrame) -> None:
    table = year_over_year_comparison(sample_df)
    YOY_COMPARISON.validate(table)
    assert "revenue_yoy_growth" in table.columns


def test_cohort_ltv_curve_contract(sample_df: pd.DataFrame) -> None:
    table = compute_cohort_ltv_curve(sample_df, cohort_period="M")
    COHORT_LTV.validate(table)
    assert table["ltv_per_customer"].min() > 0
    per_cohort = table.groupby("cohort")["cumulative_revenue"].diff().dropna()
    assert (per_cohort >= -1e-9).all()


def test_cohort_decay_rate_contract(sample_df: pd.DataFrame) -> None:
    table = compute_cohort_decay_rate(sample_df, cohort_period="M")
    COHORT_DECAY.validate(table, allow_empty=True)
    if not table.empty:
        assert table["decay_rate"].notna().all()


def test_role_retention_contract(sample_df: pd.DataFrame) -> None:
    table = compute_role_retention(sample_df, cohort_period="M", min_role_customers=1)
    ROLE_RETENTION.validate(table, allow_empty=True)
    if not table.empty:
        assert table["retention_rate"].between(0, 1).all()
        assert (table["cohort_size"] >= table["retained"]).all()
        assert table["period_index"].min() == 0
        # Each (role, cohort) starts at 100% retention
        first = table[table["period_index"] == 0]
        assert (first["retention_rate"] == 1.0).all()


def test_role_retention_empty_without_category(sample_df: pd.DataFrame) -> None:
    df = sample_df.copy().drop(columns=["category"])
    table = compute_role_retention(df, cohort_period="M")
    assert table.empty
    ROLE_RETENTION.validate(table, allow_empty=True)


def test_role_retention_deterministic(sample_df: pd.DataFrame) -> None:
    a = compute_role_retention(sample_df, cohort_period="M", min_role_customers=1)
    b = compute_role_retention(sample_df, cohort_period="M", min_role_customers=1)
    assert a.equals(b)


def test_cohort_deterministic(sample_df: pd.DataFrame) -> None:
    a = compute_cohorts(sample_df, "W")
    b = compute_cohorts(sample_df, "W")
    pd.testing.assert_frame_equal(a, b)
