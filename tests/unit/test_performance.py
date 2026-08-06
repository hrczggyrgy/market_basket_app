"""Tests for product performance analytics."""

import pandas as pd
import pytest

from src.analytics.performance import (
    abc_analysis,
    compute_product_metrics,
    compute_repeat_rate,
    compute_sku_rationalization_df,
    compute_time_to_second_purchase,
    compute_velocity,
    product_lifecycle_stage,
    xyz_analysis,
)
from src.analytics.schemas import (
    ABC_CLASSES,
    LIFECYCLE,
    PRODUCT_METRICS,
    PRODUCT_VELOCITY,
    REPEAT_RATE,
    SECOND_PURCHASE,
    SKU_RATIONALIZATION,
    XYZ_CLASSES,
    check,
)


def test_product_metrics_contract(sample_df: pd.DataFrame) -> None:
    table = compute_product_metrics(sample_df)
    check(table, PRODUCT_METRICS)
    assert len(table) >= 1
    assert table["avg_price"].notna().all()
    assert table["penetration"].between(0, 1).all()


def test_product_metrics_revenue_totals(sample_df: pd.DataFrame) -> None:
    table = compute_product_metrics(sample_df)
    expected = (sample_df["price"] * sample_df["quantity"]).sum()
    assert table["revenue"].sum() == pytest.approx(expected, rel=1e-9)


def test_abc_analysis_contract(sample_df: pd.DataFrame) -> None:
    table = abc_analysis(sample_df)
    check(table, ABC_CLASSES)
    assert set(table["abc_class"].unique()) <= {"A", "B", "C"}
    assert table["abc_class"].value_counts()["A"] >= 1


def test_abc_cumulative_share_monotonic(sample_df: pd.DataFrame) -> None:
    table = abc_analysis(sample_df).sort_values("revenue", ascending=False)
    assert table["cumulative_share"].is_monotonic_increasing


def test_xyz_analysis_contract(sample_df: pd.DataFrame) -> None:
    table = xyz_analysis(sample_df)
    check(table, XYZ_CLASSES)
    assert set(table["xyz_class"].unique()) <= {"X", "Y", "Z"}


def test_lifecycle_stage_contract(sample_df: pd.DataFrame) -> None:
    table = product_lifecycle_stage(sample_df)
    if table.empty:
        return
    check(table, LIFECYCLE)
    assert set(table["stage"].unique()) <= {"growth", "mature", "decline"}


def test_velocity_contract(sample_df: pd.DataFrame) -> None:
    table = compute_velocity(sample_df)
    check(table, PRODUCT_VELOCITY)
    assert (table["velocity"] > 0).all()


def test_repeat_rate_contract(sample_df: pd.DataFrame) -> None:
    table = compute_repeat_rate(sample_df)
    check(table, REPEAT_RATE)
    assert table["repeat_rate"].between(0, 1).all()


def test_repeat_rate_single_purchase_per_customer() -> None:
    df = pd.DataFrame(
        {
            "stockcode": ["A", "B"],
            "customer_id": [1, 2],
            "quantity": [1, 1],
        }
    )
    table = compute_repeat_rate(df)
    assert (table["repeat_rate"] == 0.0).all()


def test_time_to_second_purchase_contract(sample_df: pd.DataFrame) -> None:
    table = compute_time_to_second_purchase(sample_df)
    if table.empty:
        return
    check(table, SECOND_PURCHASE)
    assert (table["median_days_to_second"] >= 0).all()


def test_sku_rationalization_contract(sample_df: pd.DataFrame) -> None:
    table = compute_sku_rationalization_df(sample_df)
    check(table, SKU_RATIONALIZATION)
    assert set(table["action"].unique()) <= {"keep", "review", "delist_candidate"}


def test_sku_rationalization_consistent_classes(sample_df: pd.DataFrame) -> None:
    table = compute_sku_rationalization_df(sample_df)
    abc = abc_analysis(sample_df).set_index("stockcode")["abc_class"]
    for _, row in table.iterrows():
        if row["abc_class"] != "C":
            assert abc.loc[row["stockcode"]] == row["abc_class"]
