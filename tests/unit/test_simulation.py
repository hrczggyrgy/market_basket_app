"""Tests for the config-driven transaction simulator."""

import numpy as np
import pandas as pd

from src.analytics.simulation import (
    SCENARIOS,
    SimulationConfig,
    calibration_report,
    config_for,
    generate_sample_transactions,
)

SCHEMA = {
    "date",
    "transaction_id",
    "stockcode",
    "product",
    "customer_id",
    "price",
    "quantity",
    "category",
    "brand",
    "size",
    "flavor",
    "promo_flag",
    "cost",
}


def _small(**overrides) -> SimulationConfig:
    base = dict(n_customers=40, n_products=25, n_days=45, seed=42)
    base.update(overrides)
    return SimulationConfig(**base)


def test_schema_and_basic_sanity() -> None:
    df = generate_sample_transactions(_small())
    assert set(df.columns) == SCHEMA
    assert df["price"].min() > 0
    assert df["quantity"].min() >= 1
    assert df["date"].is_monotonic_increasing
    assert df["promo_flag"].dtype == bool


def test_deterministic_given_same_seed() -> None:
    a = generate_sample_transactions(_small())
    b = generate_sample_transactions(_small())
    pd.testing.assert_frame_equal(a, b)


def test_seed_changes_output() -> None:
    a = generate_sample_transactions(_small(seed=1))
    b = generate_sample_transactions(_small(seed=2))
    assert not a.equals(b)


def test_categories_assigned() -> None:
    df = generate_sample_transactions(_small())
    assert df["category"].nunique() >= 3


def test_scenarios_differ() -> None:
    cfg_a = config_for("standard")
    cfg_b = config_for("promo_heavy")
    df_a = generate_sample_transactions(cfg_a)
    df_b = generate_sample_transactions(cfg_b)
    assert len(df_a) != len(df_b) or not df_a.equals(df_b)


def test_all_scenario_keys_render() -> None:
    assert set(SCENARIOS) == {"standard", "promo_heavy", "seasonal", "high_switching"}
    for key in SCENARIOS:
        df = generate_sample_transactions(SCENARIOS[key])
        assert set(df.columns) == SCHEMA
        assert len(df) > 0


def test_promo_heavy_has_more_promo_rows() -> None:
    heavy = generate_sample_transactions(config_for("promo_heavy"))
    std = generate_sample_transactions(config_for("standard"))
    assert heavy["promo_flag"].mean() > std["promo_flag"].mean()


def test_returns_produce_negative_rows() -> None:
    df = generate_sample_transactions(_small(return_rate=0.2))
    neg = df[df["price"] < 0]
    assert len(neg) > 0
    assert (neg["quantity"] < 0).all()
    assert neg["transaction_id"].str.contains("-R").all()


def test_no_returns_by_default() -> None:
    df = generate_sample_transactions(_small())
    assert (df["price"] >= 0).all()
    assert (df["quantity"] >= 0).all()


def test_calibration_report() -> None:
    df = generate_sample_transactions(_small())
    report = calibration_report(df, _small())
    assert {"metric", "value", "target"} <= set(report.columns)
    assert (report["value"] >= 0).all()
    assert report["metric"].isin(["revenue_pareto_top20", "avg_basket_size"]).any()
    pareto = report.loc[report["metric"] == "revenue_pareto_top20", "value"].iloc[0]
    assert 0.4 <= pareto <= 1.0


def test_basket_sizes_positive() -> None:
    df = generate_sample_transactions(_small())
    baskets = df[df["quantity"] > 0].groupby("transaction_id")["quantity"].sum()
    assert (baskets > 0).all()
    assert baskets.mean() >= 1.0
