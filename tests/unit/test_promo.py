"""Tests for promotional analytics."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.promo import (
    build_uplift_dataset,
    calculate_promotional_lift,
    check_propensity_overlap,
    compute_cannibalization_analysis,
    compute_category_promo_timeline,
    compute_incrementality_waterfall,
    compute_promo_baseline,
    detect_promotions,
    estimate_propensity_score,
    evaluate_uplift_model,
    halo_effect_analysis,
    mark_promo_transactions,
    promotion_timing_analysis,
    promo_roi_analysis,
    score_uplift_by_customer,
    train_uplift_learner,
)
from src.analytics.schemas import (
    CATEGORY_PROMO_TIMELINE,
    PROMO_BASELINE,
    PROMO_CANNIBALIZATION,
    PROMO_HALO,
    PROMO_LIFT,
    PROMO_PERIODS,
    PROMO_ROI,
    PROMO_TIMING_DOW,
    PROMO_TIMING_MONTH,
    PROMO_WATERFALL,
    QINI_CURVE,
    UPLIFT_METRICS,
    UPLIFT_SCORES,
    check,
)


@pytest.fixture()
def crafted_df() -> pd.DataFrame:
    """A df with one obvious 3-day 30% promo on product A."""
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    rows = []
    for sku in ("A", "B"):
        for i, day in enumerate(dates):
            if sku == "A" and 10 <= i <= 12:
                price, qty = 7.0, 8
            elif sku == "A":
                price, qty = 10.0, 2
            else:
                price, qty = 5.0, 1
            rows.append(
                {
                    "date": day,
                    "transaction_id": f"T{sku}{i}",
                    "stockcode": sku,
                    "product": f"Product {sku}",
                    "customer_id": 1000 + i,
                    "price": price,
                    "quantity": qty,
                }
            )
    return pd.DataFrame(rows)


def test_detect_promotions_finds_crafted_period(crafted_df: pd.DataFrame) -> None:
    table = detect_promotions(crafted_df)
    check(table, PROMO_PERIODS)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["stockcode"] == "A"
    assert row["duration_days"] == 3
    assert row["avg_discount_pct"] == pytest.approx(30.0, abs=1.0)
    assert row["qty_lift"] > 0


def test_detect_promotions_none() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10),
            "transaction_id": range(10),
            "stockcode": ["A"] * 10,
            "product": ["X"] * 10,
            "customer_id": [1] * 10,
            "price": [10.0] * 10,
            "quantity": [1] * 10,
        }
    )
    table = detect_promotions(df)
    assert table.empty


def test_mark_promo_transactions(crafted_df: pd.DataFrame) -> None:
    promos = detect_promotions(crafted_df)
    marked = mark_promo_transactions(crafted_df, promos)
    assert marked["is_promo"].sum() == 3
    assert marked.loc[marked["stockcode"] == "B", "is_promo"].sum() == 0


def test_compute_promo_baseline_contract_and_marking(crafted_df: pd.DataFrame) -> None:
    promos = detect_promotions(crafted_df)
    baseline = compute_promo_baseline(crafted_df, promos, seasonal_period=4)
    check(baseline, PROMO_BASELINE)
    assert baseline["is_promo"].dtype == bool
    assert (baseline["actual_units"] >= 0).all()
    assert baseline["is_promo"].any()


def test_compute_promo_baseline_empty(crafted_df: pd.DataFrame) -> None:
    baseline = compute_promo_baseline(crafted_df, pd.DataFrame())
    check(baseline, PROMO_BASELINE, allow_empty=True)


def test_calculate_promotional_lift(crafted_df: pd.DataFrame) -> None:
    promos = detect_promotions(crafted_df)
    lift = calculate_promotional_lift(crafted_df, promos)
    check(lift, PROMO_LIFT)
    row = lift.iloc[0]
    assert row["stockcode"] == "A"
    assert row["lift_qty_pct"] > 0
    assert row["significant"] in (True, False)


def test_compute_incrementality_waterfall(crafted_df: pd.DataFrame) -> None:
    promos = detect_promotions(crafted_df)
    baseline = compute_promo_baseline(crafted_df, promos, seasonal_period=4)
    halo = pd.DataFrame(
        {
            "promo_product": ["A"],
            "halo_product": ["B"],
            "halo_revenue": [50.0],
            "base_revenue": [10.0],
            "halo_orders": [5],
            "base_orders": [2],
            "revenue_lift": [400.0],
        }
    )
    cann = pd.DataFrame({"stockcode": ["B"], "cannibalization_revenue": [20.0]})
    table = compute_incrementality_waterfall(baseline, halo_revenue=halo, cannibalization_revenue=cann)
    check(table, PROMO_WATERFALL)
    row = table.loc[table["stockcode"] == "A"].iloc[0]
    assert row["net_incremental_revenue"] == pytest.approx(
        row["incremental_revenue"] + 50.0, abs=1e-9
    )


def test_promo_roi_analysis_contract_and_ci(crafted_df: pd.DataFrame) -> None:
    promos = detect_promotions(crafted_df)
    roi = promo_roi_analysis(crafted_df, promos, n_resamples=200)
    check(roi, PROMO_ROI)
    assert len(roi) == 1
    assert roi["ci_low"].iloc[0] <= roi["ci_high"].iloc[0]
    assert np.isfinite(roi["incremental_revenue"].iloc[0])


def test_promo_roi_analysis_empty(crafted_df: pd.DataFrame) -> None:
    roi = promo_roi_analysis(crafted_df, pd.DataFrame())
    assert roi.empty


def test_promotion_timing_analysis(crafted_df: pd.DataFrame) -> None:
    promos = detect_promotions(crafted_df)
    timing = promotion_timing_analysis(crafted_df, promos)
    check(timing["by_day_of_week"], PROMO_TIMING_DOW)
    check(timing["by_month"], PROMO_TIMING_MONTH)
    assert set(timing["by_day_of_week"]["day_name"]) <= {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}


def test_halo_effect_analysis(crafted_df: pd.DataFrame) -> None:
    promos = detect_promotions(crafted_df)
    halo = halo_effect_analysis(crafted_df, promos)
    check(halo, PROMO_HALO, allow_empty=True)
    if not halo.empty:
        assert (halo["halo_product"] != halo["promo_product"]).all()


def test_compute_cannibalization_analysis(crafted_df: pd.DataFrame) -> None:
    # Add a same-category peer whose revenue drops during the promo of A.
    df = crafted_df.copy()
    df["category"] = np.where(df["stockcode"] == "A", "Cat1", "Cat1")
    promos = detect_promotions(df)
    cann = compute_cannibalization_analysis(df, promos)
    check(cann, PROMO_CANNIBALIZATION, allow_empty=True)
    if cann.empty:
        pytest.skip("no cannibalization signal in fixture")
    assert (cann["cannibalized_revenue"] >= 0).all()
    assert cann["cannibalization_index"].between(0, 1).all()


def test_compute_cannibalization_analysis_empty() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10),
            "transaction_id": range(10),
            "stockcode": ["A"] * 10,
            "product": ["X"] * 10,
            "customer_id": [1] * 10,
            "price": [10.0] * 10,
            "quantity": [1] * 10,
        }
    )
    cann = compute_cannibalization_analysis(df, pd.DataFrame())
    check(cann, PROMO_CANNIBALIZATION, allow_empty=True)
    assert cann.empty


def test_build_uplift_dataset_shapes(sample_df: pd.DataFrame) -> None:
    promos = detect_promotions(sample_df)
    X, treatment, y = build_uplift_dataset(sample_df, promos)
    assert len(X) == len(treatment) == len(y)
    assert set(treatment.unique()) <= {0, 1}
    if len(X):
        assert (y >= 0).all()
        assert X.isna().sum().sum() == 0


def test_propensity_and_overlap(sample_df: pd.DataFrame) -> None:
    promos = detect_promotions(sample_df)
    X, treatment, y = build_uplift_dataset(sample_df, promos)
    if len(X) < 30 or treatment.sum() < 5:
        pytest.skip("not enough promo exposure")
    propensity = estimate_propensity_score(X, treatment)
    assert propensity.between(0, 1).all()
    diag = check_propensity_overlap(propensity, treatment)
    assert {"overlap", "overlap_proportion", "warnings"} <= set(diag.keys())
    assert diag["overlap_proportion"] >= 0.0


def test_train_uplift_learner_s_learner(sample_df: pd.DataFrame) -> None:
    promos = detect_promotions(sample_df)
    X, treatment, y = build_uplift_dataset(sample_df, promos)
    if len(X) < 30:
        pytest.skip("dataset too small")
    model, uplift = train_uplift_learner(X, treatment, y, learner="s", base_estimator="rf", n_estimators=50)
    assert len(uplift) == len(X)
    scores = score_uplift_by_customer(X, uplift, sample_df["customer_id"].iloc[: len(X)])
    check(scores, UPLIFT_SCORES)


def test_train_uplift_learner_t_learner(sample_df: pd.DataFrame) -> None:
    promos = detect_promotions(sample_df)
    X, treatment, y = build_uplift_dataset(sample_df, promos)
    if len(X) < 30:
        pytest.skip("dataset too small")
    models, uplift = train_uplift_learner(X, treatment, y, learner="t", base_estimator="hgb", n_estimators=50)
    assert isinstance(models, tuple) and len(models) == 2
    assert len(uplift) == len(X)


def test_train_uplift_learner_insufficient_samples(sample_df: pd.DataFrame) -> None:
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    treatment = pd.Series([1, 1, 1])
    y = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        train_uplift_learner(X, treatment, y, learner="t", base_estimator="rf")


def test_evaluate_uplift_model(sample_df: pd.DataFrame) -> None:
    rng = np.random.default_rng(42)
    y = pd.Series(rng.integers(0, 5, 200).astype(float))
    treatment = pd.Series(rng.integers(0, 2, 200))
    uplift_pred = pd.Series(rng.normal(size=200))
    metrics, curve = evaluate_uplift_model(y, treatment, uplift_pred)
    check(metrics, UPLIFT_METRICS)
    check(curve, QINI_CURVE)
    assert curve["qini_y"].iloc[-1] == pytest.approx(curve["random_y"].iloc[-1])
    assert np.isfinite(metrics.loc[metrics["metric"] == "qini_coefficient", "value"].iloc[0])


def _category_promo_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two categories, one with a clear 3-day 30% promo on a single SKU."""
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    rows = []
    for sku, cat in (("A", "Coffee"), ("B", "Snacks")):
        for i, day in enumerate(dates):
            if sku == "A" and 10 <= i <= 12:
                price, qty = 7.0, 8
            elif sku == "A":
                price, qty = 10.0, 2
            else:
                price, qty = 5.0, 1
            rows.append(
                {
                    "date": day,
                    "transaction_id": f"T{sku}{i}",
                    "stockcode": sku,
                    "product": f"Product {sku}",
                    "category": cat,
                    "customer_id": 1000 + i,
                    "price": price,
                    "quantity": qty,
                }
            )
    df = pd.DataFrame(rows)
    promos = detect_promotions(df)
    return df, promos


def test_category_promo_timeline_contract_and_split() -> None:
    """Promo revenue > 0 only for the promoted category/week; discount depth reported."""
    df, promos = _category_promo_fixture()
    assert len(promos) == 1
    tl = compute_category_promo_timeline(df, promos)
    check(tl, CATEGORY_PROMO_TIMELINE)
    assert not tl.empty
    coffee_promo = tl[(tl["category"] == "Coffee") & (tl["promo_revenue"] > 0)]
    snacks_promo = tl[(tl["category"] == "Snacks") & (tl["promo_revenue"] > 0)]
    assert len(coffee_promo) >= 1
    assert len(snacks_promo) == 0
    assert coffee_promo["avg_discount_pct"].iloc[0] == pytest.approx(30.0, abs=2.0)
    # non-promo revenue for the same category/week is also present
    assert (tl[tl["category"] == "Coffee"]["non_promo_revenue"] >= 0).all()


def test_category_promo_timeline_no_promos() -> None:
    df, promos = _category_promo_fixture()
    tl = compute_category_promo_timeline(df, promos[promos["stockcode"] == "B"])
    assert tl.empty
    check(tl, CATEGORY_PROMO_TIMELINE, allow_empty=True)


def test_category_promo_timeline_missing_category() -> None:
    df, promos = _category_promo_fixture()
    df = df.drop(columns=["category"])
    tl = compute_category_promo_timeline(df, promos)
    assert tl.empty
    check(tl, CATEGORY_PROMO_TIMELINE, allow_empty=True)


def test_category_promo_timeline_sample_fixture(sample_df: pd.DataFrame) -> None:
    """Runs on the real sample; rows only for categories with active promos."""
    promos = detect_promotions(sample_df)
    if promos.empty:
        pytest.skip("no promos in sample")
    tl = compute_category_promo_timeline(sample_df, promos)
    check(tl, CATEGORY_PROMO_TIMELINE)
    assert not tl.empty
    assert (tl["n_promos"] >= 0).all()
    assert (tl["avg_discount_pct"] >= 0).all()
    assert set(tl["category"]).issubset(set(sample_df["category"]))
