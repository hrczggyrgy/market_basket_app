"""Unit tests for the Pricing package."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.pricing import (
    classify_elasticity_confidence,
    compute_kvi_score,
    diagnose_price_curves_1d,
    diagnose_price_curves_multivariate,
    estimate_cross_price_elasticity,
    estimate_hierarchical_elasticity,
    estimate_iv_elasticity,
    estimate_loglog_elasticity,
    estimate_rdd_elasticity,
    estimate_synthetic_control_elasticity,
)
from src.analytics.schemas import (
    CROSS_ELASTICITY,
    ELASTICITY,
    ELASTICITY_CONFIDENCE,
    HIERARCHICAL_ELASTICITY,
    IV_ELASTICITY,
    KVI_SCORES,
    PRICE_CURVE_1D,
    PRICE_CURVE_MULTI,
    RDD_ELASTICITY,
    SYNTHETIC_CONTROL,
)


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    from src.analytics.data import load_transactions

    df, _, _, _ = load_transactions("sample_data/sample_transactions.csv")
    return df


def test_loglog_elasticity(sample_df: pd.DataFrame) -> None:
    elast = estimate_loglog_elasticity(sample_df, min_periods=5)
    ELASTICITY.validate(elast, allow_empty=True)
    if not elast.empty:
        assert elast["elasticity"].notna().all()
        assert elast["n_obs"].ge(5).all()


def test_hierarchical_elasticity(sample_df: pd.DataFrame) -> None:
    elast = estimate_hierarchical_elasticity(sample_df, min_periods=5)
    HIERARCHICAL_ELASTICITY.validate(elast, allow_empty=True)
    if not elast.empty:
        assert elast["elasticity_shrunk"].notna().all()
        assert elast["shrink_weight"].between(0.05, 0.95).all()


def test_cross_price_elasticity(sample_df: pd.DataFrame) -> None:
    revenue = (sample_df["price"] * sample_df["quantity"]).groupby(sample_df["stockcode"]).sum()
    top5 = revenue.nlargest(5).index.tolist()
    pairs = [(top5[i], top5[j]) for i in range(len(top5)) for j in range(i + 1, len(top5))][:5]
    cross = estimate_cross_price_elasticity(sample_df, pairs, min_periods=5)
    CROSS_ELASTICITY.validate(cross, allow_empty=True)
    if not cross.empty:
        assert cross["n_obs"].ge(5).all()


def test_kvi_heuristic(sample_df: pd.DataFrame) -> None:
    kvi = compute_kvi_score(sample_df, method="heuristic")
    KVI_SCORES.validate(kvi, allow_empty=True)
    if not kvi.empty:
        assert kvi["kvi_score"].notna().all()


def test_kvi_receives_elasticity_and_category(sample_df: pd.DataFrame) -> None:
    """KVI must receive elasticity_df so abs_elasticity reflects real estimates,
    and carry the category column (provided or inferred) instead of UNKNOWN."""
    elast = estimate_loglog_elasticity(sample_df, min_periods=5)
    kvi = compute_kvi_score(sample_df, elasticity_df=elast, method="heuristic")
    KVI_SCORES.validate(kvi, allow_empty=True)
    if elast.empty:
        return
    assert kvi["abs_elasticity"].sum() > 0
    assert set(sample_df["category"].unique()).issubset(set(kvi["category"].unique()))


def test_kvi_xgb_requires_xgboost(sample_df: pd.DataFrame, monkeypatch) -> None:
    import sys
    import src.analytics.pricing.kvi as kvi_mod
    # If xgboost/shap not installed, should fall back to heuristic
    monkeypatch.setitem(sys.modules, "xgboost", None)
    monkeypatch.setitem(sys.modules, "shap", None)
    kvi = compute_kvi_score(sample_df, method="xgb")
    KVI_SCORES.validate(kvi, allow_empty=True)


def test_price_curves_1d(sample_df: pd.DataFrame) -> None:
    curves = diagnose_price_curves_1d(sample_df, n_tiers=3)
    PRICE_CURVE_1D.validate(curves, allow_empty=True)
    if not curves.empty:
        assert curves["tier_label"].isin({"Value", "Mainstream", "Premium", "Ultra", "Luxury"}).all()
        assert "has_violation" in curves.columns


def test_price_curves_1d_empty_categories() -> None:
    """Test diagnose_price_curves_1d with categories having fewer than n_tiers products.
    
    This regression test ensures the fix for KeyError: "['tier', 'tier_label'] not in index"
    when categories have fewer products than n_tiers.
    """
    # Create synthetic data where each category has only 1 product (less than n_tiers=3)
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=20, freq="D"),
        "transaction_id": [f"T{i}" for i in range(20)],
        "stockcode": [f"SKU{i}" for i in range(20)],
        "product": [f"Product {i}" for i in range(20)],
        "customer_id": [f"C{i}" for i in range(20)],
        "price": [10.0 + i * 0.5 for i in range(20)],
        "quantity": [1] * 20,
        "category": [f"Cat{i}" for i in range(20)],  # Each product in its own category
        "brand": ["Brand A"] * 20,
        "size": ["1L"] * 20,
    })
    
    curves = diagnose_price_curves_1d(df, n_tiers=3)
    PRICE_CURVE_1D.validate(curves, allow_empty=True)
    
    # Should not crash and should have correct columns even with 0 rows or categories < n_tiers
    assert "tier" in curves.columns
    assert "tier_label" in curves.columns
    assert "has_violation" in curves.columns
    assert "stockcode" in curves.columns
    assert "product_name" in curves.columns
    assert "category" in curves.columns
    assert "tier_label" in curves.columns


def test_price_curves_1d_normal_categories() -> None:
    """Test diagnose_price_curves_1d with categories having enough products for tiering."""
    # Create synthetic data where categories have enough products for tiering
    np.random.seed(42)
    n_products = 15
    n_customers = 10
    n_transactions = 30 * n_products
    
    stockcodes = [f"SKU{i}" for i in range(n_products)]
    products = [f"Product {i}" for i in range(n_products)]
    categories = [f"Cat{i % 3}" for i in range(n_products)]
    
    # Generate transactions
    np.random.seed(42)
    rows = []
    for _ in range(n_transactions):
        idx = np.random.randint(0, n_products)
        rows.append({
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=np.random.randint(0, 30)),
            "transaction_id": f"T{np.random.randint(100000, 999999)}",
            "stockcode": stockcodes[idx],
            "product": products[idx],
            "customer_id": f"C{np.random.randint(0, 10)}",
            "price": np.random.uniform(5.0, 20.0),
            "quantity": np.random.randint(1, 5),
            "category": categories[idx],
            "brand": "Brand A",
            "size": "1L",
        })
    df = pd.DataFrame(rows)
    
    curves = diagnose_price_curves_1d(df, n_tiers=3)
    PRICE_CURVE_1D.validate(curves, allow_empty=True)
    
    if not curves.empty:
        # With 15 products across 3 categories (5 per category), should have tiering
        assert "tier" in curves.columns
        assert "tier_label" in curves.columns
        # Should have tier labels from the expected set
        assert curves["tier_label"].isin({"Value", "Mainstream", "Premium", "Ultra", "Luxury"}).all()


def test_price_curves_multivariate(sample_df: pd.DataFrame) -> None:
    elast = estimate_loglog_elasticity(sample_df, min_periods=5)
    curves = diagnose_price_curves_multivariate(sample_df, elasticity_df=elast, n_tiers=3)
    PRICE_CURVE_MULTI.validate(curves, allow_empty=True)
    if not curves.empty:
        assert curves["tier_label"].notna().all()


def test_iv_elasticity(sample_df: pd.DataFrame) -> None:
    iv = estimate_iv_elasticity(sample_df, instrument_col="cost", min_periods=5)
    IV_ELASTICITY.validate(iv, allow_empty=True)
    if not iv.empty:
        assert iv["weak_instrument"].notna().all()


def test_rdd_elasticity(sample_df: pd.DataFrame) -> None:
    rdd = estimate_rdd_elasticity(sample_df, min_periods=5, bandwidth=0.5)
    RDD_ELASTICITY.validate(rdd, allow_empty=True)


def test_synthetic_control_elasticity(sample_df: pd.DataFrame) -> None:
    revenue = (sample_df["price"] * sample_df["quantity"]).groupby(sample_df["stockcode"]).sum()
    top_products = revenue.nlargest(5).index.tolist()
    treatment = top_products[0]
    donors = top_products[1:4]
    sc = estimate_synthetic_control_elasticity(
        sample_df, treatment, donors, pre_periods=5, post_periods=3
    )
    SYNTHETIC_CONTROL.validate(sc)
    required = {"treatment_effect_log", "treatment_effect_pct", "pre_period_rmse", "n_donors"}
    assert required <= set(sc["metric"])


def test_loglog_elasticity_excludes_degenerate_cases() -> None:
    """Regression: SKUs with constant qty or too few distinct prices should be
    excluded cleanly, not produce NaN p_value that violates schema."""
    from src.analytics.data import load_transactions
    import io

    # SKU with constant quantity across weeks, varying price
    dates = pd.date_range("2025-01-01", periods=20, freq="W")
    raw = pd.DataFrame({
        "date": dates.tolist(),
        "transaction_id": [f"T{i}" for i in range(1, 21)],
        "stockcode": ["CONST_QTY"] * 20,
        "product": ["p"] * 20,
        "customer_id": ["C1"] * 20,
        "price": [10]*5 + [12]*5 + [15]*5 + [20]*5,
        "quantity": [100]*20,
    })
    df, *_ = load_transactions(io.BytesIO(raw.to_csv(index=False).encode()))
    
    elast = estimate_loglog_elasticity(df, min_periods=5, use_robust_se=True)
    # Should be excluded (0 SKUs), not crash or produce NaN p_value
    assert len(elast) == 0
    ELASTICITY.validate(elast, allow_empty=True)


def test_loglog_elasticity_normal_case_unchanged(sample_df: pd.DataFrame) -> None:
    """Regression: normal well-varied SKUs should produce valid estimates."""
    elast = estimate_loglog_elasticity(sample_df, min_periods=5, use_robust_se=True)
    ELASTICITY.validate(elast, allow_empty=True)
    if not elast.empty:
        assert elast["elasticity"].notna().all()
        assert elast["p_value"].notna().all()
        assert (elast["p_value"] >= 0).all() and (elast["p_value"] <= 1).all()
        assert elast["r_squared"].between(0, 1).all()
        assert (elast["n_obs"] >= 5).all()


def test_loglog_elasticity_few_distinct_prices_excluded() -> None:
    """SKU with only 2 distinct prices should be excluded."""
    from src.analytics.data import load_transactions
    import io

    dates = pd.date_range("2025-01-01", periods=20, freq="W")
    raw = pd.DataFrame({
        "date": dates.tolist(),
        "transaction_id": [f"T{i}" for i in range(1, 21)],
        "stockcode": ["TWO_PRICES"] * 20,
        "product": ["p"] * 20,
        "customer_id": ["C1"] * 20,
        "price": [10]*10 + [20]*10,
        "quantity": [100, 90, 80, 70, 60]*2 + [50, 45, 40, 35, 30]*2,
    })
    df, *_ = load_transactions(io.BytesIO(raw.to_csv(index=False).encode()))
    
    elast = estimate_loglog_elasticity(df, min_periods=5, use_robust_se=True)
    assert len(elast) == 0
    ELASTICITY.validate(elast, allow_empty=True)


def test_hierarchical_elasticity_excludes_degenerate_cases() -> None:
    """Hierarchical estimation should also skip degenerate SKUs."""
    from src.analytics.data import load_transactions
    import io

    dates = pd.date_range("2025-01-01", periods=20, freq="W")
    raw = pd.DataFrame({
        "date": dates.tolist() * 2,
        "transaction_id": [f"T{i}" for i in range(1, 41)],
        "stockcode": ["CONST_QTY"] * 20 + ["NORMAL"] * 20,
        "product": ["p"] * 40,
        "customer_id": ["C1"] * 40,
        "price": ([10]*5 + [12]*5 + [15]*5 + [20]*5) + ([10, 12, 15, 11, 13, 14, 16, 10, 12, 15, 11, 13, 14, 16, 10, 12, 15, 11, 13, 14]),
        "quantity": [100]*20 + [100, 90, 80, 95, 85, 88, 75, 105, 92, 78, 93, 87, 82, 77, 102, 91, 79, 94, 86, 83],
    })
    df, *_ = load_transactions(io.BytesIO(raw.to_csv(index=False).encode()))
    
    hier = estimate_hierarchical_elasticity(df, min_periods=5)
    # CONST_QTY should be excluded, only NORMAL remains
    assert len(hier) == 1
    assert hier.iloc[0]["stockcode"] == "NORMAL"
    HIERARCHICAL_ELASTICITY.validate(hier, allow_empty=True)


def _elast_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stockcode": ["TIGHT", "WIDE", "INSIG", "UNIT", "NEG"],
            "elasticity": [-1.5, -0.4, -2.0, -1.0, -1.2],
            "ci_lower": [-1.6, -1.3, -3.5, -1.4, -3.0],
            "ci_upper": [-1.4, 0.5, -0.5, -0.6, 0.6],
            "p_value": [0.001, 0.2, 0.4, 0.01, 0.9],
            "n_obs": [40, 15, 12, 30, 10],
            "r_squared": [0.7, 0.3, 0.5, 0.8, 0.2],
            "std_err": [0.1, 0.4, 0.8, 0.2, 0.9],
            "avg_price": [10.0, 5.0, 8.0, 12.0, 7.0],
            "avg_weekly_qty": [100.0, 50.0, 60.0, 120.0, 80.0],
            "price_cv": [0.2, 0.3, 0.4, 0.25, 0.35],
        }
    )


def test_elasticity_confidence_contract() -> None:
    conf = classify_elasticity_confidence(_elast_fixture())
    ELASTICITY_CONFIDENCE.validate(conf)
    assert len(conf) == 5
    # TIGHT: significant + relative width 0.5/1.5 = 0.33 -> high
    assert conf.loc[conf["stockcode"] == "TIGHT", "confidence"].iloc[0] == "high"
    # WIDE: relative width 1.8/0.4 = huge -> low
    assert conf.loc[conf["stockcode"] == "WIDE", "confidence"].iloc[0] == "low"
    # INSIG: not significant, relative width 3.0/2.0 = 1.5 -> medium (tight CI but not significant)


def test_elasticity_confidence_empty() -> None:
    conf = classify_elasticity_confidence(pd.DataFrame())
    assert conf.empty
    ELASTICITY_CONFIDENCE.validate(conf, allow_empty=True)


def test_elasticity_confidence_round_trip(sample_df: pd.DataFrame) -> None:
    elast = estimate_loglog_elasticity(sample_df, min_periods=5)
    if elast.empty:
        pytest.skip("sample data does not support elasticity estimation")
    conf = classify_elasticity_confidence(elast)
    ELASTICITY_CONFIDENCE.validate(conf, allow_empty=True)
    if not conf.empty:
        assert set(conf["stockcode"]).issubset(set(elast["stockcode"]))