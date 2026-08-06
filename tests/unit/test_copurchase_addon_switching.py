"""Tests for co-purchase, add-on, and switching analytics."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import contingency

from src.analytics.addon import (
    get_addon_by_category,
    get_addon_recommendations,
    get_anchor_addon_matrix,
)
from src.analytics.copurchase import (
    compute_affinity_matrix,
    get_product_affinity_profile,
    get_top_affinity_pairs,
)
from src.analytics.schemas import (
    ADDON_RECS,
    AFFINITY_PAIRS,
    LOYALTY_METRICS,
    SWITCHING_MATRIX,
    SchemaError,
)
from src.analytics.switching import (
    compute_switching_matrix,
    compute_transition_matrix,
    get_customer_loyalty_metrics,
    get_top_switching_paths,
)


def _theme_skus(sample_df: pd.DataFrame, theme: tuple[str, ...]) -> tuple[str, ...]:
    from src.analytics.data import derive_product_lookup

    lookup = derive_product_lookup(sample_df).set_index("product")["stockcode"].to_dict()
    skus = tuple(lookup[t] for t in theme if t in lookup)
    assert len(skus) == len(theme), "theme products must exist in the fixture"
    return skus


# --- copurchase ---


def test_affinity_matches_phi_formula() -> None:
    rng = np.random.default_rng(5)
    M = rng.random((300, 4)) > 0.6
    M[~M.any(axis=1)] = True
    df = _basket_df(M, ["P1", "P2", "P3", "P4"])
    affinity = compute_affinity_matrix(df, min_cooccurrence=1)
    for i in range(4):
        for j in range(i + 1, 4):
            n11 = int((M[:, i] & M[:, j]).sum())
            n00 = int((~M[:, i] & ~M[:, j]).sum())
            n10 = int((M[:, i] & ~M[:, j]).sum())
            n01 = int((~M[:, i] & M[:, j]).sum())
            expected = (n11 * n00 - n10 * n01) / np.sqrt(
                (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
            )
            got = affinity.iloc[i, j]
            assert np.isclose(got, expected, atol=1e-9), (i, j, got, expected)


def test_affinity_matches_sklearn_mcc() -> None:
    from sklearn.metrics import matthews_corrcoef

    rng = np.random.default_rng(11)
    M = rng.random((400, 3)) > 0.55
    M[~M.any(axis=1)] = True
    df = _basket_df(M, ["P1", "P2", "P3"])
    affinity = compute_affinity_matrix(df, min_cooccurrence=1)
    for i in range(3):
        for j in range(i + 1, 3):
            expected = matthews_corrcoef(M[:, i], M[:, j])
            assert np.isclose(affinity.iloc[i, j], expected, atol=1e-9)


def test_affinity_matrix_shape_and_diagonal(sample_df: pd.DataFrame) -> None:
    affinity = compute_affinity_matrix(sample_df, min_cooccurrence=5)
    n = sample_df["stockcode"].nunique()
    assert affinity.shape == (n, n)
    assert affinity.index.equals(affinity.columns)
    assert np.allclose(np.diag(affinity.to_numpy()), 1.0)


def test_affinity_min_cooccurrence_masks(sample_df: pd.DataFrame) -> None:
    strict = compute_affinity_matrix(sample_df, min_cooccurrence=100)
    loose = compute_affinity_matrix(sample_df, min_cooccurrence=1)
    assert strict.isna().sum().sum() >= loose.isna().sum().sum()


def test_affinity_positive_for_theme_pairs(sample_df: pd.DataFrame) -> None:
    affinity = compute_affinity_matrix(sample_df, min_cooccurrence=5)
    from src.analytics.sample_data import THEMES

    hits = 0
    for theme in THEMES:
        a, b, *_ = _theme_skus(sample_df, theme)
        value = affinity.loc[a, b]
        if not np.isnan(value):
            hits += 1
            assert value > 0, f"{theme[0]}/{theme[1]} theme pair should be positively associated"
    assert hits >= 5, "expected most theme pairs to pass the co-occurrence floor"


def test_top_affinity_pairs_contract(sample_df: pd.DataFrame) -> None:
    pairs = get_top_affinity_pairs(sample_df, top_n=10, min_cooccurrence=5)
    AFFINITY_PAIRS.validate(pairs)
    assert len(pairs) <= 10
    if len(pairs) > 1:
        assert pairs["affinity"].is_monotonic_decreasing


def test_top_affinity_pairs_no_duplicates(sample_df: pd.DataFrame) -> None:
    pairs = get_top_affinity_pairs(sample_df, top_n=50, min_cooccurrence=5)
    keys = set(zip(pairs["product_a"], pairs["product_b"]))
    assert len(keys) == len(pairs)


def test_product_affinity_profile(sample_df: pd.DataFrame) -> None:
    product = sample_df["stockcode"].iloc[0]
    profile = get_product_affinity_profile(sample_df, product)
    AFFINITY_PAIRS.validate(profile, allow_empty=True)
    assert set(profile["product_a"]).union(profile["product_b"]) - {product} == set(
        profile[["product_a", "product_b"]].stack().unique()
    ) - {product}


def test_affinity_unknown_product_empty(sample_df: pd.DataFrame) -> None:
    profile = get_product_affinity_profile(sample_df, "NO_SUCH_SKU")
    assert profile.empty


# --- add-on ---


def test_addon_recommendations_contract(sample_df: pd.DataFrame) -> None:
    anchor = sample_df["stockcode"].iloc[0]
    recs = get_addon_recommendations(sample_df, anchor, top_n=5, min_lift=1.0)
    ADDON_RECS.validate(recs, allow_empty=True)
    assert set(recs["anchor"]) == {anchor}
    assert len(recs) <= 5


def test_addon_lift_above_threshold(sample_df: pd.DataFrame) -> None:
    anchor = sample_df["stockcode"].iloc[0]
    recs = get_addon_recommendations(sample_df, anchor, top_n=20, min_lift=1.5)
    if not recs.empty:
        assert recs["lift"].min() >= 1.5 - 1e-9


def test_addon_theme_members_suggested(sample_df: pd.DataFrame) -> None:
    from src.analytics.sample_data import THEMES

    anchor = _theme_skus(sample_df, THEMES[4])[0]
    recs = get_addon_recommendations(sample_df, anchor, top_n=20, min_lift=1.0)
    partners = set(recs["addon"])
    assert any(t in partners for t in _theme_skus(sample_df, THEMES[4])[1:]), "theme partners should appear as add-ons"


def test_anchor_addon_matrix(sample_df: pd.DataFrame) -> None:
    matrix = get_anchor_addon_matrix(sample_df, top_n_anchors=10)
    assert len(matrix) <= 10
    assert (matrix.to_numpy() >= 0).all()


def test_addon_by_category(sample_df: pd.DataFrame) -> None:
    category = sample_df["category"].iloc[0]
    recs = get_addon_by_category(sample_df, category, top_n=5, min_lift=1.0)
    ADDON_RECS.validate(recs, allow_empty=True)
    anchors = set(sample_df[sample_df["category"] == category]["stockcode"])
    assert set(recs["anchor"]).issubset(anchors)
    assert set(recs["addon"]).isdisjoint(anchors)


# --- switching ---


def _switching_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01", "2024-01-10", "2024-02-10", "2024-03-10"] * 2
            ),
            "transaction_id": [f"T{i}" for i in range(12)],
            "stockcode": ["A", "B", "A", "B", "A", "B", "C", "A", "B", "C", "B", "A"],
            "product": ["p"] * 12,
            "customer_id": ["C1"] * 6 + ["C2"] * 6,
            "price": [1.0] * 12,
            "quantity": [1] * 12,
        }
    )


def test_switching_matrix_contract() -> None:
    matrix = compute_switching_matrix(_switching_df(), window_days=90, min_transactions=2)
    SWITCHING_MATRIX.validate(matrix)
    assert not matrix.empty
    assert "A" in matrix["from_product"].unique() or "B" in matrix["from_product"].unique()


def test_switching_matrix_window_filters() -> None:
    tight = compute_switching_matrix(_switching_df(), window_days=20, min_transactions=2)
    wide = compute_switching_matrix(_switching_df(), window_days=120, min_transactions=2)
    assert len(tight) <= len(wide)


def test_transition_matrix_rows_sum_to_one() -> None:
    pivot = compute_transition_matrix(_switching_df(), window_days=120, min_transactions=2)
    if not pivot.empty:
        np.testing.assert_allclose(pivot.sum(axis=1), 1.0, atol=1e-9)


def test_top_switching_paths_sorted() -> None:
    top = get_top_switching_paths(_switching_df(), top_n=3, window_days=120, min_transactions=2)
    SWITCHING_MATRIX.validate(top)
    if len(top) > 1:
        assert top["count"].is_monotonic_decreasing


def test_loyalty_metrics_contract(sample_df: pd.DataFrame) -> None:
    metrics = get_customer_loyalty_metrics(sample_df)
    LOYALTY_METRICS.validate(metrics)
    assert metrics["customer_id"].is_unique
    assert metrics["switching_rate"].between(0, 1).all()
    assert (metrics["n_transactions"] >= 1).all()


def test_loyalty_metrics_single_customer() -> None:
    df = _switching_df().head(6).copy()
    metrics = get_customer_loyalty_metrics(df)
    LOYALTY_METRICS.validate(metrics)
    assert metrics.loc[0, "n_transactions"] == 6
    assert metrics.loc[0, "repeat_purchase_rate"] == 1.0
    assert metrics.loc[0, "switching_count"] == 5
    assert metrics.loc[0, "switching_rate"] == 1.0


def test_switching_empty_on_static_customer() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
            "transaction_id": ["T1", "T2", "T3"],
            "stockcode": ["A", "A", "A"],
            "product": ["p"] * 3,
            "customer_id": ["C1"] * 3,
            "price": [1.0] * 3,
            "quantity": [1] * 3,
        }
    )
    matrix = compute_switching_matrix(df, window_days=90, min_transactions=2)
    assert matrix.empty


def _basket_df(M: np.ndarray, products: list[str]) -> pd.DataFrame:
    rows = []
    for txn_i, row in enumerate(M):
        for j, present in enumerate(row):
            if present:
                rows.append(
                    {
                        "date": pd.Timestamp("2024-01-01"),
                        "transaction_id": f"T{txn_i}",
                        "stockcode": products[j],
                        "product": "p",
                        "customer_id": f"C{txn_i % 10}",
                        "price": 1.0,
                        "quantity": 1,
                    }
                )
    return pd.DataFrame(rows)
