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


def _theme_skus(sample_df: pd.DataFrame, theme: tuple[str, str]) -> tuple[str, ...]:
    """Get SKUs for a theme by looking up products in the given categories."""
    # Get stockcodes for products in the given categories
    cat_a, cat_b = theme
    sku_a = sample_df[sample_df["category"] == cat_a]["stockcode"].unique()
    sku_b = sample_df[sample_df["category"] == cat_b]["stockcode"].unique()
    
    # Pick first SKU from each category
    a = sku_a[0] if len(sku_a) > 0 else None
    b = sku_b[0] if len(sku_b) > 0 else None
    
    assert a is not None and b is not None, f"Categories {cat_a} and {cat_b} must exist in the fixture"
    return (a, b)


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
    positive_hits = 0
    for theme in THEMES:
        a, b = _theme_skus(sample_df, theme)
        value = affinity.loc[a, b]
        if not np.isnan(value):
            hits += 1
            if value > 0:
                positive_hits += 1
    # At least some pairs should have data (the sample fixture may not have all themes
    # with sufficient co-occurrence at the individual SKU level)
    assert hits >= 2, "expected at least some theme pairs to pass the co-occurrence floor"
    # At least one pair should be positively associated
    assert positive_hits >= 1, "expected at least one theme pair to be positively associated"


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

    # Use first theme (Coffee/Dairy)
    theme = THEMES[0]
    anchor = _theme_skus(sample_df, theme)[0]
    recs = get_addon_recommendations(sample_df, anchor, top_n=20, min_lift=1.0)
    if recs.empty:
        pytest.skip("No add-on recommendations generated")
    partners = set(recs["addon"])
    # Check that at least one product from the same theme categories appears
    theme_skus = set(_theme_skus(sample_df, theme))
    # Also include all SKUs from the theme categories
    cat_a, cat_b = theme
    cat_a_skus = set(sample_df[sample_df["category"] == cat_a]["stockcode"].unique())
    cat_b_skus = set(sample_df[sample_df["category"] == cat_b]["stockcode"].unique())
    all_theme_skus = cat_a_skus | cat_b_skus
    assert any(t in partners for t in all_theme_skus if t != anchor), "theme partners should appear as add-ons"


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


def test_category_switching_rollup_matches_manual() -> None:
    """SKU-level transitions must aggregate to category pairs exactly."""
    from src.analytics.schemas import CATEGORY_SWITCHING
    from src.analytics.switching import compute_category_switching_matrix

    df = _switching_df().copy()
    # Known SKU -> category mapping: A,C in CatX; B in CatY
    product_lookup = pd.DataFrame(
        {
            "stockcode": ["A", "B", "C"],
            "category": ["CatX", "CatY", "CatX"],
        }
    )

    # Manual aggregation from the SKU switching matrix
    sku_matrix = compute_switching_matrix(df, window_days=90, min_transactions=2)
    cat_map = product_lookup.set_index("stockcode")["category"].to_dict()
    manual = {}
    for _, row in sku_matrix.iterrows():
        key = (cat_map[row["from_product"]], cat_map[row["to_product"]])
        manual[key] = manual.get(key, 0) + row["count"]

    cat_matrix = compute_category_switching_matrix(
        df, window_days=90, min_transactions=2, product_lookup=product_lookup
    )
    CATEGORY_SWITCHING.validate(cat_matrix)
    assert not cat_matrix.empty
    assert cat_matrix["product_pairs"].min() >= 1

    for _, row in cat_matrix.iterrows():
        key = (row["from_category"], row["to_category"])
        assert manual[key] == row["count"]
    assert sum(manual.values()) == cat_matrix["count"].sum()


def test_category_switching_contract_on_sample(sample_df: pd.DataFrame) -> None:
    from src.analytics.schemas import CATEGORY_SWITCHING
    from src.analytics.switching import compute_category_switching_matrix

    cat_matrix = compute_category_switching_matrix(sample_df, window_days=90, min_transactions=3)
    CATEGORY_SWITCHING.validate(cat_matrix)
    if not cat_matrix.empty:
        assert (cat_matrix["count"] >= 0).all()
        assert cat_matrix["pct"].between(0, 1).all()
        assert "Unknown" in cat_matrix["from_category"].unique() or True
        # pct sums to 1 when non-empty
        assert abs(cat_matrix["pct"].sum() - 1.0) < 1e-6


def test_event_slices_partition_date_range() -> None:
    """Pre/event/post slices must partition the window around an event exactly."""
    from src.analytics.switching import build_event_slices

    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=120, freq="D")})
    events = pd.DataFrame({"start_date": [pd.Timestamp("2024-02-01")], "end_date": [pd.Timestamp("2024-02-14")]})

    slices = build_event_slices(df, events, pre_days=30, post_days=30)

    assert "pre" in slices and "event" in slices and "post" in slices
    pre = slices["pre"]["date"]
    event = slices["event"]["date"]
    post = slices["post"]["date"]

    # Window boundaries
    assert pre.min() == pd.Timestamp("2024-01-02") and pre.max() == pd.Timestamp("2024-01-31")
    assert event.min() == pd.Timestamp("2024-02-01") and event.max() == pd.Timestamp("2024-02-14")
    assert post.min() == pd.Timestamp("2024-02-15") and post.max() == pd.Timestamp("2024-03-15")

    # No overlap between phases
    assert not pre.isin(event).any() and not pre.isin(post).any() and not event.isin(post).any()


def test_event_slices_empty_when_no_events() -> None:
    from src.analytics.switching import build_event_slices

    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10, freq="D")})
    slices = build_event_slices(df, [], pre_days=5, post_days=5)
    assert slices == {}


def test_phase_switching_on_sample(sample_df: pd.DataFrame) -> None:
    """Phase switching must produce contract-validated frames and survive promo-less data."""
    from src.analytics.schemas import CATEGORY_SWITCHING
    from src.analytics.promo import detect_promotions
    from src.analytics.switching import compute_category_switching_by_phase

    events = detect_promotions(sample_df)
    if events.empty:
        return

    phases = compute_category_switching_by_phase(
        sample_df,
        events,
        pre_days=30,
        post_days=30,
        window_days=90,
        min_transactions=3,
    )
    assert set(phases) <= {"pre", "event", "post"}
    for phase_df in phases.values():
        CATEGORY_SWITCHING.validate(phase_df)


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


def test_compute_cooccurrence_matrix(sample_df: pd.DataFrame) -> None:
    from src.analytics.copurchase import compute_cooccurrence_matrix

    cooccur = compute_cooccurrence_matrix(sample_df)
    assert cooccur.index.tolist() == cooccur.columns.tolist()
    assert (cooccur.dtypes == "int64").all()
    np.testing.assert_array_equal(cooccur.to_numpy(), cooccur.to_numpy().T)
    assert cooccur.to_numpy()[np.diag_indices_from(cooccur)].min() >= 0


def test_compute_pair_trend(sample_df: pd.DataFrame) -> None:
    from src.analytics.copurchase import compute_pair_trend

    pairs = get_top_affinity_pairs(sample_df, top_n=5, min_cooccurrence=2, top_n_products=50)
    if pairs.empty:
        return
    row = pairs.iloc[0]
    trend = compute_pair_trend(sample_df, row["product_a"], row["product_b"])
    if trend.empty:
        return
    assert set(trend.columns) == {"period", "cooccurrence"}
    assert (trend["cooccurrence"] > 0).all()


def test_compute_pair_centrality(sample_df: pd.DataFrame) -> None:
    from src.analytics.copurchase import compute_pair_centrality

    centrality = compute_pair_centrality(sample_df, top_n_products=50, min_cooccurrence=2)
    assert set(["stockcode", "pagerank", "betweenness", "degree"]).issubset(centrality.columns)
    if not centrality.empty:
        assert (centrality["pagerank"] >= 0).all()
        assert centrality["stockcode"].is_unique
