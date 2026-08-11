"""Unit tests for the Segmentation package."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.schemas import (
    BEHAVIORAL_FEATURES,
    BEHAVIORAL_SEGMENTS,
    CLUSTER_QUALITY,
    CLUSTER_STABILITY,
    RFM_FEATURES,
    RFM_SEGMENTS,
    SEGMENT_MIGRATION,
    SEGMENT_RADAR,
    SURVIVAL_DIAGNOSTICS,
    SURVIVAL_PREDICTIONS,
    VALUE_BASED_SEGMENTS,
)
from src.analytics.segmentation import (
    behavioral_segmentation,
    compute_cluster_quality_metrics,
    compute_cluster_stability,
    compute_rfm_features,
    compute_segment_migration,
    compute_segment_radar,
    format_quality_metrics,
    format_stability_metrics,
    rfm_segmentation,
    survival_analysis,
    value_based_segmentation,
)


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    from src.analytics.data import load_transactions

    df, _, _, _ = load_transactions("sample_data/sample_transactions.csv")
    return df


def test_rfm_features_contract(sample_df: pd.DataFrame) -> None:
    rfm = compute_rfm_features(sample_df)
    RFM_FEATURES.validate(rfm)
    assert rfm["customer_id"].nunique() == rfm.shape[0]
    assert rfm["frequency"].min() >= 1
    assert (rfm["monetary"] > 0).all()


def test_rfm_single_line_item_customers_no_nan_cv() -> None:
    """Regression: single-line-item customers had std_order_value = NaN, so
    order_value_cv was NaN and the 'must be non-negative' validator counted
    NaN as a violation (NaN >= 0 is False), raising SchemaError.
    """
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=6, freq="D"),
            "transaction_id": ["T1", "T1", "T2", "T2", "T3", "T4"],
            "customer_id": ["A", "A", "B", "B", "C", "D"],
            "stockcode": ["P1", "P2", "P1", "P3", "P1", "P2"],
            "price": [10.0, 5.0, 10.0, 7.0, 4.0, 6.0],
            "quantity": [1, 2, 1, 1, 2, 1],
        }
    )
    rfm = compute_rfm_features(df)
    single = rfm.set_index("customer_id").loc[["C", "D"]]
    assert (single["order_value_cv"] == 0).all()
    assert rfm["order_value_cv"].notna().all()
    assert (rfm["order_value_cv"] >= 0).all()
    RFM_FEATURES.validate(rfm)


def test_rfm_quantile_segments(sample_df: pd.DataFrame) -> None:
    rfm = compute_rfm_features(sample_df)
    segments = rfm_segmentation(rfm, method="quantile")
    RFM_SEGMENTS.validate(segments)
    assert segments["segment"].notna().all()
    assert set(segments["segment"]).intersection({"Champions", "Loyal", "Lost", "Other"})
    # Scores are categorical with values 4,3,2,1 - check they map to valid range
    assert segments["recency_score"].notna().all()


def test_rfm_kmeans_segments(sample_df: pd.DataFrame) -> None:
    rfm = compute_rfm_features(sample_df)
    segments = rfm_segmentation(rfm, method="kmeans", n_segments=5)
    RFM_SEGMENTS.validate(segments)
    assert segments["cluster"].between(0, 4).all()
    assert segments["segment"].notna().all()


def test_behavioral_features(sample_df: pd.DataFrame) -> None:
    from src.analytics.segmentation.behavioral import _create_behavioral_features_pandas

    features = _create_behavioral_features_pandas(sample_df)
    BEHAVIORAL_FEATURES.validate(features)
    assert features["customer_id"].nunique() == features.shape[0]


def test_behavioral_segmentation_kmeans(sample_df: pd.DataFrame) -> None:
    segments, metrics = behavioral_segmentation(
        sample_df, n_clusters=4, return_metrics=True, method="kmeans"
    )
    BEHAVIORAL_SEGMENTS.validate(segments)
    # Some clusters may be dropped as outliers (-1)
    assert segments["cluster"].ge(-1).all()
    assert segments["segment"].notna().all()
    assert "cluster_confidence" in segments.columns


def test_behavioral_segmentation_gmm(sample_df: pd.DataFrame) -> None:
    segments = behavioral_segmentation(sample_df, n_clusters=3, method="gmm")
    BEHAVIORAL_SEGMENTS.validate(segments)
    assert segments["cluster"].between(0, 2).all()


def test_behavioral_min_samples_fallback(sample_df: pd.DataFrame) -> None:
    segments = behavioral_segmentation(sample_df, n_clusters=100, method="kmeans")
    BEHAVIORAL_SEGMENTS.validate(segments)
    assert (segments["segment"] == "Other").all()


def test_survival_analysis(sample_df: pd.DataFrame) -> None:
    surv, diag = survival_analysis(sample_df, prediction_horizon_days=30)
    SURVIVAL_PREDICTIONS.validate(surv)
    SURVIVAL_DIAGNOSTICS.validate(diag)
    assert surv["survival_prob"].between(0, 1).all()
    assert surv["churn_risk"].between(0, 1).all()
    assert diag["metric"].isin(["concordance_index", "n_events", "n_censored", "model_params"]).all()


def test_value_based_segmentation(sample_df: pd.DataFrame) -> None:
    segments = value_based_segmentation(sample_df, prediction_horizon_days=60)
    VALUE_BASED_SEGMENTS.validate(segments)
    assert segments["value_segment"].isin(
        {"VIP", "High Potential", "Loyal", "New", "Churned", "Regular"}
    ).all()
    assert segments["predicted_clv"].notna().all()


def test_cluster_quality_metrics() -> None:
    X = np.random.randn(50, 5)
    labels = np.random.randint(0, 3, 50)
    metrics = compute_cluster_quality_metrics(X, labels)
    assert "silhouette_score" in metrics
    assert "davies_bouldin_score" in metrics
    assert "n_clusters" in metrics

    df = format_quality_metrics(metrics)
    CLUSTER_QUALITY.validate(df)


def test_cluster_stability(sample_df: pd.DataFrame) -> None:
    stability = compute_cluster_stability(
        transactions_df=sample_df, n_clusters=4, n_iterations=3, seed=42
    )
    assert "mean_ari" in stability
    assert 0 <= stability["mean_ari"] <= 1

    df = format_stability_metrics(stability)
    CLUSTER_STABILITY.validate(df)


def test_invalid_methods_raise(sample_df: pd.DataFrame) -> None:
    rfm = compute_rfm_features(sample_df)
    with pytest.raises(ValueError):
        rfm_segmentation(rfm, method="unknown")
    with pytest.raises(ValueError):
        behavioral_segmentation(sample_df, method="unknown")


def test_segment_radar_contract(sample_df: pd.DataFrame) -> None:
    radar = compute_segment_radar(sample_df, n_clusters=4)
    SEGMENT_RADAR.validate(radar)
    assert not radar.empty
    # one row per (segment, feature)
    pairs = radar[["segment", "feature"]].drop_duplicates()
    assert len(pairs) == len(radar)
    # each feature min-max spans [0, 1] across segments
    per_feature = radar.groupby("feature")["normalized_value"]
    assert (per_feature.max() - per_feature.min()).abs().max() <= 1.0
    assert per_feature.max().max() <= 1.0 and per_feature.min().min() >= 0.0
    # means validate non-negative
    assert (radar["mean_value"] >= 0).all()


def test_segment_radar_deterministic(sample_df: pd.DataFrame) -> None:
    a = compute_segment_radar(sample_df, n_clusters=4)
    b = compute_segment_radar(sample_df, n_clusters=4)
    pd.testing.assert_frame_equal(a, b)


def test_segment_migration_contract(sample_df: pd.DataFrame) -> None:
    migration = compute_segment_migration(sample_df, n_clusters=4)
    SEGMENT_MIGRATION.validate(migration)
    assert not migration.empty
    assert set(migration["period_from"]).issubset({"first_half"})
    assert set(migration["period_to"]).issubset({"second_half"})
    assert (migration["customers"] > 0).all()
    # off-diagonal flows do not count as retention
    off_diag = migration[migration["segment_from"] != migration["segment_to"]]
    assert (off_diag["retention_rate"] == 0).all()


def test_segment_migration_empty_on_tiny_sample() -> None:
    tiny = pd.DataFrame(
        {
            "date": list(pd.date_range("2025-01-01", periods=20)),
            "transaction_id": [f"I{i}" for i in range(20)],
            "stockcode": ["A"] * 20,
            "product": ["a"] * 20,
            "customer_id": ["c1", "c2"] * 10,
            "price": [1.0] * 20,
            "quantity": [1] * 20,
        }
    )
    migration = compute_segment_migration(tiny, n_clusters=4)
    # either empty or passes contract
    if not migration.empty:
        SEGMENT_MIGRATION.validate(migration)