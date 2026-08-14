"""Behavioral Segmentation based on purchase patterns."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.analytics.schemas import (
    BEHAVIORAL_FEATURES,
    BEHAVIORAL_SEGMENTS,
    SEGMENT_MIGRATION,
    SEGMENT_RADAR,
    check,
)
from src.analytics.segmentation.core import (
    MIN_CLUSTER_SIZE,
    _label_behavioral_clusters,
    compute_cluster_quality_metrics,
)


def _create_behavioral_features_pandas(
    transactions_df: pd.DataFrame,
    as_of_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Create behavioral features for each customer using pandas.

    Args:
        transactions_df: Transaction data
        as_of_date: If provided, compute features ONLY on data up to this date
            (temporal holdout to prevent leakage).
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # Apply temporal holdout if as_of_date provided
    if as_of_date is not None:
        df = df[df["date"] <= pd.Timestamp(as_of_date)]

    cat_col_behav = "category" if "category" in df.columns else "stockcode"

    behavioral = (
        df.groupby("customer_id")
        .agg(
            days_active=("date", lambda x: (x.max() - x.min()).days + 1),
            purchase_frequency=("transaction_id", "nunique"),
            avg_days_between=(
                "date",
                lambda x: (x.max() - x.min()).days / max(x.nunique() - 1, 1),
            ),
            total_revenue=("revenue", "sum"),
            avg_order_value=("revenue", "mean"),
            revenue_std=("revenue", "std"),
            n_products=("stockcode", "nunique"),
            n_categories=(cat_col_behav, "nunique"),
            avg_basket_size=("quantity", "mean"),
            max_basket_size=("quantity", "max"),
            avg_price=("price", "mean"),
            price_cv=("price", lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
            weekend_ratio=("date", lambda x: (x.dt.dayofweek >= 5).mean()),
        )
        .reset_index()
    )

    behavioral = behavioral.fillna(0)
    return check(behavioral, BEHAVIORAL_FEATURES)


def behavioral_segmentation(
    transactions_df: pd.DataFrame,
    n_clusters: int = 6,
    return_metrics: bool = False,
    method: str = "kmeans",
    interactive: bool = True,  # WF-5: adaptive n_init
    as_of_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Behavioral segmentation based on purchase patterns.

    Args:
        transactions_df: Transaction data
        n_clusters: Number of clusters
        return_metrics: Also return cluster quality metrics dict
        method: Clustering algorithm ('kmeans', 'gmm')
        interactive: If True, use n_init=3 for fast interactive runs; if False, use n_init=10 for quality
        as_of_date: If provided, compute features ONLY on data up to this date
            (temporal holdout to prevent leakage). Use for training on historical data.

    Returns:
        DataFrame with cluster assignments; if return_metrics is True,
        returns (DataFrame, metrics_dict).
    """
    behavioral = _create_behavioral_features_pandas(transactions_df, as_of_date=as_of_date)

    feature_cols = [c for c in behavioral.columns if c != "customer_id"]
    n_samples = len(behavioral)

    # Minimum sample-size guard
    min_required = max(n_clusters * MIN_CLUSTER_SIZE, 10)
    if n_samples < min_required:
        behavioral["cluster"] = 0
        behavioral["segment"] = "Other"
        behavioral["cluster_distance"] = 0.0
        behavioral["cluster_confidence"] = 1.0
        result = check(behavioral, BEHAVIORAL_SEGMENTS)
        if return_metrics:
            return result, {}
        return result

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(behavioral[feature_cols])

    # WF-5: Adaptive n_init - 3 for interactive, 10 for final quality run
    n_init = 3 if interactive else 10

    if method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=n_init)
        behavioral["cluster"] = kmeans.fit_predict(X_scaled)

        # Distance-to-centroid confidence
        distances = kmeans.transform(X_scaled)
        behavioral["cluster_distance"] = distances.min(axis=1)
        max_dist = distances.max(axis=1)
        behavioral["cluster_confidence"] = np.where(
            max_dist > 0, 1 - behavioral["cluster_distance"] / max_dist, 1.0
        )
    elif method == "gmm":
        gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=5)
        behavioral["cluster"] = gmm.fit_predict(X_scaled)
        behavioral["cluster_distance"] = 1.0 - gmm.predict_proba(X_scaled).max(axis=1)
        behavioral["cluster_confidence"] = 1.0 - behavioral["cluster_distance"]
    else:
        raise ValueError(f"Unknown method: {method}")

    # Cluster quality metrics
    quality_metrics = compute_cluster_quality_metrics(X_scaled, behavioral["cluster"].values)

    # Label clusters
    cluster_profiles = behavioral.groupby("cluster")[feature_cols].mean()
    labels = _label_behavioral_clusters(cluster_profiles)

    cluster_sizes = behavioral["cluster"].value_counts()
    small_clusters = cluster_sizes[cluster_sizes < MIN_CLUSTER_SIZE].index
    if not small_clusters.empty and len(small_clusters) < n_clusters:
        for sc in small_clusters:
            if sc == -1:
                continue
            behavioral.loc[behavioral["cluster"] == sc, "cluster"] = -1
            labels.pop(sc, None)

    behavioral["segment"] = behavioral["cluster"].map(labels).fillna("Outliers")

    result = check(behavioral, BEHAVIORAL_SEGMENTS)
    if return_metrics:
        return result, quality_metrics
    return result


_RADAR_FEATURES = (
    "days_active",
    "purchase_frequency",
    "total_revenue",
    "avg_order_value",
    "n_products",
    "n_categories",
    "avg_basket_size",
)


def compute_segment_radar(
    transactions_df: pd.DataFrame,
    n_clusters: int = 4,
    method: str = "kmeans",
) -> pd.DataFrame:
    """Radar-profile of behavioral segments on the key customer dimensions.

    Runs behavioral segmentation, then averages each feature per segment and
    min-max normalizes across segments so every feature spans [0, 1] over the
    chart. Contracts: one row per (segment, feature).

    Args:
        transactions_df: Transaction data.
        n_clusters: Number of behavioral segments.
        method: Clustering algorithm ('kmeans', 'gmm').

    Returns:
        DataFrame validated against SEGMENT_RADAR (empty when segmentation
        falls back to a single "Other" segment).
    """
    empty = pd.DataFrame(columns=list(SEGMENT_RADAR.columns))
    segmented = behavioral_segmentation(transactions_df, n_clusters=n_clusters, method=method)
    if segmented["segment"].nunique() < 2:
        return check(empty, SEGMENT_RADAR, allow_empty=True)

    features = _create_behavioral_features_pandas(transactions_df)
    merged = segmented[["customer_id", "segment"]].merge(features, on="customer_id", how="inner")

    present = [f for f in _RADAR_FEATURES if f in merged.columns]
    if not present:
        return check(empty, SEGMENT_RADAR, allow_empty=True)

    profile = merged.groupby("segment")[present].mean().reset_index()
    long = profile.melt(id_vars="segment", var_name="feature", value_name="mean_value")

    # Replace outliers with the cross-segment mean for a stable axis
    long = long.copy()
    long.loc[long["mean_value"] < 0, "mean_value"] = np.nan

    normalizer = long.groupby("feature")["mean_value"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.0
    )
    long["normalized_value"] = normalizer.clip(0.0, 1.0)
    long["mean_value"] = long["mean_value"].fillna(0.0)

    result = (
        long[list(SEGMENT_RADAR.columns)].sort_values(["segment", "feature"]).reset_index(drop=True)
    )
    return check(result, SEGMENT_RADAR)


def compute_segment_migration(
    transactions_df: pd.DataFrame,
    n_clusters: int = 4,
    method: str = "kmeans",
    min_customers: int = 5,
) -> pd.DataFrame:
    """Customer migration between behavioral segments across the first/second half.

    Splits transactions at the median date, runs behavioral segmentation on
    each half independently, then counts customers flowing from their
    first-half segment to their second-half segment. Retention rate per source
    segment = share of its customers that stayed put (diagonal).

    Args:
        transactions_df: Transaction data.
        n_clusters: Number of behavioral segments per half.
        method: Clustering algorithm ('kmeans', 'gmm').
        min_customers: Source segments with fewer customers are dropped noise.

    Returns:
        DataFrame validated against SEGMENT_MIGRATION (empty when either half
        falls back to the single "Other" segment).
    """
    empty = pd.DataFrame(columns=list(SEGMENT_MIGRATION.columns))
    if transactions_df.empty:
        return check(empty, SEGMENT_MIGRATION, allow_empty=True)

    dates = pd.to_datetime(transactions_df["date"])
    split = dates.median()
    first = transactions_df[dates <= split]
    second = transactions_df[dates > split]

    def _assign(half: pd.DataFrame) -> pd.DataFrame:
        if half.empty:
            return pd.DataFrame(columns=["customer_id", "segment"])
        seg = behavioral_segmentation(half, n_clusters=n_clusters, method=method)
        if isinstance(seg, tuple):
            seg = seg[0]
        if seg["segment"].nunique() < 2:
            return pd.DataFrame(columns=["customer_id", "segment"])
        return seg[["customer_id", "segment"]]

    a = _assign(first)
    b = _assign(second)
    if a.empty or b.empty:
        return check(empty, SEGMENT_MIGRATION, allow_empty=True)

    merged = a.rename(columns={"segment": "segment_from"}).merge(
        b.rename(columns={"segment": "segment_to"}),
        on="customer_id",
        how="inner",
    )
    counts = (
        merged.groupby(["segment_from", "segment_to"], as_index=False)
        .size()
        .rename(columns={"size": "customers"})
    )
    totals = counts.groupby("segment_from")["customers"].transform("sum")
    counts["retention_rate"] = np.where(
        counts["segment_from"] == counts["segment_to"],
        counts["customers"] / totals.where(totals > 0),
        0.0,
    )
    if counts.empty:
        return check(empty, SEGMENT_MIGRATION, allow_empty=True)

    # Drop sparse source segments (fewer than min_customers total flows)
    source_totals = counts.groupby("segment_from")["customers"].transform("sum")
    counts = counts[source_totals >= min_customers]
    if counts.empty:
        return check(empty, SEGMENT_MIGRATION, allow_empty=True)

    counts["period_from"] = "first_half"
    counts["period_to"] = "second_half"

    result = (
        counts[list(SEGMENT_MIGRATION.columns)]
        .sort_values(["segment_from", "segment_to"])
        .reset_index(drop=True)
    )
    return check(result, SEGMENT_MIGRATION)
