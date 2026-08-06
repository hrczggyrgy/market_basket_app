"""Behavioral Segmentation based on purchase patterns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.analytics.segmentation.core import (
    MIN_CLUSTER_SIZE,
    _label_behavioral_clusters,
    compute_cluster_quality_metrics,
)
from src.analytics.schemas import BEHAVIORAL_FEATURES, BEHAVIORAL_SEGMENTS, check


def _create_behavioral_features_pandas(
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create behavioral features for each customer using pandas."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

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
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Behavioral segmentation based on purchase patterns.

    Args:
        transactions_df: Transaction data
        n_clusters: Number of clusters
        return_metrics: Also return cluster quality metrics dict
        method: Clustering algorithm ('kmeans', 'gmm')
        interactive: If True, use n_init=3 for fast interactive runs; if False, use n_init=10 for quality

    Returns:
        DataFrame with cluster assignments; if return_metrics is True,
        returns (DataFrame, metrics_dict).
    """
    behavioral = _create_behavioral_features_pandas(transactions_df)

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