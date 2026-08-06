"""Core segmentation utilities: labeling, quality metrics, stability."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.utils import resample

from src.analytics.schemas import CLUSTER_QUALITY, CLUSTER_STABILITY, check

MIN_CLUSTER_SIZE = 5

_RFM_ARCHETYPES = [
    "Champions",
    "Loyal",
    "Big Spenders",
    "Frequent Buyers",
    "Promising",
    "Regular",
    "At Risk",
    "Dormant",
]


def _label_rfm_clusters(profiles: pd.DataFrame) -> dict:
    """Label RFM clusters by their relative rank across dimensions.

    Every cluster gets a meaningful name — no 'Cluster N' fallback.
    Uses the _RFM_ARCHETYPES pool, with descriptive fallbacks for >8 clusters.
    """
    n_clusters = len(profiles)
    ranked = pd.DataFrame(
        {
            "rec_rank": profiles["recency_days"].rank(),
            "freq_rank": profiles["frequency"].rank(ascending=False),
            "mon_rank": profiles["monetary"].rank(ascending=False),
        }
    )
    labels = {}
    for c in profiles.index:
        r = ranked.loc[c, "rec_rank"]
        mr = ranked.loc[c, "mon_rank"]
        fr = ranked.loc[c, "freq_rank"]

        if mr <= 2 and r <= 2:
            labels[c] = "Champions"
        elif mr <= 2:
            labels[c] = "Big Spenders"
        elif fr <= 2 and r <= 2:
            labels[c] = "Frequent Buyers"
        elif fr <= 2:
            labels[c] = "Loyal"
        elif r <= 2 and mr <= 3:
            labels[c] = "Promising"
        elif r >= n_clusters - 1:
            labels[c] = "Dormant"
        elif fr >= n_clusters - 1 and mr >= n_clusters - 1:
            labels[c] = "At Risk"
        else:
            p = profiles.loc[c]
            labels[c] = (
                f"Regular ({int(p['recency_days'])}d, {p['frequency']:.1f}x, ${p['monetary']:.0f})"
            )
    return labels


def _label_behavioral_clusters(profiles: pd.DataFrame) -> dict:
    """Label behavioral clusters by relative rank across key dimensions."""
    key_dims = ["total_revenue", "purchase_frequency", "n_products"]
    present = [d for d in key_dims if d in profiles.columns]
    if not present:
        return {c: f"Segment {c}" for c in profiles.index}

    n_clusters = len(profiles)
    ranked = pd.DataFrame({d: profiles[d].rank(ascending=False) for d in present})
    has_weekend = "weekend_ratio" in profiles.columns
    labels = {}

    for c in profiles.index:
        rev = ranked.loc[c, "total_revenue"]
        freq = ranked.loc[c, "purchase_frequency"]
        prod = ranked.loc[c, "n_products"]

        if rev <= 2 and freq <= 2:
            labels[c] = "High Value"
        elif has_weekend and profiles.loc[c, "weekend_ratio"] > 0.5:
            labels[c] = "Weekend Shoppers"
        elif freq <= 2 and prod <= 2:
            labels[c] = "Frequent Buyers"
        elif prod <= 2:
            labels[c] = "Variety Seekers"
        elif freq <= 2:
            labels[c] = "Regular Shoppers"
        elif rev <= 2:
            labels[c] = "Big Spenders"
        elif freq >= n_clusters - 1 and rev >= n_clusters - 1:
            labels[c] = "Light Buyers"
        else:
            p = profiles.loc[c]
            details = ", ".join(f"{d}={p[d]:.1f}" for d in present[:2])
            labels[c] = f"Mid-Tier ({details})"
    return labels


def compute_cluster_quality_metrics(
    features: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute silhouette score and Davies-Bouldin index for a clustering.

    Args:
        features: Scaled feature matrix (n_samples x n_features)
        labels: Cluster assignments (n_samples,)

    Returns:
        Dict with 'silhouette_score', 'davies_bouldin_score', 'n_clusters',
        and 'cluster_size_min' entries (empty dict if fewer than 2 clusters
        or evaluation fails).
    """
    unique = set(labels)
    n_clusters = len(unique - {-1}) if -1 in unique else len(unique)
    if n_clusters < 2:
        return {}

    mask = labels != -1
    valid_count = mask.sum()
    if valid_count < n_clusters or valid_count < MIN_CLUSTER_SIZE:
        return {}

    try:
        sample_size = min(5000, valid_count)
        sil = silhouette_score(
            features[mask], labels[mask], sample_size=sample_size, random_state=42
        )
        db = davies_bouldin_score(features[mask], labels[mask])
        sizes = pd.Series(labels[mask]).value_counts()
        return {
            "silhouette_score": round(sil, 4),
            "davies_bouldin_score": round(db, 4),
            "n_clusters": n_clusters,
            "cluster_size_min": int(sizes.min()),
            "cluster_size_max": int(sizes.max()),
            "cluster_size_mean": round(sizes.mean(), 1),
            "cluster_size_std": round(sizes.std(), 1),
        }
    except Exception as e:
        import warnings

        warnings.warn(f"Cluster stats computation failed: {e}", stacklevel=2)
        return {}


def compute_cluster_stability(
    transactions_df: pd.DataFrame | None = None,
    features_matrix: np.ndarray | None = None,
    n_clusters: int = 6,
    n_iterations: int = 10,
    method: str = "kmeans",
    sample_frac: float = 0.8,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate cluster stability across random seeds and subsamples.

    Runs the clustering multiple times and measures pairwise agreement
    using the adjusted Rand index against a reference clustering.

    Args:
        transactions_df: Raw transaction data (used if features_matrix not provided)
        features_matrix: Pre-computed scaled feature matrix (n_samples x n_features)
        n_clusters: Number of clusters
        n_iterations: Number of bootstrap iterations
        method: Clustering method ('kmeans')
        sample_frac: Fraction of data to sample each iteration
        seed: Random seed

    Returns:
        Dict with 'mean_ari', 'std_ari', 'min_ari', 'max_ari' (empty if
        fewer than 2 clusters).
    """
    if features_matrix is None:
        if transactions_df is None:
            raise ValueError("Must provide either transactions_df or features_matrix")

        df = transactions_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["revenue"] = df["price"] * df["quantity"]

        cat_col = "category" if "category" in df.columns else "stockcode"

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
                n_products=("stockcode", "nunique"),
                n_categories=(cat_col, "nunique"),
                avg_basket_size=("quantity", "mean"),
                avg_price=("price", "mean"),
                weekend_ratio=("date", lambda x: (x.dt.dayofweek >= 5).mean()),
            )
            .fillna(0)
            .reset_index()
        )

        feature_cols = [c for c in behavioral.columns if c != "customer_id"]
        if len(behavioral) < max(n_clusters * 2, 10):
            return {}

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(behavioral[feature_cols])
    else:
        X = features_matrix

    ref = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(X)

    scores_list: list[float] = []
    for i in range(n_iterations):
        rs = seed + i + 1
        subsample_idx = resample(
            range(len(X)), replace=False, n_samples=int(len(X) * sample_frac), random_state=rs
        )
        X_sub = X[subsample_idx]
        pred = KMeans(n_clusters=n_clusters, random_state=rs, n_init=10).fit_predict(X_sub)
        ref_sub = ref[subsample_idx]
        scores_list.append(adjusted_rand_score(ref_sub, pred))

    scores = np.array(scores_list, dtype=float)
    return {
        "mean_ari": round(scores.mean(), 4),
        "std_ari": round(scores.std(), 4),
        "min_ari": round(scores.min(), 4),
        "max_ari": round(scores.max(), 4),
    }


def format_quality_metrics(metrics: Dict[str, float]) -> pd.DataFrame:
    """Format quality metrics as a contract-compliant DataFrame."""
    if not metrics:
        return check(pd.DataFrame(columns=list(CLUSTER_QUALITY.columns)), CLUSTER_QUALITY, allow_empty=True)
    rows = [{"metric": k, "value": v} for k, v in metrics.items()]
    return check(pd.DataFrame(rows, columns=list(CLUSTER_QUALITY.columns)), CLUSTER_QUALITY)


def format_stability_metrics(metrics: Dict[str, float]) -> pd.DataFrame:
    """Format stability metrics as a contract-compliant DataFrame."""
    if not metrics:
        return check(pd.DataFrame(columns=list(CLUSTER_STABILITY.columns)), CLUSTER_STABILITY, allow_empty=True)
    rows = [{"metric": k, "value": v} for k, v in metrics.items()]
    return check(pd.DataFrame(rows, columns=list(CLUSTER_STABILITY.columns)), CLUSTER_STABILITY)