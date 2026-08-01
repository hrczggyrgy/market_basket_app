"""Customer Segmentation Analytics - RFM, Behavioral, Value-based."""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# Optional Polars for accelerated groupby
try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

# Optional lifetimes for BG/NBD CLV model (ALG-2)
try:
    from lifetimes import GammaGammaFitter

    LIFETIMES_AVAILABLE = True
except ImportError:
    LIFETIMES_AVAILABLE = False

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

_BEHAVIORAL_ARCHETYPES = [
    "High Value",
    "Frequent Buyers",
    "Regular Shoppers",
    "Variety Seekers",
    "Weekend Shoppers",
    "Big Spenders",
    "At Risk",
    "Light Buyers",
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
        # ALG-6: Use sampled silhouette for large datasets to avoid O(n²) memory
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
    transactions_df: pd.DataFrame = None,
    features_matrix: np.ndarray = None,
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
    from sklearn.metrics import adjusted_rand_score
    from sklearn.utils import resample

    # Build features from transactions if not provided
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

        scaler = StandardScaler()
        X = scaler.fit_transform(behavioral[feature_cols])
    else:
        X = features_matrix

    # Reference clustering
    ref = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(X)

    scores = []
    for i in range(n_iterations):
        rs = seed + i + 1
        subsample_idx = resample(
            range(len(X)), replace=False, n_samples=int(len(X) * sample_frac), random_state=rs
        )
        X_sub = X[subsample_idx]
        pred = KMeans(n_clusters=n_clusters, random_state=rs, n_init=10).fit_predict(X_sub)
        ref_sub = ref[subsample_idx]
        scores.append(adjusted_rand_score(ref_sub, pred))

    scores = np.array(scores)
    return {
        "mean_ari": round(scores.mean(), 4),
        "std_ari": round(scores.std(), 4),
        "min_ari": round(scores.min(), 4),
        "max_ari": round(scores.max(), 4),
    }


def compute_rfm_features(
    transactions_df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Compute comprehensive RFM features per customer."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    if snapshot_date is None:
        snapshot_date = df["date"].max() + pd.Timedelta(1, unit="D")

    cat_col = "category" if "category" in df.columns else "stockcode"

    rfm = (
        df.groupby("customer_id")
        .agg(
            recency_days=("date", lambda x: (snapshot_date - x.max()).days),
            frequency=("transaction_id", "nunique"),
            monetary=("revenue", "sum"),
            avg_order_value=("revenue", "mean"),
            max_order_value=("revenue", "max"),
            n_items=("quantity", "sum"),
            n_unique_products=("stockcode", "nunique"),
            n_unique_categories=(cat_col, "nunique"),
            first_purchase=("date", "min"),
            last_purchase=("date", "max"),
            avg_price_paid=("price", "mean"),
            std_order_value=("revenue", "std"),
        )
        .reset_index()
    )

    # Derived features
    rfm["customer_lifetime_days"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days
    rfm["purchase_interval"] = np.where(
        rfm["frequency"] > 1,
        rfm["customer_lifetime_days"] / (rfm["frequency"] - 1),
        rfm["customer_lifetime_days"],
    )
    rfm["items_per_order"] = rfm["n_items"] / rfm["frequency"]
    rfm["revenue_per_item"] = rfm["monetary"] / rfm["n_items"].replace(0, np.nan)
    rfm["order_value_cv"] = rfm["std_order_value"] / rfm["avg_order_value"].replace(0, np.nan)

    # Recency segments
    rfm["recency_segment"] = pd.qcut(
        rfm["recency_days"],
        q=4,
        labels=["Recent", "Active", "Lapsing", "Churned"],
        duplicates="drop",
    )
    rfm["frequency_segment"] = pd.qcut(
        rfm["frequency"].rank(method="first"),
        q=4,
        labels=["Low", "Medium", "High", "Very High"],
        duplicates="drop",
    )
    rfm["monetary_segment"] = pd.qcut(
        rfm["monetary"].rank(method="first"),
        q=4,
        labels=["Low", "Medium", "High", "Very High"],
        duplicates="drop",
    )

    return rfm


def compute_rfm_features_polars(
    transactions_df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Compute comprehensive RFM features per customer using Polars (5-20x faster).

    Falls back to pandas if Polars is not available.
    """
    if not POLARS_AVAILABLE:
        return compute_rfm_features(transactions_df, snapshot_date)

    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    if snapshot_date is None:
        snapshot_date = df["date"].max() + pd.Timedelta(1, unit="D")

    cat_col = "category" if "category" in df.columns else "stockcode"

    # Convert to Polars LazyFrame for parallel execution
    lf = pl.from_pandas(df).lazy()

    # Compute all aggregations in one pass
    rfm_pl = lf.group_by("customer_id").agg(
        [
            (pl.lit(snapshot_date) - pl.col("date").max()).dt.total_days().alias("recency_days"),
            pl.col("transaction_id").n_unique().alias("frequency"),
            pl.col("revenue").sum().alias("monetary"),
            pl.col("revenue").mean().alias("avg_order_value"),
            pl.col("revenue").max().alias("max_order_value"),
            pl.col("quantity").sum().alias("n_items"),
            pl.col("stockcode").n_unique().alias("n_unique_products"),
            pl.col(cat_col).n_unique().alias("n_unique_categories"),
            pl.col("date").min().alias("first_purchase"),
            pl.col("date").max().alias("last_purchase"),
            pl.col("price").mean().alias("avg_price_paid"),
            pl.col("revenue").std().alias("std_order_value"),
        ]
    )

    rfm = rfm_pl.collect().to_pandas()

    # Derived features (vectorized in pandas)
    rfm["customer_lifetime_days"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days
    rfm["purchase_interval"] = np.where(
        rfm["frequency"] > 1,
        rfm["customer_lifetime_days"] / (rfm["frequency"] - 1),
        rfm["customer_lifetime_days"],
    )
    rfm["items_per_order"] = rfm["n_items"] / rfm["frequency"]
    rfm["revenue_per_item"] = rfm["monetary"] / rfm["n_items"].replace(0, np.nan)
    rfm["order_value_cv"] = rfm["std_order_value"] / rfm["avg_order_value"].replace(0, np.nan)

    # Recency segments
    rfm["recency_segment"] = pd.qcut(
        rfm["recency_days"],
        q=4,
        labels=["Recent", "Active", "Lapsing", "Churned"],
        duplicates="drop",
    )
    rfm["frequency_segment"] = pd.qcut(
        rfm["frequency"].rank(method="first"),
        q=4,
        labels=["Low", "Medium", "High", "Very High"],
        duplicates="drop",
    )
    rfm["monetary_segment"] = pd.qcut(
        rfm["monetary"].rank(method="first"),
        q=4,
        labels=["Low", "Medium", "High", "Very High"],
        duplicates="drop",
    )

    return rfm


def behavioral_segmentation_polars(
    transactions_df: pd.DataFrame,
    n_clusters: int = 6,
    return_metrics: bool = False,
    method: str = "kmeans",
) -> pd.DataFrame:
    """Behavioral segmentation using Polars for feature computation (5-20x faster)."""
    if not POLARS_AVAILABLE:
        return behavioral_segmentation(transactions_df, n_clusters, return_metrics, method)

    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    cat_col_behav = "category" if "category" in df.columns else "stockcode"

    # Use Polars for the heavy groupby
    lf = pl.from_pandas(df).lazy()

    behavioral_pl = (
        lf.group_by("customer_id")
        .agg(
            [
                (pl.col("date").max() - pl.col("date").min()).dt.total_days().alias("days_active"),
                pl.col("transaction_id").n_unique().alias("purchase_frequency"),
                (
                    (pl.col("date").max() - pl.col("date").min()).dt.total_days()
                    / (pl.col("transaction_id").n_unique() - 1)
                ).alias("avg_days_between"),
                pl.col("revenue").sum().alias("total_revenue"),
                pl.col("revenue").mean().alias("avg_order_value"),
                pl.col("revenue").std().alias("revenue_std"),
                pl.col("stockcode").n_unique().alias("n_products"),
                pl.col(cat_col_behav).n_unique().alias("n_categories"),
                pl.col("quantity").mean().alias("avg_basket_size"),
                pl.col("quantity").max().alias("max_basket_size"),
                pl.col("price").mean().alias("avg_price"),
                (pl.col("price").std() / pl.col("price").mean()).alias("price_cv"),
                (pl.col("date").dt.weekday() >= 5).mean().alias("weekend_ratio"),
            ]
        )
        .fill_null(0)
    )

    behavioral = behavioral_pl.collect().to_pandas()

    feature_cols = [c for c in behavioral.columns if c != "customer_id"]
    n_samples = len(behavioral)

    # Minimum sample-size guard
    min_required = max(n_clusters * MIN_CLUSTER_SIZE, 10)
    if n_samples < min_required:
        behavioral["cluster"] = 0
        behavioral["segment"] = "Other"
        behavioral["cluster_distance"] = 0.0
        if return_metrics:
            return behavioral, {}
        return behavioral

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(behavioral[feature_cols])

    if method == "kmeans":
        # WF-5: Adaptive n_init for interactive vs final runs
        interactive = n_clusters <= 8  # heuristic: small k = interactive
        n_init = 3 if interactive else 10
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=n_init)
        behavioral["cluster"] = kmeans.fit_predict(X_scaled)

        # Distance-to-centroid confidence
        distances = kmeans.transform(X_scaled)
        behavioral["cluster_distance"] = distances.min(axis=1)
        max_dist = distances.max(axis=1)
        behavioral["cluster_confidence"] = np.where(
            max_dist > 0, 1 - behavioral["cluster_distance"] / max_dist, 1.0
        )
    else:
        raise ValueError(f"Polars accelerated method only supports 'kmeans', got: {method}")

    # Cluster quality metrics
    quality_metrics = compute_cluster_quality_metrics(X_scaled, behavioral["cluster"].values)

    # Label clusters
    cluster_profiles = behavioral.groupby("cluster")[feature_cols].mean()
    labels = _label_behavioral_clusters(cluster_profiles)

    # Drop clusters below MIN_CLUSTER_SIZE
    cluster_sizes = behavioral["cluster"].value_counts()
    small_clusters = cluster_sizes[cluster_sizes < MIN_CLUSTER_SIZE].index
    if not small_clusters.empty and len(small_clusters) < n_clusters:
        for sc in small_clusters:
            if sc == -1:
                continue
            behavioral.loc[behavioral["cluster"] == sc, "cluster"] = -1
            labels.pop(sc, None)

    behavioral["segment"] = behavioral["cluster"].map(labels).fillna("Outliers")

    if return_metrics:
        return behavioral, quality_metrics
    return behavioral


def rfm_segmentation(
    rfm_df: pd.DataFrame, method: str = "quantile", n_segments: int = 8
) -> pd.DataFrame:
    df = rfm_df.copy()

    if method == "quantile":
        # Classic RFM scoring (1-4 per dimension)
        for dim in ["recency_days", "frequency", "monetary"]:
            if dim == "recency_days":
                df[f"{dim}_score"] = pd.qcut(
                    df[dim].rank(method="first"),
                    q=4,
                    labels=[4, 3, 2, 1],
                    duplicates="drop",
                )
            else:
                df[f"{dim}_score"] = pd.qcut(
                    df[dim].rank(method="first"),
                    q=4,
                    labels=[1, 2, 3, 4],
                    duplicates="drop",
                )

        df["rfm_score"] = (
            df["recency_days_score"].astype(str)
            + df["frequency_score"].astype(str)
            + df["monetary_score"].astype(str)
        )

        # Segment mapping
        segment_map = {
            "444": "Champions",
            "443": "Champions",
            "434": "Champions",
            "344": "Champions",
            "442": "Loyal",
            "433": "Loyal",
            "432": "Loyal",
            "343": "Loyal",
            "334": "Loyal",
            "424": "Potential Loyalists",
            "423": "Potential Loyalists",
            "333": "Potential Loyalists",
            "324": "Potential Loyalists",
            "441": "New Customers",
            "431": "New Customers",
            "422": "New Customers",
            "342": "New Customers",
            "421": "Promising",
            "332": "Promising",
            "323": "Promising",
            "322": "Need Attention",
            "233": "Need Attention",
            "232": "Need Attention",
            "223": "About to Sleep",
            "222": "About to Sleep",
            "133": "About to Sleep",
            "221": "At Risk",
            "212": "At Risk",
            "123": "At Risk",
            "122": "At Risk",
            "211": "Cannot Lose Them",
            "113": "Cannot Lose Them",
            "112": "Cannot Lose Them",
            "111": "Lost",
        }

        df["segment"] = df["rfm_score"].map(segment_map).fillna("Other")

    elif method == "kmeans":
        # K-means clustering on RFM
        features = ["recency_days", "frequency", "monetary"]
        if len(df) < n_segments:
            df["cluster"] = 0
            df["segment"] = "Other"
            return df
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])

        kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(X_scaled)

        cluster_profiles = df.groupby("cluster")[["recency_days", "frequency", "monetary"]].mean()
        cluster_labels = _label_rfm_clusters(cluster_profiles)
        df["segment"] = df["cluster"].map(cluster_labels)

    return df


def behavioral_segmentation(
    transactions_df: pd.DataFrame,
    n_clusters: int = 6,
    return_metrics: bool = False,
    method: str = "kmeans",
    interactive: bool = True,  # WF-5: adaptive n_init
) -> pd.DataFrame:
    """Behavioral segmentation based on purchase patterns.

    Args:
        transactions_df: Transaction data
        n_clusters: Number of clusters
        return_metrics: Also return cluster quality metrics dict
        method: Clustering algorithm ('kmeans', 'gmm', 'hdbscan')
        interactive: If True, use n_init=3 for fast interactive runs; if False, use n_init=10 for quality

    Returns:
        DataFrame with cluster assignments; if return_metrics is True,
        returns (DataFrame, metrics_dict).
    """
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

    feature_cols = [c for c in behavioral.columns if c != "customer_id"]
    n_samples = len(behavioral)

    # Minimum sample-size guard
    min_required = max(n_clusters * MIN_CLUSTER_SIZE, 10)
    if n_samples < min_required:
        behavioral["cluster"] = 0
        behavioral["segment"] = "Other"
        behavioral["cluster_distance"] = 0.0
        if return_metrics:
            return behavioral, {}
        return behavioral

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
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=5)
        behavioral["cluster"] = gmm.fit_predict(X_scaled)
        behavioral["cluster_distance"] = 1.0 - gmm.predict_proba(X_scaled).max(axis=1)
    elif method == "hdbscan":
        try:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=3)
            behavioral["cluster"] = clusterer.fit_predict(X_scaled)
            behavioral["cluster_distance"] = np.where(behavioral["cluster"] == -1, 1.0, 0.0)
            n_clusters = len(set(behavioral["cluster"])) - (
                1 if -1 in behavioral["cluster"].values else 0
            )
        except ImportError:
            raise ImportError("hdbscan required: pip install hdbscan")
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

    if return_metrics:
        return behavioral, quality_metrics
    return behavioral


def value_based_segmentation(
    transactions_df: pd.DataFrame, prediction_horizon_days: int = 90
) -> pd.DataFrame:
    """Value-based segmentation with predicted CLV."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    snapshot_date = df["date"].max()
    cutoff_date = snapshot_date - pd.Timedelta(prediction_horizon_days, unit="D")

    # Historical (before cutoff) and future (after cutoff)
    hist = df[df["date"] < cutoff_date]
    future = df[df["date"] >= cutoff_date]

    if hist.empty:
        raise ValueError(
            f"No historical data before cutoff ({cutoff_date.date()}). "
            f"Reduce prediction_horizon_days (currently {prediction_horizon_days})."
        )

    # Historical features
    features = (
        hist.groupby("customer_id")
        .agg(
            recency=("date", lambda x: int((cutoff_date - x.max()).days)),
            frequency=("transaction_id", "nunique"),
            monetary=("revenue", "sum"),
            avg_order=("revenue", "mean"),
            n_products=("stockcode", "nunique"),
            lifetime_days=("date", lambda x: int((x.max() - x.min()).days + 1)),
        )
        .reset_index()
    )

    # Ensure lifetime_days is numeric
    features["lifetime_days"] = pd.to_numeric(features["lifetime_days"], errors="coerce").fillna(1)

    # Ensure recency is numeric (days as int)
    features["recency"] = pd.to_numeric(features["recency"], errors="coerce").fillna(0).astype(int)

    # Future actuals (for validation)
    future_rev = future.groupby("customer_id")["revenue"].sum().reset_index()
    future_rev.columns = ["customer_id", "future_revenue"]
    features = features.merge(future_rev, on="customer_id", how="left").fillna(
        {"future_revenue": 0}
    )

    # CLV prediction with churn adjustment
    # Annualized historical spend per customer
    annual_value = features["monetary"] / (features["lifetime_days"].clip(lower=1) / 365)
    # Survival probability: customers with long recency relative to lifetime likely churned
    survival_prob = np.clip(
        1 - features["recency"] / (features["lifetime_days"].clip(lower=1) + features["recency"]),
        0,
        1,
    )
    features["predicted_clv"] = annual_value * survival_prob * 2  # 2-year horizon

    # Segments — first matching condition wins (priority: high CLV + recent > loyal > new > churned)
    conditions = [
        (features["predicted_clv"] > features["predicted_clv"].quantile(0.8))
        & (features["recency"] < 30)
        & (features["frequency"] > 1),
        (features["predicted_clv"] > features["predicted_clv"].quantile(0.6))
        & (features["recency"] < 60)
        & (features["frequency"] > 1),
        (features["frequency"] > 5) & (features["recency"] < 90),
        (features["frequency"] == 1) & (features["recency"] < 30),
        (features["recency"] > 180),
    ]
    choices = ["VIP", "High Potential", "Loyal", "New", "Churned"]
    features["value_segment"] = np.select(conditions, choices, default="Regular")

    return features


def predict_clv_bg_nbd(
    transactions_df: pd.DataFrame,
    prediction_horizon_days: int = 90,
    freq: str = "D",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Predict Customer Lifetime Value using BG/NBD + Gamma-Gamma model (ALG-2).

    This is the industry-standard probabilistic CLV model (Fader, Hardie, Lee 2005).
    BG/NBD models purchase frequency/recency, Gamma-Gamma models monetary value.

    Args:
        transactions_df: Transaction data with customer_id, date, revenue columns
        prediction_horizon_days: Days into future to predict
        freq: 'D' for daily, 'W' for weekly

    Returns:
        (DataFrame with customer_id, predicted_clv, predicted_purchases,
         expected_avg_value, p_alive, clv_segment, model_diagnostics)
    """
    if not LIFETIMES_AVAILABLE:
        raise ImportError("lifetimes required: pip install lifetimes")

    from lifetimes import BetaGeoFitter, GammaGammaFitter
    from lifetimes.utils import summary_data_from_transaction_data

    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # Filter out customers with zero/negative revenue
    customer_revenue = df.groupby("customer_id")["revenue"].sum()
    valid_customers = customer_revenue[customer_revenue > 0].index
    df = df[df["customer_id"].isin(valid_customers)].copy()

    # Prepare summary data for BG/NBD
    observation_period_end = df["date"].max() - pd.Timedelta(prediction_horizon_days, unit="D")

    summary = summary_data_from_transaction_data(
        df,
        customer_id_col="customer_id",
        datetime_col="date",
        monetary_value_col="revenue",
        observation_period_end=observation_period_end,
        freq=freq,
    )

    # Filter customers with at least 1 repeat purchase for Gamma-Gamma
    summary_cal = summary[summary["frequency"] > 0].copy()

    # Filter out non-positive monetary values for Gamma-Gamma
    summary_cal = summary_cal[summary_cal["monetary_value"] > 0].copy()

    if len(summary_cal) < 10:
        raise ValueError("Insufficient repeat customers for BG/NBD (need >= 10)")

    # Fit BG/NBD for purchase prediction
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(summary_cal["frequency"], summary_cal["recency"], summary_cal["T"])

    # Fit Gamma-Gamma for monetary value prediction
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(summary_cal["frequency"], summary_cal["monetary_value"])

    # Predict expected purchases in horizon
    t = prediction_horizon_days if freq == "D" else prediction_horizon_days / 7
    summary_cal["predicted_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        t, summary_cal["frequency"], summary_cal["recency"], summary_cal["T"]
    )

    # Predict expected average order value
    summary_cal["expected_avg_value"] = ggf.conditional_expected_average_profit(
        summary_cal["frequency"], summary_cal["monetary_value"]
    )

    # CLV = predicted_purchases * expected_avg_value
    summary_cal["predicted_clv"] = (
        summary_cal["predicted_purchases"] * summary_cal["expected_avg_value"]
    )

    # Probability alive
    summary_cal["p_alive"] = bgf.conditional_probability_alive(
        summary_cal["frequency"], summary_cal["recency"], summary_cal["T"]
    )

    # Prepare output
    result = summary_cal[
        ["predicted_purchases", "expected_avg_value", "predicted_clv", "p_alive"]
    ].copy()
    result = result.reset_index()
    result.columns = [
        "customer_id",
        "predicted_purchases",
        "expected_avg_value",
        "predicted_clv",
        "p_alive",
    ]

    # Add segment labels
    result["clv_segment"] = pd.qcut(
        result["predicted_clv"],
        q=4,
        labels=["Bronze", "Silver", "Gold", "Platinum"],
        duplicates="drop",
    )

    # Model diagnostics
    diagnostics = {
        "bgf_params": {
            "r": float(bgf.params_["r"]),
            "alpha": float(bgf.params_["alpha"]),
            "a": float(bgf.params_["a"]),
            "b": float(bgf.params_["b"]),
            "log_likelihood": float(-bgf._negative_log_likelihood_),
        },
        "ggf_params": {
            "p": float(ggf.params_["p"]),
            "q": float(ggf.params_["q"]),
            "v": float(ggf.params_["v"]),
            "log_likelihood": float(-ggf._negative_log_likelihood_),
        },
        "n_customers_fit": len(summary_cal),
        "n_customers_total": len(summary),
    }

    return result, diagnostics


def get_segment_profiles(
    transactions_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    segment_col: str = "segment",
) -> pd.DataFrame:
    """Get detailed profiles for each segment."""
    df = transactions_df.merge(
        segments_df[["customer_id", segment_col]], on="customer_id", how="left"
    )
    df["revenue"] = df["price"] * df["quantity"]

    profiles = (
        df.groupby(segment_col)
        .agg(
            n_customers=("customer_id", "nunique"),
            n_transactions=("transaction_id", "nunique"),
            total_revenue=("revenue", "sum"),
            avg_order_value=("revenue", "mean"),
            avg_items_per_order=("quantity", "mean"),
            n_products=("stockcode", "nunique"),
            top_category=(
                "category",
                lambda x: (
                    x.mode().iloc[0] if "category" in df.columns and not x.mode().empty else "N/A"
                ),
            ),
            top_brand=(
                "brand",
                lambda x: (
                    x.mode().iloc[0] if "brand" in df.columns and not x.mode().empty else "N/A"
                ),
            ),
        )
        .reset_index()
    )

    profiles["avg_products_per_customer"] = profiles["n_products"] / profiles["n_customers"]
    profiles["repeat_rate"] = profiles["n_transactions"] / profiles["n_customers"]

    profiles["revenue_per_customer"] = profiles["total_revenue"] / profiles["n_customers"]
    profiles["revenue_share"] = profiles["total_revenue"] / profiles["total_revenue"].sum()
    profiles["customer_share"] = profiles["n_customers"] / profiles["n_customers"].sum()

    return profiles


def compute_customer_entropy(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    customer_col: str = "customer_id",
) -> pd.DataFrame:
    """
    Compute Customer Entropy Score: H_c = -Σ s_p log s_p
    where s_p is the share of customer c's spend on product p.

    Low entropy = highly concentrated buyer (loyalty risk)
    High entropy = variety seeker
    """
    df = transactions_df.copy()
    df["revenue"] = df["price"] * df["quantity"]

    # Spend per customer per product
    spend = df.groupby([customer_col, product_col])["revenue"].sum().reset_index()

    # Total spend per customer
    total_spend = spend.groupby(customer_col)["revenue"].sum().rename("total_spend")
    spend = spend.merge(total_spend, on=customer_col)

    # Share per product per customer
    spend["share"] = spend["revenue"] / spend["total_spend"]

    # Entropy per customer
    def entropy(group):
        shares = group["share"].values
        shares = shares[shares > 0]
        if len(shares) == 0:
            return 0
        return -np.sum(shares * np.log(shares))

    entropy_df = spend.groupby(customer_col).apply(entropy).reset_index(name="entropy")

    # Normalize entropy (max entropy = log(n_products))
    max_entropy = np.log(transactions_df[product_col].nunique())
    entropy_df["normalized_entropy"] = entropy_df["entropy"] / max_entropy if max_entropy > 0 else 0

    return entropy_df


def compute_ipt_cv(
    transactions_df: pd.DataFrame,
    customer_col: str = "customer_id",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Compute Inter-Purchase Time Coefficient of Variation (IPT-CV) per customer.

    IPT-CV = std(inter_purchase_times) / mean(inter_purchase_times)
    Low CV = regular buyer, High CV = irregular buyer
    """
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Sort by customer and date
    df = df.sort_values([customer_col, date_col])

    # Compute inter-purchase times per customer
    df["prev_date"] = df.groupby(customer_col)[date_col].shift(1)
    df["inter_purchase_days"] = (df[date_col] - df["prev_date"]).dt.days

    # Drop first purchase per customer (no previous purchase)
    repeat_purchases = df.dropna(subset=["inter_purchase_days"])

    if repeat_purchases.empty:
        return pd.DataFrame(columns=[customer_col, "ipt_mean", "ipt_std", "ipt_cv"])

    # Compute CV per customer
    ipt_stats = (
        repeat_purchases.groupby(customer_col)["inter_purchase_days"]
        .agg(
            ipt_mean="mean",
            ipt_std="std",
        )
        .reset_index()
    )

    # Avoid division by zero
    ipt_stats["ipt_cv"] = np.where(
        ipt_stats["ipt_mean"] > 0,
        ipt_stats["ipt_std"] / ipt_stats["ipt_mean"],
        np.nan,
    )

    return ipt_stats[["customer_id", "ipt_mean", "ipt_std", "ipt_cv"]]


def compute_bg_nbd_p_alive(
    transactions_df: pd.DataFrame,
    prediction_horizon_days: int = 90,
    freq: str = "D",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute P(alive) using BG/NBD model from lifetimes library.

    Returns customer-level P(alive) with diagnostics.
    """
    if not LIFETIMES_AVAILABLE:
        raise ImportError("lifetimes library required: pip install lifetimes")

    from lifetimes import BetaGeoFitter
    from lifetimes.utils import summary_data_from_transaction_data

    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # Prepare summary data for BG/NBD
    observation_period_end = df["date"].max() - pd.Timedelta(prediction_horizon_days, unit="D")

    summary = summary_data_from_transaction_data(
        df,
        customer_id_col="customer_id",
        datetime_col="date",
        monetary_value_col="revenue",
        observation_period_end=observation_period_end,
        freq=freq,
    )

    # Filter customers with at least 1 repeat purchase for Gamma-Gamma
    summary_cal = summary[summary["frequency"] > 0].copy()

    if len(summary_cal) < 10:
        raise ValueError("Insufficient repeat customers for BG/NBD (need >= 10)")

    # Fit BG/NBD for purchase prediction
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(summary_cal["frequency"], summary_cal["recency"], summary_cal["T"])

    # Predict probability alive
    summary_cal["p_alive"] = bgf.conditional_probability_alive(
        summary_cal["frequency"], summary_cal["recency"], summary_cal["T"]
    )

    # Predict expected purchases in horizon
    t = prediction_horizon_days if freq == "D" else prediction_horizon_days / 7
    summary_cal["predicted_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        t, summary_cal["frequency"], summary_cal["recency"], summary_cal["T"]
    )

    # Fit Gamma-Gamma for monetary value
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(summary_cal["frequency"], summary_cal["monetary_value"])
    summary_cal["expected_avg_value"] = ggf.conditional_expected_average_profit(
        summary_cal["frequency"], summary_cal["monetary_value"]
    )

    # CLV = predicted_purchases * expected_avg_value
    summary_cal["predicted_clv"] = (
        summary_cal["predicted_purchases"] * summary_cal["expected_avg_value"]
    )

    # Prepare output
    result = summary_cal[
        ["predicted_purchases", "expected_avg_value", "predicted_clv", "p_alive"]
    ].copy()
    result = result.reset_index()
    result.columns = [
        "customer_id",
        "predicted_purchases",
        "expected_avg_value",
        "predicted_clv",
        "p_alive",
    ]

    # Add segment labels
    result["clv_segment"] = pd.qcut(
        result["predicted_clv"],
        q=4,
        labels=["Bronze", "Silver", "Gold", "Platinum"],
        duplicates="drop",
    )

    # Model diagnostics
    diagnostics = {
        "bgf_params": {
            "r": float(bgf.params_["r"]),
            "alpha": float(bgf.params_["alpha"]),
            "a": float(bgf.params_["a"]),
            "b": float(bgf.params_["b"]),
            "log_likelihood": float(-bgf._negative_log_likelihood_),
        },
        "ggf_params": {
            "p": float(ggf.params_["p"]),
            "q": float(ggf.params_["q"]),
            "v": float(ggf.params_["v"]),
            "log_likelihood": float(-ggf._negative_log_likelihood_),
        },
        "n_customers_fit": len(summary_cal),
        "n_customers_total": len(summary),
    }

    return result, diagnostics
