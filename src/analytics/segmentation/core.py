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
        return check(
            pd.DataFrame(columns=list(CLUSTER_QUALITY.columns)), CLUSTER_QUALITY, allow_empty=True
        )
    rows = [{"metric": k, "value": v} for k, v in metrics.items()]
    return check(pd.DataFrame(rows, columns=list(CLUSTER_QUALITY.columns)), CLUSTER_QUALITY)


def format_stability_metrics(metrics: Dict[str, float]) -> pd.DataFrame:
    """Format stability metrics as a contract-compliant DataFrame."""
    if not metrics:
        return check(
            pd.DataFrame(columns=list(CLUSTER_STABILITY.columns)),
            CLUSTER_STABILITY,
            allow_empty=True,
        )
    rows = [{"metric": k, "value": v} for k, v in metrics.items()]
    return check(pd.DataFrame(rows, columns=list(CLUSTER_STABILITY.columns)), CLUSTER_STABILITY)


def calculate_segment_value_metrics(
    segments_df: pd.DataFrame, transactions_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate segment size and economic importance metrics.

    Args:
        segments_df: DataFrame with customer_id and segment columns
        transactions_df: Transaction data with customer_id, price, quantity, etc.

    Returns:
        DataFrame with segment-level economic metrics
    """
    # Calculate revenue if not present
    df = transactions_df.copy()
    if "revenue" not in df.columns:
        df["revenue"] = df["price"] * df["quantity"]

    # Merge segment info with transaction data
    df = df.merge(segments_df[["customer_id", "segment"]], on="customer_id", how="left")

    # Calculate segment-level metrics
    segment_metrics = (
        df.groupby("segment")
        .agg(
            customers=("customer_id", "nunique"),
            revenue=("revenue", "sum"),
            transactions=("transaction_id", "nunique"),
            units=("quantity", "sum"),
        )
        .reset_index()
    )

    # Calculate totals for share calculations
    total_customers = segment_metrics["customers"].sum()
    total_revenue = segment_metrics["revenue"].sum()
    total_transactions = segment_metrics["transactions"].sum()
    total_units = segment_metrics["units"].sum()

    # Calculate share metrics
    segment_metrics["customer_share_pct"] = (
        (segment_metrics["customers"] / total_customers * 100) if total_customers > 0 else 0
    )
    segment_metrics["revenue_share_pct"] = (
        (segment_metrics["revenue"] / total_revenue * 100) if total_revenue > 0 else 0
    )
    segment_metrics["transaction_share_pct"] = (
        (segment_metrics["transactions"] / total_transactions * 100)
        if total_transactions > 0
        else 0
    )
    segment_metrics["unit_share_pct"] = (
        (segment_metrics["units"] / total_units * 100) if total_units > 0 else 0
    )

    # Calculate per-customer metrics
    segment_metrics["revenue_per_customer"] = (
        segment_metrics["revenue"] / segment_metrics["customers"]
    )
    segment_metrics["transactions_per_customer"] = (
        segment_metrics["transactions"] / segment_metrics["customers"]
    )
    segment_metrics["units_per_customer"] = segment_metrics["units"] / segment_metrics["customers"]
    segment_metrics["revenue_per_transaction"] = (
        segment_metrics["revenue"] / segment_metrics["transactions"]
    )

    # Value Concentration Index = Revenue Share / Customer Share
    segment_metrics["value_concentration_index"] = (
        (segment_metrics["revenue_share_pct"] / segment_metrics["customer_share_pct"])
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    return segment_metrics


def calculate_segment_engagement_metrics(
    segments_df: pd.DataFrame, transactions_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate customer engagement metrics for segments.

    Args:
        segments_df: DataFrame with customer_id and segment columns
        transactions_df: Transaction data with customer_id, date, etc.

    Returns:
        DataFrame with segment-level engagement metrics
    """
    # Merge segment info with transaction data
    df = transactions_df.merge(
        segments_df[["customer_id", "segment"]], on="customer_id", how="left"
    )
    df["date"] = pd.to_datetime(df["date"])

    # Calculate engagement metrics per customer first
    customer_engagement = (
        df.groupby(["customer_id", "segment"])
        .agg(
            purchase_frequency=("transaction_id", "nunique"),
            recency_days=("date", lambda x: (df["date"].max() - x.max()).days),
            avg_days_between=(
                "date",
                lambda x: (
                    (x.max() - x.min()).days / max(x.nunique() - 1, 1) if x.nunique() > 1 else 0
                ),
            ),
            active_weeks=("date", lambda x: x.dt.isocalendar().week.nunique()),
            active_months=("date", lambda x: (x.dt.year * 12 + x.dt.month).nunique()),
            purchase_streak=("date", lambda x: _calculate_purchase_streak(x.sort_values())),
        )
        .reset_index()
    )

    # Aggregate to segment level
    segment_metrics = (
        customer_engagement.groupby("segment")
        .agg(
            purchase_frequency=("purchase_frequency", "mean"),
            recency_days=("recency_days", "mean"),
            avg_days_between=("avg_days_between", "mean"),
            active_weeks=("active_weeks", "mean"),
            active_months=("active_months", "mean"),
            purchase_streak=("purchase_streak", "mean"),
        )
        .reset_index()
    )

    # Calculate Active Customer Rate (purchased in last 30 days)
    thirty_days_ago = df["date"].max() - pd.Timedelta(days=30)
    active_customers = (
        df[df["date"] >= thirty_days_ago].groupby("segment")["customer_id"].nunique().reset_index()
    )
    active_customers.columns = ["segment", "active_customers"]

    segment_totals = segments_df.groupby("segment")["customer_id"].nunique().reset_index()
    segment_totals.columns = ["segment", "total_customers"]

    engagement_metrics = segment_metrics.merge(active_customers, on="segment", how="left")
    engagement_metrics = engagement_metrics.merge(segment_totals, on="segment", how="left")
    engagement_metrics["active_customer_rate_pct"] = (
        engagement_metrics["active_customers"] / engagement_metrics["total_customers"] * 100
    ).fillna(0)

    # Dormancy rate (no purchase in last 90 days)
    ninety_days_ago = df["date"].max() - pd.Timedelta(days=90)
    dormant_customers = (
        df[df["date"] < ninety_days_ago].groupby("segment")["customer_id"].nunique().reset_index()
    )
    dormant_customers.columns = ["segment", "dormant_customers"]

    engagement_metrics = engagement_metrics.merge(dormant_customers, on="segment", how="left")
    engagement_metrics["dormant_customers"] = engagement_metrics["dormant_customers"].fillna(0)
    engagement_metrics["dormancy_rate_pct"] = (
        engagement_metrics["dormant_customers"] / engagement_metrics["total_customers"] * 100
    ).fillna(0)

    return engagement_metrics[
        [
            "segment",
            "purchase_frequency",
            "recency_days",
            "avg_days_between",
            "active_weeks",
            "active_months",
            "purchase_streak",
            "active_customer_rate_pct",
            "dormancy_rate_pct",
        ]
    ]


def _calculate_purchase_streak(dates: pd.Series) -> int:
    """Calculate the longest streak of consecutive days with purchases."""
    if len(dates) < 2:
        return len(dates)

    dates_sorted = dates.sort_values()
    date_gaps = (dates_sorted.diff().dt.days > 1).cumsum()
    streak_lengths = date_gaps.value_counts()
    return streak_lengths.max() if len(streak_lengths) > 0 else 0


def calculate_segment_retention_metrics(
    transactions_df: pd.DataFrame, segments_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate retention and lifecycle metrics for segments.

    Args:
        transactions_df: Transaction data
        segments_df: DataFrame with customer_id and segment columns

    Returns:
        DataFrame with segment-level retention metrics
    """
    # Calculate revenue if not present
    df = transactions_df.copy()
    if "revenue" not in df.columns:
        df["revenue"] = df["price"] * df["quantity"]

    # Merge segment info with transaction data
    df = df.merge(segments_df[["customer_id", "segment"]], on="customer_id", how="left")
    df["date"] = pd.to_datetime(df["date"])

    # Calculate customer lifecycle metrics
    customer_lifecycle = (
        df.groupby(["customer_id", "segment"])
        .agg(
            first_purchase=("date", "min"),
            last_purchase=("date", "max"),
            customer_lifetime_days=("date", lambda x: (x.max() - x.min()).days + 1),
            total_revenue=("revenue", "sum"),
            purchase_frequency=("transaction_id", "nunique"),
        )
        .reset_index()
    )

    # Calculate repeat purchase rate
    customer_lifecycle["is_repeat_customer"] = customer_lifecycle["purchase_frequency"] > 1
    repeat_customers = (
        customer_lifecycle.groupby("segment")["is_repeat_customer"].mean().reset_index()
    )
    repeat_customers.columns = ["segment", "repeat_purchase_rate_pct"]
    repeat_customers["repeat_purchase_rate_pct"] *= 100

    # Aggregate to segment level
    segment_metrics = (
        customer_lifecycle.groupby("segment")
        .agg(
            customer_lifetime_days=("customer_lifetime_days", "mean"),
            tenure=("customer_lifetime_days", "mean"),  # Same as lifetime for now
            avg_revenue_per_customer=("total_revenue", "mean"),
        )
        .reset_index()
    )

    # Merge with repeat purchase rate
    segment_metrics = segment_metrics.merge(repeat_customers, on="segment", how="left")

    # Calculate churn/lapse rate (no purchase in last 60 days)
    sixty_days_ago = df["date"].max() - pd.Timedelta(days=60)
    active_customers = (
        df[df["date"] >= sixty_days_ago].groupby("segment")["customer_id"].nunique().reset_index()
    )
    active_customers.columns = ["segment", "active_customers"]

    segment_totals = segments_df.groupby("segment")["customer_id"].nunique().reset_index()
    segment_totals.columns = ["segment", "total_customers"]

    lifecycle_metrics = segment_metrics.merge(active_customers, on="segment", how="left")
    lifecycle_metrics = lifecycle_metrics.merge(segment_totals, on="segment", how="left")
    lifecycle_metrics["lapse_rate_pct"] = (
        (lifecycle_metrics["total_customers"] - lifecycle_metrics["active_customers"])
        / lifecycle_metrics["total_customers"]
        * 100
    ).fillna(0)

    # Reactivation rate (customers who purchased after 60+ days of inactivity)
    # This requires more complex logic - for now we'll calculate a simplified version
    lifecycle_metrics["reactivation_rate_pct"] = 0.0  # Placeholder

    return lifecycle_metrics[
        [
            "segment",
            "customer_lifetime_days",
            "tenure",
            "repeat_purchase_rate_pct",
            "lapse_rate_pct",
            "reactivation_rate_pct",
        ]
    ]


def calculate_segment_basket_metrics(
    segments_df: pd.DataFrame, transactions_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate basket economics metrics for segments.

    Args:
        segments_df: DataFrame with customer_id and segment columns
        transactions_df: Transaction data with customer_id, price, quantity, etc.

    Returns:
        DataFrame with segment-level basket metrics
    """
    # Calculate revenue if not present
    df = transactions_df.copy()
    if "revenue" not in df.columns:
        df["revenue"] = df["price"] * df["quantity"]

    # Merge segment info with transaction data
    df = df.merge(segments_df[["customer_id", "segment"]], on="customer_id", how="left")

    # Calculate basket metrics per transaction
    basket_metrics = (
        df.groupby(["customer_id", "segment", "transaction_id"])
        .agg(
            basket_value=("revenue", "sum"),
            basket_units=("quantity", "sum"),
            basket_skus=("stockcode", "nunique"),
            basket_categories=("category", "nunique")
            if "category" in df.columns
            else ("stockcode", lambda x: 1),  # Fallback if no category
        )
        .reset_index()
    )

    # Handle case where category column doesn't exist
    if "category" not in df.columns:
        basket_metrics["basket_categories"] = 1
    else:
        basket_metrics["basket_categories"] = basket_metrics["basket_categories"]

    # Aggregate to segment level
    segment_metrics = (
        basket_metrics.groupby("segment")
        .agg(
            avg_basket_value=("basket_value", "mean"),
            avg_units_per_basket=("basket_units", "mean"),
            avg_skus_per_basket=("basket_skus", "mean"),
            avg_categories_per_basket=("basket_categories", "mean"),
        )
        .reset_index()
    )

    # Calculate basket concentration (Gini coefficient of basket values)
    def _gini(array: np.ndarray) -> float:
        """Calculate Gini coefficient of array of values."""
        if len(array) == 0:
            return 0
        sorted_array = np.sort(array)
        n = len(sorted_array)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_array) - (n + 1) * np.sum(sorted_array)) / (
            n * np.sum(sorted_array)
        )

    basket_gini = basket_metrics.groupby("segment")["basket_value"].apply(_gini).reset_index()
    basket_gini.columns = ["segment", "basket_concentration_gini"]

    # Calculate basket diversity (entropy of SKU/category distribution)
    def _entropy(array: np.ndarray) -> float:
        """Calculate Shannon entropy of array."""
        if len(array) == 0:
            return 0
        _, counts = np.unique(array, return_counts=True)
        probs = counts / len(array)
        return -np.sum(probs * np.log(probs + 1e-10))

    basket_entropy = basket_metrics.groupby("segment")["basket_skus"].apply(_entropy).reset_index()
    basket_entropy.columns = ["segment", "basket_diversity_entropy"]

    # Calculate large/small basket rates
    basket_median = basket_metrics.groupby("segment")["basket_value"].transform("median")
    basket_metrics["is_large_basket"] = basket_metrics["basket_value"] > basket_median
    basket_metrics["is_small_basket"] = basket_metrics["basket_value"] < (basket_median * 0.5)

    basket_rates = (
        basket_metrics.groupby("segment")
        .agg(
            large_basket_rate_pct=("is_large_basket", "mean"),
            small_basket_rate_pct=("is_small_basket", "mean"),
        )
        .reset_index()
    )
    basket_rates["large_basket_rate_pct"] *= 100
    basket_rates["small_basket_rate_pct"] *= 100

    # Calculate basket penetration (percentage of baskets containing specific categories)
    # For now, we'll calculate overall basket diversity as a proxy
    # In a full implementation, this would be calculated per category
    segment_metrics = segment_metrics.merge(basket_gini, on="segment", how="left")
    segment_metrics = segment_metrics.merge(basket_entropy, on="segment", how="left")
    segment_metrics = segment_metrics.merge(basket_rates, on="segment", how="left")

    return segment_metrics[
        [
            "segment",
            "avg_basket_value",
            "avg_units_per_basket",
            "avg_skus_per_basket",
            "avg_categories_per_basket",
            "basket_concentration_gini",
            "basket_diversity_entropy",
            "large_basket_rate_pct",
            "small_basket_rate_pct",
        ]
    ]


def calculate_segment_price_behavior_metrics(
    segments_df: pd.DataFrame, transactions_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate price behavior metrics for segments.

    Args:
        segments_df: DataFrame with customer_id and segment columns
        transactions_df: Transaction data with customer_id, price, etc.

    Returns:
        DataFrame with segment-level price behavior metrics
    """
    # Calculate revenue if not present
    df = transactions_df.copy()
    if "revenue" not in df.columns:
        df["revenue"] = df["price"] * df["quantity"]

    # Merge segment info with transaction data
    df = df.merge(segments_df[["customer_id", "segment"]], on="customer_id", how="left")

    # Calculate overall price metrics for comparison
    overall_avg_price = df["price"].mean()

    # Calculate price metrics per customer
    customer_price_metrics = (
        df.groupby(["customer_id", "segment"])
        .agg(
            avg_price_paid=("price", "mean"),
            price_std=("price", "std"),
            min_price=("price", "min"),
            max_price=("price", "max"),
        )
        .reset_index()
    )

    # Calculate price dispersion metrics
    customer_price_metrics["price_cv"] = (
        customer_price_metrics["price_std"] / customer_price_metrics["avg_price_paid"]
    )
    customer_price_metrics["price_cv"] = customer_price_metrics["price_cv"].fillna(0)
    customer_price_metrics["price_range"] = (
        customer_price_metrics["max_price"] - customer_price_metrics["min_price"]
    )

    # Aggregate to segment level
    segment_metrics = (
        customer_price_metrics.groupby("segment")
        .agg(
            avg_price_paid=("avg_price_paid", "mean"),
            price_cv=("price_cv", "mean"),
            price_range=("price_range", "mean"),
        )
        .reset_index()
    )

    # Calculate Price Index vs overall customer base
    segment_metrics["price_index_vs_overall"] = (
        segment_metrics["avg_price_paid"] / overall_avg_price * 100
    ).fillna(100)

    # Calculate Price Orientation Index (deviation from overall average price)
    segment_metrics["price_orientation_index"] = segment_metrics["price_index_vs_overall"] - 100

    # Premium/value product share (simplified - based on price tiers)
    price_75th = df["price"].quantile(0.75)
    price_25th = df["price"].quantile(0.25)

    def _calculate_premium_share(group: pd.Series) -> float:
        if len(group) == 0:
            return 0
        premium_count = (group >= price_75th).sum()
        return (premium_count / len(group)) * 100

    def _calculate_value_share(group: pd.Series) -> float:
        if len(group) == 0:
            return 0
        value_count = (group <= price_25th).sum()
        return (value_count / len(group)) * 100

    premium_share = (
        df.groupby(["customer_id", "segment"])["price"]
        .apply(_calculate_premium_share)
        .reset_index()
    )
    premium_share.columns = ["customer_id", "segment", "premium_product_share_pct"]

    value_share = (
        df.groupby(["customer_id", "segment"])["price"].apply(_calculate_value_share).reset_index()
    )
    value_share.columns = ["customer_id", "segment", "value_product_share_pct"]

    segment_premium = (
        premium_share.groupby("segment")["premium_product_share_pct"].mean().reset_index()
    )
    segment_value = value_share.groupby("segment")["value_product_share_pct"].mean().reset_index()

    segment_metrics = segment_metrics.merge(segment_premium, on="segment", how="left")
    segment_metrics = segment_metrics.merge(segment_value, on="segment", how="left")

    # KVI penetration (placeholder - would require KVI data)
    segment_metrics["kvi_penetration_pct"] = 0.0  # Placeholder

    return segment_metrics[
        [
            "segment",
            "avg_price_paid",
            "price_cv",
            "price_range",
            "price_index_vs_overall",
            "price_orientation_index",
            "premium_product_share_pct",
            "value_product_share_pct",
            "kvi_penetration_pct",
        ]
    ]


def calculate_segment_growth_metrics(
    transactions_df: pd.DataFrame, segments_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate growth and momentum metrics for segments.

    Args:
        transactions_df: Transaction data
        segments_df: DataFrame with customer_id and segment columns (from most recent period)

    Returns:
        DataFrame with segment-level growth metrics
    """
    # Calculate revenue if not present
    df = transactions_df.copy()
    if "revenue" not in df.columns:
        df["revenue"] = df["price"] * df["quantity"]

    # This would require historical data for comparison
    # For now, we'll calculate period-over-period growth if we can split the data
    df["date"] = pd.to_datetime(df["date"])

    if df.empty or df["date"].nunique() < 2:
        # Not enough data for growth calculation
        segment_ids = segments_df["segment"].unique() if not segments_df.empty else []
        return pd.DataFrame(
            {
                "segment": segment_ids,
                "revenue_growth_pct": 0.0,
                "customer_growth_pct": 0.0,
                "frequency_growth_pct": 0.0,
                "avg_order_value_growth_pct": 0.0,
                "segment_share_change_pct": 0.0,
            }
        )

    # Split data into two halves for period-over-period comparison
    median_date = df["date"].median()
    first_half = df[df["date"] <= median_date]
    second_half = df[df["date"] > median_date]

    if first_half.empty or second_half.empty:
        segment_ids = segments_df["segment"].unique() if not segments_df.empty else []
        return pd.DataFrame(
            {
                "segment": segment_ids,
                "revenue_growth_pct": 0.0,
                "customer_growth_pct": 0.0,
                "frequency_growth_pct": 0.0,
                "avg_order_value_growth_pct": 0.0,
                "segment_share_change_pct": 0.0,
            }
        )

    # Calculate metrics for each period
    def _calculate_period_metrics(period_df: pd.DataFrame, period_segments_df: pd.DataFrame) -> pd.DataFrame:
        if period_df.empty:
            return pd.DataFrame()

        # Merge with segment info (using most recent segmentation for simplicity)
        merged = period_df.merge(
            period_segments_df[["customer_id", "segment"]], on="customer_id", how="left"
        )

        if merged.empty or "segment" not in merged.columns:
            return pd.DataFrame()

        metrics = (
            merged.groupby("segment")
            .agg(
                revenue=("revenue", "sum"),
                customers=("customer_id", "nunique"),
                transactions=("transaction_id", "nunique"),
                avg_order_value=("revenue", "mean"),
            )
            .reset_index()
        )

        return metrics

    first_metrics = _calculate_period_metrics(first_half, segments_df)
    second_metrics = _calculate_period_metrics(second_half, segments_df)

    if first_metrics.empty or second_metrics.empty:
        segment_ids = segments_df["segment"].unique() if not segments_df.empty else []
        return pd.DataFrame(
            {
                "segment": segment_ids,
                "revenue_growth_pct": 0.0,
                "customer_growth_pct": 0.0,
                "frequency_growth_pct": 0.0,
                "avg_order_value_growth_pct": 0.0,
                "segment_share_change_pct": 0.0,
            }
        )

    # Merge and calculate growth rates
    growth_metrics = first_metrics.merge(
        second_metrics, on="segment", suffixes=("_first", "_second")
    )

    # Calculate growth rates
    growth_metrics["revenue_growth_pct"] = (
        (growth_metrics["revenue_second"] - growth_metrics["revenue_first"])
        / growth_metrics["revenue_first"].replace(0, np.nan)
        * 100
    ).fillna(0)

    growth_metrics["customer_growth_pct"] = (
        (growth_metrics["customers_second"] - growth_metrics["customers_first"])
        / growth_metrics["customers_first"].replace(0, np.nan)
        * 100
    ).fillna(0)

    growth_metrics["frequency_growth_pct"] = (
        (growth_metrics["transactions_second"] - growth_metrics["transactions_first"])
        / growth_metrics["transactions_first"].replace(0, np.nan)
        * 100
    ).fillna(0)

    growth_metrics["avg_order_value_growth_pct"] = (
        (growth_metrics["avg_order_value_second"] - growth_metrics["avg_order_value_first"])
        / growth_metrics["avg_order_value_first"].replace(0, np.nan)
        * 100
    ).fillna(0)

    # Calculate segment share change
    total_revenue_first = growth_metrics["revenue_first"].sum()
    total_revenue_second = growth_metrics["revenue_second"].sum()

    growth_metrics["revenue_share_first"] = (
        growth_metrics["revenue_first"] / total_revenue_first * 100
    ).fillna(0)

    growth_metrics["revenue_share_second"] = (
        growth_metrics["revenue_second"] / total_revenue_second * 100
    ).fillna(0)

    growth_metrics["segment_share_change_pct"] = (
        growth_metrics["revenue_share_second"] - growth_metrics["revenue_share_first"]
    ).fillna(0)

    return growth_metrics[
        [
            "segment",
            "revenue_growth_pct",
            "customer_growth_pct",
            "frequency_growth_pct",
            "avg_order_value_growth_pct",
            "segment_share_change_pct",
        ]
    ]


def calculate_segment_concentration_metrics(
    segments_df: pd.DataFrame, transactions_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate customer concentration metrics for segments.

    Args:
        segments_df: DataFrame with customer_id and segment columns
        transactions_df: Transaction data with customer_id, price, quantity, etc.

    Returns:
        DataFrame with segment-level concentration metrics
    """
    # Calculate revenue if not present
    df = transactions_df.copy()
    if "revenue" not in df.columns:
        df["revenue"] = df["price"] * df["quantity"]

    # Merge segment info with transaction data
    df = df.merge(segments_df[["customer_id", "segment"]], on="customer_id", how="left")

    # Calculate revenue concentration per segment (Gini coefficient)
    def _gini(array: np.ndarray) -> float:
        """Calculate Gini coefficient of array of values."""
        if len(array) == 0:
            return 0
        sorted_array = np.sort(array)
        n = len(sorted_array)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_array) - (n + 1) * np.sum(sorted_array)) / (
            n * np.sum(sorted_array)
        )

    revenue_gini = df.groupby("segment")["revenue"].apply(_gini).reset_index()
    revenue_gini.columns = ["segment", "revenue_concentration_gini"]

    # Calculate top customer concentration
    def _top_pct_share(group: pd.DataFrame, pct: float) -> float:
        if len(group) == 0:
            return 0
        sorted_revenue = np.sort(group["revenue"])[::-1]  # Descending order
        n_top = max(1, int(len(sorted_revenue) * pct / 100))
        top_revenue = sorted_revenue[:n_top].sum()
        total_revenue = group["revenue"].sum()
        return (top_revenue / total_revenue * 100) if total_revenue > 0 else 0

    top_1pct = df.groupby("segment").apply(lambda g: _top_pct_share(g, 1.0)).reset_index()
    top_1pct.columns = ["segment", "top_1pct_revenue_share"]

    top_5pct = df.groupby("segment").apply(lambda g: _top_pct_share(g, 5.0)).reset_index()
    top_5pct.columns = ["segment", "top_5pct_revenue_share"]

    top_10pct = df.groupby("segment").apply(lambda g: _top_pct_share(g, 10.0)).reset_index()
    top_10pct.columns = ["segment", "top_10pct_revenue_share"]

    # Calculate revenue distribution metrics
    revenue_stats = (
        df.groupby("segment")["revenue"]
        .agg(revenue_mean="mean", revenue_median="median", revenue_std="std")
        .reset_index()
    )

    # Combine all metrics
    concentration_metrics = revenue_gini.merge(top_1pct, on="segment", how="left")
    concentration_metrics = concentration_metrics.merge(top_5pct, on="segment", how="left")
    concentration_metrics = concentration_metrics.merge(top_10pct, on="segment", how="left")
    concentration_metrics = concentration_metrics.merge(revenue_stats, on="segment", how="left")

    return concentration_metrics[
        [
            "segment",
            "revenue_concentration_gini",
            "top_1pct_revenue_share",
            "top_5pct_revenue_share",
            "top_10pct_revenue_share",
            "revenue_mean",
            "revenue_median",
            "revenue_std",
        ]
    ]


def calculate_segment_stability_score(
    segments_df: pd.DataFrame, transactions_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate segment stability score based on multiple factors.

    Args:
        segments_df: DataFrame with customer_id and segment columns
        transactions_df: Transaction data

    Returns:
        DataFrame with segment stability scores
    """
    # Calculate revenue if not present
    df = transactions_df.copy()
    if "revenue" not in df.columns:
        df["revenue"] = df["price"] * df["quantity"]

    # This is a simplified stability score - in practice this would be more complex
    # For now, we'll base it on segment size and consistency

    segment_metrics = (
        segments_df.groupby("segment").agg(customer_count=("customer_id", "nunique")).reset_index()
    )

    # Size score (normalized by log of customer count)
    max_customers = segment_metrics["customer_count"].max()
    if max_customers > 0:
        segment_metrics["size_score"] = np.log(segment_metrics["customer_count"] + 1) / np.log(
            max_customers + 1
        )
    else:
        segment_metrics["size_score"] = 0

    # Consistency score (placeholder - would require temporal data)
    segment_metrics["consistency_score"] = 0.8  # Placeholder

    # Sample size adequacy (based on minimum viable segment size)
    min_viable = 30  # Minimum customers for reliable segmentation
    segment_metrics["sample_size_adequacy"] = np.minimum(
        segment_metrics["customer_count"] / min_viable, 1.0
    )

    # Overall stability score (weighted average)
    segment_metrics["stability_score"] = (
        segment_metrics["size_score"] * 0.3
        + segment_metrics["consistency_score"] * 0.4
        + segment_metrics["sample_size_adequacy"] * 0.3
    ) * 100  # Convert to 0-100 scale

    # Cap at 100
    segment_metrics["stability_score"] = segment_metrics["stability_score"].clip(0, 100)

    # Evidence level (placeholder)
    segment_metrics["evidence_level"] = 3  # Moderate evidence

    return segment_metrics[["segment", "customer_count", "stability_score", "evidence_level"]]


def calculate_segment_distinctiveness(
    segments_df: pd.DataFrame, transactions_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate segment distinctiveness metrics.

    Args:
        segments_df: DataFrame with customer_id and segment columns
        transactions_df: Transaction data

    Returns:
        DataFrame with segment distinctiveness metrics
    """
    # This would require calculating distances between segment centroids
    # For now, we'll provide a simplified measure based on feature variance

    # Calculate revenue if not present
    df = transactions_df.copy()
    if "revenue" not in df.columns:
        df["revenue"] = df["price"] * df["quantity"]

    # Merge segment info with transaction data
    df = df.merge(segments_df[["customer_id", "segment"]], on="customer_id", how="left")

    # Calculate overall means for comparison
    overall_means = df.select_dtypes(include=[np.number]).mean()

    # Calculate segment means
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ["customer_id"]]

    if len(numeric_cols) == 0:
        return pd.DataFrame(
            {
                "segment": segments_df["segment"].unique() if not segments_df.empty else [],
                "distinctiveness_score": 0.0,
                "defining_characteristics": "",
            }
        )

    segment_means = df.groupby("segment")[numeric_cols].mean()

    # Calculate Euclidean distance from overall mean for each segment
    distances = []
    for segment in segment_means.index:
        segment_mean = segment_means.loc[segment]
        # Calculate normalized Euclidean distance
        squared_diffs = ((segment_mean - overall_means) / (overall_means.std() + 1e-10)) ** 2
        distance = np.sqrt(squared_diffs.sum())
        distances.append(distance)

    distinctiveness_df = pd.DataFrame(
        {"segment": segment_means.index, "distinctiveness_score": distances}
    )

    # Normalize distinctiveness score to 0-100 scale
    if distinctiveness_df["distinctiveness_score"].max() > 0:
        distinctiveness_df["distinctiveness_score"] = (
            distinctiveness_df["distinctiveness_score"]
            / distinctiveness_df["distinctiveness_score"].max()
            * 100
        )
    else:
        distinctiveness_df["distinctiveness_score"] = 0

    # Identify defining characteristics (features with highest deviation)
    defining_chars = []
    overall_std = overall_means.std() + 1e-10
    for segment in segment_means.index:
        segment_mean = segment_means.loc[segment]
        # Calculate z-scores for each feature
        z_scores = ((segment_mean - overall_means) / overall_std).abs()
        # Get top 3 defining characteristics
        top_features = z_scores.nlargest(3).index.tolist()
        defining_chars.append(", ".join(top_features))

    distinctiveness_df["defining_characteristics"] = defining_chars

    return distinctiveness_df[["segment", "distinctiveness_score", "defining_characteristics"]]
