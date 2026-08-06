"""RFM (Recency, Frequency, Monetary) Segmentation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.analytics.segmentation.core import _label_rfm_clusters
from src.analytics.schemas import RFM_FEATURES, RFM_SEGMENTS, check


def compute_rfm_features(
    transactions_df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Compute comprehensive RFM features per customer.

    Args:
        transactions_df: Transaction data with date, transaction_id, customer_id,
            stockcode, price, quantity
        snapshot_date: Analysis snapshot date (default: max date + 1 day)

    Returns:
        DataFrame with RFM features and derived metrics per customer.
    """
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

    return check(rfm, RFM_FEATURES)


def rfm_segmentation(
    rfm_df: pd.DataFrame, method: str = "quantile", n_segments: int = 8
) -> pd.DataFrame:
    """Segment customers based on RFM features.

    Args:
        rfm_df: DataFrame with RFM features (output of compute_rfm_features)
        method: 'quantile' for classic RFM scoring, 'kmeans' for clustering
        n_segments: Number of segments for k-means

    Returns:
        DataFrame with segment assignments and scores.
    """
    df = rfm_df.copy()

    if method == "quantile":
        # Classic RFM scoring (1-4 per dimension)
        for dim, score_name in [("recency_days", "recency_score"), ("frequency", "frequency_score"), ("monetary", "monetary_score")]:
            if dim == "recency_days":
                df[score_name] = pd.qcut(
                    df[dim].rank(method="first"),
                    q=4,
                    labels=[4, 3, 2, 1],
                    duplicates="drop",
                )
            else:
                df[score_name] = pd.qcut(
                    df[dim].rank(method="first"),
                    q=4,
                    labels=[1, 2, 3, 4],
                    duplicates="drop",
                )

        df["rfm_score"] = (
            df["recency_score"].astype(str)
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
        df["cluster"] = -1

    elif method == "kmeans":
        # K-means clustering on RFM
        features = ["recency_days", "frequency", "monetary"]
        if len(df) < n_segments:
            df["cluster"] = 0
            df["segment"] = "Other"
            return check(df, RFM_SEGMENTS)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])

        kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(X_scaled)

        cluster_profiles = df.groupby("cluster")[["recency_days", "frequency", "monetary"]].mean()
        cluster_labels = _label_rfm_clusters(cluster_profiles)
        df["segment"] = df["cluster"].map(cluster_labels)

        # Also compute quantile scores for compatibility
        for dim, score_name in [("recency_days", "recency_score"), ("frequency", "frequency_score"), ("monetary", "monetary_score")]:
            if dim == "recency_days":
                df[score_name] = pd.qcut(
                    df[dim].rank(method="first"),
                    q=4,
                    labels=[4, 3, 2, 1],
                    duplicates="drop",
                )
            else:
                df[score_name] = pd.qcut(
                    df[dim].rank(method="first"),
                    q=4,
                    labels=[1, 2, 3, 4],
                    duplicates="drop",
                )
        df["rfm_score"] = (
            df["recency_score"].astype(str)
            + df["frequency_score"].astype(str)
            + df["monetary_score"].astype(str)
        )

    return check(df, RFM_SEGMENTS)