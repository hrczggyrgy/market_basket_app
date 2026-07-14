"""Customer Segmentation Analytics - RFM, Behavioral, Value-based."""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def compute_rfm_features(
    transactions_df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Compute comprehensive RFM features per customer."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    if snapshot_date is None:
        snapshot_date = df["date"].max() + pd.Timedelta(days=1)

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
            n_unique_categories=(
                ("category", "nunique") if "category" in df.columns else ("stockcode", "nunique")
            ),
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


def rfm_segmentation(
    rfm_df: pd.DataFrame, method: str = "quantile", n_segments: int = 8
) -> pd.DataFrame:
    """Perform RFM-based customer segmentation."""
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
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])

        kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(X_scaled)

        # Label clusters by ranking them relative to each other
        cluster_profiles = df.groupby("cluster")[["recency_days", "frequency", "monetary"]].mean()

        recency_rank = cluster_profiles["recency_days"].rank()
        freq_rank = cluster_profiles["frequency"].rank(ascending=False)
        mon_rank = cluster_profiles["monetary"].rank(ascending=False)

        composite = recency_rank + freq_rank + mon_rank

        cluster_labels = {}
        best_idx = composite.idxmin()
        cluster_labels[best_idx] = "High Value"

        worst_idx = composite.idxmax()
        cluster_labels[worst_idx] = "Churned/At Risk"

        for c in range(n_segments):
            if c in cluster_labels:
                continue
            if mon_rank[c] == mon_rank.min():
                cluster_labels[c] = "Big Spenders"
            elif freq_rank[c] == freq_rank.min():
                cluster_labels[c] = "Frequent Buyers"
            elif recency_rank[c] == recency_rank.min():
                cluster_labels[c] = "Recent Customers"
            else:
                cluster_labels[c] = f"Segment {c}"

        df["segment"] = df["cluster"].map(cluster_labels)

    return df


def behavioral_segmentation(transactions_df: pd.DataFrame, n_clusters: int = 6) -> pd.DataFrame:
    """Behavioral segmentation based on purchase patterns."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # Build behavioral features
    behavioral = (
        df.groupby("customer_id")
        .agg(
            # Temporal
            days_active=("date", lambda x: (x.max() - x.min()).days + 1),
            purchase_frequency=("transaction_id", "nunique"),
            avg_days_between=(
                "date",
                lambda x: (x.max() - x.min()).days / max(x.nunique() - 1, 1),
            ),
            # Monetary
            total_revenue=("revenue", "sum"),
            avg_order_value=("revenue", "mean"),
            revenue_std=("revenue", "std"),
            # Product
            n_products=("stockcode", "nunique"),
            n_categories=(
                ("category", "nunique") if "category" in df.columns else ("stockcode", "nunique")
            ),
            # Basket
            avg_basket_size=("quantity", "mean"),
            max_basket_size=("quantity", "max"),
            # Price
            avg_price=("price", "mean"),
            price_cv=("price", lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
            # Temporal patterns
            weekend_ratio=("date", lambda x: (x.dt.dayofweek >= 5).mean()),
        )
        .reset_index()
    )

    # Fill NaN
    behavioral = behavioral.fillna(0)

    # Features for clustering
    feature_cols = [c for c in behavioral.columns if c != "customer_id"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(behavioral[feature_cols])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    behavioral["cluster"] = kmeans.fit_predict(X_scaled)

    # Label clusters
    cluster_profiles = behavioral.groupby("cluster")[feature_cols].mean()

    labels = {}
    for c in range(n_clusters):
        profile = cluster_profiles.loc[c]
        if profile["total_revenue"] > cluster_profiles["total_revenue"].quantile(0.75):
            labels[c] = "High Value"
        elif profile["purchase_frequency"] > cluster_profiles["purchase_frequency"].quantile(0.75):
            labels[c] = "Frequent Buyers"
        elif profile["avg_days_between"] < cluster_profiles["avg_days_between"].quantile(0.25):
            labels[c] = "Regular Shoppers"
        elif profile["n_products"] > cluster_profiles["n_products"].quantile(0.75):
            labels[c] = "Variety Seekers"
        elif profile["weekend_ratio"] > 0.5:
            labels[c] = "Weekend Shoppers"
        else:
            labels[c] = f"Segment {c}"

    behavioral["segment"] = behavioral["cluster"].map(labels)

    return behavioral


def value_based_segmentation(
    transactions_df: pd.DataFrame, prediction_horizon_days: int = 90
) -> pd.DataFrame:
    """Value-based segmentation with predicted CLV."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    snapshot_date = df["date"].max()
    cutoff_date = snapshot_date - pd.Timedelta(days=prediction_horizon_days)

    # Historical (before cutoff) and future (after cutoff)
    hist = df[df["date"] < cutoff_date]
    future = df[df["date"] >= cutoff_date]

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
        0, 1,
    )
    features["predicted_clv"] = annual_value * survival_prob * 2  # 2-year horizon

    # Segments — first matching condition wins (priority: high CLV + recent > loyal > new > churned)
    conditions = [
        (features["predicted_clv"] > features["predicted_clv"].quantile(0.8))
        & (features["recency"] < 30),
        (features["predicted_clv"] > features["predicted_clv"].quantile(0.6))
        & (features["recency"] < 60),
        (features["frequency"] > 5) & (features["recency"] < 90),
        (features["frequency"] == 1) & (features["recency"] < 30),
        (features["recency"] > 180),
    ]
    choices = ["VIP", "High Potential", "Loyal", "New", "Churned"]
    features["value_segment"] = np.select(conditions, choices, default="Regular")

    return features


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
