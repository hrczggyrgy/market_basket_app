"""Category Overview Analytics - NLP-based category inference and scorecard.

Provides category-level KPIs without requiring a pre-existing category column.
Uses TF-IDF + KMeans on product names to infer product categories.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.analytics.basket_metrics import (
    compute_basket_penetration,
    compute_basket_value_uplift,
    basket_penetration_over_time,
)


def infer_categories_nlp(
    transactions_df: pd.DataFrame,
    n_clusters: int = 8,
    min_products_for_nlp: int = 30,
) -> pd.DataFrame:
    """Infer product categories from product names using TF-IDF + KMeans.

    Returns DataFrame with columns: stockcode, product, category (cluster label).
    Falls back to first-word heuristic if too few products.
    """
    products = (
        transactions_df.groupby("stockcode")
        .agg(product=("product", "first"))
        .reset_index()
    )

    n_products = len(products)
    if n_products < min_products_for_nlp:
        products["category"] = products["product"].apply(
            lambda x: str(x).split()[0] if str(x).strip() else "Unknown"
        )
        return products[["stockcode", "product", "category"]]

    # TF-IDF on product names
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=min(200, n_products * 2),
        stop_words="english",
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(products["product"].fillna(""))

    # KMeans clustering
    actual_clusters = min(n_clusters, n_products // 3, 15)
    if actual_clusters < 2:
        actual_clusters = 2

    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    products["cluster_id"] = kmeans.fit_predict(tfidf_matrix)

    # Generate human-readable category names from cluster centroids
    feature_names = vectorizer.get_feature_names_out()
    category_names = {}
    for cluster_id in range(actual_clusters):
        cluster_center = kmeans.cluster_centers_[cluster_id]
        top_indices = cluster_center.argsort()[-3:][::-1]
        top_terms = [feature_names[i] for i in top_indices if cluster_center[i] > 0]
        if top_terms:
            category_names[cluster_id] = " ".join(top_terms[:2]).title()
        else:
            category_names[cluster_id] = f"Category {cluster_id + 1}"

    products["category"] = products["cluster_id"].map(category_names)
    return products[["stockcode", "product", "category"]]


def compute_category_kpis(
    transactions_df: pd.DataFrame,
    categories_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute category-level KPIs from transaction data.

    One row per inferred category with all scorecard metrics.
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # Merge category info
    df = df.merge(categories_df[["stockcode", "category"]], on="stockcode", how="left")
    
    # Fill missing categories with "Unknown" to avoid KeyError in groupby
    df["category"] = df["category"].fillna("Unknown")

    total_revenue = df["revenue"].sum()
    total_baskets = df.groupby(["customer_id", "date"]).ngroups
    total_customers = df["customer_id"].nunique()

    # Get basket penetration per product
    basket_pen = compute_basket_penetration(df)
    basket_pen = basket_pen.merge(categories_df[["stockcode", "category"]], on="stockcode")

    # Get basket value uplift per product
    uplift = compute_basket_value_uplift(df, top_n=len(df["stockcode"].unique()))
    uplift = uplift.merge(categories_df[["stockcode", "category"]], on="stockcode")

    # Weekly revenue per category for trends and PoP
    df["week"] = df["date"].dt.to_period("W")
    weekly_cat = df.groupby(["category", "week"])["revenue"].sum().reset_index()

    # Compute prior period (4 weeks ago) comparison
    max_week = weekly_cat["week"].max()
    current_weeks = weekly_cat[weekly_cat["week"] > max_week - 4]
    prior_weeks = weekly_cat[
        (weekly_cat["week"] <= max_week - 4) & (weekly_cat["week"] > max_week - 8)
    ]

    current_rev = current_weeks.groupby("category")["revenue"].sum()
    prior_rev = prior_weeks.groupby("category")["revenue"].sum()

    # Category aggregations
    cat_agg = (
        df.groupby("category")
        .agg(
            revenue=("revenue", "sum"),
            units=("quantity", "sum"),
            sku_count=("stockcode", "nunique"),
            unique_customers=("customer_id", "nunique"),
            baskets_with=("customer_id", lambda x: x.nunique()),  # placeholder
        )
        .reset_index()
    )

    # Revenue share
    cat_agg["revenue_share_pct"] = cat_agg["revenue"] / total_revenue * 100

    # MoM delta (4-week vs prior 4-week)
    cat_agg["revenue_mom_delta_pct"] = cat_agg["category"].map(
        lambda c: (
            (current_rev.get(c, 0) - prior_rev.get(c, 0)) / prior_rev.get(c, 1) * 100
            if prior_rev.get(c, 0) > 0
            else np.nan
        )
    )

    # Basket penetration (baskets with category / total baskets)
    cat_baskets = df.groupby("category").apply(
        lambda x: x.groupby(["customer_id", "date"]).ngroups
    ).rename("category_baskets")
    cat_agg = cat_agg.merge(cat_baskets.reset_index(), on="category")
    cat_agg["basket_penetration_pct"] = cat_agg["category_baskets"] / total_baskets * 100

    # Shopper penetration (unique buyers / total buyers)
    cat_agg["shopper_penetration_pct"] = (
        cat_agg["unique_customers"] / total_customers * 100
    )

    # Average purchase frequency (transactions per unique buyer)
    cat_transactions = df.groupby("category")["transaction_id"].nunique().rename("transactions")
    cat_agg = cat_agg.merge(cat_transactions.reset_index(), on="category")
    cat_agg["avg_purchase_frequency"] = (
        cat_agg["transactions"] / cat_agg["unique_customers"].replace(0, np.nan)
    )

    # Basket attachment rate (same as basket penetration for category)
    cat_agg["basket_attachment_rate_pct"] = cat_agg["basket_penetration_pct"]

    # Promo dependency: weeks where price_cv > 0.15 / total weeks
    weekly_price_cv = (
        df.groupby(["category", "week"])["price"]
        .apply(lambda x: x.std() / x.mean() if x.mean() > 0 else 0)
        .reset_index(name="price_cv")
    )
    promo_weeks = weekly_price_cv[weekly_price_cv["price_cv"] > 0.15]
    promo_counts = promo_weeks.groupby("category").size().rename("promo_weeks")
    total_weeks_cat = weekly_price_cv.groupby("category").size().rename("total_weeks")
    promo_dep = (promo_counts / total_weeks_cat * 100).fillna(0).rename("promo_dependency_pct")
    cat_agg = cat_agg.merge(promo_dep.reset_index(), on="category", how="left")
    cat_agg["promo_dependency_pct"] = cat_agg["promo_dependency_pct"].fillna(0)

    # HHI concentration within category (revenue share squared sum)
    def compute_hhi(group):
        shares = group["revenue"] / group["revenue"].sum()
        return (shares**2).sum()

    hhi = df.groupby("category").apply(compute_hhi).rename("hhi_concentration")
    cat_agg = cat_agg.merge(hhi.reset_index(), on="category")

    # Suggested role (Dunnhumby 4-role framework)
    cat_agg["suggested_role"] = cat_agg.apply(_suggest_category_role, axis=1)

    # RAG status vs prior period
    cat_agg["rag_status"] = cat_agg["revenue_mom_delta_pct"].apply(_compute_rag)

    # Weekly revenue series for sparklines (last 12 weeks)
    recent_weeks = weekly_cat[weekly_cat["week"] >= max_week - 11]
    cat_agg["weekly_revenue_series"] = cat_agg["category"].map(
        lambda c: recent_weeks[recent_weeks["category"] == c].sort_values("week")["revenue"].tolist()
    )

    return cat_agg


def _suggest_category_role(row) -> str:
    """Dunnhumby 8-step category role framework simplified to 4 roles.

    Based on shopper_penetration (reach) and basket_attachment (trip share).
    """
    pen = row["shopper_penetration_pct"]
    attach = row["basket_attachment_rate_pct"]

    # Use medians as quadrant boundaries (computed globally)
    # These will be replaced with actual medians in the UI
    if pen >= 50 and attach >= 50:
        return "Destination"
    elif pen >= 50 and attach < 50:
        return "Routine"
    elif pen < 50 and attach >= 50:
        return "Seasonal"
    else:
        return "Convenience"


def _compute_rag(delta_pct) -> str:
    """RAG status: Green if improving, Amber if flat, Red if declining >5%."""
    if pd.isna(delta_pct):
        return "Grey"
    if delta_pct > 5:
        return "Green"
    elif delta_pct < -5:
        return "Red"
    else:
        return "Amber"


def compute_category_scorecard(
    transactions_df: pd.DataFrame,
    n_clusters: int = 8,
) -> pd.DataFrame:
    """Main entry point: infer categories + compute full scorecard.

    Returns category_scorecard_df with all KPIs.
    """
    categories = infer_categories_nlp(transactions_df, n_clusters=n_clusters)
    scorecard = compute_category_kpis(transactions_df, categories)
    return scorecard


def get_category_medians(scorecard_df: pd.DataFrame) -> tuple:
    """Get median shopper_penetration and basket_attachment for quadrant lines."""
    x_med = scorecard_df["shopper_penetration_pct"].median()
    y_med = scorecard_df["basket_attachment_rate_pct"].median()
    return x_med, y_med