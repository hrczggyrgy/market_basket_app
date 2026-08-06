"""Category-level analytics: KPIs, scorecard, role classification, and text-inferred categories."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import variation
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.analytics.schemas import (
    CATEGORY_KPIS,
    CATEGORY_ROLES,
    CATEGORY_SCORECARD,
    INFERRED_CATEGORIES,
    check,
)


def compute_category_kpis(df: pd.DataFrame, n_periods: int = 8) -> pd.DataFrame:
    """Per-category KPIs: revenue, transactions, customers, penetration, AOV, growth."""
    if "category" not in df.columns or df.empty:
        return check(pd.DataFrame(columns=list(CATEGORY_KPIS.columns)), CATEGORY_KPIS, allow_empty=True)
    df = df.copy()
    revenue = df["price"] * df["quantity"]
    n_baskets = df["transaction_id"].nunique()
    df["_period"] = df["date"].dt.to_period("W").astype(str)
    current = df["_period"].max()
    prior_periods = sorted(df["_period"].unique())[-n_periods - 1 : -1]
    prior = df["_period"].isin(prior_periods)
    table = pd.DataFrame(
        {
            "category": df.groupby("category")["stockcode"].count().index,
            "revenue": revenue.groupby(df["category"]).sum(),
            "transactions": df.groupby("category")["transaction_id"].nunique(),
            "customers": df.groupby("category")["customer_id"].nunique(),
        }
    ).reset_index(drop=True)
    table["penetration"] = table["transactions"] / n_baskets
    table["aov"] = table["revenue"] / table["transactions"]
    table["revenue_share"] = table["revenue"] / table["revenue"].sum()
    recent = revenue.groupby(df["category"]).sum()
    prior_rev = revenue[prior].groupby(df["category"]).sum()
    prior_rev = prior_rev.reindex(recent.index).fillna(0.0).mask(lambda s: s == 0)
    table["growth_pct"] = ((recent - prior_rev) / prior_rev * 100).fillna(0.0)
    table = table.sort_values("revenue", ascending=False).reset_index(drop=True)
    return check(table, CATEGORY_KPIS)


def compute_category_scorecard(df: pd.DataFrame, n_periods: int = 8) -> pd.DataFrame:
    """Category scorecard with role and RAG status."""
    kpis = compute_category_kpis(df, n_periods=n_periods)
    if kpis.empty:
        return check(pd.DataFrame(columns=list(CATEGORY_SCORECARD.columns)), CATEGORY_SCORECARD, allow_empty=True)
    table = kpis.copy()
    table["role"] = table.apply(_suggest_category_role, axis=1)
    table["rag"] = table["growth_pct"].apply(_compute_rag)
    return check(
        table[["category", "revenue", "revenue_share", "transactions", "customers", "aov", "growth_pct", "role", "rag"]],
        CATEGORY_SCORECARD,
    )


def infer_categories_nlp(
    df: pd.DataFrame,
    n_categories: int = 8,
    product_col: str = "product",
) -> pd.DataFrame:
    """Infer product categories by clustering product descriptions (TF-IDF + KMeans)."""
    if product_col not in df.columns:
        return check(pd.DataFrame(columns=list(INFERRED_CATEGORIES.columns)), INFERRED_CATEGORIES, allow_empty=True)
    lookup = df[["stockcode", product_col]].drop_duplicates(subset="stockcode").copy()
    texts = lookup[product_col].fillna("").astype(str)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    features = vectorizer.fit_transform(texts)
    k = min(n_categories, max(2, features.shape[0]))
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(features)
    lookup["inferred_category"] = [f"Cluster {i + 1}" for i in labels]
    return check(lookup.rename(columns={product_col: "product"}), INFERRED_CATEGORIES)


def compute_category_roles(
    df: pd.DataFrame,
    *,
    trip_gen_threshold: float = 0.15,
    demand_cv_threshold: float = 0.25,
    seasonality_amplitude_threshold: float = 0.30,
    attachment_threshold: float = 0.20,
    n_periods: int = 52,
) -> pd.DataFrame:
    """Classify each category into Destination / Routine / Seasonal / Convenience.

    Signals computed per category:
    - trip_generation_rate: % of baskets where this category is the dominant
      category by revenue share within the basket.
    - demand_cv: coefficient of variation of weekly revenue (reusing XYZ logic).
    - seasonality_amplitude: range of monthly revenue indexed to annual average.
    - attachment_rate: % of baskets containing this category that also contain
      a Destination category (computed after Destination is identified).

    Classification logic:
    1. Destination: high trip_generation_rate AND low demand_cv
    2. Seasonal: high seasonality_amplitude
    3. Convenience: low trip_generation_rate AND high attachment_rate to Destination
    4. Routine: everything else (stable, frequent, not strongly seasonal or destination)

    Args:
        df: Transaction DataFrame with columns date, transaction_id, stockcode,
            category, customer_id, price, quantity.
        trip_gen_threshold: Minimum trip_generation_rate for Destination.
        demand_cv_threshold: Maximum demand_cv for Destination.
        seasonality_amplitude_threshold: Minimum amplitude for Seasonal.
        attachment_threshold: Minimum attachment_rate for Convenience.
        n_periods: Number of periods for seasonal analysis (weeks).

    Returns:
        DataFrame with category, role, and all signal values.
    """
    if "category" not in df.columns or df.empty:
        return check(
            pd.DataFrame(columns=list(CATEGORY_ROLES.columns)), CATEGORY_ROLES, allow_empty=True
        )

    df = df.copy()
    revenue = df["price"] * df["quantity"]
    df["revenue"] = revenue

# --- 1. trip_generation_rate ---
    # For each basket, find the dominant category by revenue share within the basket
    basket_cat_rev = df.groupby(["transaction_id", "category"])["revenue"].sum().reset_index()
    basket_total = basket_cat_rev.groupby("transaction_id")["revenue"].transform("sum")
    basket_cat_rev["rev_share"] = basket_cat_rev["revenue"] / basket_total
    dominant_cat = basket_cat_rev.loc[basket_cat_rev.groupby("transaction_id")["rev_share"].idxmax()]
    trip_gen = dominant_cat.groupby("category").size().rename("trip_count")
    total_baskets = df["transaction_id"].nunique()
    trip_generation_rate = (trip_gen / total_baskets).fillna(0.0)

    # Ensure all categories are represented in trip_generation_rate
    all_categories = df["category"].unique()
    trip_generation_rate = trip_generation_rate.reindex(all_categories, fill_value=0.0)

    # --- 2. demand_cv (reuse XYZ logic at category level) ---
    df["_period"] = df["date"].dt.to_period("W").astype(str)
    cat_weekly = df.groupby(["category", "_period"])["revenue"].sum().unstack(fill_value=0)
    demand_cv = cat_weekly.apply(lambda row: variation(row.replace(0, np.nan)), axis=1).fillna(0.0)
    demand_cv = demand_cv.reindex(all_categories, fill_value=0.0)

    # --- 3. seasonality_amplitude ---
    # Monthly revenue indexed to category's own annual average
    df["_month"] = df["date"].dt.to_period("M").astype(str)
    cat_monthly = df.groupby(["category", "_month"])["revenue"].sum().unstack(fill_value=0)
    # Only compute amplitude if we have at least 3 months of data
    valid_months = (cat_monthly > 0).sum(axis=1) >= 3
    cat_avg = cat_monthly.mean(axis=1).replace(0, np.nan)
    monthly_index = cat_monthly.div(cat_avg, axis=0)
    seasonality_amplitude = (monthly_index.max(axis=1) - monthly_index.min(axis=1)).fillna(0.0)
    # Only use seasonality for categories with enough data points
    seasonality_amplitude = seasonality_amplitude.where(valid_months, 0.0)
    seasonality_amplitude = seasonality_amplitude.reindex(all_categories, fill_value=0.0)

    # --- 4. attachment_rate ---
    # First, identify Destination categories based on trip_generation and demand_cv
    temp = pd.DataFrame({
        "trip_generation_rate": trip_generation_rate,
        "demand_cv": demand_cv,
    })
    temp = temp.fillna(0.0)
    temp["is_destination"] = (
        (temp["trip_generation_rate"] >= trip_gen_threshold)
        & (temp["demand_cv"] <= demand_cv_threshold)
    )
    destination_cats = set(temp[temp["is_destination"]].index)

    # For each non-destination category, compute % of its baskets that also contain a destination category
    attachment_rate = pd.Series(0.0, index=all_categories, dtype=float)
    if destination_cats:
        # Find baskets containing destination categories
        dest_transactions = set(
            df[df["category"].isin(destination_cats)]["transaction_id"].unique()
        )
        for cat in all_categories:
            if cat in destination_cats:
                continue
            cat_transactions = set(df[df["category"] == cat]["transaction_id"].unique())
            if cat_transactions:
                overlap = cat_transactions & dest_transactions
                attachment_rate[cat] = len(overlap) / len(cat_transactions)

    # --- Classification ---
    def _classify(row: pd.Series) -> str:
        is_dest = (
            row["trip_generation_rate"] >= trip_gen_threshold
            and row["demand_cv"] <= demand_cv_threshold
        )
        if is_dest:
            return "Destination"

        if row["seasonality_amplitude"] >= seasonality_amplitude_threshold:
            return "Seasonal"

        if (
            row["trip_generation_rate"] < trip_gen_threshold
            and row["attachment_rate"] >= attachment_threshold
        ):
            return "Convenience"

        return "Routine"

    result = pd.DataFrame({
        "category": all_categories,
        "trip_generation_rate": trip_generation_rate.values,
        "demand_cv": demand_cv.values,
        "seasonality_amplitude": seasonality_amplitude.values,
        "attachment_rate": attachment_rate.values,
    }).fillna(0.0)

    result["destination_categories"] = ", ".join(sorted(destination_cats)) if destination_cats else ""
    result["role"] = result.apply(_classify, axis=1)

    return check(result, CATEGORY_ROLES)


def _suggest_category_role(row: pd.Series) -> str:
    if row["revenue_share"] >= 0.2 and row["growth_pct"] > 0:
        return "growth"
    if row["revenue_share"] >= 0.2:
        return "parity"
    if row["growth_pct"] > 15:
        return "traffic_driver"
    return "niche"


def _compute_rag(growth_pct: float) -> str:
    if growth_pct > 10:
        return "green"
    if growth_pct < -10:
        return "red"
    return "amber"
