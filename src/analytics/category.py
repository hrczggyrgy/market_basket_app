"""Category-level analytics: KPIs, scorecard, and text-inferred categories."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.analytics.schemas import CATEGORY_KPIS, CATEGORY_SCORECARD, INFERRED_CATEGORIES, check


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
