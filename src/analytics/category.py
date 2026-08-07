"""Category-level analytics: KPIs, scorecard, role classification, and text-inferred categories."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import variation
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.analytics.schemas import (
    ASSORTMENT_EFFICIENCY,
    CATEGORY_GROWTH_MATRIX,
    CATEGORY_KPIS,
    CATEGORY_MANAGER_SCORECARD,
    CATEGORY_ROLES,
    CATEGORY_SCORECARD,
    CATEGORY_TREND,
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
    category_source: str | None = None,
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
        category_source: Source of the category column. One of:
            "inferred_nlp" (TF-IDF + KMeans on product descriptions),
            "sample_themes" (sample data THEMES),
            "provided" (explicit column in source data).
            If None, attempts to auto-detect from data patterns.

    Returns:
        DataFrame with category, role, and all signal values.
    """
    if "category" not in df.columns or df.empty:
        return check(
            pd.DataFrame(columns=list(CATEGORY_ROLES.columns)), CATEGORY_ROLES, allow_empty=True
        )

    # Auto-detect category source if not provided
    if category_source is None:
        from src.analytics.sample_data import THEMES as SAMPLE_THEMES
        all_cats = set(df["category"].unique())
        theme_cats = set()
        for theme in SAMPLE_THEMES:
            theme_cats.update(theme)
        if all_cats.issubset(theme_cats) and all_cats:
            category_source = "sample_themes"
        else:
            category_source = "provided"

    df = df.copy()
    revenue = df["price"] * df["quantity"]
    df["revenue"] = revenue
    all_categories = df["category"].unique()

    # ... rest of the function unchanged until result creation ...
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
        "category_source": category_source,
    }).fillna(0.0)

    result["destination_categories"] = ", ".join(sorted(destination_cats)) if destination_cats else ""
    result["role"] = result.apply(_classify, axis=1)

    return check(result, CATEGORY_ROLES)


def compute_category_trend(
    transactions_df: pd.DataFrame,
    freq: str = "W",
) -> pd.DataFrame:
    """Weekly (or per-period) revenue and basket penetration per category.

    One row per (category, period). Penetration is the share of baskets in
    that period that contain the category. Reuses the TRANSACTIONS contract
    columns and returns a CATEGORY_TREND-validated table.
    """
    required = {"category", "date", "transaction_id", "price", "quantity"}
    if not required.issubset(transactions_df.columns) or transactions_df.empty:
        return check(
            pd.DataFrame(columns=list(CATEGORY_TREND.columns)),
            CATEGORY_TREND,
            allow_empty=True,
        )

    df = transactions_df.copy()
    df["_revenue"] = df["price"] * df["quantity"]
    df["_period"] = df["date"].dt.to_period(freq).astype(str)

    per_period = df.groupby("_period")
    total_baskets = per_period["transaction_id"].nunique().rename("n_baskets")

    rows: list[dict] = []
    for (cat, period), grp in df.groupby(["category", "_period"]):
        n_cat_baskets = int(grp["transaction_id"].nunique())
        rows.append(
            {
                "category": cat,
                "period": period,
                "revenue": float(grp["_revenue"].sum()),
                "basket_penetration": n_cat_baskets / total_baskets.get(period, 1),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return check(
            pd.DataFrame(columns=list(CATEGORY_TREND.columns)),
            CATEGORY_TREND,
            allow_empty=True,
        )
    table = table.sort_values(["category", "period"]).reset_index(drop=True)
    return check(table, CATEGORY_TREND)


def compute_assortment_efficiency(
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-category assortment efficiency: SKU share vs revenue share.

    - sku_share: share of total distinct SKUs held by the category.
    - revenue_share: share of total revenue produced by the category.
    - efficiency_index: revenue_share / sku_share (how well SKUs convert
      to revenue; > 1 means revenue outruns SKU weight).
    - efficiency_label: efficient (index > 1.1), balanced, heavy (index < 0.9).

    Requires a ``category`` column present in transactions_df.
    """
    if "category" not in transactions_df.columns or transactions_df.empty:
        return check(
            pd.DataFrame(columns=list(ASSORTMENT_EFFICIENCY.columns)),
            ASSORTMENT_EFFICIENCY,
            allow_empty=True,
        )

    df = transactions_df.copy()
    df["_revenue"] = df["price"] * df["quantity"]

    total_skus = int(df["stockcode"].nunique())
    total_revenue = float(df["_revenue"].sum())

    per_cat_rev = df.groupby("category")["_revenue"].sum()
    per_cat_skus = df.groupby("category")["stockcode"].nunique()
    per_cat_role = compute_category_roles(df)[["category", "role"]].set_index("category")["role"]

    rows: list[dict] = []
    for cat in per_cat_rev.index:
        revenue = float(per_cat_rev.get(cat, 0.0))
        n_skus = int(per_cat_skus.get(cat, 0))
        sku_share = n_skus / total_skus if total_skus else 0.0
        revenue_share = revenue / total_revenue if total_revenue else 0.0
        index = revenue_share / sku_share if sku_share > 0 else 0.0
        if index > 1.1:
            label = "efficient"
        elif index < 0.9:
            label = "under_efficient"
        else:
            label = "balanced"
        rows.append(
            {
                "category": cat,
                "role": per_cat_role.get(cat, "Routine"),
                "sku_share": sku_share,
                "revenue_share": revenue_share,
                "total_revenue": revenue,
                "efficiency_index": index,
                "efficiency_label": label,
            }
        )

    table = pd.DataFrame(rows)
    return check(table, ASSORTMENT_EFFICIENCY)


def compute_category_growth_matrix(
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Internal BCG matrix: revenue share (x) vs growth % (y) per category.

    Quadrants split at the dataset medians:
    - star: high revenue_share AND high growth_pct
    - cash_cow: high share, low growth
    - question_mark: low share, high growth
    - avoid: low share, low growth

    Reuses compute_category_manager_scorecard for revenue_share, growth,
    total_revenue and role, so single source of truth.
    """
    sc = compute_category_manager_scorecard(transactions_df)
    if sc.empty:
        return check(
            pd.DataFrame(columns=list(CATEGORY_GROWTH_MATRIX.columns)),
            CATEGORY_GROWTH_MATRIX,
            allow_empty=True,
        )

    table = sc[["category", "role", "revenue_share", "revenue_yoy_growth", "total_revenue"]].copy()
    table = table.rename(columns={"revenue_yoy_growth": "growth_pct"})
    table = table[table["growth_pct"].notna()]  # growth not computable -> drop
    if table.empty:
        return check(
            pd.DataFrame(columns=list(CATEGORY_GROWTH_MATRIX.columns)),
            CATEGORY_GROWTH_MATRIX,
            allow_empty=True,
        )

    share_med = float(table["revenue_share"].median())
    growth_med = float(table["growth_pct"].median())

    def _quadrant(row: pd.Series) -> str:
        hi_share = row["revenue_share"] >= share_med
        hi_growth = row["growth_pct"] >= growth_med
        if hi_share and hi_growth:
            return "star"
        if hi_share:
            return "cash_cow"
        if hi_growth:
            return "question_mark"
        return "dog"

    table["quadrant"] = table.apply(_quadrant, axis=1)
    table["growth_pct"] = table["growth_pct"].round(2)
    table["total_revenue"] = table["total_revenue"].round(2)
    return check(table, CATEGORY_GROWTH_MATRIX)


def enrich_with_categories(
    df: pd.DataFrame,
    n_categories: int = 8,
    product_col: str = "product",
) -> tuple[pd.DataFrame, bool]:
    """Add a ``category`` column when the dataset lacks one.

    Uses ``infer_categories_nlp`` (TF-IDF + KMeans on product descriptions)
    to derive a pseudo-category per SKU. Datasets that already carry a
    ``category`` column pass through untouched.

    Returns (df, was_inferred) where was_inferred is True only when the
    category column was synthesized.
    """
    if "category" in df.columns or df.empty:
        return df, False
    if product_col not in df.columns:
        return df, False
    inferred = infer_categories_nlp(df, n_categories=n_categories, product_col=product_col)
    if inferred.empty:
        return df, False
    mapping = dict(zip(inferred["stockcode"], inferred["inferred_category"]))
    out = df.copy()
    out["category"] = out["stockcode"].map(mapping).fillna("Unknown")
    return out, True


def compute_category_manager_scorecard(
    transactions_df: pd.DataFrame,
    yoy_window: str = "YE",
) -> pd.DataFrame:
    """Manager-facing category scorecard: one row per category.

    Metrics per column:
    - category: pseudo-category name.
    - role: Destination / Routine / Seasonal / Convenience (from CATEGORY_ROLES).
    - total_revenue: category revenue.
    - revenue_yoy_growth: YoY % growth of revenue (annual windows when the data
      spans >= 2 years, else period-on-period growth on weekly revenue).
    - basket_penetration: % of baskets containing the category.
    - repeat_purchase_rate: share of category customers with >1 transaction.
    - sku_share: % of total SKUs belonging to the category.
    - revenue_share: % of total revenue from the category.
    - kvi_count / kvi_share: number and share of Key Value Items in the category.

    Reuses compute_category_kpis (revenue, penetration, revenue_share),
    compute_category_roles (role), KVI_SCORES (kvi_count/kvi_share).
    """
    if "category" not in transactions_df.columns or transactions_df.empty:
        return check(
            pd.DataFrame(columns=list(CATEGORY_MANAGER_SCORECARD.columns)),
            CATEGORY_MANAGER_SCORECARD,
            allow_empty=True,
        )

    df = transactions_df.copy()
    df["_revenue"] = df["price"] * df["quantity"]

    kpis = compute_category_kpis(df, n_periods=8)
    roles = compute_category_roles(df)
    role_map = dict(zip(roles["category"], roles["role"]))

    total_baskets = max(int(df["transaction_id"].nunique()), 1)
    total_skus = max(int(df["stockcode"].nunique()), 1)
    total_revenue = float(df["_revenue"].sum())

    cat_rows: list[dict] = []
    for cat in kpis["category"].tolist():
        cat_df = df[df["category"] == cat]
        revenue = float(cat_df["_revenue"].sum())
        n_cat_skus = int(cat_df["stockcode"].nunique())
        n_cat_baskets = int(cat_df["transaction_id"].nunique())
        n_cat_customers = int(cat_df["customer_id"].nunique())

        # repeat purchase rate: customers with >1 distinct transaction
        cust_tx = cat_df.groupby("customer_id")["transaction_id"].nunique()
        repeat = int((cust_tx > 1).sum()) if n_cat_customers else 0

        # YoY growth on weekly revenue
        weekly = (
            cat_df.set_index("date")["_revenue"]
            .resample("W")
            .sum()
            .replace(0, np.nan)
            .dropna()
        )
        yoy = _yoy_growth(weekly, yoy_window)
        if yoy is None:
            yoy = float("nan")

        cat_rows.append(
            {
                "category": cat,
                "role": role_map.get(cat, "Routine"),
                "total_revenue": round(revenue, 2),
                "revenue_yoy_growth": round(yoy, 2) if np.isfinite(yoy) else float("nan"),
                "basket_penetration": n_cat_baskets / total_baskets,
                "repeat_purchase_rate": repeat / n_cat_customers if n_cat_customers else 0.0,
                "sku_share": n_cat_skus / total_skus,
                "revenue_share": revenue / total_revenue if total_revenue else 0.0,
                "kvi_count": 0,
                "kvi_share": 0.0,
            }
        )

    # KVI counts per category from KVI_SCORES
    try:
        from src.analytics.pricing.kvi import compute_kvi_score

        kvi = compute_kvi_score(df, method="heuristic")
        if not kvi.empty:
            kvi_by_cat = kvi.groupby("category")["stockcode"].nunique()
            total_kvi = int(kvi["stockcode"].nunique())
            for row in cat_rows:
                kc = int(kvi_by_cat.get(row["category"], 0))
                row["kvi_count"] = kc
                row["kvi_share"] = kc / total_kvi if total_kvi else 0.0
    except Exception:
        pass

    table = pd.DataFrame(cat_rows)
    return check(table, CATEGORY_MANAGER_SCORECARD)


def _yoy_growth(weekly_revenue: pd.Series, window: str) -> float | None:
    """Annualized growth of a weekly revenue series, or None when not computable."""
    if len(weekly_revenue) < 2:
        return None
    grp = weekly_revenue.groupby(pd.Grouper(freq=window))
    periods = list(grp.groups)
    if len(periods) >= 2:
        recent = float(grp.get_group(periods[-1]).sum())
        prior = float(grp.get_group(periods[-2]).sum())
    else:
        recent = float(grp.get_group(periods[-1]).sum())
        prior = float(weekly_revenue[weekly_revenue.index < periods[-1]].sum())
    if prior <= 0:
        return None
    return (recent - prior) / prior * 100


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
