"""Basket-level shopper metrics.

Implements the Dunnhumby / Circana / Oracle standard basket-layer KPIs that
operate at the shopping-trip level rather than the individual product line.
All metrics are derived exclusively from transaction data.

References
----------
- Circana Basket Analysis 101 for CPG (2026)
- Dunnhumby Complete Journey methodology
- Oracle Retail AI Foundation Cloud Service docs (23.2)
- Martin et al. (2020) "Fundamental basket size patterns", J. Retailing & Consumer Svcs.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _total_baskets(transactions_df: pd.DataFrame) -> int:
    """Total unique baskets (trips) in the dataset."""
    if "transaction_id" in transactions_df.columns:
        return transactions_df["transaction_id"].nunique()
    # Fallback: (customer_id, date) pairs as proxy baskets
    return transactions_df.groupby(["customer_id", "date"]).ngroups


def _basket_ids(transactions_df: pd.DataFrame) -> pd.Series:
    """Return a basket-id column (transaction_id if available, else customer+date)."""
    if "transaction_id" in transactions_df.columns:
        return transactions_df["transaction_id"]
    return (
        transactions_df["customer_id"].astype(str)
        + "_"
        + pd.to_datetime(transactions_df["date"]).dt.strftime("%Y%m%d")
    )


# ---------------------------------------------------------------------------
# 1. Basket penetration
# ---------------------------------------------------------------------------

def compute_basket_penetration(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
) -> pd.DataFrame:
    """Compute basket penetration metrics for every product.

    Metrics returned
    ----------------
    basket_penetration      : baskets containing product / total baskets
    unique_shopper_pen      : unique customers buying product / total unique customers
    category_basket_pen     : baskets with product / baskets with any product in same category
                              (requires 'category' column; NaN otherwise)
    trip_incidence          : alias for basket_penetration (industry standard name)

    Parameters
    ----------
    transactions_df : DataFrame with at least customer_id, stockcode, date columns.
    product_col     : Column identifying products (default 'stockcode').
    """
    df = transactions_df.copy()
    df["_basket"] = _basket_ids(df)
    df["date"] = pd.to_datetime(df["date"])

    total_baskets = df["_basket"].nunique()
    total_customers = df["customer_id"].nunique()

    # Baskets per product
    baskets_per_product = (
        df.groupby(product_col)["_basket"].nunique().rename("baskets_with_product")
    )
    # Customers per product
    customers_per_product = (
        df.groupby(product_col)["customer_id"].nunique().rename("unique_customers")
    )

    result = pd.concat([baskets_per_product, customers_per_product], axis=1).reset_index()
    result["basket_penetration"] = result["baskets_with_product"] / total_baskets
    result["unique_shopper_penetration"] = result["unique_customers"] / total_customers
    result["trip_incidence"] = result["basket_penetration"]  # alias

    # Category-conditional basket penetration
    if "category" in df.columns:
        cat_map = df.groupby(product_col)["category"].first()
        result["category"] = result[product_col].map(cat_map)
        cat_baskets = (
            df.groupby("category")["_basket"]
            .nunique()
            .rename("category_total_baskets")
        )
        result = result.merge(
            cat_baskets.reset_index(), on="category", how="left"
        )
        result["category_basket_penetration"] = (
            result["baskets_with_product"] / result["category_total_baskets"]
        )
        result.drop(columns=["category_total_baskets"], inplace=True)
    else:
        result["category_basket_penetration"] = np.nan

    return result.rename(columns={product_col: "stockcode"}).reset_index(drop=True)


def basket_penetration_over_time(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    freq: str = "ME",
) -> pd.DataFrame:
    """Monthly/weekly basket penetration trend per product.

    Returns a long DataFrame: stockcode | period | basket_penetration
    Useful for sparkline / time-series visualizations.
    """
    df = transactions_df.copy()
    df["_basket"] = _basket_ids(df)
    df["date"] = pd.to_datetime(df["date"])
    df["period"] = df["date"].dt.to_period(freq)

    period_totals = df.groupby("period")["_basket"].nunique().rename("total_baskets")
    product_period = (
        df.groupby([product_col, "period"])["_basket"]
        .nunique()
        .rename("baskets_with_product")
        .reset_index()
    )
    product_period = product_period.merge(
        period_totals.reset_index(), on="period", how="left"
    )
    product_period["basket_penetration"] = (
        product_period["baskets_with_product"] / product_period["total_baskets"]
    )
    return product_period.rename(columns={product_col: "stockcode"})


# ---------------------------------------------------------------------------
# 2. Basket composition KPIs
# ---------------------------------------------------------------------------

def compute_basket_composition(
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Store-level basket composition statistics.

    Metrics
    -------
    avg_basket_value     : mean revenue per basket
    avg_basket_size_units: mean units per basket
    avg_basket_depth_skus: mean unique SKUs per basket
    median_basket_value
    p25_basket_value
    p75_basket_value
    """
    df = transactions_df.copy()
    df["_basket"] = _basket_ids(df)
    df["revenue"] = df["price"] * df["quantity"]

    basket_stats = (
        df.groupby("_basket")
        .agg(
            basket_value=("revenue", "sum"),
            basket_units=("quantity", "sum"),
            basket_depth=("stockcode", "nunique"),
        )
    )

    return pd.DataFrame({
        "avg_basket_value": [basket_stats["basket_value"].mean()],
        "median_basket_value": [basket_stats["basket_value"].median()],
        "p25_basket_value": [basket_stats["basket_value"].quantile(0.25)],
        "p75_basket_value": [basket_stats["basket_value"].quantile(0.75)],
        "avg_basket_size_units": [basket_stats["basket_units"].mean()],
        "avg_basket_depth_skus": [basket_stats["basket_depth"].mean()],
        "total_baskets": [len(basket_stats)],
    })


def compute_basket_value_uplift(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    top_n: int = 50,
) -> pd.DataFrame:
    """Basket value uplift: mean basket value WITH product vs WITHOUT.

    Uplift > 0 means the product is associated with higher-value trips.
    This is a key Dunnhumby halo metric — products with high uplift justify
    prominent placement even if their own revenue is modest.

    Returns DataFrame: stockcode | avg_basket_with | avg_basket_without | uplift | uplift_pct
    """
    df = transactions_df.copy()
    df["_basket"] = _basket_ids(df)
    df["revenue"] = df["price"] * df["quantity"]

    basket_value = df.groupby("_basket")["revenue"].sum().rename("basket_value")
    basket_products = df.groupby("_basket")[product_col].apply(set)

    # Top-N products by frequency
    top_products = (
        df.groupby(product_col)["_basket"].nunique()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    rows = []
    all_baskets = basket_value.index
    for prod in top_products:
        baskets_with = basket_value[
            basket_products.index[basket_products.apply(lambda s: prod in s)]
        ]
        baskets_without = basket_value[
            basket_products.index[basket_products.apply(lambda s: prod not in s)]
        ]
        avg_with = baskets_with.mean() if len(baskets_with) > 0 else np.nan
        avg_without = baskets_without.mean() if len(baskets_without) > 0 else np.nan
        uplift = avg_with - avg_without if not np.isnan(avg_with) and not np.isnan(avg_without) else np.nan
        uplift_pct = uplift / avg_without * 100 if avg_without and avg_without > 0 else np.nan
        rows.append({
            product_col: prod,
            "baskets_with": len(baskets_with),
            "avg_basket_value_with": avg_with,
            "avg_basket_value_without": avg_without,
            "basket_value_uplift": uplift,
            "basket_value_uplift_pct": uplift_pct,
        })

    return pd.DataFrame(rows).sort_values("basket_value_uplift", ascending=False).reset_index(drop=True)


def compute_cross_category_basket_rate(
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-category basket co-occurrence rate.

    Returns a square DataFrame (categories x categories) where each cell
    = % of baskets containing BOTH cat_A and cat_B.
    Diagonal = category basket penetration (single-category).
    Requires 'category' column.
    """
    if "category" not in transactions_df.columns:
        return pd.DataFrame()

    df = transactions_df.copy()
    df["_basket"] = _basket_ids(df)

    basket_cats = df.groupby("_basket")["category"].apply(set)
    categories = sorted({c for cats in basket_cats for c in cats})
    total_baskets = len(basket_cats)

    matrix = pd.DataFrame(0.0, index=categories, columns=categories)
    for cats in basket_cats:
        cats_list = list(cats)
        for i, ca in enumerate(cats_list):
            for cb in cats_list[i:]:
                matrix.loc[ca, cb] += 1
                if ca != cb:
                    matrix.loc[cb, ca] += 1

    return (matrix / total_baskets).round(4)


# ---------------------------------------------------------------------------
# 3. Co-purchase index (Dunnhumby / Circana indexed affinity)
# ---------------------------------------------------------------------------

def compute_copurchase_index(
    transactions_df: pd.DataFrame,
    top_n_products: int = 50,
) -> pd.DataFrame:
    """Co-purchase index = lift * 100 presented as an indexed affinity metric.

    Index > 100: above-average pairing.
    Index > 180: strong complement (Circana threshold).
    Index < 50 : products rarely paired — potential substitutes or unrelated.

    Returns long DataFrame: product_a | product_b | copurchase_index | baskets_both
    """
    df = transactions_df.copy()
    df["_basket"] = _basket_ids(df)
    total_baskets = df["_basket"].nunique()

    top_products = (
        df.groupby("stockcode")["_basket"].nunique()
        .sort_values(ascending=False)
        .head(top_n_products)
        .index.tolist()
    )

    basket_product = (
        df[df["stockcode"].isin(top_products)]
        .groupby("_basket")["stockcode"]
        .apply(set)
    )
    penetration = {
        p: sum(1 for s in basket_product if p in s) / total_baskets
        for p in top_products
    }

    rows = []
    for i, pa in enumerate(top_products):
        for pb in top_products[i + 1:]:
            both = sum(1 for s in basket_product if pa in s and pb in s)
            p_ab = both / total_baskets
            p_a = penetration[pa]
            p_b = penetration[pb]
            if p_a > 0 and p_b > 0 and p_ab > 0:
                index = (p_ab / (p_a * p_b)) * 100
                rows.append({
                    "product_a": pa,
                    "product_b": pb,
                    "copurchase_index": round(index, 1),
                    "baskets_both": both,
                    "basket_penetration_a": round(p_a, 4),
                    "basket_penetration_b": round(p_b, 4),
                })

    return (
        pd.DataFrame(rows)
        .sort_values("copurchase_index", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 4. Shopper loyalty metrics (transaction-derived)
# ---------------------------------------------------------------------------

def compute_shopper_loyalty_metrics(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    snapshot_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Per-product shopper loyalty metrics derived from transaction history.

    Metrics
    -------
    repeat_rate          : % of buyers who purchased 2+ times
    reorder_interval_days: median days between consecutive purchases (per customer)
    loyalty_index        : mean share of category purchases devoted to this product
                           per customer (requires 'category' column)
    switcher_rate        : % of first-time buyers who never returned (1 - repeat_rate)
    exclusivity_rate     : % of buyers who bought ONLY this product in its CDT cluster
                           (approximated as % who bought only 1 unique product in category)
    time_to_second_purchase: median days from first to second purchase per customer
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if snapshot_date is None:
        snapshot_date = df["date"].max()

    rows = []
    products = df[product_col].unique()

    for prod in products:
        prod_df = df[df[product_col] == prod].copy()
        buyers = prod_df["customer_id"].unique()
        n_buyers = len(buyers)
        if n_buyers == 0:
            continue

        # Repeat rate
        repeat_buyers = (
            prod_df.groupby("customer_id")["date"]
            .nunique()
            .gt(1)
            .sum()
        )
        repeat_rate = repeat_buyers / n_buyers

        # Reorder interval & time to second purchase
        intervals = []
        tt2p_list = []
        for cid, grp in prod_df.groupby("customer_id"):
            dates = grp["date"].drop_duplicates().sort_values().tolist()
            if len(dates) >= 2:
                gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
                intervals.extend(gaps)
                tt2p_list.append((dates[1] - dates[0]).days)

        reorder_interval = np.median(intervals) if intervals else np.nan
        tt2p = np.median(tt2p_list) if tt2p_list else np.nan

        # Loyalty index (share of category wallet, requires category)
        loyalty_index = np.nan
        exclusivity_rate = np.nan
        if "category" in df.columns:
            prod_cat = prod_df["category"].iloc[0] if len(prod_df) > 0 else None
            if prod_cat:
                cat_df = df[(df["category"] == prod_cat) & (df["customer_id"].isin(buyers))]
                customer_cat_purchases = cat_df.groupby("customer_id")["date"].nunique()
                customer_prod_purchases = prod_df.groupby("customer_id")["date"].nunique()
                shared = customer_cat_purchases.index.intersection(customer_prod_purchases.index)
                if len(shared) > 0:
                    shares = customer_prod_purchases[shared] / customer_cat_purchases[shared]
                    loyalty_index = shares.mean()

                # Exclusivity: buyers who bought only 1 unique product in category
                cat_unique_per_buyer = cat_df.groupby("customer_id")[product_col].nunique()
                exclusive = (cat_unique_per_buyer == 1).sum()
                exclusivity_rate = exclusive / len(cat_unique_per_buyer) if len(cat_unique_per_buyer) > 0 else np.nan

        rows.append({
            product_col: prod,
            "n_buyers": n_buyers,
            "repeat_rate": round(repeat_rate, 4),
            "switcher_rate": round(1 - repeat_rate, 4),
            "reorder_interval_days": round(reorder_interval, 1) if not np.isnan(reorder_interval) else np.nan,
            "time_to_second_purchase_days": round(tt2p, 1) if not np.isnan(tt2p) else np.nan,
            "loyalty_index": round(loyalty_index, 4) if not np.isnan(loyalty_index) else np.nan,
            "exclusivity_rate": round(exclusivity_rate, 4) if not np.isnan(exclusivity_rate) else np.nan,
        })

    return (
        pd.DataFrame(rows)
        .rename(columns={product_col: "stockcode"})
        .sort_values("repeat_rate", ascending=False)
        .reset_index(drop=True)
    )
