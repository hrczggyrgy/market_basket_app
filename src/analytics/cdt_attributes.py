"""Transaction-derived CDT product attributes.

Generates all product attributes required by the CDT tree builder
exclusively from transaction data — no external product master needed.
This makes the CDT fully self-contained, matching Oracle's recommendation
that CDT runs on customer-linked transaction data alone.

Derived attributes
------------------
price_tier          : Budget / Mainstream / Premium  (from median selling price)
velocity_tier       : Slow / Medium / Fast  (from monthly units per active month)
basket_size_affinity: Small-Trip / Regular / Large-Mission  (mean basket size when present)
seasonality_class   : Seasonal / Steady / Sporadic  (from seasonality CV)
substitution_tier   : Low-Sub / Med-Sub / High-Sub  (mean phi sim to nearest neighbours)

References
----------
- Oracle Retail Science Cloud Services 19.1 — Attribute Processing (ch. 11)
- Oracle Retail AI Foundation 23.2 — CDT Implementation Guide
- Arxiv 2405.05218: Clustering Retail Products Based on Customer Behaviour
"""

from typing import Optional

import numpy as np
import pandas as pd


def derive_price_tier(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    n_tiers: int = 3,
    labels: Optional[list] = None,
) -> pd.Series:
    """Segment products into price tiers from transaction selling prices.

    Uses median selling price per product (robust to promotional outliers).
    Returns a Series indexed by product_col with tier labels.
    """
    if labels is None:
        labels = ["Budget", "Mainstream", "Premium"][:n_tiers]

    med_price = transactions_df.groupby(product_col)["price"].median()
    tier = pd.qcut(med_price, q=n_tiers, labels=labels, duplicates="drop")
    return tier.rename("price_tier")


def derive_velocity_tier(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    n_tiers: int = 3,
    labels: Optional[list] = None,
) -> pd.Series:
    """Classify products by sales velocity: units per active selling month.

    Active months = number of distinct calendar months the product was sold.
    Normalising by active months avoids penalising newer or seasonal products.
    """
    if labels is None:
        labels = ["Slow-Moving", "Medium", "Fast-Moving"][:n_tiers]

    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    total_units = df.groupby(product_col)["quantity"].sum()
    active_months = df.groupby(product_col)["month"].nunique()
    velocity = (total_units / active_months).rename("monthly_units")

    tier = pd.qcut(velocity, q=n_tiers, labels=labels, duplicates="drop")
    return tier.rename("velocity_tier")


def derive_basket_size_affinity(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    n_tiers: int = 3,
    labels: Optional[list] = None,
) -> pd.Series:
    """Classify products by the typical basket size of trips that include them.

    Products bought in small top-up trips vs. large stock-up missions differ
    fundamentally in customer decision context — a key CDT attribute.
    Mean basket size (# unique SKUs) when product is present is computed.
    """
    if labels is None:
        labels = ["Top-Up", "Regular", "Stock-Up"][:n_tiers]

    df = transactions_df.copy()
    if "transaction_id" in df.columns:
        basket_depth = (
            df.groupby("transaction_id")[product_col]
            .nunique()
            .rename("basket_depth")
        )
        df = df.merge(basket_depth.reset_index(), on="transaction_id", how="left")
    else:
        df["_bid"] = (
            df["customer_id"].astype(str) + "_"
            + pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
        )
        basket_depth = df.groupby("_bid")[product_col].nunique().rename("basket_depth")
        df = df.merge(basket_depth.reset_index(), left_on="_bid", right_on="_bid", how="left")

    mean_basket_depth = df.groupby(product_col)["basket_depth"].mean()
    tier = pd.qcut(mean_basket_depth, q=n_tiers, labels=labels, duplicates="drop")
    return tier.rename("basket_size_affinity")


def derive_seasonality_class(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    seasonal_cv_threshold: float = 0.35,
    sporadic_support_threshold: float = 0.3,
) -> pd.Series:
    """Classify products as Seasonal, Steady, or Sporadic.

    Method
    ------
    1. Compute monthly demand series per product (detrended by annual mean).
    2. CV > seasonal_cv_threshold AND >= 6 months data  -> Seasonal
    3. Observed in < sporadic_support_threshold of all months          -> Sporadic
    4. Otherwise                                                        -> Steady

    Seasonality class is a functional-fit-style CDT attribute because it
    captures demand timing — a structural customer decision constraint.
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    all_months = df["month"].nunique()
    monthly_qty = (
        df.groupby([product_col, "month"])["quantity"]
        .sum()
        .reset_index()
    )

    classes = {}
    for prod, grp in monthly_qty.groupby(product_col):
        n_active = grp["month"].nunique()
        support = n_active / all_months if all_months > 0 else 0

        if support < sporadic_support_threshold:
            classes[prod] = "Sporadic"
            continue

        if n_active >= 6:
            # Detrend by year mean
            grp = grp.copy()
            grp["year"] = grp["month"].dt.year
            annual_mean = grp.groupby("year")["quantity"].transform("mean").replace(0, np.nan)
            detrended = grp["quantity"] / annual_mean
            cv = detrended.std() / detrended.mean() if detrended.mean() > 0 else 0
            classes[prod] = "Seasonal" if cv > seasonal_cv_threshold else "Steady"
        else:
            classes[prod] = "Steady"

    return pd.Series(classes, name="seasonality_class")


def derive_substitution_tier(
    similarity_matrix: pd.DataFrame,
    top_k: int = 3,
    n_tiers: int = 3,
    labels: Optional[list] = None,
) -> pd.Series:
    """Classify products by how substitutable they are, from the similarity matrix.

    Mean similarity to top-K nearest neighbours (excluding self).
    High score = many close substitutes in market (High-Sub).
    Low score  = unique product with few substitutes (Low-Sub).

    This attribute acts as a transaction-derived proxy for the Oracle
    'functional fit' concept: products with high substitution tier are
    the ones CDT should group together at deep levels of the tree.
    """
    if labels is None:
        labels = ["Low-Sub", "Med-Sub", "High-Sub"][:n_tiers]

    sim = similarity_matrix.copy()
    np.fill_diagonal(sim.values, np.nan)  # exclude self-similarity

    mean_top_k = sim.apply(
        lambda row: row.nlargest(top_k).mean(), axis=1
    ).rename("mean_top_k_similarity")

    tier = pd.qcut(mean_top_k, q=n_tiers, labels=labels, duplicates="drop")
    return tier.rename("substitution_tier")


def build_transaction_derived_attributes(
    transactions_df: pd.DataFrame,
    similarity_matrix: Optional[pd.DataFrame] = None,
    product_col: str = "stockcode",
    n_tiers: int = 3,
    top_k_sim: int = 3,
) -> pd.DataFrame:
    """Build the full set of transaction-derived CDT attributes in one call.

    Returns a DataFrame indexed by product_col with columns:
        price_tier, velocity_tier, basket_size_affinity,
        seasonality_class, [substitution_tier if similarity_matrix provided]

    This DataFrame is ready to pass directly to build_cdt() as attributes_df.
    """
    attrs = [
        derive_price_tier(transactions_df, product_col, n_tiers),
        derive_velocity_tier(transactions_df, product_col, n_tiers),
        derive_basket_size_affinity(transactions_df, product_col, n_tiers),
        derive_seasonality_class(transactions_df, product_col),
    ]

    if similarity_matrix is not None and not similarity_matrix.empty:
        attrs.append(derive_substitution_tier(similarity_matrix, top_k_sim, n_tiers))

    df = pd.concat(attrs, axis=1)
    df.index.name = product_col
    return df


# Functional-fit attribute registry
# Attributes here are forced to the root of the CDT (Oracle spec).
# Seasonality and substitution tier are structural constraints that
# must be resolved before brand/price splits.
FUNCTIONAL_FIT_ATTRIBUTES = ["seasonality_class", "substitution_tier"]


def get_candidate_attributes(
    attributes_df: pd.DataFrame,
    functional_fit_first: bool = True,
) -> list:
    """Return attribute columns in CDT-recommended split order.

    Functional-fit attributes (seasonality_class, substitution_tier) come first
    per Oracle CDT spec; remaining attributes follow in arbitrary order.
    """
    cols = attributes_df.columns.tolist()
    if functional_fit_first:
        ff = [c for c in FUNCTIONAL_FIT_ATTRIBUTES if c in cols]
        rest = [c for c in cols if c not in FUNCTIONAL_FIT_ATTRIBUTES]
        return ff + rest
    return cols
