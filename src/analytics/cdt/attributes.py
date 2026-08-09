"""CDT product attribute derivation.

Builds the explanatory product attributes used to explain cluster structure
in the Customer Decision Tree: price tier, velocity tier, seasonality class,
basket-size affinity (top-up vs. stock-up missions) and substitution tier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.schemas import CDT_ATTRIBUTES, check


def derive_price_tier(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    n_tiers: int = 3,
    labels: tuple[str, ...] = ("Budget", "Mainstream", "Premium"),
) -> pd.Series:
    """Tier products by robust median selling price (robust to promo spikes).

    NOTE: Price tiers are RELATIVE to the current catalog (quantile-based).
    They are NOT absolute price thresholds. Use for relative positioning only.
    """
    med_price = transactions_df.groupby(product_col)["price"].median()
    effective_q = min(n_tiers, med_price.nunique())
    if effective_q < 2 or med_price.nunique() == 1:
        return pd.Series([labels[0]] * len(med_price), index=med_price.index, name="price_tier")
    tier = pd.qcut(med_price, q=effective_q, labels=labels[:effective_q], duplicates="drop")
    return tier.rename("price_tier")


def derive_velocity_tier(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    n_tiers: int = 3,
    labels: tuple[str, ...] = ("Slow-Moving", "Medium", "Fast-Moving"),
) -> pd.Series:
    """Tier products by units sold per TOTAL observation month (including zero months).

    Also computes active_rate = active_months / total_months as a separate metric.
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")
    
    total_months = df["month"].nunique()
    total_units = df.groupby(product_col)["quantity"].sum()
    active_months = df.groupby(product_col)["month"].nunique()
    
    # Velocity: total units per TOTAL observation month (including zero months)
    velocity = (total_units / total_months).rename("velocity_per_month")
    
    # Active rate: fraction of months with sales
    active_rate = (active_months / total_months).rename("active_rate")
    
    # Tier velocity
    effective_q = min(n_tiers, velocity.nunique())
    if velocity.nunique() <= 1:
        return pd.Series([labels[0]] * len(velocity), index=velocity.index, name="velocity_tier")
    tier = pd.qcut(velocity, q=effective_q, labels=labels[:effective_q], duplicates="drop")
    return tier.rename("velocity_tier")


def derive_basket_size_affinity(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    n_tiers: int = 3,
    labels: tuple[str, ...] = ("Top-Up", "Regular", "Stock-Up"),
) -> pd.Series:
    """Tier products by the mean basket depth (# distinct SKUs) of their trips."""
    df = transactions_df.copy()
    basket_depth = df.groupby("transaction_id")[product_col].nunique().rename("basket_depth")
    df = df.merge(basket_depth.reset_index(), on="transaction_id", how="left")
    mean_depth = df.groupby(product_col)["basket_depth"].mean()
    if mean_depth.nunique() <= 1:
        return pd.Series(
            [labels[0]] * len(mean_depth), index=mean_depth.index, name="basket_size_affinity"
        )
    effective_q = min(n_tiers, mean_depth.nunique())
    tier = pd.qcut(mean_depth, q=effective_q, labels=labels[:effective_q], duplicates="drop")
    return tier.rename("basket_size_affinity")


def derive_seasonality_class(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    seasonal_cv_threshold: float = 0.35,
    sporadic_support_threshold: float = 0.3,
) -> pd.Series:
    """Classify products as Seasonal, Steady, or Sporadic by monthly CV.

    Monthly demand is normalized by the product's annual mean; CV above
    ``seasonal_cv_threshold`` with at least 6 active months marks Seasonal.
    Products sold in fewer than ``sporadic_support_threshold`` of all months
    are Sporadic; otherwise Steady.

    FIX: CV is calculated using ALL months (including zeros for missing months),
    not just active months. This prevents inflating CV for sporadic products.
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")
    all_months = sorted(df["month"].unique())
    n_months = len(all_months)
    
    # Create complete monthly grid with zeros for missing months
    monthly = df.groupby([product_col, "month"])["quantity"].sum().reset_index()
    
    classes: dict[str, str] = {}
    for prod, grp in monthly.groupby(product_col):
        # Create full monthly series including zeros
        demand_series = grp.set_index("month").reindex(all_months, fill_value=0)["quantity"]
        demand = demand_series.to_numpy()
        annual_mean = demand.mean()
        if annual_mean > 0:
            cv = float(demand.std(ddof=1) / annual_mean)
        else:
            cv = 0.0
        
        n_active = (demand > 0).sum()
        if n_active / n_months < sporadic_support_threshold:
            classes[prod] = "Sporadic"
        elif cv > seasonal_cv_threshold and n_active >= 6:
            classes[prod] = "Seasonal"
        else:
            classes[prod] = "Steady"
    return pd.Series(classes, name="seasonality_class")


def derive_substitution_tier(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    n_tiers: int = 3,
    labels: tuple[str, ...] = ("Unique", "Moderately-Substitutable", "Highly-Substitutable"),
    external_dt: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Tier products by substitutability via demand transference SDP scores.

    WARNING: If external_dt is not provided, this uses the SAME data for
    transference computation, creating circularity with CDT clustering.
    To avoid circularity, pass pre-computed transference from a holdout period.
    """
    from src.analytics.transference import (
        compute_demand_transference_matrix,
        compute_substitutable_demand_percentage,
    )

    if external_dt is not None:
        dt = external_dt
    else:
        dt = compute_demand_transference_matrix(transactions_df)
    if dt.empty:
        return pd.Series(
            [labels[0]] * len(transactions_df[product_col].unique()),
            index=transactions_df[product_col].unique(),
            name="substitution_tier",
        )
    sdp = compute_substitutable_demand_percentage(dt, transactions_df).set_index(product_col)["sdp"]
    if sdp.nunique() <= 1:
        return pd.Series([labels[0]] * len(sdp), index=sdp.index, name="substitution_tier")
    effective_q = min(n_tiers, sdp.nunique())
    tier = pd.qcut(sdp, q=effective_q, labels=labels[:effective_q], duplicates="drop")
    return tier.rename("substitution_tier")


def build_transaction_derived_attributes(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    train_period_end: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Assemble the CDT attribute table for every product.

    Args:
        transactions_df: Full transaction dataset
        product_col: Product column name
        train_period_end: If provided, compute attributes ONLY on data up to this date
            (temporal holdout to prevent circularity). Use for training CDT on historical data.
    """
    if train_period_end is not None:
        # Filter to training period only
        transactions_df = transactions_df.copy()
        transactions_df["date"] = pd.to_datetime(transactions_df["date"])
        transactions_df = transactions_df[transactions_df["date"] <= pd.Timestamp(train_period_end)]
    
    tiers: dict[str, pd.Series] = {
        "price_tier": derive_price_tier(transactions_df, product_col),
        "velocity_tier": derive_velocity_tier(transactions_df, product_col),
        "seasonality_class": derive_seasonality_class(transactions_df, product_col),
        "basket_size_affinity": derive_basket_size_affinity(transactions_df, product_col),
        "substitution_tier": derive_substitution_tier(transactions_df, product_col),
    }
    table = pd.DataFrame(tiers)
    table.index.name = product_col
    table = table.reset_index().rename(columns={product_col: "stockcode"})
    return check(table, CDT_ATTRIBUTES)


def get_candidate_attributes() -> list[str]:
    """Column names available for CDT splitting."""
    return ["price_tier", "velocity_tier", "seasonality_class", "basket_size_affinity", "substitution_tier"]