"""Price curve diagnostics: univariate and multivariate clustering by price-per-unit."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.analytics.schemas import PRICE_CURVE_1D, PRICE_CURVE_MULTI, check


def _parse_size(size_str: str | float) -> float:
    """Parse pack size string to numeric value in base units (L or KG)."""
    if pd.isna(size_str):
        return 1.0
    size_str = str(size_str).upper()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(ML|L|G|KG|PK|PCS)", size_str)
    if match:
        val = float(match.group(1))
        unit = match.group(2)
        if unit in ("ML", "G"):
            return val / 1000
        elif unit in ("PK", "PCS"):
            return val
        return val
    return 1.0


def _detect_price_curve_violations(cat_data: pd.DataFrame) -> pd.DataFrame:
    """Detect violations: larger pack cheaper per unit (5% tolerance)."""
    if "pack_size_numeric" not in cat_data.columns or "price_per_unit" not in cat_data.columns:
        return pd.DataFrame()

    sorted_data = cat_data.sort_values("pack_size_numeric")
    violations = []

    for i in range(len(sorted_data) - 1):
        row1 = sorted_data.iloc[i]
        row2 = sorted_data.iloc[i + 1]
        if row1["price_per_unit"] > row2["price_per_unit"] * 1.05:
            violations.append(
                {
                    "category": row2["category"],
                    "larger_pack": row2["product_name"],
                    "larger_size": row2["pack_size_numeric"],
                    "larger_price_per_unit": row2["price_per_unit"],
                    "smaller_pack": row1["product_name"],
                    "smaller_size": row1["pack_size_numeric"],
                    "smaller_price_per_unit": row1["price_per_unit"],
                    "violation_pct": (row1["price_per_unit"] / row2["price_per_unit"] - 1) * 100,
                }
            )
    return pd.DataFrame(violations)


def _cluster_and_label(
    cat_data: pd.DataFrame,
    feature_cols: list[str],
    n_tiers: int = 3,
    method: str = "kmeans",
) -> pd.DataFrame:
    """Standardize features, cluster, sort tiers by price_per_unit, label."""
    if len(cat_data) < 2:
        # For single-item categories, assign default tier
        cat_data = cat_data.copy()
        cat_data["tier"] = 0
        tier_labels = {0: "Value", 1: "Mainstream", 2: "Premium", 3: "Ultra", 4: "Luxury"}
        cat_data["tier_label"] = cat_data["tier"].map(tier_labels).fillna("Tier " + cat_data["tier"].astype(str))
        return cat_data

    X = cat_data[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = min(n_tiers, len(cat_data))
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:
        model = GaussianMixture(n_components=n_clusters, random_state=42)

    cat_data["tier"] = model.fit_predict(X_scaled)

    # Sort tiers by mean price_per_unit
    tier_order = cat_data.groupby("tier")["price_per_unit"].mean().sort_values().index
    tier_map = {old: new for new, old in enumerate(tier_order)}
    cat_data["tier"] = cat_data["tier"].map(tier_map)

    tier_labels = {0: "Value", 1: "Mainstream", 2: "Premium", 3: "Ultra", 4: "Luxury"}
    cat_data["tier_label"] = (
        cat_data["tier"].map(tier_labels).fillna("Tier " + cat_data["tier"].astype(str))
    )
    return cat_data


def diagnose_price_curves_1d(
    transactions_df: pd.DataFrame,
    category_col: str = "category",
    price_col: str = "price",
    qty_col: str = "quantity",
    n_tiers: int = 3,
    method: str = "kmeans",
) -> pd.DataFrame:
    """Univariate price-per-unit clustering within category."""
    # Product median price and pack size
    product_info = (
        transactions_df.groupby("stockcode")
        .agg(
            product_name=("product", "first"),
            category=(category_col, "first")
            if category_col in transactions_df.columns
            else ("stockcode", "first"),
            brand=("brand", "first")
            if "brand" in transactions_df.columns
            else ("stockcode", "first"),
            median_price=(price_col, "median"),
            size=("size", "first") if "size" in transactions_df.columns else ("stockcode", "first"),
        )
        .reset_index()
    )

    product_info["pack_size_numeric"] = product_info["size"].apply(_parse_size)
    product_info["price_per_unit"] = product_info["median_price"] / product_info[
        "pack_size_numeric"
    ].replace(0, np.nan)

    all_results = []
    for cat in product_info["category"].unique():
        cat_data = product_info[product_info["category"] == cat].copy()
        cat_data = _cluster_and_label(cat_data, ["price_per_unit"], n_tiers, method)
        all_results.append(cat_data)

    if not all_results:
        return check(pd.DataFrame(columns=list(PRICE_CURVE_1D.columns)), PRICE_CURVE_1D, allow_empty=True)

    result_df = pd.concat(all_results, ignore_index=True)

    # Ensure tier columns always exist with proper defaults
    if "tier" not in result_df.columns:
        result_df["tier"] = 0
    if "tier_label" not in result_df.columns:
        result_df["tier_label"] = "Value"

    # Detect violations
    violations = _detect_price_curve_violations(result_df)
    result_df["has_violation"] = result_df["stockcode"].isin(violations.get("larger_pack", [])) | result_df[
        "stockcode"
    ].isin(violations.get("smaller_pack", []))

    output_cols = [
        "stockcode",
        "product_name",
        "category",
        "brand",
        "median_price",
        "pack_size_numeric",
        "price_per_unit",
        "tier",
        "tier_label",
        "has_violation",
    ]
    table = result_df[output_cols]
    return check(table, PRICE_CURVE_1D)


def diagnose_price_curves_multivariate(
    transactions_df: pd.DataFrame,
    category_col: str = "category",
    price_col: str = "price",
    qty_col: str = "quantity",
    n_tiers: int = 3,
    method: str = "kmeans",
    elasticity_df: Optional[pd.DataFrame] = None,
    cost_col: Optional[str] = None,
) -> pd.DataFrame:
    """Multivariate clustering: price_per_unit + elasticity + basket_pen + margin."""
    from src.analytics.basket_metrics import compute_basket_penetration

    # Product median price and pack size
    product_info = (
        transactions_df.groupby("stockcode")
        .agg(
            product_name=("product", "first"),
            category=(category_col, "first")
            if category_col in transactions_df.columns
            else ("stockcode", "first"),
            brand=("brand", "first")
            if "brand" in transactions_df.columns
            else ("stockcode", "first"),
            median_price=(price_col, "median"),
            size=("size", "first") if "size" in transactions_df.columns else ("stockcode", "first"),
        )
        .reset_index()
    )

    product_info["pack_size_numeric"] = product_info["size"].apply(_parse_size)
    product_info["price_per_unit"] = product_info["median_price"] / product_info[
        "pack_size_numeric"
    ].replace(0, np.nan)

    # Add basket penetration
    basket_pen = compute_basket_penetration(transactions_df)[
        ["stockcode", "penetration", "basket_count"]
    ].rename(columns={"penetration": "basket_penetration"})
    basket_pen["trip_incidence"] = basket_pen["basket_penetration"]
    product_info = product_info.merge(basket_pen, on="stockcode", how="left")

    # Add elasticity if provided
    if elasticity_df is not None and not elasticity_df.empty:
        elast_cols = ["stockcode"]
        if "elasticity" in elasticity_df.columns:
            elast_cols.append("elasticity")
        if "r_squared" in elasticity_df.columns:
            elast_cols.append("r_squared")
        if "price_cv" in elasticity_df.columns:
            elast_cols.append("price_cv")
        product_info = product_info.merge(elasticity_df[elast_cols], on="stockcode", how="left")

    # Add margin if cost column available
    if cost_col and cost_col in transactions_df.columns:
        cost_info = (
            transactions_df.groupby("stockcode").agg(median_cost=(cost_col, "median")).reset_index()
        )
        product_info = product_info.merge(cost_info, on="stockcode", how="left")
        product_info["margin_per_unit"] = (
            product_info["price_per_unit"] - product_info["median_cost"]
        ) / product_info["price_per_unit"].replace(0, np.nan)
    else:
        product_info["margin_per_unit"] = np.nan

    # Clustering features
    feature_cols = ["price_per_unit", "basket_penetration"]
    if "elasticity" in product_info.columns:
        feature_cols.append("elasticity")
    if "margin_per_unit" in product_info.columns:
        feature_cols.append("margin_per_unit")

    all_results = []
    for cat in product_info["category"].unique():
        cat_data = product_info[product_info["category"] == cat].copy()
        cat_data = _cluster_and_label(cat_data, feature_cols, n_tiers, method)
        all_results.append(cat_data)

    if not all_results:
        return check(pd.DataFrame(columns=list(PRICE_CURVE_MULTI.columns)), PRICE_CURVE_MULTI, allow_empty=True)

    result_df = pd.concat(all_results, ignore_index=True)

    # Ensure tier columns always exist with proper defaults
    if "tier" not in result_df.columns:
        result_df["tier"] = 0
    if "tier_label" not in result_df.columns:
        result_df["tier_label"] = "Value"

    # Detect violations
    violations = _detect_price_curve_violations(result_df)
    result_df["has_violation"] = result_df["stockcode"].isin(violations.get("larger_pack", [])) | result_df[
        "stockcode"
    ].isin(violations.get("smaller_pack", []))

    output_cols = [
        "stockcode",
        "product_name",
        "category",
        "brand",
        "median_price",
        "pack_size_numeric",
        "price_per_unit",
        "basket_penetration",
        "trip_incidence",
        "elasticity",
        "margin_per_unit",
        "tier",
        "tier_label",
        "has_violation",
    ]
    output_cols = [c for c in output_cols if c in result_df.columns]
    table = result_df[output_cols]
    return check(table, PRICE_CURVE_MULTI)
