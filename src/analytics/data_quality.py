"""Data quality summaries and method readiness checks.

Pure functions for data health reporting. No persistence, no UI.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def summarize_data_quality(df: pd.DataFrame) -> Dict:
    """Comprehensive data quality summary for transaction data.

    Returns dict with counts, coverage, and anomaly flags.
    """
    if df is None or df.empty:
        return {"error": "Empty or None DataFrame"}

    result = {}

    # Basic counts
    result["n_rows"] = len(df)
    result["n_transactions"] = df["transaction_id"].nunique() if "transaction_id" in df.columns else 0
    result["n_customers"] = df["customer_id"].nunique() if "customer_id" in df.columns else 0
    result["n_products"] = df["stockcode"].nunique() if "stockcode" in df.columns else 0

    # Date range
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        result["date_min"] = dates.min()
        result["date_max"] = dates.max()
        result["date_span_days"] = (result["date_max"] - result["date_min"]).days if pd.notna(result["date_min"]) else 0
    else:
        result["date_min"] = result["date_max"] = result["date_span_days"] = None

    # Missing values
    result["missing"] = df.isnull().sum().to_dict()
    result["missing_pct"] = (df.isnull().mean() * 100).round(2).to_dict()

    # Customer ID coverage
    if "customer_id" in df.columns:
        null_cust = df["customer_id"].isnull().sum()
        result["missing_customer_id"] = int(null_cust)
        result["missing_customer_id_pct"] = round(null_cust / len(df) * 100, 2)
    else:
        result["missing_customer_id"] = result["missing_customer_id_pct"] = None

    # Duplicate transaction lines (same transaction_id + stockcode)
    if "transaction_id" in df.columns and "stockcode" in df.columns:
        dup_mask = df.duplicated(subset=["transaction_id", "stockcode"], keep=False)
        result["duplicate_txn_lines"] = int(dup_mask.sum())
        result["duplicate_txn_lines_pct"] = round(dup_mask.sum() / len(df) * 100, 2)
    else:
        result["duplicate_txn_lines"] = result["duplicate_txn_lines_pct"] = None

    # Non-positive price/quantity
    if "price" in df.columns:
        nonpos_price = (df["price"] <= 0).sum()
        result["nonpositive_price"] = int(nonpos_price)
        result["nonpositive_price_pct"] = round(nonpos_price / len(df) * 100, 2)
    else:
        result["nonpositive_price"] = result["nonpositive_price_pct"] = None

    if "quantity" in df.columns:
        nonpos_qty = (df["quantity"] <= 0).sum()
        result["nonpositive_quantity"] = int(nonpos_qty)
        result["nonpositive_quantity_pct"] = round(nonpos_qty / len(df) * 100, 2)
    else:
        result["nonpositive_quantity"] = result["nonpositive_quantity_pct"] = None

    # Revenue
    if "price" in df.columns and "quantity" in df.columns:
        result["total_revenue"] = float((df["price"] * df["quantity"]).sum())
    else:
        result["total_revenue"] = None

    # Sparsity: SKUs with very few transactions
    if "stockcode" in df.columns and "transaction_id" in df.columns:
        sku_txn_counts = df.groupby("stockcode")["transaction_id"].nunique()
        sparse_threshold = max(1, sku_txn_counts.quantile(0.1))
        result["sparse_sku_count"] = int((sku_txn_counts <= sparse_threshold).sum())
        result["sparse_sku_threshold"] = int(sparse_threshold)
    else:
        result["sparse_sku_count"] = result["sparse_sku_threshold"] = None

    # Category/attribute coverage
    attr_cols = ["category", "brand", "size", "flavor", "variant", "flavour"]
    result["attribute_coverage"] = {}
    for col in attr_cols:
        if col in df.columns:
            nonnull = df[col].notna().sum()
            result["attribute_coverage"][col] = {
                "covered": int(nonnull),
                "pct": round(nonnull / len(df) * 100, 2),
            }

    return result


def validate_price_quantity(df: pd.DataFrame) -> Dict[str, List]:
    """Validate price and quantity columns for common issues.

    Returns dict with 'errors', 'warnings', 'info' lists of messages.
    """
    errors, warnings, info = [], [], []

    if df is None or df.empty:
        errors.append("DataFrame is empty or None")
        return {"errors": errors, "warnings": warnings, "info": info}

    # Price checks
    if "price" in df.columns:
        prices = df["price"]
        if prices.isnull().any():
            warnings.append(f"Price has {prices.isnull().sum()} null values")
        if (prices < 0).any():
            errors.append(f"Price has {(prices < 0).sum()} negative values (returns?)")
        if (prices == 0).any():
            warnings.append(f"Price has {(prices == 0).sum()} zero values")
        cv = prices.std() / prices.mean() if prices.mean() > 0 else np.inf
        if cv < 0.01:
            warnings.append(f"Overall price CV very low ({cv:.4f}) — limited price variation")

    # Quantity checks
    if "quantity" in df.columns:
        qtys = df["quantity"]
        if qtys.isnull().any():
            warnings.append(f"Quantity has {qtys.isnull().sum()} null values")
        if (qtys < 0).any():
            info.append(f"Quantity has {(qtys < 0).sum()} negative values (likely returns)")
        if (qtys == 0).any():
            warnings.append(f"Quantity has {(qtys == 0).sum()} zero values")

    # Price × quantity consistency
    if "price" in df.columns and "quantity" in df.columns:
        revenue = df["price"] * df["quantity"]
        if (revenue < 0).any():
            info.append(f"{(revenue < 0).sum()} rows have negative revenue (price×qty)")

    return {"errors": errors, "warnings": warnings, "info": info}


def find_sku_description_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Find SKUs with multiple different product descriptions.

    Returns DataFrame with stockcode, n_descriptions, descriptions list.
    """
    if df is None or df.empty or "stockcode" not in df.columns or "product" not in df.columns:
        return pd.DataFrame()

    conflicts = (
        df.groupby("stockcode")["product"]
        .nunique()
        .reset_index(name="n_descriptions")
        .query("n_descriptions > 1")
    )

    if conflicts.empty:
        return pd.DataFrame()

    # Get the actual descriptions
    desc_map = df.groupby("stockcode")["product"].apply(lambda x: list(x.unique())).reset_index(name="descriptions")
    conflicts = conflicts.merge(desc_map, on="stockcode")
    return conflicts.sort_values("n_descriptions", ascending=False)


def calculate_method_readiness(
    df: pd.DataFrame,
    analysis_name: str,
    params: Optional[Dict] = None
) -> Dict:
    """Assess readiness for a specific analysis method.

    Returns dict with 'status' (ready/directional/blocked), 'details', 'requirements'.
    """
    params = params or {}
    summary = summarize_data_quality(df)
    validation = validate_price_quantity(df)

    requirements = []
    details = {}

    if analysis_name in ["elasticity", "elasticity_analysis"]:
        # Need price variation, sufficient periods per SKU
        min_periods = params.get("min_periods", 10)
        min_cv = params.get("min_price_variation", 0.05)

        if "price" in df.columns and "stockcode" in df.columns:
            price_cv = df.groupby("stockcode")["price"].apply(
                lambda x: x.std() / x.mean() if x.mean() > 0 else 0
            )
            eligible = (price_cv >= min_cv).sum()
            total = len(price_cv)
            details["eligible_skus"] = int(eligible)
            details["total_skus"] = int(total)
            details["eligible_pct"] = round(eligible / total * 100, 1) if total > 0 else 0
            details["median_price_cv"] = round(price_cv.median(), 4)
            requirements.append(f"Min {min_periods} price periods per SKU")
            requirements.append(f"Min price CV ≥ {min_cv:.0%}")

            if eligible == 0:
                return {"status": "blocked", "details": details, "requirements": requirements,
                        "reason": "No SKUs meet minimum price variation"}
            elif eligible < 5:
                return {"status": "directional", "details": details, "requirements": requirements,
                        "reason": f"Only {eligible} SKUs eligible — results exploratory"}
            else:
                return {"status": "ready", "details": details, "requirements": requirements}

    elif analysis_name in ["promo_uplift", "promo_uplift_modeling"]:
        # Need treatment/control overlap, promo periods
        n_transactions = summary.get("n_transactions", 0)
        n_customers = summary.get("n_customers", 0)

        details["n_transactions"] = n_transactions
        details["n_customers"] = n_customers
        requirements.append("Promo detection feasibility (price drop threshold)")
        requirements.append("Treatment/control overlap for causal inference")

        if n_transactions < 1000 or n_customers < 100:
            return {"status": "blocked", "details": details, "requirements": requirements,
                    "reason": "Insufficient data for uplift modeling"}
        else:
            return {"status": "directional", "details": details, "requirements": requirements,
                    "reason": "Causal uplift requires strong overlap; treat as exploratory"}

    elif analysis_name in ["cdt", "cdt_builder", "demand_transference", "assortment_optimizer"]:
        # Need sufficient co-occurrence, products, customers
        n_products = summary.get("n_products", 0)
        n_customers = summary.get("n_customers", 0)
        n_transactions = summary.get("n_transactions", 0)
        min_cooc = params.get("min_cooccurrence", 5)

        details["n_products"] = n_products
        details["n_customers"] = n_customers
        details["n_transactions"] = n_transactions
        details["min_cooccurrence"] = min_cooc

        attr_cov = summary.get("attribute_coverage", {})
        details["attribute_coverage"] = attr_cov
        requirements.append(f"Min co-occurrence ≥ {min_cooc}")
        requirements.append("Sufficient product co-purchase density")
        requirements.append("Attribute columns (category/brand/size) for tree enrichment")

        if n_products < 20 or n_customers < 50 or n_transactions < 500:
            return {"status": "directional", "details": details, "requirements": requirements,
                    "reason": "Small dataset — CDT structure may be unstable"}
        else:
            return {"status": "ready", "details": details, "requirements": requirements}

    elif analysis_name in ["segmentation", "customer_segmentation"]:
        n_customers = summary.get("n_customers", 0)
        n_transactions = summary.get("n_transactions", 0)

        details["n_customers"] = n_customers
        details["n_transactions"] = n_transactions
        requirements.append("Sufficient customers for cluster stability")
        requirements.append("Multiple transactions per customer for behavioral features")

        if n_customers < 100:
            return {"status": "blocked", "details": details, "requirements": requirements,
                    "reason": "Too few customers for reliable segmentation"}
        elif n_customers < 300:
            return {"status": "directional", "details": details, "requirements": requirements,
                    "reason": "Limited customers — use fewer segments, interpret cautiously"}
        else:
            return {"status": "ready", "details": details, "requirements": requirements}

    elif analysis_name in ["switching", "product_switching"]:
        n_customers = summary.get("n_customers", 0)
        n_transactions = summary.get("n_transactions", 0)
        date_span = summary.get("date_span_days", 0)

        details["n_customers"] = n_customers
        details["n_transactions"] = n_transactions
        details["date_span_days"] = date_span
        requirements.append("Repeated purchases per customer over time window")
        requirements.append("Sufficient date span for transition observation")

        if n_customers < 50 or date_span < 30:
            return {"status": "directional", "details": details, "requirements": requirements,
                    "reason": "Limited switching signal — few repeat customers or short span"}
        else:
            return {"status": "ready", "details": details, "requirements": requirements}

    elif analysis_name in ["cohort", "cohort_analysis"]:
        n_customers = summary.get("n_customers", 0)
        date_span = summary.get("date_span_days", 0)

        details["n_customers"] = n_customers
        details["date_span_days"] = date_span
        requirements.append("Multiple cohort periods with sufficient base sizes")
        requirements.append("Date span covering several periods")

        if n_customers < 100 or date_span < 60:
            return {"status": "directional", "details": details, "requirements": requirements,
                    "reason": "Small cohort bases — retention estimates noisy"}
        else:
            return {"status": "ready", "details": details, "requirements": requirements}

    # Default
    return {"status": "ready", "details": details, "requirements": requirements,
            "reason": "General analysis — no specific readiness gate"}


def format_readiness_for_ui(readiness: Dict) -> str:
    """Format readiness dict for UI display."""
    status = readiness.get("status", "unknown")
    reason = readiness.get("reason", "")
    details = readiness.get("details", {})
    requirements = readiness.get("requirements", [])

    badge = {"ready": "✅", "directional": "⚠️", "blocked": "❌"}.get(status, "❓")
    lines = [f"**Method Readiness:** {badge} `{status.upper()}`"]

    if reason:
        lines.append(f"  *{reason}*")

    if details:
        lines.append("  **Details:**")
        for k, v in details.items():
            lines.append(f"    - {k}: {v}")

    if requirements:
        lines.append("  **Requirements:**")
        for r in requirements:
            lines.append(f"    - {r}")

    return "\n".join(lines)