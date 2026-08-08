"""Transaction data loading, normalization, and capability detection.

Pure pandas — no Streamlit imports. Every function takes a plain DataFrame
or file source and returns plain values.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.analytics.schemas import TRANSACTIONS
from src.analytics.data_quality import DataQualityReport, assess_data_quality

REQUIRED_COLUMNS = list(TRANSACTIONS.columns)

_CANONICAL_MAPPING = {
    "date": ["date", "transaction_date", "dt", "order_date"],
    "transaction_id": ["transaction_id", "txn_id", "order_id", "order_no", "basket_id", "invoice_no"],
    "stockcode": ["stockcode", "item_code", "sku", "sku_code", "product_code", "stock_code"],
    "product": ["product", "product_name", "item_name", "item_desc", "description"],
    "customer_id": ["customer_id", "cust_id", "client_id", "user_id", "buyer", "customer"],
    "price": ["price", "unit_price", "amount", "sales_price"],
    "quantity": ["quantity", "qty", "units", "units_sold", "quantity_sold"],
}


def detect_column_mapping(columns: list[str]) -> dict[str, str]:
    """Auto-detect the canonical column for each required field."""
    normalized = {c.strip().lower(): c for c in columns}
    mapping: dict[str, str] = {}
    for canonical, candidates in _CANONICAL_MAPPING.items():
        for candidate in candidates:
            if candidate in normalized:
                mapping[canonical] = normalized[candidate]
                break
    return mapping


def _clean_id(value: str) -> str:
    """Render IDs from numeric sources without a trailing '.0' (85123.0 -> 85123)."""
    value = value.strip()
    try:
        if "." in value and float(value).is_integer():
            return str(int(float(value)))
    except ValueError:
        pass
    return value


def load_transactions(
    source: str | Path | io.BytesIO,
    column_mapping: dict[str, str] | None = None,
    assess_quality: bool = True,
) -> tuple[pd.DataFrame, str, int, Optional[DataQualityReport]]:
    """Load and normalize a transaction CSV.

    Returns (df, warning_message, dropped_rows, quality_report). The returned df satisfies the
    TRANSACTIONS contract. Missing optional columns are simply absent.
    """
    if isinstance(source, (str, Path)):
        raw = pd.read_csv(source, low_memory=False)
    else:
        raw = pd.read_csv(source, low_memory=False)

    mapping = dict(column_mapping) if column_mapping else detect_column_mapping(raw.columns.tolist())
    missing = [c for c in REQUIRED_COLUMNS if c not in mapping or mapping[c] not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = pd.DataFrame({c: raw[mapping[c]] for c in REQUIRED_COLUMNS})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["customer_id"] = df["customer_id"].apply(lambda v: _clean_id(str(v)))
    df["stockcode"] = df["stockcode"].apply(lambda v: _clean_id(str(v)))
    df["transaction_id"] = df["transaction_id"].apply(lambda v: _clean_id(str(v)))

    return_mask = (df["price"] < 0) | (df["quantity"] < 0)
    return_count = int(return_mask.sum())
    return_value = float(abs((df.loc[return_mask, "price"] * df.loc[return_mask, "quantity"]).sum())) if return_count else 0.0

    before = len(df)
    valid = (
        df["date"].notna()
        & df["price"].notna()
        & df["price"].gt(0)
        & df["quantity"].notna()
        & df["quantity"].gt(0)
        & df["transaction_id"].ne("")
        & df["stockcode"].ne("")
    )
    df = df.loc[valid].copy()
    df["quantity"] = df["quantity"].astype(int)
    dropped = int(before - len(df))

    for extra in ("category", "brand", "size", "flavor", "variant", "promo_flag", "cost", "is_online", "channel"):
        if extra in raw.columns:
            df[extra] = raw.loc[df.index, extra]

    warning = ""
    if dropped > 0:
        warning = f"Removed {dropped} rows with missing/invalid data"
    if return_count > 0:
        warning = (warning + "; " if warning else "") + f"Detected {return_count} return row(s) worth {return_value:.2f} (excluded)"

    quality_report = None
    if assess_quality:
        quality_report = assess_data_quality(df)
        if quality_report.volume_warning:
            warning = (warning + "; " if warning else "") + quality_report.volume_warning

    return df.reset_index(drop=True), warning, dropped, quality_report


def build_dataset_capabilities(df: pd.DataFrame) -> dict[str, bool]:
    """Detect which optional analyses are possible given available columns."""
    return {
        "has_category": "category" in df.columns,
        "has_brand": "brand" in df.columns,
        "has_size": "size" in df.columns,
        "has_flavor": "flavor" in df.columns or "variant" in df.columns,
        "has_promo_flag": "promo_flag" in df.columns,
        "has_cost": "cost" in df.columns,
        "has_is_online": "is_online" in df.columns,
        "has_channel": "channel" in df.columns,
    }


def get_data_summary(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Basic dataset statistics as a flat dict."""
    revenue = float((df["price"] * df["quantity"]).sum())
    baskets = df["transaction_id"].nunique()
    return {
        "n_transactions": int(baskets),
        "n_line_items": int(len(df)),
        "n_customers": int(df["customer_id"].nunique()),
        "n_products": int(df["stockcode"].nunique()),
        "total_revenue": round(revenue, 2),
        "avg_basket_size": round(len(df) / baskets, 2) if baskets else 0.0,
        "avg_basket_value": round(revenue / baskets, 2) if baskets else 0.0,
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
    }


def derive_product_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """Unique product-level attributes keyed by stockcode."""
    cols = [c for c in ("stockcode", "product", "category", "brand", "size", "flavor") if c in df.columns]
    lookup = df[cols].drop_duplicates(subset="stockcode")
    lookup["product"] = lookup["product"].fillna(lookup["stockcode"])
    for col in ("category", "brand", "size", "flavor"):
        if col in lookup.columns:
            lookup[col] = lookup[col].fillna("Unknown")
    return lookup.reset_index(drop=True)


def revenue_column(df: pd.DataFrame) -> pd.Series:
    """Line revenue as a Series (price * quantity)."""
    return df["price"] * df["quantity"]


def is_positive_price_series(series: pd.Series) -> bool:
    """True if series has no NaN and all values > 0."""
    return bool(series.notna().all() and series.gt(0).all() and len(series) > 0)


def safe_divide(
    numerator: float | np.ndarray, denominator: float | np.ndarray
) -> float | np.ndarray:
    """Division that yields 0.0 where the denominator is zero."""
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator != 0)
    return result


def assign_basket_mission(
    df: pd.DataFrame,
    product_col: str = "stockcode",
    labels: tuple[str, ...] = ("Top-Up", "Regular", "Stock-Up"),
) -> pd.DataFrame:
    """Assign basket-size mission (Top-Up/Regular/Stock-Up) to each transaction.
    
    Based on the mean basket depth of each product. Products that tend to appear
    in small baskets are "Top-Up", large baskets are "Stock-Up".
    
    Args:
        df: Transaction dataframe
        product_col: Column with product codes
        labels: Labels for the three tiers
        
    Returns:
        DataFrame with added 'basket_mission' column per transaction
    """
    from src.analytics.cdt.attributes import derive_basket_size_affinity
    
    product_mission = derive_basket_size_affinity(df, product_col=product_col, labels=labels)
    # For each transaction, assign the mission of its products
    # If multiple products, use the mode (most common)
    df = df.copy()
    df["product_mission"] = df[product_col].map(product_mission)
    # Aggregate to transaction level
    txn_mission = df.groupby("transaction_id")["product_mission"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else labels[1]
    ).rename("basket_mission")
    return df.merge(txn_mission, on="transaction_id", how="left")


def add_segment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add customer segment columns from available segmentations.
    
    Tries RFM segment first (cheap quantile-based), then behavioral segment.
    
    Args:
        df: Transaction dataframe with customer_id
        
    Returns:
        DataFrame with added segment columns if available
    """
    df = df.copy()
    segments_added = []
    
    # Try RFM segment (cheap quantile-based)
    try:
        from src.analytics.segmentation import rfm_segmentation, compute_rfm_features
        rfm_features = compute_rfm_features(df)
        rfm_seg = rfm_segmentation(rfm_features)
        if "segment" in rfm_seg.columns:
            seg_map = rfm_seg.set_index("customer_id")["segment"].to_dict()
            df["rfm_segment"] = df["customer_id"].map(seg_map)
            segments_added.append("rfm_segment")
    except Exception:
        pass
    
    # Try behavioral segment
    try:
        from src.analytics.segmentation import behavioral_segmentation
        beh_seg = behavioral_segmentation(df)
        if "segment" in beh_seg.columns:
            seg_map = beh_seg.set_index("customer_id")["segment"].to_dict()
            df["behavioral_segment"] = df["customer_id"].map(seg_map)
            segments_added.append("behavioral_segment")
    except Exception:
        pass
    
    return df