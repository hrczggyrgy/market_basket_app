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

from src.analytics.data_quality import DataQualityReport, assess_data_quality
from src.analytics.schemas import TRANSACTIONS

REQUIRED_COLUMNS = list(TRANSACTIONS.columns)

_CANONICAL_MAPPING = {
    "date": ["date", "transaction_date", "dt", "order_date"],
    "transaction_id": [
        "transaction_id",
        "txn_id",
        "order_id",
        "order_no",
        "basket_id",
        "invoice_no",
    ],
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


def _clean_id(value: str | float | None) -> str | pd.NA:
    """Render IDs from numeric sources without a trailing '.0' (85123.0 -> 85123).

    Returns pd.NA for missing/NaN values instead of the string "nan".
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NA
    value = str(value).strip()
    if value == "" or value.lower() == "nan":
        return pd.NA
    try:
        if "." in value and float(value).is_integer():
            return str(int(float(value)))
    except ValueError:
        pass
    return value


def _is_valid_id(series: pd.Series) -> pd.Series:
    """Check if IDs are valid (not pd.NA and not empty string)."""
    return series.notna() & series.ne("")


def load_transactions(
    source: str | Path | io.BytesIO,
    column_mapping: dict[str, str] | None = None,
    assess_quality: bool = True,
    require_customer_id: bool = True,
) -> tuple[pd.DataFrame, str, int, Optional[DataQualityReport]]:
    """Load and normalize a transaction CSV.

    Returns (df, warning_message, dropped_rows, quality_report). The returned df satisfies the
    TRANSACTIONS contract. Missing optional columns are simply absent.

    Enhanced with consistent data quality validation and error handling.

    Args:
        source: Path, URL, or BytesIO object containing CSV data.
        column_mapping: Optional mapping of canonical column names to source columns.
        assess_quality: Whether to run data quality assessment (default: True).
        require_customer_id: If True (default), filter out rows with missing/invalid customer_id.
            If False, keep all rows regardless of customer_id validity. This is useful for
            general transaction analyses that don't require customer-level data.
    """
    from src.analytics.config import get_config
    from src.analytics.data_quality import DataQualityError

    config = get_config()

    if isinstance(source, (str, Path)):
        raw = pd.read_csv(source, low_memory=False)
    else:
        raw = pd.read_csv(source, low_memory=False)

    mapping = (
        dict(column_mapping) if column_mapping else detect_column_mapping(raw.columns.tolist())
    )
    missing = [c for c in REQUIRED_COLUMNS if c not in mapping or mapping[c] not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = pd.DataFrame({c: raw[mapping[c]] for c in REQUIRED_COLUMNS})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["customer_id"] = df["customer_id"].apply(_clean_id)
    df["stockcode"] = df["stockcode"].apply(_clean_id)
    df["transaction_id"] = df["transaction_id"].apply(_clean_id)

    return_mask = (df["price"] < 0) | (df["quantity"] < 0)
    return_count = int(return_mask.sum())
    return_value = (
        float(abs((df.loc[return_mask, "price"] * df.loc[return_mask, "quantity"]).sum()))
        if return_count
        else 0.0
    )

    before = len(df)
    # Transaction-level validity (required for all analyses)
    transaction_valid = (
        df["date"].notna()
        & df["price"].notna()
        & df["price"].gt(0)
        & df["quantity"].notna()
        & df["quantity"].gt(0)
        & df["transaction_id"].notna()
        & df["transaction_id"].ne("")
        & df["stockcode"].notna()
        & df["stockcode"].ne("")
    )
    # Customer-level validity (required for customer analytics when require_customer_id=True)
    customer_valid = df["customer_id"].notna() & df["customer_id"].ne("")

    # Apply filters based on require_customer_id parameter
    if require_customer_id:
        valid_mask = transaction_valid & customer_valid
    else:
        valid_mask = transaction_valid

    df = df.loc[valid_mask].copy()
    # Preserve fractional quantities (e.g., weighted goods)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    dropped = int(before - len(df))

    # Optional columns: auto-detect via same canonical mapping (case-insensitive)
    optional_candidates = {
        "category": ["category", "cat", "product_category", "category_name"],
        "brand": ["brand", "brand_name", "manufacturer"],
        "size": ["size", "package_size", "unit_size"],
        "flavor": ["flavor", "flavour", "variant", "flavor_name"],
        "variant": ["variant", "variant_name", "flavor"],
        "promo_flag": ["promo_flag", "promo", "is_promo", "on_promotion", "promotion"],
        "cost": ["cost", "unit_cost", "cost_price", "wholesale_price"],
        "is_online": ["is_online", "online", "channel_online", "ecommerce"],
        "channel": ["channel", "sales_channel", "channel_name", "store_type"],
    }
    # Use the same canonical mapping normalization for optional columns
    raw_cols_normalized = {c.strip().lower(): c for c in raw.columns}
    for canonical, candidates in optional_candidates.items():
        for candidate in candidates:
            candidate_norm = candidate.strip().lower()
            if candidate_norm in raw_cols_normalized:
                src_col = raw_cols_normalized[candidate_norm]
                if src_col in raw.columns:
                    df[canonical] = raw.loc[df.index, src_col]
                break

    warning = ""
    if dropped > 0:
        warning = f"Removed {dropped} rows with missing/invalid data"
    if return_count > 0:
        warning = (
            warning + "; " if warning else ""
        ) + f"Detected {return_count} return row(s) worth {return_value:.2f} (excluded)"

    quality_report = None
    if assess_quality:
        quality_report = assess_data_quality(df)
        if quality_report.volume_warning:
            warning = (warning + "; " if warning else "") + quality_report.volume_warning

        # Consistent quality gate enforcement
        if getattr(config, "fail_on_quality_issues", False) and quality_report.has_issues():
            raise DataQualityError(quality_report)

    return df.reset_index(drop=True), warning, dropped, quality_report


def build_dataset_capabilities(df: pd.DataFrame) -> dict[str, bool]:
    """Detect which optional analyses are possible given available columns and data quality.

    Note on has_price_variation: This is an aggregate measure computed as the maximum
    coefficient of variation (CV) across all SKUs. It returns True if ANY SKU has
    price variation >= 5% CV, even if most SKUs have zero variation.
    """
    # Price variation check (aggregate: max CV across all SKUs)
    price_cv = 0.0
    if "price" in df.columns and "stockcode" in df.columns:
        price_stats = df.groupby("stockcode")["price"].agg(["mean", "std"]).reset_index()
        price_stats["cv"] = price_stats["std"] / price_stats["mean"].replace(0, pd.NA)
        price_cv = price_stats["cv"].max() if not price_stats["cv"].isna().all() else 0.0

    # Distinct price points per SKU
    min_distinct_prices = 0
    if "price" in df.columns and "stockcode" in df.columns:
        distinct_prices = df.groupby("stockcode")["price"].nunique()
        min_distinct_prices = int(distinct_prices.min()) if len(distinct_prices) > 0 else 0

    n_customers = df["customer_id"].nunique() if "customer_id" in df.columns else 0
    n_skus = df["stockcode"].nunique() if "stockcode" in df.columns else 0
    n_baskets = df["transaction_id"].nunique() if "transaction_id" in df.columns else 0

    return {
        # Column-based capabilities
        "has_category": "category" in df.columns,
        "has_brand": "brand" in df.columns,
        "has_size": "size" in df.columns,
        "has_flavor": "flavor" in df.columns or "variant" in df.columns,
        "has_promo_flag": "promo_flag" in df.columns,
        "has_cost": "cost" in df.columns,
        "has_is_online": "is_online" in df.columns,
        "has_channel": "channel" in df.columns,
        # Data quality / volume capabilities
        "has_price_variation": price_cv >= 0.05,  # 5% min CV
        "min_distinct_prices_3": min_distinct_prices >= 3,
        "sufficient_customers_100": n_customers >= 100,
        "sufficient_customers_500": n_customers >= 500,
        "sufficient_skus_20": n_skus >= 20,
        "sufficient_skus_50": n_skus >= 50,
        "sufficient_baskets_200": n_baskets >= 200,
        "sufficient_baskets_500": n_baskets >= 500,
        "sufficient_baskets_1000": n_baskets >= 1000,
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
    cols = [
        c
        for c in ("stockcode", "product", "category", "brand", "size", "flavor")
        if c in df.columns
    ]
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
    """Division that yields 0.0 where the denominator is zero.

    Enhanced with numerical stability checks and warnings for edge cases.
    """
    import warnings as _warnings

    denominator = np.asarray(denominator, dtype=float)
    numerator = np.asarray(numerator, dtype=float)

    # Check for near-zero denominators
    near_zero_mask = np.abs(denominator) < 1e-10
    if np.any(near_zero_mask):
        _warnings.warn(
            f"Division by near-zero values detected in {np.sum(near_zero_mask)} cases. "
            "Results may be numerically unstable.",
            UserWarning,
            stacklevel=2,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(
            numerator, denominator, out=np.zeros_like(denominator), where=denominator != 0
        )
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
    txn_mission = (
        df.groupby("transaction_id")["product_mission"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else labels[1])
        .rename("basket_mission")
    )
    return df.merge(txn_mission, on="transaction_id", how="left")


def add_segment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add customer segment columns from available segmentations.

    Tries RFM segment first (cheap quantile-based), then behavioral segment.

    Args:
        df: Transaction dataframe with customer_id

    Returns:
        DataFrame with added segment columns if available
    """
    import inspect
    import logging

    logger = logging.getLogger(__name__)

    df = df.copy()
    segments_added = []

    # Try RFM segment (cheap quantile-based)
    try:
        from src.analytics.segmentation import compute_rfm_features, rfm_segmentation

        rfm_features = compute_rfm_features(df)
        rfm_seg = rfm_segmentation(rfm_features)
        if "segment" in rfm_seg.columns:
            seg_map = rfm_seg.set_index("customer_id")["segment"].to_dict()
            df["rfm_segment"] = df["customer_id"].map(seg_map)
            segments_added.append("rfm_segment")
    except Exception as e:
        logger.warning(f"RFM segmentation failed: {e}")

    # Try behavioral segment
    try:
        from src.analytics.segmentation import behavioral_segmentation

        # Runtime safety check: verify behavioral_segmentation supports return_metrics parameter
        sig = inspect.signature(behavioral_segmentation)
        supports_return_metrics = "return_metrics" in sig.parameters

        if supports_return_metrics:
            beh_seg = behavioral_segmentation(df, return_metrics=False)
        else:
            beh_seg = behavioral_segmentation(df)

        if isinstance(beh_seg, tuple):
            beh_seg = beh_seg[0]
        if "segment" in beh_seg.columns:
            seg_map = beh_seg.set_index("customer_id")["segment"].to_dict()
            df["behavioral_segment"] = df["customer_id"].map(seg_map)
            segments_added.append("behavioral_segment")
    except Exception as e:
        logger.warning(f"Behavioral segmentation failed: {e}")

    if not segments_added:
        logger.info("No segment columns could be added")

    return df


def test_behavioral_segmentation_return_metrics() -> None:
    """Test that behavioral_segmentation supports return_metrics parameter and returns correct type."""
    import inspect
    from src.analytics.segmentation import behavioral_segmentation
    import pandas as pd
    import numpy as np

    # Verify signature has return_metrics parameter
    sig = inspect.signature(behavioral_segmentation)
    assert "return_metrics" in sig.parameters, "behavioral_segmentation missing return_metrics parameter"

    # Create minimal test data
    test_df = pd.DataFrame({
        "customer_id": ["C1", "C1", "C2", "C2", "C3", "C3"],
        "transaction_id": ["T1", "T2", "T3", "T4", "T5", "T6"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03", "2024-01-01", "2024-01-04"]),
        "stockcode": ["S1", "S2", "S1", "S3", "S2", "S1"],
        "product": ["P1", "P2", "P1", "P3", "P2", "P1"],
        "price": [10.0, 20.0, 10.0, 30.0, 20.0, 10.0],
        "quantity": [1, 2, 1, 1, 3, 1],
    })

    # Test with return_metrics=False (should return DataFrame)
    result = behavioral_segmentation(test_df, return_metrics=False)
    assert isinstance(result, pd.DataFrame), f"Expected DataFrame, got {type(result)}"
    assert "segment" in result.columns, "Result missing 'segment' column"
    assert "customer_id" in result.columns, "Result missing 'customer_id' column"

    # Test with return_metrics=True (should return tuple of DataFrame, dict)
    result_with_metrics = behavioral_segmentation(test_df, return_metrics=True)
    assert isinstance(result_with_metrics, tuple), f"Expected tuple, got {type(result_with_metrics)}"
    assert len(result_with_metrics) == 2, f"Expected tuple of length 2, got {len(result_with_metrics)}"
    assert isinstance(result_with_metrics[0], pd.DataFrame), "First element should be DataFrame"
    assert isinstance(result_with_metrics[1], dict), "Second element should be dict"

    print("test_behavioral_segmentation_return_metrics passed")


def test_load_transactions_customer_id_control() -> None:
    """Test that require_customer_id parameter controls customer_id filtering."""
    import tempfile
    import pandas as pd

    csv_data = """date,transaction_id,stockcode,product,customer_id,price,quantity
2024-01-01,TXN001,SKU001,Product A,CUST001,10.0,1
2024-01-01,TXN002,SKU002,Product B,,15.0,2
2024-01-01,TXN003,SKU003,Product C,CUST003,20.0,1
2024-01-01,TXN004,SKU004,Product D,,25.0,3
2024-01-01,TXN005,SKU005,Product E,CUST005,30.0,1
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_data)
        temp_path = f.name

    try:
        df_with_cust, _, dropped_with, _ = load_transactions(temp_path, require_customer_id=True)
        assert len(df_with_cust) == 3, f"Expected 3 rows with customer_id, got {len(df_with_cust)}"
        assert dropped_with == 2, f"Expected 2 dropped rows, got {dropped_with}"
        assert df_with_cust["customer_id"].notna().all()

        df_without_cust, _, dropped_without, _ = load_transactions(temp_path, require_customer_id=False)
        assert len(df_without_cust) == 5, f"Expected 5 rows without customer_id filter, got {len(df_without_cust)}"
        assert dropped_without == 0, f"Expected 0 dropped rows, got {dropped_without}"
        assert df_without_cust["customer_id"].isna().sum() == 2

    finally:
        import os
        os.unlink(temp_path)

    print("test_load_transactions_customer_id_control passed")
