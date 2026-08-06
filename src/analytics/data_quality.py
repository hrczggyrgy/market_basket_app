"""Data quality checks and reporting for transaction data.

Provides configurable checks for common data quality issues in retail transaction
data, with actionable reports for users to review before analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set

import numpy as np
import pandas as pd

from src.analytics.config import get_config


class DataQualityError(Exception):
    """Raised when fail_on_quality_issues is enabled and data quality gates fail.

    Carries the DataQualityReport so callers can render the findings even in
    hard-fail mode (e.g. a CI regression run).
    """

    def __init__(self, report: "DataQualityReport") -> None:
        self.report = report
        super().__init__(generate_quality_summary(report))


@dataclass
class DataQualityReport:
    """Report of data quality issues found in transaction data."""
    
    # Products with fewer than min_transactions threshold
    low_freq_products: List[str] = field(default_factory=list)
    low_freq_counts: Dict[str, int] = field(default_factory=dict)
    
    # Basket size outliers (above percentile threshold)
    basket_outlier_txn_ids: List[str] = field(default_factory=list)
    basket_size_percentile: float = 0.99
    basket_outlier_threshold: int = 0
    
    # Duplicate transactions
    duplicate_txn_ids: List[str] = field(default_factory=list)
    duplicate_count: int = 0
    
    # Incomplete rows (missing required fields)
    incomplete_rows: int = 0
    incomplete_row_details: Dict[str, int] = field(default_factory=dict)
    
    # Volume warning
    volume_warning: Optional[str] = None
    n_transactions: int = 0
    n_products: int = 0
    
    # User-selected exclusions (set via UI)
    excluded_products: List[str] = field(default_factory=list)
    excluded_txn_ids: List[str] = field(default_factory=list)
    
    def has_issues(self) -> bool:
        """Return True if any quality issues were found."""
        return any([
            len(self.low_freq_products) > 0,
            len(self.basket_outlier_txn_ids) > 0,
            self.duplicate_count > 0,
            self.incomplete_rows > 0,
            self.volume_warning is not None,
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "low_freq_products": self.low_freq_products,
            "low_freq_counts": self.low_freq_counts,
            "basket_outlier_txn_ids": self.basket_outlier_txn_ids,
            "basket_size_percentile": self.basket_size_percentile,
            "basket_outlier_threshold": self.basket_outlier_threshold,
            "duplicate_txn_ids": self.duplicate_txn_ids,
            "duplicate_count": self.duplicate_count,
            "incomplete_rows": self.incomplete_rows,
            "incomplete_row_details": self.incomplete_row_details,
            "volume_warning": self.volume_warning,
            "n_transactions": self.n_transactions,
            "n_products": self.n_products,
            "excluded_products": self.excluded_products,
            "excluded_txn_ids": self.excluded_txn_ids,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataQualityReport":
        """Create from dictionary."""
        return cls(
            low_freq_products=d.get("low_freq_products", []),
            low_freq_counts=d.get("low_freq_counts", {}),
            basket_outlier_txn_ids=d.get("basket_outlier_txn_ids", []),
            basket_size_percentile=d.get("basket_size_percentile", 0.99),
            basket_outlier_threshold=d.get("basket_outlier_threshold", 0),
            duplicate_txn_ids=d.get("duplicate_txn_ids", []),
            duplicate_count=d.get("duplicate_count", 0),
            incomplete_rows=d.get("incomplete_rows", 0),
            incomplete_row_details=d.get("incomplete_row_details", {}),
            volume_warning=d.get("volume_warning"),
            n_transactions=d.get("n_transactions", 0),
            n_products=d.get("n_products", 0),
            excluded_products=d.get("excluded_products", []),
            excluded_txn_ids=d.get("excluded_txn_ids", []),
        )


def assess_data_quality(
    df: pd.DataFrame,
    min_product_transactions: int = 50,
    basket_outlier_percentile: float = 0.99,
    min_viable_transactions: Optional[Dict[int, int]] = None,
) -> DataQualityReport:
    """Assess data quality of transaction DataFrame.
    
    Args:
        df: Transaction DataFrame with columns: transaction_id, stockcode, 
            customer_id, date, price, quantity
        min_product_transactions: Minimum transactions per product to not be flagged
        basket_outlier_percentile: Percentile threshold for basket size outliers
        min_viable_transactions: Dict mapping n_products thresholds to min transactions
            e.g., {200: 2000, 1000: 5000, float('inf'): 10000}
    
    Returns:
        DataQualityReport with all findings
    """
    config = get_config()
    
    # Use config defaults if not provided
    if min_product_transactions is None:
        min_product_transactions = getattr(config, "min_product_transactions", 50)
    if basket_outlier_percentile is None:
        basket_outlier_percentile = getattr(config, "basket_outlier_percentile", 0.99)
    if min_viable_transactions is None:
        min_viable_transactions = getattr(config, "min_viable_transactions", {
            200: 2000,
            1000: 5000,
            float('inf'): 10000,
        })
    
    report = DataQualityReport()
    report.n_transactions = df["transaction_id"].nunique()
    report.n_products = df["stockcode"].nunique()
    
    # 1. Check for low-frequency products
    product_txn_counts = df.groupby("stockcode")["transaction_id"].nunique()
    low_freq_mask = product_txn_counts < min_product_transactions
    if low_freq_mask.any():
        low_freq_products = product_txn_counts[low_freq_mask]
        report.low_freq_products = low_freq_products.index.tolist()
        report.low_freq_counts = low_freq_products.to_dict()
    
    # 2. Check for basket size outliers
    basket_sizes = df.groupby("transaction_id")["stockcode"].nunique()
    if len(basket_sizes) > 0:
        threshold = basket_sizes.quantile(basket_outlier_percentile)
        report.basket_size_percentile = basket_outlier_percentile
        report.basket_outlier_threshold = int(np.ceil(threshold))
        outliers = basket_sizes[basket_sizes > threshold]
        report.basket_outlier_txn_ids = outliers.index.tolist()
    
    # 3. Check for duplicate transactions (same transaction_id + stockcode + customer_id + date + price + qty)
    # We consider a row duplicate if all required columns match
    dup_cols = ["transaction_id", "stockcode", "customer_id", "date", "price", "quantity"]
    available_dup_cols = [c for c in dup_cols if c in df.columns]
    if len(available_dup_cols) >= 3:  # Need at least transaction_id + stockcode + one more
        dup_mask = df.duplicated(subset=available_dup_cols, keep="first")
        report.duplicate_count = int(dup_mask.sum())
        if report.duplicate_count > 0:
            report.duplicate_txn_ids = df.loc[dup_mask, "transaction_id"].tolist()
    
    # 4. Check for incomplete rows (missing required fields)
    required_cols = ["transaction_id", "stockcode", "customer_id", "date", "price", "quantity"]
    available_required = [c for c in required_cols if c in df.columns]
    if available_required:
        missing_per_col = df[available_required].isna().sum()
        report.incomplete_row_details = missing_per_col[missing_per_col > 0].to_dict()
        # Count rows with ANY missing required field
        incomplete_mask = df[available_required].isna().any(axis=1)
        report.incomplete_rows = int(incomplete_mask.sum())
    
    # 5. Volume warning based on catalog size
    n_skus = report.n_products
    min_txns = None
    for threshold, min_t in sorted(min_viable_transactions.items()):
        if n_skus <= threshold:
            min_txns = min_t
            break
    if min_txns is not None and report.n_transactions < min_txns:
        report.volume_warning = (
            f"Low transaction volume: {report.n_transactions:,} transactions for "
            f"{n_skus:,} SKUs. Minimum recommended: {min_txns:,} for catalog of this size. "
            f"Results may be unreliable."
        )

    if getattr(config, "fail_on_quality_issues", False) and report.has_issues():
        raise DataQualityError(report)

    return report


def filter_data_by_quality(
    df: pd.DataFrame,
    report: DataQualityReport,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Filter DataFrame based on quality report exclusions.
    
    Args:
        df: Original transaction DataFrame
        report: DataQualityReport with excluded_products and excluded_txn_ids set
        
    Returns:
        Tuple of (filtered_df, updated_report)
    """
    filtered = df.copy()
    
    # Exclude low-frequency products if user selected
    if report.excluded_products:
        filtered = filtered[~filtered["stockcode"].isin(report.excluded_products)]
    
    # Exclude basket outliers if user selected
    if report.excluded_txn_ids:
        filtered = filtered[~filtered["transaction_id"].isin(report.excluded_txn_ids)]
    
    # Update report with filtered stats
    report.n_transactions = filtered["transaction_id"].nunique()
    report.n_products = filtered["stockcode"].nunique()
    
    return filtered, report


def generate_quality_summary(report: DataQualityReport) -> str:
    """Generate human-readable summary of data quality issues."""
    lines = []
    
    if report.low_freq_products:
        lines.append(
            f"⚠️ **Low-frequency products**: {len(report.low_freq_products)} products "
            f"appear in fewer than 50 transactions (configurable). "
            f"Consider excluding from mining."
        )
    
    if report.basket_outlier_txn_ids:
        lines.append(
            f"⚠️ **Basket size outliers**: {len(report.basket_outlier_txn_ids)} transactions "
            f"exceed the {report.basket_size_percentile:.0%} percentile "
            f"(threshold: {report.basket_outlier_threshold} items/basket)."
        )
    
    if report.duplicate_count > 0:
        lines.append(
            f"⚠️ **Duplicate transactions**: {report.duplicate_count} duplicate rows detected "
            f"(same transaction_id + stockcode + customer_id + date + price + qty)."
        )
    
    if report.incomplete_rows > 0:
        details = ", ".join(f"{col}: {cnt}" for col, cnt in report.incomplete_row_details.items())
        lines.append(
            f"⚠️ **Incomplete rows**: {report.incomplete_rows} rows with missing required fields "
            f"({details}). These were dropped during loading."
        )
    
    if report.volume_warning:
        lines.append(f"⚠️ **Volume warning**: {report.volume_warning}")
    
    if not lines:
        lines.append("✅ No data quality issues detected.")
    
    return "\n\n".join(lines)