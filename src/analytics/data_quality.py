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
    
    # Incomplete rows (missing required fields)
    incomplete_rows: int = 0
    incomplete_row_details: Dict[str, int] = field(default_factory=dict)
    
    # Duplicate transactions (exact duplicate rows)
    duplicate_count: int = 0
    
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
            "incomplete_rows": self.incomplete_rows,
            "incomplete_row_details": self.incomplete_row_details,
            "duplicate_count": self.duplicate_count,
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
            incomplete_rows=d.get("incomplete_rows", 0),
            incomplete_row_details=d.get("incomplete_row_details", {}),
            duplicate_count=d.get("duplicate_count", 0),
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
    
    # 3. Check for incomplete rows (missing required fields)
    required_cols = ["transaction_id", "stockcode", "customer_id", "date", "price", "quantity"]
    available_required = [c for c in required_cols if c in df.columns]
    if available_required:
        missing_per_col = df[available_required].isna().sum()
        report.incomplete_row_details = missing_per_col[missing_per_col > 0].to_dict()
        # Count rows with ANY missing required field
        incomplete_mask = df[available_required].isna().any(axis=1)
        report.incomplete_rows = int(incomplete_mask.sum())
    
    # 4. Check for duplicate transactions (exact duplicate rows)
    report.duplicate_count = int(df.duplicated().sum())
    
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


def compute_quality_score(report: DataQualityReport) -> float:
    """Compute a 0-1 quality score from a DataQualityReport.
    
    Score components:
    - Volume adequacy: 30% weight
    - Low-frequency products: 25% weight  
    - Basket outliers: 15% weight
    - Incomplete rows: 20% weight
    - Coverage: 10% weight
    
    Returns:
        Float in [0, 1] where 1 = perfect quality
    """
    score = 1.0
    
    # Volume adequacy (30%)
    if report.volume_warning:
        # Extract actual vs recommended
        try:
            parts = report.volume_warning.split(": ")
            actual_str = parts[1].split(" transactions")[0]
            actual = int(actual_str.replace(",", ""))
            recommended_str = parts[2].split(" for")[0]
            recommended = int(recommended_str.replace(",", ""))
            volume_ratio = min(1.0, actual / recommended)
            score -= 0.30 * (1.0 - volume_ratio)
        except (ValueError, IndexError, AttributeError):
            score -= 0.15  # Default penalty if parsing fails
    
    # Low-frequency products (25%)
    n_low_freq = len(report.low_freq_products)
    n_total = max(1, report.n_products)
    low_freq_ratio = n_low_freq / n_total
    score -= 0.25 * min(1.0, low_freq_ratio * 2)  # Cap at 2x penalty
    
    # Basket outliers (15%)
    n_outliers = len(report.basket_outlier_txn_ids)
    n_transactions = max(1, report.n_transactions)
    outlier_ratio = n_outliers / n_transactions
    score -= 0.15 * min(1.0, outlier_ratio * 10)  # Cap penalty
    
    # Incomplete rows (20%)
    incomplete_ratio = report.incomplete_rows / max(1, report.n_transactions)
    score -= 0.20 * min(1.0, incomplete_ratio * 100)
    
    # Coverage bonus (10% - if we have category/brand/etc)
    coverage_bonus = 0.0
    # This would be set by the caller if they have optional columns
    
    return max(0.0, min(1.0, score + coverage_bonus))


def attach_quality_metadata(
    df: pd.DataFrame, 
    quality_score: float,
    quality_report: Optional[DataQualityReport] = None
) -> pd.DataFrame:
    """Attach quality metadata to a DataFrame output.
    
    Adds a '_quality' column with JSON-serialized quality info.
    """
    import json
    result = df.copy()
    meta = {
        "quality_score": round(quality_score, 3),
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    if quality_report:
        meta["report"] = quality_report.to_dict()
    result["_quality"] = [json.dumps(meta)] * len(result)
    return result


def get_quality_score(df: pd.DataFrame) -> Optional[float]:
    """Extract quality score from a DataFrame with _quality column."""
    if "_quality" not in df.columns or len(df) == 0:
        return None
    try:
        import json
        meta = json.loads(df["_quality"].iloc[0])
        return meta.get("quality_score")
    except (json.JSONDecodeError, KeyError, IndexError):
        return None