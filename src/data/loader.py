"""Data loading and validation."""

from typing import Optional

import pandas as pd
import streamlit as st

REQUIRED_COLUMNS = [
    "date",
    "transaction_id",
    "stockcode",
    "product",
    "customer_id",
    "price",
    "quantity",
]


def load_transactions(file, column_mapping=None, **kwargs) -> pd.DataFrame:
    """
    Load and validate transaction CSV.

    Args:
        file: Uploaded file object or path
        column_mapping: Optional dict mapping standard names to actual column names
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        Validated DataFrame

    Raises:
        ValueError: If required columns missing or data invalid
    """
    df = pd.read_csv(file, **kwargs)

    # Apply column mapping if provided
    if column_mapping:
        # Reverse mapping: actual_col -> standard_col
        reverse_mapping = {v: k for k, v in column_mapping.items() if v in df.columns}
        df = df.rename(columns=reverse_mapping)

    return validate_and_clean(df)


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and clean data."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Convert date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Convert numeric columns
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    # Convert IDs to string
    df["transaction_id"] = df["transaction_id"].astype(str)
    df["stockcode"] = df["stockcode"].astype(str)
    df["customer_id"] = df["customer_id"].astype(str)
    df["product"] = df["product"].astype(str)

    # Drop rows with invalid critical data
    initial_len = len(df)
    df = df.dropna(
        subset=[
            "date",
            "transaction_id",
            "stockcode",
            "customer_id",
            "price",
            "quantity",
        ]
    )

    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} rows with missing/invalid data")

    # BUG 4 FIX: Warn about dropped returns/refunds (negative/zero prices and quantities)
    before_filter = len(df)
    df = df[(df["price"] > 0) & (df["quantity"] > 0)]
    dropped_returns = before_filter - len(df)
    if dropped_returns > 0:
        st.warning(
            f"Dropped {dropped_returns} rows with zero/negative price or quantity "
            "(likely returns/refunds). This may affect revenue and retention calculations."
        )

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """Get summary statistics of transaction data."""
    # BUG 7 FIX: Vectorized approach instead of groupby().apply()
    df = df.copy()
    df["revenue"] = df["price"] * df["quantity"]
    basket_revenue = df.groupby("transaction_id")["revenue"].sum()
    basket_size = df.groupby("transaction_id")["quantity"].sum()

    return {
        "n_transactions": df["transaction_id"].nunique(),
        "n_customers": df["customer_id"].nunique(),
        "n_products": df["stockcode"].nunique(),
        "date_range": (df["date"].min(), df["date"].max()),
        "total_revenue": df["revenue"].sum(),
        "avg_basket_size": basket_size.mean(),
        "avg_basket_value": basket_revenue.mean(),
    }


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Filter transactions by date range."""
    filtered = df.copy()
    if start_date:
        filtered = filtered[filtered["date"] >= start_date]
    if end_date:
        filtered = filtered[filtered["date"] <= end_date]
    return filtered


def filter_top_products(df: pd.DataFrame, n: int = 100, by: str = "frequency") -> pd.DataFrame:
    """Filter to top N products by frequency or revenue."""
    if by == "frequency":
        top_products = df["stockcode"].value_counts().head(n).index
    elif by == "revenue":
        df = df.copy()
        df["revenue"] = df["price"] * df["quantity"]
        revenue = df.groupby("stockcode")["revenue"].sum()
        top_products = revenue.nlargest(n).index
    else:
        raise ValueError("by must be 'frequency' or 'revenue'")

    return df[df["stockcode"].isin(top_products)]


def get_customer_product_matrix(df: pd.DataFrame, min_transactions: int = 1) -> pd.DataFrame:
    """
    Create customer x product matrix (one-hot encoded).

    Returns:
        DataFrame with customers as rows, products as columns, 1/0 values
    """
    # Filter customers with minimum transactions
    cust_counts = df["customer_id"].value_counts()
    valid_customers = cust_counts[cust_counts >= min_transactions].index
    df = df[df["customer_id"].isin(valid_customers)]

    # Create matrix
    matrix = pd.crosstab(df["customer_id"], df["stockcode"])
    matrix = (matrix > 0).astype(int)

    return matrix


def add_rfm_features(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Add RFM (Recency, Frequency, Monetary) features per customer.

    Args:
        df: Transaction DataFrame
        snapshot_date: Date to calculate recency from (default: max date in data)

    Returns:
        DataFrame with RFM features per customer
    """
    if snapshot_date is None:
        snapshot_date = df["date"].max() + pd.Timedelta(1, unit="D")

    # BUG 3 FIX: Compute revenue column first instead of using broken lambda
    df_with_revenue = df.copy()
    df_with_revenue["revenue"] = df_with_revenue["price"] * df_with_revenue["quantity"]

    rfm = (
        df_with_revenue.groupby("customer_id")
        .agg(
            recency=("date", lambda x: (snapshot_date - x.max()).days),
            frequency=("transaction_id", "nunique"),
            monetary=("revenue", "sum"),  # Now uses precomputed revenue column
        )
        .reset_index()
    )

    # Add quartile-based scores
    for col in ["recency", "frequency", "monetary"]:
        if col == "recency":
            # Lower recency is better
            rfm[f"{col}_score"] = pd.qcut(rfm[col], 4, labels=[4, 3, 2, 1], duplicates="drop")
        else:
            rfm[f"{col}_score"] = pd.qcut(rfm[col], 4, labels=[1, 2, 3, 4], duplicates="drop")

    rfm["rfm_score"] = (
        rfm["recency_score"].astype(str)
        + rfm["frequency_score"].astype(str)
        + rfm["monetary_score"].astype(str)
    )

    return rfm
