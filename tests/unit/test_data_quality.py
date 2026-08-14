"""Tests for data quality assessment."""

from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.data_quality import (
    DataQualityReport,
    assess_data_quality,
    filter_data_by_quality,
    generate_quality_summary,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    from src.analytics.data import load_transactions

    df, _, _, _ = load_transactions("sample_data/sample_transactions.csv")
    return df


def test_assess_data_quality_basic(sample_df: pd.DataFrame) -> None:
    report = assess_data_quality(sample_df)
    assert isinstance(report, DataQualityReport)
    assert report.n_transactions > 0
    assert report.n_products > 0


def test_low_freq_products_detected() -> None:
    # Create a fixture with a product that appears very few times
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=100, freq="D"),
            "transaction_id": [f"T{i}" for i in range(100)],
            "stockcode": ["A"] * 95 + ["B"] * 5,  # B appears only 5 times
            "product": ["Product A"] * 95 + ["Product B"] * 5,
            "customer_id": [f"C{i % 10}" for i in range(100)],
            "price": [10.0] * 100,
            "quantity": [1] * 100,
        }
    )
    report = assess_data_quality(df, min_product_transactions=10)
    assert "B" in report.low_freq_products
    assert report.low_freq_counts["B"] == 5


def test_basket_outliers_detected() -> None:
    # Create data with many small baskets and one very large basket
    # 20 baskets with 2 items each, 1 basket with 10 items
    dates = ["2024-01-01"] * 21
    txn_ids = [f"T{i}" for i in range(21)]
    # First 20 baskets have 2 items each (SKU A and B)
    # Last basket has 10 items (SKU A, B, C, D, E, F, G, H, I, J)
    stockcodes = [("A", "B")] * 20 + [("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")]

    rows = []
    for i in range(21):
        for sku in stockcodes[i]:
            rows.append(
                {
                    "date": dates[i],
                    "transaction_id": txn_ids[i],
                    "stockcode": sku,
                    "product": f"Product {sku}",
                    "customer_id": "C1",
                    "price": 10.0,
                    "quantity": 1,
                }
            )
    df = pd.DataFrame(rows)

    # 20 baskets with 2 unique products, 1 basket with 10 unique products
    # 95th percentile of [2,2,2,...,2,10] = 10 (20th out of 21)
    report = assess_data_quality(df, basket_outlier_percentile=0.95)
    assert len(report.basket_outlier_txn_ids) >= 1
    assert "T20" in report.basket_outlier_txn_ids  # The large basket


def test_incomplete_rows_detected() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", None],
            "transaction_id": ["T1", "T2", "T3"],
            "stockcode": ["A", "B", "C"],
            "product": ["Product A", "Product B", "Product C"],
            "customer_id": ["C1", "C2", "C3"],
            "price": [10.0, 20.0, 30.0],
            "quantity": [1, 1, 1],
        }
    )
    report = assess_data_quality(df)
    assert report.incomplete_rows == 1
    assert "date" in report.incomplete_row_details


def test_volume_warning() -> None:
    # Create data with few transactions but many SKUs
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"] * 100,
            "transaction_id": [f"T{i}" for i in range(100)],
            "stockcode": [f"SKU{i}" for i in range(100)],  # 100 SKUs, only 100 transactions
            "product": [f"Product {i}" for i in range(100)],
            "customer_id": ["C1"] * 100,
            "price": [10.0] * 100,
            "quantity": [1] * 100,
        }
    )
    # With 100 SKUs, min viable is 2000 transactions
    report = assess_data_quality(df)
    assert report.volume_warning is not None
    assert "Low transaction volume" in report.volume_warning


def test_filter_data_by_quality(sample_df: pd.DataFrame) -> None:
    report = assess_data_quality(sample_df, min_product_transactions=1)
    # Manually set some exclusions
    report.excluded_products = ["SKU001"]
    report.excluded_txn_ids = ["T123"]

    filtered_df, filtered_report = filter_data_by_quality(sample_df, report)

    assert "SKU001" not in filtered_df["stockcode"].values
    assert "T123" not in filtered_df["transaction_id"].values
    assert filtered_report.n_products < sample_df["stockcode"].nunique()


def test_generate_quality_summary() -> None:
    # Empty report
    empty_report = DataQualityReport(n_transactions=100, n_products=50)
    summary = generate_quality_summary(empty_report)
    assert "No data quality issues detected" in summary

    # Report with issues
    report = DataQualityReport(
        n_transactions=100,
        n_products=50,
        low_freq_products=["SKU001", "SKU002"],
        low_freq_counts={"SKU001": 5, "SKU002": 3},
        basket_outlier_txn_ids=["T1", "T2"],
        basket_outlier_threshold=10,
        basket_size_percentile=0.99,
        incomplete_rows=3,
        incomplete_row_details={"date": 2, "price": 1},
        volume_warning="Low volume",
    )
    summary = generate_quality_summary(report)
    assert "Low-frequency products" in summary
    assert "Basket size outliers" in summary
    assert "Incomplete rows" in summary
    assert "Volume warning" in summary


def test_report_serialization() -> None:
    report = DataQualityReport(
        low_freq_products=["A", "B"],
        low_freq_counts={"A": 5, "B": 3},
        basket_outlier_txn_ids=["T1"],
        basket_size_percentile=0.99,
        basket_outlier_threshold=10,
        incomplete_rows=2,
        incomplete_row_details={"date": 2},
        volume_warning="Test warning",
        n_transactions=100,
        n_products=50,
        excluded_products=["A"],
        excluded_txn_ids=["T1"],
    )
    d = report.to_dict()
    restored = DataQualityReport.from_dict(d)

    assert restored.low_freq_products == ["A", "B"]
    assert restored.low_freq_counts == {"A": 5, "B": 3}
    assert restored.basket_outlier_txn_ids == ["T1"]
    assert restored.basket_size_percentile == 0.99
    assert restored.basket_outlier_threshold == 10
    assert restored.incomplete_rows == 2
    assert restored.incomplete_row_details == {"date": 2}
    assert restored.volume_warning == "Test warning"
    assert restored.n_transactions == 100
    assert restored.n_products == 50
    assert restored.excluded_products == ["A"]
    assert restored.excluded_txn_ids == ["T1"]


def test_filter_data_by_quality_with_empty_exclusions(sample_df: pd.DataFrame) -> None:
    report = DataQualityReport()
    filtered_df, filtered_report = filter_data_by_quality(sample_df, report)
    assert len(filtered_df) == len(sample_df)
    assert filtered_report.n_products == sample_df["stockcode"].nunique()
