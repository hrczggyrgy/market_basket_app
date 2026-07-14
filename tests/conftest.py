"""Test configuration and fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing."""
    return pd.DataFrame(
        {
            "transaction_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5],
            "stockcode": ["A", "B", "C", "A", "B", "A", "C", "D", "B", "C", "A", "D"],
            "quantity": [1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1],
            "customer_id": [100, 100, 100, 101, 101, 102, 102, 102, 103, 103, 104, 104],
            "price": [10.0, 20.0, 15.0, 10.0, 20.0, 10.0, 15.0, 25.0, 20.0, 15.0, 10.0, 25.0],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-05",
                ]
            ),
        }
    )


@pytest.fixture
def sample_rules():
    """Create sample rules DataFrame for testing."""
    return pd.DataFrame(
        {
            "antecedents": [
                frozenset(["A"]),
                frozenset(["B"]),
                frozenset(["A", "B"]),
                frozenset(["C"]),
            ],
            "consequents": [
                frozenset(["B"]),
                frozenset(["C"]),
                frozenset(["C"]),
                frozenset(["A"]),
            ],
            "antecedent support": [0.4, 0.3, 0.2, 0.25],
            "consequent support": [0.3, 0.2, 0.2, 0.4],
            "support": [0.2, 0.15, 0.1, 0.1],
            "confidence": [0.5, 0.5, 0.5, 0.4],
            "lift": [1.67, 2.5, 2.5, 1.0],
        }
    )
