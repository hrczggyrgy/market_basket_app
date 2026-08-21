"""Efficient test configuration using single CSV data source."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analytics.data import load_transactions

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_FIXTURE = REPO_ROOT / "sample_data" / "sample_transactions.csv"


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Load the canonical sample transaction data once per session."""
    df, _, _, _ = load_transactions(SAMPLE_FIXTURE)
    assert len(df) > 0, "sample fixture must not be empty"
    return df


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    return SAMPLE_FIXTURE


@pytest.fixture(scope="session")
def app_path() -> Path:
    return REPO_ROOT / "app.py"


@pytest.fixture(scope="session")
def product_lookup(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Unique product-level attributes keyed by stockcode."""
    from src.analytics.data import derive_product_lookup
    return derive_product_lookup(sample_df)


@pytest.fixture(scope="session")
def revenue_series(sample_df: pd.DataFrame) -> pd.Series:
    """Line revenue as a Series (price * quantity)."""
    from src.analytics.data import revenue_column
    return revenue_column(sample_df)


@pytest.fixture(scope="session")
def capabilities(sample_df: pd.DataFrame) -> dict[str, bool]:
    """Dataset capabilities for conditional test execution."""
    from src.analytics.data import build_dataset_capabilities
    return build_dataset_capabilities(sample_df)


# Type hints for common test patterns
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest