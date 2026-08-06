"""Shared fixtures: every test uses the real checked-in sample fixture."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analytics.data import load_transactions

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FIXTURE = REPO_ROOT / "sample_data" / "sample_transactions.csv"


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    df, _, _, _ = load_transactions(SAMPLE_FIXTURE)
    assert len(df) > 0, "sample fixture must not be empty"
    return df


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    return SAMPLE_FIXTURE


@pytest.fixture(scope="session")
def app_path() -> Path:
    return REPO_ROOT / "app.py"
