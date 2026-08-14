"""Tests for scenario grid analytics."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.scenarios import compute_scenario_grid
from src.analytics.schemas import SCENARIO_GRID


def test_scenario_grid_contract(sample_df: pd.DataFrame) -> None:
    table = compute_scenario_grid(sample_df)
    SCENARIO_GRID.validate(table, allow_empty=True)
    if not table.empty:
        assert set(table["scenario"].unique()) == {"pessimistic", "neutral", "optimistic"}
        assert table["projected_revenue"].min() >= 0
        assert table["weekly_growth_pct"].notna().all()
        assert table["feasible"].isin({True, False}).all()
        # Each category has exactly its 3 scenarios
        per_cat = table.groupby("category")["scenario"].nunique()
        assert (per_cat == 3).all()


def test_scenario_grid_ordering(sample_df: pd.DataFrame) -> None:
    """Neutral must sit between pessimistic and optimistic in projected revenue."""
    table = compute_scenario_grid(sample_df)
    if table.empty:
        pytest.skip("insufficient history in fixture")
    for cat, g in table.groupby("category"):
        rev = g.set_index("scenario")["projected_revenue"]
        assert rev["pessimistic"] <= rev["neutral"] + 1e-6
        assert rev["neutral"] <= rev["optimistic"] + 1e-6


def test_scenario_grid_empty_without_category() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "transaction_id": range(60),
            "stockcode": ["A"] * 60,
            "product": ["A"] * 60,
            "customer_id": ["c1"] * 60,
            "price": [1.0] * 60,
            "quantity": [1] * 60,
        }
    )
    table = compute_scenario_grid(df)
    assert table.empty
    SCENARIO_GRID.validate(table, allow_empty=True)


def test_scenario_grid_insufficient_history_returns_empty() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "transaction_id": range(10),
            "stockcode": ["A"] * 10,
            "product": ["A"] * 10,
            "category": ["C1"] * 10,
            "customer_id": ["c1"] * 10,
            "price": [1.0] * 10,
            "quantity": [1] * 10,
        }
    )
    table = compute_scenario_grid(df)
    assert table.empty


def test_scenario_grid_deterministic(sample_df: pd.DataFrame) -> None:
    a = compute_scenario_grid(sample_df)
    b = compute_scenario_grid(sample_df)
    assert a.equals(b)
