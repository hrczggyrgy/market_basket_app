"""Tests for causal promotional incrementality TWFE estimation."""

import pytest

import numpy as np
import pandas as pd

from src.analytics.promo.causal import build_promo_causal_panel, estimate_twfe_promo_effect
from src.analytics.promo import detect_promotions
from src.analytics.schemas import check, PROMO_TWFE_RESULT


@pytest.fixture()
def crafted_df() -> pd.DataFrame:
    """A df with one obvious 3-day 30% promo on product A."""
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    rows = []
    for sku in ("A", "B"):
        for i, day in enumerate(dates):
            if sku == "A" and 10 <= i <= 12:
                price, qty = 7.0, 8
            elif sku == "A":
                price, qty = 10.0, 2
            else:
                price, qty = 5.0, 1
            rows.append(
                {
                    "date": day,
                    "transaction_id": f"T{sku}{i}",
                    "stockcode": sku,
                    "product": f"Product {sku}",
                    "customer_id": 1000 + i,
                    "price": price,
                    "quantity": qty,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def panel_with_promo(crafted_df: pd.DataFrame) -> pd.DataFrame:
    """A panel built from crafted promo data."""
    promos = detect_promotions(crafted_df)
    panel = build_promo_causal_panel(crafted_df, promos)
    # Convert week period to timestamp for PanelOLS compatibility
    panel_copy = panel.copy()
    panel_copy["week"] = panel_copy["week"].dt.to_timestamp()
    return panel_copy


def test_twfe_non_null_coefficients(panel_with_promo: pd.DataFrame) -> None:
    """Test that TWFE estimation returns non-NaN promo_coefficient and n_obs > 0."""
    result = estimate_twfe_promo_effect(
        panel_with_promo, outcome="log_units", price_control=True, cluster_se=True
    )

    check(result, PROMO_TWFE_RESULT, allow_empty=True)

    # Assert non-NaN promo coefficient
    assert not np.isnan(result["promo_coefficient"].iloc[0]), (
        f"promo_coefficient should not be NaN, got {result['promo_coefficient'].iloc[0]}"
    )

    # Assert n_obs > 0
    assert result["n_obs"].iloc[0] > 0, f"n_obs should be > 0, got {result['n_obs'].iloc[0]}"


def test_twfe_non_null_coefficients_no_price_control(panel_with_promo: pd.DataFrame) -> None:
    """Test TWFE estimation without price control also returns non-NaN results."""
    result = estimate_twfe_promo_effect(
        panel_with_promo, outcome="log_units", price_control=False, cluster_se=True
    )

    check(result, PROMO_TWFE_RESULT, allow_empty=True)

    # Assert non-NaN promo coefficient
    assert not np.isnan(result["promo_coefficient"].iloc[0]), (
        f"promo_coefficient should not be NaN, got {result['promo_coefficient'].iloc[0]}"
    )

    # Assert n_obs > 0
    assert result["n_obs"].iloc[0] > 0, f"n_obs should be > 0, got {result['n_obs'].iloc[0]}"


def test_twfe_with_log_revenue(panel_with_promo: pd.DataFrame) -> None:
    """Test TWFE estimation with log_revenue outcome."""
    result = estimate_twfe_promo_effect(
        panel_with_promo, outcome="log_revenue", price_control=True, cluster_se=True
    )

    check(result, PROMO_TWFE_RESULT, allow_empty=True)

    # Assert non-NaN promo coefficient
    assert not np.isnan(result["promo_coefficient"].iloc[0]), (
        f"promo_coefficient should not be NaN, got {result['promo_coefficient'].iloc[0]}"
    )

    # Assert n_obs > 0
    assert result["n_obs"].iloc[0] > 0, f"n_obs should be > 0, got {result['n_obs'].iloc[0]}"


def test_twfe_with_units_outcome(panel_with_promo: pd.DataFrame) -> None:
    """Test TWFE estimation with units (non-log) outcome."""
    result = estimate_twfe_promo_effect(
        panel_with_promo, outcome="units", price_control=True, cluster_se=True
    )

    check(result, PROMO_TWFE_RESULT, allow_empty=True)

    # Assert non-NaN promo coefficient
    assert not np.isnan(result["promo_coefficient"].iloc[0]), (
        f"promo_coefficient should not be NaN, got {result['promo_coefficient'].iloc[0]}"
    )

    # Assert n_obs > 0
    assert result["n_obs"].iloc[0] > 0, f"n_obs should be > 0, got {result['n_obs'].iloc[0]}"


def test_build_panel_week_accessor() -> None:
    """Test that build_promo_causal_panel uses isocalendar week accessor (pandas 2.3+ compat)."""
    from src.analytics.promo import detect_promotions

    crafted_df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=40, freq="D"),
            "transaction_id": ["T0"] * 40,
            "stockcode": ["A"] * 40,
            "product": ["Product A"] * 40,
            "customer_id": list(range(1000, 1040)),
            "price": [10.0] * 40,
            "quantity": [2] * 40,
        }
    )
    promos = detect_promotions(crafted_df)
    panel = build_promo_causal_panel(crafted_df, promos)

    # Verify week_num column exists and has integer values
    assert "week_num" in panel.columns
    assert panel["week_num"].dtype in (int, np.int64)

    # Verify no AttributeError from deprecated dt.week accessor
    # Use to_timestamp().dt.isocalendar().week path (required for Period dtype in pandas 2.3.3)
    week_num_from_accessor = panel["week"].dt.to_timestamp().dt.isocalendar().week.astype(int)
    import pandas._testing as tm
    tm.assert_series_equal(panel["week_num"], week_num_from_accessor.rename("week_num"))
