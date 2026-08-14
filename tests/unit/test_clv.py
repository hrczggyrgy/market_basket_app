"""Tests for probabilistic CLV (BG/NBD + Gamma-Gamma)."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.clv import compute_clv_customer_df, predict_clv_bg_nbd
from src.analytics.schemas import CLV_CUSTOMER, CLV_DIAGNOSTICS, CLV_PREDICTIONS, check


def test_predict_clv_contracts(sample_df: pd.DataFrame) -> None:
    predictions, diagnostics = predict_clv_bg_nbd(sample_df)
    check(predictions, CLV_PREDICTIONS)
    check(diagnostics, CLV_DIAGNOSTICS)
    assert len(predictions) >= 10
    assert (predictions["predicted_clv"] >= 0).all()
    assert predictions["p_alive"].between(0, 1).all()
    assert set(predictions["clv_segment"].unique()) <= {"Bronze", "Silver", "Gold", "Platinum"}


def test_predict_clv_diagnostics_content(sample_df: pd.DataFrame) -> None:
    _, diagnostics = predict_clv_bg_nbd(sample_df)
    metrics = set(diagnostics["metric"])
    assert {"bgf_r", "bgf_alpha", "bgf_a", "bgf_b", "ggf_p", "ggf_q", "ggf_v"} <= metrics
    numeric = pd.to_numeric(diagnostics["value"], errors="coerce")
    assert numeric.notna().all()
    assert np.isfinite(numeric).all()


def test_predict_clv_monotonic_with_frequency(sample_df: pd.DataFrame) -> None:
    predictions, _ = predict_clv_bg_nbd(sample_df)
    grouped = predictions.groupby("frequency")["predicted_purchases"].mean()
    if len(grouped) >= 3:
        assert grouped.is_monotonic_increasing


def test_predict_clv_insufficient_repeat_customers() -> None:
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame(
        {
            "date": pd.to_datetime("2026-01-01") + pd.Timedelta(days=1) * rng.integers(0, 180, n),
            "transaction_id": range(n),
            "customer_id": rng.integers(0, 5, n),
            "price": rng.uniform(5, 20, n),
            "quantity": rng.integers(1, 3, n),
        }
    )
    with pytest.raises(ValueError):
        predict_clv_bg_nbd(df)


def test_compute_clv_customer_contract(sample_df: pd.DataFrame) -> None:
    table = compute_clv_customer_df(sample_df)
    check(table, CLV_CUSTOMER)
    assert (table["clv_12m"] >= 0).all()
    assert table["clv_12m"].is_monotonic_decreasing
    assert (table["recency_days"] >= 0).all()
    assert table["normalized_entropy"].between(0, 1).all()
