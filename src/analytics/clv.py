"""Customer Lifetime Value: BG/NBD + Gamma-Gamma via the lifetimes library.

Implements the industry-standard probabilistic CLV model (Fader, Hardie &
Lee 2005). BG/NBD models purchase frequency/recency; the Gamma-Gamma submodel
models average monetary value. All model fitting/prediction is delegated to
lifetimes (no custom reimplementation).
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import ConvergenceError, summary_data_from_transaction_data

from src.analytics.basket_metrics import compute_customer_entropy
from src.analytics.schemas import CLV_CUSTOMER, CLV_DIAGNOSTICS, CLV_PREDICTIONS, check

_SEGMENTS = ["Bronze", "Silver", "Gold", "Platinum"]
_PENALIZERS = (0.01, 0.1, 0.5, 1.0, 2.0)


def predict_clv_bg_nbd(
    df: pd.DataFrame,
    prediction_horizon_days: int = 90,
    freq: str = "D",
    min_repeat_customers: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BG/NBD + Gamma-Gamma CLV predictions keyed by customer.

    Returns (clv_predictions, clv_diagnostics) both contract-validated.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    df = df[df["revenue"] > 0]

    # Collapse line items to one purchase event per basket.
    purchases = (
        df.groupby(["customer_id", "transaction_id"])
        .agg(date=("date", "min"), revenue=("revenue", "sum"))
        .reset_index()
    )
    observation_period_end = df["date"].max() - pd.Timedelta(days=prediction_horizon_days)
    summary = summary_data_from_transaction_data(
        purchases,
        customer_id_col="customer_id",
        datetime_col="date",
        monetary_value_col="revenue",
        observation_period_end=observation_period_end,
        freq=freq,
    )
    calibration = summary[(summary["frequency"] >= 1) & (summary["monetary_value"] > 0)].copy()
    if len(calibration) < min_repeat_customers:
        raise ValueError(
            f"Insufficient repeat customers for BG/NBD (need >= {min_repeat_customers}, got {len(calibration)})"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bgf, bg_penalizer = _fit_bg_nbd(calibration)
        ggf, gg_penalizer = _fit_gamma_gamma(calibration)

    t = prediction_horizon_days if freq == "D" else prediction_horizon_days / 7
    calibration["predicted_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        t, calibration["frequency"], calibration["recency"], calibration["T"]
    )
    calibration["expected_avg_value"] = ggf.conditional_expected_average_profit(
        calibration["frequency"], calibration["monetary_value"]
    )
    calibration["predicted_clv"] = calibration["predicted_purchases"] * calibration["expected_avg_value"]
    calibration["p_alive"] = bgf.conditional_probability_alive(
        calibration["frequency"], calibration["recency"], calibration["T"]
    )

    ci_lower, ci_upper = _bootstrap_clv_ci(calibration, prediction_horizon_days, freq)
    calibration["ci_lower"] = ci_lower
    calibration["ci_upper"] = ci_upper

    table = calibration.reset_index()
    table = table.rename(columns={"index": "customer_id"})
    table.columns = [("customer_id" if c == "index" else c) for c in table.columns]
    cols = [
        "customer_id",
        "frequency",
        "recency",
        "T",
        "monetary_value",
        "predicted_purchases",
        "expected_avg_value",
        "predicted_clv",
        "ci_lower",
        "ci_upper",
        "p_alive",
    ]
    table = table[[c for c in cols if c in table.columns]].copy()
    table["clv_segment"] = _segment_labels(table["predicted_clv"])
    predictions = check(table, CLV_PREDICTIONS)
    diagnostics = check(
        _build_diagnostics(bgf, ggf, len(calibration), len(summary), bg_penalizer, gg_penalizer),
        CLV_DIAGNOSTICS,
    )
    return predictions, diagnostics


def _bootstrap_clv_ci(
    calibration: pd.DataFrame,
    prediction_horizon_days: int,
    freq: str,
    n_resamples: int = 30,
    random_seed: int = 42,
    time_budget_s: float = 20.0,
) -> tuple[pd.Series, pd.Series]:
    """Percentile bootstrap CI on predicted CLV, resampling customers.

    Refits BG/NBD + Gamma-Gamma on each customer-level resample and
    predicts CLV for the observed customers present in that resample.
    Falls back to the point estimate (CI == estimate) if the resample
    cannot be fit or if the wall-clock budget is exhausted.
    """
    rng = np.random.default_rng(random_seed)
    t = prediction_horizon_days if freq == "D" else prediction_horizon_days / 7
    n_customers = len(calibration)
    replicates: dict[str, list[float]] = {c: [] for c in calibration.index}
    deadline = time.monotonic() + time_budget_s

    for _ in range(n_resamples):
        if time.monotonic() > deadline:
            break
        idx = rng.integers(0, n_customers, size=n_customers)
        sample = calibration.iloc[idx].copy()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                bgf, _ = _fit_bg_nbd(sample)
                ggf, _ = _fit_gamma_gamma(sample)
            pred_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
                t, sample["frequency"], sample["recency"], sample["T"]
            )
            pred_value = ggf.conditional_expected_average_profit(
                sample["frequency"], sample["monetary_value"]
            )
            clv = pred_purchases * pred_value
            for c, v in zip(sample.index, clv):
                replicates[c].append(float(v))
        except ValueError:
            continue

    lower, upper = [], []
    for c in calibration.index:
        vals = np.asarray(replicates.get(c, []))
        if len(vals) >= 10:
            alpha = 0.05
            lower.append(float(np.percentile(vals, 100 * alpha / 2)))
            upper.append(float(np.percentile(vals, 100 * (1 - alpha / 2))))
        else:
            point = float(calibration.loc[c, "predicted_clv"])
            lower.append(point)
            upper.append(point)
    return pd.Series(lower, index=calibration.index), pd.Series(upper, index=calibration.index)


def _fit_bg_nbd(calibration: pd.DataFrame) -> tuple[BetaGeoFitter, float]:
    last_error: Exception | None = None
    for penalizer in _PENALIZERS:
        model = BetaGeoFitter(penalizer_coef=penalizer)
        try:
            model.fit(calibration["frequency"], calibration["recency"], calibration["T"])
            return model, penalizer
        except ConvergenceError as exc:
            last_error = exc
    raise ValueError(f"BG/NBD failed to converge at any penalizer: {last_error}")


def _fit_gamma_gamma(calibration: pd.DataFrame) -> tuple[GammaGammaFitter, float]:
    last_error: Exception | None = None
    for penalizer in _PENALIZERS:
        model = GammaGammaFitter(penalizer_coef=penalizer)
        try:
            model.fit(calibration["frequency"], calibration["monetary_value"])
            return model, penalizer
        except (ConvergenceError, ValueError, TypeError) as exc:
            last_error = exc
    raise ValueError(f"Gamma-Gamma failed to converge at any penalizer: {last_error}")


def _segment_labels(predicted_clv: pd.Series) -> pd.Series:
    try:
        return pd.qcut(predicted_clv, q=4, labels=_SEGMENTS, duplicates="drop")
    except (ValueError, TypeError):
        median = predicted_clv.median()
        return pd.Series(
            np.where(predicted_clv >= median, "Gold", "Silver"),
            index=predicted_clv.index,
        )


def _build_diagnostics(
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    n_fit: int,
    n_total: int,
    bg_penalizer: float,
    gg_penalizer: float,
) -> pd.DataFrame:
    rows = [
        ("model", "BG/NBD + Gamma-Gamma"),
        ("bgf_r", float(bgf.params_["r"])),
        ("bgf_alpha", float(bgf.params_["alpha"])),
        ("bgf_a", float(bgf.params_["a"])),
        ("bgf_b", float(bgf.params_["b"])),
        ("ggf_p", float(ggf.params_["p"])),
        ("ggf_q", float(ggf.params_["q"])),
        ("ggf_v", float(ggf.params_["v"])),
        ("bgf_penalizer_used", float(bg_penalizer)),
        ("ggf_penalizer_used", float(gg_penalizer)),
        ("n_customers_fit", float(n_fit)),
        ("n_customers_total", float(n_total)),
    ]
    bg_nll = getattr(bgf, "_negative_log_likelihood_", None)
    gg_nll = getattr(ggf, "_negative_log_likelihood_", None)
    if bg_nll is not None:
        rows.append(("bgf_negative_log_likelihood", float(bg_nll)))
    if gg_nll is not None:
        rows.append(("ggf_negative_log_likelihood", float(gg_nll)))
    return pd.DataFrame(rows, columns=["metric", "value"])


def compute_clv_customer_df(
    df: pd.DataFrame,
    prediction_horizon_days: int = 90,
    freq: str = "D",
) -> pd.DataFrame:
    """Customer CLV view: BG/NBD predictions joined with behavior metrics."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    df = df[df["revenue"] > 0]

    predictions, _ = predict_clv_bg_nbd(df, prediction_horizon_days=prediction_horizon_days, freq=freq)

    metrics = (
        df.groupby("customer_id")
        .agg(
            frequency=("transaction_id", "nunique"),
            total_revenue=("revenue", "sum"),
            first_purchase=("date", "min"),
            last_purchase=("date", "max"),
        )
        .reset_index()
    )
    metrics["avg_order_value"] = metrics["total_revenue"] / metrics["frequency"].replace(0, np.nan)
    metrics["customer_lifetime_days"] = (metrics["last_purchase"] - metrics["first_purchase"]).dt.days
    metrics["recency_days"] = (df["date"].max() - metrics["last_purchase"]).dt.days

    entropy = compute_customer_entropy(df)[["customer_id", "entropy", "normalized_entropy"]]

    predictions = predictions.drop(columns=["frequency", "recency", "T", "monetary_value"], errors="ignore")
    result = predictions.merge(metrics, on="customer_id", how="left")
    result = result.merge(entropy, on="customer_id", how="left")

    annualization = 365.0 / prediction_horizon_days
    result["clv_12m"] = result["expected_avg_value"] * result["predicted_purchases"] * annualization

    cols = [
        "customer_id",
        "frequency",
        "recency_days",
        "customer_lifetime_days",
        "total_revenue",
        "avg_order_value",
        "p_alive",
        "predicted_purchases",
        "expected_avg_value",
        "predicted_clv",
        "clv_12m",
        "clv_segment",
        "entropy",
        "normalized_entropy",
    ]
    result = result[cols].sort_values("clv_12m", ascending=False).reset_index(drop=True)
    return check(result, CLV_CUSTOMER)