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
from scipy.stats import pearsonr, spearmanr

from src.analytics.basket_metrics import compute_customer_entropy
from src.analytics.schemas import CLV_CUSTOMER, CLV_DIAGNOSTICS, CLV_PREDICTIONS, check

_SEGMENTS = ["Bronze", "Silver", "Gold", "Platinum"]
_PENALIZERS = (0.01, 0.1, 0.5, 1.0, 2.0)


def predict_clv_bg_nbd(
    df: pd.DataFrame,
    prediction_horizon_days: int = 90,
    freq: str = "D",
    min_repeat_customers: int = 10,
    discount_rate_pct: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BG/NBD + Gamma-Gamma CLV predictions keyed by customer.

    Returns (clv_predictions, clv_diagnostics) both contract-validated.

    ``discount_rate_pct`` is an annual discount rate applied to future
    expected purchases (monthly compounding, uniform purchase timing).
    A rate of 0 means no discounting.
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
    discount = _discount_factor(prediction_horizon_days, discount_rate_pct)
    calibration["predicted_clv"] = calibration["predicted_purchases"] * calibration["expected_avg_value"] * discount
    calibration["p_alive"] = bgf.conditional_probability_alive(
        calibration["frequency"], calibration["recency"], calibration["T"]
    )

    ci_lower, ci_upper = _bootstrap_clv_ci(calibration, prediction_horizon_days, freq)
    calibration["ci_lower"] = ci_lower * discount
    calibration["ci_upper"] = ci_upper * discount

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
        _build_diagnostics(bgf, ggf, len(calibration), len(summary), bg_penalizer, gg_penalizer, purchases, discount_rate_pct),
        CLV_DIAGNOSTICS,
    )
    return predictions, diagnostics


def _discount_factor(horizon_days: int, annual_rate_pct: float) -> float:
    """Mean present-value factor for purchases spread uniformly over the horizon.

    Monthly compounding: factor = (1 - (1+r)^-n) / (r*n) is the per-dollar
    discount applied to a uniform stream of purchases over the horizon.
    A rate of 0 (or a horizon <= 0) yields 1.0 (no discounting).
    """
    if annual_rate_pct <= 0 or horizon_days <= 0:
        return 1.0
    r_m = (annual_rate_pct / 100.0) / 12.0
    n_months = max(1.0, horizon_days / 30.44)
    if r_m == 0:
        return 1.0
    return float((1 - (1 + r_m) ** -n_months) / (r_m * n_months))


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
    purchases: pd.DataFrame | None = None,
    discount_rate_pct: float = 0.0,
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
        ("discount_rate_pct", float(discount_rate_pct)),
    ]
    bg_nll = getattr(bgf, "_negative_log_likelihood_", None)
    gg_nll = getattr(ggf, "_negative_log_likelihood_", None)
    if bg_nll is not None:
        rows.append(("bgf_negative_log_likelihood", float(bg_nll)))
    if gg_nll is not None:
        rows.append(("ggf_negative_log_likelihood", float(gg_nll)))

    if purchases is not None and len(purchases) > 0:
        customers = (
            purchases.groupby("customer_id")
            .agg(frequency=("transaction_id", "size"), monetary_value=("revenue", "mean"))
            .reset_index()
        )
        customers = customers[customers["frequency"] >= 2]
        if len(customers) >= 5:
            pearson_corr, spearman_corr = _monetary_frequency_correlations(customers)
            rows.append(("gg_freq_value_pearson", float(pearson_corr)))
            rows.append(("gg_freq_value_spearman", float(spearman_corr)))
            rows.append(("gg_independence_status", float(_gg_independence_status(pearson_corr, spearman_corr))))
            if len(customers) >= 10:
                stationarity = _avg_order_value_stationarity(purchases)
                if np.isfinite(stationarity):
                    rows.append(("gg_value_stationarity_pct", float(stationarity)))
    return pd.DataFrame(rows, columns=["metric", "value"])


def _monetary_frequency_pair(customers: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = customers["frequency"].to_numpy(dtype=float)
    y = customers["monetary_value"].to_numpy(dtype=float)
    return x, y


def _monetary_frequency_correlations(customers: pd.DataFrame) -> tuple[float, float]:
    """Pearson/Spearman correlation between purchase frequency and avg order value."""
    x, y = _monetary_frequency_pair(customers)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 0.0
    pearson_corr, _ = pearsonr(x, y)
    spearman_corr, _ = spearmanr(x, y)
    return float(pearson_corr), float(spearman_corr)


def _gg_independence_status(pearson_corr: float, spearman_corr: float) -> float:
    """Numeric Gamma-Gamma independence code: 0 = largely met, 1 = partially met, 2 = violated."""
    strongest = max(abs(pearson_corr), abs(spearman_corr))
    if strongest < 0.2:
        return 0.0
    if strongest < 0.4:
        return 1.0
    return 2.0


def _avg_order_value_stationarity(purchases: pd.DataFrame) -> float:
    """Share of customers with >= 3 purchases whose first-half vs second-half
    AOV ratio stays within [0.5, 2.0] (rough per-customer spend stationarity)."""
    purchases = purchases.copy()
    purchases["date"] = pd.to_datetime(purchases["date"])
    purchases = purchases.sort_values(["customer_id", "date"])
    consistent = []
    for _, grp in purchases.groupby("customer_id"):
        if len(grp) < 3:
            continue
        half = len(grp) // 2
        early = float(grp["revenue"].iloc[:half].mean())
        late = float(grp["revenue"].iloc[half:].mean())
        if early <= 0 or late <= 0:
            continue
        ratio = late / early
        consistent.append(0.5 <= ratio <= 2.0)
    if not consistent:
        return float("nan")
    return float(np.mean(consistent))


def compute_clv_customer_df(
    df: pd.DataFrame,
    prediction_horizon_days: int = 90,
    freq: str = "D",
    discount_rate_pct: float = 0.0,
) -> pd.DataFrame:
    """Customer CLV view: BG/NBD predictions joined with behavior metrics."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    df = df[df["revenue"] > 0]

    predictions, _ = predict_clv_bg_nbd(
        df,
        prediction_horizon_days=prediction_horizon_days,
        freq=freq,
        discount_rate_pct=discount_rate_pct,
    )

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
    result["clv_12m_discounted"] = (
        result["clv_12m"] * _discount_factor(365, discount_rate_pct)
        if discount_rate_pct > 0
        else result["clv_12m"]
    )

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
        "clv_12m_discounted",
        "clv_segment",
        "entropy",
        "normalized_entropy",
    ]
    result = result[cols].sort_values("clv_12m", ascending=False).reset_index(drop=True)
    return check(result, CLV_CUSTOMER)