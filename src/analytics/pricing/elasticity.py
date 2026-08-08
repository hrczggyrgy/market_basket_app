"""Elasticity estimation: OLS log-log, hierarchical shrinkage, cross-price."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.analytics.schemas import (
    CROSS_ELASTICITY,
    ELASTICITY,
    ELASTICITY_CONFIDENCE,
    HIERARCHICAL_ELASTICITY,
    check,
)


_EPS = 1e-10
_MIN_DISTINCT_PRICES = 3


def _check_estimable(log_price: pd.Series, log_qty: pd.Series) -> Optional[str]:
    """Return error reason if regression is numerically degenerate, else None."""
    if len(log_price) < _MIN_DISTINCT_PRICES:
        return "insufficient distinct price points"
    if log_price.std() < _EPS:
        return "near-constant price"
    if log_qty.std() < _EPS:
        return "near-constant quantity"
    # Check near-perfect collinearity via correlation
    try:
        r = log_price.corr(log_qty)
        if pd.notna(r) and abs(r) >= 1.0 - _EPS:
            return "near-perfect collinearity"
    except Exception:
        return "correlation computation failed"
    return None


def _ols_loglog(
    log_price: pd.Series,
    log_qty: pd.Series,
    use_robust: bool = True,
) -> Tuple[float, float, float, float, float, float, float]:
    """Single SKU log-log OLS: returns elasticity, std_err, p_value, r2, ci_low, ci_high, n_obs."""
    if len(log_price) < 3:
        raise ValueError("insufficient observations")

    # Align on common index and drop any remaining NaN
    common_idx = log_price.index.intersection(log_qty.index)
    log_price = log_price.loc[common_idx].dropna()
    log_qty = log_qty.loc[common_idx].dropna()
    # Re-align after dropna
    common_idx = log_price.index.intersection(log_qty.index)
    log_price = log_price.loc[common_idx]
    log_qty = log_qty.loc[common_idx]

    if len(log_price) < 3:
        raise ValueError("insufficient observations after alignment")

    reason = _check_estimable(log_price, log_qty)
    if reason:
        raise ValueError(f"degenerate case: {reason}")

    if use_robust:
        X = sm.add_constant(log_price)
        model = sm.OLS(log_qty, X).fit(cov_type="HC3")
        elasticity = float(model.params[log_price.name])
        std_err = float(model.bse[log_price.name])
        p_value = float(model.pvalues[log_price.name])
        r2 = float(model.rsquared)
        conf = model.conf_int().loc[log_price.name]
        ci_low, ci_high = float(conf[0]), float(conf[1])
    else:
        slope, intercept, r, p, se = stats.linregress(log_price, log_qty)
        # Guard against NaN from linregress (constant x or y at certain n)
        if not np.isfinite(slope) or not np.isfinite(p) or not np.isfinite(se):
            raise ValueError("linregress produced non-finite result")
        elasticity = float(slope)
        std_err = float(se)
        p_value = float(p)
        r2 = float(r**2)
        ci_low = elasticity - 1.96 * std_err
        ci_high = elasticity + 1.96 * std_err

    # Final sanity checks
    if not (0.0 <= p_value <= 1.0):
        raise ValueError(f"p_value out of range: {p_value}")
    if not np.isfinite(elasticity) or not np.isfinite(std_err) or not np.isfinite(r2):
        raise ValueError("non-finite regression output")

    return elasticity, std_err, p_value, r2, ci_low, ci_high, len(log_price)


def estimate_loglog_elasticity(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    price_col: str = "price",
    qty_col: str = "quantity",
    date_col: str = "date",
    freq: str = "W",
    min_periods: int = 10,
    min_price_variation: float = 0.05,
    use_robust_se: bool = True,
) -> pd.DataFrame:
    """Per-SKU log-log OLS elasticity with diagnostics.

    Returns DataFrame with one row per SKU meeting minimum data requirements.
    """
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["revenue"] = df[price_col] * df[qty_col]

    results: list[dict[str, float | int | str]] = []

    for product_id in df[product_col].unique():
        prod_df = df[df[product_col] == product_id].copy()

        weekly = (
            prod_df.set_index(date_col)
            .groupby(pd.Grouper(freq=freq))
            .agg(avg_price=(price_col, "mean"), total_qty=(qty_col, "sum"))
            .dropna()
        )

        if len(weekly) < min_periods:
            continue

        price_cv = weekly["avg_price"].std() / weekly["avg_price"].mean()
        if price_cv < min_price_variation:
            continue

        log_price = np.log(weekly["avg_price"].replace(0, np.nan).dropna())
        log_qty = np.log(weekly.loc[log_price.index, "total_qty"].replace(0, np.nan).dropna())

        # Align indices (fixes misaligned dropna bug)
        common_idx = log_price.index.intersection(log_qty.index)
        log_price = log_price.loc[common_idx]
        log_qty = log_qty.loc[common_idx]

        if len(log_price) < min_periods:
            continue

        # Require minimum distinct price points (not just total obs)
        if log_price.nunique() < _MIN_DISTINCT_PRICES:
            continue

        try:
            elast, se, pval, r2, ci_low, ci_high, n_obs = _ols_loglog(
                log_price, log_qty, use_robust_se
            )
        except Exception:
            continue

        results.append(
            {
                "stockcode": product_id,
                "elasticity": elast,
                "r_squared": r2,
                "p_value": pval,
                "std_err": se,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "n_obs": n_obs,
                "avg_price": float(weekly["avg_price"].mean()),
                "avg_weekly_qty": float(weekly["total_qty"].mean()),
                "price_cv": float(price_cv),
            }
        )

    if not results:
        return check(pd.DataFrame(columns=list(ELASTICITY.columns)), ELASTICITY, allow_empty=True)

    table = pd.DataFrame(results, columns=list(ELASTICITY.columns))
    return check(table, ELASTICITY)


def estimate_hierarchical_elasticity(
    transactions_df: pd.DataFrame,
    category_col: str = "category",
    freq: str = "W",
    min_periods: int = 10,
    min_price_variation: float = 0.05,
) -> pd.DataFrame:
    """Empirical Bayes / partial pooling elasticity with James-Stein shrinkage.

    Individual SKU elasticities are shrunk toward their category mean using
    variance-weighted shrinkage: weight = between_var / (within_var + between_var).
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    ols_results: list[dict[str, float | int | str]] = []

    for product_id in df["stockcode"].unique():
        prod_df = df[df["stockcode"] == product_id]
        cat = prod_df[category_col].iloc[0] if category_col in prod_df.columns else "UNKNOWN"

        weekly = (
            prod_df.set_index("date")
            .groupby(pd.Grouper(freq=freq))
            .agg(avg_price=("price", "mean"), total_qty=("quantity", "sum"))
            .dropna()
        )

        if len(weekly) < min_periods:
            continue

        price_cv = weekly["avg_price"].std() / weekly["avg_price"].mean()
        if price_cv < min_price_variation:
            continue

        log_price = np.log(weekly["avg_price"].replace(0, np.nan).dropna())
        log_qty = np.log(weekly.loc[log_price.index, "total_qty"].replace(0, np.nan).dropna())

        # Align indices
        common_idx = log_price.index.intersection(log_qty.index)
        log_price = log_price.loc[common_idx]
        log_qty = log_qty.loc[common_idx]

        if len(log_price) < min_periods:
            continue

        if log_price.nunique() < _MIN_DISTINCT_PRICES:
            continue

        try:
            elast, se, pval, r2, _, _, n_obs = _ols_loglog(log_price, log_qty, use_robust=False)
        except Exception:
            continue

        ols_results.append(
            {
                "stockcode": product_id,
                "category": cat,
                "elasticity_ols": elast,
                "r_squared": r2,
                "p_value": pval,
                "n_obs": n_obs,
                "avg_price": float(weekly["avg_price"].mean()),
                "std_err": se,
            }
        )

    if not ols_results:
        return check(pd.DataFrame(columns=list(HIERARCHICAL_ELASTICITY.columns)), HIERARCHICAL_ELASTICITY, allow_empty=True)

    ols_df = pd.DataFrame(ols_results)

    # Category-level stats
    cat_vars = ols_df.groupby("category")["elasticity_ols"].var().rename("cat_var")
    cat_means = ols_df.groupby("category")["elasticity_ols"].mean().rename("elasticity_cat")
    cat_n = ols_df.groupby("category").size().rename("cat_n")

    ols_df = ols_df.merge(cat_means, on="category", how="left")
    ols_df = ols_df.merge(cat_vars, on="category", how="left")
    ols_df = ols_df.merge(cat_n, on="category", how="left")

    ols_df["within_var"] = ols_df["std_err"] ** 2
    ols_df["between_var"] = ols_df["cat_var"].fillna(ols_df["within_var"].mean())

    # Shrinkage weight
    ols_df["shrink_weight"] = ols_df["between_var"] / (
        ols_df["within_var"] + ols_df["between_var"] + 1e-8
    )
    ols_df["shrink_weight"] = ols_df["shrink_weight"].clip(0.05, 0.95)

    ols_df["elasticity_shrunk"] = (
        ols_df["shrink_weight"] * ols_df["elasticity_cat"]
        + (1 - ols_df["shrink_weight"]) * ols_df["elasticity_ols"]
    )

    table = ols_df[list(HIERARCHICAL_ELASTICITY.columns)]
    return check(table, HIERARCHICAL_ELASTICITY)


def classify_elasticity_confidence(
    elasticity_df: pd.DataFrame,
    ci_relative_width_threshold: float = 2.0,
    ci_relative_width_low: float = 4.0,
) -> pd.DataFrame:
    """Classify elasticity estimates into confidence tiers and demand direction.

    Confidence tiers (per SKU, from its 95% CI width and significance):
    - high:   significant (p < 0.05) and CI width < ``ci_relative_width_threshold``
              times the magnitude of the point estimate.
    - medium: significant but wider CI, or non-significant with a tight CI.
    - low:    otherwise (wide CI and/or non-significant) — not actionable.

    ``direction`` labels the demand regime: elastic (|e| > 1.05),
    unit_elastic (|e| in [0.95, 1.05]), inelastic (|e| < 0.95).

    Args:
        elasticity_df: Output of ``estimate_loglog_elasticity`` (ELASTICITY contract).
        ci_relative_width_threshold: Max CI width (as multiple of |elasticity|)
            for a high-confidence label.
        ci_relative_width_low: CI width multiple above which the estimate is
            considered low confidence.

    Returns:
        DataFrame validated against ELASTICITY_CONFIDENCE (empty input yields
        an empty, validated frame).
    """
    empty = pd.DataFrame(columns=list(ELASTICITY_CONFIDENCE.columns))
    if elasticity_df is None or elasticity_df.empty:
        return check(empty, ELASTICITY_CONFIDENCE, allow_empty=True)

    required = {"stockcode", "elasticity", "ci_lower", "ci_upper", "p_value", "n_obs"}
    if not required.issubset(elasticity_df.columns):
        return check(empty, ELASTICITY_CONFIDENCE, allow_empty=True)

    df = elasticity_df.copy()
    df["ci_width"] = df["ci_upper"] - df["ci_lower"]
    df["significant"] = df["p_value"] < 0.05

    def _tier(row: pd.Series) -> str:
        magnitude = max(abs(row["elasticity"]), 1e-6)
        relative = row["ci_width"] / magnitude
        if row["significant"] and relative < ci_relative_width_threshold:
            return "high"
        if relative < ci_relative_width_low:
            return "medium"
        return "low"

    df["confidence"] = df.apply(_tier, axis=1)

    def _direction(e: float) -> str:
        if abs(e) > 1.05:
            return "elastic"
        if abs(e) >= 0.95:
            return "unit_elastic"
        return "inelastic"

    df["direction"] = df["elasticity"].apply(_direction)

    table = df[list(ELASTICITY_CONFIDENCE.columns)].reset_index(drop=True)
    return check(table, ELASTICITY_CONFIDENCE)


def estimate_cross_price_elasticity(
    transactions_df: pd.DataFrame,
    product_pairs: List[Tuple[str, str]],
    category_col: str = "category",
    freq: str = "W",
    min_periods: int = 10,
    min_price_variation: float = 0.05,
) -> pd.DataFrame:
    """Bivariate log-log OLS cross-price elasticity for specified pairs.

    log(qty_A) = alpha + beta_own * log(price_A) + beta_cross * log(price_B)
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    results: list[dict[str, float | int | str]] = []

    for prod_a, prod_b in product_pairs:
        prod_a_df = df[df["stockcode"] == prod_a].copy()
        prod_b_df = df[df["stockcode"] == prod_b].copy()

        if len(prod_a_df) < min_periods or len(prod_b_df) < min_periods:
            continue

        weekly_a = (
            prod_a_df.set_index("date")
            .groupby(pd.Grouper(freq=freq))
            .agg(avg_price_a=("price", "mean"), total_qty_a=("quantity", "sum"))
            .dropna()
        )
        weekly_b = (
            prod_b_df.set_index("date")
            .groupby(pd.Grouper(freq=freq))
            .agg(avg_price_b=("price", "mean"), total_qty_b=("quantity", "sum"))
            .dropna()
        )

        weekly = weekly_a.join(weekly_b, how="inner")
        if len(weekly) < min_periods:
            continue

        cv_a = weekly["avg_price_a"].std() / weekly["avg_price_a"].mean()
        cv_b = weekly["avg_price_b"].std() / weekly["avg_price_b"].mean()
        if cv_a < min_price_variation or cv_b < min_price_variation:
            continue

        log_price_a = np.log(weekly["avg_price_a"].replace(0, np.nan))
        log_price_b = np.log(weekly["avg_price_b"].replace(0, np.nan))
        log_qty_a = np.log(weekly["total_qty_a"].replace(0, np.nan))

        # Align indices and drop any NaN
        common_idx = log_price_a.index.intersection(log_price_b.index).intersection(log_qty_a.index)
        log_price_a = log_price_a.loc[common_idx].dropna()
        log_price_b = log_price_b.loc[common_idx].dropna()
        log_qty_a = log_qty_a.loc[common_idx].dropna()
        # Re-align after dropna
        common_idx = log_price_a.index.intersection(log_price_b.index).intersection(log_qty_a.index)
        log_price_a = log_price_a.loc[common_idx]
        log_price_b = log_price_b.loc[common_idx]
        log_qty_a = log_qty_a.loc[common_idx]

        if len(log_price_a) < min_periods:
            continue

        if log_price_a.nunique() < _MIN_DISTINCT_PRICES or log_price_b.nunique() < _MIN_DISTINCT_PRICES:
            continue

        X = np.column_stack([log_price_a.values, log_price_b.values])
        X = sm.add_constant(X)
        y = log_qty_a.values

        try:
            model = sm.OLS(y, X).fit(cov_type="HC3")
            own_elast = float(model.params[1])
            cross_elast = float(model.params[2])
            own_se = float(model.bse[1])
            cross_se = float(model.bse[2])
            own_p = float(model.pvalues[1])
            cross_p = float(model.pvalues[2])
            r2 = float(model.rsquared)
        except Exception:
            continue

        results.append(
            {
                "product_a": prod_a,
                "product_b": prod_b,
                "own_elasticity": own_elast,
                "own_elasticity_se": own_se,
                "own_elasticity_p": own_p,
                "cross_elasticity": cross_elast,
                "cross_elasticity_se": cross_se,
                "cross_elasticity_p": cross_p,
                "r_squared": r2,
                "n_obs": len(log_price_a),
                "avg_price_a": float(weekly["avg_price_a"].mean()),
                "avg_price_b": float(weekly["avg_price_b"].mean()),
            }
        )

    if not results:
        return check(pd.DataFrame(columns=list(CROSS_ELASTICITY.columns)), CROSS_ELASTICITY, allow_empty=True)

    table = pd.DataFrame(results, columns=list(CROSS_ELASTICITY.columns))
    return check(table, CROSS_ELASTICITY)