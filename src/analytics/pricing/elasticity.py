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
    ELASTICITY_STATUS,
    HIERARCHICAL_ELASTICITY,
    check,
)

_EPS = 1e-10
_MIN_DISTINCT_PRICES = 3


def _check_estimable(log_price: pd.Series, log_qty: pd.Series) -> Optional[str]:
    """Return error reason if regression is numerically degenerate, else None.

    Returns specific error codes for better user understanding:
    - insufficient_price_points: fewer than 3 distinct price points
    - near_constant_price: price variation below threshold
    - near_constant_quantity: quantity variation below threshold
    - near_perfect_collinearity: price and quantity are perfectly correlated
    - extreme_values: log values indicate data quality issues
    - correlation_failed: correlation computation error

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> prices = pd.Series([1.0, 1.0, 1.0])  # Constant price
        >>> qtys = pd.Series([10, 12, 8])
        >>> _check_estimable(np.log(prices), np.log(qtys))
        'near_constant_price'
    """
    if len(log_price) < _MIN_DISTINCT_PRICES:
        return "insufficient_price_points"
    if log_price.std() < _EPS:
        return "near_constant_price"
    if log_qty.std() < _EPS:
        return "near_constant_quantity"

    # Check for extreme log values (data quality issue)
    if np.any(np.abs(log_price) > 10) or np.any(np.abs(log_qty) > 10):
        return "extreme_values"

    # Check near-perfect collinearity via correlation
    try:
        r = log_price.corr(log_qty)
        if pd.notna(r) and abs(r) >= 1.0 - _EPS:
            return "near_perfect_collinearity"
    except Exception:
        return "correlation_failed"
    return None


def _ols_loglog(
    log_price: pd.Series,
    log_qty: pd.Series,
    use_robust: bool = True,
    time_dummies: Optional[pd.DataFrame] = None,
) -> Tuple[float, float, float, float, float, float, float]:
    """Single SKU log-log OLS: returns elasticity, std_err, p_value, r2, ci_low, ci_high, n_obs.

    Enhanced with numerical stability checks and robust error handling.
    Optional time_dummies can be included as fixed effects.
    """
    import warnings as _warnings

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

    # Additional numerical stability checks
    if np.any(np.abs(log_price) > 10):  # Check for extreme log values
        _warnings.warn(
            f"Extreme log price values detected (max: {log_price.max():.2f}). "
            "Elasticity estimates may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    if np.any(np.abs(log_qty) > 10):  # Check for extreme log values
        _warnings.warn(
            f"Extreme log quantity values detected (max: {log_qty.max():.2f}). "
            "Elasticity estimates may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    try:
        if use_robust:
            X = sm.add_constant(log_price)
            if time_dummies is not None:
                # Align time dummies with log_price index
                time_dummies_aligned = time_dummies.loc[log_price.index]
                X = pd.concat([X, time_dummies_aligned], axis=1)
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

        # Check for economically implausible elasticities
        if abs(elasticity) > 10:
            _warnings.warn(
                f"Extreme elasticity value detected: {elasticity:.2f}. "
                "This may indicate data quality issues or model misspecification.",
                UserWarning,
                stacklevel=2,
            )

        return elasticity, std_err, p_value, r2, ci_low, ci_high, len(log_price)

    except Exception as e:
        _warnings.warn(
            f"OLS regression failed: {e}. This SKU will be skipped in elasticity estimation.",
            UserWarning,
            stacklevel=2,
        )
        raise


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
    add_time_fe: bool = True,
) -> pd.DataFrame:
    """Per-SKU log-log OLS elasticity with diagnostics (vectorized preprocessing).

    WARNING: This estimates OBSERVED price response, NOT causal elasticity.
    - Endogeneity: price and quantity are simultaneously determined
    - No instrument for price; OLS is biased if demand/supply shocks correlate
    - Results are descriptive: "how quantity co-varies with price historically"
    - For causal inference, use IV, RDD, or experimental methods (with valid instruments)

    Returns DataFrame with one row per SKU meeting minimum data requirements.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2024-01-01', periods=20, freq='W'),
        ...     'stockcode': ['A'] * 20,
        ...     'price': [10.0, 10.5, 11.0, 10.8, 10.2] * 4,
        ...     'quantity': [100, 95, 90, 92, 98] * 4,
        ... })
        >>> result = estimate_loglog_elasticity(df, min_periods=5)
        >>> 'elasticity' in result.columns
        True
    """
    import warnings

    warnings.warn(
        "estimate_loglog_elasticity estimates OBSERVED price response, NOT causal elasticity. "
        "Price endogeneity is not addressed. Results are descriptive only.",
        UserWarning,
        stacklevel=2,
    )
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # VECTORIZED: Compute weekly aggregates for ALL SKUs in one groupby
    weekly = (
        df.set_index(date_col)
        .groupby([product_col, pd.Grouper(freq=freq)])
        .agg(avg_price=(price_col, "mean"), total_qty=(qty_col, "sum"))
        .dropna()
        .reset_index()
    )

    if weekly.empty:
        return check(pd.DataFrame(columns=list(ELASTICITY.columns)), ELASTICITY, allow_empty=True)

    # VECTORIZED: Filter SKUs meeting minimum criteria using groupby transform
    sku_counts = weekly.groupby(product_col).size().rename("n_obs")
    weekly = weekly.merge(sku_counts, on=product_col)

    # Filter by minimum periods
    weekly = weekly[weekly["n_obs"] >= min_periods].copy()
    if weekly.empty:
        return check(pd.DataFrame(columns=list(ELASTICITY.columns)), ELASTICITY, allow_empty=True)

    # VECTORIZED: Compute price CV per SKU
    price_stats = weekly.groupby(product_col).agg(
        price_mean=("avg_price", "mean"),
        price_std=("avg_price", "std"),
        price_nunique=("avg_price", "nunique"),
        qty_zeros=("total_qty", lambda x: (x == 0).sum()),
        price_zeros=("avg_price", lambda x: (x == 0).sum()),
    ).reset_index()

    price_stats["price_cv"] = price_stats["price_std"] / price_stats["price_mean"]

    # Filter by price variation and distinct price points
    valid_skus = price_stats[
        (price_stats["price_cv"] >= min_price_variation)
        & (price_stats["price_nunique"] >= _MIN_DISTINCT_PRICES)
        & (price_stats["qty_zeros"] == 0)
        & (price_stats["price_zeros"] == 0)
    ][product_col].tolist()

    if not valid_skus:
        return check(pd.DataFrame(columns=list(ELASTICITY.columns)), ELASTICITY, allow_empty=True)

    weekly = weekly[weekly[product_col].isin(valid_skus)].copy()

    # Add log columns
    weekly["log_price"] = np.log(weekly["avg_price"])
    weekly["log_qty"] = np.log(weekly["total_qty"])

    # Now process each valid SKU (OLS still needs per-SKU but data prep is vectorized)
    results: list[dict[str, float | int | str]] = []

    for product_id in valid_skus:
        sku_weekly = weekly[weekly[product_col] == product_id].sort_values(date_col)

        log_price = sku_weekly["log_price"]
        log_qty = sku_weekly["log_qty"]

        # Align indices
        common_idx = log_price.index.intersection(log_qty.index)
        log_price = log_price.loc[common_idx]
        log_qty = log_qty.loc[common_idx]

        if len(log_price) < min_periods:
            continue

        # Create time fixed effects if requested
        time_dummies = None
        try:
            if add_time_fe and len(sku_weekly) > 3:  # Need enough obs for dummies
                # Create week-of-year and month dummies
                if hasattr(sku_weekly[date_col].dt, "isocalendar"):
                    week_values = sku_weekly[date_col].dt.isocalendar().week
                else:
                    week_values = sku_weekly[date_col].dt.week
                week_dummies = pd.get_dummies(week_values, prefix="week", drop_first=True)
                week_dummies.index = sku_weekly.index
                month_dummies = pd.get_dummies(sku_weekly[date_col].dt.month, prefix="month", drop_first=True)
                month_dummies.index = sku_weekly.index
                time_dummies = pd.concat([week_dummies, month_dummies], axis=1).astype(float)
                time_dummies = time_dummies.loc[log_price.index]
                
                # Limit dummies to avoid overfitting (max 1/3 of observations)
                max_dummies = max(1, len(log_price) // 3)
                if time_dummies.shape[1] > max_dummies:
                    time_dummies = time_dummies.iloc[:, :max_dummies]

            elast, se, pval, r2, ci_low, ci_high, n_obs = _ols_loglog(
                log_price, log_qty, use_robust_se, time_dummies=time_dummies
            )
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            continue

        avg_price = float(sku_weekly["avg_price"].mean())
        avg_weekly_qty = float(sku_weekly["total_qty"].mean())
        price_cv = float(price_stats[price_stats[product_col] == product_id]["price_cv"].iloc[0])

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
                "avg_price": avg_price,
                "avg_weekly_qty": avg_weekly_qty,
                "price_cv": price_cv,
            }
        )

    if not results:
        return check(pd.DataFrame(columns=list(ELASTICITY.columns)), ELASTICITY, allow_empty=True)

    table = pd.DataFrame(results, columns=list(ELASTICITY.columns))
    return check(table, ELASTICITY)


def compute_elasticity_status(
    transactions_df: pd.DataFrame,
    elasticity_df: Optional[pd.DataFrame] = None,
    product_col: str = "stockcode",
    price_col: str = "price",
    qty_col: str = "quantity",
    date_col: str = "date",
    freq: str = "W",
    min_periods: int = 10,
    min_price_variation: float = 0.05,
) -> pd.DataFrame:
    """Per-SKU elasticity estimability status for every SKU (coverage view) - VECTORIZED.

    Unlike ``estimate_loglog_elasticity`` (which returns only SKUs with a
    usable estimate), this returns one row per SKU with an explicit
    ``elasticity_status`` so callers can answer "why is this SKU missing?"
    instead of silently dropping it or misreading it as perfectly inelastic.

    Status values:
    - estimated:                 usable estimate available.
    - weak:                      estimate available but low confidence (wide CI
                                 or not significant).
    - insufficient_observations: fewer than ``min_periods`` weekly periods.
    - insufficient_variation:    price CV < ``min_price_variation``.
    - insufficient_price_points: fewer than 3 distinct price points.
    - near_constant_price:       price variation below numerical threshold.
    - near_constant_quantity:    quantity variation below numerical threshold.
    - near_perfect_collinearity: price and quantity are perfectly correlated.
    - extreme_values:            log values indicate data quality issues.
    - correlation_failed:        correlation computation error.
    - model_failed:              regression convergence failure.
    - not_significant:           p-value >= 0.05 (used when confidence is low).
    - unavailable:               no estimate (zero prices/quantities, or no
                                 ``elasticity_df`` supplied).

    Args:
        transactions_df: Transaction data with date, price, quantity, stockcode.
        elasticity_df: Optional output of ``estimate_loglog_elasticity`` used to
            refine candidate statuses (estimated vs weak) and carry the point
            estimate, r-squared and confidence tier.
        freq: Resampling frequency for weekly aggregates.
        min_periods: Minimum weekly periods for an estimate.
        min_price_variation: Minimum price CV for an estimate.

    Returns:
        DataFrame validated against ELASTICITY_STATUS (one row per SKU).

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2024-01-01', periods=20, freq='W'),
        ...     'stockcode': ['A'] * 20,
        ...     'price': [10.0, 10.5, 11.0, 10.8, 10.2] * 4,
        ...     'quantity': [100, 95, 90, 92, 98] * 4,
        ... })
        >>> status = compute_elasticity_status(df, min_periods=5)
        >>> 'elasticity_status' in status.columns
        True
        >>> status['elasticity_status'].iloc[0]
        'estimated'
    """
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # VECTORIZED: Compute weekly aggregates for ALL SKUs in one groupby
    weekly = (
        df.set_index(date_col)
        .groupby([product_col, pd.Grouper(freq=freq)])
        .agg(avg_price=(price_col, "mean"), total_qty=(qty_col, "sum"))
        .dropna()
        .reset_index()
    )

    if weekly.empty:
        all_skus = df[product_col].unique()
        status_df = pd.DataFrame({
            "stockcode": all_skus,
            "elasticity_status": "unavailable",
            "n_obs": 0,
            "price_cv": np.nan,
        })
        return _finalize_status_df(status_df, elasticity_df)

    # VECTORIZED: Compute all stats per SKU using groupby
    sku_stats = weekly.groupby(product_col).agg(
        n_obs=("avg_price", "size"),
        price_mean=("avg_price", "mean"),
        price_std=("avg_price", "std"),
        price_nunique=("avg_price", "nunique"),
        qty_zeros=("total_qty", lambda x: (x == 0).sum()),
        price_zeros=("avg_price", lambda x: (x == 0).sum()),
    ).reset_index()

    sku_stats["price_cv"] = sku_stats["price_std"] / sku_stats["price_mean"]

    # VECTORIZED: Determine status for all SKUs at once
    conditions = [
        (sku_stats["n_obs"] == 0),
        (sku_stats["n_obs"] < min_periods),
        (sku_stats["qty_zeros"] > 0) | (sku_stats["price_zeros"] > 0),
        (sku_stats["price_nunique"] < _MIN_DISTINCT_PRICES),
        (sku_stats["price_cv"] < min_price_variation),
    ]
    choices = [
        "unavailable",
        "insufficient_observations",
        "unavailable",
        "insufficient_price_points",
        "insufficient_variation",
    ]

    # Default status for SKUs that pass all checks
    sku_stats["elasticity_status"] = np.select(conditions, choices, default="estimated")

    # Build rows for ALL SKUs (including those with no weekly data)
    all_skus = df[product_col].unique()
    status_rows = pd.DataFrame({"stockcode": all_skus})
    status_rows = status_rows.merge(
        sku_stats[[product_col, "elasticity_status", "n_obs", "price_cv"]],
        on=product_col,
        how="left"
    )
    status_rows["elasticity_status"] = status_rows["elasticity_status"].fillna("unavailable")
    status_rows["n_obs"] = status_rows["n_obs"].fillna(0).astype(int)
    status_rows["price_cv"] = status_rows["price_cv"]

    return _finalize_status_df(status_rows, elasticity_df)


def _finalize_status_df(
    status_df: pd.DataFrame,
    elasticity_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Finalize status DataFrame with elasticity data and confidence."""
    status_df["elasticity"] = np.nan
    status_df["confidence"] = np.nan
    status_df["r_squared"] = np.nan

    if elasticity_df is not None and not elasticity_df.empty:
        est = elasticity_df[["stockcode", "elasticity", "r_squared"]].copy()
        status_df = status_df.merge(est, on="stockcode", how="left", suffixes=("", "_est"))
        status_df["elasticity"] = status_df["elasticity"].combine_first(status_df["elasticity_est"])
        status_df["r_squared"] = status_df["r_squared"].combine_first(status_df["r_squared_est"])
        status_df = status_df.drop(columns=["elasticity_est", "r_squared_est"])

        conf = classify_elasticity_confidence(elasticity_df)[["stockcode", "confidence"]]
        status_df = status_df.merge(conf, on="stockcode", how="left", suffixes=("", "_conf"))
        status_df["confidence"] = status_df["confidence"].combine_first(status_df["confidence_conf"])
        status_df = status_df.drop(columns=["confidence_conf"])

        has_estimate = status_df["elasticity"].notna()
        refine = status_df["elasticity_status"].eq("estimated")
        status_df.loc[refine & has_estimate, "elasticity_status"] = "estimated"
        status_df.loc[refine & has_estimate & status_df["confidence"].eq("low"), "elasticity_status"] = "weak"
        status_df.loc[refine & ~has_estimate, "elasticity_status"] = "unavailable"
        status_df.loc[~has_estimate, "confidence"] = np.nan

    table = status_df[list(ELASTICITY_STATUS.columns)]
    return check(table, ELASTICITY_STATUS)


def estimate_hierarchical_elasticity(
    transactions_df: pd.DataFrame,
    category_col: str = "category",
    freq: str = "W",
    min_periods: int = 10,
    min_price_variation: float = 0.05,
) -> pd.DataFrame:
    """Empirical Bayes / partial pooling elasticity with James-Stein shrinkage.

    WARNING: This estimates OBSERVED price response, NOT causal elasticity.
    - Endogeneity: price and quantity are simultaneously determined
    - Shrinkage is toward category mean, does not address endogeneity
    - Results are descriptive: "how quantity co-varies with price historically"
    - For causal inference, use IV, RDD, or experimental methods (with valid instruments)

    Individual SKU elasticities are shrunk toward their category mean using
    variance-weighted shrinkage: weight = between_var / (within_var + between_var).
    """
    import warnings

    warnings.warn(
        "estimate_hierarchical_elasticity estimates OBSERVED price response, NOT causal elasticity. "
        "Price endogeneity is not addressed. Results are descriptive only.",
        UserWarning,
        stacklevel=2,
    )
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

        # Validate data before log transformation
        zero_prices = (weekly["avg_price"] == 0).sum()
        zero_qty = (weekly["total_qty"] == 0).sum()

        if zero_prices > 0 or zero_qty > 0:
            continue  # Skip products with zero prices/quantities

        log_price = np.log(weekly["avg_price"])
        log_qty = np.log(weekly["total_qty"])

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
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
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
        return check(
            pd.DataFrame(columns=list(HIERARCHICAL_ELASTICITY.columns)),
            HIERARCHICAL_ELASTICITY,
            allow_empty=True,
        )

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

    WARNING: This estimates OBSERVED cross-price response, NOT causal elasticity.
    - Endogeneity: prices and quantities are simultaneously determined
    - No instruments for prices; OLS is biased if demand/supply shocks correlate
    - Results are descriptive: "how quantity of A co-varies with price of B historically"
    - Positive cross_elasticity suggests substitutes; negative suggests complements
    - For causal inference, use IV or experimental methods

    log(qty_A) = alpha + beta_own * log(price_A) + beta_cross * log(price_B)
    """
    import warnings

    warnings.warn(
        "estimate_cross_price_elasticity estimates OBSERVED cross-price response, NOT causal elasticity. "
        "Price endogeneity is not addressed. Results are descriptive only.",
        UserWarning,
        stacklevel=2,
    )
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

        # Validate data before log transformation
        if (weekly[["avg_price_a", "avg_price_b", "total_qty_a"]] == 0).any().any():
            continue  # Skip products with zero prices/quantities

        log_price_a = np.log(weekly["avg_price_a"])
        log_price_b = np.log(weekly["avg_price_b"])
        log_qty_a = np.log(weekly["total_qty_a"])

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

        if (
            log_price_a.nunique() < _MIN_DISTINCT_PRICES
            or log_price_b.nunique() < _MIN_DISTINCT_PRICES
        ):
            continue

        # Create time fixed effects
        time_dummies = None
        if len(weekly) > 1:
            # Use isocalendar().week for pandas >= 2.0 compatibility
            if hasattr(weekly.index, "isocalendar"):
                week_values = weekly.index.isocalendar().week
            else:
                week_values = weekly.index.week
            # get_dummies yields bool columns regardless of the source dtype,
            # which breaks sm.OLS, and month values lose the datetime index;
            # cast to float64 and restore the index before combining.
            week_dummies = pd.get_dummies(week_values, prefix="week", drop_first=True)
            week_dummies.index = weekly.index
            month_dummies = pd.get_dummies(weekly.index.month, prefix="month", drop_first=True)
            month_dummies.index = weekly.index
            time_dummies = pd.concat([week_dummies, month_dummies], axis=1).astype(float)
            time_dummies = time_dummies.loc[log_price_a.index]

        X = np.column_stack([log_price_a.values, log_price_b.values])
        X = sm.add_constant(X)
        if time_dummies is not None:
            X = np.column_stack([X, time_dummies.values])
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
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
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
        return check(
            pd.DataFrame(columns=list(CROSS_ELASTICITY.columns)), CROSS_ELASTICITY, allow_empty=True
        )

    table = pd.DataFrame(results, columns=list(CROSS_ELASTICITY.columns))
    return check(table, CROSS_ELASTICITY)
