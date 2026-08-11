"""Causal & econometric methods: IV/2SLS, RDD, Synthetic Control, T/S uplift."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.analytics.schemas import (
    CAUSAL_UPLIFT,
    IV_ELASTICITY_EXTERNAL,
    LOCAL_PRICE_RESPONSE,
    SYNTHETIC_CONTROL_ELASTICITY,
    check,
)


def _estimate_propensity_score(X: pd.DataFrame, treatment: pd.Series) -> pd.Series:
    """P(T=1|X) via logistic regression on standardized features."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X, treatment)
    return pd.Series(model.predict_proba(X)[:, 1], index=X.index)


def _check_propensity_overlap(
    propensity: pd.Series,
    treatment: pd.Series,
    min_overlap: float = 0.1,
) -> dict:
    """Overlap diagnostics: ranges, overlap proportion, warnings."""
    treated_ps = propensity[treatment == 1]
    control_ps = propensity[treatment == 0]
    if len(treated_ps) == 0 or len(control_ps) == 0:
        return {"overlap": False, "overlap_proportion": 0.0, "warnings": ["No treated or control units"]}
    treated_range = (treated_ps.min(), treated_ps.max())
    control_range = (control_ps.min(), control_ps.max())
    lower, upper = max(treated_range[0], control_range[0]), min(treated_range[1], control_range[1])
    if lower >= upper:
        overlap = 0.0
    else:
        treated_in = ((treated_ps >= lower) & (treated_ps <= upper)).mean()
        control_in = ((control_ps >= lower) & (control_ps <= upper)).mean()
        overlap = float(min(treated_in, control_in))
    return {"overlap": overlap > min_overlap, "overlap_proportion": overlap, "warnings": []}


from sklearn.pipeline import make_pipeline


def iv_elasticity_manual_2sls(
    transactions_df: pd.DataFrame,
    instrument_col: str,
    price_col: str = "price",
    qty_col: str = "quantity",
    date_col: str = "date",
    freq: str = "W",
    min_periods: int = 10,
    min_price_variation: float = 0.05,
    product_col: str = "stockcode",
) -> pd.DataFrame:
    """Two-Stage Least Squares IV elasticity using a MANUAL 2SLS procedure.

    WARNING: This is a manual 2SLS implementation, NOT a full IV inference framework.
    - Requires a valid external instrument (e.g., cost shifter, tax change, exchange rate shock)
    - Standard POS schema does NOT include cost or other valid instruments
    - Standard errors are HC3 (heteroskedasticity-robust) but NOT proper 2SLS SEs
    - First-stage F-statistic is reported; weak instruments (F < 10) are flagged
    - Results should be interpreted as descriptive, NOT causal inference

    Stage 1: log(price) ~ log(instrument)
    Stage 2: log(qty) ~ log(price_hat)
    """
    import warnings
    warnings.warn(
        "iv_elasticity_manual_2sls uses manual 2SLS; standard errors are NOT proper 2SLS SEs. "
        "Results are descriptive, not causal inference. Requires valid external instrument.",
        UserWarning,
        stacklevel=2
    )
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["revenue"] = df[price_col] * df[qty_col]

    results: list[dict[str, float | int | str | bool]] = []

    for product_id in df[product_col].unique():
        prod_df = df[df[product_col] == product_id].copy()

        weekly = (
            prod_df.set_index(date_col)
            .groupby(pd.Grouper(freq=freq))
            .agg(
                avg_price=(price_col, "mean"),
                total_qty=(qty_col, "sum"),
                avg_instrument=(instrument_col, "mean"),
            )
            .dropna()
        )

        if len(weekly) < min_periods:
            continue

        price_cv = weekly["avg_price"].std() / weekly["avg_price"].mean()
        instr_cv = weekly["avg_instrument"].std() / weekly["avg_instrument"].mean()
        if price_cv < min_price_variation or instr_cv < min_price_variation:
            continue

        # Validate data before log transformation
        if (weekly[["avg_price", "total_qty", "avg_instrument"]] == 0).any().any():
            continue  # Skip products with zero values

        log_price = np.log(weekly["avg_price"])
        log_qty = np.log(weekly["total_qty"])
        log_instr = np.log(weekly["avg_instrument"])

        if len(log_price) < min_periods:
            continue

        # Stage 1
        X1 = sm.add_constant(log_instr)
        try:
            model1 = sm.OLS(log_price, X1).fit()
            log_price_hat = model1.predict(X1)
            f_stat = float(model1.fvalue)
        except Exception:
            continue

        # Stage 2
        X2 = sm.add_constant(log_price_hat)
        try:
            model2 = sm.OLS(log_qty, X2).fit(cov_type="HC3")
            elasticity = float(model2.params[1])
            std_err = float(model2.bse[1])
            p_value = float(model2.pvalues[1])
            r2 = float(model2.rsquared)
        except Exception:
            continue

        weak_instrument = f_stat < 10

        results.append(
            {
                "stockcode": product_id,
                "iv_elasticity": elasticity,
                "iv_elasticity_se": std_err,
                "iv_elasticity_p": p_value,
                "iv_r_squared": r2,
                "first_stage_f": f_stat,
                "weak_instrument": weak_instrument,
                "n_obs": len(log_price),
                "avg_price": float(weekly["avg_price"].mean()),
                "avg_weekly_qty": float(weekly["total_qty"].mean()),
                "avg_instrument": float(weekly["avg_instrument"].mean()),
            }
        )

    if not results:
        return check(pd.DataFrame(columns=list(IV_ELASTICITY_EXTERNAL.columns)), IV_ELASTICITY_EXTERNAL, allow_empty=True)

    table = pd.DataFrame(results, columns=list(IV_ELASTICITY_EXTERNAL.columns))
    return check(table, IV_ELASTICITY_EXTERNAL)


def local_price_response(
    transactions_df: pd.DataFrame,
    price_col: str = "price",
    qty_col: str = "quantity",
    date_col: str = "date",
    product_col: str = "stockcode",
    threshold_price: Optional[float | List[float]] = None,
    bandwidth: float = 0.5,
    kernel: str = "triangular",
    freq: str = "W",
    min_periods: int = 10,
) -> pd.DataFrame:
    """Local price response around psychological price thresholds (NOT a true RDD).

    THIS IS NOT A REGRESSION DISCONTINUITY DESIGN (RDD). It is a local weighted regression
    around psychological price points with the following limitations:
    - No treatment assignment rule changes discontinuously at the threshold
    - No local randomization assumption
    - Just a descriptive local price elasticity near round prices
    - Standard errors are NOT valid RDD inference
    - No placebo tests, no bandwidth sensitivity analysis

    USE FOR EXPLORATORY ANALYSIS ONLY.
    """
    import warnings
    warnings.warn(
        "local_price_response is NOT a valid RDD. It is a descriptive local regression "
        "around psychological price points. Results are NOT causal estimates.",
        UserWarning,
        stacklevel=2
    )
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["revenue"] = df[price_col] * df[qty_col]

    if threshold_price is None:
        threshold_price = [0.99, 1.99, 2.99, 4.99, 9.99, 19.99, 49.99, 99.99]
    elif isinstance(threshold_price, float):
        threshold_price = [threshold_price]

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

        # Validate data before log transformation
        if (weekly[["avg_price", "total_qty"]] == 0).any().any():
            continue  # Skip products with zero values

        log_price = np.log(weekly["avg_price"])
        log_qty = np.log(weekly["total_qty"])

        if len(log_price) < min_periods:
            continue

        price_vals = weekly["avg_price"].values

        for thresh in threshold_price:
            in_window = (price_vals >= thresh - bandwidth) & (price_vals <= thresh + bandwidth)
            if in_window.sum() < min_periods:
                continue

            x = price_vals[in_window] - thresh
            y = log_qty[in_window]

            # Kernel weights
            if kernel == "triangular":
                weights = 1 - np.abs(x) / bandwidth
            elif kernel == "epanechnikov":
                weights = 0.75 * np.maximum(0, 1 - (x / bandwidth) ** 2)
            else:
                weights = np.ones_like(x)

            # Weighted regression
            X = np.column_stack([np.ones_like(x), x])
            W = np.diag(weights)

            try:
                beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
                elasticity = float(beta[1])
            except Exception:
                continue

            results.append(
                {
                    "product_a": product_id,
                    "product_b": "RDD_threshold",
                    "threshold_price": thresh,
                    "cross_elasticity": elasticity,
                    "n_obs": int(in_window.sum()),
                    "bandwidth": bandwidth,
                }
            )

    if not results:
        return check(pd.DataFrame(columns=list(LOCAL_PRICE_RESPONSE.columns)), LOCAL_PRICE_RESPONSE, allow_empty=True)

    table = pd.DataFrame(results, columns=list(LOCAL_PRICE_RESPONSE.columns))
    return check(table, LOCAL_PRICE_RESPONSE)


def synthetic_control_estimate(
    transactions_df: pd.DataFrame,
    treatment_product: str,
    donor_products: List[str],
    price_col: str = "price",
    qty_col: str = "quantity",
    date_col: str = "date",
    freq: str = "W",
    pre_periods: int = 20,
    post_periods: int = 10,
    product_col: str = "stockcode",
    promo_col: str | None = "promo_flag",
) -> pd.DataFrame:
    """Synthetic Control Method: weights donors to match pre-treatment trajectory.

    WARNING: This is a basic synthetic control implementation with limitations:
    - Promo periods are masked before fitting (if promo_col provided)
    - NO placebo tests are performed
    - NO permutation inference
    - Standard errors are NOT provided
    - Weights are chosen to minimize pre-period RMSE without regularization

    USE FOR EXPLORATORY ANALYSIS ONLY. NOT FOR CAUSAL INFERENCE.
    """
    import warnings
    warnings.warn(
        "synthetic_control_estimate is a basic implementation without placebo tests, "
        "permutation inference, or proper standard errors. Results are NOT causal estimates.",
        UserWarning,
        stacklevel=2
    )
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["revenue"] = df[price_col] * df[qty_col]

    weekly_all = (
        df.set_index(date_col)
        .groupby([product_col, pd.Grouper(freq="W")])
        .agg(
            avg_price=(price_col, "mean"),
            total_qty=(qty_col, "sum"),
            has_promo=(promo_col, "max") if promo_col and promo_col in df.columns else (price_col, "first")
        )
        .dropna()
        .reset_index()
    )
    if date_col != "date" and date_col in weekly_all.columns:
        weekly_all = weekly_all.rename(columns={date_col: "date"})

    products = [treatment_product] + donor_products
    weekly = weekly_all[weekly_all[product_col].isin(products)].copy()

    # Mask promo periods before fitting
    if "has_promo" in weekly.columns:
        weekly = weekly[weekly["has_promo"] == 0].copy()

    weekly["log_price"] = np.log(weekly["avg_price"].clip(lower=1e-6))
    weekly["log_qty"] = np.log(weekly["total_qty"].clip(lower=1e-6))

    all_dates = sorted(weekly["date"].unique())
    cutoff_idx = len(all_dates) - post_periods
    if cutoff_idx < pre_periods:
        raise ValueError("Not enough data for synthetic control")

    pre_dates = all_dates[:cutoff_idx]
    post_dates = all_dates[cutoff_idx:]

    pre_data = weekly[weekly["date"].isin(pre_dates)]
    post_data = weekly[weekly["date"].isin(post_dates)]

    donor_matrix = (
        pre_data[pre_data[product_col].isin(donor_products)]
        .pivot_table(index=product_col, columns="date", values="log_qty", aggfunc="mean")
        .ffill(axis=1)
        .bfill(axis=1)
    )

    treatment_pre = pre_data[pre_data[product_col] == treatment_product].set_index("date")["log_qty"]
    treatment_pre = treatment_pre.reindex(pre_dates).ffill().bfill()

    n_donors = len(donor_matrix)
    if n_donors == 0:
        raise ValueError("No donor products available")

    def objective(w: np.ndarray) -> float:
        synth = donor_matrix.T @ w
        return float(np.sum((synth - treatment_pre.values) ** 2))

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0.0, 1.0)] * n_donors
    w0 = np.ones(n_donors) / n_donors

    res = minimize(objective, w0, bounds=bounds, constraints=constraints, method='SLSQP')
    weights = res.x if res.success else w0

    # Post-period synthetic control
    post_donor = post_data[post_data[product_col].isin(donor_products)].pivot_table(
        index=product_col, columns="date", values="log_qty", aggfunc="mean"
    ).ffill(axis=1).bfill(axis=1)

    synthetic_post = post_donor.T @ weights
    treatment_post = post_data[post_data[product_col] == treatment_product].set_index("date")["log_qty"]
    treatment_post = treatment_post.reindex(post_dates).ffill().bfill()

    effect = float(np.mean(treatment_post - synthetic_post))
    effect_pct = float(np.mean((np.exp(treatment_post) - np.exp(synthetic_post)) / np.exp(synthetic_post)))

    # Pre-period fit diagnostics
    pre_synth = donor_matrix.T @ weights
    pre_rmse = float(np.sqrt(np.mean((pre_synth - treatment_pre.values) ** 2)))

    rows = [
        {"metric": "treatment_effect_log", "value": effect},
        {"metric": "treatment_effect_pct", "value": effect_pct},
        {"metric": "pre_period_rmse", "value": pre_rmse},
        {"metric": "n_donors", "value": float(n_donors)},
    ]
    for i, d in enumerate(donor_matrix.index):
        rows.append({"metric": f"weight_{d}", "value": float(weights[i])})

    table = pd.DataFrame(rows, columns=list(SYNTHETIC_CONTROL_ELASTICITY.columns))
    return check(table, SYNTHETIC_CONTROL_ELASTICITY)


def causal_uplift_t_s(
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    learner: str = "s_learner",
    model_type: str = "rf",
    random_seed: int = 42,
) -> pd.DataFrame:
    """T-learner or S-learner causal uplift.

    T-learner: two separate models for treatment/control, uplift = mu1 - mu0
    S-learner: single model with treatment as feature, uplift = mu(t=1) - mu(t=0)

    Args:
        X: Feature matrix
        treatment: Binary treatment indicator
        outcome: Outcome variable
        learner: 't_learner' or 's_learner'
        model_type: 'rf' (RandomForest) or 'hgb' (HistGradientBoosting)
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with customer_id, uplift, treatment, propensity
    """
    if len(X) != len(treatment) or len(X) != len(outcome):
        raise ValueError("X, treatment, outcome must have same length")

    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import train_test_split

    # Propensity score
    propensity = _estimate_propensity_score(X, treatment)
    overlap_diag = _check_propensity_overlap(propensity, treatment)
    if not overlap_diag["overlap"]:
        raise ValueError(f"Propensity overlap insufficient: {overlap_diag['warnings']}")

    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
        X, treatment, outcome, test_size=0.3, random_state=random_seed, stratify=treatment
    )

    if model_type == "rf":
        base_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=random_seed)
    else:
        base_model = HistGradientBoostingRegressor(max_iter=200, random_state=random_seed)

    if learner == "t_learner":
        # Two separate models
        model_t = base_model.__class__(**base_model.get_params())
        model_c = base_model.__class__(**base_model.get_params())
        model_t.fit(X_train[T_train == 1], y_train[T_train == 1])
        model_c.fit(X_train[T_train == 0], y_train[T_train == 0])
        uplift = model_t.predict(X_test) - model_c.predict(X_test)
    else:
        # S-learner: single model with treatment feature
        X_train_s = X_train.copy()
        X_train_s["treatment"] = T_train
        X_test_s = X_test.copy()
        X_test_s["treatment"] = T_test
        model = base_model.__class__(**base_model.get_params())
        model.fit(X_train_s, y_train)
        X_test_s["treatment"] = 1
        mu1 = model.predict(X_test_s)
        X_test_s["treatment"] = 0
        mu0 = model.predict(X_test_s)
        uplift = mu1 - mu0

    result = pd.DataFrame(
        {
            "customer_id": X_test.index,
            "uplift": uplift,
            "treatment": T_test.values,
            "propensity": propensity.loc[X_test.index].values,
        }
    )
    return check(result, CAUSAL_UPLIFT)
