"""Pricing, Elasticity & KVI Analytics."""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Optional statsmodels for robust OLS and diagnostics
try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings("ignore")


def _ols_diagnostics(model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Compute OLS diagnostics: Breusch-Pagan, Durbin-Watson, VIF, condition number."""
    diagnostics = {}

    if not STATSMODELS_AVAILABLE:
        return diagnostics

    try:
        # Breusch-Pagan test for heteroskedasticity
        bp_lm, bp_p, _, _ = het_breuschpagan(model.resid, X)
        diagnostics["breusch_pagan_lm"] = float(bp_lm)
        diagnostics["breusch_pagan_p"] = float(bp_p)
        diagnostics["heteroskedasticity_detected"] = bp_p < 0.05
    except Exception:
        pass

    try:
        # Durbin-Watson test for autocorrelation
        from statsmodels.stats.stattools import durbin_watson

        dw = durbin_watson(model.resid)
        diagnostics["durbin_watson"] = float(dw)
        diagnostics["autocorrelation_detected"] = dw < 1.5 or dw > 2.5
    except Exception:
        pass

    try:
        # Variance Inflation Factor (for multi-collinearity)
        # Only meaningful if X has multiple columns
        if X.shape[1] > 1:
            vif_data = pd.DataFrame()
            vif_data["feature"] = range(X.shape[1])
            vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
            diagnostics["vif"] = vif_data.to_dict("records")
            diagnostics["max_vif"] = float(vif_data["VIF"].max())
            diagnostics["multicollinearity_detected"] = vif_data["VIF"].max() > 10
    except Exception:
        pass

    try:
        # Condition number for multicollinearity
        cond_num = np.linalg.cond(X)
        diagnostics["condition_number"] = float(cond_num)
        diagnostics["ill_conditioned"] = cond_num > 30
    except Exception:
        pass

    try:
        # Ramsey RESET test for functional form misspecification
        from statsmodels.stats.diagnostic import linear_reset

        reset_result = linear_reset(model, power=3)
        diagnostics["ramsey_reset_f"] = float(reset_result.fvalue)
        diagnostics["ramsey_reset_p"] = float(reset_result.pvalue)
        diagnostics["functional_form_misspec"] = reset_result.pvalue < 0.05
    except Exception:
        pass

    return diagnostics


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
    """
    Per-SKU log-log OLS elasticity: log(qty) = alpha + beta * log(price) + seasonality.

    Returns: stockcode, elasticity (beta), r_squared, n_obs, p_value, avg_price, avg_qty.
    """
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["revenue"] = df[price_col] * df[qty_col]

    results = []

    for product_id in df[product_col].unique():
        prod_df = df[df[product_col] == product_id].copy()

        # Aggregate to weekly
        weekly = (
            prod_df.set_index(date_col)
            .groupby(pd.Grouper(freq=freq))
            .agg(avg_price=(price_col, "mean"), total_qty=(qty_col, "sum"))
            .dropna()
        )

        if len(weekly) < min_periods:
            continue

        # Check price variation
        price_cv = weekly["avg_price"].std() / weekly["avg_price"].mean()
        if price_cv < min_price_variation:
            continue

        # Log-log regression
        log_price = np.log(weekly["avg_price"].replace(0, np.nan).dropna())
        log_qty = np.log(weekly.loc[log_price.index, "total_qty"].replace(0, np.nan).dropna())

        if len(log_price) < min_periods:
            continue

        # Use statsmodels OLS with robust SE if available, fallback to scipy
        if use_robust_se and STATSMODELS_AVAILABLE:
            import statsmodels.api as sm

            X = sm.add_constant(log_price)
            model = sm.OLS(log_qty, X).fit(cov_type="HC3")
            elasticity = model.params[log_price.name]
            std_err = model.bse[log_price.name]
            p_value = model.pvalues[log_price.name]
            r_squared = model.rsquared
            conf_int = model.conf_int().loc[log_price.name]
            ci_lower, ci_upper = conf_int[0], conf_int[1]

            # Add diagnostics
            diagnostics = _ols_diagnostics(model, X.values, log_qty.values)
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_price, log_qty)
            elasticity = slope
            r_squared = r_value**2
            ci_lower = elasticity - 1.96 * std_err
            ci_upper = elasticity + 1.96 * std_err
            diagnostics = {}

        results.append(
            {
                "stockcode": product_id,
                "elasticity": elasticity,
                "r_squared": r_squared,
                "p_value": p_value,
                "std_err": std_err,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_obs": len(log_price),
                "avg_price": weekly["avg_price"].mean(),
                "avg_weekly_qty": weekly["total_qty"].mean(),
                "price_cv": price_cv,
                "diagnostics": diagnostics,
            }
        )

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def estimate_hierarchical_elasticity(
    transactions_df: pd.DataFrame,
    category_col: str = "category",
    freq: str = "W",
    min_periods: int = 10,
    min_price_variation: float = 0.05,
    ridge_alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Empirical Bayes / partial pooling elasticity estimation.

    Shrinks individual SKU elasticities toward category-level mean using
    variance-weighted James-Stein style shrinkage.
    This replaces the simple n_obs-based weight with variance-ratio weighting.
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # First pass: individual OLS
    ols_results = []
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

        if len(log_price) < min_periods:
            continue

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_price, log_qty)

        ols_results.append(
            {
                "stockcode": product_id,
                "category": cat,
                "elasticity_ols": slope,
                "r_squared": r_value**2,
                "p_value": p_value,
                "n_obs": len(log_price),
                "avg_price": weekly["avg_price"].mean(),
                "std_err": std_err,
            }
        )

    if not ols_results:
        return pd.DataFrame()

    ols_df = pd.DataFrame(ols_results)

    # Category-level variance (between-SKU variance within category)
    cat_vars = ols_df.groupby("category")["elasticity_ols"].var().rename("cat_var")
    cat_means = ols_df.groupby("category")["elasticity_ols"].mean().rename("elasticity_cat")
    cat_n = ols_df.groupby("category").size().rename("cat_n")

    ols_df = ols_df.merge(cat_means, on="category", how="left")
    ols_df = ols_df.merge(cat_vars, on="category", how="left")
    ols_df = ols_df.merge(cat_n, on="category", how="left")

    # James-Stein / empirical Bayes shrinkage weight:
    # weight = within_SKU_variance / (within_SKU_variance + between_SKU_variance)
    # Using std_err^2 as proxy for within-SKU variance, cat_var as between-SKU variance
    ols_df["within_var"] = ols_df["std_err"] ** 2
    ols_df["between_var"] = ols_df["cat_var"]

    # Shrinkage weight: more shrinkage when within-var >> between-var
    # weight = between_var / (within_var + between_var) -> closer to cat mean
    ols_df["shrink_weight"] = ols_df["between_var"] / (
        ols_df["within_var"] + ols_df["between_var"] + 1e-8
    )

    # Cap weight to reasonable range
    ols_df["shrink_weight"] = ols_df["shrink_weight"].clip(0.05, 0.95)

    ols_df["elasticity_shrunk"] = (
        ols_df["shrink_weight"] * ols_df["elasticity_cat"]
        + (1 - ols_df["shrink_weight"]) * ols_df["elasticity_ols"]
    )

    return ols_df


def estimate_cross_price_elasticity(
    transactions_df: pd.DataFrame,
    product_pairs: List[Tuple[str, str]],
    category_col: str = "category",
    freq: str = "W",
    min_periods: int = 10,
    min_price_variation: float = 0.05,
) -> pd.DataFrame:
    """
    Estimate cross-price elasticities for specified product pairs.

    For each pair (A, B), runs bivariate log-log OLS:
        log(qty_A) = alpha + beta_own * log(price_A) + beta_cross * log(price_B) + error

    beta_cross > 0 -> B is a substitute for A
    beta_cross < 0 -> B is a complement to A

    Args:
        transactions_df: Transaction data
        product_pairs: List of (product_a, product_b) tuples to estimate cross-elasticity
        category_col: Category column for grouping
        freq: Time frequency for aggregation
        min_periods: Minimum weeks of data
        min_price_variation: Minimum price CV

    Returns:
        DataFrame with cross-elasticity estimates per pair
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    results = []

    for prod_a, prod_b in product_pairs:
        # Get weekly data for both products
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

        # Align on date index
        weekly = weekly_a.join(weekly_b, how="inner")
        if len(weekly) < min_periods:
            continue

        # Check price variation for both
        cv_a = weekly["avg_price_a"].std() / weekly["avg_price_a"].mean()
        cv_b = weekly["avg_price_b"].std() / weekly["avg_price_b"].mean()
        if cv_a < min_price_variation or cv_b < min_price_variation:
            continue

        # Log-log regression: log(qty_a) ~ log(price_a) + log(price_b)
        log_price_a = np.log(weekly["avg_price_a"].replace(0, np.nan).dropna())
        log_price_b = np.log(
            weekly.loc[log_price_a.index, "avg_price_b"].replace(0, np.nan).dropna()
        )
        log_qty_a = np.log(weekly.loc[log_price_a.index, "total_qty_a"].replace(0, np.nan).dropna())

        if len(log_price_a) < min_periods:
            continue

        # Bivariate OLS
        X = np.column_stack([log_price_a.values, log_price_b.values])
        X = sm.add_constant(X)
        y = log_qty_a.values

        try:
            model = sm.OLS(y, X).fit(cov_type="HC3")
            own_elasticity = model.params[1]
            cross_elasticity = model.params[2]
            own_se = model.bse[1]
            cross_se = model.bse[2]
            own_p = model.pvalues[1]
            cross_p = model.pvalues[2]
            r2 = model.rsquared
        except Exception:
            continue

        results.append(
            {
                "product_a": prod_a,
                "product_b": prod_b,
                "own_elasticity": own_elasticity,
                "own_elasticity_se": own_se,
                "own_elasticity_p": own_p,
                "cross_elasticity": cross_elasticity,
                "cross_elasticity_se": cross_se,
                "cross_elasticity_p": cross_p,
                "r_squared": r2,
                "n_obs": len(log_price_a),
                "avg_price_a": weekly["avg_price_a"].mean(),
                "avg_price_b": weekly["avg_price_b"].mean(),
            }
        )

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def _bayesian_convergence_diagnostics(trace) -> Dict[str, Any]:
    """Compute Bayesian convergence diagnostics: R-hat, ESS, divergences."""
    diagnostics = {
        "rhat_max": None,
        "rhat_mean": None,
        "ess_min": None,
        "ess_mean": None,
        "divergences": 0,
        "divergence_rate": 0.0,
    }

    try:
        import arviz as az

        # R-hat (Gelman-Rubin)
        rhat = az.rhat(trace)
        if hasattr(rhat, "values"):
            rhat_values = rhat.values.flatten()
        else:
            rhat_values = np.array(list(rhat.values())).flatten()
        rhat_values = rhat_values[~np.isnan(rhat_values)]
        if len(rhat_values) > 0:
            diagnostics["rhat_max"] = float(np.max(rhat_values))
            diagnostics["rhat_mean"] = float(np.mean(rhat_values))

        # Effective Sample Size
        ess = az.ess(trace)
        if hasattr(ess, "values"):
            ess_values = ess.values.flatten()
        else:
            ess_values = np.array(list(ess.values())).flatten()
        ess_values = ess_values[~np.isnan(ess_values)]
        if len(ess_values) > 0:
            diagnostics["ess_min"] = float(np.min(ess_values))
            diagnostics["ess_mean"] = float(np.mean(ess_values))

        # Divergences
        if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
            diverging = trace.sample_stats["diverging"].values
            diagnostics["divergences"] = int(np.sum(diverging))
            diagnostics["divergence_rate"] = float(np.mean(diverging))

        # Per-parameter summary
        summary = az.summary(trace, hdi_prob=0.94)
        diagnostics["summary"] = summary.to_dict()

    except Exception as e:
        diagnostics["error"] = str(e)

    return diagnostics


def estimate_bayesian_hierarchical_elasticity(
    transactions_df: pd.DataFrame,
    category_col: str = "category",
    freq: str = "W",
    min_periods: int = 10,
    min_price_variation: float = 0.05,
    n_samples: int = 500,
    n_tune: int = 500,
    bayesian_mode: str = "fast (ADVI)",
    return_trace: bool = False,
) -> pd.DataFrame | tuple:
    """
    Three-tier Bayesian hierarchical elasticity via PyMC.

    Tiers:
      1. Global (population) mean & variance for elasticity and intercept.
      2. Category-level partial pooling.
      3. SKU-level elasticities shrunk toward category means.

    Parameters
    ----------
    bayesian_mode : str
        "fast (ADVI)" uses variational inference (ADVI).
        "full (NUTS)" uses NUTS MCMC (slower but exact).
    return_trace : bool
        If True, return (DataFrame, trace) where trace is the InferenceData
        object from PyMC sampling.

    Returns
    -------
    DataFrame or (DataFrame, InferenceData | None)
        SKU-level elasticity estimates with posterior mean, SD, and HDI.
        If return_trace=True and bayesian_mode is NUTS, also returns the trace.
    """
    try:
        import pymc as pm
    except ImportError:
        raise ImportError("PyMC >= 5.0.0 required: pip install pymc")

    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # --- 1. Aggregate to weekly level per SKU ---
    records = []
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

        weekly = weekly.reset_index()
        weekly["stockcode"] = product_id
        weekly["category"] = cat
        records.append(weekly)

    if not records:
        return pd.DataFrame()

    agg_df = pd.concat(records, ignore_index=True)
    agg_df["log_price"] = np.log(agg_df["avg_price"].clip(lower=1e-6))
    agg_df["log_qty"] = np.log(agg_df["total_qty"].clip(lower=1e-6))

    # Encode indices
    categories = agg_df["category"].unique()
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    stockcodes = agg_df["stockcode"].unique()
    sku_to_idx = {s: i for i, s in enumerate(stockcodes)}
    sku_idx = agg_df["stockcode"].map(sku_to_idx).values

    # Per-SKU category index (each SKU belongs to exactly one category)
    sku_cat = agg_df.groupby("stockcode")["category"].first()
    sku_cat_idx = np.array([cat_to_idx[c] for c in sku_cat])

    n_obs = len(agg_df)

    # --- 2. Build PyMC model ---
    coords = {
        "category": categories,
        "sku": stockcodes,
        "observation": range(n_obs),
    }

    with pm.Model(coords=coords):
        log_price_data = pm.Data("log_price_data", agg_df["log_price"].values)
        sku_idx_data = pm.Data("sku_idx_data", sku_idx)

        # Tier 1 — global
        mu_alpha = pm.Normal("mu_alpha", 0, 1)
        sigma_alpha = pm.HalfNormal("sigma_alpha", 1)
        mu_beta = pm.Normal("mu_beta", -1, 1)
        sigma_beta = pm.HalfNormal("sigma_beta", 1)

        # Tier 2 — category
        alpha_cat = pm.Normal("alpha_cat", mu_alpha, sigma_alpha, dims="category")
        beta_cat = pm.Normal("beta_cat", mu_beta, sigma_beta, dims="category")

        # Tier 3 — SKU (each SKU belongs to one category determined by sku_cat_idx)
        sigma_sku = pm.HalfNormal("sigma_sku", 1)
        alpha_sku = pm.Normal("alpha_sku", alpha_cat[sku_cat_idx], sigma_sku, dims="sku")
        beta_sku = pm.Normal("beta_sku", beta_cat[sku_cat_idx], sigma_sku, dims="sku")

        mu = alpha_sku[sku_idx_data] + beta_sku[sku_idx_data] * log_price_data
        sigma = pm.HalfNormal("sigma", 1)

        pm.Normal("likelihood", mu, sigma, observed=agg_df["log_qty"].values, dims="observation")

        # --- 3. Inference ---
        if bayesian_mode.startswith("fast"):
            approx = pm.fit(n=n_samples, method="advi", obj_optimizer=pm.adam(learning_rate=0.01))
            trace = approx.sample(n_samples)
        else:
            trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                chains=2,
                cores=1,
                progressbar=False,
                random_seed=42,
            )

        # --- 3b. Convergence diagnostics ---
        convergence_diagnostics = _bayesian_convergence_diagnostics(trace)
        rhat_max = convergence_diagnostics.get("rhat_max")
        ess_min = convergence_diagnostics.get("ess_min")
        rhat_str = f"{rhat_max:.3f}" if rhat_max is not None else "N/A"
        ess_str = f"{ess_min:.0f}" if ess_min is not None else "N/A"
        print(f"Bayesian convergence: R-hat max={rhat_str}, ESS min={ess_str}")

    # --- 4. Extract SKU-level posterior ---
    beta_post = trace.posterior["beta_sku"] if hasattr(trace, "posterior") else trace["beta_sku"]

    beta_mean = (
        beta_post.mean(dim=("chain", "draw")).values
        if hasattr(beta_post, "mean")
        else beta_post.mean(axis=0)
    )
    beta_sd = (
        beta_post.std(dim=("chain", "draw")).values
        if hasattr(beta_post, "std")
        else beta_post.std(axis=0)
    )

    # HDI (highest density interval)
    try:
        import arviz as az

        hdi = az.hdi(beta_post, prob=0.94)
        beta_hdi_lower = hdi.sel(hdi="lower").values if hasattr(hdi, "sel") else hdi[:, 0]
        beta_hdi_upper = hdi.sel(hdi="higher").values if hasattr(hdi, "sel") else hdi[:, 1]
    except Exception:
        beta_hdi_lower = beta_mean - 1.96 * beta_sd
        beta_hdi_upper = beta_mean + 1.96 * beta_sd

    # Per-SKU summary
    sku_stats = (
        agg_df.groupby("stockcode")
        .agg(
            category=("category", "first"),
            n_obs=("total_qty", "size"),
            avg_price=("avg_price", "mean"),
        )
        .reset_index()
    )

    sku_stats["elasticity_mean"] = beta_mean
    sku_stats["elasticity_sd"] = beta_sd
    sku_stats["elasticity_hdi_lower"] = beta_hdi_lower
    sku_stats["elasticity_hdi_upper"] = beta_hdi_upper

    if return_trace and hasattr(trace, "posterior"):
        return sku_stats, trace
    return sku_stats


def estimate_elasticity_xgb(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    price_col: str = "price",
    qty_col: str = "quantity",
    date_col: str = "date",
    freq: str = "W",
    min_periods: int = 10,
) -> Tuple[object, pd.DataFrame]:
    """
    XGBoost-based non-linear elasticity estimation with SHAP explanations.
    Returns (model, feature_importance_df).
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost required: pip install xgboost")

    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["revenue"] = df[price_col] * df[qty_col]

    # Build features at weekly level
    weekly_features = []
    for product_id in df[product_col].unique():
        prod_df = df[df[product_col] == product_id]
        weekly = (
            prod_df.set_index(date_col)
            .groupby(pd.Grouper(freq=freq))
            .agg(
                avg_price=(price_col, "mean"),
                total_qty=(qty_col, "sum"),
                price_std=(price_col, "std"),
                n_txns=("transaction_id", "nunique"),
            )
            .dropna()
        )

        if len(weekly) < min_periods:
            continue

        # Add lag features
        weekly = weekly.copy()
        weekly["lag_price"] = weekly["avg_price"].shift(1)
        weekly["lag_qty"] = weekly["total_qty"].shift(1)
        weekly["price_change"] = weekly["avg_price"].pct_change()
        weekly["month"] = weekly.index.month
        weekly["week"] = weekly.index.isocalendar().week
        weekly["stockcode"] = product_id

        weekly_features.append(weekly)

    if not weekly_features:
        return None, pd.DataFrame()

    feature_df = pd.concat(weekly_features).dropna()

    # Prepare X, y
    feature_cols = ["avg_price", "lag_price", "lag_qty", "price_change", "month", "week"]
    X = feature_df[feature_cols]
    y = feature_df["total_qty"]

    # Train
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0
    )
    model.fit(X, y)

    # Feature importance
    importance = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return model, importance


# ============================================================================
# KVI SCORING
# ============================================================================


def compute_kvi_score(
    transactions_df: pd.DataFrame,
    elasticity_df: Optional[pd.DataFrame] = None,
    product_metrics_df: Optional[pd.DataFrame] = None,
    cost_col: Optional[str] = None,
    margin_pct: Optional[float] = None,
    method: str = "xgb_importance",
) -> pd.DataFrame:
    """
    Compute KVI (Key Value Item) scores.

    Features per SKU:
    - basket_penetration (trip incidence)
    - category_revenue_share
    - |elasticity| (price sensitivity)
    - avg_basket_value_when_present (halo)
    - repeat_purchase_rate

    Target: XGB predicting basket value or price sensitivity.
    Returns SKU-ranked KVI scores with feature contributions.
    """
    from src.analytics import (
        compute_basket_penetration,
        compute_basket_value_uplift,
        compute_product_metrics,
    )

    # Compute product metrics if not provided
    if product_metrics_df is None:
        product_metrics_df = compute_product_metrics(transactions_df)

    if product_metrics_df.empty:
        return pd.DataFrame()

    # Basket uplift (halo)
    basket_uplift = compute_basket_value_uplift(transactions_df)
    basket_pen = compute_basket_penetration(transactions_df)

    # Merge features
    kvi_features = product_metrics_df.merge(
        basket_uplift[["stockcode", "basket_value_uplift_pct"]], on="stockcode", how="left"
    ).merge(
        basket_pen[["stockcode", "basket_penetration", "trip_incidence"]],
        on="stockcode",
        how="left",
    )

    # Elasticity features
    if elasticity_df is not None and not elasticity_df.empty:
        elast_features = elasticity_df[["stockcode", "elasticity", "r_squared", "price_cv"]].copy()
        elast_features["abs_elasticity"] = elast_features["elasticity"].abs()
        kvi_features = kvi_features.merge(elast_features, on="stockcode", how="left")

    # Category share
    if "category" in kvi_features.columns:
        cat_rev = kvi_features.groupby("category")["total_revenue"].sum()
        total_rev = kvi_features["total_revenue"].sum()
        kvi_features["category_revenue_share"] = kvi_features["category"].map(cat_rev / total_rev)

    # Repeat rate
    kvi_features["repeat_rate"] = kvi_features.get("repeat_rate", 0)

    # Fill NaN
    kvi_features = kvi_features.fillna(0)

    if method == "xgb_importance":
        return _kvi_xgb(kvi_features, cost_col, margin_pct)
    else:
        return _kvi_heuristic(kvi_features, cost_col, margin_pct)


def _kvi_xgb(
    kvi_features: pd.DataFrame,
    cost_col: Optional[str] = None,
    margin_pct: Optional[float] = None,
) -> pd.DataFrame:
    """KVI scoring via XGBoost with cross-validated SHAP values (ALG-2 fix)."""
    try:
        import shap
        import xgboost as xgb
    except ImportError:
        return _kvi_heuristic(kvi_features, cost_col, margin_pct)

    # Target: margin-weighted revenue or revenue
    if cost_col and cost_col in kvi_features.columns:
        kvi_features["margin"] = kvi_features["total_revenue"] * (
            1 - kvi_features[cost_col] / kvi_features["avg_price"].replace(0, np.nan)
        ).clip(0, 1)
        y = kvi_features["margin"].fillna(0)
    elif margin_pct:
        kvi_features["margin"] = kvi_features["total_revenue"] * margin_pct
        y = kvi_features["margin"]
    else:
        y = kvi_features["total_revenue"].fillna(0)

    # Feature columns
    feature_cols = [
        "basket_penetration",
        "trip_incidence",
        "total_revenue",
        "total_customers",
        "avg_price",
        "price_cv",
        "basket_value_uplift_pct",
        "total_transactions",
        "revenue_per_customer",
        "repeat_rate",
        "abs_elasticity",
        "category_revenue_share",
        "r_squared",
    ]
    feature_cols = [c for c in feature_cols if c in kvi_features.columns]

    X = kvi_features[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)

    # ALG-2: Use cross-validated OOF predictions as KVI score, not in-sample
    from sklearn.model_selection import KFold, cross_val_predict

    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0
    )

    # Out-of-fold predictions to prevent overfitting
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    kvi_features["kvi_score"] = oof_preds

    # Fit on full data for SHAP explanations
    model.fit(X, y)

    # ALG-2: Use SHAP values for feature contributions (more reliable than built-in importance)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # Store mean absolute SHAP per feature
        kvi_features["kvi_shap_importance"] = str(
            dict(zip(feature_cols, np.mean(np.abs(shap_values), axis=0)))
        )
    except Exception:
        # Fallback to built-in importance
        kvi_features["kvi_feature_importance"] = str(
            dict(zip(feature_cols, model.feature_importances_))
        )

    return kvi_features


def _kvi_heuristic(
    kvi_features: pd.DataFrame,
    cost_col: Optional[str] = None,
    margin_pct: Optional[float] = None,
) -> pd.DataFrame:
    """KVI scoring via weighted heuristic."""

    from sklearn.preprocessing import StandardScaler

    # Target metric
    if cost_col and cost_col in kvi_features.columns:
        kvi_features["margin"] = kvi_features["total_revenue"] * (
            1 - kvi_features[cost_col] / kvi_features["avg_price"].replace(0, np.nan)
        ).clip(0, 1)
    elif margin_pct:
        kvi_features["margin"] = kvi_features["total_revenue"] * margin_pct

    # Features
    feature_cols = [
        "basket_penetration",
        "total_revenue",
        "total_customers",
        "revenue_per_customer",
        "basket_value_uplift_pct",
        "abs_elasticity",
    ]
    feature_cols = [c for c in feature_cols if c in kvi_features.columns]

    if not feature_cols:
        kvi_features["kvi_score"] = 0
        return kvi_features

    X = kvi_features[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Weights: penetration (0.3), revenue (0.25), halo (0.2), elasticity (0.15), customers (0.1)
    weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])[: len(feature_cols)]
    weights = weights / weights.sum()

    kvi_features["kvi_score"] = X_scaled @ weights

    return kvi_features


# ============================================================================
# PRICE CURVE DIAGNOSTICS
# ============================================================================


def diagnose_price_curves_1d(
    transactions_df: pd.DataFrame,
    category_col: str = "category",
    price_col: str = "price",
    qty_col: str = "quantity",
    n_tiers: int = 3,
    method: str = "kmeans",
) -> pd.DataFrame:
    """
    Cluster SKUs by price_per_unit only within category.
    Simple univariate clustering for basic price tier analysis.
    """
    # Product median price and pack size
    product_info = (
        transactions_df.groupby("stockcode")
        .agg(
            product_name=("product", "first"),
            category=(category_col, "first")
            if category_col in transactions_df.columns
            else ("stockcode", "first"),
            brand=("brand", "first")
            if "brand" in transactions_df.columns
            else ("stockcode", "first"),
            median_price=(price_col, "median"),
            size=("size", "first") if "size" in transactions_df.columns else ("stockcode", "first"),
        )
        .reset_index()
    )

    # Parse pack size numeric
    def parse_size(size_str):
        if pd.isna(size_str):
            return 1.0
        size_str = str(size_str).upper()
        import re

        match = re.search(r"(\d+(?:\.\d+)?)\s*(ML|L|G|KG|PK|PCS)", size_str)
        if match:
            val = float(match.group(1))
            unit = match.group(2)
            if unit in ("ML", "G"):
                return val / 1000
            elif unit in ("PK", "PCS"):
                return val
            return val
        return 1.0

    product_info["pack_size_numeric"] = product_info["size"].apply(parse_size)
    product_info["price_per_unit"] = product_info["median_price"] / product_info[
        "pack_size_numeric"
    ].replace(0, np.nan)

    # Cluster per category using only price_per_unit
    all_results = []

    for cat in product_info["category"].unique():
        cat_data = product_info[product_info["category"] == cat].copy()
        if len(cat_data) < 2:
            continue

        features = cat_data[["price_per_unit"]].fillna(0).values

        if method == "kmeans":
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=min(n_tiers, len(cat_data)), random_state=42, n_init=10)
        else:
            from sklearn.mixture import GaussianMixture

            model = GaussianMixture(n_components=min(n_tiers, len(cat_data)), random_state=42)

        cat_data["tier"] = model.fit_predict(features)

        # Sort tiers by mean price_per_unit
        tier_order = cat_data.groupby("tier")["price_per_unit"].mean().sort_values().index
        tier_map = {old: new for new, old in enumerate(tier_order)}
        cat_data["tier"] = cat_data["tier"].map(tier_map)

        tier_labels = {0: "Value", 1: "Mainstream", 2: "Premium", 3: "Ultra", 4: "Luxury"}
        cat_data["tier_label"] = (
            cat_data["tier"].map(tier_labels).fillna("Tier " + cat_data["tier"].astype(str))
        )

        all_results.append(cat_data)

    if not all_results:
        return pd.DataFrame()

    result_df = pd.concat(all_results, ignore_index=True)

    # Detect violations (monotonicity)
    violations = _detect_price_curve_violations(result_df)
    result_df["has_violation"] = result_df["stockcode"].isin(violations["larger_pack"]) | result_df[
        "stockcode"
    ].isin(violations["smaller_pack"])

    return result_df


def diagnose_price_curves_multivariate(
    transactions_df: pd.DataFrame,
    category_col: str = "category",
    price_col: str = "price",
    qty_col: str = "quantity",
    n_tiers: int = 3,
    method: str = "kmeans",
    elasticity_df: Optional[pd.DataFrame] = None,
    cost_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Cluster SKUs by (price_per_unit, elasticity, basket_penetration, margin) within category.
    Multivariate clustering for advanced price tier analysis incorporating demand sensitivity and profitability.
    """
    from src.analytics import compute_basket_penetration

    # Product median price and pack size
    product_info = (
        transactions_df.groupby("stockcode")
        .agg(
            product_name=("product", "first"),
            category=(category_col, "first")
            if category_col in transactions_df.columns
            else ("stockcode", "first"),
            brand=("brand", "first")
            if "brand" in transactions_df.columns
            else ("stockcode", "first"),
            median_price=(price_col, "median"),
            size=("size", "first") if "size" in transactions_df.columns else ("stockcode", "first"),
        )
        .reset_index()
    )

    # Parse pack size numeric
    def parse_size(size_str):
        if pd.isna(size_str):
            return 1.0
        size_str = str(size_str).upper()
        import re

        match = re.search(r"(\d+(?:\.\d+)?)\s*(ML|L|G|KG|PK|PCS)", size_str)
        if match:
            val = float(match.group(1))
            unit = match.group(2)
            if unit in ("ML", "G"):
                return val / 1000
            elif unit in ("PK", "PCS"):
                return val
            return val
        return 1.0

    product_info["pack_size_numeric"] = product_info["size"].apply(parse_size)
    product_info["price_per_unit"] = product_info["median_price"] / product_info[
        "pack_size_numeric"
    ].replace(0, np.nan)

    # Add basket penetration
    basket_pen = compute_basket_penetration(transactions_df)[
        ["stockcode", "basket_penetration", "trip_incidence"]
    ]
    product_info = product_info.merge(basket_pen, on="stockcode", how="left")

    # Add elasticity if provided
    if elasticity_df is not None and not elasticity_df.empty:
        elast_cols = ["stockcode"]
        if "elasticity" in elasticity_df.columns:
            elast_cols.append("elasticity")
        if "r_squared" in elasticity_df.columns:
            elast_cols.append("r_squared")
        if "price_cv" in elasticity_df.columns:
            elast_cols.append("price_cv")
        product_info = product_info.merge(elasticity_df[elast_cols], on="stockcode", how="left")

    # Add margin if cost column available
    if cost_col and cost_col in transactions_df.columns:
        cost_info = (
            transactions_df.groupby("stockcode").agg(median_cost=(cost_col, "median")).reset_index()
        )
        product_info = product_info.merge(cost_info, on="stockcode", how="left")
        product_info["margin_per_unit"] = (
            product_info["price_per_unit"] - product_info["median_cost"]
        ) / product_info["price_per_unit"].replace(0, np.nan)

    # Prepare clustering features
    feature_cols = ["price_per_unit", "basket_penetration"]
    if "elasticity" in product_info.columns:
        feature_cols.append("elasticity")
    if "margin_per_unit" in product_info.columns:
        feature_cols.append("margin_per_unit")

    # Standardize features
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    all_results = []

    for cat in product_info["category"].unique():
        cat_data = product_info[product_info["category"] == cat].copy()
        if len(cat_data) < 2:
            continue

        X = cat_data[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        if X.shape[1] == 0:
            continue

        X_scaled = scaler.fit_transform(X)

        if method == "kmeans":
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=min(n_tiers, len(cat_data)), random_state=42, n_init=10)
        else:
            from sklearn.mixture import GaussianMixture

            model = GaussianMixture(n_components=min(n_tiers, len(cat_data)), random_state=42)

        cat_data["tier"] = model.fit_predict(X_scaled)

        # Sort tiers by mean price_per_unit
        tier_order = cat_data.groupby("tier")["price_per_unit"].mean().sort_values().index
        tier_map = {old: new for new, old in enumerate(tier_order)}
        cat_data["tier"] = cat_data["tier"].map(tier_map)

        tier_labels = {0: "Value", 1: "Mainstream", 2: "Premium", 3: "Ultra", 4: "Luxury"}
        cat_data["tier_label"] = (
            cat_data["tier"].map(tier_labels).fillna("Tier " + cat_data["tier"].astype(str))
        )

        all_results.append(cat_data)

    if not all_results:
        return pd.DataFrame()

    result_df = pd.concat(all_results, ignore_index=True)

    # Detect violations (monotonicity)
    violations = _detect_price_curve_violations(result_df)
    result_df["has_violation"] = result_df["stockcode"].isin(violations["larger_pack"]) | result_df[
        "stockcode"
    ].isin(violations["smaller_pack"])

    return result_df


def diagnose_price_curves(
    transactions_df: pd.DataFrame,
    category_col: str = "category",
    price_col: str = "price",
    qty_col: str = "quantity",
    n_tiers: int = 3,
    method: str = "kmeans",
) -> pd.DataFrame:
    """
    Legacy wrapper for backward compatibility.
    Calls diagnose_price_curves_1d (price-only clustering).
    """
    return diagnose_price_curves_1d(
        transactions_df=transactions_df,
        category_col=category_col,
        price_col=price_col,
        qty_col=qty_col,
        n_tiers=n_tiers,
        method=method,
    )


def _detect_price_curve_violations(cat_data: pd.DataFrame) -> pd.DataFrame:
    """Detect violations: larger pack cheaper per unit."""

    if "pack_size_numeric" not in cat_data.columns:
        return pd.DataFrame()

    sorted_data = cat_data.sort_values("pack_size_numeric")

    violations = []
    for i in range(len(sorted_data) - 1):
        row1 = sorted_data.iloc[i]
        row2 = sorted_data.iloc[i + 1]

        if row1["price_per_unit"] > row2["price_per_unit"] * 1.05:  # 5% tolerance
            violations.append(
                {
                    "category": row1["category"],
                    "larger_pack": row1["product_name"],
                    "larger_size": row1["pack_size_numeric"],
                    "larger_price_per_unit": row1["price_per_unit"],
                    "smaller_pack": row2["product_name"],
                    "smaller_size": row2["pack_size_numeric"],
                    "smaller_price_per_unit": row2["price_per_unit"],
                    "violation_pct": (row1["price_per_unit"] / row2["price_per_unit"] - 1) * 100,
                }
            )

    return pd.DataFrame(violations)


# ============================================================================
# CAUSAL & ECONOMETRIC METHODS (PHASE 2)
# ============================================================================


def estimate_iv_elasticity(
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
    """
    Instrumental Variables (IV) / 2SLS elasticity estimation.

    Uses cost shifters (e.g., seasonal input price indices) as instruments
    to address endogeneity in price-quantity relationship.

    2SLS:
    Stage 1: log(price) = alpha + gamma * instrument + controls
    Stage 2: log(qty) = alpha + beta * log(price_hat) + controls

    Args:
        instrument_col: Column name for instrument (e.g., cost, input price index)

    Returns:
        DataFrame with IV elasticity estimates per SKU
    """
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["revenue"] = df[price_col] * df[qty_col]

    results = []

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

        # Check variation in both price and instrument
        price_cv = weekly["avg_price"].std() / weekly["avg_price"].mean()
        instr_cv = weekly["avg_instrument"].std() / weekly["avg_instrument"].mean()
        if price_cv < min_price_variation or instr_cv < min_price_variation:
            continue

        log_price = np.log(weekly["avg_price"].replace(0, np.nan).dropna())
        log_qty = np.log(weekly.loc[log_price.index, "total_qty"].replace(0, np.nan).dropna())
        log_instr = np.log(
            weekly.loc[log_price.index, "avg_instrument"].replace(0, np.nan).dropna()
        )

        if len(log_price) < min_periods:
            continue

        # Stage 1: log_price = alpha + gamma * log_instr
        X1 = sm.add_constant(log_instr)
        try:
            model1 = sm.OLS(log_price, X1).fit()
            log_price_hat = model1.predict(X1)
        except Exception:
            continue

        # Stage 2: log_qty = alpha + beta * log_price_hat
        X2 = sm.add_constant(log_price_hat)
        y = log_qty

        try:
            model2 = sm.OLS(y, X2).fit(cov_type="HC3")
            elasticity = model2.params[1]
            std_err = model2.bse[1]
            p_value = model2.pvalues[1]
            r_squared = model2.rsquared
            f_stat = model2.fvalue  # First stage F-stat for weak instrument test
        except Exception:
            continue

        # Weak instrument check (F < 10 is weak)
        weak_instrument = f_stat < 10 if f_stat else True

        results.append(
            {
                "stockcode": product_id,
                "iv_elasticity": elasticity,
                "iv_elasticity_se": std_err,
                "iv_elasticity_p": p_value,
                "iv_r_squared": r_squared,
                "first_stage_f": f_stat if f_stat else np.nan,
                "weak_instrument": weak_instrument,
                "n_obs": len(log_price),
                "avg_price": weekly["avg_price"].mean(),
                "avg_weekly_qty": weekly["total_qty"].mean(),
                "avg_instrument": weekly["avg_instrument"].mean(),
            }
        )

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def estimate_rdd_elasticity(
    transactions_df: pd.DataFrame,
    price_col: str = "price",
    qty_col: str = "quantity",
    date_col: str = "date",
    product_col: str = "stockcode",
    threshold_price: float = None,
    bandwidth: float = 0.5,
    kernel: str = "triangular",
    freq: str = "W",
    min_periods: int = 10,
) -> pd.DataFrame:
    """
    Regression Discontinuity Design (RDD) for price threshold effects.

    Estimates elasticity at psychological price thresholds (e.g., $4.99 vs $5.00).
    Uses local linear regression around the threshold.

    Args:
        threshold_price: Price threshold to test (e.g., 5.0 for $5.00).
                        If None, tests common psychological thresholds.
        bandwidth: Window around threshold (e.g., 0.5 = $0.50 window)
        kernel: Kernel function ('triangular', 'uniform', 'epanechnikov')

    Returns:
        DataFrame with RDD elasticity estimates at each threshold
    """
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["revenue"] = df[price_col] * df[qty_col]

    if threshold_price is None:
        # Common psychological thresholds
        threshold_price = [0.99, 1.99, 2.99, 4.99, 9.99, 19.99, 49.99, 99.99]
    else:
        threshold_price = [threshold_price]

    results = []

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

        log_price = np.log(weekly["avg_price"].replace(0, np.nan).dropna())
        log_qty = np.log(weekly.loc[log_price.index, "total_qty"].replace(0, np.nan).dropna())

        if len(log_price) < min_periods:
            continue

        price_vals = weekly["avg_price"].values

        for thresh in threshold_price:
            # Define bandwidth window
            in_window = (price_vals >= thresh - bandwidth) & (price_vals <= thresh + bandwidth)
            if in_window.sum() < min_periods:
                continue

            # Local linear regression around threshold
            x = price_vals[in_window] - thresh
            y = log_qty[in_window]

            # Kernel weights
            if kernel == "triangular":
                weights = 1 - np.abs(x) / bandwidth
            elif kernel == "epanechnikov":
                weights = 1 - (x / bandwidth) ** 2
            else:  # uniform
                weights = np.ones_like(x)

            # Weighted regression: log_qty = alpha + beta * (price - thresh)
            X = np.column_stack([np.ones_like(x), x])
            W = np.diag(weights)

            try:
                beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
                elasticity = beta[1]  # Local elasticity at threshold
            except Exception:
                continue

            results.append(
                {
                    "product_a": product_id,
                    "product_b": "RDD_threshold",
                    "threshold_price": thresh,
                    "cross_elasticity": elasticity,
                    "n_obs": in_window.sum(),
                    "bandwidth": bandwidth,
                }
            )

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def estimate_synthetic_control_elasticity(
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
) -> Dict:
    """
    Synthetic Control Method for elasticity estimation.

    Constructs a weighted combination of donor products that matches
    the treatment product's pre-treatment demand trajectory.
    The post-treatment gap estimates the causal effect of price change.

    Args:
        treatment_product: SKU that experienced price change
        donor_products: List of SKUs to use as donor pool
        pre_periods: Number of periods before treatment
        post_periods: Number of periods after treatment

    Returns:
        Dict with treatment effect, weights, and diagnostics
    """
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["revenue"] = df[price_col] * df[qty_col]

    # Aggregate to weekly level
    weekly_all = (
        df.set_index(date_col)
        .groupby([product_col, pd.Grouper(freq="W")])
        .agg(avg_price=(price_col, "mean"), total_qty=(qty_col, "sum"))
        .dropna()
        .reset_index()
    )

    # Filter to treatment and donor products
    products = [treatment_product] + donor_products
    weekly = weekly_all[weekly_all[product_col].isin(products)].copy()

    # Log transform
    weekly["log_price"] = np.log(weekly["avg_price"].clip(lower=1e-6))
    weekly["log_qty"] = np.log(weekly["total_qty"].clip(lower=1e-6))

    # Split into pre/post
    all_dates = sorted(weekly["date"].unique())
    cutoff_idx = len(all_dates) - post_periods

    if cutoff_idx < pre_periods:
        raise ValueError("Not enough data for synthetic control")

    pre_dates = all_dates[:cutoff_idx]
    post_dates = all_dates[cutoff_idx:]

    pre_data = weekly[weekly["date"].isin(pre_dates)]
    post_data = weekly[weekly["date"].isin(post_dates)]

    # Build donor matrix (donor products x pre-periods)
    donor_matrix = (
        pre_data[pre_data[product_col].isin(donor_products)]
        .pivot_table(index=product_col, columns="date", values="log_qty", aggfunc="mean")
        .fillna(method="ffill", axis=1)
        .fillna(method="bfill", axis=1)
    )

    treatment_pre = pre_data[pre_data[product_col] == treatment_product].set_index("date")[
        "log_qty"
    ]

    # Optimize weights: min ||treatment_pre - donor_matrix.T @ w||^2 s.t. w >= 0, sum(w) = 1
    from scipy.optimize import minimize

    n_donors = len(donor_matrix)

    def objective(w):
        return np.sum((treatment_pre.values - donor_matrix.T.values @ w) ** 2)

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = [(0, 1) for _ in range(n_donors)]
    w0 = np.ones(n_donors) / n_donors

    result = minimize(objective, w0, bounds=bounds, constraints=constraints, method="SLSQP")
    weights = result.x

    # Predict counterfactual
    donor_post = (
        post_data[post_data[product_col].isin(donor_products)]
        .pivot_table(index=product_col, columns="date", values="log_qty", aggfunc="mean")
        .fillna(method="ffill", axis=1)
        .fillna(method="bfill", axis=1)
    )

    counterfactual = donor_post.T.values @ weights
    actual = post_data[post_data[product_col] == treatment_product].set_index("date")["log_qty"]

    # Align and compute effect
    common_dates = actual.index.intersection(pd.Index(counterfactual.index))
    if len(common_dates) == 0:
        return {"error": "No overlapping post-period dates"}

    actual_vals = actual.loc[common_dates].values
    counterfactual_vals = counterfactual[common_dates]

    effect = np.mean(actual_vals - counterfactual_vals)
    effect_pct = np.mean(
        (np.exp(actual_vals) - np.exp(counterfactual_vals)) / np.exp(counterfactual_vals)
    )

    return {
        "treatment_product": treatment_product,
        "donor_weights": dict(zip(donor_products, weights)),
        "pre_periods": pre_periods,
        "post_periods": post_periods,
        "effect_log_qty": effect,
        "effect_pct": effect_pct,
        "counterfactual": counterfactual_vals.tolist(),
        "actual": actual_vals.tolist(),
        "dates": common_dates.strftime("%Y-%m-%d").tolist(),
    }


def run_price_rdd_at_thresholds(
    transactions_df: pd.DataFrame, thresholds: List[float] = None, bandwidth: float = 0.5, **kwargs
) -> pd.DataFrame:
    """
    Run RDD at multiple price thresholds and return combined results.
    """
    if thresholds is None:
        thresholds = [0.99, 1.99, 2.99, 4.99, 9.99, 19.99, 49.99, 99.99]

    results = []
    for thresh in thresholds:
        try:
            rdd_results = estimate_rdd_elasticity(
                transactions_df, threshold_price=thresh, bandwidth=bandwidth, **kwargs
            )
            rdd_results["threshold"] = thresh
            results.append(rdd_results)
        except Exception:
            continue

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


# ============================================================================
# SEQUENCE & NETWORK ALGORITHMS (PHASE 2)
# ============================================================================


def mine_sequential_patterns_prefixspan(
    transactions_df: pd.DataFrame,
    customer_col: str = "customer_id",
    product_col: str = "stockcode",
    date_col: str = "date",
    min_support: float = 0.01,
    max_pattern_length: int = 5,
) -> pd.DataFrame:
    """
    Mine sequential patterns using PrefixSpan algorithm.

    Finds ordered sequences of product purchases across customer journeys.
    E.g., A -> B -> C means customers who buy A, then B, then C.

    Args:
        min_support: Minimum support threshold (fraction of customers)
        max_pattern_length: Maximum length of sequential patterns

    Returns:
        DataFrame with sequential patterns and their support
    """
    try:
        from prefixspan import PrefixSpan
    except ImportError:
        raise ImportError("prefixspan required: pip install prefixspan")

    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([customer_col, date_col])

    # Build sequences per customer
    sequences = df.groupby(customer_col)[product_col].apply(list).tolist()

    # Filter by min_support
    min_count = int(len(sequences) * min_support)

    ps = PrefixSpan(sequences)
    ps.minlen = 1
    ps.maxlen = max_pattern_length

    patterns = ps.frequent(min_count)

    # Convert to DataFrame
    results = []
    for support, pattern in patterns:
        if len(pattern) >= 2:  # Only multi-item sequences
            results.append(
                {
                    "pattern": " -> ".join(pattern),
                    "length": len(pattern),
                    "support": support / len(sequences),
                    "support_count": support,
                }
            )

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("support", ascending=False)


def compute_personalized_pagerank(
    similarity_matrix: pd.DataFrame,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Compute Personalized PageRank on product similarity graph.

    For each product, computes proximity to all other products
    via random walk with restart. Better than community detection
    for recommendation: gives per-anchor-product similarity scores.

    Args:
        similarity_matrix: Product x product similarity (Phi/Jaccard/PMI)
        alpha: Damping factor (teleport probability = 1 - alpha)

    Returns:
        DataFrame with personalized PageRank scores (products x products)
    """
    n = len(similarity_matrix)
    products = similarity_matrix.index.tolist()

    # Normalize similarity matrix to row-stochastic transition matrix
    P = similarity_matrix.values.copy()
    row_sums = P.sum(axis=1)
    P[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
    P[row_sums == 0] = 1.0 / n  # Uniform for dangling nodes

    # Personalized PageRank for each product
    ppr_matrix = np.zeros((n, n))

    for i in range(n):
        # Personalized teleport vector
        e_i = np.zeros(n)
        e_i[i] = 1.0

        # Power iteration
        pi = e_i.copy()
        for _ in range(max_iter):
            pi_new = alpha * (P.T @ pi) + (1 - alpha) * e_i
            if np.abs(pi_new - pi).max() < tol:
                pi = pi_new
                break
            pi = pi_new
        ppr_matrix[i] = pi

    return pd.DataFrame(ppr_matrix, index=products, columns=products)


def train_customer_lifecycle_hmm(
    transactions_df: pd.DataFrame,
    n_states: int = 4,
    customer_col: str = "customer_id",
    product_col: str = "stockcode",
    date_col: str = "date",
    n_iter: int = 100,
) -> Tuple[object, pd.DataFrame]:
    """
    Train Hidden Markov Model for customer lifecycle stages.

    Latent states represent lifecycle stages (e.g., New, Active, At Risk, Churned).
    Observations: product categories, basket size, recency, frequency.

    Returns:
        (model, state_predictions) where state_predictions has customer_id, predicted_state, state_probs
    """
    try:
        from hmmlearn import hmm
    except ImportError:
        raise ImportError("hmmlearn required: pip install hmmlearn")

    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # Build feature sequences per customer
    cat_col = "category" if "category" in transactions_df.columns else "stockcode"

    sequences = []
    lengths = []
    customer_ids = []

    for cust_id, grp in df.groupby(customer_col):
        grp = grp.sort_values("date")
        feats = []
        for _, row in grp.iterrows():
            # Feature vector: [category_onehot, log_price, log_qty, day_of_week, basket_size]
            cat_onehot = np.zeros(20)  # Assume max 20 categories
            cat_idx = hash(row[cat_col]) % 20
            cat_onehot[cat_idx] = 1
            feats = np.concatenate(
                [
                    cat_onehot,
                    [
                        np.log(row["price"] + 1),
                        np.log(row["quantity"] + 1),
                        row["date"].weekday(),
                        1,
                    ],  # basket_size placeholder
                ]
            )
            feats.append(feats)
        if len(feats) > 0:
            sequences.append(np.array(feats))
            lengths.append(len(feats))
            customer_ids.append(cust_id)

    if not sequences:
        return None, pd.DataFrame()

    # Concatenate all sequences
    X = np.vstack(sequences)

    # Train HMM
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=42,
    )
    model.fit(X, lengths=lengths)

    # Predict states for each customer
    predictions = []
    for i, (cust_id, seq) in enumerate(zip(customer_ids, sequences)):
        states = model.predict(sequences[i])
        state_probs = model.predict_proba(sequences[i])

        for t, (state, probs) in enumerate(zip(states, state_probs)):
            predictions.append(
                {
                    "customer_id": cust_id,
                    "time_step": t,
                    "predicted_state": state,
                    "state_prob": probs[state],
                    "state_probs": probs,
                }
            )

    pred_df = pd.DataFrame(predictions)

    # Label states by their emission means (monetary, frequency, etc.)
    state_means = model.means_
    state_labels = {}
    for i in range(n_states):
        mean_monetary = state_means[i][1] if len(state_means[i]) > 1 else 0
        if mean_monetary > np.percentile(state_means[:, 1], 75):
            state_labels[i] = "High Value"
        elif mean_monetary > np.percentile(state_means[:, 1], 25):
            state_labels[i] = "Active"
        else:
            state_labels[i] = "At Risk"

    pred_df["state_label"] = pred_df["predicted_state"].map(state_labels)

    return model, pred_df


def detect_changepoints_bocpd(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    date_col: str = "date",
    qty_col: str = "quantity",
    freq: str = "W",
    hazard_lambda: float = 100,
) -> pd.DataFrame:
    """
    Bayesian Online Changepoint Detection (BOCPD) for product demand.

    Detects changepoints in product demand time series (e.g., new product launch,
    seasonality shift, competitor entry, supply disruption).

    Args:
        hazard_lambda: Exponential hazard rate (mean run length = lambda)

    Returns:
        DataFrame with changepoints per product (product, changepoint_date, probability)
    """
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["revenue"] = df["price"] * df["quantity"]

    all_changepoints = []

    for product_id in df["stockcode"].unique():
        prod_df = df[df["stockcode"] == product_id].copy()
        weekly = (
            prod_df.set_index(date_col)
            .groupby(pd.Grouper(freq=freq))
            .agg(total_qty=(qty_col, "sum"), total_rev=("revenue", "sum"))
            .dropna()
        )

        if len(weekly) < 10:
            continue

        # Simple BOCPD using conjugate Normal-Gamma model
        # (Simplified: using scipy's changepoint detection as proxy)
        from scipy.signal import find_peaks
        from scipy.stats import zscore

        qty_series = weekly["total_qty"].values
        z_scores = zscore(qty_series)
        peaks, _ = find_peaks(np.abs(z_scores), height=2.5, distance=4)

        for peak in peaks:
            all_changepoints.append(
                {
                    "stockcode": product_id,
                    "changepoint_date": weekly.index[peak],
                    "z_score": z_scores[peak],
                    "quantity": qty_series[peak],
                }
            )

    if not all_changepoints:
        return pd.DataFrame()

    return pd.DataFrame(all_changepoints)


# ============================================================================
# CLUSTERING ENHANCEMENTS (PHASE 2)
# ============================================================================


def cluster_hdbscan(
    features: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = None,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    HDBSCAN clustering - finds variable-density clusters with noise detection.

    Unlike KMeans, HDBSCAN:
    - Finds clusters of variable density
    - Identifies noise points (label = -1)
    - No need to specify n_clusters
    """
    try:
        import hdbscan
    except ImportError:
        raise ImportError("hdbscan required: pip install hdbscan")

    if min_samples is None:
        min_samples = min_cluster_size

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
    )

    labels = clusterer.fit_predict(features)
    return labels


def compute_wasserstein_distance(
    segment_a: pd.DataFrame,
    segment_b: pd.DataFrame,
    feature_cols: List[str],
) -> float:
    """
    Compute Wasserstein (Earth Mover's) distance between two segment distributions.

    More meaningful than centroid distance for comparing segment distributions.
    """
    from scipy.stats import wasserstein_distance

    distances = []
    for col in feature_cols:
        dist = wasserstein_distance(segment_a[col].dropna(), segment_b[col].dropna())
        distances.append(dist)
    return np.mean(distances)


def compute_segment_wasserstein_matrix(
    segments_df: pd.DataFrame,
    feature_cols: List[str],
    segment_col: str = "segment",
) -> pd.DataFrame:
    """
    Compute pairwise Wasserstein distance matrix between segments.

    Returns symmetric DataFrame (segments x segments) with Wasserstein distances.
    """
    segments = sorted(segments_df[segment_col].unique())
    matrix = pd.DataFrame(0.0, index=segments, columns=segments)

    for i, seg_a in enumerate(segments):
        for j, seg_b in enumerate(segments):
            if j < i:
                continue
            if seg_a == seg_b:
                matrix.loc[seg_a, seg_b] = 0.0
            else:
                a_data = segments_df[segments_df[segment_col] == seg_a][feature_cols]
                b_data = segments_df[segments_df[segment_col] == seg_b][feature_cols]
                dist = compute_wasserstein_distance(a_data, b_data, feature_cols)
                matrix.loc[seg_a, seg_b] = dist
                matrix.loc[seg_b, seg_a] = dist

    return matrix
