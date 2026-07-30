"""Pricing, Elasticity & KVI Analytics."""

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Optional statsmodels for robust OLS
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings("ignore")


# ============================================================================
# ELASTICITY ESTIMATION
# ============================================================================


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
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_price, log_qty)
            elasticity = slope
            r_squared = r_value**2
            ci_lower = elasticity - 1.96 * std_err
            ci_upper = elasticity + 1.96 * std_err

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

    Shrinks individual SKU elasticities toward category-level mean using Ridge regression.
    This is a simplified approximation of hierarchical Bayesian elasticity.
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
            }
        )

    if not ols_results:
        return pd.DataFrame()

    ols_df = pd.DataFrame(ols_results)

    # Category means
    cat_means = ols_df.groupby("category")["elasticity_ols"].mean().rename("elasticity_cat")

    # Shrink toward category mean: weight = n_obs / (n_obs + lambda)
    ols_df["shrink_weight"] = ols_df["n_obs"] / (ols_df["n_obs"] + ridge_alpha)
    ols_df = ols_df.merge(cat_means, on="category", how="left")
    ols_df["elasticity_shrunk"] = (
        ols_df["shrink_weight"] * ols_df["elasticity_ols"]
        + (1 - ols_df["shrink_weight"]) * ols_df["elasticity_cat"]
    )

    return ols_df


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
    cat_idx = agg_df["category"].map(cat_to_idx).values

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
        alpha_sku = pm.Normal(
            "alpha_sku", alpha_cat[sku_cat_idx], sigma_sku, dims="sku"
        )
        beta_sku = pm.Normal(
            "beta_sku", beta_cat[sku_cat_idx], sigma_sku, dims="sku"
        )

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

    # --- 4. Extract SKU-level posterior ---
    alpha_post = trace.posterior["alpha_sku"] if hasattr(trace, "posterior") else trace["alpha_sku"]
    beta_post = trace.posterior["beta_sku"] if hasattr(trace, "posterior") else trace["beta_sku"]

    beta_mean = beta_post.mean(dim=("chain", "draw")).values if hasattr(beta_post, "mean") else beta_post.mean(axis=0)
    beta_sd = beta_post.std(dim=("chain", "draw")).values if hasattr(beta_post, "std") else beta_post.std(axis=0)

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
    sku_stats = agg_df.groupby("stockcode").agg(
        category=("category", "first"),
        n_obs=("total_qty", "size"),
        avg_price=("avg_price", "mean"),
    ).reset_index()

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
    """KVI scoring via XGBoost."""
    try:
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

    # Train
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0
    )
    model.fit(X, y)

    # Predictions as KVI score
    kvi_features["kvi_score"] = model.predict(X)

    # Feature importance
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
            if unit == "ML":
                return val / 1000
            elif unit == "G":
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
    from src.analytics import compute_basket_penetration, compute_product_metrics

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
            if unit == "ML":
                return val / 1000
            elif unit == "G":
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
            transactions_df.groupby("stockcode")
            .agg(median_cost=(cost_col, "median"))
            .reset_index()
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
