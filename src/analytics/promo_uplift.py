"""Promotional Uplift Modeling — Causal Inference for Promo Impact.

Implements uplift modeling (T-learner, S-learner) and propensity stratification
to estimate true incremental impact of promotions, decomposing into:
- True Incrementality (new demand)
- Forward Buy (stockpiling)
- Substitution (cannibalization)

References
----------
- Jaskowski & Jaroszewicz (2012) "Uplift Modeling for Clinical Trial Data"
- Kunzel et al. (2019) "Metalearners for Estimating Heterogeneous Treatment Effects"
- Gutierrez & Gerardy (2017) "Causal Inference and Uplift Modeling"
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from causalml.inference.tree import UpliftRandomForestClassifier, UpliftTreeClassifier

    CAUSALML_AVAILABLE = True
except ImportError:
    CAUSALML_AVAILABLE = False

warnings.filterwarnings("ignore")


# ============================================================================
# PROMO DETECTION
# ============================================================================


def derive_promo_flag(
    transactions_df: pd.DataFrame,
    price_col: str = "price",
    product_col: str = "stockcode",
    date_col: str = "date",
    window_days: int = 28,
    drop_threshold: float = 0.15,
    baseline_quantile: float = 0.9,
) -> pd.DataFrame:
    """
    Detect promotional periods from price drops vs rolling baseline.

    Baseline = rolling quantile (default 90th percentile) over window_days.
    Flag promo when price < baseline * (1 - drop_threshold).
    """
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([product_col, date_col])

    promos = []

    for sku, grp in df.groupby(product_col):
        prices = grp[price_col].values
        dates = grp[date_col].values

        if len(prices) < window_days:
            continue

        baseline = pd.Series(prices).rolling(window_days, min_periods=1).quantile(baseline_quantile)

        for i, (price, date) in enumerate(zip(prices, dates)):
            base = baseline.iloc[i]
            if base > 0 and price < base * (1 - drop_threshold):
                discount = (base - price) / base
                promos.append(
                    {
                        "stockcode": sku,
                        "date": date,
                        "price": price,
                        "baseline_price": base,
                        "discount_pct": discount * 100,
                        "is_promo": True,
                    }
                )

    return pd.DataFrame(promos)


def build_uplift_dataset(
    transactions_df: pd.DataFrame,
    promo_df: pd.DataFrame,
    product_col: str = "stockcode",
    customer_col: str = "customer_id",
    date_col: str = "date",
    price_col: str = "price",
    qty_col: str = "quantity",
    prediction_window_days: int = 7,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build uplift modeling dataset at customer-product-week level.

    Returns:
        X: features (customer + product + context)
        treatment: binary promo exposure
        y: outcome (units purchased in prediction window)
    """
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["revenue"] = df[price_col] * df[qty_col]

    # Aggregate to customer-product-week
    df["week"] = df[date_col].dt.to_period("W")

    # Merge promo flags
    promo_df = promo_df.copy()
    promo_df[date_col] = pd.to_datetime(promo_df[date_col])
    promo_df["week"] = promo_df[date_col].dt.to_period("W")

    weekly = (
        df.groupby([customer_col, product_col, "week"])
        .agg(
            total_qty=(qty_col, "sum"),
            total_rev=("revenue", "sum"),
            avg_price=(price_col, "mean"),
            n_txns=("transaction_id", "nunique"),
        )
        .reset_index()
    )

    # Merge promo flag
    promo_weekly = promo_df.groupby([product_col, "week"])["is_promo"].any().reset_index()
    weekly = weekly.merge(promo_weekly, on=[product_col, "week"], how="left")
    weekly["treatment"] = weekly["is_promo"].fillna(False).astype(int)

    # Target: quantity in next week
    weekly = weekly.sort_values([customer_col, product_col, "week"])
    weekly["next_week_qty"] = weekly.groupby([customer_col, product_col])["total_qty"].shift(-1)
    weekly = weekly.dropna(subset=["next_week_qty"])

    # Features
    # Note: Features are built dynamically below based on available columns

    # Add customer-level features
    cust_features = (
        df.groupby(customer_col)
        .agg(
            cust_total_qty=(qty_col, "sum"),
            cust_total_rev=("revenue", "sum"),
            cust_n_products=(product_col, "nunique"),
            cust_avg_price=(price_col, "mean"),
            cust_n_txns=("transaction_id", "nunique"),
            cust_first_date=(date_col, "min"),
            cust_last_date=(date_col, "max"),
        )
        .reset_index()
    )

    cust_features["cust_lifetime_days"] = (
        cust_features["cust_last_date"] - cust_features["cust_first_date"]
    ).dt.days
    cust_features["cust_freq"] = cust_features["cust_n_txns"] / cust_features[
        "cust_lifetime_days"
    ].clip(lower=1)

    weekly = weekly.merge(cust_features, on=customer_col, how="left")

    # Add product-level features
    prod_features = (
        df.groupby(product_col)
        .agg(
            prod_total_qty=(qty_col, "sum"),
            prod_total_rev=("revenue", "sum"),
            prod_avg_price=(price_col, "mean"),
            prod_price_cv=(price_col, lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
        )
        .reset_index()
    )

    weekly = weekly.merge(prod_features, on=product_col, how="left")

    # Lag features
    weekly["total_qty_lag1"] = weekly.groupby([customer_col, product_col])["total_qty"].shift(1)
    weekly["avg_price_lag1"] = weekly.groupby([customer_col, product_col])["avg_price"].shift(1)

    # Time features
    weekly["week_of_year"] = weekly["week"].dt.week
    weekly["month"] = weekly["week"].dt.month

    # Fill NaN
    weekly = weekly.fillna(0)

    # Define X, treatment, y
    X_cols = [
        c
        for c in weekly.columns
        if c
        not in [
            customer_col,
            product_col,
            "week",
            "is_promo",
            "treatment",
            "next_week_qty",
            "cust_first_date",
            "cust_last_date",
        ]
    ]
    X = weekly[X_cols]
    treatment = weekly["treatment"]
    y = weekly["next_week_qty"]

    return X, treatment, y


# ============================================================================
# UPLIFT LEARNERS
# ============================================================================


def train_t_learner_uplift(
    X: pd.DataFrame,
    treatment: pd.Series,
    y: pd.Series,
    base_learner: str = "xgb",
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> Tuple[object, object, pd.Series]:
    """
    T-Learner: Two separate models for treated and control.

    Uplift = mu_1(X) - mu_0(X)
    """
    if base_learner == "xgb":
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost required for xgb base learner")
        model_class = xgb.XGBRegressor
        model_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbosity=0,
        )
    elif base_learner == "rf":
        model_class = RandomForestRegressor
        model_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown base_learner: {base_learner}")

    # Split treated/control
    X_treated = X[treatment == 1]
    y_treated = y[treatment == 1]
    X_control = X[treatment == 0]
    y_control = y[treatment == 0]

    if len(X_treated) < 10 or len(X_control) < 10:
        raise ValueError("Insufficient treated/control samples")

    # Train
    model_treated = model_class(**model_params)
    model_control = model_class(**model_params)

    model_treated.fit(X_treated, y_treated)
    model_control.fit(X_control, y_control)

    # Predict uplift on full dataset
    mu_1 = model_treated.predict(X)
    mu_0 = model_control.predict(X)
    uplift = mu_1 - mu_0

    return model_treated, model_control, pd.Series(uplift, index=X.index)


def train_s_learner_uplift(
    X: pd.DataFrame,
    treatment: pd.Series,
    y: pd.Series,
    base_learner: str = "xgb",
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> Tuple[object, pd.Series]:
    """
    S-Learner: Single model with treatment as feature.

    Uplift = pred(X, T=1) - pred(X, T=0)
    """
    if base_learner == "xgb":
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost required for xgb base learner")
        model_class = xgb.XGBRegressor
        model_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbosity=0,
        )
    elif base_learner == "rf":
        model_class = RandomForestRegressor
        model_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown base_learner: {base_learner}")

    # Add treatment as feature
    X_with_t = X.copy()
    X_with_t["treatment"] = treatment

    model = model_class(**model_params)
    model.fit(X_with_t, y)

    # Counterfactual predictions
    X_t1 = X.copy()
    X_t1["treatment"] = 1
    X_t0 = X.copy()
    X_t0["treatment"] = 0

    mu_1 = model.predict(X_t1)
    mu_0 = model.predict(X_t0)
    uplift = mu_1 - mu_0

    return model, pd.Series(uplift, index=X.index)


def train_xgb_uplift(
    X: pd.DataFrame,
    treatment: pd.Series,
    y: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> Tuple[object, pd.Series]:
    """
    Direct uplift optimization using modified objective (causalml-style).

    Uses 'distribution' = 'kl' or 'euclidean' for uplift-specific split criterion.
    """
    if not CAUSALML_AVAILABLE:
        raise ImportError("causalml required for direct uplift: pip install causalml")

    # Use UpliftRandomForestClassifier for binary outcomes or regressor for continuous
    model = UpliftRandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=10,
        min_samples_treatment=10,
        evaluationFunction="KL",
        random_state=random_state,
    )

    model.fit(X.values, treatment.values, y.values)

    # Predict uplift
    uplift_pred = model.predict(X.values)[:, 1]  # uplift for treatment=1

    return model, pd.Series(uplift_pred, index=X.index)


def train_uplift_tree(
    X: pd.DataFrame,
    treatment: pd.Series,
    y: pd.Series,
    max_depth: int = 5,
    min_samples_leaf: int = 10,
    min_samples_treatment: int = 10,
    evaluation_function: str = "KL",
    random_state: int = 42,
) -> Tuple[object, pd.Series]:
    """
    Train a single uplift decision tree (interpretable).

    Uses causalml's UpliftTreeClassifier.
    """
    if not CAUSALML_AVAILABLE:
        raise ImportError("causalml required: pip install causalml")

    model = UpliftTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_treatment=min_samples_treatment,
        evaluationFunction=evaluation_function,
        random_state=random_state,
    )

    model.fit(X.values, treatment.values, y.values)

    uplift_pred = model.predict(X.values)[:, 1]

    return model, pd.Series(uplift_pred, index=X.index)


# ============================================================================
# PROPENSITY SCORE STRATIFICATION
# ============================================================================


def estimate_propensity_score(
    X: pd.DataFrame,
    treatment: pd.Series,
    method: str = "xgb",
    **model_params,
) -> pd.Series:
    """
    Estimate propensity score P(T=1|X) for stratification or IPW.

    Returns propensity scores (probability of treatment).
    """
    if method == "xgb":
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost required")
        model = xgb.XGBClassifier(
            n_estimators=model_params.get("n_estimators", 100),
            max_depth=model_params.get("max_depth", 5),
            learning_rate=model_params.get("learning_rate", 0.1),
            random_state=model_params.get("random_state", 42),
            verbosity=0,
        )
    elif method == "lr":
        model = LogisticRegression(
            max_iter=model_params.get("max_iter", 1000),
            random_state=model_params.get("random_state", 42),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    model.fit(X, treatment)
    propensity = model.predict_proba(X)[:, 1]

    return pd.Series(propensity, index=X.index)


def check_propensity_overlap(
    propensity: pd.Series,
    treatment: pd.Series,
    min_overlap: float = 0.1,
    trim_threshold: float = 0.01,
) -> Dict:
    """
    Check propensity score overlap (common support) between treated and control.

    Overlap is crucial for valid causal inference. If treated and control units
    have disjoint propensity score ranges, causal estimates are unreliable.

    Args:
        propensity: Propensity scores (0-1)
        treatment: Binary treatment indicator (0/1)
        min_overlap: Minimum required overlap proportion (default 0.1)
        trim_threshold: Trim observations with propensity < trim_threshold or > 1-trim_threshold

    Returns:
        Dict with overlap metrics, warnings, and trimmed indices if applicable
    """
    treated_ps = propensity[treatment == 1]
    control_ps = propensity[treatment == 0]

    if len(treated_ps) == 0 or len(control_ps) == 0:
        return {
            "overlap": False,
            "warning": "No treated or control units",
            "treated_range": None,
            "control_range": None,
            "overlap_proportion": 0.0,
            "trimmed": False,
        }

    treated_range = (treated_ps.min(), treated_ps.max())
    control_range = (control_ps.min(), control_ps.max())

    # Compute overlap proportion
    overlap_lower = max(treated_range[0], control_range[0])
    overlap_upper = min(treated_range[1], control_range[1])

    if overlap_lower >= overlap_upper:
        overlap_proportion = 0.0
    else:
        # Proportion of each group in the overlap region
        treated_in_overlap = ((treated_ps >= overlap_lower) & (treated_ps <= overlap_upper)).mean()
        control_in_overlap = ((control_ps >= overlap_lower) & (control_ps <= overlap_upper)).mean()
        overlap_proportion = min(treated_in_overlap, control_in_overlap)

    # Check for extreme propensity scores (near 0 or 1)
    trim_low = (propensity < trim_threshold).sum()
    trim_high = (propensity > 1 - trim_threshold).sum()

    results = {
        "overlap": overlap_proportion >= min_overlap,
        "overlap_proportion": overlap_proportion,
        "treated_range": treated_range,
        "control_range": control_range,
        "overlap_range": (overlap_lower, overlap_upper),
        "treated_in_overlap": treated_in_overlap,
        "control_in_overlap": control_in_overlap,
        "min_overlap_required": min_overlap,
        "trim_low": int(trim_low),
        "trim_high": int(trim_high),
        "trim_threshold": trim_threshold,
        "warnings": [],
    }

    if overlap_proportion < min_overlap:
        results["warnings"].append(
            f"Insufficient propensity overlap: {overlap_proportion:.1%} < {min_overlap:.1%}. "
            f"Causal estimates may be unreliable."
        )

    if trim_low > 0 or trim_high > 0:
        results["warnings"].append(
            f"Extreme propensity scores detected: {trim_low} below {trim_threshold}, "
            f"{trim_high} above {1 - trim_threshold}. Consider trimming."
        )

    # Check for complete separation
    if treated_range[0] > control_range[1] or control_range[0] > treated_range[1]:
        results["warnings"].append(
            "Complete separation: treated and control propensity ranges do not overlap at all."
        )
        results["overlap"] = False

    return results


def trim_propensity_scores(
    X: pd.DataFrame,
    treatment: pd.Series,
    propensity: pd.Series,
    trim_threshold: float = 0.01,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Trim observations with extreme propensity scores.

    Removes units with propensity < trim_threshold or > 1-trim_threshold
    to improve overlap and reduce extrapolation.

    Returns trimmed X, treatment, y, propensity.
    """
    mask = (propensity >= trim_threshold) & (propensity <= 1 - trim_threshold)

    return X[mask], treatment[mask], propensity[mask], mask


def validate_uplift_assumptions(
    X: pd.DataFrame,
    treatment: pd.Series,
    y: pd.Series,
    propensity: pd.Series,
    min_overlap: float = 0.1,
) -> Dict:
    """
    Comprehensive validation of uplift modeling assumptions.

    Checks:
    1. Propensity score overlap (common support)
    2. Balance before/after stratification
    3. Treatment effect heterogeneity
    4. Sample size adequacy

    Returns dict with validation results and recommendations.
    """
    # 1. Overlap check
    overlap_results = check_propensity_overlap(propensity, treatment, min_overlap=min_overlap)

    # 2. Covariate balance (standardized mean differences)
    treated_mask = treatment == 1
    control_mask = treatment == 0

    balance_results = {}
    for col in X.select_dtypes(include=[np.number]).columns:
        treated_mean = X.loc[treated_mask, col].mean()
        control_mean = X.loc[control_mask, col].mean()
        treated_std = X.loc[treated_mask, col].std()
        control_std = X.loc[control_mask, col].std()
        pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)

        if pooled_std > 0:
            smd = (treated_mean - control_mean) / pooled_std
        else:
            smd = 0

        balance_results[col] = {
            "smd": float(smd),
            "balanced": abs(smd) < 0.1,  # SMD < 0.1 is good balance
            "treated_mean": float(treated_mean),
            "control_mean": float(control_mean),
        }

    # Overall balance
    max_smd = max(abs(v["smd"]) for v in balance_results.values()) if balance_results else 0

    # 3. Sample size check
    n_treated = treated_mask.sum()
    n_control = control_mask.sum()
    min_group = min(n_treated, n_control)

    # 4. Outcome variance check
    treated_var = y[treated_mask].var()
    control_var = y[control_mask].var()

    results = {
        "overlap": overlap_results,
        "covariate_balance": balance_results,
        "max_smd": max_smd,
        "balance_ok": max_smd < 0.2,
        "sample_sizes": {
            "treated": int(n_treated),
            "control": int(n_control),
            "min_group": int(min_group),
            "total": int(len(treatment)),
        },
        "outcome_variance": {
            "treated": float(treated_var) if not np.isnan(treated_var) else 0,
            "control": float(control_var) if not np.isnan(control_var) else 0,
        },
        "warnings": overlap_results.get("warnings", []),
        "recommendations": [],
    }

    # Add recommendations
    if not overlap_results["overlap"]:
        results["recommendations"].append(
            "Insufficient propensity overlap. Consider: trimming extreme scores, "
            "adding covariates, or using different propensity model."
        )

    if max_smd > 0.2:
        results["recommendations"].append(
            f"Covariate imbalance detected (max SMD={max_smd:.2f}). "
            "Consider propensity score matching or weighting."
        )

    if min_group < 30:
        results["recommendations"].append(
            f"Small minimum group size ({min_group}). Results may be underpowered."
        )

    if min_group < 10:
        results["recommendations"].append(
            f"Very small group size ({min_group}). Consider collecting more data."
        )

    return results


def propensity_stratification_uplift(
    X: pd.DataFrame,
    treatment: pd.Series,
    y: pd.Series,
    propensity: pd.Series,
    n_strata: int = 5,
) -> Dict:
    """
    Estimate uplift within propensity score strata (ATS - Average Treatment Effect on Stratified).

    Reduces selection bias by comparing treated vs control within similar propensity groups.
    """
    # Create strata
    strata = pd.qcut(propensity, q=n_strata, labels=False, duplicates="drop")
    strata = pd.Series(strata, index=propensity.index)

    results = {}

    for s in range(n_strata):
        mask = strata == s
        if mask.sum() < 10:
            continue

        T_s = treatment[mask]
        y_s = y[mask]

        if T_s.nunique() < 2:
            continue

        # Simple difference in means
        ate = y_s[T_s == 1].mean() - y_s[T_s == 0].mean()

        # Mann-Whitney U test
        if len(T_s[T_s == 1]) > 1 and len(T_s[T_s == 0]) > 1:
            u_stat, p_val = mannwhitneyu(y_s[T_s == 1], y_s[T_s == 0], alternative="two-sided")
        else:
            u_stat, p_val = np.nan, np.nan

        results[s] = {
            "stratum": s,
            "n": int(mask.sum()),
            "n_treated": int(T_s.sum()),
            "n_control": int((T_s == 0).sum()),
            "ate": ate,
            "u_statistic": u_stat,
            "p_value": p_val,
            "propensity_mean": propensity[mask].mean(),
            "propensity_range": (propensity[mask].min(), propensity[mask].max()),
        }

    # Overall ATE (weighted by stratum size)
    total_n = sum(r["n"] for r in results.values())
    overall_ate = sum(r["ate"] * r["n"] / total_n for r in results.values())

    return {
        "strata": results,
        "overall_ate": overall_ate,
        "n_strata": len(results),
    }


# ============================================================================
# UPLIFT EVALUATION
# ============================================================================


def evaluate_uplift_model(
    X_test: pd.DataFrame,
    treatment_test: pd.Series,
    y_test: pd.Series,
    uplift_pred: pd.Series,
    n_bins: int = 10,
) -> Dict:
    """
    Evaluate uplift model using Qini coefficient, AUUC, and uplift@k.

    Returns dict with metrics and curves for plotting.
    """
    # Sort by predicted uplift (descending)
    order = np.argsort(-uplift_pred.values)
    y_sorted = y_test.iloc[order].values
    t_sorted = treatment_test.iloc[order].values

    # Qini curve
    n = len(y_sorted)
    bin_size = n // n_bins

    qini_x = [0]
    qini_y = [0]
    cum_uplift = 0

    for i in range(n_bins):
        start = i * bin_size
        end = min((i + 1) * bin_size, n)

        t_bin = t_sorted[start:end]
        y_bin = y_sorted[start:end]

        n_treated = t_bin.sum()
        n_control = len(t_bin) - n_treated

        if n_treated > 0 and n_control > 0:
            uplift_bin = y_bin[t_bin == 1].mean() - y_bin[t_bin == 0].mean()
            cum_uplift += uplift_bin * n_treated
        else:
            cum_uplift += 0

        qini_x.append(end / n)
        qini_y.append(cum_uplift / n)

    # Qini coefficient (area under Qini curve)
    qini = np.trapz(qini_y, qini_x)

    # AUUC (Area Under Uplift Curve) - normalized
    random_curve_x = qini_x
    random_curve_y = [x * qini_y[-1] for x in qini_x]
    auuc = np.trapz(qini_y, qini_x) - np.trapz(random_curve_y, random_curve_x)
    auuc = max(
        0, auuc / (qini_y[-1] - random_curve_y[-1]) if qini_y[-1] != random_curve_y[-1] else 0
    )

    # Uplift @ k
    top_k = min(n_bins, 3)
    top_k_end = min(top_k * bin_size, n)
    t_top = t_sorted[:top_k_end]
    y_top = y_sorted[:top_k_end]
    if t_top.sum() > 0 and (len(t_top) - t_top.sum()) > 0:
        uplift_at_k = y_top[t_top == 1].mean() - y_top[t_top == 0].mean()
    else:
        uplift_at_k = 0

    return {
        "qini_coefficient": qini,
        "auuc": auuc,
        "uplift_at_top_k": uplift_at_k,
        "qini_curve_x": qini_x,
        "qini_curve_y": qini_y,
        "random_curve_x": random_curve_x,
        "random_curve_y": random_curve_y,
    }


def qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_pred: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Qini curve points."""
    order = np.argsort(-uplift_pred)
    y_sorted = y_true[order]
    t_sorted = treatment[order]
    n = len(y_sorted)
    bin_size = n // n_bins

    qini_x = [0]
    qini_y = [0]
    cum_uplift = 0

    for i in range(n_bins):
        start = i * bin_size
        end = min((i + 1) * bin_size, n)
        t_bin = t_sorted[start:end]
        y_bin = y_sorted[start:end]

        n_t = t_bin.sum()
        n_c = len(t_bin) - n_t

        if n_t > 0 and n_c > 0:
            cum_uplift += (y_bin[t_bin == 1].mean() - y_bin[t_bin == 0].mean()) * n_t

        qini_x.append(end / n)
        qini_y.append(cum_uplift / n)

    return np.array(qini_x), np.array(qini_y)


# ============================================================================
# PROMO DECOMPOSITION (Incrementality / Forward Buy / Substitution)
# ============================================================================


def decompose_promo_lift(
    transactions_df: pd.DataFrame,
    promo_df: pd.DataFrame,
    baseline_window_days: int = 28,
    promo_window_days: int = 14,
) -> Dict:
    """
    Decompose promotional lift into:
    - True Incrementality: new demand that wouldn't exist otherwise
    - Forward Buy: stockpiling (pull-forward from future periods)
    - Substitution: cannibalization from other products
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # Mark promo periods
    df["is_promo"] = False
    for _, promo in promo_df.iterrows():
        mask = (
            (df["stockcode"] == promo["stockcode"])
            & (df["date"] >= promo["date"])
            & (df["date"] < promo["date"] + pd.Timedelta(days=promo_window_days))
        )
        df.loc[mask, "is_promo"] = True

    # Baseline: non-promo daily rate
    non_promo = df[~df["is_promo"]]
    promo = df[df["is_promo"]]

    baseline_daily_qty = (
        non_promo.groupby("stockcode")["quantity"].sum() / non_promo["date"].nunique()
    )
    promo_daily_qty = promo.groupby("stockcode")["quantity"].sum() / promo["date"].nunique()

    # Total lift
    lift = (promo_daily_qty / baseline_daily_qty - 1).fillna(0)

    # Forward buy: look at post-promo dip
    forward_buy = {}
    for sku in promo["stockcode"].unique():
        sku_promo = promo[promo["stockcode"] == sku]
        promo_end = sku_promo["date"].max()

        # Post-promo period
        post_start = promo_end + pd.Timedelta(days=1)
        post_end = post_start + pd.Timedelta(days=promo_window_days)

        post_promo = df[
            (df["stockcode"] == sku)
            & (df["date"] >= post_start)
            & (df["date"] <= post_end)
            & (~df["is_promo"])
        ]

        if len(post_promo) > 0:
            post_rate = post_promo["quantity"].sum() / post_promo["date"].nunique()
            base_rate = baseline_daily_qty.get(sku, 0)
            if base_rate > 0:
                forward_buy[sku] = max(0, (base_rate - post_rate) / base_rate)
            else:
                forward_buy[sku] = 0
        else:
            forward_buy[sku] = 0

    # Substitution: other products in same promo baskets
    substitution = {}
    for sku in promo["stockcode"].unique():
        promo_txns = promo[promo["stockcode"] == sku]["transaction_id"].unique()
        basket_other = df[
            (df["transaction_id"].isin(promo_txns)) & (df["stockcode"] != sku) & (~df["is_promo"])
        ]

        if len(basket_other) > 0:
            # Compare to baseline co-purchase
            baseline_other = df[(df["stockcode"] == sku) & (~df["is_promo"])][
                "transaction_id"
            ].unique()
            baseline_basket = df[
                (df["transaction_id"].isin(baseline_other)) & (df["stockcode"] != sku)
            ]

            promo_rate = basket_other["quantity"].sum() / len(promo_txns)
            base_rate = baseline_basket["quantity"].sum() / max(len(baseline_other), 1)
            substitution[sku] = max(0, (promo_rate - base_rate) / base_rate if base_rate > 0 else 0)
        else:
            substitution[sku] = 0

    return {
        "total_lift": lift.to_dict(),
        "forward_buy": forward_buy,
        "substitution": substitution,
        "incrementality": {
            sku: lift.get(sku, 0) - forward_buy.get(sku, 0) - substitution.get(sku, 0)
            for sku in lift.index
        },
    }


def promo_roi_analysis(
    transactions_df: pd.DataFrame,
    promo_df: pd.DataFrame,
    cost_per_promo: Optional[Dict[str, float]] = None,
    margin_pct: float = 0.3,
    promo_cost_pct: float = 0.15,
) -> pd.DataFrame:
    """Calculate ROI of promotions using incremental revenue."""
    decomp = decompose_promo_lift(transactions_df, promo_df)

    results = []
    for sku in decomp["total_lift"]:
        total_lift = decomp["total_lift"].get(sku, 0)
        inc = decomp["incrementality"].get(sku, 0)
        fwd = decomp["forward_buy"].get(sku, 0)
        sub = decomp["substitution"].get(sku, 0)

        # Estimate incremental revenue
        base_rev = transactions_df[transactions_df["stockcode"] == sku]["revenue"].sum()
        promo_days = promo_df[promo_df["stockcode"] == sku]["date"].nunique()
        if promo_days > 0:
            inc_rev = base_rev * inc / max(promo_days, 1) * promo_days
        else:
            inc_rev = 0

        # Promo cost
        if cost_per_promo and sku in cost_per_promo:
            promo_cost = cost_per_promo[sku]
        else:
            promo_cost = inc_rev * promo_cost_pct

        incremental_profit = inc_rev * margin_pct - promo_cost
        roi = (incremental_profit / promo_cost * 100) if promo_cost > 0 else 0

        results.append(
            {
                "stockcode": sku,
                "total_lift_pct": total_lift * 100,
                "incrementality_pct": inc * 100,
                "forward_buy_pct": fwd * 100,
                "substitution_pct": sub * 100,
                "incremental_revenue": inc_rev,
                "incremental_profit": incremental_profit,
                "promo_cost": promo_cost,
                "roi_pct": roi,
            }
        )

    return pd.DataFrame(results).sort_values("roi_pct", ascending=False)
