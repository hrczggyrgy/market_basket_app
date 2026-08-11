"""KVI (Key Value Item) scoring.

Computes KVI scores via XGBoost + SHAP (preferred) or weighted heuristic.
Features: basket penetration, revenue, halo (basket uplift), elasticity, repeat rate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.analytics.schemas import KVI_ELASTICITY_QUADRANT, KVI_SCORES, check


def _create_kvi_features(
    transactions_df: pd.DataFrame,
    elasticity_df: Optional[pd.DataFrame] = None,
    elasticity_status_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Assemble KVI feature table per SKU."""
    from src.analytics.basket_metrics import compute_basket_penetration
    from src.analytics.performance import compute_product_metrics

    product_metrics = compute_product_metrics(transactions_df)
    if product_metrics.empty:
        return pd.DataFrame()

    # Basket penetration - drop product_metrics' 'penetration' (customer penetration)
    # and use basket_penetration's 'penetration' (basket/trip incidence)
    basket_pen = compute_basket_penetration(transactions_df)
    pm = product_metrics.drop(columns=["penetration"], errors="ignore")

    kvi_features = pm.merge(
        basket_pen[["stockcode", "basket_count", "penetration", "revenue_share"]], on="stockcode", how="left"
    )

    # Carry category through to the feature table (may be provided or inferred upstream)
    if "category" not in kvi_features.columns and "category" in transactions_df.columns:
        cat_lookup = (
            transactions_df[["stockcode", "category"]]
            .drop_duplicates(subset="stockcode")
            .dropna(subset=["category"])
        )
        kvi_features = kvi_features.merge(cat_lookup, on="stockcode", how="left")
    total_baskets = float(transactions_df["transaction_id"].nunique()) if "transaction_id" in transactions_df.columns else 0.0
    kvi_features["trip_incidence"] = (
        kvi_features["basket_count"] / total_baskets if total_baskets > 0 else 0.0
    )
    kvi_features = kvi_features.rename(columns={"penetration": "basket_penetration", "revenue": "total_revenue"})

    # Elasticity features
    if elasticity_df is not None and not elasticity_df.empty:
        elast_cols = ["stockcode"]
        if "elasticity" in elasticity_df.columns:
            elast_cols.append("elasticity")
        if "r_squared" in elasticity_df.columns:
            elast_cols.append("r_squared")
        if "price_cv" in elasticity_df.columns:
            elast_cols.append("price_cv")
        kvi_features = kvi_features.merge(elasticity_df[elast_cols], on="stockcode", how="left")

    # |elasticity|: leave NaN when not estimable instead of fabricating a 0.0.
    # A missing elasticity means "not estimable", never "perfectly inelastic".
    if "abs_elasticity" not in kvi_features.columns and "elasticity" in kvi_features.columns:
        kvi_features["abs_elasticity"] = kvi_features["elasticity"].abs()
    if "abs_elasticity" not in kvi_features.columns:
        kvi_features["abs_elasticity"] = np.nan

    # Explicit estimability status carried on every row.
    if elasticity_status_df is not None and not elasticity_status_df.empty:
        lookup = elasticity_status_df[["stockcode", "elasticity_status"]]
        kvi_features = kvi_features.merge(lookup, on="stockcode", how="left")
        kvi_features["elasticity_status"] = kvi_features["elasticity_status"].fillna("unavailable")
    elif "elasticity" in kvi_features.columns:
        kvi_features["elasticity_status"] = "estimated"
    else:
        kvi_features["elasticity_status"] = "unavailable"

    # Category revenue share - use stockcode as fallback category
    if "category" not in kvi_features.columns:
        kvi_features["category"] = "UNKNOWN"
    cat_rev = kvi_features.groupby("category")["total_revenue"].sum()
    total_rev = kvi_features["total_revenue"].sum()
    kvi_features["category_revenue_share"] = kvi_features["category"].map(cat_rev / total_rev)

    # Repeat rate - use transactions/customers as proxy
    if "customers" in kvi_features.columns and "transactions" in kvi_features.columns:
        kvi_features["repeat_rate"] = kvi_features["transactions"] / kvi_features["customers"]
    else:
        kvi_features["repeat_rate"] = 0.0

    # Fill numeric gaps EXCEPT abs_elasticity: NaN there means "not estimable",
    # which is information, not a missing value to impute as 0.
    fill_cols = [
        c
        for c in kvi_features.columns
        if c not in ("abs_elasticity", "stockcode", "category", "elasticity_status")
        and pd.api.types.is_numeric_dtype(kvi_features[c])
    ]
    kvi_features[fill_cols] = kvi_features[fill_cols].fillna(0).replace([np.inf, -np.inf], 0)
    return kvi_features


def compute_kvi_score(
    transactions_df: pd.DataFrame,
    elasticity_df: Optional[pd.DataFrame] = None,
    cost_col: Optional[str] = None,
    margin_pct: Optional[float] = None,
    method: str = "heuristic",
    elasticity_status_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """KVI scoring: XGBoost + SHAP (method='xgb') or heuristic (method='heuristic').

    ``elasticity_status_df`` (optional output of ``compute_elasticity_status``)
    carries the per-SKU estimability state so that SKUs without a usable
    elasticity are labelled explicitly (e.g. ``insufficient_variation``) rather
    than silently treated as perfectly inelastic.
    """
    features = _create_kvi_features(transactions_df, elasticity_df, elasticity_status_df)
    if features.empty:
        return check(pd.DataFrame(columns=list(KVI_SCORES.columns)), KVI_SCORES, allow_empty=True)

    if method == "xgb":
        return _kvi_xgb(features, cost_col, margin_pct)
    return _kvi_heuristic(features, cost_col, margin_pct)


def _kvi_heuristic(
    features: pd.DataFrame,
    cost_col: Optional[str] = None,
    margin_pct: Optional[float] = None,
) -> pd.DataFrame:
    """Weighted heuristic KVI score."""
    # Target metric
    if cost_col and cost_col in features.columns:
        features["margin"] = features["total_revenue"] * (
            1 - features[cost_col] / features["avg_price"].replace(0, np.nan)
        ).clip(0, 1)
    elif margin_pct:
        features["margin"] = features["total_revenue"] * margin_pct

    # Revenue-per-customer (customer value), derived when missing
    if (
        "revenue_per_customer" not in features.columns
        and "total_revenue" in features.columns
        and "customers" in features.columns
    ):
        features["revenue_per_customer"] = (
            features["total_revenue"] / features["customers"].replace(0, np.nan)
        )

    feature_cols = [
        "basket_penetration",
        "total_revenue",
        "abs_elasticity",
        "customers",
        "revenue_per_customer",
    ]
    feature_cols = [c for c in feature_cols if c in features.columns]

    if not feature_cols:
        features["kvi_score"] = 0
        return _format_kvi_output(features)

    X = features[feature_cols].replace([np.inf, -np.inf], np.nan)
    # Missing elasticity (not estimable) is imputed to the median of estimable
    # SKUs rather than a fabricated 0, so it neither vanishes nor reads as
    # "perfectly inelastic" in the composite score.
    if "abs_elasticity" in X.columns:
        med = X["abs_elasticity"].median()
        if pd.isna(med):
            med = 0.5
        X["abs_elasticity"] = X["abs_elasticity"].fillna(med)
    X = X.fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Weights: penetration (0.30), revenue (0.25), elasticity (0.15),
    # customer reach (0.10), revenue-per-customer / customer value (0.20).
    weights = np.array([0.30, 0.25, 0.15, 0.10, 0.20])[: len(feature_cols)]
    weights = weights / weights.sum()

    features["kvi_score"] = X_scaled @ weights
    return _format_kvi_output(features)


def _kvi_xgb(
    features: pd.DataFrame,
    cost_col: Optional[str] = None,
    margin_pct: Optional[float] = None,
) -> pd.DataFrame:
    """XGBoost + SHAP KVI scoring with out-of-fold predictions."""
    try:
        import shap
        import xgboost as xgb
    except ImportError:
        return _kvi_heuristic(features, cost_col, margin_pct)

    if cost_col and cost_col in features.columns:
        features["margin"] = features["total_revenue"] * (
            1 - features[cost_col] / features["avg_price"].replace(0, np.nan)
        ).clip(0, 1)
        y = features["margin"].fillna(0)
    elif margin_pct:
        features["margin"] = features["total_revenue"] * margin_pct
        y = features["margin"]
    else:
        y = features["total_revenue"].fillna(0)

    if (
        "revenue_per_customer" not in features.columns
        and "total_revenue" in features.columns
        and "customers" in features.columns
    ):
        features["revenue_per_customer"] = (
            features["total_revenue"] / features["customers"].replace(0, np.nan)
        )

    feature_cols = [
        "basket_penetration",
        "trip_incidence",
        "total_revenue",
        "customers",
        "avg_price",
        "price_cv",
        "revenue_per_customer",
        "abs_elasticity",
        "category_revenue_share",
        "r_squared",
    ]
    feature_cols = [c for c in feature_cols if c in features.columns]

    X = features[feature_cols].replace([np.inf, -np.inf], np.nan)
    # Missing elasticity imputed to median of estimable SKUs (not a fabricated 0).
    if "abs_elasticity" in X.columns:
        med = X["abs_elasticity"].median()
        if pd.isna(med):
            med = 0.5
        X["abs_elasticity"] = X["abs_elasticity"].fillna(med)
    X = X.fillna(0)

    # Cross-validated out-of-fold predictions as KVI score
    from sklearn.model_selection import KFold, cross_val_predict

    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    features["kvi_score"] = oof_preds

    # SHAP explanations on full-data fit
    model.fit(X, y)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        features["kvi_shap_importance"] = str(
            dict(zip(feature_cols, np.mean(np.abs(shap_values), axis=0), strict=True))
        )
    except Exception:
        features["kvi_feature_importance"] = str(
            dict(zip(feature_cols, model.feature_importances_, strict=True))
        )

    return _format_kvi_output(features)


def _format_kvi_output(features: pd.DataFrame) -> pd.DataFrame:
    """Select and validate KVI output columns."""
    output_cols = [
        "stockcode",
        "category",
        "kvi_score",
        "total_revenue",
        "basket_penetration",
        "trip_incidence",
        "abs_elasticity",
        "elasticity_status",
    ]
    output_cols = [c for c in output_cols if c in features.columns]
    table = features[output_cols].copy()
    if "kvi_score" in table.columns and len(table) > 0:
        lo, hi = float(table["kvi_score"].min()), float(table["kvi_score"].max())
        if hi > lo:
            table["kvi_score"] = (table["kvi_score"] - lo) / (hi - lo)
        else:
            table["kvi_score"] = 0.5
    return check(table, KVI_SCORES)


def compute_kvi_elasticity_quadrant(
    kvi_df: pd.DataFrame,
    elasticity_threshold: float = 1.0,
) -> pd.DataFrame:
    """Map SKUs into a KVI x elasticity strategy quadrant.

    Axes:
    - kvi_score in [0, 1] (higher = more strategically important).
    - abs_elasticity = |own-price elasticity| (higher = more price sensitive).

    Quadrants (median split on KVI, |elasticity| = 1 as the demand regime):
    - advocate:   high KVI + elastic  -> traffic drivers; price defensively,
      never give them up to private label.
    - protect:    high KVI + inelastic-> keep in assortment and on shelf; they
      can carry margin without volume risk.
    - promote:    low KVI + elastic  -> use as promotional bait / price levers;
      safe to trade down or delist.
    - defer:      low KVI + inelastic-> slow movers with no price lever; review
      assortment depth last.
    - unknown:    elasticity not estimable -> KVI x elasticity strategy cannot
      be assigned. This is a separate state: it is NOT "inelastic" (0.0).

    Args:
        kvi_df: Output of ``compute_kvi_score`` (KVI_SCORES contract).
        elasticity_threshold: |elasticity| at which demand flips to elastic.

    Returns:
        DataFrame validated against KVI_ELASTICITY_QUADRANT (empty input yields
        an empty, validated frame).
    """
    empty = pd.DataFrame(columns=list(KVI_ELASTICITY_QUADRANT.columns))
    if kvi_df is None or kvi_df.empty:
        return check(empty, KVI_ELASTICITY_QUADRANT, allow_empty=True)

    required = {"stockcode", "kvi_score", "abs_elasticity", "category", "total_revenue"}
    if not required.issubset(kvi_df.columns):
        return check(empty, KVI_ELASTICITY_QUADRANT, allow_empty=True)

    df = kvi_df.copy()
    # A missing elasticity must NOT be coerced to 0.0 (which reads as inelastic).
    if "elasticity_status" not in df.columns:
        df["elasticity_status"] = np.where(df["abs_elasticity"].notna(), "estimated", "unavailable")
    kvi_median = float(df["kvi_score"].median())

    def _quadrant(row: pd.Series) -> str:
        if row["elasticity_status"] not in ("estimated", "weak") or pd.isna(row["abs_elasticity"]):
            return "unknown"
        high_kvi = row["kvi_score"] >= kvi_median
        elastic = row["abs_elasticity"] >= elasticity_threshold
        if high_kvi and elastic:
            return "advocate"
        if high_kvi:
            return "protect"
        if elastic:
            return "promote"
        return "defer"

    df["quadrant"] = df.apply(_quadrant, axis=1)
    table = df[list(KVI_ELASTICITY_QUADRANT.columns)].reset_index(drop=True)
    return check(table, KVI_ELASTICITY_QUADRANT)
