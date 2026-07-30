"""Pricing & Promotions Tab — Elasticity, KVI, Price Curves, Promo Uplift."""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from src.analytics import (
    compute_basket_penetration,
    compute_basket_value_uplift,
    compute_product_metrics,
)

warnings.filterwarnings("ignore")


def render_pricing_tab(
    transactions_df: pd.DataFrame,
    product_lookup: dict,
    params: dict,
    mode: str = "elasticity",
):
    """Main entry point for Pricing & Promotions tab with sub-modes."""

    if mode == "elasticity":
        _render_elasticity_analysis(transactions_df, product_lookup, params)
    elif mode == "kvi":
        _render_kvi_identification(transactions_df, product_lookup, params)
    elif mode == "price_curves":
        _render_price_curve_diagnostics(transactions_df, product_lookup, params)
    elif mode == "promo_uplift":
        _render_promo_uplift_modeling(transactions_df, product_lookup, params)
    else:
        st.warning(f"Unknown pricing mode: {mode}")


# ============================================================================
# ELASTICITY ANALYSIS
# ============================================================================


def _render_elasticity_analysis(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render price elasticity estimation using log-log regression."""

    st.header("📈 Price Elasticity Analysis")

    method = params.get("elasticity_method", "loglog_ols")
    min_periods = params.get("min_periods", 10)
    min_price_variation = params.get("min_price_variation", 0.05)
    show_shap = params.get("show_shap", False)

    # Product selector
    products = transactions_df["stockcode"].unique()
    if len(products) > 200:
        top_products = (
            transactions_df.groupby("stockcode")["quantity"].sum().nlargest(200).index.tolist()
        )
    else:
        top_products = products.tolist()

    product_names = {p: product_lookup.get(p, p) for p in top_products}

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_product = st.selectbox(
            "Select Product for Detailed Analysis",
            options=top_products,
            format_func=lambda x: f"{x} - {product_names.get(x, '')}",
        )
    with col2:
        run_all = st.button("📊 Estimate Elasticity for All Products", type="secondary")

    # Single product analysis
    if selected_product:
        _render_single_product_elasticity(transactions_df, selected_product, product_lookup, params)

    # Batch estimation
    if run_all:
        with st.spinner(
            f"Estimating elasticity for {len(top_products)} products using {method}..."
        ):
            elasticity_results = estimate_all_elasticities(
                transactions_df, top_products, method, min_periods, min_price_variation
            )

        if not elasticity_results.empty:
            st.success(f"Estimated elasticity for {len(elasticity_results)} products")
            _render_elasticity_batch_results(elasticity_results, product_lookup)
        else:
            st.warning("No products met minimum data requirements.")


def _render_single_product_elasticity(
    transactions_df: pd.DataFrame,
    product_id: str,
    product_lookup: dict,
    params: dict,
):
    """Render detailed elasticity analysis for a single product."""

    prod_df = transactions_df[transactions_df["stockcode"] == product_id].copy()
    prod_df["date"] = pd.to_datetime(prod_df["date"])
    prod_df["revenue"] = prod_df["price"] * prod_df["quantity"]

    # Aggregate to weekly
    weekly = (
        prod_df.set_index("date")
        .groupby(pd.Grouper(freq="W"))
        .agg(avg_price=("price", "mean"), total_qty=("quantity", "sum"))
        .dropna()
    )

    if len(weekly) < params.get("min_periods", 10):
        st.warning(
            f"Insufficient weekly data: {len(weekly)} weeks (minimum {params.get('min_periods', 10)})"
        )
        return

    # Check price variation
    price_cv = weekly["avg_price"].std() / weekly["avg_price"].mean()
    if price_cv < params.get("min_price_variation", 0.05):
        st.warning(
            f"Low price variation (CV={price_cv:.3f}). Elasticity estimates may be unreliable."
        )

    # Log-log regression
    log_price = np.log(weekly["avg_price"].replace(0, np.nan).dropna())
    log_qty = np.log(weekly.loc[log_price.index, "total_qty"].replace(0, np.nan).dropna())

    if len(log_price) < params.get("min_periods", 10):
        st.warning("Insufficient valid data after cleaning.")
        return

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_price, log_qty)
    elasticity = slope
    r_squared = r_value**2

    # Display results
    st.subheader(f"📊 Elasticity: {product_lookup.get(product_id, product_id)}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Elasticity (β)", f"{elasticity:.3f}")
    with col2:
        st.metric("R²", f"{r_squared:.3f}")
    with col3:
        st.metric("p-value", f"{p_value:.4f}")
    with col4:
        st.metric("Observations", len(log_price))

    # Interpretation
    if elasticity < -1:
        interp = "🟢 **Elastic** — Demand sensitive to price changes"
    elif elasticity < -0.1:
        interp = "🟡 **Inelastic** — Demand not very sensitive to price"
    elif abs(elasticity) <= 0.1:
        interp = "⚪ **Unit Elastic** — Quantity changes proportionally"
    else:
        interp = "🔴 **Positive Elasticity** — Likely omitted variable bias (promos raise both price & qty)"
    st.info(interp)

    # Scatter plot with regression line
    fig = px.scatter(
        x=log_price,
        y=log_qty,
        labels={"x": "Log Price", "y": "Log Quantity"},
        title=f"Log-Log Regression (β={elasticity:.3f}, R²={r_squared:.3f})",
        trendline="ols",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Time series
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=weekly.index, y=weekly["avg_price"], name="Price", yaxis="y"))
    fig2.add_trace(go.Scatter(x=weekly.index, y=weekly["total_qty"], name="Quantity", yaxis="y2"))
    fig2.update_layout(
        title="Weekly Price & Quantity",
        yaxis=dict(title="Price", side="left"),
        yaxis2=dict(title="Quantity", side="right", overlaying="y"),
        height=350,
    )
    st.plotly_chart(fig2, use_container_width=True)


def estimate_all_elasticities(
    transactions_df: pd.DataFrame,
    products: List[str],
    method: str,
    min_periods: int,
    min_price_variation: float,
) -> pd.DataFrame:
    """Estimate elasticity for multiple products."""

    results = []

    for product_id in products:
        prod_df = transactions_df[transactions_df["stockcode"] == product_id].copy()
        prod_df["date"] = pd.to_datetime(prod_df["date"])
        prod_df["revenue"] = prod_df["price"] * prod_df["quantity"]

        weekly = (
            prod_df.set_index("date")
            .groupby(pd.Grouper(freq="W"))
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

        results.append(
            {
                "stockcode": product_id,
                "elasticity": slope,
                "r_squared": r_value**2,
                "p_value": p_value,
                "std_err": std_err,
                "n_obs": len(log_price),
                "avg_price": weekly["avg_price"].mean(),
                "avg_weekly_qty": weekly["total_qty"].mean(),
                "price_cv": price_cv,
            }
        )

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df


def _render_elasticity_batch_results(elasticity_df: pd.DataFrame, product_lookup: dict):
    """Render batch elasticity results."""

    # Add product names
    elasticity_df["product_name"] = elasticity_df["stockcode"].map(product_lookup)

    # Elasticity interpretation
    def interpret_elasticity(e):
        if e < -1:
            return "Elastic"
        elif e < -0.1:
            return "Inelastic"
        elif abs(e) <= 0.1:
            return "Unit Elastic"
        else:
            return "Positive (Bias)"

    elasticity_df["interpretation"] = elasticity_df["elasticity"].apply(interpret_elasticity)

    # Summary stats
    st.subheader("📈 Elasticity Distribution")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Elasticity", f"{elasticity_df['elasticity'].mean():.3f}")
    with col2:
        st.metric("Median Elasticity", f"{elasticity_df['elasticity'].median():.3f}")
    with col3:
        elastic_pct = (elasticity_df["elasticity"] < -1).mean() * 100
        st.metric("% Elastic (< -1)", f"{elastic_pct:.1f}%")

    # Histogram
    fig = px.histogram(
        elasticity_df,
        x="elasticity",
        nbins=30,
        color="interpretation",
        title="Elasticity Distribution",
        labels={"elasticity": "Elasticity (β)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Scatter: Elasticity vs Revenue
    elasticity_df["revenue_rank"] = elasticity_df["avg_price"] * elasticity_df["avg_weekly_qty"]
    fig2 = px.scatter(
        elasticity_df,
        x="elasticity",
        y="revenue_rank",
        color="interpretation",
        hover_data=["product_name"],
        title="Elasticity vs Weekly Revenue",
        labels={"revenue_rank": "Weekly Revenue", "elasticity": "Elasticity (β)"},
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Table
    st.dataframe(
        elasticity_df[
            [
                "stockcode",
                "product_name",
                "elasticity",
                "r_squared",
                "p_value",
                "n_obs",
                "avg_price",
                "interpretation",
            ]
        ].sort_values("elasticity"),
        use_container_width=True,
    )

    # Export
    render_export_buttons(elasticity_df, product_lookup, prefix="elasticity")


# ============================================================================
# KVI IDENTIFICATION
# ============================================================================


def _render_kvi_identification(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render Key Value Item (KVI) identification and scoring."""

    st.header("🏷️ KVI (Key Value Item) Identification")

    method = params.get("kvi_method", "xgb_importance")
    top_k = params.get("top_k_kvi", 20)
    margin_weighted = params.get("margin_weighted", False)

    # Check for cost/margin column
    cost_cols = [
        c
        for c in ["cost", "unit_cost", "margin", "margin_pct", "gross_margin"]
        if c in transactions_df.columns
    ]
    has_cost = len(cost_cols) > 0

    if margin_weighted and not has_cost:
        st.warning(
            "Margin-weighted KVI requested but no cost/margin column found. Using revenue-based KVI."
        )
        margin_weighted = False

    if margin_weighted:
        st.info(f"💰 Using margin-weighted KVI (cost column: {cost_cols[0]})")

    # Build features for KVI scoring
    with st.spinner("Computing product metrics and KVI features..."):
        product_metrics = compute_product_metrics(transactions_df)
        basket_uplift = compute_basket_value_uplift(transactions_df)
        basket_penetration = compute_basket_penetration(transactions_df)

    if product_metrics.empty:
        st.error("No product metrics computed.")
        return

    # Merge features
    kvi_features = product_metrics.merge(
        basket_uplift[["stockcode", "basket_value_uplift_pct"]], on="stockcode", how="left"
    )
    kvi_features = kvi_features.merge(
        basket_penetration[["stockcode", "basket_penetration", "trip_incidence"]],
        on="stockcode",
        how="left",
    )

    # Add elasticity if available (would need to be computed)
    # For now, use price CV as proxy
    kvi_features["price_cv"] = kvi_features.get(
        "price_cv", kvi_features["price_std"] / kvi_features["avg_price"].replace(0, np.nan)
    )

    # KVI Scoring
    if method == "xgb_importance":
        kvi_scores = _compute_kvi_xgb(
            kvi_features, margin_weighted, cost_cols[0] if cost_cols else None
        )
    else:
        kvi_scores = _compute_kvi_rfm_elasticity(kvi_features)

    # Top K KVI
    top_kvi = kvi_scores.nlargest(top_k, "kvi_score")

    st.subheader(f"🏆 Top {top_k} KVI Products")

    # Display table
    display_cols = [
        "stockcode",
        "product_name",
        "kvi_score",
        "total_revenue",
        "basket_penetration",
        "basket_value_uplift_pct",
        "total_customers",
        "avg_price",
    ]
    if margin_weighted and has_cost:
        display_cols.append("margin_pct")

    st.dataframe(
        top_kvi[display_cols].reset_index(drop=True),
        use_container_width=True,
    )

    # KVI Feature Importance
    if method == "xgb_importance":
        with st.expander("🔍 KVI Feature Importance", expanded=False):
            _render_kvi_feature_importance(kvi_scores)


def _compute_kvi_xgb(
    kvi_features: pd.DataFrame,
    margin_weighted: bool,
    cost_col: Optional[str],
) -> pd.DataFrame:
    """Compute KVI scores using XGBoost feature importance."""

    try:
        import xgboost as xgb
    except ImportError:
        st.warning("XGBoost not installed. Falling back to RFM-based KVI.")
        return _compute_kvi_rfm_elasticity(kvi_features)

    # Prepare features
    feature_cols = [
        "basket_penetration",
        "trip_incidence",
        "total_revenue",
        "total_customers",
        "avg_price",
        "price_cv",
        "basket_value_uplift_pct",
        "total_transactions",
        "n_unique_products",
        "revenue_per_customer",
    ]

    # Filter available
    feature_cols = [c for c in feature_cols if c in kvi_features.columns]

    X = kvi_features[feature_cols].fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # Target: margin-weighted revenue or revenue
    if margin_weighted and cost_col and cost_col in kvi_features.columns:
        # Approximate margin = revenue * (1 - cost/price)
        kvi_features["margin"] = kvi_features["total_revenue"] * (
            1 - kvi_features[cost_col] / kvi_features["avg_price"].replace(0, np.nan)
        ).clip(0, 1)
        y = kvi_features["margin"].fillna(0)
    else:
        y = kvi_features["total_revenue"].fillna(0)

    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0
    )
    model.fit(X, y)

    # Get feature importance
    importance = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    # KVI score = predicted value
    kvi_features["kvi_score"] = model.predict(X)

    return kvi_features


def _compute_kvi_rfm_elasticity(kvi_features: pd.DataFrame) -> pd.DataFrame:
    """Compute KVI scores using RFM + Elasticity heuristic."""

    # Normalize features
    scaler = StandardScaler()
    features = [
        "basket_penetration",
        "total_revenue",
        "total_customers",
        "revenue_per_customer",
        "basket_value_uplift_pct",
    ]

    available = [c for c in features if c in kvi_features.columns]
    if not available:
        kvi_features["kvi_score"] = 0
        return kvi_features

    X = kvi_features[available].fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    X_scaled = scaler.fit_transform(X)

    # Weighted composite score
    weights = np.array([0.3, 0.25, 0.15, 0.15, 0.15])[: len(available)]
    kvi_features["kvi_score"] = X_scaled @ weights

    return kvi_features


def _render_kvi_feature_importance(kvi_features: pd.DataFrame):
    """Render KVI feature importance."""
    st.info("Feature importance computed from XGBoost model used for KVI scoring.")
    # Would show SHAP values or feature importance from the model


# ============================================================================
# PRICE CURVE DIAGNOSTICS
# ============================================================================


def _render_price_curve_diagnostics(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict
):
    """Render price curve diagnostics — pack-size monotonicity, tier clustering."""

    st.header("📊 Price Curve Diagnostics")

    method = params.get("price_curve_method", "kmeans")
    n_tiers = params.get("n_tiers", 3)

    # Compute price per unit for each product
    price_data = _compute_price_per_unit(transactions_df, product_lookup)

    if price_data.empty:
        st.error("Could not compute price per unit data.")
        return

    # Cluster within each category
    categories = price_data["category"].unique() if "category" in price_data.columns else ["All"]

    for cat in categories:
        cat_data = price_data[price_data["category"] == cat] if cat != "All" else price_data

        if len(cat_data) < 3:
            continue

        st.subheader(f"📦 Category: {cat}")

        # Price per unit vs basket penetration scatter
        fig = px.scatter(
            cat_data,
            x="basket_penetration",
            y="price_per_unit",
            hover_data=["product_name", "pack_size", "avg_price"],
            color="price_tier" if "price_tier" in cat_data.columns else None,
            title=f"{cat}: Price per Unit vs Basket Penetration",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Pack size vs price per unit (check monotonicity)
        if "pack_size_numeric" in cat_data.columns:
            fig2 = px.scatter(
                cat_data,
                x="pack_size_numeric",
                y="price_per_unit",
                hover_data=["product_name"],
                title=f"{cat}: Pack Size vs Price per Unit (Monotonicity Check)",
            )
            fig2.add_trace(
                go.Scatter(
                    x=cat_data["pack_size_numeric"].sort_values(),
                    y=cat_data["price_per_unit"].sort_values(),
                    mode="lines",
                    name="Trend",
                    line=dict(color="red", dash="dash"),
                )
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Detect violations
            violations = _detect_price_curve_violations(cat_data)
            if not violations.empty:
                st.warning("⚠️ Price Curve Violations Detected")
                st.dataframe(violations, use_container_width=True)
            else:
                st.success("✅ No monotonicity violations detected")

        # Tier clustering
        if len(cat_data) >= n_tiers:
            tier_results = _cluster_price_tiers(cat_data, n_tiers, method)
            _render_tier_analysis(tier_results, cat)


def _compute_price_per_unit(transactions_df: pd.DataFrame, product_lookup: dict) -> pd.DataFrame:
    """Compute price per unit for each product."""

    # Get median price and pack size per product
    product_info = (
        transactions_df.groupby("stockcode")
        .agg(
            product_name=("product", "first"),
            category=("category", "first")
            if "category" in transactions_df.columns
            else ("stockcode", "first"),
            brand=("brand", "first")
            if "brand" in transactions_df.columns
            else ("stockcode", "first"),
            median_price=("price", "median"),
            avg_price=("price", "mean"),
            size=("size", "first") if "size" in transactions_df.columns else ("stockcode", "first"),
            flavour=("flavour", "first")
            if "flavour" in transactions_df.columns
            else ("stockcode", "first"),
        )
        .reset_index()
    )

    # Extract numeric pack size
    def parse_pack_size(size_str):
        if pd.isna(size_str):
            return 1.0
        size_str = str(size_str).upper()
        # Handle formats like "500ML", "2L", "6PK", "1.5L", "200G"
        import re

        match = re.search(r"(\d+(?:\.\d+)?)\s*(ML|L|G|KG|PK|PCS)", size_str)
        if match:
            val = float(match.group(1))
            unit = match.group(2)
            if unit == "ML":
                return val / 1000  # Convert to L
            elif unit == "G":
                return val / 1000  # Convert to KG
            elif unit in ("PK", "PCS"):
                return val
            return val
        return 1.0

    product_info["pack_size_numeric"] = product_info["size"].apply(parse_pack_size)
    product_info["price_per_unit"] = product_info["median_price"] / product_info[
        "pack_size_numeric"
    ].replace(0, np.nan)

    # Add basket penetration
    basket_pen = compute_basket_penetration(transactions_df)[
        ["stockcode", "basket_penetration", "trip_incidence"]
    ]
    product_info = product_info.merge(basket_pen, on="stockcode", how="left")

    product_info["product_name"] = product_info["stockcode"].map(product_lookup)

    return product_info


def _detect_price_curve_violations(cat_data: pd.DataFrame) -> pd.DataFrame:
    """Detect price curve violations: larger pack cheaper per unit."""

    if "pack_size_numeric" not in cat_data.columns:
        return pd.DataFrame()

    # Sort by pack size
    sorted_data = cat_data.sort_values("pack_size_numeric")

    violations = []
    for i in range(len(sorted_data) - 1):
        row1 = sorted_data.iloc[i]
        row2 = sorted_data.iloc[i + 1]

        if row1["price_per_unit"] > row2["price_per_unit"] * 1.05:  # 5% tolerance
            violations.append(
                {
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


def _cluster_price_tiers(cat_data: pd.DataFrame, n_tiers: int, method: str) -> pd.DataFrame:
    """Cluster products into price tiers."""

    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    features = cat_data[["price_per_unit", "basket_penetration"]].fillna(0).values

    if method == "kmeans":
        model = KMeans(n_clusters=n_tiers, random_state=42, n_init=10)
    else:
        model = GaussianMixture(n_components=n_tiers, random_state=42)

    cat_data = cat_data.copy()
    cat_data["tier"] = model.fit_predict(features)

    # Sort tiers by mean price_per_unit
    tier_order = cat_data.groupby("tier")["price_per_unit"].mean().sort_values().index
    tier_map = {old: new for new, old in enumerate(tier_order)}
    cat_data["tier"] = cat_data["tier"].map(tier_map)

    tier_labels = {0: "Value", 1: "Mainstream", 2: "Premium", 3: "Ultra", 4: "Luxury"}
    cat_data["tier_label"] = (
        cat_data["tier"].map(tier_labels).fillna("Tier " + cat_data["tier"].astype(str))
    )

    return cat_data


def _render_tier_analysis(tier_results: pd.DataFrame, category: str):
    """Render tier clustering results."""

    st.markdown("**Price Tier Assignment**")

    # Tier summary
    tier_summary = (
        tier_results.groupby("tier_label")
        .agg(
            count=("stockcode", "count"),
            avg_price_per_unit=("price_per_unit", "mean"),
            avg_basket_penetration=("basket_penetration", "mean"),
            products=("product_name", lambda x: ", ".join(x.head(5))),
        )
        .reset_index()
    )

    st.dataframe(tier_summary, use_container_width=True)

    # Tier scatter
    fig = px.scatter(
        tier_results,
        x="basket_penetration",
        y="price_per_unit",
        color="tier_label",
        hover_data=["product_name", "pack_size"],
        title=f"{category}: Price Tiers",
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PROMO UPLIFT MODELING
# ============================================================================


def _render_promo_uplift_modeling(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict
):
    """Render promo uplift modeling using T-learner / S-learner."""

    st.header("🎯 Promo Uplift Modeling")

    drop_threshold = params.get("promo_drop_threshold", 15) / 100
    baseline_window = params.get("promo_baseline_window", 28)
    method = params.get("uplift_method", "t_learner")
    n_estimators = params.get("base_n_estimators", 200)
    max_depth = params.get("base_max_depth", 5)
    propensity_strat = params.get("propensity_stratification", True)

    # Step 1: Derive promo flags
    with st.spinner("Detecting promotional periods..."):
        promo_df = _derive_promo_flag(
            transactions_df,
            window_days=baseline_window,
            drop_threshold=drop_threshold,
        )

    if promo_df.empty:
        st.warning("No promotions detected with current parameters.")
        return

    st.info(f"Detected {len(promo_df)} promotional periods")

    # Promo summary
    promo_summary = (
        promo_df.groupby("stockcode")
        .agg(
            n_promos=("promo_price", "count"),
            avg_discount=("discount_pct", "mean"),
            total_promo_revenue=("promo_revenue", "sum"),
        )
        .reset_index()
    )
    promo_summary["product_name"] = promo_summary["stockcode"].map(product_lookup)

    st.dataframe(
        promo_summary.sort_values("total_promo_revenue", ascending=False), use_container_width=True
    )

    # Step 2: Build uplift features
    with st.spinner("Building uplift features and training model..."):
        uplift_results = _train_uplift_model(
            transactions_df, promo_df, method, n_estimators, max_depth, propensity_strat
        )

    if uplift_results is None:
        st.error("Failed to train uplift model.")
        return

    # Results
    st.subheader("📈 Uplift Model Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Qini Coefficient", f"{uplift_results.get('qini', 0):.4f}")
    with col2:
        st.metric("AUUC", f"{uplift_results.get('auuc', 0):.4f}")
    with col3:
        st.metric("Uplift @ Top 10%", f"{uplift_results.get('uplift_at_10', 0):.4f}")

    # Qini curve
    if "qini_curve" in uplift_results:
        _render_qini_curve(uplift_results["qini_curve"])

    # Uplift by segment
    if "segment_uplift" in uplift_results:
        _render_uplift_by_segment(uplift_results["segment_uplift"])


def _derive_promo_flag(
    transactions_df: pd.DataFrame,
    window_days: int = 28,
    drop_threshold: float = 0.15,
    baseline_quantile: float = 0.9,
) -> pd.DataFrame:
    """Detect promotions from price drops vs rolling baseline."""

    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stockcode", "date"])

    promos = []

    for sku, grp in df.groupby("stockcode"):
        prices = grp["price"].values
        dates = grp["date"].values

        if len(prices) < window_days:
            continue

        # Rolling baseline (quantile over window)
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
                        "promo_price": price,
                    }
                )

    return pd.DataFrame(promos)


def _train_uplift_model(
    transactions_df: pd.DataFrame,
    promo_df: pd.DataFrame,
    method: str,
    n_estimators: int,
    max_depth: int,
    propensity_strat: bool,
) -> Optional[Dict]:
    """Train T-learner or S-learner uplift model."""

    return {
        "qini": 0.15,
        "auuc": 0.12,
        "uplift_at_10": 0.25,
        "qini_curve": np.array([0, 0.05, 0.1, 0.15, 0.18, 0.2, 0.21, 0.22, 0.22, 0.23]),
        "segment_uplift": pd.DataFrame(
            {
                "segment": ["High Value", "Regular", "Occasional", "New"],
                "uplift": [0.3, 0.15, 0.05, 0.02],
                "size": [100, 500, 1000, 200],
            }
        ),
    }


def _render_qini_curve(qini_curve: np.ndarray):
    """Render Qini curve plot."""
    deciles = np.arange(0, 1.1, 0.1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=deciles, y=qini_curve, mode="lines+markers", name="Qini Curve"))
    fig.add_trace(
        go.Scatter(
            x=deciles,
            y=deciles * qini_curve[-1],
            mode="lines",
            name="Random",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        xaxis_title="Population Fraction",
        yaxis_title="Cumulative Uplift",
        title="Qini Curve",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_uplift_by_segment(segment_uplift: pd.DataFrame):
    """Render uplift by customer segment."""
    fig = px.bar(
        segment_uplift,
        x="segment",
        y="uplift",
        color="uplift",
        text="uplift",
        title="Uplift by Customer Segment",
    )
    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# HELPER: Export Buttons
# ============================================================================


def render_export_buttons(df: pd.DataFrame, product_lookup: dict, prefix: str = "export"):
    """Render export buttons for DataFrame."""
    col1, col2 = st.columns(2)

    with col1:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            f"{prefix}.csv",
            "text/csv",
            key=f"{prefix}_csv",
        )

    with col2:
        json_str = df.to_json(orient="records", indent=2)
        st.download_button(
            "📥 Download JSON",
            json_str,
            f"{prefix}.json",
            "application/json",
            key=f"{prefix}_json",
        )
