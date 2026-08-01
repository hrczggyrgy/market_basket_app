"""Pricing & Promotions Tab — Elasticity, KVI, Price Curves, Promo Uplift."""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.preprocessing import StandardScaler

from src.analytics import (
    build_uplift_dataset,
    compute_basket_penetration,
    compute_basket_value_uplift,
    compute_product_metrics,
    compute_kvi_composite_df,
    compute_price_ladder_df,
    diagnose_price_curves_1d,
    diagnose_price_curves_multivariate,
    estimate_bayesian_hierarchical_elasticity,
    evaluate_uplift_model,
    run_validation,
    train_s_learner_uplift,
    train_t_learner_uplift,
)
from src.analytics.sufficiency import (
    assess_data_sufficiency,
    format_sufficiency_summary,
)
from src.utils.cache import get_trace_cache, trace_cache_key
from src.ui.insight_header import render_result_context, render_elasticity_context, render_uplift_context
from src.ui.data_quality import render_data_quality_expander

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Module-level helpers (must be defined before any function that calls them)
# ---------------------------------------------------------------------------


def _derive_promo_flag(
    transactions_df: pd.DataFrame,
    window_days: int = 28,
    drop_threshold: float = 0.15,
) -> pd.DataFrame:
    """Detect promotional periods by comparing price to rolling baseline.

    For each SKU, computes a rolling median baseline price and flags
    observations where the price drops more than `drop_threshold` below
    the baseline as promotional.

    Args:
        transactions_df: Raw transaction DataFrame with columns:
            stockcode, date, price, quantity.
        window_days: Rolling window (in days) for baseline price.
        drop_threshold: Relative price drop to flag as promo (e.g. 0.15 = 15%).

    Returns:
        DataFrame with promo periods: stockcode, date, price, baseline_price,
        discount_pct, promo_price, promo_revenue.
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    promos = []
    for sku, group in df.groupby("stockcode"):
        group = group.sort_values("date")
        prices = group["price"].values
        dates = group["date"].values
        quantities = group["quantity"].values

        # Rolling median baseline
        price_series = pd.Series(prices, index=group["date"])
        baseline = price_series.rolling(f"{window_days}D", min_periods=3).median().values

        for i, (price, date, qty) in enumerate(zip(prices, dates, quantities)):
            base = baseline[i]
            if base is not None and not np.isnan(base) and base > 0:
                if price < base * (1 - drop_threshold):
                    discount = (base - price) / base
                    promos.append(
                        {
                            "stockcode": sku,
                            "date": date,
                            "price": price,
                            "baseline_price": base,
                            "discount_pct": discount * 100,
                            "promo_price": price,
                            "promo_revenue": price * qty,
                        }
                    )

    return pd.DataFrame(promos)


def persistent_tabs(tab_labels: List[str], key: str, default_tab: int = 0) -> int:
    """Render Streamlit tabs and persist the selected index in session_state.

    Args:
        tab_labels: List of tab display labels.
        key: Unique session_state key to persist selection.
        default_tab: Index of the tab selected on first render.

    Returns:
        Index (int) of the currently selected tab.
    """
    tabs = st.tabs(tab_labels)
    selected = st.session_state.get(key, default_tab)
    # st.tabs renders all tabs; we track which one the user is on via
    # a radio hidden in the sidebar so the state survives reruns.
    with st.sidebar:
        selected = st.radio(
            f"__{key}_nav",
            options=list(range(len(tab_labels))),
            format_func=lambda i: tab_labels[i],
            index=selected,
            key=key,
            label_visibility="collapsed",
        )
    return int(selected)


def render_analytics_export(df: pd.DataFrame, prefix: str = "export"):
    """Render CSV + JSON download buttons for any analytics result DataFrame."""
    if df is None or df.empty:
        return
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            f"{prefix}.csv",
            "text/csv",
            key=f"{prefix}_csv_export",
        )
    with col2:
        st.download_button(
            "📥 Download JSON",
            df.to_json(orient="records", indent=2),
            f"{prefix}.json",
            "application/json",
            key=f"{prefix}_json_export",
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def render_pricing_tab(
    transactions_df: pd.DataFrame,
    product_lookup: dict,
    params: dict,
    pipeline: Any = None,
    mode: str = "elasticity",
):
    """Main entry point for Pricing & Promotions tab with sub-modes."""

    if mode == "elasticity":
        _render_elasticity_analysis(transactions_df, product_lookup, params, pipeline)
    elif mode == "kvi":
        _render_kvi_identification(transactions_df, product_lookup, params, pipeline)
    elif mode == "kvi_composite":
        _render_kvi_composite(transactions_df, product_lookup, params, pipeline)
    elif mode == "price_ladder":
        _render_price_ladder(transactions_df, product_lookup, params, pipeline)
    elif mode == "price_curves":
        _render_price_curve_diagnostics(transactions_df, product_lookup, params, pipeline)
    elif mode == "promo_uplift":
        _render_promo_uplift_modeling(transactions_df, product_lookup, params, pipeline)
    elif mode == "benchmark":
        _render_elasticity_benchmark(params, pipeline)
    else:
        st.warning(f"Unknown pricing mode: {mode}")


# ============================================================================
# ELASTICITY ANALYSIS
# ============================================================================


def _render_elasticity_analysis(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict, pipeline: dict = None
):
    """Render price elasticity estimation using log-log regression."""

    st.header("📈 Price Elasticity Analysis")
    st.caption(
        "Estimates price elasticity of demand from observational transaction data. "
        "**Observational only — not causal.** Confounded by promotions, seasonality, stockouts, competitor actions."
    )

    render_data_quality_expander(transactions_df, "elasticity", params, expanded=False)

    method = params.get("elasticity_method", "loglog_ols")
    min_periods = params.get("min_periods", 10)
    min_price_variation = params.get("min_price_variation", 0.05)

    cached_elasticity = pipeline.elasticity_results if pipeline else None
    if cached_elasticity is not None and not cached_elasticity.empty:
        st.info("Using cached elasticity results from pipeline")

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

    if selected_product:
        _render_single_product_elasticity(transactions_df, selected_product, product_lookup, params)

    if run_all:
        with st.spinner(
            f"Estimating elasticity for {len(top_products)} products using {method}..."
        ):
            elasticity_results = estimate_all_elasticities(
                transactions_df, top_products, method, min_periods, min_price_variation, params
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

    method = params.get("elasticity_method", "loglog_ols")

    if method == "bayesian_hierarchical":
        with st.spinner("Running Bayesian hierarchical model..."):
            all_results = _run_bayesian_elasticity_cached(
                transactions_df,
                params,
                params.get("min_periods", 10),
                params.get("min_price_variation", 0.05),
            )
        if all_results.empty:
            st.warning("Bayesian model did not converge.")
            return
        row = all_results[all_results["stockcode"] == product_id]
        if row.empty:
            st.info("This product did not meet minimum data requirements for Bayesian estimation.")
            return
        row = row.iloc[0]

        # Fix #24: _run_bayesian_elasticity_cached renames elasticity_mean -> elasticity;
        # read 'elasticity', not 'elasticity_mean'.
        render_elasticity_context(
            sku=product_lookup.get(product_id, product_id),
            elasticity=row["elasticity"],
            n_obs=row.get("n_obs", 0),
            price_cv=row.get("price_cv", 0),
            n_price_points=row.get("n_price_points", 0),
            method="Bayesian Hierarchical",
            hdi_lower=row["elasticity_hdi_lower"],
            hdi_upper=row["elasticity_hdi_upper"],
        )

        if params.get("bayesian_mode", "").startswith("full"):
            _render_trace_diagnostics()
        return

    prod_df = transactions_df[transactions_df["stockcode"] == product_id].copy()
    prod_df["date"] = pd.to_datetime(prod_df["date"])
    prod_df["revenue"] = prod_df["price"] * prod_df["quantity"]

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

    price_cv = weekly["avg_price"].std() / weekly["avg_price"].mean()
    n_price_points = weekly["avg_price"].nunique()

    if price_cv < params.get("min_price_variation", 0.05):
        st.warning(
            f"Low price variation (CV={price_cv:.3f}). Elasticity estimates may be unreliable."
        )
    else:
        st.info(f"Price CV: {price_cv:.3f} — sufficient variation ✅")

    log_price = np.log(weekly["avg_price"].replace(0, np.nan).dropna())
    log_qty = np.log(weekly.loc[log_price.index, "total_qty"].replace(0, np.nan).dropna())

    if len(log_price) < params.get("min_periods", 10):
        st.warning("Insufficient valid data after cleaning.")
        return

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_price, log_qty)
    elasticity = slope
    r_squared = r_value**2

    render_elasticity_context(
        sku=product_lookup.get(product_id, product_id),
        elasticity=elasticity,
        n_obs=len(log_price),
        price_cv=price_cv,
        n_price_points=n_price_points,
        method="Log-log OLS",
    )

    fig = px.scatter(
        x=log_price,
        y=log_qty,
        labels={"x": "Log Price", "y": "Log Quantity"},
        title=f"Log-Log Regression (β={elasticity:.3f}, R²={r_squared:.3f})",
        trendline="ols",
    )
    st.plotly_chart(fig, use_container_width=True)

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


def _run_bayesian_elasticity_cached(
    transactions_df: pd.DataFrame,
    params: dict,
    min_periods: int = 10,
    min_price_variation: float = 0.05,
) -> pd.DataFrame:
    """Run Bayesian hierarchical elasticity with trace caching."""
    bayesian_mode = params.get("bayesian_mode", "fast (ADVI)")
    want_trace = bayesian_mode.startswith("full")
    model_config = {
        "min_periods": min_periods,
        "min_price_variation": min_price_variation,
        "bayesian_mode": bayesian_mode,
    }
    cache_key = trace_cache_key(transactions_df, model_config)

    cache = get_trace_cache()
    if cache_key in cache:
        return cache[cache_key]

    if want_trace:
        result, trace_obj = estimate_bayesian_hierarchical_elasticity(
            transactions_df,
            min_periods=min_periods,
            min_price_variation=min_price_variation,
            bayesian_mode=bayesian_mode,
            return_trace=True,
        )
        if trace_obj is not None:
            st.session_state["_bayesian_trace"] = trace_obj
    else:
        result = estimate_bayesian_hierarchical_elasticity(
            transactions_df,
            min_periods=min_periods,
            min_price_variation=min_price_variation,
            bayesian_mode=bayesian_mode,
        )
    if not result.empty:
        result = result.rename(columns={"elasticity_mean": "elasticity"})
        result["method"] = "bayesian_hierarchical"
    cache[cache_key] = result
    return result


def estimate_all_elasticities(
    transactions_df: pd.DataFrame,
    products: List[str],
    method: str,
    min_periods: int,
    min_price_variation: float,
    params: dict | None = None,
) -> pd.DataFrame:
    """Estimate elasticity for multiple products."""

    if method == "bayesian_hierarchical":
        return _run_bayesian_elasticity_cached(
            transactions_df, params or {}, min_periods, min_price_variation
        )

    if method == "loglog_ols":
        return _estimate_all_elasticities_vectorized(
            transactions_df, products, min_periods, min_price_variation
        )

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

    return pd.DataFrame(results)


def _estimate_all_elasticities_vectorized(
    transactions_df: pd.DataFrame,
    products: List[str],
    min_periods: int,
    min_price_variation: float,
) -> pd.DataFrame:
    """Vectorized elasticity estimation using block-diagonal matrix solve.

    Fix #25: The design matrix has shape (total_obs, 2 * n_valid) where the
    first n_valid columns are per-SKU intercepts and the last n_valid columns
    are per-SKU slopes. Previously the shape was (total_obs, n_valid + 1)
    which caused only one intercept slot and n_valid - 1 slope overflows.
    """
    df = transactions_df[transactions_df["stockcode"].isin(products)].copy()
    df["date"] = pd.to_datetime(df["date"])

    weekly_all = (
        df.set_index("date")
        .groupby(["stockcode", pd.Grouper(freq="W")])
        .agg(avg_price=("price", "mean"), total_qty=("quantity", "sum"))
        .dropna()
        .reset_index()
    )

    sku_counts = weekly_all.groupby("stockcode").size()
    valid_skus_count = sku_counts[sku_counts >= min_periods].index

    price_cv = weekly_all.groupby("stockcode").apply(
        lambda x: x["avg_price"].std() / x["avg_price"].mean()
    )
    valid_skus_cv = price_cv[price_cv >= min_price_variation].index

    valid_skus = sorted(set(valid_skus_count) & set(valid_skus_cv))
    if not valid_skus:
        return pd.DataFrame()

    weekly_all = weekly_all[weekly_all["stockcode"].isin(valid_skus)].copy()
    weekly_all["log_price"] = np.log(weekly_all["avg_price"].clip(lower=1e-6))
    weekly_all["log_qty"] = np.log(weekly_all["total_qty"].clip(lower=1e-6))

    n_valid = len(valid_skus)
    sku_to_idx = {sku: i for i, sku in enumerate(valid_skus)}

    total_obs = len(weekly_all)
    # Fix #25: shape is (total_obs, 2 * n_valid): first n_valid cols = intercepts,
    # last n_valid cols = slopes.
    X = np.zeros((total_obs, 2 * n_valid))
    y = weekly_all["log_qty"].values

    row_offset = 0
    sku_stats: dict = {}

    for sku, group in weekly_all.groupby("stockcode"):
        idx = sku_to_idx[sku]
        n = len(group)
        X[row_offset : row_offset + n, idx] = 1.0                          # intercept
        X[row_offset : row_offset + n, n_valid + idx] = group["log_price"].values  # slope
        sku_stats[sku] = {
            "n_obs": n,
            "avg_price": group["avg_price"].mean(),
            "avg_weekly_qty": group["total_qty"].mean(),
            "price_cv": group["avg_price"].std() / group["avg_price"].mean(),
        }
        row_offset += n

    X = X[:row_offset]
    y = y[:row_offset]

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    intercepts = beta[:n_valid]
    slopes = beta[n_valid:]

    results = []
    for i, sku in enumerate(valid_skus):
        st_info = sku_stats[sku]
        slope = slopes[i]
        mask = weekly_all["stockcode"] == sku
        y_true = weekly_all.loc[mask, "log_qty"].values
        y_pred = intercepts[i] + slope * weekly_all.loc[mask, "log_price"].values
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        results.append(
            {
                "stockcode": sku,
                "elasticity": float(slope),
                "r_squared": float(r_squared),
                "p_value": 1.0,
                "std_err": 0.0,
                "n_obs": st_info["n_obs"],
                "avg_price": st_info["avg_price"],
                "avg_weekly_qty": st_info["avg_weekly_qty"],
                "price_cv": st_info["price_cv"],
            }
        )

    return pd.DataFrame(results)


def _render_elasticity_batch_results(elasticity_df: pd.DataFrame, product_lookup: dict):
    """Render batch elasticity results."""

    elasticity_df["product_name"] = elasticity_df["stockcode"].map(product_lookup)

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

    st.subheader("📈 Elasticity Distribution")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Elasticity", f"{elasticity_df['elasticity'].mean():.3f}")
    with col2:
        st.metric("Median Elasticity", f"{elasticity_df['elasticity'].median():.3f}")
    with col3:
        elastic_pct = (elasticity_df["elasticity"] < -1).mean() * 100
        st.metric("% Elastic (< -1)", f"{elastic_pct:.1f}%")

    if "elasticity_sd" in elasticity_df.columns:
        st.subheader("🔬 Bayesian Posterior Diagnostics")
        col_sd1, col_sd2, col_sd3 = st.columns(3)
        with col_sd1:
            st.metric("Mean Posterior SD", f"{elasticity_df['elasticity_sd'].mean():.3f}")
        with col_sd2:
            st.metric(
                "Mean HDI Width (94%)",
                f"{(elasticity_df['elasticity_hdi_upper'] - elasticity_df['elasticity_hdi_lower']).mean():.3f}",
            )
        with col_sd3:
            cross_zero = (
                (elasticity_df["elasticity_hdi_lower"] < 0)
                & (elasticity_df["elasticity_hdi_upper"] > 0)
            ).mean() * 100
            st.metric("% HDI Crosses Zero", f"{cross_zero:.1f}%")
        hdi_fig = px.scatter(
            elasticity_df,
            x="elasticity",
            y="elasticity_sd",
            color="interpretation",
            hover_data=["product_name"],
            title="Posterior Mean vs SD (tighter = more certain)",
        )
        st.plotly_chart(hdi_fig, use_container_width=True)
        _render_trace_diagnostics()

    fig = px.histogram(
        elasticity_df, x="elasticity", nbins=30, color="interpretation",
        title="Elasticity Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    elasticity_df["revenue_rank"] = elasticity_df["avg_price"] * elasticity_df["avg_weekly_qty"]
    fig2 = px.scatter(
        elasticity_df, x="elasticity", y="revenue_rank", color="interpretation",
        hover_data=["product_name"], title="Elasticity vs Weekly Revenue",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        elasticity_df[
            ["stockcode", "product_name", "elasticity", "r_squared", "p_value",
             "n_obs", "avg_price", "interpretation"]
        ].sort_values("elasticity"),
        use_container_width=True,
    )

    render_export_buttons(elasticity_df, product_lookup, prefix="elasticity")


def _render_trace_diagnostics():
    """Render MCMC trace and posterior density plots from st.session_state."""
    trace = st.session_state.get("_bayesian_trace")
    if trace is None or not hasattr(trace, "posterior"):
        return

    try:
        import arviz as az
        import matplotlib.pyplot as plt

        st.subheader("🔬 MCMC Trace Diagnostics")
        var_names = [v for v in ["mu_beta", "beta_cat", "beta_sku"] if v in trace.posterior]
        if not var_names:
            return

        with st.expander("Trace & Density Plots", expanded=False):
            fig = az.plot_trace(trace, var_names=var_names[:3], compact=True, backend="matplotlib")
            st.pyplot(fig[0][0].figure)
            plt.close("all")

        with st.expander("R-hat Convergence Diagnostics", expanded=False):
            summary = az.summary(trace, var_names=var_names[:3], round_to=3)
            st.dataframe(summary, use_container_width=True)
    except Exception:
        pass


# ============================================================================
# ELASTICITY BENCHMARK
# ============================================================================


def _render_elasticity_benchmark(params: dict, pipeline: Any = None):  # fix #16/#17: accept optional pipeline
    """Run and display synthetic-data validation across all elasticity methods."""

    st.header(" Elasticity Benchmark (Synthetic Data)")
    st.caption(
        "Generates data with **known ground-truth elasticities** and compares "
        "how well each method recovers them."
    )

    n_skus = params.get("benchmark_n_skus", 20)
    n_weeks = params.get("benchmark_n_weeks", 52)
    n_cats = params.get("benchmark_n_categories", 3)

    if st.button("▶️ Run Benchmark", type="primary", key="bench_run"):
        with st.spinner(f"Generating {n_skus} SKUs × {n_weeks} weeks with known elasticities..."):
            results = run_validation(n_skus=n_skus, n_weeks=n_weeks, n_categories=n_cats, n_samples=300)

        if results.empty:
            st.warning("Benchmark produced no results.")
            return

        st.success(f"Benchmark complete — {len(results)} methods compared")

        st.subheader("Recovery Metrics")
        display = results[["method", "rmse", "bias", "coverage_94", "n_products"]].copy()
        display["rmse"] = display["rmse"].round(4)
        display["bias"] = display["bias"].round(4)
        display["coverage_94"] = display["coverage_94"].round(3)
        st.dataframe(display, use_container_width=True)

        fig = px.bar(results, x="method", y="rmse", color="method",
                     title="RMSE by Method (lower = better)", text_auto=".4f")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(results, x="method", y="bias", color="method",
                      title="Bias by Method (closer to 0 = better)", text_auto=".4f")
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig2, use_container_width=True)

        cov_results = results[results["coverage_94"].notna()].copy()
        if not cov_results.empty:
            fig3 = px.bar(cov_results, x="method", y="coverage_94", color="method",
                          title="94% HDI Coverage (target = 0.94)", text_auto=".3f")
            fig3.add_hline(y=0.94, line_dash="dash", line_color="green")
            st.plotly_chart(fig3, use_container_width=True)

        best_rmse = results.loc[results["rmse"].idxmin()]
        best_bias = results.loc[results["bias"][results["bias"].notna()].abs().idxmin()]
        st.info(
            f" Lowest RMSE: **{best_rmse['method']}** ({best_rmse['rmse']:.4f})  ·  "
            f"Lowest |Bias|: **{best_bias['method']}** ({best_bias['bias']:.4f})"
        )


# ============================================================================
# KVI IDENTIFICATION
# ============================================================================


def _render_kvi_identification(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict, pipeline: dict = None
):
    """Render Key Value Item (KVI) identification and scoring."""

    st.header("🏷️ KVI (Key Value Item) Identification")
    st.caption(
        "Identifies items where price changes have outsized impact on customer perception and traffic. "
        "**Internal transaction-data KVI proxy only.**"
    )

    method = params.get("kvi_method", "xgb_importance")
    top_k = params.get("top_k_kvi", 20)
    margin_weighted = params.get("margin_weighted", False)

    cost_cols = [
        c for c in ["cost", "unit_cost", "margin", "margin_pct", "gross_margin"]
        if c in transactions_df.columns
    ]
    has_cost = len(cost_cols) > 0

    if margin_weighted and not has_cost:
        st.warning("Margin-weighted KVI requested but no cost/margin column found. Using revenue-based KVI.")
        margin_weighted = False

    with st.spinner("Computing product metrics and KVI features..."):
        product_metrics = compute_product_metrics(transactions_df)
        basket_uplift = compute_basket_value_uplift(transactions_df)
        basket_penetration = compute_basket_penetration(transactions_df)

    if product_metrics.empty:
        st.error("No product metrics computed.")
        return

    kvi_features = product_metrics.merge(
        basket_uplift[["stockcode", "basket_value_uplift_pct"]], on="stockcode", how="left"
    )
    kvi_features = kvi_features.merge(
        basket_penetration[["stockcode", "basket_penetration", "trip_incidence"]],
        on="stockcode", how="left",
    )

    kvi_features["price_cv"] = kvi_features.get(
        "price_cv", kvi_features["price_std"] / kvi_features["avg_price"].replace(0, np.nan)
    )

    if method == "xgb_importance":
        kvi_scores = _compute_kvi_xgb(kvi_features, margin_weighted, cost_cols[0] if cost_cols else None)
    else:
        kvi_scores = _compute_kvi_rfm_elasticity(kvi_features)

    top_kvi = kvi_scores.nlargest(top_k, "kvi_score")
    st.subheader(f"🏆 Top {top_k} KVI Products")

    display_cols = [
        "stockcode", "product_name", "kvi_score", "total_revenue",
        "basket_penetration", "basket_value_uplift_pct", "total_customers", "avg_price",
    ]
    if margin_weighted and has_cost:
        display_cols.append("margin_pct")

    st.dataframe(top_kvi[display_cols].reset_index(drop=True), use_container_width=True)

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

    feature_cols = [
        "basket_penetration", "trip_incidence", "total_revenue", "total_customers",
        "avg_price", "price_cv", "basket_value_uplift_pct", "total_transactions",
        "n_unique_products", "revenue_per_customer",
    ]
    feature_cols = [c for c in feature_cols if c in kvi_features.columns]
    X = kvi_features[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

    if margin_weighted and cost_col and cost_col in kvi_features.columns:
        kvi_features["margin"] = kvi_features["total_revenue"] * (
            1 - kvi_features[cost_col] / kvi_features["avg_price"].replace(0, np.nan)
        ).clip(0, 1)
        y = kvi_features["margin"].fillna(0)
    else:
        y = kvi_features["total_revenue"].fillna(0)

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
    model.fit(X, y)
    kvi_features["kvi_score"] = model.predict(X)
    return kvi_features


def _compute_kvi_rfm_elasticity(kvi_features: pd.DataFrame) -> pd.DataFrame:
    """Compute KVI scores using RFM + Elasticity heuristic."""
    scaler = StandardScaler()
    features = [
        "basket_penetration", "total_revenue", "total_customers",
        "revenue_per_customer", "basket_value_uplift_pct",
    ]
    available = [c for c in features if c in kvi_features.columns]
    if not available:
        kvi_features["kvi_score"] = 0
        return kvi_features

    X = kvi_features[available].fillna(0).replace([np.inf, -np.inf], 0)
    X_scaled = scaler.fit_transform(X)
    weights = np.array([0.3, 0.25, 0.15, 0.15, 0.15])[: len(available)]
    kvi_features["kvi_score"] = X_scaled @ weights
    return kvi_features


def _render_kvi_feature_importance(kvi_features: pd.DataFrame):
    st.info("Feature importance computed from XGBoost model used for KVI scoring.")


# ============================================================================
# KVI COMPOSITE (NielsenIQ 4-signal framework)
# ============================================================================


def _render_kvi_composite(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict, pipeline: dict = None
):
    """Render KVI Composite using 4-signal NielsenIQ framework."""
    st.header("🏷️ KVI Composite Score (NielsenIQ 4-Signal Framework)")
    st.caption(
        "Signals: Elasticity | Penetration | Frequency | Price Recall Proxy  |  "
        "**Internal transaction-data proxy only.**"
    )

    elasticity_df = params.get("elasticity_results")

    @st.cache_data
    def get_kvi_composite_cached(df, elast_df):
        return compute_kvi_composite_df(df, elast_df)

    with st.spinner("Computing KVI composite scores..."):
        kvi_df = get_kvi_composite_cached(transactions_df, elasticity_df)

    if kvi_df.empty:
        st.warning("No KVI data available")
        return

    st.success(f"Scored {len(kvi_df)} SKUs")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total SKUs Scored", len(kvi_df))
    with col2:
        st.metric("Tier 1 (Top) SKUs", f"{(kvi_df['kvi_tier'] == 'Tier 1 (Top)').sum()}")
    with col3:
        st.metric("True KVI Quadrant", f"{(kvi_df['kvi_quadrant'] == 'True KVI').sum()}")
    with col4:
        st.metric("Mean KVI Score", f"{kvi_df['kvi_score'].mean():.3f}")

    # Fix #26: persistent_tabs and render_analytics_export are now module-level functions.
    kvi_tab_labels = [
        " KVI Quadrant", " Price Ladder", " True Price View",
        " Basket Segments", " Tier Table", " Signal Breakdown",
    ]
    selected = persistent_tabs(kvi_tab_labels, "kvi_composite_tabs", default_tab=0)

    if selected == 0:
        _render_kvi_quadrant_chart(kvi_df, product_lookup)
    elif selected == 1:
        _render_price_ladder_chart(transactions_df, product_lookup, params)
    elif selected == 2:
        _render_true_price_view(transactions_df, product_lookup, params)
    elif selected == 3:
        _render_basket_segment_pricing_tab(transactions_df, product_lookup, params)
    elif selected == 4:
        _render_kvi_tier_table(kvi_df, product_lookup)
    elif selected == 5:
        _render_kvi_signal_breakdown(kvi_df, product_lookup)

    render_analytics_export(kvi_df, "KVI_Composite")


def _render_kvi_quadrant_chart(kvi_df: pd.DataFrame, product_lookup: dict):
    """Render KVI Quadrant: Elasticity vs Price Recall Proxy."""
    st.subheader("KVI Quadrant Chart")

    quad_colors = {
        "True KVI": "#2E7D32",
        "Promo Lever": "#FF8F00",
        "Price Anchor": "#1565C0",
        "Margin Recovery": "#C62828",
    }

    fig = px.scatter(
        kvi_df,
        x="abs_elasticity",
        y="price_recall_proxy",
        size="total_revenue",
        color="kvi_quadrant",
        color_discrete_map=quad_colors,
        hover_data=["product_name", "category", "kvi_score", "recommended_price_action", "avg_price"],
        title="KVI Quadrant: Elasticity vs Price Recall",
        size_max=50,
    )

    x_med = kvi_df["abs_elasticity"].median()
    y_med = kvi_df["price_recall_proxy"].median()
    fig.add_vline(x=x_med, line_dash="dash", line_color="gray", line_width=1)
    fig.add_hline(y=y_med, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    quad_summary = kvi_df.groupby("kvi_quadrant").agg(
        SKUs=("stockcode", "count"),
        Total_Revenue=("total_revenue", "sum"),
        Avg_KVI_Score=("kvi_score", "mean"),
    ).reset_index()
    st.dataframe(quad_summary, use_container_width=True)


def _render_price_ladder_chart(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render Price Ladder Chart with tier bands and violations."""
    st.subheader("Price Ladder Chart")

    n_tiers = params.get("n_tiers", 3)

    @st.cache_data
    def get_ladder_cached(df, tiers):
        return compute_price_ladder_df(df, n_tiers=tiers)

    with st.spinner("Computing price ladder..."):
        ladder_df = get_ladder_cached(transactions_df, n_tiers)

    if ladder_df.empty:
        st.warning("No price ladder data available")
        return

    ladder_df["product_name"] = ladder_df["stockcode"].map(product_lookup)
    st.dataframe(ladder_df, use_container_width=True)


def _render_kvi_tier_table(kvi_df: pd.DataFrame, product_lookup: dict):
    """Render KVI Tier table with action recommendations."""
    st.subheader("KVI Tiers & Recommended Actions")

    tier_order = ["Tier 1 (Top)", "Tier 2", "Tier 3", "Tier 4 (Background)"]
    kvi_df["kvi_tier"] = pd.Categorical(kvi_df["kvi_tier"], categories=tier_order, ordered=True)
    kvi_df = kvi_df.sort_values("kvi_tier")

    display_cols = [
        "product_name", "stockcode", "category", "kvi_tier",
        "kvi_score", "kvi_quadrant", "recommended_price_action",
        "total_revenue", "basket_penetration",
    ]
    display_cols = [c for c in display_cols if c in kvi_df.columns]
    st.dataframe(kvi_df[display_cols], use_container_width=True, hide_index=True)


def _render_kvi_signal_breakdown(kvi_df: pd.DataFrame, product_lookup: dict):
    """Render signal breakdown heatmap."""
    st.subheader("KVI Signal Breakdown")

    signal_cols = ["elasticity_signal", "penetration_signal", "frequency_signal", "recall_signal"]
    available = [c for c in signal_cols if c in kvi_df.columns]

    if not available:
        st.info("Signal breakdown not available")
        return

    fig = go.Figure(go.Heatmap(
        z=kvi_df[available].values.T,
        x=kvi_df["product_name"].tolist(),
        y=available,
        colorscale="RdYlGn",
    ))
    fig.update_layout(height=300, xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


def _render_basket_segment_pricing_tab(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict
):
    """Render basket-size segment analysis for pricing context."""
    st.subheader("Basket-Size Segment Analysis")

    from src.analytics.basket_metrics import compute_basket_size_segments, compute_basket_segment_profile

    with st.spinner("Computing basket segments..."):
        basket_segments = compute_basket_size_segments(transactions_df)
        segment_profile = compute_basket_segment_profile(transactions_df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Small Baskets (1-2 SKUs)",
                  f"{basket_segments[basket_segments['basket_segment']=='Small']['_basket'].count():,}")
    with col2:
        st.metric("Medium Baskets (3-7 SKUs)",
                  f"{basket_segments[basket_segments['basket_segment']=='Medium']['_basket'].count():,}")
    with col3:
        st.metric("Large Baskets (8+ SKUs)",
                  f"{basket_segments[basket_segments['basket_segment']=='Large']['_basket'].count():,}")

    st.subheader("Segment Profile")
    st.dataframe(segment_profile, use_container_width=True)

    # Fix #21: pre-compute revenue column; avoid invalid groupby lambda referencing outer df.
    st.subheader("Average Price by Basket Segment")
    from src.analytics.basket_metrics import get_basket_segment_for_product

    with st.spinner("Computing ASP by segment..."):
        product_segments = get_basket_segment_for_product(transactions_df)

    if not product_segments.empty:
        # Pre-compute revenue before aggregation (fix #21)
        tx = transactions_df.copy()
        tx["revenue"] = tx["price"] * tx["quantity"]
        asp_agg = tx.groupby("stockcode").agg(
            avg_price=("price", "mean"),
            total_revenue=("revenue", "sum"),
        ).reset_index()

        product_segments = product_segments.merge(asp_agg, on="stockcode", how="left")
        product_segments["product_name"] = product_segments["stockcode"].map(product_lookup)

        fig = px.box(
            product_segments, x="basket_segment", y="avg_price", color="basket_segment",
            hover_data=["product_name", "total_revenue"],
            title="ASP Distribution by Basket Segment",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No product-level segment data available.")


def _render_true_price_view(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict
):
    """Render NIQ-style True Price View."""
    st.subheader("💲 True Price View (NIQ-Style)")

    price_data = _compute_price_per_unit(transactions_df, product_lookup)
    if price_data.empty:
        st.error("Could not compute price per unit data.")
        return

    # Fix #23: pre-compute revenue before aggregation; avoid lambda over outer df.
    tx = transactions_df.copy()
    tx["revenue"] = tx["price"] * tx["quantity"]

    asp_data = (
        tx.groupby("stockcode")
        .agg(
            avg_price=("price", "mean"),
            median_price=("price", "median"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            total_revenue=("revenue", "sum"),
            total_units=("quantity", "sum"),
        )
        .reset_index()
    )
    asp_data["product_name"] = asp_data["stockcode"].map(product_lookup)

    n_tiers = params.get("n_tiers", 3)
    from sklearn.cluster import KMeans

    asp_values = asp_data["avg_price"].values.reshape(-1, 1)
    if len(asp_values) >= n_tiers:
        kmeans = KMeans(n_clusters=n_tiers, random_state=42, n_init=10)
        asp_data["price_tier"] = kmeans.fit_predict(asp_values)
        tier_order = asp_data.groupby("price_tier")["avg_price"].mean().sort_values().index
        tier_map = {old: new for new, old in enumerate(tier_order)}
        asp_data["price_tier"] = asp_data["price_tier"].map(tier_map)
        tier_centers_sorted = np.sort(kmeans.cluster_centers_.flatten())
    else:
        asp_data["price_tier"] = 0
        tier_centers_sorted = [asp_data["avg_price"].median()]

    tier_colors = {0: "#2E7D32", 1: "#1565C0", 2: "#FF8F00", 3: "#C62828", 4: "#673AB7"}

    fig = px.scatter(
        asp_data, x="avg_price", y="total_revenue",
        color="price_tier", color_discrete_map=tier_colors,
        hover_data=["product_name", "total_units"],
        title="True Price View: ASP vs Revenue by Price Tier",
    )
    for center in tier_centers_sorted:
        fig.add_vline(x=center, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PRICE LADDER (standalone)
# ============================================================================


def _render_price_ladder(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict, pipeline: dict = None
):
    _render_price_ladder_chart(transactions_df, product_lookup, params)


# ============================================================================
# PRICE CURVE DIAGNOSTICS
# ============================================================================


def _render_price_curve_diagnostics(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict, pipeline: dict = None
):
    """Render price curve diagnostics — pack-size monotonicity, tier clustering."""

    st.header("📊 Price Curve Diagnostics")

    multivariate_mode = st.sidebar.checkbox(
        "Multivariate (Price + Elasticity + Margin)",
        value=params.get("price_curve_multivariate", False),
        key="price_curve_multivariate",
    )

    method = params.get("price_curve_method", "kmeans")
    n_tiers = params.get("n_tiers", 3)

    elasticity_df = None
    cost_col = None
    if multivariate_mode:
        cost_cols = [
            c for c in ["cost", "unit_cost", "margin", "margin_pct", "gross_margin"]
            if c in transactions_df.columns
        ]
        if cost_cols:
            cost_col = st.sidebar.selectbox("Cost Column", cost_cols, key="price_curve_cost_col")
        if "elasticity_results" in params:
            elasticity_df = params["elasticity_results"]

    price_data = _compute_price_per_unit(transactions_df, product_lookup)
    if price_data.empty:
        st.error("Could not compute price per unit data.")
        return

    if multivariate_mode:
        result_df = diagnose_price_curves_multivariate(
            transactions_df, n_tiers=n_tiers, method=method,
            elasticity_df=elasticity_df, cost_col=cost_col,
        )
    else:
        result_df = diagnose_price_curves_1d(transactions_df, n_tiers=n_tiers, method=method)

    if result_df.empty:
        st.warning("Insufficient data for price curve diagnostics.")
        return

    # Fix #19: merge basket_penetration into result_df so the scatter doesn't crash.
    if "basket_penetration" not in result_df.columns:
        bp = compute_basket_penetration(transactions_df)[["stockcode", "basket_penetration"]]
        result_df = result_df.merge(bp, on="stockcode", how="left")
        result_df["basket_penetration"] = result_df["basket_penetration"].fillna(0)

    if "product_name" not in result_df.columns:
        result_df["product_name"] = result_df["stockcode"].map(product_lookup)

    categories = result_df["category"].unique() if "category" in result_df.columns else ["All"]

    for cat in categories:
        cat_data = result_df[result_df["category"] == cat] if cat != "All" else result_df
        if len(cat_data) < 3:
            continue

        st.subheader(f"📦 Category: {cat}")

        fig = px.scatter(
            cat_data, x="basket_penetration", y="price_per_unit",
            hover_data=["product_name", "pack_size_numeric", "avg_price"],
            color="tier_label" if "tier_label" in cat_data.columns else None,
            title=f"{cat}: Price per Unit vs Basket Penetration",
        )
        st.plotly_chart(fig, use_container_width=True)

        if "pack_size_numeric" in cat_data.columns:
            fig2 = px.scatter(
                cat_data, x="pack_size_numeric", y="price_per_unit",
                # Fix #20: replaced invalid `pack_size` with `pack_size_numeric` in hover_data
                hover_data=["product_name", "pack_size_numeric"],
                title=f"{cat}: Pack Size vs Price per Unit (Monotonicity Check)",
            )
            st.plotly_chart(fig2, use_container_width=True)

            violations = _detect_price_curve_violations(cat_data)
            if not violations.empty:
                st.warning("⚠️ Price Curve Violations Detected")
                st.dataframe(violations, use_container_width=True)
            else:
                st.success("✅ No monotonicity violations detected")

        if len(cat_data) >= n_tiers:
            _render_tier_analysis(cat_data, cat)


def _compute_price_per_unit(transactions_df: pd.DataFrame, product_lookup: dict) -> pd.DataFrame:
    """Compute price per unit for each product."""
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
        )
        .reset_index()
    )

    def parse_pack_size(size_str):
        if pd.isna(size_str):
            return 1.0
        import re
        size_str = str(size_str).upper()
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

    product_info["pack_size_numeric"] = product_info["size"].apply(parse_pack_size)
    product_info["price_per_unit"] = (
        product_info["median_price"] / product_info["pack_size_numeric"].replace(0, np.nan)
    )

    bp = compute_basket_penetration(transactions_df)[["stockcode", "basket_penetration", "trip_incidence"]]
    product_info = product_info.merge(bp, on="stockcode", how="left")
    product_info["product_name"] = product_info["stockcode"].map(product_lookup)

    return product_info


def _detect_price_curve_violations(cat_data: pd.DataFrame) -> pd.DataFrame:
    """Detect price curve violations: larger pack cheaper per unit."""
    if "pack_size_numeric" not in cat_data.columns:
        return pd.DataFrame()

    sorted_data = cat_data.sort_values("pack_size_numeric")
    violations = []
    for i in range(len(sorted_data) - 1):
        row1 = sorted_data.iloc[i]
        row2 = sorted_data.iloc[i + 1]
        if row1["price_per_unit"] > row2["price_per_unit"] * 1.05:
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
    model = KMeans(n_clusters=n_tiers, random_state=42, n_init=10) if method == "kmeans" \
        else GaussianMixture(n_components=n_tiers, random_state=42)

    cat_data = cat_data.copy()
    cat_data["tier"] = model.fit_predict(features)
    tier_order = cat_data.groupby("tier")["price_per_unit"].mean().sort_values().index
    tier_map = {old: new for new, old in enumerate(tier_order)}
    cat_data["tier"] = cat_data["tier"].map(tier_map)
    tier_labels = {0: "Value", 1: "Mainstream", 2: "Premium", 3: "Ultra", 4: "Luxury"}
    cat_data["tier_label"] = cat_data["tier"].map(tier_labels).fillna("Tier " + cat_data["tier"].astype(str))
    return cat_data


def _render_tier_analysis(tier_results: pd.DataFrame, category: str):
    """Render tier clustering results."""
    st.markdown("**Price Tier Assignment**")

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

    # Fix #20: use pack_size_numeric (always present) instead of pack_size (doesn't exist).
    hover = ["product_name", "pack_size_numeric"] if "pack_size_numeric" in tier_results.columns else ["product_name"]
    fig = px.scatter(
        tier_results, x="basket_penetration", y="price_per_unit",
        color="tier_label", hover_data=hover,
        title=f"{category}: Price Tiers",
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PROMO UPLIFT MODELING
# ============================================================================


def _render_promo_uplift_modeling(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict, pipeline: dict = None
):
    """Render promo uplift modeling using T-learner / S-learner.

    Causal Assumptions (SUTVA & Confounders) — Issue #6
    ---------------------------------------------------
    This module estimates Conditional Average Treatment Effect (CATE) from
    observational transaction data. The following causal assumptions apply:

    1. SUTVA (Stable Unit Treatment Value Assumption): No interference between
       customers — one customer's exposure to a promotion does not affect
       another's purchasing behaviour. Violated if promotions drive store traffic
       (halo effect) or cannibalise category-level volume.

    2. Unconfoundedness (No hidden confounders): Treatment assignment (promo/no-
       promo) must be conditionally independent of potential outcomes given
       observed covariates. Key confounders NOT controlled for include:
       - Seasonal demand cycles coinciding with promotions
       - Competitor price changes during promo windows
       - Stockout effects masking true sales lift
       - Customer self-selection (price-sensitive shoppers over-index promo buys)

    3. Overlap (Positivity): Every customer must have non-zero probability of
       receiving either treatment. Blocked automatically when overlap < 60%.

    Results should be treated as indicative correlational estimates, not
    validated causal incrementality. Use A/B test data for decision-grade uplift.
    """

    st.header("🎯 Promo Uplift Modeling")
    st.caption(
        "Causal uplift estimation from observational data. "
        "**Experimental — not validated incrementality.** Requires strong treatment/control overlap. "
        "Results blocked if propensity overlap < 60% or validation score < 0.7."
    )

    render_data_quality_expander(transactions_df, "promo_uplift", params, expanded=False)

    drop_threshold = params.get("promo_drop_threshold", 15) / 100
    baseline_window = params.get("promo_baseline_window", 28)
    method = params.get("uplift_method", "t_learner")
    n_estimators = params.get("base_n_estimators", 200)
    max_depth = params.get("base_max_depth", 5)
    propensity_strat = params.get("propensity_stratification", True)

    # Fix #18/#15: _derive_promo_flag is now a module-level function (no NameError).
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
    st.subheader("Detected Promotions Summary")
    st.dataframe(promo_summary, use_container_width=True)

    # Fix #22: build_uplift_dataset returns a DataFrame; pass it directly to trainers.
    with st.spinner("Building uplift dataset..."):
        uplift_data = build_uplift_dataset(
            transactions_df,
            promo_df,
            baseline_window=baseline_window,
        )

    if uplift_data.empty:
        st.warning("Insufficient data for uplift modeling.")
        return

    treatment_n = uplift_data["treatment"].sum()
    control_n = (~uplift_data["treatment"].astype(bool)).sum()
    overlap_pct = (
        min(treatment_n, control_n) / max(treatment_n, control_n) * 100
        if max(treatment_n, control_n) > 0 else 0
    )

    st.subheader("Treatment/Control Balance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Treatment (Promo)", f"{int(treatment_n):,}")
    col2.metric("Control (Non-Promo)", f"{int(control_n):,}")
    col3.metric("Overlap", f"{overlap_pct:.1f}%")

    if overlap_pct < 60:
        st.error(
            f"🚫 **Blocked**: Treatment/control overlap ({overlap_pct:.1f}%) < 60%."
        )
        return

    # Fix #22: pass uplift_data DataFrame directly; trainers expect a single DataFrame.
    with st.spinner(f"Training {method} uplift model..."):
        if method == "t_learner":
            model, metrics = train_t_learner_uplift(
                uplift_data,
                n_estimators=n_estimators,
                max_depth=max_depth,
                propensity_stratification=propensity_strat,
            )
        else:
            model, metrics = train_s_learner_uplift(
                uplift_data,
                n_estimators=n_estimators,
                max_depth=max_depth,
                propensity_stratification=propensity_strat,
            )

    if model is None:
        st.error(f"Model training failed: {metrics.get('error', 'Unknown error')}")
        return

    validation_score = metrics.get("validation_score", 0)
    ate = metrics.get("ate", None)

    if validation_score < 0.7:
        st.error(
            f"🚫 **Blocked**: Validation score ({validation_score:.3f}) < 0.7."
        )
        return

    render_uplift_context(
        treatment_n=int(treatment_n),
        control_n=int(control_n),
        overlap_pct=overlap_pct,
        validation_score=validation_score,
        method=method,
        ate=ate,
    )

    predictions = metrics.get("predictions")
    if predictions is not None:
        fig = px.histogram(
            predictions, nbins=50,
            title="Predicted Uplift Distribution",
            labels={"value": "Uplift"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Uplift")
        st.plotly_chart(fig, use_container_width=True)

    if "qini_curve" in metrics:
        st.subheader("Qini Curve")
        qini = metrics["qini_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=qini["population_pct"], y=qini["qini"], mode="lines", name="Model",
        ))
        fig.add_trace(go.Scatter(
            x=qini["population_pct"], y=qini["random"], mode="lines", name="Random",
            line=dict(dash="dash"),
        ))
        fig.update_layout(xaxis_title="Population %", yaxis_title="Qini Coefficient", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔬 Expert Diagnostics (SHAP, Feature Importance)", expanded=False):
        _render_uplift_expert_diagnostics(metrics, uplift_data)


def _render_uplift_expert_diagnostics(metrics: dict, uplift_data: pd.DataFrame):
    """Render SHAP / feature importance diagnostics for uplift model."""
    st.caption("Expert diagnostics: feature importance and SHAP values from the uplift model.")
    feature_importance = metrics.get("feature_importance")
    if feature_importance is not None and not feature_importance.empty:
        fig = px.bar(
            feature_importance.head(20), x="importance", y="feature",
            orientation="h", title="Feature Importance (Top 20)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for this model.")


# ============================================================================
# HELPER: Export Buttons
# ============================================================================


def render_export_buttons(df: pd.DataFrame, product_lookup: dict, prefix: str = "export"):
    """Render export buttons for DataFrame."""
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV", csv, f"{prefix}.csv", "text/csv", key=f"{prefix}_csv",
        )
    with col2:
        json_str = df.to_json(orient="records", indent=2)
        st.download_button(
            "📥 Download JSON", json_str, f"{prefix}.json", "application/json",
            key=f"{prefix}_json",
        )
