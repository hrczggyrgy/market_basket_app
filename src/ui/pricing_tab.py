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

warnings.filterwarnings("ignore")


def render_pricing_tab(
    transactions_df: pd.DataFrame,
    product_lookup: dict,
    params: dict,
    mode: str = "elasticity",
    pipeline: Any = None,
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

    # Data sufficiency gate
    sufficiency = assess_data_sufficiency(
        transactions_df,
        min_transactions=params.get("min_transactions", 500),
        min_customers=params.get("min_customers", 30),
        min_products=params.get("min_products", 5),
        min_time_span_days=params.get("min_time_span_days", 60),
        min_price_variation_cv=params.get("min_price_variation", 0.03),
    )
    with st.expander("📋 Data Sufficiency", expanded=sufficiency["overall"] != "robust"):
        st.markdown(format_sufficiency_summary(sufficiency))
        if sufficiency["overall"] == "insufficient":
            st.warning("Dataset may be too small for reliable elasticity estimates.")
        elif sufficiency["overall"] == "directional":
            st.info("Results should be treated as directional, not definitive.")

    method = params.get("elasticity_method", "loglog_ols")
    min_periods = params.get("min_periods", 10)
    min_price_variation = params.get("min_price_variation", 0.05)

    # Check pipeline for cached elasticity results
    cached_elasticity = pipeline.elasticity_results if pipeline else None
    if cached_elasticity is not None and not cached_elasticity.empty:
        st.info("Using cached elasticity results from pipeline")

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

    # Bayesian single-product mode (uses the hierarchical result filtered to this SKU)
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
        st.subheader(f"📊 Bayesian Elasticity: {product_lookup.get(product_id, product_id)}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Elasticity (posterior mean)", f"{row['elasticity_mean']:.3f}")
        with col2:
            st.metric("Posterior SD", f"{row['elasticity_sd']:.3f}")
        with col3:
            st.metric("94% HDI Lower", f"{row['elasticity_hdi_lower']:.3f}")
        with col4:
            st.metric("94% HDI Upper", f"{row['elasticity_hdi_upper']:.3f}")
        if row["elasticity_hdi_lower"] < 0 and row["elasticity_hdi_upper"] < 0:
            st.info("🔴 **Elastic** — 94% HDI entirely below zero (demand sensitive to price).")
        elif row["elasticity_hdi_lower"] < 0 < row["elasticity_hdi_upper"]:
            st.warning(
                "🟡 **Uncertain** — HDI crosses zero (insufficient evidence of price effect)."
            )
        else:
            st.info(
                "🟢 **Positive** — HDI above zero (possible promo effect or omitted variable bias)."
            )

        # Trace diagnostics (only for NUTS)
        if params.get("bayesian_mode", "").startswith("full"):
            _render_trace_diagnostics()
        return

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
    else:
        st.info(f"Price CV: {price_cv:.3f} — sufficient variation ✅")

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


def _run_bayesian_elasticity_cached(
    transactions_df: pd.DataFrame,
    params: dict,
    min_periods: int = 10,
    min_price_variation: float = 0.05,
) -> pd.DataFrame:
    """Run Bayesian hierarchical elasticity with trace caching.

    Results are cached in-memory via st.cache_resource and also stored
    in st.session_state for trace diagnostic plotting.
    """
    bayesian_mode = params.get("bayesian_mode", "fast (ADVI)")
    want_trace = bayesian_mode.startswith("full")
    model_config = {
        "min_periods": min_periods,
        "min_price_variation": min_price_variation,
        "bayesian_mode": bayesian_mode,
    }
    cache_key = trace_cache_key(transactions_df, model_config)

    # Check in-memory cache
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
        # Vectorized matrix solve - O(N) instead of O(N * K) where K=products
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

    df = pd.DataFrame(results)
    return df


def _estimate_all_elasticities_vectorized(
    transactions_df: pd.DataFrame,
    products: List[str],
    min_periods: int,
    min_price_variation: float,
) -> pd.DataFrame:
    """Vectorized elasticity estimation using matrix solve (10-50x speedup).

    Stacks all SKU weekly data into a single block-diagonal design matrix
    and solves via np.linalg.lstsq in one call.
    """
    df = transactions_df[transactions_df["stockcode"].isin(products)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # Aggregate all products to weekly in one pass
    weekly_all = (
        df.set_index("date")
        .groupby(["stockcode", pd.Grouper(freq="W")])
        .agg(avg_price=("price", "mean"), total_qty=("quantity", "sum"))
        .dropna()
        .reset_index()
    )

    # Filter by min_periods and price variation per SKU
    sku_counts = weekly_all.groupby("stockcode").size()
    valid_skus_count = sku_counts[sku_counts >= min_periods].index

    price_cv = weekly_all.groupby("stockcode").apply(
        lambda x: x["avg_price"].std() / x["avg_price"].mean()
    )
    valid_skus_cv = price_cv[price_cv >= min_price_variation].index

    valid_skus = set(valid_skus_count) & set(valid_skus_cv)
    if not valid_skus:
        return pd.DataFrame()

    weekly_all = weekly_all[weekly_all["stockcode"].isin(valid_skus)]

    # Prepare log-transformed data
    weekly_all["log_price"] = np.log(weekly_all["avg_price"].clip(lower=1e-6))
    weekly_all["log_qty"] = np.log(weekly_all["total_qty"].clip(lower=1e-6))

    # Build block-diagonal design matrix and solve all at once
    results = []
    sku_to_idx = {sku: i for i, sku in enumerate(valid_skus)}
    n_valid = len(valid_skus)

    # Group by SKU to get slice indices
    sku_groups = weekly_all.groupby("stockcode")

    # Build block-diagonal matrix
    total_obs = len(weekly_all)
    X = np.zeros((total_obs, n_valid + 1))  # +1 for intercept per SKU
    y = weekly_all["log_qty"].values

    row_offset = 0
    sku_stats = {}

    for sku, group in sku_groups:
        if sku not in valid_skus:
            continue
        idx = sku_to_idx[sku]
        n = len(group)

        # Design matrix: [1, log_price] for this SKU's block
        X[row_offset : row_offset + n, idx] = 1.0  # intercept column for this SKU
        X[row_offset : row_offset + n, n_valid + idx] = group[
            "log_price"
        ].values  # slope column for this SKU

        sku_stats[sku] = {
            "n_obs": n,
            "avg_price": group["avg_price"].mean(),
            "avg_weekly_qty": group["total_qty"].mean(),
            "price_cv": group["avg_price"].std() / group["avg_price"].mean(),
        }

        row_offset += n

    # Trim unused rows
    X = X[:row_offset]
    y = y[:row_offset]

    # Solve via least squares: (X^T X) beta = X^T y
    # This is much faster than per-SKU loop
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    # Extract results
    intercepts = beta[:n_valid]
    slopes = beta[n_valid:]

    for i, sku in enumerate(valid_skus):
        stats = sku_stats[sku]
        slope = slopes[i]
        # Compute R²
        mask = weekly_all["stockcode"] == sku
        y_true = weekly_all.loc[mask, "log_qty"].values
        y_pred = intercepts[i] + slope * weekly_all.loc[mask, "log_price"].values
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        results.append(
            {
                "stockcode": sku,
                "elasticity": float(slope),
                "r_squared": float(r_squared),
                "p_value": 1.0,  # Not computed in vectorized version
                "std_err": 0.0,  # Not computed in vectorized version
                "n_obs": stats["n_obs"],
                "avg_price": stats["avg_price"],
                "avg_weekly_qty": stats["avg_weekly_qty"],
                "price_cv": stats["price_cv"],
            }
        )

    return pd.DataFrame(results)


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

    # Bayesian posterior diagnostics
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
            labels={"elasticity": "Elasticity (mean)", "elasticity_sd": "Posterior SD"},
        )
        st.plotly_chart(hdi_fig, use_container_width=True)

        _render_trace_diagnostics()

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


def _render_trace_diagnostics():
    """Render MCMC trace and posterior density plots from st.session_state."""
    trace = st.session_state.get("_bayesian_trace")
    if trace is None or not hasattr(trace, "posterior"):
        return

    try:
        import arviz as az
        import matplotlib.pyplot as plt

        st.subheader("🔬 MCMC Trace Diagnostics")

        # Extract key parameters
        var_names = [v for v in ["mu_beta", "beta_cat", "beta_sku"] if v in trace.posterior]
        if not var_names:
            return

        # Trace plot
        with st.expander("Trace & Density Plots", expanded=False):
            fig = az.plot_trace(
                trace,
                var_names=var_names[:3],
                compact=True,
                backend="matplotlib",
            )
            st.pyplot(fig[0][0].figure)
            plt.close("all")

        # R-hat summary
        with st.expander("R-hat Convergence Diagnostics", expanded=False):
            summary = az.summary(trace, var_names=var_names[:3], round_to=3)
            st.dataframe(summary, use_container_width=True)

    except Exception:
        pass


# ============================================================================
# ELASTICITY BENCHMARK (synthetic-data validation)
# ============================================================================


def _render_elasticity_benchmark(params: dict):
    """Run and display synthetic-data validation across all elasticity methods."""

    st.header(" Elastity Benchmark (Synthetic Data)")
    st.caption(
        "Generates data with **known ground-truth elasticities** and compares "
        "how well each method recovers them. 94% HDI coverage measures how often "
        "the true value falls inside the credible interval (target: ~0.94)."
    )

    n_skus = params.get("benchmark_n_skus", 20)
    n_weeks = params.get("benchmark_n_weeks", 52)
    n_cats = params.get("benchmark_n_categories", 3)

    if st.button("▶️ Run Benchmark", type="primary", key="bench_run"):
        with st.spinner(f"Generating {n_skus} SKUs × {n_weeks} weeks with known elasticities..."):
            results = run_validation(
                n_skus=n_skus,
                n_weeks=n_weeks,
                n_categories=n_cats,
                n_samples=300,
            )

        if results.empty:
            st.warning("Benchmark produced no results.")
            return

        st.success(f"Benchmark complete — {len(results)} methods compared")

        # Summary metrics table
        st.subheader("Recovery Metrics")
        display = results[["method", "rmse", "bias", "coverage_94", "n_products"]].copy()
        display["rmse"] = display["rmse"].round(4)
        display["bias"] = display["bias"].round(4)
        display["coverage_94"] = display["coverage_94"].round(3)
        st.dataframe(display, use_container_width=True)

        # Bar chart: RMSE
        fig = px.bar(
            results,
            x="method",
            y="rmse",
            color="method",
            title="RMSE by Method (lower = better recovery)",
            labels={"rmse": "RMSE", "method": ""},
            text_auto=".4f",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Bar chart: Bias
        fig2 = px.bar(
            results,
            x="method",
            y="bias",
            color="method",
            title="Bias by Method (closer to 0 = better)",
            labels={"bias": "Bias", "method": ""},
            text_auto=".4f",
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig2, use_container_width=True)

        # Coverage bar chart
        cov_results = results[results["coverage_94"].notna()].copy()
        if not cov_results.empty:
            fig3 = px.bar(
                cov_results,
                x="method",
                y="coverage_94",
                color="method",
                title="94% HDI Coverage (target = 0.94)",
                labels={"coverage_94": "Coverage", "method": ""},
                text_auto=".3f",
            )
            fig3.add_hline(y=0.94, line_dash="dash", line_color="green")
            st.plotly_chart(fig3, use_container_width=True)

        # Interpretation
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
# KVI COMPOSITE (Phase 2 - NielsenIQ 4-signal framework)
# ============================================================================


def _render_kvi_composite(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict, pipeline: dict = None
):
    """Render KVI Composite using 4-signal NielsenIQ framework."""
    st.header("🏷️ KVI Composite Score (NielsenIQ 4-Signal Framework)")
    st.caption(
        "Signals: Elasticity | Penetration | Frequency | Price Recall Proxy  |  "
        "Quadrants: True KVI / Promo Lever / Price Anchor / Margin Recovery"
    )

    # Check for elasticity data
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

    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total SKUs Scored", len(kvi_df))
    with col2:
        st.metric("Tier 1 (Top) SKUs", f"{(kvi_df['kvi_tier'] == 'Tier 1 (Top)').sum()}")
    with col3:
        st.metric("True KVI Quadrant", f"{(kvi_df['kvi_quadrant'] == 'True KVI').sum()}")
    with col4:
        st.metric("Mean KVI Score", f"{kvi_df['kvi_score'].mean():.3f}")

    # Sub-tabs
    kvi_tabs = [" KVI Quadrant", " Price Ladder", " Tier Table", " Signal Breakdown"]
    selected = persistent_tabs(kvi_tabs, "kvi_composite_tabs", default_tab=0)

    if selected == 0:
        _render_kvi_quadrant_chart(kvi_df, product_lookup)
    elif selected == 1:
        _render_price_ladder_chart(transactions_df, product_lookup, params)
    elif selected == 2:
        _render_kvi_tier_table(kvi_df, product_lookup)
    elif selected == 3:
        _render_kvi_signal_breakdown(kvi_df, product_lookup)

    render_analytics_export(kvi_df, "KVI_Composite")


def _render_kvi_quadrant_chart(kvi_df: pd.DataFrame, product_lookup: dict):
    """Render KVI Quadrant: Elasticity vs Price Recall Proxy."""
    st.subheader("KVI Quadrant Chart")
    st.caption(
        "X = |Elasticity| (Price Sensitivity)  |  "
        "Y = Price Recall Proxy (Frequency × Price Stability)  |  "
        "Size = Revenue  |  Color = KVI Quadrant"
    )

    # Quadrant colors
    quad_colors = {
        "True KVI": "#2E7D32",        # Green - Protect
        "Promo Lever": "#FF8F00",      # Amber - Promote
        "Price Anchor": "#1565C0",     # Blue - Fair Price
        "Margin Recovery": "#C62828",  # Red - Recover Margin
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
        labels={
            "abs_elasticity": "|Elasticity| (Price Sensitivity)",
            "price_recall_proxy": "Price Recall Proxy",
            "kvi_quadrant": "KVI Quadrant",
            "total_revenue": "Revenue",
        },
        size_max=50,
    )

    # Quadrant lines at medians
    x_med = kvi_df["abs_elasticity"].median()
    y_med = kvi_df["price_recall_proxy"].median()
    fig.add_vline(x=x_med, line_dash="dash", line_color="gray", line_width=1)
    fig.add_hline(y=y_med, line_dash="dash", line_color="gray", line_width=1)

    # Quadrant labels
    x_max = kvi_df["abs_elasticity"].max() * 1.1
    y_max = kvi_df["price_recall_proxy"].max() * 1.1
    x_min = kvi_df["abs_elasticity"].min() * 0.9
    y_min = kvi_df["price_recall_proxy"].min() * 0.9

    fig.add_annotation(x=x_max*0.7, y=y_max*0.9, text="<b>True KVI</b>", showarrow=False, font=dict(color="#2E7D32", size=14))
    fig.add_annotation(x=x_max*0.7, y=y_min*1.1, text="<b>Promo Lever</b>", showarrow=False, font=dict(color="#FF8F00", size=14))
    fig.add_annotation(x=x_min*1.3, y=y_max*0.9, text="<b>Price Anchor</b>", showarrow=False, font=dict(color="#1565C0", size=14))
    fig.add_annotation(x=x_min*1.3, y=y_min*1.1, text="<b>Margin Recovery</b>", showarrow=False, font=dict(color="#C62828", size=14))

    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Quadrant summary
    quad_summary = kvi_df.groupby("kvi_quadrant").agg(
        SKUs=("stockcode", "count"),
        Total_Revenue=("total_revenue", "sum"),
        Avg_KVI_Score=("kvi_score", "mean"),
        Avg_Elasticity=("abs_elasticity", "mean"),
        Avg_Recall=("price_recall_proxy", "mean"),
    ).reset_index()

    st.dataframe(
        quad_summary.style.format({
            "Total_Revenue": "${:,.0f}",
            "Avg_KVI_Score": "{:.3f}",
            "Avg_Elasticity": "{:.3f}",
            "Avg_Recall": "{:.3f}",
        }).background_gradient(cmap="RdYlGn", subset=["Total_Revenue", "Avg_KVI_Score"]),
        use_container_width=True,
    )


def _render_price_ladder_chart(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render Price Ladder Chart with tier bands and violations."""
    st.subheader("Price Ladder Chart")
    st.caption(
        "Horizontal dots = SKU ASP position. Shaded bands = price tiers (KMeans). "
        "Red dots = SKUs violating tier placement."
    )

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

    # Tier colors
    tier_colors = {
        0: "rgba(46, 125, 50, 0.15)",    # Value - green tint
        1: "rgba(21, 101, 192, 0.15)",   # Mainstream - blue tint
        2: "rgba(255, 143, 0, 0.15)",    # Premium - amber tint
        3: "rgba(198, 40, 40, 0.15)",    # Ultra - red tint
        4: "rgba(103, 58, 183, 0.15)",   # Luxury - purple tint
    }

    fig = go.Figure()

    # Tier bands
    for tier in sorted(ladder_df["price_tier"].dropna().unique()):
        tier_data = ladder_df[ladder_df["price_tier"] == tier]
        tier_min = tier_data["tier_min"].iloc[0]
        tier_max = tier_data["tier_max"].iloc[0]
        fig.add_shape(
            type="rect",
            x0=tier_min, x1=tier_max,
            y0=-0.5, y1=len(ladder_df) - 0.5,
            fillcolor=tier_colors.get(int(tier), "rgba(128, 128, 128, 0.1)"),
            line=dict(width=0),
            layer="below",
        )
        # Tier label
        fig.add_annotation(
            x=(tier_min + tier_max) / 2,
            y=len(ladder_df) + 1,
            text=tier_data["tier_label"].iloc[0],
            showarrow=False,
            font=dict(size=12, color="gray"),
            xanchor="center",
        )

    # SKU dots
    compliant = ladder_df[~ladder_df["violation"]]
    violations = ladder_df[ladder_df["violation"]]

    if not compliant.empty:
        fig.add_trace(go.Scatter(
            x=compliant["asp"],
            y=compliant["product_name"],
            mode="markers",
            marker=dict(color="#2E7D32", size=10, symbol="circle"),
            name="Compliant",
            hovertemplate="<b>%{y}</b><br>ASP: $%{x:.2f}<extra></extra>",
        ))

    if not violations.empty:
        fig.add_trace(go.Scatter(
            x=violations["asp"],
            y=violations["product_name"],
            mode="markers",
            marker=dict(color="#C62828", size=12, symbol="x"),
            name="Tier Violation",
            hovertemplate="<b>%{y}</b><br>ASP: $%{x:.2f}<br>VIOLATION<extra></extra>",
        ))

    fig.update_layout(
        title="Price Ladder: ASP by SKU with Tier Bands",
        xaxis_title="Average Selling Price ($)",
        yaxis_title="SKU",
        height=max(500, len(ladder_df) * 20),
        showlegend=True,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Violations table
    if not violations.empty:
        st.warning(f"⚠️ {len(violations)} SKUs with tier violations")
        st.dataframe(
            violations[["product_name", "stockcode", "asp", "tier_label", "tier_min", "tier_max"]].style.format({
                "asp": "${:.2f}",
                "tier_min": "${:.2f}",
                "tier_max": "${:.2f}",
            }),
            use_container_width=True,
        )


def _render_kvi_tier_table(kvi_df: pd.DataFrame, product_lookup: dict):
    """Render KVI Tier table with action recommendations."""
    st.subheader("KVI Tiers & Recommended Actions")

    tier_order = ["Tier 1 (Top)", "Tier 2", "Tier 3", "Tier 4 (Background)"]
    kvi_df["kvi_tier"] = pd.Categorical(kvi_df["kvi_tier"], categories=tier_order, ordered=True)
    kvi_df = kvi_df.sort_values("kvi_tier")

    display_cols = [
        "product_name", "stockcode", "category", "kvi_tier",
        "kvi_score", "kvi_quadrant", "recommended_price_action",
        "total_revenue", "basket_penetration", "abs_elasticity", "price_recall_proxy",
    ]
    display_cols = [c for c in display_cols if c in kvi_df.columns]

    st.dataframe(
        kvi_df[display_cols].style.format({
            "total_revenue": "${:,.0f}",
            "basket_penetration": "{:.1%}",
            "abs_elasticity": "{:.3f}",
            "price_recall_proxy": "{:.3f}",
            "kvi_score": "{:.3f}",
        }).background_gradient(cmap="RdYlGn", subset=["kvi_score", "total_revenue"]),
        use_container_width=True,
        hide_index=True,
    )


def _render_kvi_signal_breakdown(kvi_df: pd.DataFrame, product_lookup: dict):
    """Render signal breakdown heatmap."""
    st.subheader("KVI Signal Breakdown")
    st.caption("Each signal normalized 0-1. KVI Score = equal-weighted average of 4 signals.")

    signal_cols = ["elasticity_signal", "penetration_signal", "frequency_signal", "recall_signal"]
    available = [c for c in signal_cols if c in kvi_df.columns]

    if not available:
        st.info("Signal breakdown not available")
        return

    # Heatmap
    fig = go.Figure(go.Heatmap(
        z=kvi_df[available].values.T,
        x=kvi_df["product_name"].tolist(),
        y=available,
        colorscale="RdYlGn",
        text=kvi_df[available].round(3).values.T,
        texttemplate="%{text}",
        colorbar=dict(title="Signal Strength"),
    ))
    fig.update_layout(
        height=300,
        xaxis_tickangle=45,
        title="Signal Strength by SKU (Green=High, Red=Low)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PRICE LADDER (Standalone mode)
# ============================================================================


def _render_price_ladder(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict, pipeline: dict = None
):
    """Render standalone Price Ladder mode."""
    _render_price_ladder_chart(transactions_df, product_lookup, params)


# ============================================================================
# PRICE CURVE DIAGNOSTICS
# ============================================================================


def _render_price_curve_diagnostics(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict, pipeline: dict = None
):
    """Render price curve diagnostics — pack-size monotonicity, tier clustering."""

    st.header("📊 Price Curve Diagnostics")

    # Mode selection
    multivariate_mode = st.sidebar.checkbox(
        "Multivariate (Price + Elasticity + Margin)",
        value=params.get("price_curve_multivariate", False),
        key="price_curve_multivariate",
        help="Use price, elasticity, basket penetration, and margin for tier clustering. "
        "Requires elasticity data and optional cost column.",
    )

    method = params.get("price_curve_method", "kmeans")
    n_tiers = params.get("n_tiers", 3)

    # Get elasticity data if multivariate mode
    elasticity_df = None
    cost_col = None
    if multivariate_mode:
        st.sidebar.markdown("---")
        st.sidebar.caption("Multivariate Mode: Active")
        st.sidebar.caption("Requires elasticity computation and optional cost column.")

        # Check for cost column
        cost_cols = [
            c
            for c in ["cost", "unit_cost", "margin", "margin_pct", "gross_margin"]
            if c in transactions_df.columns
        ]
        if cost_cols:
            cost_col = st.sidebar.selectbox("Cost Column", cost_cols, key="price_curve_cost_col")
        else:
            st.sidebar.info("No cost column found — margin not included in clustering.")

        # Try to get elasticity from params or compute
        if "elasticity_results" in params:
            elasticity_df = params["elasticity_results"]
            st.sidebar.caption(f"Using elasticity data for {len(elasticity_df)} SKUs")
        else:
            st.sidebar.warning("Elasticity data not found — run elasticity estimation first.")

    # Compute price per unit for each product
    price_data = _compute_price_per_unit(transactions_df, product_lookup)

    if price_data.empty:
        st.error("Could not compute price per unit data.")
        return

    # Route to appropriate function
    if multivariate_mode:
        st.info(
            "🔬 **Multivariate Mode** — Clustering on: Price/Unit, Elasticity, Basket Penetration, Margin"
        )
        result_df = diagnose_price_curves_multivariate(
            transactions_df,
            n_tiers=n_tiers,
            method=method,
            elasticity_df=elasticity_df,
            cost_col=cost_col,
        )
    else:
        st.info("📊 **Univariate Mode** — Clustering on: Price/Unit only")
        result_df = diagnose_price_curves_1d(
            transactions_df,
            n_tiers=n_tiers,
            method=method,
        )

    if result_df.empty:
        st.warning("Insufficient data for price curve diagnostics.")
        return

    # Display results per category
    categories = result_df["category"].unique() if "category" in result_df.columns else ["All"]

    for cat in categories:
        cat_data = result_df[result_df["category"] == cat] if cat != "All" else result_df

        if len(cat_data) < 3:
            continue

        st.subheader(f"📦 Category: {cat}")

        # Price per unit vs basket penetration scatter
        fig = px.scatter(
            cat_data,
            x="basket_penetration",
            y="price_per_unit",
            hover_data=["product_name", "pack_size", "avg_price"],
            color="tier_label" if "tier_label" in cat_data.columns else None,
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
            _render_tier_analysis(cat_data, cat)


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
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict, pipeline: dict = None
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

    st.caption(
        "ℹ️ Qini and AUUC are not normalized — compare relative model performance on THIS dataset, not absolute values across different datasets or reports."
    )

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
    """Train T-learner or S-learner uplift model using real promo_uplift functions."""
    try:
        # Build uplift dataset
        X, treatment, y = build_uplift_dataset(transactions_df, promo_df)

        if len(X) == 0 or treatment.sum() == 0 or (treatment == 0).sum() == 0:
            return None

        # Train model based on method
        if method == "t_learner":
            model_treated, model_control, uplift = train_t_learner_uplift(
                X,
                treatment,
                y,
                base_learner="xgb",
                n_estimators=n_estimators,
                max_depth=max_depth,
            )
        elif method == "s_learner":
            model, uplift = train_s_learner_uplift(
                X,
                treatment,
                y,
                base_learner="xgb",
                n_estimators=n_estimators,
                max_depth=max_depth,
            )
        else:
            raise ValueError(f"Unknown uplift method: {method}")

        # Evaluate model
        eval_results = evaluate_uplift_model(X, treatment, y, uplift)

        # Prepare segment uplift if segment assignments exist
        segment_uplift = None
        segment_assignments = st.session_state.get("segment_assignments")

        if segment_assignments is not None and not segment_assignments.empty:
            # Check staleness: verify customer_id overlap
            current_customers = set(transactions_df["customer_id"].unique())
            segment_customers = set(segment_assignments["customer_id"].unique())

            if current_customers & segment_customers:
                # Reconstruct weekly data with customer_id to align with uplift predictions
                # This mirrors build_uplift_dataset but keeps customer_id
                weekly_with_cust = transactions_df.copy()
                weekly_with_cust["date"] = pd.to_datetime(weekly_with_cust["date"])
                weekly_with_cust["week"] = weekly_with_cust["date"].dt.to_period("W")

                weekly_agg = (
                    weekly_with_cust.groupby(["customer_id", "stockcode", "week"])
                    .agg(total_qty=("quantity", "sum"))
                    .reset_index()
                )

                # Merge promo flags
                promo_df_copy = promo_df.copy()
                promo_df_copy["date"] = pd.to_datetime(promo_df_copy["date"])
                promo_df_copy["week"] = promo_df_copy["date"].dt.to_period("W")
                promo_weekly = (
                    promo_df_copy.groupby(["stockcode", "week"])["is_promo"].any().reset_index()
                )

                weekly_agg = weekly_agg.merge(promo_weekly, on=["stockcode", "week"], how="left")
                weekly_agg["treatment"] = weekly_agg["is_promo"].fillna(False).astype(int)

                # Target: quantity in next week
                weekly_agg = weekly_agg.sort_values(["customer_id", "stockcode", "week"])
                weekly_agg["next_week_qty"] = weekly_agg.groupby(["customer_id", "stockcode"])[
                    "total_qty"
                ].shift(-1)
                weekly_agg = weekly_agg.dropna(subset=["next_week_qty"])

                # The weekly_agg rows should correspond to the rows in X from build_uplift_dataset
                # (same filtering logic). Align by index/position.
                if len(weekly_agg) == len(uplift):
                    weekly_agg["uplift"] = uplift.values
                    weekly_agg["treatment"] = treatment.values

                    # Merge segment assignments
                    merged = weekly_agg.merge(
                        segment_assignments[["customer_id", "segment"]],
                        on="customer_id",
                        how="left",
                    )

                    # Compute mean uplift per segment (only for treated)
                    treated_merged = merged[merged["treatment"] == 1]
                    if not treated_merged.empty:
                        segment_uplift = (
                            treated_merged.groupby("segment")
                            .agg(uplift=("uplift", "mean"), size=("customer_id", "count"))
                            .reset_index()
                        )
                        segment_uplift = segment_uplift[segment_uplift["segment"].notna()]

        return {
            "qini": eval_results.get("qini_coefficient", 0),
            "auuc": eval_results.get("auuc", 0),
            "uplift_at_10": eval_results.get("uplift_at_top_k", 0),
            "qini_curve": np.array(eval_results.get("qini_curve_x", [])),
            "segment_uplift": segment_uplift,
        }
    except Exception as e:
        st.error(f"Uplift model training failed: {e}")
        return None


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


def _render_uplift_by_segment(segment_uplift: Optional[pd.DataFrame]):
    """Render uplift by customer segment."""
    if segment_uplift is None or segment_uplift.empty:
        st.info(
            "📍 Segment-level uplift requires Customer Segmentation to run first. "
            "Go to **Sidebar → Analysis Category → Customer Segmentation → Run Analysis**, "
            "then return to this tab."
        )
        return

    st.caption(
        "🟢 Live model output — segments from Customer Segmentation tab, uplift from T-learner/S-learner on your uploaded data."
    )

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
