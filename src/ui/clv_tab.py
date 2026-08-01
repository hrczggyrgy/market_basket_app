"""CLV Analytics Tab - BG/NBD customer lifetime value with basket metrics."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.clv_customer import (
    compute_clv_customer_df,
    compute_clv_segment_profiles,
    get_rfm_heatmap_data,
)
from src.analytics.sufficiency import assess_data_sufficiency, format_sufficiency_summary
from src.ui.export import render_analytics_export
from src.ui.tabs import persistent_tabs
from src.ui.insight_header import render_result_context
from src.ui.data_quality import render_data_quality_expander


def _render_clv_readiness_notes():
    """Render BG/NBD applicability and readiness notes."""
    with st.expander(" BG/NBD Model Readiness & Applicability Notes", expanded=False):
        st.markdown("""
        **When BG/NBD Works Well:**
        - ✅ Frequent, repeat-purchase categories (groceries, consumables, coffee, personal care)
        - ✅ Non-contractual settings (customers can come and go freely)
        - ✅ Sufficient repeat customers (≥100 with ≥2 purchases each)
        - ✅ Sufficient time span (≥6 months for weekly, ≥12 months for monthly)
        - ✅ Customers make independent purchase decisions

        **When BG/NBD May Be Unreliable:**
        - ❌ Durable/one-time purchase categories (electronics, furniture, apparel)
        - ❌ Contractual/subscription settings (use Pareto/NBD instead)
        - ❌ Too few repeat customers (<50 with repeat purchases)
        - ❌ Too short observation window (<3 months)
        - ❌ Strong seasonality not captured by model
        - ❌ Major marketing interventions during observation period

        **Model Assumptions (Know Before You Trust):**
        - Purchases follow Poisson process while "alive"
        - Lifetime follows exponential distribution
        - Customers are independent and homogeneous within segments
        - No covariates (marketing, seasonality) included in base model
        - "Alive" means still in the buying process, not literally alive

        **Validation Checklist Before Using CLV:**
        - [ ] P(alive) heatmap shows smooth diagonal gradient (top-left green → bottom-right red)
        - [ ] Top-left (recent + frequent) ≈ 1.0, Bottom-right ≈ 0.0
        - [ ] No green cells in bottom-right (old + low freq = churned)
        - [ ] Customer counts per heatmap cell ≥ 5 for reliable estimates
        - [ ] Segment profiles show clear differentiation (Champions ≠ Lost)
        """)
from src.ui.insight_header import render_result_context
from src.ui.data_quality import render_data_quality_expander


def render_clv_tab(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict, pipeline: dict = None
):
    """Render CLV Analytics tab with persistent sub-tabs."""
    st.header(" CLV Analytics")
    st.caption(
        "BG/NBD + Gamma-Gamma CLV  |  P(alive) from Fader & Hardie  |  "
        "Joined with IPT-CV & Customer Entropy from basket metrics"
    )

    if transactions_df.empty:
        st.warning("No transaction data available")
        return

    # BG/NBD Applicability & Readiness Notes
    _render_clv_readiness_notes()

    # Data sufficiency gate
    sufficiency = assess_data_sufficiency(transactions_df)
    with st.expander(" Data Sufficiency", expanded=sufficiency["overall"] != "robust"):
        st.markdown(format_sufficiency_summary(sufficiency))
        if sufficiency["overall"] == "insufficient":
            st.warning("Dataset may be too small for reliable CLV modeling.")
        elif sufficiency["overall"] == "directional":
            st.info("CLV results should be treated as directional.")

    # Data quality & readiness at top
    render_data_quality_expander(transactions_df, "clv", params, expanded=False)

    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        horizon_days = st.slider(
            "Prediction Horizon (days)",
            30,
            365,
            params.get("prediction_horizon_days", 90),
            key="clv_horizon_days",
            help="Days into future for BG/NBD purchase prediction.",
        )
    with col2:
        freq = st.selectbox(
            "Time Frequency",
            ["D", "W"],
            format_func=lambda x: "Daily" if x == "D" else "Weekly",
            index=0,
            key="clv_freq",
            help="Daily fits more data points; Weekly is more stable for sparse data.",
        )
    with col3:
        if st.button(" Re-run CLV Model", key="clv_rerun"):
            st.cache_data.clear()
            st.rerun()

    # Compute CLV customer dataframe (cached)
    @st.cache_data
    def get_clv_cached(df, horizon, freq_val):
        return compute_clv_customer_df(df, prediction_horizon_days=horizon, freq=freq_val)

    with st.spinner("Fitting BG/NBD model and computing customer metrics..."):
        clv_df = get_clv_cached(transactions_df, horizon_days, freq)

    if clv_df.empty:
        st.warning("No CLV data available - insufficient repeat customers")
        return

    st.success(f"Computed CLV for {len(clv_df):,} customers")

    # Insight header for CLV overview
    active_pct = (clv_df["p_alive"] > 0.5).mean() * 100
    median_palive = clv_df["p_alive"].median()
    mean_clv = clv_df["clv_12m"].mean()
    render_result_context(
        title="BG/NBD CLV Model Overview",
        finding=(
            f"BG/NBD model fit for {len(clv_df):,} customers | "
            f"{active_pct:.1f}% Active (P(alive) > 0.5) | "
            f"Median P(alive): {median_palive:.2f} | "
            f"Mean 12m CLV: ${mean_clv:,.2f}"
        ),
        evidence=f"Horizon: {horizon_days}d | Freq: {'Daily' if freq == 'D' else 'Weekly'} | "
                 f"Segments: {segment_profiles.shape[0]} | "
                 f"Total 12m CLV: ${clv_df['clv_12m'].sum():,.0f}",
        confidence="Directional",
        limitation="BG/NBD assumes Poisson purchases while alive + exponential lifetime. "
                   "Best for frequent, repeat-purchase categories (groceries, consumables). "
                   "Not suitable for durable/one-time purchases or contractual settings. "
                   "No covariates included (marketing, seasonality). "
                   "P(alive) is model probability, not observed ground truth.",
    )

    # Compute segment profiles with correct median
    segment_profiles = compute_clv_segment_profiles(clv_df)

    # Persistent sub-tabs
    tab_labels = [
        " P(alive) Distribution",
        " RFM Heatmap",
        " CLV Waterfall",
        " Entropy × IPT-CV",
    ]
    selected = persistent_tabs(tab_labels, "clv_main_tabs", default_tab=0)

    if selected == 0:
        _render_palive_histogram(clv_df, segment_profiles)
    elif selected == 1:
        _render_rfm_heatmap(clv_df)
    elif selected == 2:
        _render_clv_waterfall(clv_df, segment_profiles)
    elif selected == 3:
        _render_entropy_ipt_cv(clv_df, segment_profiles)

    # Export
    export_df = clv_df.drop(columns=["customer_id"], errors="ignore")
    render_analytics_export(export_df, "CLV_Customer_Metrics")


def _render_palive_histogram(clv_df: pd.DataFrame, segment_profiles: pd.DataFrame):
    """V1 — P(alive) Distribution Histogram."""
    st.subheader("P(alive) Distribution")
    st.caption(
        "Probability customer is still active (BG/NBD). "
        "Blue = Active (P>0.5), Red = At-Risk (P≤0.5). "
        "Healthy base = majority in blue."
    )

    # Active vs at-risk split
    clv_df["alive_status"] = clv_df["p_alive"].apply(lambda x: "Active" if x > 0.5 else "At-Risk")
    active_pct = (clv_df["alive_status"] == "Active").mean() * 100

    # Histogram
    fig = px.histogram(
        clv_df,
        x="p_alive",
        nbins=40,
        color="alive_status",
        color_discrete_map={"Active": "#1565C0", "At-Risk": "#C62828"},
        title=f"Customer P(alive) Distribution — {active_pct:.1f}% Active (P>0.5)",
        labels={"p_alive": "P(alive)", "count": "Customers"},
        hover_data=["count"],
    )
    fig.add_vline(
        x=0.5,
        line_dash="dash",
        line_color="black",
        line_width=2,
        annotation_text="Active/At-Risk Boundary",
        annotation_position="top",
    )
    fig.update_layout(
        height=500,
        bargap=0.1,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(clv_df):,}")
    with col2:
        st.metric("Active Customers (P>0.5)", f"{(clv_df['p_alive'] > 0.5).sum():,}")
    with col3:
        st.metric("Mean P(alive)", f"{clv_df['p_alive'].mean():.2f}")
    with col4:
        st.metric("Median P(alive)", f"{clv_df['p_alive'].median():.2f}")

    # Segment breakdown
    st.subheader("Segment Distribution")
    seg_counts = clv_df["clv_segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]

    fig2 = px.pie(
        seg_counts,
        values="Customers",
        names="Segment",
        title="CLV Segments (P(alive) × CLV_12m)",
        color="Segment",
        color_discrete_map={
            "Champions": "#2E7D32",
            "Promising": "#66BB6A",
            "At Risk": "#FF8F00",
            "Lost": "#C62828",
        },
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Segment profile table
    st.dataframe(
        segment_profiles.style.format(
            {
                "avg_p_alive": "{:.2f}",
                "avg_predicted_purchases": "{:.2f}",
                "avg_expected_avg_value": "${:,.2f}",
                "avg_predicted_clv": "${:,.2f}",
                "avg_clv_12m": "${:,.2f}",
                "total_clv_12m": "${:,.0f}",
                "avg_frequency": "{:.1f}",
                "avg_recency": "{:.0f}",
                "avg_ipt_cv": "{:.2f}",
                "avg_entropy": "{:.2f}",
                "customer_share": "{:.1%}",
                "revenue_share": "{:.1%}",
            }
        ).background_gradient(cmap="RdYlGn", subset=["total_clv_12m", "revenue_share"]),
        use_container_width=True,
    )


def _render_rfm_heatmap(clv_df: pd.DataFrame):
    """V2 — Frequency × Recency Heatmap (RFM Matrix)."""
    st.subheader("RFM Heatmap: Mean P(alive) by Recency × Frequency Decile")
    st.caption(
        "X = Frequency Decile (D1=lowest, D10=highest)  |  "
        "Y = Recency Decile (D1=most recent, D10=oldest)  |  "
        "Color = Mean P(alive). Canonical BG/NBD diagnostic."
    )

    heatmap_pivot = get_rfm_heatmap_data(clv_df)

    if heatmap_pivot.empty:
        st.warning("Insufficient data for heatmap")
        return

    # Also get customer counts for annotation
    df = clv_df.copy()
    df["recency_decile"] = pd.qcut(
        df["recency_days"].rank(method="first"),
        q=10,
        labels=[f"D{i+1}" for i in range(10)],
        duplicates="drop",
    )
    df["frequency_decile"] = pd.qcut(
        df["frequency"].rank(method="first"),
        q=10,
        labels=[f"D{i+1}" for i in range(10)],
        duplicates="drop",
    )
    count_pivot = df.groupby(["recency_decile", "frequency_decile"]).size().unstack(fill_value=0)

    # Create annotated heatmap
    fig = go.Figure(go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns.tolist(),
        y=heatmap_pivot.index.tolist(),
        colorscale="RdYlGn",
        zmin=0,
        zmax=1,
        text=count_pivot.values,
        texttemplate="P=%{z:.2f}<br>N=%{text}",
        hovertemplate=(
            "Recency: %{y}<br>Frequency: %{x}<br>"
            "Mean P(alive): %{z:.3f}<br>Customers: %{text}<extra></extra>"
        ),
        colorbar=dict(title="Mean P(alive)"),
    ))
    fig.update_layout(
        title="BG/NBD Diagnostic: P(alive) by Recency × Frequency",
        xaxis_title="Frequency Decile (D1=Low → D10=High)",
        yaxis_title="Recency Decile (D1=Recent → D10=Old)",
        height=600,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interpretation guide
    with st.expander(" How to Read This Heatmap"):
        st.markdown("""
        **BG/NBD Diagnostic (Fader & Hardie):**

        - **Top-Left (Recent + High Freq)**: Should be dark green (P≈1) — active loyal customers
        - **Bottom-Right (Old + Low Freq)**: Should be red (P≈0) — churned customers
        - **Diagonal Gradient**: Healthy model shows smooth transition
        - **Anomalies**: Green cells in bottom-right = model uncertainty (need more data)
        - **Customer Count (N)**: Low N cells (<5) are unreliable

        **Action Items:**
        - **High P(alive) + Low Frequency** → Reactivation campaigns
        - **Low P(alive) + High Past Value** → Win-back offers
        - **High P(alive) + High Frequency** → VIP retention
        """)


def _render_clv_waterfall(clv_df: pd.DataFrame, segment_profiles: pd.DataFrame):
    """V3 — CLV Segment Waterfall."""
    st.subheader("CLV Segment Waterfall: 12-Month Projected Revenue by Segment")
    st.caption(
        "Stacked bars = sum of CLV_12m per segment. "
        "Secondary axis = customer count. "
        "Shows revenue concentration by customer quality."
    )

    # Sort segments by CLV descending
    seg_order = segment_profiles.sort_values("avg_clv_12m", ascending=False)["clv_segment"].tolist()
    plot_df = segment_profiles.set_index("clv_segment").loc[seg_order].reset_index()

    fig = go.Figure()

    # CLV_12m bars
    fig.add_trace(go.Bar(
        x=plot_df["clv_segment"],
        y=plot_df["total_clv_12m"],
        name="Projected CLV (12m)",
        marker_color=["#2E7D32", "#66BB6A", "#FF8F00", "#C62828"][:len(plot_df)],
        text=[f"${v:,.0f}" for v in plot_df["total_clv_12m"]],
        textposition="outside",
        yaxis="y",
    ))

    # Customer count line
    fig.add_trace(go.Scatter(
        x=plot_df["clv_segment"],
        y=plot_df["n_customers"],
        name="Customer Count",
        mode="lines+markers",
        line=dict(color="black", width=2, dash="dot"),
        marker=dict(size=8),
        yaxis="y2",
    ))

    fig.update_layout(
        title="12-Month CLV by Segment (Bar) vs Customer Count (Line)",
        xaxis_title="Segment",
        yaxis=dict(title="Total CLV 12m ($)", side="left"),
        yaxis2=dict(title="Customers", side="right", overlaying="y", showgrid=False),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Revenue share analysis
    st.subheader("Revenue Concentration")
    col1, col2 = st.columns(2)
    with col1:
        # Pareto of CLV
        clv_sorted = clv_df.sort_values("clv_12m", ascending=False).reset_index(drop=True)
        clv_sorted["cum_clv"] = clv_sorted["clv_12m"].cumsum()
        clv_sorted["cum_pct"] = clv_sorted["cum_clv"] / clv_sorted["clv_12m"].sum() * 100
        clv_sorted["cust_pct"] = (clv_sorted.index + 1) / len(clv_sorted) * 100

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=clv_sorted["cust_pct"],
            y=clv_sorted["cum_pct"],
            mode="lines",
            name="Cumulative CLV %",
            line=dict(color="#1565C0", width=2),
        ))
        fig2.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100],
            mode="lines",
            name="Perfect Equality",
            line=dict(color="gray", width=1, dash="dash"),
        ))
        fig2.add_hline(y=80, line_dash="dot", line_color="red", annotation_text="80% Revenue")
        fig2.add_vline(x=20, line_dash="dot", line_color="red", annotation_text="20% Customers")
        fig2.update_layout(
            title="CLV Concentration (Pareto)",
            xaxis_title="Customers (%)",
            yaxis_title="Cumulative CLV (%)",
            height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Segment contribution table
        st.dataframe(
            plot_df[["clv_segment", "n_customers", "total_clv_12m", "customer_share", "revenue_share"]]
            .style.format({
                "n_customers": "{:,}",
                "total_clv_12m": "${:,.0f}",
                "customer_share": "{:.1%}",
                "revenue_share": "{:.1%}",
            })
            .background_gradient(cmap="RdYlGn", subset=["total_clv_12m", "revenue_share"]),
            use_container_width=True,
        )


def _render_entropy_ipt_cv(clv_df: pd.DataFrame, segment_profiles: pd.DataFrame):
    """V4 — Customer Entropy vs IPT-CV Scatter."""
    st.subheader("Customer Entropy vs. IPT-CV: Purchase Regularity vs Variety")
    st.caption(
        "X = IPT-CV (Inter-Purchase Time CV) — Low = Regular buyer, High = Irregular  |  "
        "Y = Normalized Entropy — Low = Loyal/Concentrated, High = Variety Seeker  |  "
        "4 Quadrants: Loyal Regulars / Variety Regulars / Loyal Irregulars / Explorers"
    )

    # Filter valid data
    plot_df = clv_df.dropna(subset=["ipt_cv", "normalized_entropy"]).copy()

    if plot_df.empty:
        st.warning("Insufficient IPT-CV or Entropy data")
        return

    # Quadrant boundaries (medians)
    x_med = plot_df["ipt_cv"].median()
    y_med = plot_df["normalized_entropy"].median()

    # Quadrant labels
    def assign_quadrant(row):
        if row["ipt_cv"] <= x_med and row["normalized_entropy"] <= y_med:
            return "Loyal Regulars"
        elif row["ipt_cv"] <= x_med and row["normalized_entropy"] > y_med:
            return "Variety Regulars"
        elif row["ipt_cv"] > x_med and row["normalized_entropy"] <= y_med:
            return "Loyal Irregulars"
        else:
            return "Explorers"

    plot_df["quadrant"] = plot_df.apply(assign_quadrant, axis=1)

    # Color map
    quadrant_colors = {
        "Loyal Regulars": "#2E7D32",
        "Variety Regulars": "#1565C0",
        "Loyal Irregulars": "#FF8F00",
        "Explorers": "#C62828",
    }

    fig = px.scatter(
        plot_df,
        x="ipt_cv",
        y="normalized_entropy",
        color="quadrant",
        color_discrete_map=quadrant_colors,
        hover_data=["customer_id", "p_alive", "clv_12m", "frequency", "recency_days", "clv_segment"],
        title=f"Entropy vs IPT-CV (Median IPT-CV={x_med:.2f}, Median Entropy={y_med:.2f})",
        labels={
            "ipt_cv": "IPT-CV (Purchase Regularity) — Lower = More Regular",
            "normalized_entropy": "Normalized Entropy — Lower = More Concentrated",
        },
        opacity=0.6,
    )

    # Quadrant lines
    fig.add_hline(y=y_med, line_dash="dash", line_color="gray", line_width=1)
    fig.add_vline(x=x_med, line_dash="dash", line_color="gray", line_width=1)

    # Quadrant annotations
    x_max = plot_df["ipt_cv"].max()
    x_min = plot_df["ipt_cv"].min()
    y_max = plot_df["normalized_entropy"].max()
    y_min = plot_df["normalized_entropy"].min()

    fig.add_annotation(x=x_max*0.7, y=y_max*0.9, text="<b>Explorers</b>", showarrow=False, font=dict(color="#C62828"))
    fig.add_annotation(x=x_min*1.5, y=y_max*0.9, text="<b>Variety Regulars</b>", showarrow=False, font=dict(color="#1565C0"))
    fig.add_annotation(x=x_max*0.7, y=y_min*1.2, text="<b>Loyal Irregulars</b>", showarrow=False, font=dict(color="#FF8F00"))
    fig.add_annotation(x=x_min*1.5, y=y_min*1.2, text="<b>Loyal Regulars</b>", showarrow=False, font=dict(color="#2E7D32"))

    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Quadrant summary
    st.subheader("Quadrant Summary")
    quad_summary = (
        plot_df.groupby("quadrant")
        .agg(
            n_customers=("customer_id", "count"),
            avg_clv_12m=("clv_12m", "mean"),
            avg_p_alive=("p_alive", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_recency=("recency_days", "mean"),
            avg_ipt_cv=("ipt_cv", "mean"),
            avg_entropy=("normalized_entropy", "mean"),
        )
        .reset_index()
    )
    quad_order = ["Loyal Regulars", "Variety Regulars", "Loyal Irregulars", "Explorers"]
    quad_summary["quadrant"] = pd.Categorical(quad_summary["quadrant"], categories=quad_order, ordered=True)
    quad_summary = quad_summary.sort_values("quadrant")

    st.dataframe(
        quad_summary.style.format({
            "n_customers": "{:,}",
            "avg_clv_12m": "${:,.2f}",
            "avg_p_alive": "{:.2f}",
            "avg_frequency": "{:.1f}",
            "avg_recency": "{:.0f}",
            "avg_ipt_cv": "{:.2f}",
            "avg_entropy": "{:.2f}",
        }).background_gradient(cmap="RdYlGn", subset=["avg_clv_12m", "avg_p_alive"]),
        use_container_width=True,
    )

    # Segment cross-tab
    st.subheader("CLV Segment × Quadrant Cross-Tab")
    cross_tab = pd.crosstab(plot_df["clv_segment"], plot_df["quadrant"], margins=True)
    st.dataframe(cross_tab.style.background_gradient(cmap="Blues"), use_container_width=True)