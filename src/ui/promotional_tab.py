"""Promotional Analytics Tab with persistent tab state."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.promotional import detect_promotions
from src.ui.export import render_analytics_export
from src.ui.tabs import persistent_tabs


def render_promotional_tab(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render promotional analytics tab with persistent sub-tabs."""
    st.header("📢 Promotional Analytics")

    if transactions_df.empty:
        st.warning("No transaction data available")
        return

    # Check if we have promotional data (or simulate)
    st.info(
        "💡 This analysis requires promotional period data. Using simulated promotions based on price drops and volume spikes."
    )

    # Detect promotions (cached)
    @st.cache_data
    def detect_promos_cached(df, price_thresh, min_dur, max_dur):
        return detect_promotions(
            df,
            price_change_threshold=price_thresh,
            min_duration_days=min_dur,
            max_duration_days=max_dur,
        )
    with st.spinner("Detecting promotional periods..."):
        promo_periods = detect_promos_cached(
            transactions_df,
            params.get("price_change_threshold", 0.15),
            params.get("min_duration_days", 3),
            params.get("max_duration_days", 30),
        )

    if promo_periods.empty:
        st.warning(
            "No promotional periods detected. Try adjusting thresholds or upload data with promotional flags."
        )
        st.info("Tip: Add a 'is_promotion' column to your data for best results.")
        return

    st.success(f"Detected {len(promo_periods)} promotional periods")

    # Show promo periods
    with st.expander("View Detected Promotions", expanded=False):
        promo_display = promo_periods.copy()
        promo_display["Product"] = promo_display["stockcode"].map(product_lookup)
        promo_display["Duration (days)"] = (
            promo_display["end_date"] - promo_display["start_date"]
        ).dt.days
        st.dataframe(
            promo_display[
                [
                    "Product",
                    "stockcode",
                    "start_date",
                    "end_date",
                    "Duration (days)",
                    "avg_price_drop_pct",
                    "avg_volume_lift_pct",
                ]
            ].style.format({"avg_price_drop_pct": "{:.1f}%", "avg_volume_lift_pct": "{:.1f}%"}),
            width="stretch",
        )

    # Persistent sub-tabs
    promo_tabs = [
        "📈 Promotional Lift",
        "💰 Incremental Revenue",
        "� ROI Analysis",
        "🌟 Halo Effect",
        "📅 Timing Analysis",
        "📊 Period Comparison",
    ]
    selected = persistent_tabs(promo_tabs, "promo_main_tabs", default_tab=0)

    if selected == 0:
        render_promotional_lift_tab(transactions_df, product_lookup, promo_periods, params)
    elif selected == 1:
        render_incremental_revenue_tab(transactions_df, product_lookup, promo_periods, params)
    elif selected == 2:
        render_roi_analysis_tab(transactions_df, product_lookup, promo_periods, params)
    elif selected == 3:
        render_halo_effect_tab(transactions_df, product_lookup, promo_periods, params)
    elif selected == 4:
        render_timing_analysis_tab(transactions_df, product_lookup, promo_periods, params)
    elif selected == 5:
        render_period_comparison_tab(transactions_df, promo_periods, params)


def render_promotional_lift_tab(
    transactions_df: pd.DataFrame,
    product_lookup: dict,
    promo_periods: pd.DataFrame,
    params: dict,
):
    """Render promotional lift analysis."""
    st.subheader("Promotional Lift Analysis")

    with st.spinner("Calculating promotional lift..."):
        from src.analytics.promotional import calculate_promotional_lift

        lift_results = calculate_promotional_lift(transactions_df, promo_periods)

    if isinstance(lift_results, dict) and "error" in lift_results:
        st.warning(f"Could not calculate promotional lift: {lift_results['error']}")
        return

    if isinstance(lift_results, dict) and "product_lifts" in lift_results:
        product_lifts = lift_results["product_lifts"]
    elif isinstance(lift_results, pd.DataFrame):
        product_lifts = lift_results
    else:
        st.warning("Unexpected lift results format")
        st.write(lift_results)
        return

    if product_lifts.empty:
        st.warning("No lift data available")
        return

    # Add product names
    product_lifts["Product"] = product_lifts["stockcode"].map(product_lookup)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Products Analyzed", len(product_lifts))
    with col2:
        avg_lift = product_lifts["lift_pct"].mean()
        st.metric("Avg Revenue Lift", f"{avg_lift:.1f}%")
    with col3:
        positive_lift = (product_lifts["lift_pct"] > 0).sum()
        st.metric("Products with Positive Lift", f"{positive_lift}/{len(product_lifts)}")
    with col4:
        max_lift = product_lifts["lift_pct"].max()
        st.metric("Max Revenue Lift", f"{max_lift:.1f}%")

    # Lift table
    st.subheader("Promotional Lift by Product")
    display_cols = [
        "Product",
        "stockcode",
        "promo_revenue_per_day",
        "non_promo_revenue_per_day",
        "lift_pct",
        "promo_transactions",
        "baseline_transactions",
    ]
    display_cols = [c for c in display_cols if c in product_lifts.columns]

    st.dataframe(
        product_lifts[display_cols]
        .sort_values("lift_pct", ascending=False)
        .style.format(
            {
                "promo_revenue_per_day": "${:,.2f}",
                "non_promo_revenue_per_day": "${:,.2f}",
                "lift_pct": "{:.1f}%",
            }
        )
        .background_gradient(cmap="RdYlGn", subset=["lift_pct"]),
        width="stretch",
    )

    # Lift visualization
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            product_lifts.sort_values("lift_pct", ascending=True).head(20),
            x="lift_pct",
            y="Product",
            orientation="h",
            title="Top 20 Products by Revenue Lift",
            color="lift_pct",
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.scatter(
            product_lifts,
            x="non_promo_revenue_per_day",
            y="lift_pct",
            size="promo_revenue_per_day",
            color="stockcode",
            hover_data=["Product"],
            title="Revenue Lift vs Baseline Revenue",
            labels={
                "non_promo_revenue_per_day": "Baseline Revenue/Day ($)",
                "lift_pct": "Lift (%)",
            },
        )
        st.plotly_chart(fig, width="stretch")

    render_analytics_export(product_lifts, "Promotional_Lift")


def render_incremental_revenue_tab(
    transactions_df: pd.DataFrame,
    product_lookup: dict,
    promo_periods: pd.DataFrame,
    params: dict,
):
    """Render incremental revenue analysis."""
    st.subheader("Incremental Revenue Analysis")

    with st.spinner("Calculating incremental revenue..."):
        from src.analytics.promotional import calculate_incremental_revenue

        inc_revenue = calculate_incremental_revenue(
            transactions_df,
            promo_periods,
            baseline_window=params.get("baseline_window", 30),
        )

    if inc_revenue.empty:
        st.warning("Could not calculate incremental revenue")
        return

    inc_revenue["Product"] = inc_revenue["stockcode"].map(product_lookup)

    col1, col2, col3 = st.columns(3)
    with col1:
        total_inc = inc_revenue["incremental_revenue"].sum()
        st.metric("Total Incremental Revenue", f"${total_inc:,.2f}")
    with col2:
        avg_lift = inc_revenue["revenue_lift_pct"].mean()
        st.metric("Avg Revenue Lift", f"{avg_lift:.1f}%")
    with col3:
        st.metric("Products with Incremental Revenue", len(inc_revenue))

    st.subheader("Incremental Revenue by Product")
    display_cols = [
        "Product",
        "stockcode",
        "incremental_revenue",
        "revenue_lift_pct",
        "promo_revenue",
        "baseline_revenue",
    ]
    display_cols = [c for c in display_cols if c in inc_revenue.columns]

    st.dataframe(
        inc_revenue[display_cols]
        .sort_values("incremental_revenue", ascending=False)
        .style.format(
            {
                "incremental_revenue": "${:,.2f}",
                "promo_revenue": "${:,.2f}",
                "baseline_revenue": "${:,.2f}",
                "revenue_lift_pct": "{:.1f}%",
            }
        )
        .background_gradient(cmap="RdYlGn", subset=["incremental_revenue"]),
        width="stretch",
    )

    # Visualization
    fig = px.bar(
        inc_revenue.sort_values("incremental_revenue", ascending=True).head(20),
        x="incremental_revenue",
        y="Product",
        orientation="h",
        title="Top 20 Products by Incremental Revenue",
        color="revenue_lift_pct",
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")

    render_analytics_export(inc_revenue, "Incremental_Revenue")


def render_roi_analysis_tab(
    transactions_df: pd.DataFrame,
    product_lookup: dict,
    promo_periods: pd.DataFrame,
    params: dict,
):
    """Render ROI analysis."""
    st.subheader("Promotional ROI Analysis")

    st.info(
        "ROI analysis requires promotional cost data. Using estimated costs based on margin assumptions."
    )

    margin_assumption = st.slider("Assumed Margin %", 10, 50, 30, key="roi_margin")
    promo_cost_pct = st.slider("Promotional Cost % of Revenue", 5, 50, 15, key="roi_cost_pct")

    with st.spinner("Calculating ROI..."):
        from src.analytics.promotional import promotion_roi_analysis

        roi_results = promotion_roi_analysis(
            transactions_df,
            promo_periods,
            margin_pct=margin_assumption / 100,
            promo_cost_pct=promo_cost_pct / 100,
        )

    if roi_results.empty:
        st.warning("Could not calculate ROI")
        return

    roi_results["Product"] = roi_results["stockcode"].map(product_lookup)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Promotions", len(roi_results))
    with col2:
        st.metric("Avg ROI", f"{roi_results['roi_pct'].mean():.1f}%")
    with col3:
        profitable = (roi_results["roi_pct"] > 0).sum()
        st.metric("Profitable Promotions", f"{profitable}/{len(roi_results)}")
    with col4:
        total_profit = roi_results["net_profit"].sum()
        st.metric("Total Net Profit", f"${total_profit:,.2f}")

    st.subheader("ROI by Promotion")
    display_cols = [
        "Product",
        "stockcode",
        "incremental_revenue",
        "incremental_profit",
        "promo_cost",
        "net_profit",
        "roi_pct",
    ]
    display_cols = [c for c in display_cols if c in roi_results.columns]

    st.dataframe(
        roi_results[display_cols]
        .sort_values("roi_pct", ascending=False)
        .style.format(
            {
                "incremental_revenue": "${:,.2f}",
                "incremental_profit": "${:,.2f}",
                "promo_cost": "${:,.2f}",
                "net_profit": "${:,.2f}",
                "roi_pct": "{:.1f}%",
            }
        )
        .background_gradient(cmap="RdYlGn", subset=["roi_pct", "net_profit"]),
        width="stretch",
    )

    # ROI visualization
    fig = px.scatter(
        roi_results,
        x="incremental_revenue",
        y="roi_pct",
        size="promo_cost",
        color="net_profit",
        hover_data=["Product"],
        title="ROI vs Incremental Revenue",
        labels={"incremental_revenue": "Incremental Revenue ($)", "roi_pct": "ROI (%)"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, width="stretch")

    render_analytics_export(roi_results, "Promotional_ROI")


def render_halo_effect_tab(
    transactions_df: pd.DataFrame,
    product_lookup: dict,
    promo_periods: pd.DataFrame,
    params: dict,
):
    """Render halo effect analysis."""
    st.subheader("Halo Effect Analysis")
    st.info("Analyzes impact of promotions on non-promoted products in the same transactions.")

    window_days = st.slider("Analysis Window (days)", 1, 30, 7, key="halo_window")

    with st.spinner("Analyzing halo effects..."):
        from src.analytics.promotional import halo_effect_analysis

        halo_results = halo_effect_analysis(transactions_df, promo_periods, window_days=window_days)

    if halo_results.empty:
        st.warning("No halo effects detected")
        return

    halo_results["Promo Product"] = halo_results["promo_product"].map(product_lookup)
    halo_results["Halo Product"] = halo_results["halo_product"].map(product_lookup)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Halo Effects Detected", len(halo_results))
    with col2:
        avg_lift = halo_results["revenue_lift_pct"].mean()
        st.metric("Avg Halo Lift", f"{avg_lift:.1f}%")
    with col3:
        positive = (halo_results["revenue_lift_pct"] > 0).sum()
        st.metric("Positive Halo Effects", f"{positive}/{len(halo_results)}")

    st.subheader("Top Halo Effects")
    display_cols = [
        "Promo Product",
        "Halo Product",
        "halo_revenue",
        "base_revenue",
        "revenue_lift_pct",
        "halo_orders",
        "base_orders",
    ]
    display_cols = [c for c in display_cols if c in halo_results.columns]

    st.dataframe(
        halo_results[display_cols]
        .sort_values("revenue_lift_pct", ascending=False)
        .head(20)
        .style.format(
            {
                "halo_revenue": "${:,.2f}",
                "base_revenue": "${:,.2f}",
                "revenue_lift_pct": "{:.1f}%",
            }
        )
        .background_gradient(cmap="RdYlGn", subset=["revenue_lift_pct"]),
        width="stretch",
    )

    # Visualization
    fig = px.scatter(
        halo_results,
        x="base_revenue",
        y="revenue_lift_pct",
        size="halo_revenue",
        color="Promo Product",
        hover_data=["Halo Product"],
        title="Halo Effect: Base Revenue vs Lift",
        labels={
            "base_revenue": "Base Revenue ($)",
            "revenue_lift_pct": "Revenue Lift (%)",
        },
    )
    st.plotly_chart(fig, width="stretch")

    render_analytics_export(halo_results, "Halo_Effect")


def render_timing_analysis_tab(
    transactions_df: pd.DataFrame,
    product_lookup: dict,
    promo_periods: pd.DataFrame,
    params: dict,
):
    """Render promotion timing analysis."""
    st.subheader("Promotion Timing Analysis")

    with st.spinner("Analyzing promotion timing..."):
        from src.analytics.promotional import promotion_timing_analysis

        timing = promotion_timing_analysis(transactions_df, promo_periods)

    if not timing:
        st.warning("Insufficient data for timing analysis")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("By Day of Week")
        dow_data = timing.get("by_day_of_week")
        if dow_data is None or dow_data.empty:
            st.info("No day-of-week data available")
        else:
            fig = px.bar(
                dow_data,
                x="day_name",
                y="revenue_lift",
                color="revenue_lift",
                color_continuous_scale="RdYlGn",
                title="Revenue Lift by Day of Week",
                labels={"day_name": "Day", "revenue_lift": "Revenue Lift (%)"},
            )
            st.plotly_chart(fig, width="stretch")

            st.dataframe(
                dow_data[
                    [
                        "day_name",
                        "promo_revenue",
                        "base_revenue",
                        "revenue_lift",
                        "promo_orders",
                        "base_orders",
                    ]
                ].style.format(
                    {
                        "promo_revenue": "${:,.2f}",
                        "base_revenue": "${:,.2f}",
                        "revenue_lift": "{:.1f}%",
                    }
                ),
                width="stretch",
            )

    with col2:
        st.subheader("By Month")
        month_data = timing.get("by_month")
        if month_data is None or month_data.empty:
            st.info("No monthly data available")
        else:
            fig = px.bar(
                month_data,
                x="month",
                y="revenue_lift",
                color="revenue_lift",
                color_continuous_scale="RdYlGn",
                title="Revenue Lift by Month",
                labels={"month": "Month", "revenue_lift": "Revenue Lift (%)"},
            )
            st.plotly_chart(fig, width="stretch")

            st.dataframe(
                month_data[
                    [
                        "month",
                        "promo_revenue",
                        "base_revenue",
                        "revenue_lift",
                        "promo_orders",
                        "base_orders",
                    ]
                ].style.format(
                    {
                        "promo_revenue": "${:,.2f}",
                        "base_revenue": "${:,.2f}",
                        "revenue_lift": "{:.1f}%",
                    }
                ),
                width="stretch",
            )


def render_period_comparison_tab(
    transactions_df: pd.DataFrame, promo_periods: pd.DataFrame, params: dict
):
    """Render period-over-period and YoY comparison."""
    st.subheader("Period-over-Period & Year-over-Year Comparison")

    comparison_type = st.radio(
        "Comparison Type",
        ["Period-over-Period (PoP)", "Year-over-Year (YoY)"],
        horizontal=True,
        key="promo_comparison_type",
    )

    if "Period-over-Period" in comparison_type:
        period = st.selectbox(
            "Period", ["Monthly", "Weekly", "Quarterly"], index=1, key="pop_period"
        )
        period_map = {"Monthly": "M", "Weekly": "W", "Quarterly": "Q"}

        from src.analytics.cohort import period_over_period_comparison

        pop = period_over_period_comparison(transactions_df, period=period_map[period])

        st.subheader(f"{period} Period-over-Period Comparison")

        metrics = [
            "revenue",
            "orders",
            "customers",
            "avg_order_value",
            "items_per_order",
        ]
        for metric in metrics:
            if metric in pop.columns and f"{metric}_pop_pct" in pop.columns:
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=pop["period"],
                        y=pop[metric],
                        name=metric.replace("_", " ").title(),
                        yaxis="y",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=pop["period"],
                        y=pop[f"{metric}_pop_pct"],
                        name=f"{metric} % Change",
                        yaxis="y2",
                        line=dict(color="red"),
                    )
                )
                fig.update_layout(
                    title=f"{metric.replace('_', ' ').title()} with PoP % Change",
                    yaxis=dict(title=metric),
                    yaxis2=dict(title="% Change", overlaying="y", side="right"),
                )
                st.plotly_chart(fig, width="stretch")

    else:
        st.subheader("Year-over-Year Comparison")

        from src.analytics.cohort import year_over_year_comparison

        yoy = year_over_year_comparison(transactions_df)

        if not yoy.empty:
            # Show YoY table
            st.dataframe(yoy.style.format("{:,.2f}"), width="stretch")

            # Revenue YoY chart
            rev_cols = [c for c in yoy.columns if c.startswith("revenue_")]
            if len(rev_cols) >= 2:
                yoy_melted = yoy.melt(
                    id_vars=["month", "month_name"],
                    value_vars=rev_cols,
                    var_name="Year",
                    value_name="Revenue",
                )
                yoy_melted["Year"] = yoy_melted["Year"].str.replace("revenue_", "")

                fig = px.bar(
                    yoy_melted,
                    x="month_name",
                    y="Revenue",
                    color="Year",
                    barmode="group",
                    title="Revenue by Month and Year",
                    labels={"month_name": "Month", "Revenue": "Revenue ($)"},
                )
                st.plotly_chart(fig, width="stretch")
