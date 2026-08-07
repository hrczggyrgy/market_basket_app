"""Promotional Analytics tab."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.promo import (
    calculate_promotional_lift,
    compute_incrementality_waterfall,
    compute_promo_baseline,
    detect_promotions,
    promo_roi_analysis,
    promotion_timing_analysis,
)
from src.ui.plots import PALETTE, empty_state, new_fig, render_bar_with_ci, show
from src.ui.registry import ModeSpec


def _render_promo_periods(promos: pd.DataFrame) -> None:
    st.subheader(":material/calendar_month: Detected Promotional Periods")
    if promos.empty:
        show(empty_state("No promotional periods detected"))
        return

    # Duration vs Discount scatter
    fig = px.scatter(
        promos,
        x="duration_days",
        y="avg_discount_pct",
        size="promo_revenue",
        color="stockcode",
        hover_data=["start_date", "end_date", "qty_lift", "revenue_lift"],
        size_max=40,
    )
    fig.update_layout(xaxis={"title": "Duration (days)"}, yaxis={"title": "Avg Discount %"})
    show(fig)

    # Timeline
    fig2 = go.Figure()
    for i, (_, row) in enumerate(promos.iterrows()):
        fig2.add_trace(
            go.Scatter(
                x=[row["start_date"], row["end_date"]],
                y=[row["stockcode"], row["stockcode"]],
                mode="lines",
                line={"width": 10, "color": PALETTE[i % len(PALETTE)]},
                name=row["stockcode"],
                hovertemplate=f"{row['stockcode']}<br>Start: {row['start_date']}<br>End: {row['end_date']}<br>Discount: {row['avg_discount_pct']:.1f}%<extra></extra>",
            )
        )
    fig2.update_layout(yaxis={"title": "SKU"}, xaxis={"title": "Date"}, height=max(300, 20 * len(promos)))
    show(fig2)

    st.dataframe(promos, use_container_width=True, hide_index=True)


def _render_lift_analysis(lift: pd.DataFrame) -> None:
    st.subheader(":material/trending_up: Promotional Lift (DiD)")
    if lift.empty:
        show(empty_state("No significant promotional lift detected"))
        return

    # Lift waterfall per promo
    fig = px.bar(
        lift,
        x="stockcode",
        y=["lift_revenue_pct", "lift_qty_pct", "lift_orders_pct"],
        barmode="group",
        color_discrete_sequence=PALETTE[:3],
        hover_data=["start_date", "end_date", "p_value", "significant"],
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(xaxis={"tickangle": -45}, yaxis={"title": "Lift %"})
    show(fig)

    # Significance
    sig = lift[lift["significant"]]
    st.metric("Significant Promotions", len(sig), f"out of {len(lift)} total")

    st.dataframe(lift, use_container_width=True, hide_index=True)


def _render_waterfall(waterfall: pd.DataFrame) -> None:
    st.subheader(":material/waterfall: Incrementality Waterfall")
    if waterfall.empty:
        show(empty_state("No incrementality waterfall available"))
        return

    # Stacked bar: baseline, incremental, halo, cannibalization, net
    fig = go.Figure()
    for col, name, color in [
        ("baseline_revenue", "Baseline", PALETTE[2]),
        ("incremental_revenue", "Incremental", PALETTE[0]),
        ("halo_revenue", "Halo", PALETTE[3]),
        ("cannibalization_revenue", "Cannibalization", PALETTE[4]),
        ("acceleration_revenue", "Acceleration", PALETTE[5]),
        ("switching_revenue", "Switching", PALETTE[1]),
        ("stockpiling_revenue", "Stockpiling", PALETTE[0]),
    ]:
        if col in waterfall.columns:
            fig.add_trace(
                go.Bar(
                    x=waterfall["stockcode"],
                    y=waterfall[col],
                    name=name,
                    marker={"color": color},
                )
            )
    fig.add_trace(
        go.Scatter(
            x=waterfall["stockcode"],
            y=waterfall["net_incremental_revenue"],
            mode="lines+markers",
            name="Net Incremental",
            line={"color": "black", "width": 3},
            marker={"size": 8},
        )
    )
    fig.update_layout(barmode="relative", xaxis={"tickangle": -45}, yaxis={"title": "Revenue ($)"})
    show(fig)

    # ROI
    if "roi" in waterfall.columns:
        fig2 = px.bar(
            waterfall,
            x="stockcode",
            y="roi",
            color=waterfall["roi"].apply(lambda x: "Positive" if x > 0 else "Negative"),
            color_discrete_map={"Positive": PALETTE[0], "Negative": PALETTE[4]},
            text=waterfall["roi"].apply(lambda x: f"{x:.1%}"),
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(yaxis={"title": "ROI"})
        show(fig2)

    st.dataframe(waterfall, use_container_width=True, hide_index=True)


def _render_roi(roi: pd.DataFrame) -> None:
    st.subheader(":material/attach_money: Promo ROI Analysis")
    if roi.empty:
        show(empty_state("No ROI data available"))
        return

    # ROI with CI bars
    if "ci_low" in roi.columns and "ci_high" in roi.columns:
        roi_plot = roi.copy()
        roi_plot["label"] = roi_plot["stockcode"]

        fig = render_bar_with_ci(
            df=roi_plot,
            x_col="label",
            y_col="roi_pct",
            ci_lower_col="ci_low",
            ci_upper_col="ci_high",
            y_title="ROI %",
            height=450,
        )
        show(fig)
        st.caption("ROI % with 95% bootstrap confidence intervals. Error bars show uncertainty in incremental revenue estimation.")

    # Incremental profit scatter (secondary view)
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=roi["stockcode"],
            y=roi["incremental_profit"],
            mode="markers",
            marker={"size": roi["incremental_profit"].abs().apply(lambda v: max(6, min(30, v / 1000))), "color": PALETTE[2]},
            name="Incremental Profit",
            hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
        )
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(
        xaxis={"tickangle": -45, "title": "SKU"},
        yaxis={"title": "Incremental Profit ($)"},
    )
    show(fig2)

    st.dataframe(roi, use_container_width=True, hide_index=True)


def _render_timing(timing: dict[str, pd.DataFrame]) -> None:
    st.subheader(":material/access_time: Promo Timing Effectiveness")
    if not timing:
        show(empty_state("No timing data"))
        return

    if "by_day_of_week" in timing and not timing["by_day_of_week"].empty:
        dow = timing["by_day_of_week"]
        fig = px.bar(
            dow.sort_values("day_name"),
            x="day_name",
            y="revenue_lift",
            color=dow["revenue_lift"].apply(lambda x: "Positive" if x > 0 else "Negative"),
            color_discrete_map={"Positive": PALETTE[0], "Negative": PALETTE[4]},
            text=dow["revenue_lift"].apply(lambda x: f"{x:.1f}%"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis={"title": "Day of Week"}, yaxis={"title": "Revenue Lift %"})
        show(fig)

    if "by_month" in timing and not timing["by_month"].empty:
        month = timing["by_month"]
        fig2 = px.bar(
            month.sort_values("month"),
            x="month_name",
            y="revenue_lift",
            color=month["revenue_lift"].apply(lambda x: "Positive" if x > 0 else "Negative"),
            color_discrete_map={"Positive": PALETTE[0], "Negative": PALETTE[4]},
            text=month["revenue_lift"].apply(lambda x: f"{x:.1f}%"),
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(xaxis={"title": "Month"}, yaxis={"title": "Revenue Lift %"})
        show(fig2)


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/local_offer: Promotional Analytics")

    with st.expander("Detection Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        price_change_threshold = c1.number_input("Price Drop Threshold", 0.05, 0.50, 0.15, 0.01)
        min_duration = c2.number_input("Min Duration (days)", 1, 30, 3)
        max_duration = c3.number_input("Max Duration (days)", 7, 120, 60)

    promos = detect_promotions(
        df,
        price_change_threshold=price_change_threshold,
        min_duration_days=min_duration,
        max_duration_days=max_duration,
    )

    if promos.empty:
        st.warning("No promotional periods detected with current parameters.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Promo Periods", "Lift Analysis", "Waterfall", "ROI", "Timing"])

    with tab1:
        _render_promo_periods(promos)

    with tab2:
        lift = calculate_promotional_lift(df, promo_periods=promos)
        _render_lift_analysis(lift)

    with tab3:
        baseline_df = compute_promo_baseline(df, promo_periods=promos)
        waterfall = compute_incrementality_waterfall(baseline_df)
        _render_waterfall(waterfall)

    with tab4:
        roi = promo_roi_analysis(df, promo_periods=promos)
        _render_roi(roi)

    with tab5:
        timing = promotion_timing_analysis(df, promos)
        _render_timing(timing)


MODE_SPEC: ModeSpec = ModeSpec(
    key="promo",
    label="Promotions",
    icon=":material/local_offer:",
    handler=render,
)