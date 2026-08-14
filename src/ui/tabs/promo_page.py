"""Promotional Analytics tab.

Follows the app-wide page pattern: promo scorecard (WIN / MIXED /
INEFFECTIVE / DESTROYS VALUE) -> waterfall -> ROI -> timing -> top insights ->
ranked decisions.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.insights import generate_promotion_insights
from src.analytics.insights.promotion import classify_promo_score
from src.analytics.opportunities import generate_promotion_opportunities
from src.analytics.promo import (
    compute_cannibalization_analysis,
    compute_incrementality_waterfall,
    compute_promo_baseline,
    detect_promotions,
    pre_post_promo_comparison,
    promo_roi_analysis,
    promotion_timing_analysis,
)
from src.ui.components import render_insight_cards, render_metric_row, render_opportunity_table
from src.ui.plots import PALETTE, empty_state, new_fig, render_bar_with_ci, show
from src.ui.registry import ModeSpec

_SCORE_META = {
    "WIN": {"icon": ":material/trending_up:", "color": "#59A14F"},
    "MIXED": {"icon": ":material/swap_vert:", "color": "#F28E2B"},
    "INEFFECTIVE": {"icon": ":material/remove:", "color": "#B07AA1"},
    "DESTROYS_VALUE": {"icon": ":material/error:", "color": "#E15759"},
}


def _render_scorecard(waterfall: pd.DataFrame, roi: pd.DataFrame) -> None:
    st.subheader(":material/leaderboard: Promo Scorecard")
    if waterfall.empty:
        show(empty_state("No promo waterfall data"))
        return

    work = waterfall.copy()
    if roi is not None and not roi.empty:
        work = work.merge(roi[["stockcode", "roi_pct"]], on="stockcode", how="left")
    else:
        work["roi_pct"] = None
    work["score"] = work.apply(classify_promo_score, axis=1)

    counts = work["score"].value_counts()
    metrics = []
    for score in ("WIN", "MIXED", "INEFFECTIVE", "DESTROYS_VALUE"):
        n = int(counts.get(score, 0))
        sub = work[work["score"] == score]
        net = float(sub["net_incremental_revenue"].sum())
        meta = _SCORE_META[score]
        metrics.append(
            {
                "label": f"{meta['icon']} {score}",
                "value": str(n),
                "help": f"Net incremental revenue: €{net:,.0f}. {score.replace('_', ' ').lower()} promos.",
            }
        )
    render_metric_row(metrics)

    fig = new_fig()
    order = ["WIN", "MIXED", "INEFFECTIVE", "DESTROYS_VALUE"]
    present = [s for s in order if s in counts.index]
    fig.add_trace(
        go.Bar(
            x=[_SCORE_META[s]["icon"].replace(":material/", "").replace(":", "") for s in present],
            y=[counts[s] for s in present],
            marker={"color": [_SCORE_META[s]["color"] for s in present]},
            text=[str(counts[s]) for s in present],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis={"title": ""},
        yaxis={"title": "Promotions"},
        xaxis_tickangle=0,
    )
    show(fig)
    st.caption(
        "WIN = positive net incremental with ROI ≥ break-even · MIXED = volume up but "
        "margin down · INEFFECTIVE = no incremental volume · DESTROYS VALUE = net negative. "
        "Scorecard is descriptive; validate wins/destroyers with the causal engine."
    )


def _render_incremental_vs_observed(waterfall: pd.DataFrame) -> None:
    st.subheader(":material/compare_arrows: Observed vs Incremental Revenue")
    if waterfall.empty:
        show(empty_state("No waterfall data"))
        return
    top = waterfall.sort_values(
        "actual_revenue" if "actual_revenue" in waterfall.columns else "baseline_revenue",
        ascending=False,
    ).head(15)
    if "actual_revenue" not in top.columns:
        top["actual_revenue"] = top["baseline_revenue"] + top["incremental_revenue"].fillna(0)
    top["actual_revenue"] = top["actual_revenue"].fillna(0)

    fig = new_fig()
    fig.add_trace(
        go.Bar(
            x=top["stockcode"],
            y=top["actual_revenue"],
            name="Observed revenue",
            marker={"color": PALETTE[2]},
            hovertemplate="%{x}<br>Observed: %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=top["stockcode"],
            y=top["incremental_revenue"].fillna(0),
            name="Incremental (vs baseline)",
            marker={"color": PALETTE[0]},
            hovertemplate="%{x}<br>Incremental: %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        barmode="group",
        xaxis={"title": "", "tickangle": -45},
        yaxis={"title": "Revenue"},
    )
    show(fig)
    st.caption(
        "Observed revenue is the headline; incremental revenue is what the promo "
        "actually added vs its no-promo baseline. A big gap = mostly a discount "
        "give-away on demand that existed anyway."
    )


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
    fig2.update_layout(
        yaxis={"title": "SKU"}, xaxis={"title": "Date"}, height=max(300, 20 * len(promos))
    )
    show(fig2)

    st.dataframe(promos, use_container_width=True, hide_index=True)


def _render_lift_analysis(lift: pd.DataFrame) -> None:
    st.subheader(":material/trending_up: Promotional Pre/Post Comparison")
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
    st.metric("Significant Promotions", len(sig))
    st.caption(f"{len(sig)} out of {len(lift)} total promotions are significant")

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
        st.caption(
            "ROI % with 95% bootstrap confidence intervals. Error bars show uncertainty in incremental revenue estimation."
        )

    # Incremental profit scatter (secondary view)
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=roi["stockcode"],
            y=roi["incremental_profit"],
            mode="markers",
            marker={
                "size": roi["incremental_profit"].abs().apply(lambda v: max(6, min(30, v / 1000))),
                "color": PALETTE[2],
            },
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

    lift = pre_post_promo_comparison(df, promo_periods=promos)
    baseline_df = compute_promo_baseline(df, promo_periods=promos)
    cannibalization = compute_cannibalization_analysis(df, promo_periods=promos)
    cannibalization_agg = (
        (
            cannibalization.groupby("promo_product")["cannibalized_revenue"]
            .sum()
            .rename("cannibalization_revenue")
            .reset_index()
            .rename(columns={"promo_product": "stockcode"})
        )
        if not cannibalization.empty
        else None
    )
    waterfall = compute_incrementality_waterfall(
        baseline_df,
        cannibalization_revenue=cannibalization_agg,
    )
    roi = promo_roi_analysis(df, promo_periods=promos)

    st.divider()
    _render_scorecard(waterfall, roi)

    st.divider()
    _render_incremental_vs_observed(waterfall)

    st.divider()
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Promo Periods", "Lift Analysis", "Waterfall", "ROI", "Timing"]
    )

    with tab1:
        _render_promo_periods(promos)

    with tab2:
        _render_lift_analysis(lift)

    with tab3:
        st.subheader(":material/bar_chart: Promotional Revenue Decomposition (Descriptive)")
        st.caption(
            "⚠️ **Descriptive Only**: Shows observed revenue vs. modeled baseline. "
            "Does NOT imply causal incrementality."
        )

        _render_waterfall(waterfall)

        if not cannibalization.empty:
            top = (
                cannibalization.groupby("promo_product")["cannibalization_index"]
                .mean()
                .rename("avg_cannibalization_index")
                .sort_values(ascending=False)
            )
            st.subheader(":material/swap_horiz: Cannibalization (Cross-Effect)")
            st.caption(
                "Revenue lost by same-category peers during each promo window vs the pre-promo period. "
                "The index is the peer's shortfall relative to its own pre-promo revenue (0 = none, 1 = fully cannibalized)."
            )
            if not top.empty:
                fig = go.Figure(
                    go.Bar(
                        x=[str(k) for k in top.index],
                        y=top.values,
                        marker={"color": PALETTE[4]},
                        text=[f"{v:.0%}" for v in top.values],
                        textposition="outside",
                        hovertemplate="%{x}: %{y:.1%}<extra></extra>",
                    )
                )
                fig.update_layout(
                    xaxis={"tickangle": -45},
                    yaxis={"title": "Avg Cannibalization Index", "tickformat": ".0%"},
                )
                show(fig)
            st.dataframe(
                cannibalization.sort_values("cannibalized_revenue", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

        # Causal Incrementality Layer
        st.subheader(":material/science: Causal Incrementality Engine")
        st.caption(
            "⚠️ **Causal Estimates**: Require parallel trends assumption. "
            "Validate with event study pre-trends before acting."
        )

        if st.button("Run Causal Incrementality Engine"):
            with st.spinner("Estimating causal effects..."):
                df_clean = df.copy()
                df_clean["date"] = pd.to_datetime(df_clean["date"])

                # Run causal engine
                from src.analytics.promo.causal import compute_causal_waterfall

                causal_wf = compute_causal_waterfall(df_clean, promos)

                if causal_wf.empty:
                    st.warning("Insufficient data for causal estimation.")
                else:
                    # Display causal waterfall
                    st.subheader(":material/waterfall: Causal Incrementality Waterfall")
                    wf = causal_wf.iloc[0]

                    cols = st.columns(5)
                    cols[0].metric("Direct Effect", f"€{wf['direct_effect_revenue']:,.0f}")
                    cols[1].metric("Halo Effect", f"€{wf['halo_revenue']:,.0f}")
                    cols[2].metric("Cannibalization", f"-€{wf['cannibalization_revenue']:,.0f}")
                    cols[3].metric("Stockpiling", f"€{wf['stockpiling_revenue']:,.0f}")
                    cols[4].metric("NET INCREMENTAL", f"€{wf['net_incremental_revenue']:,.0f}")

                    # Assumption checklist
                    st.subheader(":material/checklist: Causal Assumptions")
                    st.caption("Confirm before acting on causal estimates:")
                    st.checkbox("Parallel trends (pre-trends p > 0.05)", disabled=True)
                    st.checkbox("No spillover (measured via cross-SKU effects)", disabled=True)
                    st.checkbox("No anticipation (flat pre-trends)", disabled=True)
                    st.checkbox("SUTVA (no interference across SKUs)", disabled=True)

    with tab4:
        _render_roi(roi)

    with tab5:
        timing = promotion_timing_analysis(df, promos)
        _render_timing(timing)

    st.divider()
    st.subheader(":material/radar: Top Insights")
    insights = generate_promotion_insights(waterfall, roi, lift, cannibalization)
    render_insight_cards(insights)

    st.divider()
    st.subheader(":material/task_alt: Ranked Decisions")
    opportunities = generate_promotion_opportunities(waterfall, roi, top_n=10)
    render_opportunity_table(opportunities)


MODE_SPEC: ModeSpec = ModeSpec(
    key="promo",
    label="Promotions",
    icon=":material/local_offer:",
    handler=render,
    requires=("has_promo_flag", "has_price_variation", "sufficient_baskets_500"),
)
