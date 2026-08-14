"""Cohort Analysis tab."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.cohort import (
    compute_cohort_ltv_curve,
    compute_cohorts,
    compute_role_retention,
    year_over_year_comparison,
)
from src.ui.plots import PALETTE, empty_state, new_fig, show
from src.ui.registry import ModeSpec


def _render_retention_heatmap(cohort_table: pd.DataFrame) -> None:
    st.subheader(":material/table_chart: Retention Rate Heatmap")
    if cohort_table.empty:
        show(empty_state("No cohort data"))
        return

    pivot = cohort_table.pivot(index="cohort", columns="period_index", values="retention_rate")
    pivot = pivot * 100  # percentage

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            colorscale="RdYlGn",
            zmin=0,
            zmax=100,
            colorbar={"title": "Retention %"},
            text=[[f"{v:.1f}%" if pd.notna(v) else "" for v in row] for row in pivot.to_numpy()],
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )
    fig.update_layout(
        xaxis={"title": "Period index", "tickangle": 0},
        yaxis={"title": "Cohort", "tickangle": 0},
        height=max(350, 25 * len(pivot)),
    )
    show(fig)
    st.caption(
        "Rows = acquisition cohort (first period), columns = periods since first purchase. Green = high retention."
    )


def _render_revenue_heatmap(df: pd.DataFrame, cohort_period: str) -> None:
    st.subheader(":material/attach_money: Cumulative Revenue per Customer Heatmap")
    ltv = compute_cohort_ltv_curve(df, cohort_period=cohort_period)
    if ltv.empty:
        show(empty_state("No LTV data"))
        return

    pivot = ltv.pivot(index="cohort", columns="period_index", values="ltv_per_customer")

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            colorscale="Blues",
            zmin=0,
            colorbar={"title": "Cumulative $"},
            text=[[f"${v:.0f}" if pd.notna(v) else "" for v in r] for r in pivot.to_numpy()],
            texttemplate="%{text}",
            textfont={"size": 10},
        ),
    )
    fig.update_layout(
        xaxis={"title": "Period index", "tickangle": 0},
        yaxis={"title": "Cohort", "tickangle": 0},
        height=max(350, 25 * len(pivot)),
    )
    show(fig)
    st.caption("Cumulative revenue per acquired customer. Darker = higher lifetime value to date.")


def _render_aov_curves(df: pd.DataFrame, cohort_period: str) -> None:
    st.subheader(":material/show_chart: Cumulative Revenue per Retained Customer")
    ltv = compute_cohort_ltv_curve(df, cohort_period=cohort_period)
    if ltv.empty:
        show(empty_state("No LTV data"))
        return

    # cumulative_revenue / retained_customers at each period
    from src.analytics.cohort import compute_cohorts

    cohort_table = compute_cohorts(df, cohort_period=cohort_period)
    if cohort_table.empty:
        show(empty_state("No cohort data"))
        return

    # Merge LTV with retained counts
    merged = ltv.merge(
        cohort_table[["cohort", "period_index", "retained", "cohort_size"]],
        on=["cohort", "period_index"],
        how="left",
    )
    merged["rev_per_retained"] = merged["cumulative_revenue"] / merged["retained"].replace(0, pd.NA)

    fig = new_fig()
    cohorts = sorted(merged["cohort"].unique())
    for i, cohort in enumerate(cohorts):
        sub = merged[merged["cohort"] == cohort].sort_values("period_index")
        fig.add_trace(
            go.Scatter(
                x=sub["period_index"],
                y=sub["rev_per_retained"],
                mode="lines+markers",
                name=str(cohort),
                line={"width": 1.5, "color": PALETTE[i % len(PALETTE)]},
                marker={"size": 4},
            )
        )
    fig.update_layout(
        xaxis={"title": "Period index"},
        yaxis={"title": "Cumulative revenue per retained customer ($)"},
        hovermode="x unified",
    )
    show(fig)
    st.caption(
        "Cumulative revenue per retained customer, per cohort. Rising = the customers "
        "who keep buying spend more over time; falling = discounting or mix shift. "
        "This is not average order value — it is a lifetime-value-to-date view."
    )


def _render_role_retention_curves(
    df: pd.DataFrame, cohort_period: str, min_role_customers: int
) -> None:
    st.subheader(":material/group_work: Retention by Category Role")
    overview = compute_role_retention(
        df, cohort_period=cohort_period, min_role_customers=min_role_customers
    )
    if overview.empty:
        show(empty_state("No role retention data"))
        return

    # Weighted-average retention curve per role (pool retain and size across cohorts)
    agg = overview.groupby(["role", "period_index"], as_index=False).agg(
        retained=("retained", "sum"), cohort_size=("cohort_size", "sum")
    )
    agg["retention_rate"] = agg["retained"] / agg["cohort_size"]

    role_order = sorted(overview["role"].unique().tolist())
    fig = new_fig()
    for i, role in enumerate(role_order):
        sub = agg[agg["role"] == role].sort_values("period_index")
        fig.add_trace(
            go.Scatter(
                x=sub["period_index"],
                y=sub["retention_rate"] * 100,
                mode="lines+markers",
                name=role,
                line={"width": 2, "color": PALETTE[i % len(PALETTE)]},
                marker={"size": 4},
                hovertemplate="%{y:.1f}% retained<extra>" + role + "</extra>",
            )
        )
    fig.update_layout(
        xaxis={"title": "Period index since acquisition"},
        yaxis={"title": "Retention (%)", "range": [0, 105]},
        hovermode="x unified",
    )
    show(fig)

    # Cohort sizes per role
    sizes = (
        overview.groupby("cohort", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    r: int(g.loc[g["role"] == r, "cohort_size"].max())
                    if (g["role"] == r).any()
                    else 0
                    for r in role_order
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    st.dataframe(sizes, use_container_width=True, hide_index=True)
    st.caption(
        "Customers per acquisition cohort by the role of their first basket's dominant category. Curves show weighted-average retention per role."
    )


def _render_yoy_bars(df: pd.DataFrame) -> None:
    st.subheader(":material/calendar_month: Year-over-Year Revenue Growth")
    yoy = year_over_year_comparison(df)
    if yoy.empty or "revenue_yoy_growth" not in yoy.columns:
        show(empty_state("Insufficient years for YoY"))
        return

    fig = new_fig()
    fig.add_trace(
        go.Bar(
            x=yoy["week"].astype(str),
            y=yoy["revenue"],
            name="Current year",
            marker={"color": PALETTE[0]},
        )
    )
    if "prior_revenue" in yoy.columns:
        fig.add_trace(
            go.Bar(
                x=yoy["week"].astype(str),
                y=yoy["prior_revenue"],
                name="Prior year",
                marker={"color": PALETTE[1]},
            )
        )
    fig.update_layout(barmode="group", xaxis={"title": "ISO Week"}, yaxis={"title": "Revenue ($)"})
    show(fig)

    # Growth rate line
    fig2 = new_fig()
    fig2.add_trace(
        go.Scatter(
            x=yoy["week"].astype(str),
            y=yoy["revenue_yoy_growth"],
            mode="lines+markers",
            name="YoY % growth",
            line={"width": 2, "color": PALETTE[2]},
            marker={"size": 4},
        )
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(xaxis={"title": "ISO Week"}, yaxis={"title": "Revenue YoY Growth (%)"})
    show(fig2)
    st.caption(
        "Bars = weekly revenue by year; line = YoY growth rate. Positive = growth vs prior year."
    )


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/timeline: Cohort Retention")

    with st.expander("Parameters", expanded=True):
        c1, c2 = st.columns(2)
        period = c1.selectbox(
            "Cohort period",
            ["W", "M"],
            index=0,
            format_func=lambda x: "Weekly" if x == "W" else "Monthly",
        )
        min_role_customers = int(c2.number_input("Min customers per role cohort", 1, 200, 5))

    cohort_table = compute_cohorts(df, cohort_period=period)

    if cohort_table.empty:
        st.warning("No cohort data available.")
        return

    st.divider()
    _render_retention_heatmap(cohort_table)

    st.divider()
    _render_role_retention_curves(df, cohort_period=period, min_role_customers=min_role_customers)

    st.divider()
    _render_revenue_heatmap(df, cohort_period=period)

    st.divider()
    _render_aov_curves(df, cohort_period=period)

    st.divider()
    _render_yoy_bars(df)

    st.divider()
    st.subheader(":material/table_rows: Raw Cohort Data")
    st.dataframe(cohort_table, use_container_width=True, hide_index=True)


MODE_SPEC: ModeSpec = ModeSpec(
    key="cohorts",
    label="Cohorts",
    icon=":material/timeline:",
    handler=render,
)
