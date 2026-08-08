"""Category Overview tab: manager-facing category analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.category import (
    compute_assortment_efficiency,
    compute_category_growth_matrix,
    compute_category_manager_scorecard,
    compute_category_trend,
    enrich_with_categories,
)
from src.analytics.promo import (
    compute_category_cannibalization,
    compute_category_promo_timeline,
    detect_promotions,
)
from src.analytics.pricing import compute_kvi_score
from src.analytics.scenarios import compute_scenario_grid
from src.ui.plots import PALETTE, empty_state, new_fig, show
from src.ui.registry import ModeSpec

ROLE_COLORS: dict[str, str] = {
    "Destination": PALETTE[0],
    "Routine": PALETTE[2],
    "Seasonal": PALETTE[3],
    "Convenience": PALETTE[4],
    "Unclassified": PALETTE[9],
}


def _treemap(scorecard: pd.DataFrame) -> None:
    st.subheader(":material/dashboard: Category Role Treemap")
    fig = px.treemap(
        scorecard,
        path=["role", "category"],
        values="total_revenue",
        color="role",
        color_discrete_map=ROLE_COLORS,
        custom_data=[
            "category",
            "role",
            "revenue_yoy_growth",
            "basket_penetration",
            "repeat_purchase_rate",
            "sku_share",
            "revenue_share",
            "kvi_count",
        ],
        title="Category revenue by role (rectangles sized by revenue, colored by role)",
    )
    fig.update_traces(
        customdata=scorecard[
            ["category", "role", "revenue_yoy_growth", "basket_penetration", "repeat_purchase_rate", "sku_share", "revenue_share", "kvi_count"]
        ].to_numpy(),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Role: %{customdata[1]}<br>"
            "Revenue: %{value:,.0f}<br>"
            "YoY growth: %{customdata[2]:.1%}<br>"
            "Basket penetration: %{customdata[3]:.1%}<br>"
            "Repeat purchase: %{customdata[4]:.1%}<br>"
            "SKU share: %{customdata[5]:.1%} · Revenue share: %{customdata[6]:.1%}<br>"
            "KVI items: %{customdata[7]:.0f}"
            "<extra></extra>"
        ),
        texttemplate="%{label}",
    )
    fig.update_layout(height=420)
    show(fig)
    st.caption(
        "Outer rectangles = category roles; inner = categories. "
        "Size is total revenue; color encodes role (Destination / Routine / Seasonal / Convenience)."
    )


def _drilldown(df: pd.DataFrame, scorecard: pd.DataFrame) -> None:
    st.subheader(":material/vis_line_chart: Category Drill-down")

    categories = scorecard.sort_values("total_revenue", ascending=False)["category"].tolist()
    selected = st.selectbox("Select category", categories, key="category_drilldown")

    trend = compute_category_trend(df)
    cat_trend = trend[trend["category"] == selected]
    cat_row = scorecard[scorecard["category"] == selected].iloc[0]

    kvi = compute_kvi_score(df)
    cat_kvi = (
        kvi[kvi["category"] == selected]
        .sort_values("kvi_score", ascending=False)
        .head(10)
        if not kvi.empty
        else kvi
    )

    if cat_trend.empty:
        show(empty_state(f"No trend data for {selected}"))
        return

    c1, c2 = st.columns(2)

    with c1:
        fig = new_fig(height=300)
        fig.add_trace(
            {
                "x": cat_trend["period"],
                "y": cat_trend["revenue"],
                "name": "Revenue",
                "type": "scatter",
                "mode": "lines+markers",
            }
        )
        fig.update_layout(title=f"{selected} — revenue by period", margin={"l": 50, "r": 15, "t": 45, "b": 40})
        show(fig)

    with c2:
        fig = new_fig(height=300)
        fig.add_trace(
            {
                "x": cat_trend["period"],
                "y": cat_trend["basket_penetration"],
                "name": "Basket penetration",
                "type": "scatter",
                "mode": "lines+markers",
            }
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(title=f"{selected} — basket penetration by period", margin={"l": 50, "r": 15, "t": 45, "b": 40})
        show(fig)

    st.markdown("**Assortment efficiency** (full matrix lands in Phase 2 — item 3)")
    eff_agg = scorecard[["sku_share", "revenue_share"]].mean()
    eff_row = scorecard.loc[cat_row.name, ["sku_share", "revenue_share"]]
    eff_ratio = float(eff_row["revenue_share"] / eff_row["sku_share"]) if eff_row["sku_share"] > 0 else float("nan")
    st.write(
        f"{selected} has {eff_row['sku_share']:.1%} of SKUs and {eff_row['revenue_share']:.1%} of revenue "
        f"(category-median ratios: {eff_agg['sku_share']:.1%} SKU / {eff_agg['revenue_share']:.1%} revenue). "
        + (
            "Revenue outperforms its SKU count — efficiently shopped."
            if pd.notna(eff_ratio) and eff_ratio > 1
            else "Its SKU count is heavy relative to revenue — an assortment-rationalization candidate."
            if pd.notna(eff_ratio)
            else "Not enough data to assess assortment efficiency."
        )
    )

    if not cat_kvi.empty:
        st.markdown(f"**Top KVI items in {selected}**")
        st.dataframe(
            cat_kvi[["stockcode", "kvi_score", "category"]].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption(f"No KVI items computed for {selected}.")


QUADRANT_COLORS: dict[str, str] = {
    "star": PALETTE[2],
    "cash_cow": PALETTE[0],
    "question_mark": PALETTE[1],
    "dog": PALETTE[3],
}


def _assortment_efficiency(df: pd.DataFrame) -> None:
    st.subheader(":material/scatter_plot: Assortment Efficiency")

    eff = compute_assortment_efficiency(df)
    if eff.empty:
        show(empty_state("No category data for assortment efficiency"))
        return

    eff = eff.sort_values("total_revenue", ascending=False)
    label_colors = {
        "efficient": PALETTE[2],
        "balanced": PALETTE[0],
        "under_efficient": PALETTE[3],
    }

    fig = px.scatter(
        eff,
        x="sku_share",
        y="revenue_share",
        color="efficiency_label",
        color_discrete_map=label_colors,
        size="total_revenue",
        size_max=40,
        hover_data=["category", "role", "efficiency_index"],
        text="category",
    )
    # parity line: revenue_share == sku_share (index = 1)
    line_max = max(float(eff["sku_share"].max()), float(eff["revenue_share"].max())) * 1.15
    fig.add_trace(
        {
            "x": [0, line_max],
            "y": [0, line_max],
            "mode": "lines",
            "line": {"color": "#888888", "dash": "dash"},
            "name": "Parity (index = 1)",
            "hovertemplate": "<extra></extra>",
        }
    )
    fig.update_layout(
        title="SKU share vs revenue share per category",
        height=420,
        margin={"l": 55, "r": 20, "t": 45, "b": 50},
    )
    fig.update_xaxes(title="SKU share", tickformat=".0%")
    fig.update_yaxes(title="Revenue share", tickformat=".0%")
    fig.update_traces(
        customdata=eff[["category", "role", "efficiency_index"]].to_numpy(),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Role: %{customdata[1]}<br>"
            "SKU share: %{x:.1%}<br>"
            "Revenue share: %{y:.1%}<br>"
            "Efficiency index: %{customdata[2]:.2f}"
            "<extra></extra>"
        ),
        texttemplate="%{label}",
        textposition="top center",
    )
    show(fig)
    st.caption(
        "Bubble size = total revenue; color = efficiency. Points above the parity line (index > 1) "
        "turn SKUs into revenue more efficiently than their assortment weight; below it they are "
        "SKU-heavy for the revenue they produce."
    )

    st.dataframe(
        eff.sort_values("efficiency_index", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        f"**Read**: {len(eff[eff['efficiency_label'] == 'efficient'])} efficient, "
        f"{len(eff[eff['efficiency_label'] == 'balanced'])} balanced, "
        f"{len(eff[eff['efficiency_label'] == 'under_efficient'])} under-efficient."
    )


def _growth_matrix(df: pd.DataFrame) -> None:
    st.subheader(":material/timeline: Category Growth Matrix (internal BCG)")

    matrix = compute_category_growth_matrix(df)
    if matrix.empty:
        show(empty_state("No growth data for the matrix"))
        return

    share_med = float(matrix["revenue_share"].median())
    growth_med = float(matrix["growth_pct"].median())

    quadrant_labels = {
        "star": "Star (high share · high growth)",
        "cash_cow": "Cash Cow (high share · low growth)",
        "question_mark": "Question Mark (low share · high growth)",
        "dog": "Dog (low share · low growth)",
    }
    matrix["quadrant_label"] = matrix["quadrant"].map(quadrant_labels)

    fig = px.scatter(
        matrix,
        x="revenue_share",
        y="growth_pct",
        color="quadrant_label",
        color_discrete_map={
            quadrant_labels["star"]: QUADRANT_COLORS["star"],
            quadrant_labels["cash_cow"]: QUADRANT_COLORS["cash_cow"],
            quadrant_labels["question_mark"]: QUADRANT_COLORS["question_mark"],
            quadrant_labels["dog"]: QUADRANT_COLORS["dog"],
        },
        size="total_revenue",
        size_max=45,
        custom_data=["category", "role", "total_revenue"],
        text=matrix["category"],
    )
    # Median reference lines
    fig.add_vline(x=share_med, line_dash="dash", line_color="#888888")
    fig.add_hline(y=growth_med, line_dash="dash", line_color="#888888")

    fig.update_layout(
        title="Category growth vs revenue share",
        height=460,
        margin={"l": 55, "r": 20, "t": 45, "b": 50},
    )
    fig.update_xaxes(title="Revenue share", tickformat=".0%")
    fig.update_yaxes(title="Revenue growth (YoY)")
    fig.update_traces(
        customdata=matrix[["category", "role", "total_revenue"]].to_numpy(),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Role: %{customdata[1]}<br>"
            "Revenue share: %{x:.2%}<br>"
            "Growth: %{y:+.1f}%<br>"
            "Total revenue: %{customdata[2]:,.0f}"
            "<extra></extra>"
        ),
        textposition="top center",
    )
    show(fig)
    st.caption(
        "Internal BCG: x = revenue share, y = YoY revenue growth, bubble size = total revenue. "
        "Dashed lines = category medians. "
        "Stars (invest), Cash Cows (harvest), Question Marks (evaluate/turnaround), Dogs (rationalize)."
    )

    table_cols = [c for c in ("category", "role", "quadrant", "revenue_share", "growth_pct", "total_revenue") if c in matrix.columns]
    st.dataframe(
        matrix[table_cols].sort_values("total_revenue", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def _scenario_grid(df: pd.DataFrame) -> None:
    st.subheader(":material/query_stats: Scenario Grid")
    with st.expander("Parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        n_weeks = int(c1.number_input("Anchor weeks", 4, 52, 12))
        projection_weeks = int(c2.number_input("Projection weeks", 4, 52, 13))
        uplift = float(c3.number_input("Optimistic uplift (%/wk)", -5.0, 5.0, 0.10, step=0.05))

    grid = compute_scenario_grid(
        df,
        n_weeks=n_weeks,
        projection_weeks=projection_weeks,
        optimistic_uplift=uplift,
    )
    if grid.empty:
        show(empty_state("Insufficient weekly history for scenario projection"))
        return

    # Grouped bars: projected revenue by scenario for each category
    pivot = grid.pivot(index="category", columns="scenario", values="projected_revenue").reindex(
        columns=["pessimistic", "neutral", "optimistic"]
    )
    fig = go.Figure()
    scenario_colors = {
        "pessimistic": PALETTE[6],
        "neutral": PALETTE[0],
        "optimistic": PALETTE[2],
    }
    for scenario in pivot.columns:
        fig.add_trace(
            go.Bar(
                x=pivot.index,
                y=pivot[scenario],
                name=scenario.capitalize(),
                marker={"color": scenario_colors[scenario]},
            )
        )
    fig.update_layout(barmode="group", xaxis={"title": "Category"}, yaxis={"title": f"Projected revenue ({projection_weeks}wk)"})
    show(fig)
    st.caption(
        "Pessimistic = 1-sigma below trend, Neutral = historical weekly growth compounded, "
        "Optimistic = trend + manager uplift. Infeasible cells (outside the ±15%/wk planning "
        "band or more than 2x growth) are flagged below."
    )

    st.dataframe(
        grid[
            ["category", "scenario", "growth_lever", "weekly_growth_pct", "projected_revenue", "revenue_change_pct", "feasible", "guard_note"]
        ].sort_values(["category", "scenario"]),
        use_container_width=True,
        hide_index=True,
    )

    infeasible = grid[~grid["feasible"]]
    if not infeasible.empty:
        st.warning(
            f"{len(infeasible)} infeasible scenario cell(s): "
            + "; ".join(f"{r.category} ({r.scenario}): {r.guard_note}" for _, r in infeasible.head(3).iterrows())
            + (" …" if len(infeasible) > 3 else "")
        )


def _category_cannibalization(df: pd.DataFrame) -> None:
    st.subheader(":material/local_fire_department: Category Cannibalization")
    promos = detect_promotions(df)
    if promos.empty:
        st.caption("No promotional periods detected with default parameters.")
        return

    cann = compute_category_cannibalization(df, promos)
    if cann.empty:
        st.caption("No cross-category cannibalization detected.")
        return

    # Heatmap: promo_category (rows) -> peer_category (cols), intensity = index
    pivot = cann.pivot_table(
        index="promo_category",
        columns="peer_category",
        values="cannibalization_index",
        fill_value=0,
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="YlOrRd",
            zmin=0,
            zmax=1,
            colorbar={"title": "Index"},
            text=[[f"{v:.0%}" if v > 0 else "" for v in row] for row in pivot.to_numpy()],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Promo in %{y} cannibalizes %{x}: %{z:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis={"title": "Peer category (loses revenue)", "tickangle": -45},
        yaxis={"title": "Promo category"},
        height=max(300, len(pivot) * 32 + 80),
    )
    show(fig)
    st.caption(
        "Rows = category running promos; columns = categories that lose revenue during those "
        "promo windows vs the prior equal-length window. Index = cannibalized / base revenue "
        "(0-1). Diagonal excluded. High cells mean the promo shifts share out of the column "
        "category - schedule or feature them apart."
    )

    table = cann.sort_values("cannibalized_revenue", ascending=False)
    st.dataframe(table, use_container_width=True, hide_index=True)


def _promo_timeline(df: pd.DataFrame) -> None:
    st.subheader(":material/local_offer: Category Promo Timeline")

    promos = detect_promotions(df)
    if promos.empty:
        st.caption("No promotional periods detected with default parameters.")
        return

    tl = compute_category_promo_timeline(df, promos)
    if tl.empty:
        st.caption("No promo activity across the timeline.")
        return

    categories = sorted(tl["category"].unique())
    selected = st.multiselect("Categories", categories, default=categories[:3], key="promo_timeline_cats")

    data = tl[tl["category"].isin(selected)]
    if data.empty:
        show(empty_state("Select at least one category"))
        return

    fig = px.bar(
        data,
        x="period",
        y=["promo_revenue", "non_promo_revenue"],
        barmode="stack",
        color_discrete_map={"promo_revenue": PALETTE[3], "non_promo_revenue": PALETTE[0]},
        facet_col="category",
        facet_col_wrap=3,
        labels={"value": "Revenue", "period": "Week", "variable": ""},
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(
        title="Weekly revenue during promo vs non-promo (stacked, per category)",
        height=300 * min(2, (len(selected) + 2) // 3),
        margin={"l": 55, "r": 20, "t": 45, "b": 50},
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(tickformat=".2s")
    show(fig)

    st.caption(
        "Promo weeks are detected via per-SKU price drops (15%+ vs 90th-percentile baseline). "
        "Red = in-promo revenue, blue = non-promo. Hover shows active promo SKU count and discount depth."
    )

    hover = tl[tl["category"].isin(selected) & (tl["n_promos"] > 0)][
        ["category", "period", "promo_revenue", "n_promos", "avg_discount_pct"]
    ].sort_values(["category", "period"])
    if not hover.empty:
        st.markdown("**Promo activity detail**")
        st.dataframe(
            hover.assign(avg_discount_pct=hover["avg_discount_pct"].round(2)),
            use_container_width=True,
            hide_index=True,
        )


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/category: Category Overview")

    df, category_inferred = enrich_with_categories(df)
    if category_inferred:
        st.info("A `category` column was not supplied; categories were inferred from product descriptions (TF-IDF + KMeans).")

    scorecard = compute_category_manager_scorecard(df)
    if scorecard.empty:
        show(empty_state("No category data for scorecard"))
        return

    st.subheader(":material/table_rows: Category Scorecard")
    st.dataframe(
        scorecard.sort_values("total_revenue", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "One row per category. Role from CATEGORY_ROLES; revenue_yoy_growth uses annual windows "
        "(period-on-period fallback when the data spans < 2 years). KVI count/share from KVI scores."
    )

    _treemap(scorecard)
    _assortment_efficiency(df)
    _growth_matrix(df)
    _scenario_grid(df)
    _promo_timeline(df)
    _category_cannibalization(df)
    _drilldown(df, scorecard)


MODE_SPEC: ModeSpec = ModeSpec(
    key="category",
    label="Category Overview",
    icon=":material/category:",
    handler=render,
)
