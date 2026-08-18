"""Assortment Optimization tab."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.assortment import (
    build_solution_table,
    compare_assortment_scenarios,
    optimize_assortment_heuristic,
)
from src.analytics.performance import compute_product_metrics
from src.ui.plots import PALETTE, empty_state, show
from src.ui.registry import ModeSpec


def _render_coverage_gauge(metrics: dict) -> None:
    st.subheader(":material/speed: Assortment Coverage")
    coverage = metrics.get("coverage", 0.0)
    recovery = metrics.get("recovery_rate", 0.0)

    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=coverage * 100,
            title={"text": "Revenue Coverage %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": PALETTE[0]},
                "steps": [
                    {"range": [0, 50], "color": PALETTE[4]},
                    {"range": [50, 80], "color": PALETTE[2]},
                    {"range": [80, 100], "color": PALETTE[0]},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 80,
                },
            },
        )
    )
    fig.update_layout(height=250)
    show(fig)

    c1, c2 = st.columns(2)
    c1.metric("Recovery Rate", f"{recovery:.1%}")
    c2.metric("Expected Revenue", f"${metrics.get('expected_revenue', 0):,.0f}")


def _render_revenue_waterfall(kept: list[str], metrics: dict, revenue: pd.Series) -> None:
    st.subheader(":material/waterfall: Revenue Composition")
    kept_rev = metrics.get("kept_revenue", 0.0)
    recovered = metrics.get("recovered_revenue", 0.0)
    lost = metrics.get("lost_revenue", 0.0)
    total = kept_rev + recovered + lost

    fig = go.Figure(
        go.Waterfall(
            name="Revenue",
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Total Market", "Kept", "Recovered", "Lost"],
            textposition="outside",
            y=[total, kept_rev, recovered, -lost],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": PALETTE[0]}},
            decreasing={"marker": {"color": PALETTE[4]}},
            totals={"marker": {"color": PALETTE[2]}},
        )
    )
    fig.update_layout(yaxis={"title": "Revenue ($)"}, height=350)
    show(fig)
    st.caption(
        "Total market = kept + recovered + lost. Recovered = demand from delisted SKUs captured by kept substitutes."
    )


def _render_scenario_comparison(scenarios: pd.DataFrame) -> None:
    st.subheader(":material/compare_arrows: Scenario Comparison")
    if scenarios.empty:
        show(empty_state("No scenarios to compare"))
        return

    # Expected revenue bar chart
    fig = px.bar(
        scenarios.sort_values("expected_revenue", ascending=True).tail(10),
        x="expected_revenue",
        y="scenario_id",
        color="method",
        color_discrete_map={"greedy": PALETTE[0], "random": PALETTE[2], "milp": PALETTE[4]},
        orientation="h",
        hover_data=["coverage", "recovery_rate", "n_skus"],
    )
    fig.update_layout(xaxis={"title": "Expected Revenue ($)"}, yaxis={"title": "Scenario ID"})
    show(fig)

    # Coverage vs Recovery scatter
    fig2 = px.scatter(
        scenarios,
        x="coverage",
        y="recovery_rate",
        size="n_skus",
        color="method",
        color_discrete_map={"greedy": PALETTE[0], "random": PALETTE[2], "milp": PALETTE[4]},
        hover_data=["scenario_id", "kept_revenue", "expected_revenue"],
    )
    fig2.add_vline(x=0.8, line_dash="dash", line_color="gray", annotation_text="Target 80%")
    fig2.update_layout(xaxis={"title": "Coverage"}, yaxis={"title": "Recovery Rate"})
    show(fig2)
    st.caption("Bubble size = SKU count. Target: high coverage + high recovery.")


def _render_selected_assortment_table(kept: list[str], revenue: pd.Series) -> None:
    st.subheader(":material/table_rows: Selected Assortment")
    table = build_solution_table(kept, revenue)
    if table.empty:
        show(empty_state("No SKUs selected"))
        return

    fig = px.treemap(
        table.head(30),
        path=["stockcode"],
        values="revenue",
        hover_data=["rank"],
        color="revenue",
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=400)
    show(fig)
    st.caption("Treemap: top 30 selected SKUs by revenue. Click to drill down.")

    st.dataframe(
        table.sort_values("revenue", ascending=False), use_container_width=True, hide_index=True
    )


def _render_category_coverage(kept: list[str], df: pd.DataFrame) -> None:
    st.subheader(":material/category: Category Coverage")
    if "category" not in df.columns:
        st.info("No category column in data")
        return

    cat_of = dict(df.drop_duplicates("stockcode").set_index("stockcode")["category"])
    kept_cats = {cat_of[p] for p in kept if p in cat_of}
    all_cats = set(cat_of.values())

    kept_rev: dict[str, float] = {}
    for p in kept:
        if p in cat_of:
            kept_rev[cat_of[p]] = (
                kept_rev.get(cat_of[p], 0.0)
                + (df[df["stockcode"] == p]["price"] * df[df["stockcode"] == p]["quantity"]).sum()
            )

    cat_df = pd.DataFrame(
        {
            "category": list(all_cats),
            "covered": [1 if c in kept_cats else 0 for c in all_cats],
            "kept_revenue": [kept_rev.get(c, 0.0) for c in all_cats],
        }
    ).sort_values("covered", ascending=False)

    fig = px.bar(
        cat_df,
        x="category",
        y="kept_revenue",
        color=cat_df["covered"].map({1: "Covered", 0: "Not Covered"}),
        color_discrete_map={"Covered": PALETTE[0], "Not Covered": PALETTE[4]},
    )
    fig.update_layout(xaxis={"tickangle": -45}, yaxis={"title": "Kept Revenue ($)"})
    show(fig)
    st.caption(f"Categories covered: {len(kept_cats)} / {len(all_cats)}")


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/inventory_2: Assortment Optimization")

    with st.expander("Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        max_skus = c1.number_input("Max SKUs", 10, 500, 50)
        min_coverage = c2.number_input("Min Coverage", 0.1, 1.0, 0.8, 0.05)
        objective = c3.selectbox("Objective", ["revenue", "margin"])

    selected, metrics = optimize_assortment_heuristic(
        df, max_skus=max_skus, min_coverage=min_coverage, objective=objective
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SKUs Selected", len(selected))
    col2.metric("Coverage", f"{metrics.get('coverage', 0):.1%}")
    col3.metric("Kept Revenue", f"${metrics.get('kept_revenue', 0):,.0f}")
    col4.metric("Recovery Rate", f"{metrics.get('recovery_rate', 0):.1%}")

    st.divider()
    revenue = compute_product_metrics(df).set_index("stockcode")["revenue"]

    _render_coverage_gauge(metrics)
    st.divider()
    _render_revenue_waterfall(selected, metrics, revenue)
    st.divider()
    _render_category_coverage(selected, df)
    st.divider()

    scenarios = compare_assortment_scenarios(df, [])
    _render_scenario_comparison(scenarios)
    st.divider()

    _render_selected_assortment_table(selected, revenue)


MODE_SPEC: ModeSpec = ModeSpec(
    key="assortment",
    label="Assortment",
    icon=":material/inventory_2:",
    handler=render,
    requires=("has_category", "sufficient_skus_20", "sufficient_baskets_500"),
)
