"""Product Performance tab."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.category import compute_category_roles
from src.analytics.performance import (
    abc_analysis,
    compute_repeat_rate,
    compute_sku_rationalization_df,
    compute_velocity,
    product_lifecycle_stage,
    xyz_analysis,
)
from src.ui.features import get_product_metrics
from src.ui.plots import PALETTE, empty_state, new_fig, show
from src.ui.registry import ModeSpec


def _render_abc_pareto(perf: pd.DataFrame) -> None:
    st.subheader(":material/leaderboard: ABC Revenue Pareto")
    abc = abc_analysis(perf)
    if abc.empty:
        show(empty_state("No ABC data"))
        return

    abc_sorted = abc.sort_values("revenue", ascending=False).reset_index(drop=True)
    abc_sorted["cumulative_pct"] = abc_sorted["revenue"].cumsum() / abc_sorted["revenue"].sum() * 100

    fig = new_fig()
    # Bar: revenue
    fig.add_trace(
        go.Bar(
            x=abc_sorted["stockcode"],
            y=abc_sorted["revenue"],
            name="Revenue",
            marker={"color": abc_sorted["abc_class"].map({"A": PALETTE[0], "B": PALETTE[2], "C": PALETTE[4]})},
            hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
        )
    )
    # Line: cumulative %
    fig.add_trace(
        go.Scatter(
            x=abc_sorted["stockcode"],
            y=abc_sorted["cumulative_pct"],
            name="Cumulative %",
            mode="lines+markers",
            yaxis="y2",
            line={"color": PALETTE[1], "width": 2},
            marker={"size": 4},
        )
    )
    fig.add_hline(y=70, line_dash="dash", line_color="gray", annotation_text="A boundary (70%)", yref="y2")
    fig.add_hline(y=90, line_dash="dash", line_color="gray", annotation_text="B boundary (90%)", yref="y2")
    fig.update_layout(
        yaxis={"title": "Revenue ($)"},
        yaxis2={"title": "Cumulative %", "overlaying": "y", "side": "right", "range": [0, 105]},
        xaxis={"tickangle": -45},
        hovermode="x unified",
    )
    show(fig)
    st.caption("Pareto chart: A-class (green) = top 70% revenue, B (blue) = 70-90%, C (gray) = rest.")


def _render_xyz_volatility(perf: pd.DataFrame) -> None:
    st.subheader(":material/signal_cellular_alt: XYZ Demand Volatility")
    xyz = xyz_analysis(perf)
    if xyz.empty:
        show(empty_state("No XYZ data"))
        return

    fig = px.scatter(
        xyz.sort_values("revenue", ascending=False).head(50),
        x="revenue",
        y="cv",
        color="xyz_class",
        color_discrete_map={"X": PALETTE[0], "Y": PALETTE[2], "Z": PALETTE[4]},
        hover_data=["stockcode"],
        log_x=True,
    )
    fig.add_hline(y=0.10, line_dash="dash", line_color="gray", annotation_text="X boundary (10% CV)")
    fig.add_hline(y=0.25, line_dash="dash", line_color="gray", annotation_text="Y boundary (25% CV)")
    fig.update_layout(xaxis={"title": "Total Revenue (log)"}, yaxis={"title": "Coefficient of Variation"})
    show(fig)
    st.caption("X = stable demand (CV ≤ 10%), Y = moderate (10-25%), Z = erratic (>25%). Circle size = revenue.")


def _render_lifecycle_scatter(perf: pd.DataFrame) -> None:
    st.subheader(":material/trending_up: Lifecycle Stage (Growth vs Revenue)")
    lifecycle = product_lifecycle_stage(perf)
    if lifecycle.empty:
        show(empty_state("No lifecycle data"))
        return

    fig = px.scatter(
        lifecycle,
        x="prior_revenue",
        y="growth_pct",
        color="stage",
        color_discrete_map={"growth": PALETTE[0], "mature": PALETTE[2], "decline": PALETTE[4]},
        hover_data=["stockcode"],
        log_x=True,
    )
    fig.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="Growth threshold (+25%)")
    fig.add_hline(y=-25, line_dash="dash", line_color="gray", annotation_text="Decline threshold (-25%)")
    fig.update_layout(xaxis={"title": "Prior Period Revenue (log)"}, yaxis={"title": "Growth % (Recent vs Prior)"})
    show(fig)
    st.caption("Products in growth quadrant (top-right) are expanding; decline (bottom-left) need attention.")


def _render_velocity_repeat(full: pd.DataFrame) -> None:
    st.subheader(":material/speed: Velocity vs Repeat Rate")
    velocity = full[["stockcode", "velocity"]].dropna()
    repeat = full[["stockcode", "repeat_rate"]].dropna()
    if velocity.empty or repeat.empty:
        show(empty_state("No velocity/repeat data"))
        return

    merged = velocity.merge(repeat, on="stockcode", how="inner")

    fig = px.scatter(
        merged,
        x="velocity",
        y="repeat_rate",
        size="revenue",
        color=merged["revenue"].apply(lambda r: "High" if r > merged["revenue"].median() else "Low"),
        color_discrete_map={"High": PALETTE[0], "Low": PALETTE[4]},
        hover_data=["stockcode"],
        log_x=True,
    )
    fig.update_layout(xaxis={"title": "Velocity (units/active day)"}, yaxis={"title": "Repeat Purchase Rate"})
    show(fig)
    st.caption("High velocity + high repeat (top-right) = sticky, fast-moving products. Low both = slow movers with low loyalty.")


def _render_sku_rationalization(perf: pd.DataFrame) -> None:
    st.subheader(":material/check_circle: SKU Rationalization Actions")
    rational = compute_sku_rationalization_df(perf)
    if rational.empty:
        show(empty_state("No rationalization data"))
        return

    action_counts = rational["action"].value_counts()
    fig = go.Figure(
        data=[
            go.Pie(
                labels=action_counts.index,
                values=action_counts.values,
                hole=0.4,
                marker={"colors": [PALETTE[0], PALETTE[2], PALETTE[4], PALETTE[1]]},
            )
        ]
    )
    fig.update_layout(showlegend=True)
    show(fig)

    st.caption("keep = A + X/Y; delist_candidate = C + Z; review = others. Use filters below to drill down.")
    st.dataframe(
        rational.sort_values(["abc_class", "xyz_class", "revenue"], ascending=[True, True, False]),
        use_container_width=True,
        hide_index=True,
    )


def _render_category_roles(df: pd.DataFrame) -> None:
    st.subheader(":material/category: Category Role Classification")
    if "category" not in df.columns:
        show(empty_state("No category column in data"))
        return

    try:
        roles = compute_category_roles(df)
    except Exception as e:
        show(empty_state(f"Error computing category roles: {e}"))
        return

    if roles.empty:
        show(empty_state("No category roles computed"))
        return

    # Disclosure banner
    source = roles["category_source"].iloc[0] if "category_source" in roles.columns and not roles.empty else "unknown"
    source_labels = {
        "sample_themes": "📋 Categories from sample-data themes (Coffee/Dairy, Snacks/Beverages, etc.)",
        "inferred_nlp": "🤖 Categories inferred from product descriptions via TF-IDF + KMeans",
        "provided": "📁 Categories provided in source data",
    }
    st.info(source_labels.get(source, f"ℹ️ Category source: {source}"))

    # Treemap: categories sized by revenue, colored by role
    # Need to get revenue per category
    revenue_per_cat = df.copy()
    revenue_per_cat["revenue"] = revenue_per_cat["price"] * revenue_per_cat["quantity"]
    cat_revenue = revenue_per_cat.groupby("category")["revenue"].sum().reset_index()
    roles_with_rev = roles.merge(cat_revenue, on="category", how="left")

    role_colors = {
        "Destination": PALETTE[0],
        "Routine": PALETTE[2],
        "Seasonal": PALETTE[3],
        "Convenience": PALETTE[4],
    }
    colors = roles_with_rev["role"].map(role_colors)

    fig = px.treemap(
        roles_with_rev,
        path=["category"],
        values="revenue",
        color="role",
        color_discrete_map=role_colors,
        hover_data=["trip_generation_rate", "demand_cv", "seasonality_amplitude", "attachment_rate", "category_source"],
    )
    fig.update_layout(height=400)
    show(fig)
    st.caption("Categories sized by revenue, colored by role. Destination = high trip generation + stable demand; Seasonal = high seasonal amplitude; Convenience = low trip gen + high attachment to Destination; Routine = stable, frequent, not strongly seasonal/destination.")

    # Signal table
    st.subheader(":material/table_rows: Category Role Signals")
    display_cols = ["category", "role", "trip_generation_rate", "demand_cv", "seasonality_amplitude", "attachment_rate", "destination_categories", "category_source"]
    display_cols = [c for c in display_cols if c in roles.columns]
    st.dataframe(
        roles[display_cols].sort_values(["role", "category"]),
        use_container_width=True,
        hide_index=True,
    )


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/insights: Product Performance")

    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        abc_filter = c1.multiselect("ABC Class", ["A", "B", "C"], default=["A", "B", "C"])
        xyz_filter = c2.multiselect("XYZ Class", ["X", "Y", "Z"], default=["X", "Y", "Z"])
        stage_filter = c3.multiselect("Lifecycle", ["growth", "mature", "decline"], default=["growth", "mature", "decline"])

    perf = get_product_metrics(df)
    abc = abc_analysis(df)
    xyz = xyz_analysis(df)
    lifecycle = product_lifecycle_stage(df)
    velocity = compute_velocity(df)
    repeat = compute_repeat_rate(df)

    # Merge all
    full = (
        perf.merge(abc[["stockcode", "abc_class"]], on="stockcode", how="left")
        .merge(xyz[["stockcode", "xyz_class"]], on="stockcode", how="left")
        .merge(lifecycle[["stockcode", "stage"]], on="stockcode", how="left")
        .merge(velocity[["stockcode", "velocity"]], on="stockcode", how="left")
        .merge(repeat[["stockcode", "repeat_rate"]], on="stockcode", how="left")
    )

    filtered = full[
        full["abc_class"].isin(abc_filter)
        & full["xyz_class"].isin(xyz_filter)
        & full["stage"].isin(stage_filter)
    ]

    st.divider()
    _render_abc_pareto(df)

    st.divider()
    _render_xyz_volatility(df)

    st.divider()
    _render_lifecycle_scatter(df)

    st.divider()
    _render_velocity_repeat(full)

    st.divider()
    _render_category_roles(df)

    st.divider()
    _render_sku_rationalization(df)

    st.divider()
    st.subheader(":material/table_rows: Full Performance Table")
    display_cols = ["stockcode", "revenue", "units", "transactions", "customers",
                    "abc_class", "xyz_class", "stage", "velocity", "repeat_rate"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[display_cols].sort_values("revenue", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


MODE_SPEC: ModeSpec = ModeSpec(
    key="performance",
    label="Performance",
    icon=":material/insights:",
    handler=render,
)