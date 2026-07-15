"""Association rules tab with persistent tab state."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.rules.generator import filter_rules, format_rules_for_display
from src.ui.export import render_analytics_export
from src.ui.tabs import persistent_tabs
from src.viz.heatmap import create_heatmap, create_scatter_heatmap
from src.viz.network import create_network_graph


def render_rules_tab(rules: pd.DataFrame, product_lookup: dict, params: dict):
    """Render association rules analysis tab with persistent sub-tabs."""
    st.header("📋 Association Rules")
    st.caption(
        "Discovers product combinations that occur together more than chance. "
        "**Lift > 1** = positive association. **Conviction → ∞** = near-deterministic rule."
    )

    if rules.empty:
        st.warning("No rules generated. Try lowering min_support or min_confidence.")
        return

    # Filter controls
    with st.expander(" Filter Rules", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            min_supp = st.number_input(
                "Min Support",
                0.0,
                1.0,
                params.get("min_support", 0.0),
                0.001,
                format="%.4f",
                help="Fraction of transactions containing both items. Higher = more frequent pairs.",
            )
            min_conf = st.number_input(
                "Min Confidence",
                0.0,
                1.0,
                params.get("min_confidence", 0.5),
                0.05,
                help="P(B|A): probability of B given A was bought. Higher = more reliable rules.",
            )

        with col2:
            min_lift = st.number_input(
                "Min Lift",
                0.0,
                10.0,
                params.get("min_lift", 1.0),
                0.1,
                help="Lift > 1 means items co-occur more than by chance. Lift = 1 means independence.",
            )
            max_lift = st.number_input(
                "Max Lift",
                0.0,
                100.0,
                params.get("max_lift", 100.0),
                0.5,
                help="Caps very high lift values that may arise from rare items.",
            )

        with col3:
            min_lev = st.number_input(
                "Min Leverage",
                -1.0,
                1.0,
                params.get("min_leverage", -1.0),
                0.01,
                help="Leverage = support(A,B) − support(A)×support(B). Positive = synergy above baseline.",
            )
            min_conv = st.number_input(
                "Min Conviction",
                0.0,
                10.0,
                params.get("min_conviction", 0.0),
                0.1,
                help="Conviction → ∞ as confidence → 1. Values > 1.5 indicate a strong directional rule.",
            )

        with col4:
            max_ant_len = st.number_input(
                "Max Antecedent Length", 1, 10, params.get("max_antecedent_len", 3)
            )
            max_cons_len = st.number_input(
                "Max Consequent Length", 1, 10, params.get("max_consequent_len", 3)
            )

    # Apply filters
    filtered = filter_rules(
        rules,
        min_support=min_supp,
        min_confidence=min_conf,
        min_lift=min_lift,
        max_lift=max_lift,
        min_leverage=min_lev,
        min_conviction=min_conv,
        max_antecedent_len=max_ant_len,
        max_consequent_len=max_cons_len,
    )

    total_rules = len(rules)
    filtered_count = len(filtered)
    st.metric(
        "Filtered Rules",
        filtered_count,
        delta=f"{filtered_count - total_rules} from total {total_rules}",
        delta_color="off",
    )

    if filtered.empty:
        st.warning("No rules match the current filters")
        return

    # Format for display
    display_rules = format_rules_for_display(filtered, product_lookup)

    # Persistent sub-tabs for different views
    tab_labels = [" Table", " Network", " Heatmap", " Scatter", " 3D"]
    selected = persistent_tabs(tab_labels, "rules_view_tabs", default_tab=0)

    if selected == 0:
        _render_rules_table_tab(display_rules, filtered)
    elif selected == 1:
        _render_rules_network_tab(filtered, product_lookup, min_lift)
    elif selected == 2:
        _render_rules_heatmap_tab(filtered)
    elif selected == 3:
        _render_rules_scatter_tab(filtered)
    elif selected == 4:
        _render_rules_3d_tab(filtered)


def _render_rules_table_tab(display_rules: pd.DataFrame, filtered: pd.DataFrame):
    """Render the rules table view."""
    st.subheader("Rules Table")

    # Top rule insight callout
    if not display_rules.empty and "lift" in filtered.columns:
        top = filtered.nlargest(1, "lift").iloc[0]
        ant = ", ".join(str(x) for x in top["antecedents"])
        con = ", ".join(str(x) for x in top["consequents"])
        st.info(
            f"💡 **Strongest rule:** `{ant}` → `{con}`  \n"
            f"Lift **{top['lift']:.2f}** · Confidence **{top['confidence']:.2%}** · "
            f"Support **{top['support']:.4f}**"
        )

    # Column selector
    available_cols = display_rules.columns.tolist()
    default_cols = [
        "rule",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
        "zhangs_metric",
    ]
    selected_cols = st.multiselect(
        "Display Columns",
        available_cols,
        default=[c for c in default_cols if c in available_cols],
        key="rules_cols",
    )

    if selected_cols:
        numeric_cols = [
            c for c in selected_cols if display_rules[c].dtype in ["float64", "int64"]
        ]
        if numeric_cols:
            sort_col = st.selectbox(
                "Sort by",
                numeric_cols,
                index=numeric_cols.index("lift") if "lift" in numeric_cols else 0,
                key="rules_sort_col",
            )
            sort_asc = st.checkbox("Ascending", value=False, key="rules_sort_asc")
            display_rules = display_rules.sort_values(sort_col, ascending=sort_asc)

        st.dataframe(
            display_rules[selected_cols],
            width="stretch",
            hide_index=True,
            height=500,
        )

    render_analytics_export(filtered, "Association_Rules")


def _render_rules_network_tab(filtered: pd.DataFrame, product_lookup: dict, min_lift: float):
    """Render the network graph view."""
    st.subheader("Rules Network Graph")

    if len(filtered) > 0:
        min_lift_net = st.slider(
            "Min Lift for Network", 1.0, 5.0, min_lift, 0.1, key="net_min_lift"
        )
        max_nodes = st.slider("Max Nodes", 10, 100, 40, key="net_max_nodes")
        max_edges = st.slider("Max Edges", 20, 200, min(100, len(filtered)), key="net_max_edges")

        fig = create_network_graph(
            filtered,
            product_lookup=product_lookup,
            min_lift=min_lift_net,
            max_nodes=max_nodes,
            max_edges=max_edges,
            title=f"Association Rules Network (Lift ≥ {min_lift_net})",
        )
        st.plotly_chart(fig, width="stretch")


def _render_rules_heatmap_tab(filtered: pd.DataFrame):
    """Render the heatmap view."""
    st.subheader("Support-Confidence-Lift Heatmap")

    x_metric = st.selectbox(
        "X-axis", ["support", "confidence", "lift", "leverage"], index=0, key="heatmap_x"
    )
    y_metric = st.selectbox(
        "Y-axis", ["confidence", "support", "lift", "leverage"], index=1, key="heatmap_y"
    )
    color_metric = st.selectbox(
        "Color",
        ["lift", "confidence", "support", "leverage", "conviction"],
        index=0,
        key="heatmap_color",
    )

    fig = create_heatmap(
        filtered,
        x_metric=x_metric,
        y_metric=y_metric,
        color_metric=color_metric,
        title=f"{x_metric.capitalize()} vs {y_metric.capitalize()} (Color: {color_metric})",
    )
    st.plotly_chart(fig, width="stretch")


def _render_rules_scatter_tab(filtered: pd.DataFrame):
    """Render the scatter plot view."""
    st.subheader("Rules Scatter Plot")

    x_metric = st.selectbox(
        "X-axis",
        ["support", "confidence", "lift", "leverage"],
        index=0,
        key="scatter_x",
    )
    y_metric = st.selectbox(
        "Y-axis",
        ["confidence", "support", "lift", "leverage"],
        index=1,
        key="scatter_y",
    )
    color_metric = st.selectbox(
        "Color",
        ["lift", "confidence", "support", "leverage", "conviction"],
        index=0,
        key="scatter_color",
    )

    fig = create_scatter_heatmap(
        filtered, x_metric=x_metric, y_metric=y_metric, color_metric=color_metric
    )
    st.plotly_chart(fig, width="stretch")


def _render_rules_3d_tab(filtered: pd.DataFrame):
    """Render the 3D scatter view."""
    st.subheader("3D: Support × Confidence × Lift")

    fig = _create_3d_scatter(filtered)
    st.plotly_chart(fig, width="stretch")


def _create_3d_scatter(rules: pd.DataFrame) -> go.Figure:
    """Create 3D scatter plot of rules."""
    if rules.empty:
        return go.Figure()

    # Limit points
    plot_rules = rules.nlargest(500, "lift") if len(rules) > 500 else rules

    # Format hover text - vectorized approach
    ant_str = plot_rules["antecedents"].apply(lambda x: ", ".join(map(str, x)))
    cons_str = plot_rules["consequents"].apply(lambda x: ", ".join(map(str, x)))
    hover_text = (
        "A: " + ant_str + "<br>C: " + cons_str + "<br>"
        "Supp: " + plot_rules["support"].round(4).astype(str) + "<br>"
        "Conf: " + plot_rules["confidence"].round(4).astype(str) + "<br>"
        "Lift: " + plot_rules["lift"].round(4).astype(str)
    )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=plot_rules["support"],
                y=plot_rules["confidence"],
                z=plot_rules["lift"],
                mode="markers",
                marker={
                    "size": 5,
                    "color": plot_rules["lift"],
                    "colorscale": "Viridis",
                    "opacity": 0.7,
                    "colorbar": {"title": "Lift"},
                },
                text=hover_text,
                hoverinfo="text",
            )
        ]
    )

    # Lift = 1 reference plane
    x_range = [float(plot_rules["support"].min()), float(plot_rules["support"].max())]
    y_range = [float(plot_rules["confidence"].min()), float(plot_rules["confidence"].max())]
    xx, yy = np.meshgrid(x_range, y_range)
    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=np.ones_like(xx),
            opacity=0.15,
            colorscale=[[0, "red"], [1, "red"]],
            showscale=False,
            name="Lift = 1 (independence)",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="3D Rule Space: Support × Confidence × Lift  (red plane = Lift 1.0)",
        scene={"xaxis_title": "Support", "yaxis_title": "Confidence", "zaxis_title": "Lift"},
        height=600,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )

    return fig
