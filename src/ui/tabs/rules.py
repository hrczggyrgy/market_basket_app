"""Association Rules tab."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.data import derive_product_lookup
from src.analytics.rules import (
    aggregate_rules_to_categories,
    bootstrap_lift_ci,
    create_basket_matrix,
    filter_rules,
    flag_redundant_rules,
    generate_rules,
    rules_to_table,
    run_fpgrowth,
)
from src.ui.plots import PALETTE, empty_state, new_fig, show
from src.ui.registry import ModeSpec


def _rule_label(antecedent: str, consequent: str) -> str:
    return f"{antecedent}  →  {consequent}"


def _render_lift_ci_chart(rules: pd.DataFrame, table: pd.DataFrame, top_n: int) -> None:
    st.subheader(":material/error: Lift with Bootstrap Confidence Intervals")
    with_ci = rules[rules["lift_ci_lower"].notna()]
    if with_ci.empty:
        st.info("No rules have a valid bootstrap CI at current settings.")
        return

    chart_rules = with_ci.nlargest(top_n, "lift")
    labels = [
        _rule_label(table.loc[idx, "antecedent"], table.loc[idx, "consequent"])
        for idx in chart_rules.index
    ]

    fig = new_fig()
    fig.add_trace(
        go.Bar(
            x=labels[::-1],
            y=chart_rules["lift"].values[::-1],
            error_y={
                "array": (chart_rules["lift_ci_upper"] - chart_rules["lift"]).values[::-1],
                "arrayminus": (chart_rules["lift"] - chart_rules["lift_ci_lower"]).values[::-1],
                "type": "data",
                "thickness": 1.2,
            },
            marker={"color": PALETTE[0]},
            name="Lift",
            orientation="h",
        )
    )
    fig.update_layout(
        yaxis={"categoryorder": "array", "categoryarray": labels[::-1]},
        xaxis={"title": "Lift"},
        height=max(360, 28 * len(labels)),
    )
    show(fig)


def _render_strength_stability_scatter(rules: pd.DataFrame, table: pd.DataFrame) -> None:
    """Scatter: Lift (strength) vs CI Width (stability)."""
    st.subheader(":material/analytics: Rule Strength vs Stability")
    with_ci = rules[rules["lift_ci_lower"].notna() & rules["lift_ci_upper"].notna()].copy()
    if with_ci.empty:
        st.info("No rules have valid bootstrap confidence intervals at current settings.")
        return
    
    with_ci["ci_width"] = with_ci["lift_ci_upper"] - with_ci["lift_ci_lower"]
    
    # Add antecedent category for coloring
    def get_first_antecedent(itemset) -> str | None:
        if itemset:
            return list(itemset)[0]
        return None
    
    with_ci["anchor"] = with_ci["antecedents"].apply(get_first_antecedent)
    with_ci = with_ci[with_ci["anchor"].notna()]
    
    if with_ci.empty:
        st.info("No rules with identifiable anchors.")
        return
    
    # Get product names for hover
    lookup = table.set_index("antecedent")["consequent"].to_dict()  # not quite right
    # Build a simple label for each rule
    with_ci["rule_label"] = with_ci.apply(
        lambda r: f"{list(r['antecedents'])[0]} → {list(r['consequents'])[0]}", axis=1
    )
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=with_ci["lift"],
            y=with_ci["ci_width"],
            mode="markers",
            marker={
                "size": 10,
                "color": with_ci["lift"],
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "Lift"},
                "line": {"color": "white", "width": 1},
            },
            text=with_ci["rule_label"],
            hovertemplate="Rule: %{text}<br>Lift: %{x:.2f}<br>CI Width: %{y:.3f}<extra></extra>",
            name="Rules",
        )
    )
    
    # Add quadrant lines (median splits)
    med_lift = with_ci["lift"].median()
    med_ci = with_ci["ci_width"].median()
    fig.add_hline(y=med_ci, line_dash="dash", line_color="gray", annotation_text="Median CI Width")
    fig.add_vline(x=med_lift, line_dash="dash", line_color="gray", annotation_text="Median Lift")
    
    fig.update_layout(
        xaxis={"title": "Lift (Strength)"},
        yaxis={"title": "CI Width (Stability) — lower = more stable"},
        height=450,
    )
    show(fig)
    
    st.caption(
        "Quadrants: High Lift + Low CI Width (bottom-right) = Strong & Stable. "
        "High Lift + High CI Width (top-right) = Strong but Uncertain. "
        "Color = Lift magnitude."
    )


def _render_anchor_drilldown(df: pd.DataFrame, rules: pd.DataFrame, table: pd.DataFrame) -> None:
    st.subheader(":material/filter_center_focus: Anchor Product Drill-down")
    products = sorted(set(rules["antecedents"].explode()) | set(rules["consequents"].explode()))
    if not products:
        st.info("No rules available for drill-down.")
        return

    lookup = derive_product_lookup(df)
    display_options = {
        str(lookup.loc[lookup["stockcode"] == p, "product"].iloc[0]) if (lookup["stockcode"] == p).any() else p: p
        for p in products
    }

    selected = st.selectbox("Select anchor product", options=list(display_options.keys()))
    anchor = display_options[selected]

    related = rules[
        rules["antecedents"].apply(lambda s: anchor in s)
        | rules["consequents"].apply(lambda s: anchor in s)
    ]
    if related.empty:
        st.info(f"No rules involve {selected}.")
        return

    labels = [
        _rule_label(table.loc[idx, "antecedent"], table.loc[idx, "consequent"])
        for idx in related.index
    ]

    fig = new_fig()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=related["lift"],
            marker={"color": [PALETTE[1] if anchor in r else PALETTE[0] for r in related["antecedents"]]},
            name="Lift",
        )
    )
    fig.update_layout(xaxis={"tickangle": -30}, yaxis={"title": "Lift"})
    show(fig)


def _render_rule_network(df: pd.DataFrame, rules: pd.DataFrame, top_n: int) -> None:
    st.subheader(":material/hub: Rule Network")
    if rules.empty:
        show(empty_state("No rules to display"))
        return

    top = rules.nlargest(top_n, "lift")
    basket = create_basket_matrix(df)
    product_support = basket.mean(axis=0)

    graph = nx.DiGraph()
    for _, row in top.iterrows():
        for a in row["antecedents"]:
            for c in row["consequents"]:
                if a != c:
                    graph.add_edge(a, c, lift=row["lift"], support=row["support"])

    if graph.number_of_nodes() == 0:
        show(empty_state("No connected rules"))
        return

    pos = nx.spring_layout(graph, seed=42, k=0.6)

    edge_x: list[float] = []
    edge_y: list[float] = []
    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, float("nan")]
        edge_y += [y0, y1, float("nan")]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"color": "#B0B0B0", "width": 1},
        hoverinfo="none",
    )

    node_x = [pos[n][0] for n in graph.nodes()]
    node_y = [pos[n][1] for n in graph.nodes()]
    node_sizes = [6 + 18 * product_support.get(n, 0) for n in graph.nodes()]
    node_text = [n for n in graph.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="bottom center",
        marker={
            "size": node_sizes,
            "color": PALETTE[0],
            "line": {"color": "white", "width": 1},
        },
        hoverinfo="text",
    )

    fig = new_fig()
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        showlegend=False,
    )
    show(fig)


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/schema: Association Rules (FP-Growth)")

    with st.expander("Parameters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        min_support = c1.number_input("Min Support", 0.001, 0.5, 0.01, 0.001)
        max_len = c2.number_input("Max Itemset Length", 2, 5, 3)
        min_threshold = c3.number_input("Min Confidence", 0.01, 1.0, 0.05, 0.01)
        n_bootstrap = c4.number_input("Bootstrap Resamples", 5, 100, 25, 5)

    basket = create_basket_matrix(df)
    st.caption(f"Basket matrix: {basket.shape[0]} transactions × {basket.shape[1]} products")

    freq = run_fpgrowth(basket, min_support=min_support, max_len=max_len)
    st.caption(f"Frequent itemsets: {len(freq)}")

    if freq.empty:
        st.warning("No frequent itemsets found with current parameters.")
        return

    rules = generate_rules(freq, min_threshold=min_threshold)
    st.caption(f"Rules generated: {len(rules)}")

    if rules.empty:
        st.warning("No rules meet the confidence threshold.")
        return

    filtered = filter_rules(rules, min_lift=1.0, min_confidence=min_threshold)
    st.caption(f"Rules after filtering (lift ≥ 1.0): {len(filtered)}")

    if not filtered.empty:
        filtered = flag_redundant_rules(filtered)
        filtered = bootstrap_lift_ci(df, filtered, n_resamples=n_bootstrap)

        lookup = derive_product_lookup(df)
        table = rules_to_table(filtered, lookup)
        table["is_redundant"] = filtered["is_redundant"].values
        table["lift_ci_lower"] = filtered["lift_ci_lower"].values
        table["lift_ci_upper"] = filtered["lift_ci_upper"].values

        hide_redundant = st.checkbox("Hide redundant rules", value=False)
        display = table[~table["is_redundant"]] if hide_redundant else table
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.caption(f"Redundant rules: {int(filtered['is_redundant'].sum())} of {len(filtered)}")

        csv = table.to_csv(index=False)
        st.download_button(
            ":material/download: Download Rules CSV",
            csv,
            "association_rules.csv",
            "text/csv",
        )

        # Category-level rules rollup
        st.divider()
        st.subheader(":material/category: Category Affinities (Rollup)")
        cat_rules = aggregate_rules_to_categories(filtered, lookup)
        if not cat_rules.empty:
            # Format for display
            cat_display = cat_rules.copy()
            cat_display["support"] = cat_display["support"].apply(lambda x: f"{x:.4f}")
            cat_display["confidence"] = cat_display["confidence"].apply(lambda x: f"{x:.2%}")
            cat_display["lift"] = cat_display["lift"].apply(lambda x: f"{x:.2f}")
            cat_display["avg_lift"] = cat_display["avg_lift"].apply(lambda x: f"{x:.2f}")
            cat_display["max_lift"] = cat_display["max_lift"].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(
                cat_display[["antecedent_category", "consequent_category", "rule_count", "support", "confidence", "lift", "avg_lift", "max_lift"]],
                use_container_width=True,
                hide_index=True,
            )
            
            # Category affinity heatmap
            st.caption("Heatmap: Lift by Category Pair")
            pivot = cat_rules.pivot_table(
                index="antecedent_category",
                columns="consequent_category",
                values="lift",
                fill_value=0
            )
            if not pivot.empty:
                fig = go.Figure(
                    data=go.Heatmap(
                        z=pivot.values,
                        x=pivot.columns.tolist(),
                        y=pivot.index.tolist(),
                        colorscale="RdYlGn",
                        colorbar={"title": "Avg Lift"},
                        hovertemplate="From: %{y}<br>To: %{x}<br>Lift: %{z:.2f}<extra></extra>",
                    )
                )
                fig.update_layout(
                    xaxis={"title": "Consequent Category", "tickangle": -45},
                    yaxis={"title": "Antecedent Category"},
                    height=max(300, len(pivot) * 30 + 100),
                )
                show(fig)
            
            csv_cat = cat_rules.to_csv(index=False)
            st.download_button(
                ":material/download: Download Category Rules CSV",
                csv_cat,
                "category_rules.csv",
                "text/csv",
            )
        else:
            st.info("No category-level rules available (insufficient category diversity).")

        # Rule Strength vs Stability scatter
        st.divider()
        _render_strength_stability_scatter(filtered, table)

        st.divider()
        _render_lift_ci_chart(filtered, table, top_n=15)

        st.divider()
        _render_anchor_drilldown(df, filtered, table)

        st.divider()
        top_n_network = st.slider("Network: top rules by lift", 10, 100, 40)
        _render_rule_network(df, filtered, top_n=top_n_network)


MODE_SPEC: ModeSpec = ModeSpec(
    key="rules",
    label="Association Rules",
    icon=":material/schema:",
    handler=render,
)
