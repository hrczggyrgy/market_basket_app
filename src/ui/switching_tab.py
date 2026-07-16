"""Switching analysis tab with persistent tab state."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.switching import (
    compute_switching_matrix,
    compute_transition_matrix,
    get_customer_loyalty_metrics,
    get_switching_heatmap_data,
    get_top_switching_paths,
)
from src.ui.export import render_analytics_export
from src.ui.tabs import persistent_tabs


def render_switching_tab(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render product switching analysis tab with persistent sub-tabs."""
    st.header("🔀 Product Switching Analysis")
    st.caption(
        "Tracks when a customer buys product A on one visit, then product B on the next. "
        "High **switch rate** from A \u2192 B suggests substitutability or a sequential need. "
        "A new Markov view summarizes transition probabilities among the most important products."
    )

    if transactions_df.empty:
        st.warning("No transaction data available")
        return

    with st.expander(" Switching Parameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            window_days = st.slider(
                "Analysis Window (days)",
                30,
                365,
                params.get("window_days", 90),
                key="switch_tab_window_days",
            )
            min_transactions = st.slider(
                "Min Customer Transactions",
                2,
                10,
                params.get("min_transactions", 3),
                key="switch_tab_min_transactions",
            )
        with col2:
            top_n_products = st.slider(
                "Top Products for Heatmap / Markov",
                10,
                50,
                30,
                key="switch_tab_top_n_products",
            )
            min_switches = st.number_input(
                "Min Switch Count",
                1,
                50,
                params.get("min_switches", 2),
                key="switch_tab_min_switches",
            )

    with st.spinner("Computing switching patterns..."):
        switch_matrix = compute_switching_matrix(
            transactions_df, window_days=window_days, min_transactions=min_transactions
        )

    @st.cache_data
    def get_loyalty_cached(df):
        return get_customer_loyalty_metrics(df)

    @st.cache_data
    def get_top_paths_cached(df, min_sw):
        return get_top_switching_paths(df, min_switches=min_sw)

    @st.cache_data
    def get_transition_cached(df, top_n):
        return compute_transition_matrix(df, top_n=top_n)

    loyalty = get_loyalty_cached(transactions_df)
    top_paths = get_top_paths_cached(transactions_df, min_switches)
    transition_matrix = get_transition_cached(transactions_df, top_n_products)

    st.subheader("Switching Overview")

    total_events = len(switch_matrix) if not switch_matrix.empty else 0
    avg_switch = switch_matrix["switch_rate"].mean() if not switch_matrix.empty else 0
    mean_asym = (
        switch_matrix["asymmetry_ratio"].abs().mean() if not switch_matrix.empty else 0
    )
    total_customers = len(loyalty)
    loyal_n = (
        int(loyalty["loyalty_segment"].eq("Loyal").sum())
        if "loyalty_segment" in loyalty.columns
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Switching Events", total_events)
    col2.metric("Avg Switch Rate", f"{avg_switch:.1%}")
    col3.metric("Avg Flow Asymmetry", f"{mean_asym:.2f}")
    col4.metric(
        "Loyal Customers",
        loyal_n,
        delta=f"{loyal_n / total_customers:.0%} of base" if total_customers else None,
        delta_color="normal",
    )

    tab_labels = [
        " Switching Heatmap",
        " Top Switch Paths",
        " Asymmetry View",
        " Markov Chain",
        " Sankey Flow",
        " Customer Loyalty",
    ]
    selected = persistent_tabs(tab_labels, "switching_view_tabs", default_tab=0)

    if selected == 0:
        _render_heatmap_tab(switch_matrix, transactions_df, product_lookup, top_n_products)
    elif selected == 1:
        _render_top_paths_tab(top_paths, product_lookup)
    elif selected == 2:
        _render_asymmetry_tab(top_paths, product_lookup)
    elif selected == 3:
        _render_markov_tab(transition_matrix, product_lookup)
    elif selected == 4:
        _render_sankey_tab(switch_matrix, product_lookup)
    elif selected == 5:
        _render_loyalty_tab(loyalty)


def _render_heatmap_tab(
    switch_matrix: pd.DataFrame,
    transactions_df: pd.DataFrame,
    product_lookup: dict,
    top_n_products: int,
):
    """Render the switching heatmap tab."""
    st.subheader("Product Switching Heatmap")

    @st.cache_data
    def get_heatmap_data_cached(df, top_n):
        return get_switching_heatmap_data(df, top_n_products=top_n)

    heatmap_data = get_heatmap_data_cached(transactions_df, top_n_products)

    if not heatmap_data.empty:
        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns.tolist(),
                y=heatmap_data.index.tolist(),
                colorscale="RdYlGn",
                colorbar=dict(title="Switch Count"),
                hoverongaps=False,
                hovertemplate="From: %{y}<br>To: %{x}<br>Count: %{z}<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"Product Switching Matrix (Top {top_n_products})",
            xaxis_title="To Product",
            yaxis_title="From Product",
            height=600,
        )
        st.plotly_chart(fig, width="stretch")

        if not switch_matrix.empty and "from_product" in switch_matrix.columns:
            top_exits = (
                switch_matrix.groupby("from_product")["switch_count"]
                .sum()
                .nlargest(3)
                .reset_index()
            )
            top_exits["name"] = top_exits["from_product"].map(
                product_lookup if product_lookup else {}
            )
            top_exits["name"] = top_exits["name"].fillna(top_exits["from_product"])
            exit_str = " \u00b7 ".join(
                f"**{row['name']}** ({int(row['switch_count'])})"
                for _, row in top_exits.iterrows()
            )
            st.info(f"\ud83d\udce4 **Top switch-away products:** {exit_str}")

        with st.expander("View Raw Matrix"):
            st.dataframe(heatmap_data.round(2), width="stretch")
    else:
        st.info("No switching data available for heatmap")


def _render_top_paths_tab(top_paths: pd.DataFrame, product_lookup: dict):
    """Render the top switching paths tab."""
    st.subheader("Top Switching Paths")

    if not top_paths.empty:
        top_paths = top_paths.copy()
        top_paths["From Product"] = top_paths["from_product"].map(product_lookup)
        top_paths["To Product"] = top_paths["to_product"].map(product_lookup)
        top_paths["path"] = top_paths["From Product"] + " \u2192 " + top_paths["To Product"]

        display_cols = [
            "From Product",
            "To Product",
            "switch_count",
            "switch_rate",
            "avg_days_between",
            "asymmetry_ratio",
            "switches_per_customer",
        ]
        available = [c for c in display_cols if c in top_paths.columns]
        st.dataframe(
            top_paths[available].round(4),
            width="stretch",
            hide_index=True,
        )

        render_analytics_export(top_paths, "Top_Switching_Paths")

        st.subheader("Top 20 Switches by Count")
        chart_data = top_paths.head(20).copy()

        # Use asymmetry_ratio for color if available, otherwise use switch_count
        color_col = "asymmetry_ratio" if "asymmetry_ratio" in chart_data.columns else "switch_count"
        color_scale = "RdYlGn" if "asymmetry_ratio" in chart_data.columns else "Blues"

        fig = px.bar(
            chart_data,
            x="switch_count",
            y="path",
            orientation="h",
            color=color_col,
            color_continuous_scale=color_scale,
            title="Top Switching Paths",
            labels={"switch_count": "Switch Count", "path": "Path"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No significant switching paths found")


def _render_asymmetry_tab(top_paths: pd.DataFrame, product_lookup: dict):
    """Render asymmetry scatter for directional winning/losing flows."""
    st.subheader("Asymmetry View: Who Is Winning the Switch?")
    st.caption(
        "Positive asymmetry means the displayed direction dominates its reverse path. "
        "Values near zero indicate balanced switching."
    )

    if top_paths.empty:
        st.info("No significant switching paths found")
        return

    # Check required columns exist
    required_cols = ["asymmetry_ratio", "switches_per_customer"]
    if not all(c in top_paths.columns for c in required_cols):
        st.warning("Asymmetry data not available. Please re-run analysis to refresh cache.")
        return

    data = top_paths.copy()
    data["From Product"] = data["from_product"].map(product_lookup)
    data["To Product"] = data["to_product"].map(product_lookup)
    data["path"] = data["From Product"] + " \u2192 " + data["To Product"]

    fig = px.scatter(
        data,
        x="switch_count",
        y="asymmetry_ratio",
        color="asymmetry_ratio",
        size="switches_per_customer",
        color_continuous_scale="RdYlGn",
        range_color=[-1, 1],
        hover_name="path",
        hover_data=["avg_days_between", "switch_rate"],
        title="Asymmetry Ratio: +1 = shown direction dominates, 0 = balanced",
        labels={
            "asymmetry_ratio": "Asymmetry Ratio",
            "switch_count": "Switch Count",
            "switches_per_customer": "Switches / Customer",
        },
    )
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="black",
        annotation_text="Balanced reverse flow",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_markov_tab(transition_matrix: pd.DataFrame, product_lookup: dict):
    """Render first-order Markov transition matrix."""
    st.subheader("Markov Transition Matrix")
    st.caption(
        "Each row sums to 1.0 and shows the probability of the next purchase, "
        "conditional on the current product. This is a lightweight sequential model "
        "that strengthens the scientific basis of switching analysis."
    )

    if transition_matrix.empty:
        st.info("Not enough sequential transitions to build the Markov matrix")
        return

    row_labels = [
        product_lookup.get(idx, idx) if product_lookup else idx
        for idx in transition_matrix.index
    ]
    col_labels = [
        product_lookup.get(col, col) if product_lookup else col
        for col in transition_matrix.columns
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=transition_matrix.values,
            x=col_labels,
            y=row_labels,
            colorscale="Blues",
            zmin=0,
            zmax=max(float(transition_matrix.values.max()), 0.05),
            text=transition_matrix.round(2).values,
            texttemplate="%{text}",
            hovertemplate="Current: %{y}<br>Next: %{x}<br>Prob: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="First-order Product Transition Probabilities",
        height=650,
        xaxis_title="Next Product",
        yaxis_title="Current Product",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Transition Table"):
        st.dataframe(transition_matrix.round(3), width="stretch")


def _render_sankey_tab(switch_matrix: pd.DataFrame, product_lookup: dict):
    """Render the Sankey flow tab."""
    st.subheader("Switching Flow (Sankey)")

    if not switch_matrix.empty:
        top_switches = switch_matrix.nlargest(30, "switch_count")
        all_products = list(
            set(top_switches["from_product"].tolist() + top_switches["to_product"].tolist())
        )
        product_to_idx = {p: i for i, p in enumerate(all_products)}

        sources = top_switches["from_product"].map(product_to_idx).tolist()
        targets = top_switches["to_product"].map(product_to_idx).tolist()
        values = top_switches["switch_count"].tolist()

        labels = [
            (
                product_lookup.get(p, p)[:20] + "..."
                if len(product_lookup.get(p, p)) > 20
                else product_lookup.get(p, p)
            )
            if product_lookup
            else (p[:20] + "..." if len(p) > 20 else p)
            for p in all_products
        ]

        max_val = max(values) if values else 1
        node_outgoing = {
            p: top_switches.loc[top_switches["from_product"] == p, "switch_count"].sum()
            for p in all_products
        }
        node_colors = [
            "rgba(70, 130, 180, "
            f"{min(0.4 + node_outgoing.get(p, 0) / max_val * 0.6, 1.0):.2f})"
            for p in all_products
        ]

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=20,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=labels,
                        color=node_colors,
                        hovertemplate="%{label}<extra></extra>",
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color="rgba(100, 100, 100, 0.3)",
                        hovertemplate=(
                            "%{source.label} \u2192 %{target.label}: "
                            "%{value} switches<extra></extra>"
                        ),
                    ),
                )
            ]
        )

        fig.update_layout(
            title_text="Product Switching Flow",
            font=dict(size=13, family="Arial, sans-serif"),
            height=600,
            margin=dict(l=150, r=50, t=50, b=50),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No switching data for Sankey diagram")


def _render_loyalty_tab(loyalty: pd.DataFrame):
    """Render the customer loyalty segments tab."""
    st.subheader("Customer Loyalty Segments")

    if not loyalty.empty and "loyalty_segment" in loyalty.columns:
        segment_counts = loyalty["loyalty_segment"].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Segment Distribution**")
            fig = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="Customer Loyalty Segments",
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.write("**Segment Metrics**")
            segment_metrics = (
                loyalty.groupby("loyalty_segment")
                .agg(
                    Customers=("transaction_count", "count"),
                    Avg_Transactions=("transaction_count", "mean"),
                    Avg_Products=("unique_products", "mean"),
                    Repeat_Rate=("repeat_rate", "mean"),
                    Concentration=("concentration_hhi", "mean"),
                )
                .round(2)
            )
            st.dataframe(segment_metrics, width="stretch")

        st.write("**Top Customers by Loyalty**")
        top_loyal = loyalty[loyalty["loyalty_segment"] == "Loyal"].nlargest(20, "repeat_rate")
        if not top_loyal.empty:
            available_cols = [
                "transaction_count",
                "unique_products",
                "repeat_rate",
                "favorite_product",
                "concentration_hhi",
                "customer_lifespan_days",
            ]
            display_cols = [c for c in available_cols if c in top_loyal.columns]
            st.dataframe(
                top_loyal[display_cols].round(2),
                width="stretch",
            )

        render_analytics_export(loyalty, "Customer_Loyalty_Metrics")
    else:
        st.info("Loyalty metrics not available")
