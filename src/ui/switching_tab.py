"""Switching analysis tab with persistent tab state."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.switching import (
    compute_switching_matrix,
    get_customer_loyalty_metrics,
    get_switching_heatmap_data,
    get_top_switching_paths,
)
from src.ui.export import render_analytics_export
from src.ui.tabs import persistent_tabs


def render_switching_tab(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render product switching analysis tab with persistent sub-tabs."""
    st.header(" Product Switching Analysis")

    if transactions_df.empty:
        st.warning("No transaction data available")
        return

    # Parameters
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
                "Top Products for Heatmap", 10, 50, 30, key="switch_tab_top_n_products"
            )
            min_switches = st.number_input(
                "Min Switch Count",
                1,
                50,
                params.get("min_switches", 2),
                key="switch_tab_min_switches",
            )

    with st.spinner("Computing switching patterns..."):
        # Compute switching matrix
        switch_matrix = compute_switching_matrix(
            transactions_df, window_days=window_days, min_transactions=min_transactions
        )

    # Customer loyalty metrics (cached)
    @st.cache_data
    def get_loyalty_cached(df):
        return get_customer_loyalty_metrics(df)

    loyalty = get_loyalty_cached(transactions_df)

    # Top switching paths (cached)
    @st.cache_data
    def get_top_paths_cached(df, min_sw):
        return get_top_switching_paths(df, min_switches=min_sw)

    top_paths = get_top_paths_cached(transactions_df, min_switches)

    # Overview metrics
    st.subheader("Switching Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Switching Events",
            len(switch_matrix) if not switch_matrix.empty else 0,
        )
    with col2:
        st.metric(
            "Avg Switch Rate",
            (f"{switch_matrix['switch_rate'].mean():.1%}" if not switch_matrix.empty else "0%"),
        )
    with col3:
        st.metric(
            "Loyal Customers",
            (
                int(loyalty["loyalty_segment"].eq("Loyal").sum())
                if "loyalty_segment" in loyalty.columns
                else 0
            ),
        )
    with col4:
        st.metric(
            "Switchers",
            (
                int(loyalty["loyalty_segment"].eq("Switcher").sum())
                if "loyalty_segment" in loyalty.columns
                else 0
            ),
        )

    # Persistent tabs
    tab_labels = [
        " Switching Heatmap",
        " Top Switch Paths",
        " Sankey Flow",
        " Customer Loyalty",
    ]
    selected = persistent_tabs(tab_labels, "switching_view_tabs", default_tab=0)

    if selected == 0:
        _render_heatmap_tab(switch_matrix, transactions_df, product_lookup, top_n_products)
    elif selected == 1:
        _render_top_paths_tab(top_paths, product_lookup)
    elif selected == 2:
        _render_sankey_tab(switch_matrix, product_lookup)
    elif selected == 3:
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

        # Show raw data
        with st.expander("View Raw Matrix"):
            st.dataframe(heatmap_data.round(2), width="stretch")
    else:
        st.info("No switching data available for heatmap")


def _render_top_paths_tab(top_paths: pd.DataFrame, product_lookup: dict):
    """Render the top switching paths tab."""
    st.subheader("Top Switching Paths")

    if not top_paths.empty:
        # Add product names
        top_paths = top_paths.copy()
        top_paths["From Product"] = top_paths["from_product"].map(product_lookup)
        top_paths["To Product"] = top_paths["to_product"].map(product_lookup)

        display_cols = [
            "From Product",
            "To Product",
            "switch_count",
            "switch_rate",
            "avg_days_between",
        ]
        st.dataframe(
            top_paths[display_cols].round(4),
            width="stretch",
            hide_index=True,
        )

        render_analytics_export(top_paths, "Top_Switching_Paths")

        # Bar chart
        st.subheader("Top 20 Switches by Count")
        chart_data = top_paths.head(20).copy()
        chart_data["path"] = chart_data["From Product"] + " → " + chart_data["To Product"]

        fig = px.bar(
            chart_data,
            x="switch_count",
            y="path",
            orientation="h",
            title="Top Switching Paths",
            labels={"switch_count": "Switch Count", "path": "Path"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No significant switching paths found")


def _render_sankey_tab(switch_matrix: pd.DataFrame, product_lookup: dict):
    """Render the Sankey flow tab."""
    st.subheader("Switching Flow (Sankey)")

    if not switch_matrix.empty:
        # Create Sankey from switching matrix
        top_switches = switch_matrix.nlargest(30, "switch_count")

        # Get unique products
        all_products = list(
            set(top_switches["from_product"].tolist() + top_switches["to_product"].tolist())
        )
        product_to_idx = {p: i for i, p in enumerate(all_products)}

        sources = top_switches["from_product"].map(product_to_idx).tolist()
        targets = top_switches["to_product"].map(product_to_idx).tolist()
        values = top_switches["switch_count"].tolist()

        labels = [product_lookup.get(p, p) if product_lookup else p for p in all_products]

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=labels,
                        color="lightblue",
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color="rgba(100, 100, 100, 0.3)",
                    ),
                )
            ]
        )

        fig.update_layout(title_text="Product Switching Flow", font_size=10, height=500)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No switching data for Sankey diagram")


def _render_loyalty_tab(loyalty: pd.DataFrame):
    """Render the customer loyalty segments tab."""
    st.subheader("Customer Loyalty Segments")

    if not loyalty.empty and "loyalty_segment" in loyalty.columns:
        # Segment distribution
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

        # Loyalty details table
        st.write("**Top Customers by Loyalty**")
        top_loyal = loyalty[loyalty["loyalty_segment"] == "Loyal"].nlargest(20, "repeat_rate")
        if not top_loyal.empty:
            # Only display columns that exist
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
