"""Co-purchase / Affinity analysis tab with persistent tab state."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics.copurchase import (
    compute_affinity_matrix,
    get_product_affinity_profile,
    get_top_affinity_pairs,
)
from src.ui.export import render_analytics_export
from src.ui.tabs import persistent_tabs


# Bug 1: Cached wrappers for heavy computations
@st.cache_data
def _cached_compute_affinity_matrix(transactions_df, min_support, min_lift, top_n_products):
    return compute_affinity_matrix(
        transactions_df, min_support=min_support, min_lift=min_lift, top_n_products=top_n_products
    )


@st.cache_data
def _cached_get_top_affinity_pairs(transactions_df, min_support, min_lift, top_n, top_n_products):
    return get_top_affinity_pairs(
        transactions_df,
        min_support=min_support,
        min_lift=min_lift,
        top_n=top_n,
        top_n_products=top_n_products,
    )


@st.cache_data
def _cached_get_product_affinity_profile(transactions_df, target_product, min_lift, top_n):
    return get_product_affinity_profile(
        transactions_df, target_product, min_lift=min_lift, top_n=top_n
    )


def render_copurchase_tab(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render co-purchase/affinity analysis tab with persistent sub-tabs."""
    st.header("🤝 Co-purchase / Affinity Analysis")

    if transactions_df.empty:
        st.warning("No transaction data available")
        return

    min_support = params.get("min_support", 0.005)
    min_lift = params.get("min_lift", 1.2)
    top_n = params.get("top_n", 50)
    top_n_products = params.get("top_n_products", 50)

    with st.spinner("Computing affinity matrix..."):
        affinity_matrix = _cached_compute_affinity_matrix(
            transactions_df, min_support, min_lift, top_n_products
        )

        top_pairs = _cached_get_top_affinity_pairs(
            transactions_df, min_support, min_lift, top_n, top_n_products
        )

    if top_pairs.empty:
        st.warning("No significant co-purchase pairs found. Try lowering min_lift or min_support.")
        return

    # Add product names
    top_pairs["Product A Name"] = top_pairs["product_a"].map(product_lookup)
    top_pairs["Product B Name"] = top_pairs["product_b"].map(product_lookup)

    # Persistent tabs for different views
    tab_labels = ["📊 Top Pairs", "🔥 Affinity Heatmap", "🔗 Sankey Flow", "🎯 Product Profile"]
    selected_tab = persistent_tabs(tab_labels, "copurchase_view_tabs", default_tab=0)

    if selected_tab == 0:
        _render_top_pairs_tab(top_pairs, min_lift)
    elif selected_tab == 1:
        _render_heatmap_tab(affinity_matrix, product_lookup, top_n_products)
    elif selected_tab == 2:
        _render_sankey_tab(affinity_matrix, product_lookup, top_n_products)
    elif selected_tab == 3:
        _render_product_profile_tab(transactions_df, product_lookup, min_lift)


def _render_top_pairs_tab(top_pairs: pd.DataFrame, min_lift: float):
    """Render the top co-purchase pairs tab."""
    st.subheader(f"Top {len(top_pairs)} Co-purchase Pairs (Lift ≥ {min_lift})")

    display_cols = [
        "Product A Name",
        "Product B Name",
        "support",
        "confidence_a_to_b",
        "confidence_b_to_a",
        "lift",
        "leverage",
    ]

    st.dataframe(top_pairs[display_cols].round(4), width="stretch", hide_index=True)

    render_analytics_export(top_pairs, "CoPurchase_Pairs")

    # Scatter plot
    st.subheader("Support vs Lift")
    fig = px.scatter(
        top_pairs,
        x="support",
        y="lift",
        color="confidence_a_to_b",
        size="confidence_a_to_b",
        hover_data=[
            "Product A Name",
            "Product B Name",
            "confidence_a_to_b",
            "confidence_b_to_a",
        ],
        title="Co-purchase Pairs: Support vs Lift",
        labels={
            "support": "Support",
            "lift": "Lift",
            "confidence_a_to_b": "Confidence A→B",
        },
    )
    st.plotly_chart(fig, width="stretch")


def _render_heatmap_tab(affinity_matrix: pd.DataFrame, product_lookup: dict, top_n_products: int):
    """Render the affinity heatmap tab."""
    st.subheader("Product Affinity Heatmap")

    if not affinity_matrix.empty:
        from src.viz.heatmap import create_affinity_heatmap

        fig = create_affinity_heatmap(
            affinity_matrix,
            product_lookup=product_lookup,
            min_lift=1.0,
            max_products=top_n_products,
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Affinity matrix not available")


def _render_sankey_tab(affinity_matrix: pd.DataFrame, product_lookup: dict, top_n_products: int):
    """Render the Sankey flow tab."""
    st.subheader("Co-purchase Flow (Sankey)")

    if not affinity_matrix.empty:
        from src.viz.network import create_sankey_from_matrix

        fig = create_sankey_from_matrix(
            affinity_matrix.head(min(top_n_products, 15)),
            product_lookup=product_lookup,
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Not enough data for Sankey diagram")


def _render_product_profile_tab(
    transactions_df: pd.DataFrame, product_lookup: dict, min_lift: float
):
    """Render the single product affinity profile tab."""
    st.subheader("Single Product Affinity Profile")

    products = transactions_df["stockcode"].unique()
    selected_product = st.selectbox(
        "Select Product",
        options=products,
        format_func=lambda x: product_lookup.get(x, x),
        key="copurchase_tab_product_select",
    )

    if selected_product:
        profile = _cached_get_product_affinity_profile(
            transactions_df, selected_product, min_lift, 20
        )

        if not profile.empty:
            profile["Co-purchase Product Name"] = profile["co_purchase_product"].map(product_lookup)
            profile["Target Product"] = product_lookup.get(selected_product, selected_product)

            display_cols = [
                "Co-purchase Product Name",
                "support",
                "confidence_target_to_other",
                "confidence_other_to_target",
                "lift",
                "leverage",
            ]

            st.dataframe(
                profile[display_cols].round(4),
                width="stretch",
                hide_index=True,
            )

            render_analytics_export(profile, f"Affinity_{selected_product}")

            # Bar chart
            st.subheader("Top Co-purchases by Lift")
            fig = px.bar(
                profile.head(15),
                x="lift",
                y="Co-purchase Product Name",
                orientation="h",
                color="confidence_target_to_other",
                title=f"Products co-purchased with {product_lookup.get(selected_product, selected_product)}",
                labels={"lift": "Lift", "confidence_target_to_other": "Confidence"},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width="stretch")
        else:
            st.info(
                f"No strong co-purchases found for {product_lookup.get(selected_product, selected_product)}"
            )
