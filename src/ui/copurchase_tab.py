"""Co-purchase / Affinity analysis tab."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.copurchase import (
    compute_affinity_matrix,
    get_product_affinity_profile,
    get_top_affinity_pairs,
)
from src.ui.export import render_analytics_export
from src.ui.tabs import persistent_tabs


@st.cache_data
def _cached_compute_affinity_matrix(transactions_df, min_support, min_lift, top_n_products):
    return compute_affinity_matrix(
        transactions_df,
        min_support=min_support,
        min_lift=min_lift,
        top_n_products=top_n_products,
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
        transactions_df,
        target_product=target_product,
        min_lift=min_lift,
        top_n=top_n,
    )


def render_copurchase_tab(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render co-purchase/affinity analysis tab with persistent sub-tabs."""
    st.header("🛒 Co-purchase / Affinity Analysis")
    st.caption(
        "Measures how often products are bought **in the same basket**. "
        "Lift > 1 = complementary pair. Jaccard and Kulczynski add more robust, "
        "symmetric evidence for scientifically stronger pair ranking."
    )

    if transactions_df.empty:
        st.warning("No transaction data available")
        return

    with st.expander("Affinity Parameters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            min_support = st.number_input(
                "Min Support", 0.001, 1.0, params.get("min_support", 0.005), 0.001
            )
        with col2:
            min_lift = st.number_input(
                "Min Lift", 0.1, 20.0, params.get("min_lift", 1.2), 0.1
            )
        with col3:
            top_n_products = st.slider(
                "Top N Products", 10, 100, params.get("top_n_products", 30)
            )

    with st.spinner("Computing co-purchase patterns..."):
        affinity_matrix = _cached_compute_affinity_matrix(
            transactions_df, min_support, min_lift, top_n_products
        )
        top_pairs = _cached_get_top_affinity_pairs(
            transactions_df, min_support, min_lift, 50, top_n_products
        )

    if top_pairs.empty:
        st.warning("No significant co-purchase pairs found. Try lowering min_lift or min_support.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pairs Found", len(top_pairs))
    col2.metric("Max Lift", f"{top_pairs['lift'].max():.2f}")
    col3.metric("Avg Jaccard", f"{top_pairs['jaccard'].mean():.3f}")
    col4.metric("Avg Kulczynski", f"{top_pairs['kulczynski'].mean():.3f}")

    top_pairs["Product A Name"] = top_pairs["product_a"].map(product_lookup)
    top_pairs["Product B Name"] = top_pairs["product_b"].map(product_lookup)

    tab_labels = [
        "Top Pairs",
        "Quadrant View",
        "Heatmap",
        "Product Profile",
    ]
    active_tab = persistent_tabs(tab_labels, "copurchase_tabs", default_tab=0)

    if active_tab == 0:
        _render_top_pairs_tab(top_pairs)
    elif active_tab == 1:
        _render_quadrant_tab(top_pairs)
    elif active_tab == 2:
        _render_heatmap_tab(affinity_matrix, product_lookup, top_n_products)
    elif active_tab == 3:
        _render_profile_tab(transactions_df, top_pairs, product_lookup, min_lift)


def _render_top_pairs_tab(top_pairs: pd.DataFrame):
    """Render top co-purchase pairs table and classical scatter."""
    st.subheader("Top Co-purchase Pairs")

    display_cols = [
        "Product A Name",
        "Product B Name",
        "support",
        "confidence_a_to_b",
        "confidence_b_to_a",
        "lift",
        "jaccard",
        "kulczynski",
        "cosine",
        "phi_coefficient",
        "leverage",
    ]
    available = [c for c in display_cols if c in top_pairs.columns]
    st.dataframe(top_pairs[available].round(4), width="stretch", hide_index=True)

    if not top_pairs.empty:
        best = top_pairs.nlargest(1, "lift").iloc[0]
        name_a = best.get("Product A Name", best["product_a"])
        name_b = best.get("Product B Name", best["product_b"])
        st.success(
            f"🏆 **Best bundle candidate:** `{name_a}` + `{name_b}`  \n"
            f"Lift **{best['lift']:.2f}** · Jaccard **{best['jaccard']:.3f}** · "
            f"Kulczynski **{best['kulczynski']:.3f}**"
        )

    render_analytics_export(top_pairs, "CoPurchase_Pairs")

    st.subheader("Support vs Lift")
    top_pairs["label"] = (
        top_pairs["Product A Name"].str[:20] + " + " + top_pairs["Product B Name"].str[:20]
    )
    fig = px.scatter(
        top_pairs,
        x="support",
        y="lift",
        color="confidence_a_to_b",
        size="jaccard",
        color_continuous_scale="Blues",
        hover_name="label",
        hover_data=[
            "Product A Name",
            "Product B Name",
            "support",
            "lift",
            "confidence_a_to_b",
            "jaccard",
            "kulczynski",
        ],
        labels={
            "support": "Support",
            "lift": "Lift",
            "confidence_a_to_b": "Confidence A\u2192B",
        },
    )
    fig.add_hline(
        y=1.0,
        line_dash="dot",
        line_color="red",
        annotation_text="Lift = 1 (random co-occurrence)",
        annotation_position="bottom right",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_quadrant_tab(top_pairs: pd.DataFrame):
    """Render academic quadrant view using Jaccard and Kulczynski."""
    st.subheader("Quadrant View: Breadth vs Strength")
    st.caption(
        "Jaccard measures breadth of overlap; Kulczynski averages both directional confidences. "
        "Top-right pairs are typically the strongest candidates for bundling or adjacency."
    )

    qdata = top_pairs.copy()
    qdata["label"] = (
        qdata["Product A Name"].str[:20] + " + " + qdata["Product B Name"].str[:20]
    )
    median_j = qdata["jaccard"].median()
    median_k = qdata["kulczynski"].median()

    fig = px.scatter(
        qdata,
        x="jaccard",
        y="kulczynski",
        size="support",
        color="phi_coefficient",
        color_continuous_scale="RdYlGn",
        hover_name="label",
        hover_data=["lift", "support", "confidence_a_to_b", "confidence_b_to_a"],
        labels={
            "jaccard": "Jaccard (breadth of overlap)",
            "kulczynski": "Kulczynski (average directional confidence)",
            "phi_coefficient": "Phi coefficient",
        },
        title="Quadrant Map of Complementarity",
    )
    fig.add_vline(
        x=median_j,
        line_dash="dash",
        line_color="gray",
        annotation_text="Median Jaccard",
    )
    fig.add_hline(
        y=median_k,
        line_dash="dash",
        line_color="gray",
        annotation_text="Median Kulczynski",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_heatmap_tab(
    affinity_matrix: pd.DataFrame, product_lookup: dict, top_n_products: int
):
    """Render lift heatmap."""
    st.subheader("Affinity Matrix Heatmap")

    if affinity_matrix.empty:
        st.info("No affinity matrix available")
        return

    labels = [
        product_lookup.get(col, col) if product_lookup else col
        for col in affinity_matrix.columns
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=affinity_matrix.values,
            x=labels,
            y=labels,
            colorscale="RdYlGn",
            zmid=1.0,
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title=f"Lift-based Affinity Matrix (Top {top_n_products} Products)",
        height=700,
        xaxis_title="Product",
        yaxis_title="Product",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_profile_tab(
    transactions_df: pd.DataFrame,
    top_pairs: pd.DataFrame,
    product_lookup: dict,
    min_lift: float,
):
    """Render single-product affinity profile."""
    st.subheader("Affinity Profile for a Single Product")

    products = sorted(
        set(top_pairs["product_a"]).union(set(top_pairs["product_b"]))
    )
    target_product = st.selectbox(
        "Select Product",
        options=products,
        format_func=lambda x: product_lookup.get(x, x) if product_lookup else x,
        key="copurchase_target_product",
    )

    if not target_product:
        return

    profile = _cached_get_product_affinity_profile(
        transactions_df, target_product, min_lift, 20
    )

    if profile.empty:
        st.info("No affinity profile found for this product")
        return

    profile["Co-purchase Name"] = profile["co_purchase_product"].map(product_lookup)
    st.dataframe(profile.round(4), width="stretch", hide_index=True)

    fig = px.bar(
        profile,
        x="lift",
        y="Co-purchase Name",
        orientation="h",
        color="kulczynski",
        color_continuous_scale="Blues",
        title=f"Affinity Profile for {product_lookup.get(target_product, target_product)}",
        labels={"lift": "Lift", "Co-purchase Name": "Co-purchase Product"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
