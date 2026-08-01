"""Co-purchase / Affinity analysis tab."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.copurchase import (
    compute_affinity_matrix,
    get_product_affinity_profile,
    get_top_affinity_pairs,
)
from src.analytics.sufficiency import assess_data_sufficiency, format_sufficiency_summary
from src.analytics.basket_metrics import compute_basket_penetration
from src.ui.export import render_analytics_export
from src.ui.tabs import persistent_tabs
from src.ui.insight_header import render_result_context
from src.ui.data_quality import render_data_quality_expander


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
        transactions_df,
        target_product=target_product,
        min_lift=min_lift,
        top_n=top_n,
    )


def render_copurchase_tab(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict, pipeline: dict = None
):
    """Render co-purchase/affinity analysis tab with persistent sub-tabs."""
    st.header("🛒 Co-purchase / Affinity Analysis")
    st.caption(
        "Measures how often products are bought **in the same basket**. "
        "Lift > 1 = complementary pair. Jaccard and Kulczynski add more robust, "
        "symmetric evidence for scientifically stronger pair ranking. "
        "**Associative only — not causal incrementality.**"
    )

    if transactions_df.empty:
        st.warning("No transaction data available")
        return

    # Data quality & readiness at top
    render_data_quality_expander(transactions_df, "copurchase", params, expanded=False)

    # Compute basket penetration for enrichment
    basket_pen = None
    with st.spinner("Computing basket penetration..."):
        basket_pen = compute_basket_penetration(transactions_df)

    with st.expander("Affinity Parameters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            min_support = st.number_input(
                "Min Support", 0.001, 1.0, params.get("min_support", 0.005), 0.001
            )
        with col2:
            min_lift = st.number_input("Min Lift", 0.1, 20.0, params.get("min_lift", 1.2), 0.1)
        with col3:
            top_n_products = st.slider("Top N Products", 10, 100, params.get("top_n_products", 30))

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

    # Unstable result warning
    min_supp_in_filtered = top_pairs["support"].min()
    if min_supp_in_filtered < 0.001:
        st.warning(
            f"⚠️ **Unstable results**: Minimum support in top pairs is {min_supp_in_filtered:.5f}. "
            "Pairs with very low support (<0.1%) are statistically unreliable."
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pairs Found", len(top_pairs))
    col2.metric("Max Lift", f"{top_pairs['lift'].max():.2f}")
    col3.metric("Avg Jaccard", f"{top_pairs['jaccard'].mean():.3f}")
    col4.metric("Avg Kulczynski", f"{top_pairs['kulczynski'].mean():.3f}")

    # Enrich with basket penetration
    top_pairs = _enrich_pairs_with_penetration(top_pairs, basket_pen, product_lookup)

    top_pairs["Product A Name"] = top_pairs["product_a"].map(product_lookup)
    top_pairs["Product B Name"] = top_pairs["product_b"].map(product_lookup)

    tab_labels = [
        "Top Pairs",
        "Quadrant View",
        "Heatmap",
        "Product Profile",
        "Basket Segments",
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
    elif active_tab == 4:
        _render_basket_segment_tab(transactions_df, product_lookup)


def _render_top_pairs_tab(top_pairs: pd.DataFrame):
    """Render top co-purchase pairs table and classical scatter."""
    st.subheader("Top Co-purchase Pairs — Ranked Evidence")

    # Insight header for top pair
    if not top_pairs.empty:
        best = top_pairs.nlargest(1, "lift").iloc[0]
        name_a = best.get("Product A Name", best["product_a"])
        name_b = best.get("Product B Name", best["product_b"])

        evidence_parts = [
            f"Lift: {best['lift']:.2f}",
            f"Jaccard: {best['jaccard']:.3f}",
            f"Kulczynski: {best['kulczynski']:.3f}",
            f"Support: {best['support']:.4f}",
        ]
        if "basket_penetration_a" in best:
            evidence_parts.append(f"Basket Pen A: {best['basket_penetration_a']:.2%}")
        if "basket_penetration_b" in best:
            evidence_parts.append(f"Basket Pen B: {best['basket_penetration_b']:.2%}")

        render_result_context(
            title="Top Co-purchase Pair",
            finding=f"`{name_a}` + `{name_b}` co-occur {best['lift']:.1f}x more than expected — strong bundle candidate",
            evidence=" | ".join(evidence_parts),
            confidence="Directional",
            limitation="Associative only — co-occurrence does not imply causation or incrementality. No control for trip type, seasonality, or promotions.",
        )

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
    # Add penetration columns if available
    pen_cols = [c for c in top_pairs.columns if "basket_penetration" in c or "shopper_penetration" in c]
    display_cols.extend(pen_cols)
    available = [c for c in display_cols if c in top_pairs.columns]

    st.dataframe(top_pairs[available].round(4), width="stretch", hide_index=True)

    # Disclaimer
    st.caption(
        "⚠️ **Interpretation**: Lift > 1 indicates co-occurrence above chance. "
        "Jaccard/Kulczynski are symmetric affinity measures. "
        "These are **associative** metrics from observational basket data. "
        "They do not prove causation, incrementality, or that bundling will increase sales."
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


def _enrich_pairs_with_penetration(top_pairs: pd.DataFrame, basket_pen: pd.DataFrame, product_lookup: dict) -> pd.DataFrame:
    """Add basket penetration metrics to pairs display."""
    if basket_pen is None or basket_pen.empty:
        return top_pairs

    pen_lookup = basket_pen.set_index("stockcode")

    def get_pen(stockcode, col):
        try:
            return pen_lookup.loc[stockcode, col]
        except (KeyError, IndexError):
            return np.nan

    top_pairs = top_pairs.copy()
    for side in ["a", "b"]:
        col_name = f"product_{side}"
        top_pairs[f"basket_penetration_{side}"] = top_pairs[col_name].apply(lambda x: get_pen(x, "basket_penetration"))
        top_pairs[f"unique_shopper_penetration_{side}"] = top_pairs[col_name].apply(lambda x: get_pen(x, "unique_shopper_penetration"))

    return top_pairs


def _render_quadrant_tab(top_pairs: pd.DataFrame):
    """Render academic quadrant view using Jaccard and Kulczynski."""
    st.subheader("Quadrant View: Breadth vs Strength")
    st.caption(
        "Jaccard measures breadth of overlap; Kulczynski averages both directional confidences. "
        "Top-right pairs are typically the strongest candidates for bundling or adjacency."
    )

    qdata = top_pairs.copy()
    qdata["label"] = qdata["Product A Name"].str[:20] + " + " + qdata["Product B Name"].str[:20]
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


def _render_heatmap_tab(affinity_matrix: pd.DataFrame, product_lookup: dict, top_n_products: int):
    """Render lift heatmap."""
    st.subheader("Affinity Matrix Heatmap")
    st.caption("Symmetric lift matrix. **Associative only — not causal.** Red/Green = above/below independence.")

    if affinity_matrix.empty:
        st.info("No affinity matrix available")
        return

    labels = [
        product_lookup.get(col, col) if product_lookup else col for col in affinity_matrix.columns
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

    products = sorted(set(top_pairs["product_a"]).union(set(top_pairs["product_b"])))
    target_product = st.selectbox(
        "Select Product",
        options=products,
        format_func=lambda x: product_lookup.get(x, x) if product_lookup else x,
        key="copurchase_target_product",
    )

    if not target_product:
        return

    profile = _cached_get_product_affinity_profile(transactions_df, target_product, min_lift, 20)

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


def _render_basket_segment_tab(transactions_df: pd.DataFrame, product_lookup: dict):
    """Render basket-size segment analysis for co-purchase context."""
    st.subheader("Basket-Size Segment Analysis")
    st.caption(
        "Baskets segmented by number of distinct SKUs: Small (1-2), Medium (3-7), Large (8+). "
        "Shows how co-purchase patterns vary by basket size."
    )

    from src.analytics.basket_metrics import compute_basket_size_segments, get_basket_segment_for_product

    with st.spinner("Computing basket segments..."):
        basket_segments = compute_basket_size_segments(transactions_df)
        segment_profile = compute_basket_segment_profile(transactions_df)

    # Segment overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Small Baskets (1-2 SKUs)", f"{basket_segments[basket_segments['basket_segment']=='Small']['_basket'].count():,}")
    with col2:
        st.metric("Medium Baskets (3-7 SKUs)", f"{basket_segments[basket_segments['basket_segment']=='Medium']['_basket'].count():,}")
    with col3:
        st.metric("Large Baskets (8+ SKUs)", f"{basket_segments[basket_segments['basket_segment']=='Large']['_basket'].count():,}")

    # Segment profile
    st.subheader("Segment Profile")
    st.dataframe(
        segment_profile.style.format({
            "n_baskets": "{:,}",
            "n_customers": "{:,}",
            "total_revenue": "${:,.0f}",
            "avg_basket_value": "${:.2f}",
            "avg_basket_depth": "{:.1f}",
            "avg_basket_units": "{:.1f}",
            "pct_baskets": "{:.1f}%",
            "pct_revenue": "{:.1f}%",
            "pct_customers": "{:.1f}%",
        }).background_gradient(cmap="RdYlGn", subset=["pct_baskets", "pct_revenue"]),
        use_container_width=True,
    )

    # Product-level segment distribution
    st.subheader("Product-Level Segment Distribution")
    st.caption("Shows which basket segments each product appears in.")

    with st.spinner("Computing product-level segment distribution..."):
        from src.analytics.basket_metrics import get_basket_segment_for_product
        product_segments = get_basket_segment_for_product(transactions_df)

    if not product_segments.empty:
        product_segments["Product Name"] = product_segments["stockcode"].map(product_lookup)
        
        # Pivot for heatmap
        pivot = product_segments.pivot_table(
            index="Product Name",
            columns="basket_segment",
            values="pct_baskets",
            fill_value=0,
        )
        
        fig = px.imshow(
            pivot.values,
            x=pivot.columns,
            y=pivot.index,
            color_continuous_scale="RdYlGn",
            labels={"x": "Basket Segment", "y": "Product", "color": "% of Baskets"},
            title="Product Presence by Basket Segment (%)",
            aspect="auto",
        )
        fig.update_layout(height=max(400, len(pivot) * 20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No product-level segment data available.")
