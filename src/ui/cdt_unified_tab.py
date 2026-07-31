"""Unified CDT UI Tab - merges cdt_tab.py and cdt_assortment_tab.py"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import build_customer_sequences
from src.analytics.sufficiency import assess_data_sufficiency, format_sufficiency_summary
from src.application.cdt_service import CDTConfig, get_cdt_service
from src.ui.export import render_analytics_export
from src.viz.cdt_viz import (
    plot_behavioral_heatmap,
    plot_dendrogram,
    plot_silhouette_scores,
    plot_similarity_heatmap,
    plot_sunburst,
    plot_treemap,
)

# CDT result tab labels
_CDT_TABS = [
    "🌞 CDT Sunburst",
    "📦 CDT Treemap",
    "🌲 Dendrogram & Clusters",
    "🔥 Similarity Heatmap",
    "🔄 Switching Analysis",
    "🔄 Substitution Analysis",
    "🎁 Bundling Opportunities",
    "📊 CDT Benchmark",
    "📥 Export",
]


def render_cdt_tab(
    transactions_df: pd.DataFrame,
    product_lookup: dict,
    params: dict,
    mode: str = "cdt",
):
    """Main entry point for unified CDT tab with sub-modes."""

    # Data sufficiency gate (applies to all modes)
    if not transactions_df.empty:
        sufficiency = assess_data_sufficiency(transactions_df)
        with st.expander("📋 Data Sufficiency", expanded=sufficiency["overall"] != "robust"):
            st.markdown(format_sufficiency_summary(sufficiency))
            if sufficiency["overall"] == "insufficient":
                st.warning("Dataset may be too small for reliable analysis.")
            elif sufficiency["overall"] == "directional":
                st.info("Results should be treated as directional.")

    # Invalidate cached results when the data changes
    _invalidate_data_cache_if_changed(transactions_df)

    if mode == "cdt":
        _render_cdt_builder(transactions_df, product_lookup, params)
    elif mode == "transference":
        _render_demand_transference(transactions_df, product_lookup, params)
    elif mode == "assortment":
        _render_assortment_optimizer(transactions_df, product_lookup, params)
    else:
        st.warning(f"Unknown mode: {mode}")


def _invalidate_data_cache_if_changed(transactions_df: pd.DataFrame) -> None:
    """
    Invalidate cached results when the underlying transaction data changes.
    """
    if transactions_df.empty:
        return

    # Create a fingerprint of the current dataset
    data_fingerprint = hash((
        len(transactions_df),
        transactions_df["transaction_id"].nunique() if "transaction_id" in transactions_df.columns else 0,
        transactions_df["customer_id"].nunique() if "customer_id" in transactions_df.columns else 0,
        transactions_df["stockcode"].nunique() if "stockcode" in transactions_df.columns else 0,
        str(transactions_df["date"].min()) if "date" in transactions_df.columns else "",
        str(transactions_df["date"].max()) if "date" in transactions_df.columns else "",
    ))

    # Check if fingerprint matches cached one
    cached_fingerprint = st.session_state.get("cdt_unified_data_fingerprint")
    if cached_fingerprint is not None and cached_fingerprint != data_fingerprint:
        # Data changed - clear all CDT session state
        keys_to_delete = [k for k in st.session_state if k.startswith("cdt_unified_")]
        for key in keys_to_delete:
            del st.session_state[key]

    # Store/update fingerprint
    st.session_state["cdt_unified_data_fingerprint"] = data_fingerprint


# ============================================================================
# CDT BUILDER MODE
# ============================================================================


def _render_cdt_builder(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render CDT Builder with similarity methods, community detection, and tree building."""

    st.header("🌳 Customer Decision Tree (CDT) Builder")

    # Build CDT config from params
    cdt_config = CDTConfig(
        similarity_methods=params.get("similarity_methods", ["phi"]),
        min_cooccurrence=params.get("min_cooccurrence", 5),
        community_method=params.get("community_method", "label_propagation"),
        community_resolution=params.get("community_resolution", 1.0),
        graph_min_weight=params.get("graph_min_weight", 0.1),
        graph_max_degree=params.get("graph_max_degree", 50),
        linkage_method=params.get("linkage_method", "average"),
        min_k=params.get("min_k", 2),
        max_k=params.get("max_k", 15),
        min_cluster_size=params.get("min_cluster_size", 3),
        quality_threshold=params.get("quality_threshold", 60) / 100,
        split_criterion=params.get("split_criterion", "mutual_info"),
        split_alpha=params.get("split_alpha", 0.5),
        top_n_products=params.get("top_n_products", 50),
        min_lift=params.get("min_lift", 1.2),
        max_sub=params.get("max_sub", 0.3),
    )

    # Check pipeline for cached results
    has_results = "cdt_unified_root" in st.session_state

    if has_results:
        root = st.session_state["cdt_unified_root"]
        metadata = st.session_state["cdt_unified_metadata"]
        similarity_matrix = st.session_state["cdt_unified_similarity_matrix"]
        switching_df = st.session_state["cdt_unified_switching_df"]
        substitution_df = st.session_state["cdt_unified_substitution_df"]
        bundling_df = st.session_state["cdt_unified_bundling_df"]
        linkage_matrix = st.session_state["cdt_unified_linkage_matrix"]
        ordered_labels = st.session_state["cdt_unified_ordered_labels"]
        silhouette_scores = st.session_state["cdt_unified_silhouette_scores"]
        optimal_k = st.session_state["cdt_unified_optimal_k"]
        similarity_method = st.session_state.get("cdt_unified_similarity_method", "phi")
        product_lookup = st.session_state.get("cdt_unified_product_lookup", {})

        if st.session_state.pop("cdt_unified_just_built", False):
            n_products = len(similarity_matrix)
            st.toast(
                f"✅ CDT built for {n_products} products. "
                "Each panel renders only when you open it.",
                icon="📊",
            )

        col_rebuild, _ = st.columns([1, 5])
        with col_rebuild:
            if st.button(" Reconfigure & Rebuild", type="secondary"):
                for key in list(st.session_state.keys()):
                    if key.startswith("cdt_unified_"):
                        del st.session_state[key]
                st.rerun()

        _render_cdt_results_tabs(
            root,
            metadata,
            similarity_matrix,
            switching_df,
            substitution_df,
            bundling_df,
            linkage_matrix,
            ordered_labels,
            silhouette_scores,
            optimal_k,
            product_lookup,
            similarity_method,
        )
        return

    _render_cdt_config_panel(transactions_df, product_lookup, params, cdt_config)


def _render_cdt_config_panel(
    transactions_df: pd.DataFrame,
    product_lookup: dict,
    params: dict,
    cdt_config: CDTConfig,
):
    """Render the CDT configuration panel and run pipeline when button clicked."""

    with st.expander(" CDT Configuration", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Similarity")
            similarity_mode = st.selectbox(
                "Mode",
                ["Fast (single method)", "Ensemble (weighted)"],
                index=0,
                help="Fast = single similarity method. "
                "Ensemble = weighted combination of Phi, Jaccard, PMI, Cosine TF-IDF.",
            )
            if similarity_mode == "Fast (single method)":
                similarity_method = st.selectbox(
                    "Similarity Method",
                    ["phi", "jaccard"],
                    index=0,
                    help="Phi coefficient ([-1,1]) or Jaccard ([0,1])",
                )
                cdt_config.similarity_methods = [similarity_method]
            else:
                cdt_config.similarity_methods = st.multiselect(
                    "Similarity Methods",
                    ["phi", "jaccard", "pmi", "cosine_tfidf"],
                    default=["phi"],
                    help="Methods to include in ensemble",
                )

            cdt_config.min_cooccurrence = st.slider(
                "Min Co-occurrence",
                2,
                20,
                cdt_config.min_cooccurrence,
                help="Min customers buying both products",
            )

        with col2:
            st.subheader("Community Detection")
            cdt_config.community_method = st.selectbox(
                "Community Method",
                ["none", "label_propagation", "louvain", "leiden"],
                index=1,
                help="Run label-propagation community detection before clustering. "
                "Useful for large product sets with natural sub-groups.",
            )
            cdt_config.community_resolution = st.slider(
                "Resolution", 0.5, 2.0, cdt_config.community_resolution, 0.1
            )
            cdt_config.graph_min_weight = st.slider(
                "Graph Min Weight", 0.0, 0.5, cdt_config.graph_min_weight, 0.05
            )
            cdt_config.graph_max_degree = st.slider(
                "Graph Max Degree", 10, 100, cdt_config.graph_max_degree
            )

        with col3:
            st.subheader("Clustering")
            cdt_config.linkage_method = st.selectbox(
                "Linkage Method", ["average", "complete", "single"],
                index=0, help="Average = default"
            )
            cdt_config.min_k = st.slider("Min Clusters (k)", 2, 10, cdt_config.min_k)
            cdt_config.max_k = st.slider("Max Clusters (k)", 3, 20, cdt_config.max_k)

        with col4:
            st.subheader("Tree Building")
            cdt_config.min_cluster_size = st.slider(
                "Min Cluster Size",
                2, 10, cdt_config.min_cluster_size,
                help="Min products per tree node",
            )
            cdt_config.quality_threshold = (
                st.slider(
                    "Quality Threshold (%)",
                    40, 80, int(cdt_config.quality_threshold * 100),
                    help="Tree quality vs unconstrained baseline (default: 60%)",
                )
                / 100.0
            )
            cdt_config.split_criterion = st.selectbox(
                "Split Criterion",
                ["mutual_info", "gini", "entropy", "mixed"],
                index=0,
                help="Attribute split scoring method",
            )
            cdt_config.split_alpha = st.slider(
                "Split Alpha (entropy/Gini mix)",
                0.0, 1.0, cdt_config.split_alpha, 0.1
            )

    # Attribute columns
    attribute_cols = detect_attribute_columns(transactions_df)
    if attribute_cols:
        st.info(f" Detected attribute columns: {', '.join(attribute_cols)}")
        st.multiselect(
            "Attributes for Tree Enrichment",
            attribute_cols,
            default=attribute_cols,
            help="These will be tested as split criteria",
        )
    else:
        st.warning(
            "No attribute columns detected (category, brand, size, flavor, etc.). "
            "Tree will be built from similarity only."
        )

    # Category filter
    if "category" in transactions_df.columns:
        categories = ["All"] + sorted(transactions_df["category"].unique().tolist())
        selected_category = st.selectbox("Filter by Category", categories)
        if selected_category != "All":
            transactions_df = transactions_df[
                transactions_df["category"] == selected_category
            ].copy()
            st.info(
                f"Filtered to category: {selected_category} ({len(transactions_df)} transactions)"
            )

    # Behavioral parameters
    with st.expander("**Behavioral**", expanded=False):
        cdt_config.top_n_products = st.slider(
            "Top N Products", 20, 200, cdt_config.top_n_products
        )
        cdt_config.min_lift = st.slider(
            "Min Lift", 1.0, 3.0, cdt_config.min_lift, 0.1
        )
        cdt_config.max_sub = st.slider(
            "Max Substitution", 0.0, 0.5, cdt_config.max_sub, 0.05
        )

    # Expected duration hint
    n_rows = len(transactions_df)
    n_products_est = (
        transactions_df["stockcode"].nunique() if "stockcode" in transactions_df.columns else 0
    )
    if n_products_est > 100 or n_rows > 50_000:
        wait_hint = "large dataset — pipeline may take **1–3 minutes**"
    elif n_products_est > 50 or n_rows > 10_000:
        wait_hint = "medium dataset — pipeline typically takes **20–60 seconds**"
    else:
        wait_hint = "small dataset — pipeline typically completes in **< 20 seconds**"
    st.info(f"⏱️ {wait_hint}. Each visualisation tab renders on demand after completion.")

    run_button = st.button(
        " Build Customer Decision Tree", type="primary", use_container_width=True
    )

    if not run_button:
        st.info("Configure parameters above and click **Build Customer Decision Tree** to start.")
        return

    # Execute CDT pipeline
    progress_bar = st.progress(0)
    status_text = st.empty()

    cdt_service = get_cdt_service()

    try:
        status_text.info("⏳ **Step 1 / 5** — Building customer purchase sequences…")
        progress_bar.progress(10)

        status_text.info("⏳ **Step 2 / 5** — Computing similarity matrix…")
        progress_bar.progress(25)

        status_text.info("⏳ **Step 3 / 5** — Performing hierarchical clustering…")
        progress_bar.progress(40)

        status_text.info("⏳ **Step 4 / 5** — Building Customer Decision Tree…")
        progress_bar.progress(55)

        status_text.info("⏳ **Step 5 / 5** — Computing behavioral matrices…")
        progress_bar.progress(70)

        # Execute CDT pipeline
        cdt_service = get_cdt_service()
        result = cdt_service.execute_cdt(
            transactions_df=transactions_df,
            product_lookup=product_lookup,
            cdt_config=cdt_config,
        )

        if not result["success"]:
            st.error(f"CDT pipeline failed: {result.get('error', 'Unknown error')}")
            return

        # Store results in session state
        st.session_state["cdt_unified_root"] = result["tree_root"]
        st.session_state["cdt_unified_metadata"] = result["tree_metadata"]
        st.session_state["cdt_unified_similarity_matrix"] = result["similarity_matrix"]
        st.session_state["cdt_unified_linkage_matrix"] = result["linkage_matrix"]
        st.session_state["cdt_unified_ordered_labels"] = result["ordered_labels"]
        st.session_state["cdt_unified_cluster_assignments"] = result["cluster_assignments"]
        st.session_state["cdt_unified_silhouette_scores"] = result["silhouette_scores"]
        st.session_state["cdt_unified_optimal_k"] = result["optimal_k"]
        st.session_state["cdt_unified_similarity_method"] = cdt_config.similarity_methods[0]
        st.session_state["cdt_unified_product_lookup"] = product_lookup
        st.session_state["cdt_unified_just_built"] = True

        progress_bar.progress(100)
        status_text.success(
            "✅ **Complete!** Select a tab below to explore the results. "
            "Each panel renders only when you open it."
        )
        st.rerun()

    except Exception as e:
        st.error(f"Pipeline failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return


# ============================================================================
# Results: LAZY tab rendering
# ============================================================================


def _render_cdt_results_tabs(
    root,
    metadata: dict,
    similarity_matrix: pd.DataFrame,
    switching_df: pd.DataFrame,
    substitution_df: pd.DataFrame,
    bundling_df: pd.DataFrame,
    linkage_matrix: np.ndarray,
    ordered_labels: list,
    silhouette_scores: dict,
    optimal_k: int,
    product_lookup: dict,
    similarity_method: str = "phi",
):
    """Render CDT results with lazy per-tab rendering."""
    render_quality_summary(metadata)

    st.markdown("#### Explore Results")
    btn_cols = st.columns(len(_CDT_TABS))
    active = st.session_state.get("cdt_unified_active_tab", 0)
    for i, (col, label) in enumerate(zip(btn_cols, _CDT_TABS)):
        btn_type = "primary" if i == active else "secondary"
        if col.button(label, key=f"cdt_unified_tab_btn_{i}", type=btn_type, use_container_width=True):
            st.session_state["cdt_unified_active_tab"] = i
            active = i
            st.rerun()

    st.divider()

    if active == 0:
        _tab_sunburst(root, metadata)
    elif active == 1:
        _tab_treemap(root)
    elif active == 2:
        _tab_dendrogram(linkage_matrix, ordered_labels, silhouette_scores, optimal_k)
    elif active == 3:
        _tab_similarity(similarity_matrix, similarity_method)
    elif active == 4:
        _tab_switching(switching_df, product_lookup)
    elif active == 5:
        _tab_substitution(substitution_df, product_lookup)
    elif active == 6:
        _tab_bundling(bundling_df, product_lookup)
    elif active == 7:
        _tab_cdt_benchmark()
    elif active == 8:
        _tab_export(root, switching_df, substitution_df, bundling_df)


# ============================================================================
# Individual lazy tab renderers
# ============================================================================


def _tab_sunburst(root, metadata: dict):
    st.subheader("Customer Decision Tree — Sunburst View")
    st.caption(
        "Hierarchical tree from bottom-up clustering. "
        "Inner rings = higher-level splits. Outer rings = product leaves."
    )
    with st.spinner("Rendering Sunburst chart — please wait…"):
        fig = plot_sunburst(
            root,
            title=f"CDT: {metadata['n_leaves']} leaf clusters, {metadata['max_depth']} levels",
            height=700,
        )
    st.plotly_chart(fig, use_container_width=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Nodes", metadata["n_nodes"])
    col2.metric("Leaf Clusters", metadata["n_leaves"])
    col3.metric("Max Depth", metadata["max_depth"])


def _tab_treemap(root):
    st.subheader("Customer Decision Tree — Treemap View")
    st.caption("Area proportional to number of products. Color = split attribute.")
    size_metric = st.radio("Size Metric", ["size", "similarity_within"], horizontal=True)
    with st.spinner("Rendering Treemap — please wait…"):
        fig = plot_treemap(root, size_metric=size_metric, height=700)
    st.plotly_chart(fig, use_container_width=True)


def _tab_dendrogram(
    linkage_matrix: np.ndarray,
    ordered_labels: list,
    silhouette_scores: dict,
    optimal_k: int,
):
    st.subheader("Hierarchical Clustering Dendrogram")
    col1, col2 = st.columns([2, 1])
    with col1:
        with st.spinner("Rendering dendrogram — please wait…"):
            fig = plot_dendrogram(linkage_matrix, ordered_labels, height=600)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Cluster Quality")
        st.metric("Optimal k (Silhouette)", optimal_k)
        st.metric("Silhouette Score", f"{silhouette_scores.get(optimal_k, 0):.3f}")
        with st.spinner("Rendering silhouette chart…"):
            fig_sil = plot_silhouette_scores(silhouette_scores, optimal_k, height=300)
        st.plotly_chart(fig_sil, use_container_width=True)


def _tab_similarity(similarity_matrix: pd.DataFrame, similarity_method: str):
    st.subheader("Product Similarity Matrix")
    st.caption(
        f"Method: {similarity_method.upper()}. "
        "Red=dissimilar, Blue=similar / substitutable."
    )
    top_n = st.slider(
        "Top N Products",
        10,
        min(100, len(similarity_matrix)),
        min(50, len(similarity_matrix)),
        key="cdt_sim_top_n",
    )
    with st.spinner("Rendering similarity heatmap — please wait…"):
        fig = plot_similarity_heatmap(similarity_matrix, top_n=top_n, height=600)
    st.plotly_chart(fig, use_container_width=True)


def _tab_switching(switching_df: pd.DataFrame, product_lookup: dict):
    st.subheader("Switching Analysis")
    st.caption(
        "Product-to-product switching rates from customer purchase sequences. "
        "Each cell shows the rate at which customers who bought the **row product** "
        "next purchased the **column product**."
    )
    if switching_df.empty:
        st.info(
            "No switching data available. "
            "Need customers with repeat purchases across different products."
        )
        return

    # Diagnostic metrics
    n_pairs = len(switching_df)
    max_rate = switching_df["switch_rate"].max()
    st.info(
        f"📊 **{n_pairs:,}** switching pairs detected — "
        f"max switch rate: **{max_rate:.1%}**"
    )

    top_n_heatmap = st.slider(
        "Top N products in heatmap",
        5,
        50,
        min(30, len(switching_df["from_product"].unique())),
        key="cdt_switch_top_n",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        with st.spinner("Rendering switching heatmap — please wait…"):
            from src.analytics.cdt_behavioral import switching_matrix_to_heatmap
            switch_matrix = switching_matrix_to_heatmap(switching_df, top_n=top_n_heatmap)

        if switch_matrix.empty:
            st.warning(
                "Switching matrix is empty after filtering. "
                "Try lowering **Top N products in heatmap** or "
                "rebuilding CDT with a lower Min Co-occurrence."
            )
        else:
            fig_heatmap = plot_behavioral_heatmap(
                switch_matrix,
                title="Switching Rate Matrix",
                height=500,
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

    with col2:
        st.write("**Top Switching Paths**")
        from src.analytics.cdt_behavioral import get_top_switching_paths
        top_switches = get_top_switching_paths(switching_df, top_n=15)
        if not top_switches.empty:
            display_df = top_switches.copy()
            display_df["from"] = display_df["from_product"].map(
                lambda x: product_lookup.get(x, x)[:30]
            )
            display_df["to"] = display_df["to_product"].map(lambda x: product_lookup.get(x, x)[:30])
            st.dataframe(
                display_df[["from", "to", "switch_count", "switch_rate"]].round(4),
                hide_index=True,
                use_container_width=True,
            )

    st.subheader("Switching Network")
    st.caption(
        "Nodes = products. Edge width and opacity = switching rate. "
        "Only edges above the minimum rate threshold are shown; "
        "if none pass the threshold the top-50 by rate are shown instead."
    )
    min_rate_input = st.slider(
        "Min switch rate for network edges",
        0.00, 0.20, 0.05, 0.01,
        key="cdt_net_min_rate",
    )
    with st.spinner("Rendering network graph — please wait…"):
        from src.viz.cdt_viz import plot_switching_network
        fig_net = plot_switching_network(switching_df, product_lookup, min_rate=min_rate_input)
    st.plotly_chart(fig_net, use_container_width=True)
    render_analytics_export(switching_df, "CDT_Switching")


def _tab_substitution(substitution_df: pd.DataFrame, product_lookup: dict):
    st.subheader("Substitution Analysis")
    st.caption(
        "High similarity = high substitutability. "
        "Derived from co-purchase patterns (Phi coefficient)."
    )
    if substitution_df.empty:
        st.info("No substitution data available.")
        return

    top_n = st.slider("Top N Products", 10, 100, 50, key="cdt_sub_top_n")
    with st.spinner("Rendering substitution heatmap — please wait…"):
        fig = plot_similarity_heatmap(
            substitution_df,
            top_n=top_n,
            title="Substitution Score Matrix",
            height=500,
        )
    st.plotly_chart(fig, use_container_width=True)

    st.write("**Top Substitutable Pairs**")
    from src.analytics.cdt_behavioral import get_top_substitution_pairs
    top_subs = get_top_substitution_pairs(substitution_df, top_n=20)
    if not top_subs.empty:
        display_df = top_subs.copy()
        display_df["Product A"] = display_df["product_a"].map(
            lambda x: product_lookup.get(x, x)[:30]
        )
        display_df["Product B"] = display_df["product_b"].map(
            lambda x: product_lookup.get(x, x)[:30]
        )
        st.dataframe(
            display_df[["Product A", "Product B", "substitution_score"]].round(4),
            hide_index=True,
            use_container_width=True,
        )
        render_analytics_export(top_subs, "CDT_Substitution")


def _tab_bundling(bundling_df: pd.DataFrame, product_lookup: dict):
    st.subheader("Bundling Opportunities")
    st.caption("High lift + low substitution = true complements.")
    if bundling_df.empty:
        st.info("No bundling data available.")
        return

    st.write("**Top Bundling Pairs**")
    from src.analytics.cdt_behavioral import get_top_bundling_pairs
    top_bundles = get_top_bundling_pairs(bundling_df, top_n=20)
    if not top_bundles.empty:
        display_df = top_bundles.copy()
        display_df["Product A"] = display_df["product_a"].map(
            lambda x: product_lookup.get(x, x)[:30]
        )
        display_df["Product B"] = display_df["product_b"].map(
            lambda x: product_lookup.get(x, x)[:30]
        )
        st.dataframe(
            display_df[["Product A", "Product B", "lift", "substitution", "bundle_score"]].round(4),
            hide_index=True,
            use_container_width=True,
        )
        render_analytics_export(top_bundles, "CDT_Bundling")

    st.subheader("Lift vs Substitution Tradeoff")
    with st.spinner("Rendering bundle scatter — please wait…"):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=bundling_df["substitution"],
                y=bundling_df["lift"],
                mode="markers",
                marker={
                    "size": 8,
                    "color": bundling_df["bundle_score"],
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "Bundle Score"},
                },
                text=[
                    f"{product_lookup.get(a, a)} × {product_lookup.get(b, b)}"
                    for a, b in zip(
                        bundling_df["product_a"],
                        bundling_df["product_b"],
                        strict=False,
                    )
                ],
                hovertemplate=(
                    "%{text}<br>Substitution: %{x:.3f}<br>Lift: %{y:.3f}<extra></extra>"
                ),
            )
        )
        fig.add_vline(
            x=0.3,
            line_dash="dash",
            line_color="red",
            annotation_text="Max Sub for Bundling",
        )
        fig.add_hline(y=1.2, line_dash="dash", line_color="green", annotation_text="Min Lift")
        fig.update_layout(
            title="Bundling Sweet Spot: High Lift + Low Substitution",
            xaxis_title="Substitution Score",
            yaxis_title="Lift",
            height=500,
            plot_bgcolor="white",
        )
    st.plotly_chart(fig, use_container_width=True)


def _tab_cdt_benchmark():
    """CDT Benchmark tab comparing similarity methods."""
    st.subheader("📊 CDT Method Benchmark")

    st.markdown(
        "Compare how well different similarity methods recover known cluster "
        "structure from **synthetic data with ground-truth labels**."
    )

    col1, col2 = st.columns(2)
    with col1:
        bench_n_products = st.slider("Products", 10, 80, 30, key="cdt_bench_n_prod")
        bench_n_clusters = st.slider("True Clusters", 2, 6, 3, key="cdt_bench_n_clust")
    with col2:
        bench_n_customers = st.slider("Customers", 50, 500, 200, key="cdt_bench_n_cust")
        bench_noise = st.slider("Noise Level", 0.0, 0.5, 0.2, 0.05, key="cdt_bench_noise")

    if st.button("▶️ Run CDT Benchmark", type="primary", key="cdt_bench_run"):
        with st.spinner("Running CDT benchmark against synthetic ground truth…"):
            from src.analytics.cdt_validation import run_cdt_validation
            methods = [
                "legacy_phi",
                "legacy_jaccard",
                "ensemble_phi_jaccard_pmi_tfidf",
            ]
            results = run_cdt_validation(
                n_products=bench_n_products,
                n_true_clusters=bench_n_clusters,
                n_customers=bench_n_customers,
                noise_level=bench_noise,
                methods=methods,
            )

        if results.empty:
            st.warning("Benchmark did not produce results.")
            return

        st.success("Benchmark complete!")

        # ARI / NMI bar chart
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Adjusted Rand Index",
                x=results["method"],
                y=results["adjusted_rand_index"],
                marker_color="royalblue",
            )
        )
        fig.add_trace(
            go.Bar(
                name="Normalized Mutual Info",
                x=results["method"],
                y=results["normalized_mutual_info"],
                marker_color="orange",
            )
        )
        fig.update_layout(
            title="Cluster Recovery: ARI vs NMI",
            yaxis_title="Score",
            barmode="group",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Runtime bar chart
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=results["method"],
                y=results["runtime_seconds"],
                marker_color="seagreen",
            )
        )
        fig2.update_layout(
            title="Runtime Comparison",
            yaxis_title="Seconds",
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Detail table
        st.subheader("Detailed Results")
        display_cols = [
            "method",
            "adjusted_rand_index",
            "normalized_mutual_info",
            "n_clusters_found",
            "n_true_clusters",
            "runtime_seconds",
        ]
        styled = results[display_cols].copy()
        styled.columns = [
            "Method",
            "ARI",
            "NMI",
            "Clusters Found",
            "True Clusters",
            "Runtime (s)",
        ]
        st.dataframe(styled, hide_index=True, use_container_width=True)

        render_analytics_export(results, "CDT_Benchmark")


def _tab_export(
    root,
    switching_df: pd.DataFrame,
    substitution_df: pd.DataFrame,
    bundling_df: pd.DataFrame,
):
    st.subheader("Export Results")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Tree Structure (JSON)**")
        if st.button("Export CDT as JSON"):
            from src.analytics.cdt_tree_builder import tree_to_json
            json_str = tree_to_json(root)
            st.download_button(
                "Download JSON",
                json_str,
                file_name="cdt_tree.json",
                mime="application/json",
            )
    with col2:
        st.write("**Tree Structure (CSV)**")
        if st.button("Export CDT as CSV"):
            from src.analytics.cdt_tree_builder import tree_to_dataframe
            df = tree_to_dataframe(root)
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                file_name="cdt_tree.csv",
                mime="text/csv",
            )

    st.divider()
    st.write("**Behavioral Matrices**")
    export_cols = st.columns(3)
    with export_cols[0]:
        if not switching_df.empty:
            csv = switching_df.to_csv(index=False)
            st.download_button("Switching Matrix", csv, "cdt_switching.csv", "text/csv")
    with export_cols[1]:
        if not substitution_df.empty:
            from src.analytics.cdt_behavioral import get_top_substitution_pairs
            top_subs = get_top_substitution_pairs(substitution_df, top_n=100)
            csv = top_subs.to_csv(index=False)
            st.download_button("Substitution Pairs", csv, "cdt_substitution.csv", "text/csv")
    with export_cols[2]:
        if not bundling_df.empty:
            from src.analytics.cdt_behavioral import get_top_bundling_pairs
            top_bundles = get_top_bundling_pairs(bundling_df, top_n=100)
            csv = top_bundles.to_csv(index=False)
            st.download_button("Bundling Pairs", csv, "cdt_bundling.csv", "text/csv")


# ============================================================================
# DEMAND TRANSFERENCE MODE
# ============================================================================


def _render_demand_transference(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render Demand Transference / Delist Simulation."""
    st.header("🔄 Demand Transference & Delist Simulation")

    # Build CDT pipeline first
    cdt_config = CDTConfig(
        similarity_methods=params.get("similarity_methods", ["phi"]),
        min_cooccurrence=params.get("min_cooccurrence", 5),
        community_method=params.get("community_method", "label_propagation"),
        community_resolution=params.get("community_resolution", 1.0),
        graph_min_weight=params.get("graph_min_weight", 0.1),
        graph_max_degree=params.get("graph_max_degree", 50),
        linkage_method=params.get("linkage_method", "average"),
        min_k=params.get("min_k", 2),
        max_k=params.get("max_k", 15),
        min_cluster_size=params.get("min_cluster_size", 3),
        quality_threshold=params.get("quality_threshold", 60) / 100,
        split_criterion=params.get("split_criterion", "mutual_info"),
        split_alpha=params.get("split_alpha", 0.5),
        top_n_products=params.get("top_n_products", 50),
        min_lift=params.get("min_lift", 1.2),
        max_sub=params.get("max_sub", 0.3),
    )

    with st.spinner("Building CDT pipeline for demand transference..."):
        cdt_service = get_cdt_service()
        result = cdt_service.execute_cdt(
            transactions_df=transactions_df,
            product_lookup=product_lookup,
            cdt_config=cdt_config,
        )

    if not result["success"]:
        st.error(f"CDT pipeline failed: {result.get('error', 'Unknown error')}")
        return

    st.success("CDT pipeline built successfully!")

    # Build demand transference matrix
    from src.analytics.cdt_behavioral import compute_switching_matrix
    from src.analytics.demand_transference import compute_demand_transference_matrix

    with st.spinner("Computing demand transference matrix..."):
        sequences = build_customer_sequences(transactions_df)
        switching_df = compute_switching_matrix(sequences)
        dt_matrix = compute_demand_transference_matrix(
            transactions_df,
            switching_df,
            top_n=params.get("top_n_products", 50),
        )

    if dt_matrix.empty:
        st.warning("No demand transference data available.")
        return

    # Delist Simulator
    st.subheader("🗑️ Delist Impact Simulator")

    all_products = dt_matrix["from_product"].unique().tolist()
    product_names = {p: product_lookup.get(p, p) for p in all_products}

    delist_products = st.multiselect(
        "Select products to delist",
        options=all_products,
        format_func=lambda x: f"{x} - {product_names.get(x, '')}",
        key="dt_delist_products",
    )

    if delist_products:
        # Run delist impact analysis
        from src.analytics.demand_transference import delist_impact_analysis

        impact_df = delist_impact_analysis(
            transactions_df,
            dt_matrix,
            delist_products,
        )

        st.subheader("📊 Delist Impact Results")

        # Summary metrics
        total_lost = impact_df["product_revenue"].sum()
        total_recovered = impact_df["estimated_revenue_recovered"].sum()
        net_impact = total_recovered - total_lost

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Revenue Lost", f"${total_lost:,.2f}")
        with col2:
            st.metric("Revenue Recovered (est.)", f"${total_recovered:,.2f}")
        with col3:
            st.metric(
                "Net Impact",
                f"${net_impact:,.2f}",
                delta_color="normal" if net_impact >= 0 else "inverse",
            )

        # Detail table
        display_df = impact_df.copy()
        display_df["product_name"] = display_df["stockcode"].map(product_names)
        display_cols = [
            "stockcode",
            "product_name",
            "product_revenue",
            "estimated_revenue_recovered",
            "net_revenue_impact",
            "recovery_rate",
        ]
        st.dataframe(display_df[display_cols], use_container_width=True)

        # Waterfall chart
        _render_delist_waterfall(impact_df, product_names)


def _render_delist_waterfall(impact_df: pd.DataFrame, product_names: dict):
    """Render waterfall chart for delist impact."""
    fig = go.Figure()

    fig.add_trace(
        go.Waterfall(
            name="Revenue",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(impact_df) + ["total"],
            x=["Current Revenue"]
            + [
                product_names.get(row["stockcode"], row["stockcode"])
                for _, row in impact_df.iterrows()
            ]
            + ["Net Revenue"],
            textposition="outside",
            y=[impact_df["product_revenue"].sum()]
            + list(-impact_df["product_revenue"])
            + [impact_df["estimated_revenue_recovered"].sum()],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )

    fig.update_layout(title="Delist Revenue Waterfall", height=400)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# ASSORTMENT OPTIMIZER MODE
# ============================================================================


def _render_assortment_optimizer(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render Assortment Optimizer with MILP/Heuristic solver."""

    st.header("🎯 Assortment Optimizer")

    # Build CDT pipeline
    cdt_config = CDTConfig(
        similarity_methods=params.get("similarity_methods", ["phi"]),
        min_cooccurrence=params.get("min_cooccurrence", 5),
        community_method=params.get("community_method", "label_propagation"),
        community_resolution=params.get("community_resolution", 1.0),
        graph_min_weight=params.get("graph_min_weight", 0.1),
        graph_max_degree=params.get("graph_max_degree", 50),
        linkage_method=params.get("linkage_method", "average"),
        min_k=params.get("min_k", 2),
        max_k=params.get("max_k", 15),
        min_cluster_size=params.get("min_cluster_size", 3),
        quality_threshold=params.get("quality_threshold", 60) / 100,
        split_criterion=params.get("split_criterion", "mutual_info"),
        split_alpha=params.get("split_alpha", 0.5),
    )

    with st.spinner("Building CDT pipeline for assortment optimization..."):
        cdt_service = get_cdt_service()
        cdt_result = cdt_service.execute_cdt(
            transactions_df=transactions_df,
            product_lookup=product_lookup,
            cdt_config=cdt_config,
        )

    if not cdt_result["success"]:
        st.error("CDT pipeline failed for assortment optimization")
        return

    # Build required inputs for assortment optimization
    from src.analytics.assortment_opt import (
        generate_assortment_scenarios,
        optimize_assortment_heuristic,
        optimize_assortment_milp,
    )
    from src.analytics.cdt_behavioral import compute_switching_matrix
    from src.analytics.demand_transference import compute_demand_transference_matrix

    with st.spinner("Preparing assortment optimization inputs..."):
        sequences = build_customer_sequences(transactions_df)
        switching_df = compute_switching_matrix(sequences)
        dt_matrix = compute_demand_transference_matrix(transactions_df, switching_df)

        revenue_per_product = transactions_df.groupby("stockcode").apply(
            lambda x: (x["price"] * x["quantity"]).sum()
        )

    # Parameters
    max_skus = params.get("max_skus", 100)
    min_coverage = params.get("min_coverage", 0.80)
    objective = params.get("objective", "revenue")
    solver = params.get("solver", "heuristic")

    # Solver info
    if solver == "milp":
        import importlib.util
        if importlib.util.find_spec("ortools.linear_solver.pywraplp"):
            st.info("🔧 Using OR-Tools MILP solver")
        else:
            st.warning("OR-Tools not installed. Falling back to heuristic solver.")
            solver = "heuristic"
    else:
        st.info("⚡ Using greedy/SA heuristic solver")

    # Run optimization
    if st.button("🚀 Optimize Assortment", type="primary"):
        with st.spinner(f"Optimizing assortment (max {max_skus} SKUs, {min_coverage:.0%} coverage)..."):
            if solver == "milp":
                selected_skus, metrics = optimize_assortment_milp(
                    transactions_df,
                    dt_matrix,
                    dt_matrix,
                    revenue_per_product,
                    None,
                    max_skus=max_skus,
                    min_coverage=min_coverage,
                    objective=objective,
                )
            else:
                selected_skus, metrics = optimize_assortment_heuristic(
                    transactions_df,
                    dt_matrix,
                    dt_matrix,
                    revenue_per_product,
                    None,
                    max_skus=max_skus,
                    min_coverage=min_coverage,
                    objective=objective,
                )

        st.success(f"Optimization complete! Selected {len(selected_skus)} SKUs")

        # Results
        _render_assortment_results(
            selected_skus, metrics, revenue_per_product, product_lookup, transactions_df
        )

    # Scenario Generation
    if st.button("🎲 Generate Scenarios"):
        with st.spinner("Generating diverse assortment scenarios..."):
            base_assortment = (
                transactions_df["stockcode"].value_counts().head(max_skus).index.tolist()
            )
            scenarios = generate_assortment_scenarios(
                transactions_df,
                base_assortment,
                n_scenarios=10,
                max_skus_range=(int(max_skus * 0.5), max_skus),
            )

        _render_scenario_comparison(scenarios, product_lookup, revenue_per_product)


def _render_assortment_results(
    selected_skus: list,
    metrics: dict,
    revenue_per_product: pd.Series,
    product_lookup: dict,
    transactions_df: pd.DataFrame,
):
    """Render assortment optimization results."""
    st.subheader("📊 Optimized Assortment")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Selected SKUs", len(selected_skus))
    with col2:
        st.metric("Expected Revenue", f"${metrics.get('expected_revenue', 0):,.2f}")
    with col3:
        st.metric("Coverage", f"{metrics.get('coverage', 0):.1%}")
    with col4:
        st.metric("Recovery Rate", f"{metrics.get('recovery_rate', 0):.1%}")

    # SKU table
    sku_df = pd.DataFrame({"stockcode": selected_skus})
    sku_df["product_name"] = sku_df["stockcode"].map(product_lookup)
    sku_df["revenue"] = sku_df["stockcode"].map(revenue_per_product).fillna(0)
    sku_df["category"] = (
        sku_df["stockcode"].map(
            transactions_df.drop_duplicates("stockcode").set_index("stockcode")["category"]
        )
        if "category" in transactions_df.columns
        else "N/A"
    )

    st.dataframe(sku_df.sort_values("revenue", ascending=False), use_container_width=True)

    # Category distribution
    if "category" in sku_df.columns:
        cat_dist = (
            sku_df.groupby("category")
            .agg(skus=("stockcode", "count"), revenue=("revenue", "sum"))
            .reset_index()
        )
        fig = px.pie(
            cat_dist,
            values="revenue",
            names="category",
            title="Optimized Assortment by Category Revenue",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Export
    render_analytics_export(sku_df, product_lookup, prefix="assortment")


def _render_scenario_comparison(
    scenarios: list, product_lookup: dict, revenue_per_product: pd.Series
):
    """Render scenario comparison table."""
    st.subheader("📋 Scenario Comparison")

    rows = []
    for i, scen in enumerate(scenarios):
        skus = scen.get("skus", [])
        rows.append(
            {
                "Scenario": i + 1,
                "SKUs": len(skus),
                "Revenue": scen.get("revenue", 0),
                "Coverage": scen.get("coverage", 0),
                "Recovery": scen.get("recovery_rate", 0),
                "Categories": scen.get("n_categories", 0),
            }
        )

    scen_df = pd.DataFrame(rows)
    st.dataframe(scen_df, use_container_width=True)

    # Parallel coordinates
    if len(scen_df) > 1:
        fig = px.parallel_coordinates(
            scen_df,
            color="Revenue",
            dimensions=["SKUs", "Revenue", "Coverage", "Recovery", "Categories"],
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def detect_attribute_columns(df: pd.DataFrame) -> list[str]:
    """Detect common product attribute columns."""
    candidates = [
        "category", "brand", "size", "flavor", "color", "variant",
        "type", "style", "material", "collection", "line", "range",
        "pack_size", "unit", "weight", "volume", "scent", "design",
        "theme", "occasion", "target_audience", "gender", "age_group",
    ]
    return [c for c in candidates if c in df.columns]


def render_quality_summary(metadata: dict):
    """Render CDT quality metrics at top of results."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        delta = " Pass" if metadata["passed_threshold"] else " Fail"
        st.metric(
            "Tree Quality vs Baseline",
            f"{metadata['quality_ratio']:.1%}",
            delta=delta,
            help="CDT threshold: 60%",
        )
    with col2:
        st.metric("Tree Quality Score", f"{metadata['tree_quality']:.3f}")
    with col3:
        st.metric("Unconstrained Baseline", f"{metadata['unconstrained_baseline']:.3f}")
    with col4:
        st.metric("Threshold", f"{metadata['quality_threshold']:.0%}")

    if not metadata["passed_threshold"]:
        st.warning(
            f" Tree quality ({metadata['quality_ratio']:.1%}) is below the "
            f"{metadata['quality_threshold']:.0%} threshold. "
            "Consider: lowering min_cluster_size, adding more attributes, "
            "or using a different similarity method."
        )
