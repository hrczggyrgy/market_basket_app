"""Customer Decision Tree (CDT) - Streamlit UI Tab.

This tab implements the unsupervised CDT pipeline:
1. Build customer purchase sequences
2. Compute similarity matrix (Phi / Jaccard)
3. Hierarchical clustering + optimal k selection
4. Bottom-up tree construction with attribute enrichment
5. Behavioral matrices: switching, substitution, bundling
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.algorithms.fpgrowth import create_basket_matrix, run_fpgrowth
from src.analytics.cdt_behavioral import (
    build_behavioral_matrices,
    get_top_bundling_pairs,
    get_top_substitution_pairs,
    get_top_switching_paths,
    switching_matrix_to_heatmap,
)
from src.analytics.cdt_clustering import (
    find_optimal_clusters,
    get_cluster_assignments,
    perform_hierarchical_clustering,
)
from src.analytics.cdt_similarity import (
    build_customer_sequences,
    build_similarity_matrix,
)
from src.analytics.cdt_tree_builder import (
    build_cdt,
    extract_product_attributes,
    tree_to_dataframe,
    tree_to_json,
)
from src.ui.export import render_analytics_export
from src.viz.cdt_viz import (
    plot_behavioral_heatmap,
    plot_dendrogram,
    plot_silhouette_scores,
    plot_similarity_heatmap,
    plot_sunburst,
    plot_switching_network,
    plot_treemap,
)


# Bug 1: Cached wrappers for heavy CDT computations
@st.cache_data
def _cached_build_customer_sequences(transactions_df):
    return build_customer_sequences(transactions_df)


@st.cache_data
def _cached_build_similarity_matrix(transactions_df, method, min_cooccurrence):
    return build_similarity_matrix(
        transactions_df, method=method, min_cooccurrence=min_cooccurrence
    )


@st.cache_data
def _cached_perform_hierarchical_clustering(similarity_matrix, linkage_method, distance_method):
    return perform_hierarchical_clustering(
        similarity_matrix, linkage_method=linkage_method, distance_method=distance_method
    )


@st.cache_data
def _cached_find_optimal_clusters(
    linkage_matrix, similarity_matrix, distance_method, min_clusters, max_clusters
):
    return find_optimal_clusters(
        linkage_matrix,
        similarity_matrix,
        distance_method=distance_method,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
    )


@st.cache_data
def _cached_get_cluster_assignments(linkage_matrix, similarity_matrix, n_clusters):
    return get_cluster_assignments(linkage_matrix, similarity_matrix, n_clusters=n_clusters)


@st.cache_data
def _cached_extract_product_attributes(transactions_df, attribute_cols):
    return extract_product_attributes(transactions_df, attribute_cols=attribute_cols)


@st.cache_data
def _cached_build_cdt(
    similarity_matrix,
    cluster_assignments,
    attributes_df,
    min_cluster_size,
    quality_threshold,
    candidate_attributes,
):
    return build_cdt(
        similarity_matrix,
        cluster_assignments,
        attributes_df,
        min_cluster_size=min_cluster_size,
        quality_threshold=quality_threshold,
        candidate_attributes=candidate_attributes,
    )


@st.cache_data
def _cached_build_behavioral_matrices(sequences, min_cooccurrence):
    return build_behavioral_matrices(sequences, min_cooccurrence=min_cooccurrence)


@st.cache_data
def _cached_get_top_bundling_pairs(affinity_matrix, top_n):
    return get_top_bundling_pairs(affinity_matrix, top_n=top_n)


@st.cache_data
def _cached_get_top_substitution_pairs(switching_matrix, top_n):
    return get_top_substitution_pairs(switching_matrix, top_n=top_n)


@st.cache_data
def _cached_get_top_switching_paths(switching_matrix, top_n):
    return get_top_switching_paths(switching_matrix, top_n=top_n)


def render_cdt_tab(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render Customer Decision Tree & Patterns tab."""

    st.header(" Customer Decision Tree & Patterns (CDT)")

    if transactions_df.empty:
        st.warning("No transaction data available")
        return

    # ============================================================
    # CHECK FOR EXISTING RESULTS IN SESSION STATE
    # ============================================================
    has_results = "cdt_root" in st.session_state

    if has_results:
        # Results exist - render them directly
        root = st.session_state["cdt_root"]
        metadata = st.session_state["cdt_metadata"]
        similarity_matrix = st.session_state["cdt_similarity_matrix"]
        switching_df = st.session_state["cdt_switching_df"]
        substitution_df = st.session_state["cdt_substitution_df"]
        bundling_df = st.session_state["cdt_bundling_df"]
        linkage_matrix = st.session_state["cdt_linkage_matrix"]
        ordered_labels = st.session_state["cdt_ordered_labels"]
        silhouette_scores = st.session_state["cdt_silhouette_scores"]
        optimal_k = st.session_state["cdt_optimal_k"]
        similarity_method = st.session_state.get("cdt_similarity_method", "phi")
        product_lookup = st.session_state.get("cdt_product_lookup", product_lookup)

        # Add a button to clear results and reconfigure
        if st.button(" Reconfigure & Rebuild", type="secondary"):
            # Clear session state
            for key in list(st.session_state.keys()):
                if key.startswith("cdt_"):
                    del st.session_state[key]
            st.rerun()

        # Render results tabs
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

    # ============================================================
    # NO RESULTS YET - SHOW CONFIGURATION PANEL
    # ============================================================
    _render_cdt_config_panel(transactions_df, product_lookup, params)


def _render_cdt_config_panel(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render the CDT configuration panel and run pipeline when button clicked."""

    # ============================================================
    # CONFIGURATION PANEL
    # ============================================================
    with st.expander(" CDT Configuration", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Similarity")
            similarity_method = st.selectbox(
                "Similarity Method",
                ["phi", "jaccard"],
                index=0,
                help="Phi coefficient ([-1,1]) or Jaccard ([0,1])",
            )
            min_cooccurrence = st.slider(
                "Min Co-occurrence",
                2,
                20,
                params.get("min_cooccurrence", 5),
                help="Min customers buying both products",
            )

        with col2:
            st.subheader("Clustering")
            linkage_method = st.selectbox(
                "Linkage Method",
                ["average", "complete", "single"],
                index=0,
                help="Average = default",
            )
            min_k = st.slider("Min Clusters (k)", 2, 10, params.get("min_k", 2))
            max_k = st.slider("Max Clusters (k)", 3, 20, params.get("max_k", 15))

        with col3:
            st.subheader("Tree Building")
            min_cluster_size = st.slider(
                "Min Cluster Size",
                2,
                10,
                params.get("min_cluster_size", 3),
                help="Min products per tree node",
            )
            quality_threshold = (
                st.slider(
                    "Quality Threshold (%)",
                    40,
                    80,
                    params.get("quality_threshold", 60),
                    help="Tree quality vs unconstrained baseline (: 60%)",
                )
                / 100.0
            )

        with col4:
            st.subheader("Behavioral")
            top_n_products = st.slider(
                "Top N Products",
                20,
                200,
                params.get("top_n_products", 50),
                help="Limit analysis to top products",
            )
            min_lift = st.slider("Min Lift for Bundling", 1.0, 3.0, 1.2, 0.1)
            max_sub = st.slider("Max Substitution for Bundling", 0.0, 0.5, 0.3, 0.05)

    # Detect attribute columns
    attribute_cols = detect_attribute_columns(transactions_df)
    if attribute_cols:
        st.info(f" Detected attribute columns: {', '.join(attribute_cols)}")
        selected_attrs = st.multiselect(
            "Attributes for Tree Enrichment",
            attribute_cols,
            default=attribute_cols,
            help="These will be tested as split criteria",
        )
    else:
        st.warning(
            "No attribute columns detected (category, brand, size, flavor, etc.). Tree will be built from similarity only."
        )
        selected_attrs = []

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

    # ============================================================
    # RUN PIPELINE
    # ============================================================
    run_button = st.button(" Build Customer Decision Tree", type="primary", width="stretch")

    if not run_button:
        st.info("Configure parameters above and click **Build Customer Decision Tree** to start.")
        return

    # Progress tracking
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

    try:
        # Step 1: Build customer sequences
        status_text.text("Step 1/6: Building customer purchase sequences...")
        progress_bar.progress(10)

        sequences = _cached_build_customer_sequences(transactions_df)

        # Step 2: Compute similarity matrix
        status_text.text("Step 2/6: Computing similarity matrix...")
        progress_bar.progress(25)

        similarity_matrix = _cached_build_similarity_matrix(
            transactions_df,
            method=similarity_method,
            min_cooccurrence=min_cooccurrence,
        )

        if similarity_matrix.empty:
            st.error("Could not compute similarity matrix. Check data and parameters.")
            return

        # Limit to top products if specified
        if top_n_products and len(similarity_matrix) > top_n_products:
            # Use average similarity as importance
            avg_sim = similarity_matrix.mean(axis=1).sort_values(ascending=False)
            top_products = avg_sim.head(top_n_products).index.tolist()
            similarity_matrix = similarity_matrix.loc[top_products, top_products]
            sequences = {
                c: [p for p in prods if p in top_products] for c, prods in sequences.items()
            }
            sequences = {c: p for c, p in sequences.items() if p}

        # Step 3: Hierarchical clustering
        status_text.text("Step 3/6: Performing hierarchical clustering...")
        progress_bar.progress(40)

        linkage_matrix, ordered_labels = _cached_perform_hierarchical_clustering(
            similarity_matrix,
            linkage_method=linkage_method,
            distance_method=similarity_method,
        )

        # Find optimal k
        optimal_k, silhouette_scores = _cached_find_optimal_clusters(
            linkage_matrix,
            similarity_matrix,
            distance_method=similarity_method,
            min_clusters=min_k,
            max_clusters=min(max_k, len(similarity_matrix) - 1),
        )

        # Get cluster assignments
        cluster_assignments = _cached_get_cluster_assignments(
            linkage_matrix, similarity_matrix, n_clusters=optimal_k
        )

        # Step 4: Build CDT
        status_text.text("Step 4/6: Building Customer Decision Tree...")
        progress_bar.progress(55)

        # Extract product attributes
        attributes_df = extract_product_attributes(transactions_df, attribute_cols=selected_attrs)

        root, metadata = build_cdt(
            similarity_matrix,
            cluster_assignments,
            attributes_df,
            min_cluster_size=min_cluster_size,
            quality_threshold=quality_threshold,
            candidate_attributes=selected_attrs if selected_attrs else None,
        )

        # Step 5: Compute behavioral matrices
        status_text.text("Step 5/6: Computing behavioral matrices...")
        progress_bar.progress(70)

        # Need affinity matrix for bundling
        basket = create_basket_matrix(transactions_df)
        if len(basket.columns) > top_n_products:
            basket = (
                basket[top_products]
                if "top_products" in locals()
                else basket.iloc[:, :top_n_products]
            )

        freq_items = run_fpgrowth(basket, min_support=0.001, max_len=2)
        affinity_matrix = pd.DataFrame(
            1.0, index=basket.columns, columns=basket.columns, dtype=float
        )
        if not freq_items.empty:
            pairs = freq_items[freq_items["length"] == 2]
            product_probs = basket.mean()
            for _, row in pairs.iterrows():
                items = list(row["itemsets"])
                if len(items) == 2:
                    a, b = items[0], items[1]
                    p_a = product_probs.get(a, 0)
                    p_b = product_probs.get(b, 0)
                    if p_a > 0 and p_b > 0:
                        lift = row["support"] / (p_a * p_b)
                        affinity_matrix.loc[a, b] = lift
                        affinity_matrix.loc[b, a] = lift

        switching_df, substitution_df, bundling_df = build_behavioral_matrices(
            transactions_df,
            similarity_matrix,
            affinity_matrix,
            sequences,
            top_n_products=top_n_products,
        )

        # Step 6: Visualizations
        status_text.text("Step 6/6: Generating visualizations...")
        progress_bar.progress(85)

        # Clear progress
        progress_bar.progress(100)
        status_text.text(" Complete!")

        # Store in session for export
        st.session_state["cdt_root"] = root
        st.session_state["cdt_metadata"] = metadata
        st.session_state["cdt_similarity_matrix"] = similarity_matrix
        st.session_state["cdt_switching_df"] = switching_df
        st.session_state["cdt_substitution_df"] = substitution_df
        st.session_state["cdt_bundling_df"] = bundling_df
        st.session_state["cdt_linkage_matrix"] = linkage_matrix
        st.session_state["cdt_ordered_labels"] = ordered_labels
        st.session_state["cdt_silhouette_scores"] = silhouette_scores
        st.session_state["cdt_optimal_k"] = optimal_k
        st.session_state["cdt_sequences"] = sequences
        st.session_state["cdt_product_lookup"] = product_lookup
        st.session_state["cdt_similarity_method"] = similarity_method

    except Exception as e:
        st.error(f"Pipeline failed: {str(e)}")
        import traceback

        st.code(traceback.format_exc())
        return


def _render_cdt_results_tabs(
    root,
    metadata: dict,
    similarity_matrix: pd.DataFrame,
    switching_df: pd.DataFrame,
    substitution_df: pd.DataFrame,
    bundling_df: pd.DataFrame,
    linkage_matrix: np.ndarray,
    ordered_labels: list[str],
    silhouette_scores: dict[int, float],
    optimal_k: int,
    product_lookup: dict,
    similarity_method: str = "phi",
):
    """Render the CDT results tabs."""

    # Quality summary at top
    render_quality_summary(metadata)

    # Main tabs
    tabs = st.tabs(
        [
            " CDT Sunburst",
            " CDT Treemap",
            " Dendrogram & Clusters",
            " Similarity Heatmap",
            " Switching Analysis",
            " Substitution Analysis",
            " Bundling Opportunities",
            " Export",
        ]
    )

    with tabs[0]:
        st.subheader("Customer Decision Tree - Sunburst View")
        st.caption(
            "Hierarchical tree from bottom-up clustering. Inner rings = higher-level splits (attributes). Outer rings = product leaves."
        )

        fig = plot_sunburst(
            root,
            title=f"CDT: {metadata['n_leaves']} leaf clusters, {metadata['max_depth']} levels",
            height=700,
        )
        st.plotly_chart(fig, width="stretch")

        # Tree statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Nodes", metadata["n_nodes"])
        with col2:
            st.metric("Leaf Clusters", metadata["n_leaves"])
        with col3:
            st.metric("Max Depth", metadata["max_depth"])

    with tabs[1]:
        st.subheader("Customer Decision Tree - Treemap View")
        st.caption("Area proportional to number of products. Color = split attribute.")

        size_metric = st.radio("Size Metric", ["size", "similarity_within"], horizontal=True)
        fig = plot_treemap(root, size_metric=size_metric, height=700)
        st.plotly_chart(fig, width="stretch")

    with tabs[2]:
        st.subheader("Hierarchical Clustering Dendrogram")

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = plot_dendrogram(linkage_matrix, ordered_labels, height=600)
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Cluster Quality")
            st.metric("Optimal k (Silhouette)", optimal_k)
            st.metric("Silhouette Score", f"{silhouette_scores.get(optimal_k, 0):.3f}")

            # Show silhouette chart
            fig_sil = plot_silhouette_scores(silhouette_scores, optimal_k, height=300)
            st.plotly_chart(fig_sil, width="stretch")

    with tabs[3]:
        st.subheader("Product Similarity Matrix")
        st.caption(
            f"Method: {similarity_method.upper()}. Red = dissimilar, Blue = similar (substitutable)."
        )

        top_n = st.slider("Top N Products", 10, min(100, len(similarity_matrix)), 50)
        fig = plot_similarity_heatmap(similarity_matrix, top_n=top_n, height=600)
        st.plotly_chart(fig, width="stretch")

    with tabs[4]:
        st.subheader("Switching Analysis")
        st.caption("Product-to-product switching rates from customer purchase sequences.")

        if not switching_df.empty:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Heatmap
                switch_matrix = switching_matrix_to_heatmap(switching_df, top_n=30)
                if not switch_matrix.empty:
                    fig = plot_behavioral_heatmap(
                        switch_matrix,
                        title="Switching Rate Matrix",
                        height=500,
                        colorscale="Reds",
                    )
                    st.plotly_chart(fig, width="stretch")

            with col2:
                # Top switching paths
                st.write("**Top Switching Paths**")
                top_switches = get_top_switching_paths(switching_df, top_n=15)
                if not top_switches.empty:
                    display_df = top_switches.copy()
                    display_df["from"] = display_df["from_product"].map(
                        lambda x: product_lookup.get(x, x)[:30]
                    )
                    display_df["to"] = display_df["to_product"].map(
                        lambda x: product_lookup.get(x, x)[:30]
                    )
                    st.dataframe(
                        display_df[["from", "to", "switch_count", "switch_rate"]].round(4),
                        hide_index=True,
                        width="stretch",
                    )

            # Network graph
            st.subheader("Switching Network")
            fig_net = plot_switching_network(switching_df, product_lookup, min_rate=0.05)
            st.plotly_chart(fig_net, width="stretch")

            render_analytics_export(switching_df, "CDT_Switching")
        else:
            st.info(
                "No switching data available. Need customers with repeat purchases across products."
            )

    with tabs[5]:
        st.subheader("Substitution Analysis")
        st.caption(
            "High similarity = high substitutability. Derived from co-purchase patterns (Phi coefficient)."
        )

        if not substitution_df.empty:
            # Heatmap
            top_n = st.slider("Top N Products", 10, 100, 50, key="sub_top_n")
            fig = plot_similarity_heatmap(
                substitution_df,
                top_n=top_n,
                title="Substitution Score Matrix",
                height=500,
            )
            st.plotly_chart(fig, width="stretch")

            # Top pairs
            st.write("**Top Substitutable Pairs**")
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
                    width="stretch",
                )
                render_analytics_export(top_subs, "CDT_Substitution")

    with tabs[6]:
        st.subheader("Bundling Opportunities")
        st.caption("High lift + low substitution = true complements (not substitutes).")

        if not bundling_df.empty:
            st.write("**Top Bundling Pairs**")
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
                    display_df[
                        [
                            "Product A",
                            "Product B",
                            "lift",
                            "substitution",
                            "bundle_score",
                        ]
                    ].round(4),
                    hide_index=True,
                    width="stretch",
                )
                render_analytics_export(top_bundles, "CDT_Bundling")

            # Scatter: Lift vs Substitution
            st.subheader("Lift vs Substitution Tradeoff")
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
                            bundling_df["product_a"], bundling_df["product_b"], strict=False
                        )
                    ],
                    hovertemplate="%{text}<br>Substitution: %{x:.3f}<br>Lift: %{y:.3f}<extra></extra>",
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
            st.plotly_chart(fig, width="stretch")

    with tabs[7]:
        st.subheader("Export Results")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Tree Structure (JSON)**")
            if st.button("Export CDT as JSON"):
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
                df = tree_to_dataframe(root)
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    file_name="cdt_tree.csv",
                    mime="text/csv",
                )

        st.divider()

        # Export behavioral matrices
        st.write("**Behavioral Matrices (Excel)**")
        export_cols = st.columns(3)
        with export_cols[0]:
            if not switching_df.empty:
                csv = switching_df.to_csv(index=False)
                st.download_button("Switching Matrix", csv, "cdt_switching.csv", "text/csv")
        with export_cols[1]:
            if not substitution_df.empty:
                # Export top pairs only
                top_subs = get_top_substitution_pairs(substitution_df, top_n=100)
                csv = top_subs.to_csv(index=False)
                st.download_button("Substitution Pairs", csv, "cdt_substitution.csv", "text/csv")
        with export_cols[2]:
            if not bundling_df.empty:
                top_bundles = get_top_bundling_pairs(bundling_df, top_n=100)
                csv = top_bundles.to_csv(index=False)
                st.download_button("Bundling Pairs", csv, "cdt_bundling.csv", "text/csv")


def detect_attribute_columns(df: pd.DataFrame) -> list[str]:
    """Detect common product attribute columns."""
    candidates = [
        "category",
        "brand",
        "size",
        "flavor",
        "color",
        "variant",
        "type",
        "style",
        "material",
        "collection",
        "line",
        "range",
        "pack_size",
        "unit",
        "weight",
        "volume",
        "scent",
        "design",
        "theme",
        "occasion",
        "target_audience",
        "gender",
        "age_group",
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
            f"Consider: lowering min_cluster_size, adding more attributes, "
            f"or using a different similarity method."
        )
