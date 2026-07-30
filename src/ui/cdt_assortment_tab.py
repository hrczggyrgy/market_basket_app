"""CDT & Assortment Tab — Customer Decision Tree, Demand Transference, Assortment Optimization."""

from typing import Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import (
    build_cdt,
    build_customer_sequences,
    build_similarity_matrix,
    build_similarity_matrix_ensemble,
    compute_affinity_matrix,
    compute_bundling_matrix,
    compute_switching_matrix,
    extract_product_attributes,
    find_optimal_clusters,
    get_cluster_assignments,
    get_dendrogram_data,
    get_substitution_matrix,
    get_top_bundling_pairs,
    get_top_substitution_pairs,
    perform_hierarchical_clustering,
)
from src.analytics.assortment_opt import (
    generate_assortment_scenarios,
    optimize_assortment_heuristic,
    optimize_assortment_milp,
)
from src.analytics.cdt_attributes import build_transaction_derived_attributes
from src.analytics.cdt_community import (
    build_product_graph,
    detect_communities,
    hierarchical_clustering_within_communities,
    merge_community_dendrograms,
)
from src.analytics.cdt_tree_builder import tree_to_dataframe
from src.analytics.demand_transference import (
    compute_demand_transference_matrix,
    delist_impact_analysis,
    node_delist_impact,
)
from src.viz.cdt_viz import plot_sunburst, plot_treemap
from src.viz.cdt_viz import plot_dendrogram as _plot_dendrogram
from src.analytics.sufficiency import assess_data_sufficiency, format_sufficiency_summary


def render_cdt_assortment_tab(
    transactions_df: pd.DataFrame,
    product_lookup: dict,
    params: dict,
    mode: str = "cdt",
):
    """Main entry point for CDT & Assortment tab with sub-modes."""

    # Data sufficiency gate (applies to all modes)
    if not transactions_df.empty:
        sufficiency = assess_data_sufficiency(transactions_df)
        with st.expander("📋 Data Sufficiency", expanded=sufficiency["overall"] != "robust"):
            st.markdown(format_sufficiency_summary(sufficiency))
            if sufficiency["overall"] == "insufficient":
                st.warning("Dataset may be too small for reliable analysis.")
            elif sufficiency["overall"] == "directional":
                st.info("Results should be treated as directional.")

    if mode == "cdt":
        _render_cdt_builder(transactions_df, product_lookup, params)
    elif mode == "transference":
        _render_demand_transference(transactions_df, product_lookup, params)
    elif mode == "assortment":
        _render_assortment_optimizer(transactions_df, product_lookup, params)
    else:
        st.warning(f"Unknown mode: {mode}")


# ============================================================================
# CDT BUILDER MODE
# ============================================================================


def _render_cdt_builder(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render CDT Builder with similarity methods, community detection, and tree building."""

    st.header("🌳 Customer Decision Tree (CDT) Builder")

    # Step 1: Similarity Matrix
    with st.spinner("Building similarity matrices..."):
        similarity_matrices = build_similarity_matrix_ensemble(
            transactions_df,
            methods=params.get("similarity_methods", ["phi"]),
            min_cooccurrence=params.get("min_cooccurrence", 5),
        )

    if not similarity_matrices:
        st.error("Failed to build similarity matrices. Try lowering min_cooccurrence.")
        return

    # Use first method as primary, or ensemble if available
    primary_method = params.get("similarity_methods", ["phi"])[0]
    sim_matrix = similarity_matrices.get("ensemble", similarity_matrices.get(primary_method))

    st.success(f"Built similarity matrices: {', '.join(similarity_matrices.keys())}")

    # Similarity comparison
    if len(similarity_matrices) > 1:
        with st.expander("📊 Similarity Matrix Comparison", expanded=False):
            _render_similarity_comparison(similarity_matrices)

    # Step 2: Community Detection
    community_assignments = None
    community_method = params.get("community_method", "label_propagation")

    if community_method != "none":
        with st.spinner(f"Detecting communities ({community_method})..."):
            # Build graph
            graph = build_product_graph(
                sim_matrix,
                min_weight=params.get("graph_min_weight", 0.1),
                min_joint_customers=params.get("min_cooccurrence", 5),
                max_edges_per_node=params.get("graph_max_degree", 50),
            )

            # Detect communities
            community_assignments = detect_communities(
                graph,
                method=community_method,
                resolution=params.get("community_resolution", 1.0),
            )

            n_communities = len(set(community_assignments.values()))
            st.info(f"Detected {n_communities} communities using {community_method}")

            # Show community summary
            with st.expander("🏘️ Community Summary", expanded=False):
                _render_community_summary(community_assignments, graph)

    # Step 3: Hierarchical Clustering
    with st.spinner("Performing hierarchical clustering..."):
        if community_assignments:
            # Cluster within communities
            comm_dendrograms = hierarchical_clustering_within_communities(
                sim_matrix,
                community_assignments,
                linkage_method=params.get("linkage_method", "average"),
                distance_method=primary_method,
            )
            # Merge dendrograms
            linkage_matrix, ordered_labels = merge_community_dendrograms(
                comm_dendrograms, community_assignments
            )
        else:
            # Global clustering
            linkage_matrix, ordered_labels = perform_hierarchical_clustering(
                sim_matrix,
                linkage_method=params.get("linkage_method", "average"),
                distance_method=primary_method,
            )

    # Step 4: Cluster Assignments
    # When community detection is active, use community assignments directly
    # (merged community dendrograms are not valid for fcluster). Otherwise,
    # derive clusters from the global linkage matrix.
    if community_assignments is not None:
        unique_comm = sorted(set(community_assignments.values()))
        comm_map = {old: new for new, old in enumerate(unique_comm)}
        cluster_assignments = {p: comm_map[c] for p, c in community_assignments.items()}
        optimal_k = len(unique_comm)
        silhouette_scores = {}
        st.info(f"Using {optimal_k} community-based clusters")
    else:
        optimal_k, silhouette_scores = find_optimal_clusters(
            linkage_matrix,
            sim_matrix,
            distance_method=primary_method,
            min_clusters=params.get("min_k", 2),
            max_clusters=params.get("max_k", 15),
        )
        st.info(f"Optimal clusters (silhouette): **k = {optimal_k}**")
        with st.expander("📈 Silhouette Analysis", expanded=False):
            _render_silhouette_plot(silhouette_scores, optimal_k)

        cluster_assignments = get_cluster_assignments(linkage_matrix, sim_matrix, n_clusters=optimal_k)

    with st.spinner("Building Customer Decision Tree..."):
        # Extract attributes
        if params.get("extract_from_text", False):
            attr_df = build_transaction_derived_attributes(
                transactions_df,
                sim_matrix,
                n_tiers=3,
            )
        else:
            attr_df = extract_product_attributes(
                transactions_df,
                attribute_cols=["category", "brand", "size", "flavour", "flavor", "variant"],
            )

        # Combine with transaction-derived attributes
        txn_attrs = build_transaction_derived_attributes(
            transactions_df,
            sim_matrix,
            n_tiers=3,
        )
        attr_df = pd.concat([attr_df, txn_attrs], axis=1)

        # Ensure all products in attr_df
        all_products = list(sim_matrix.index)
        attr_df = attr_df.reindex(all_products)

        root, metadata = build_cdt(
            sim_matrix,
            cluster_assignments,
            attr_df,
            min_cluster_size=params.get("min_cluster_size", 3),
            quality_threshold=params.get("quality_threshold", 60) / 100,
            candidate_attributes=params.get("candidate_attributes", None),
            criterion=params.get("split_criterion", "mutual_info"),
            alpha=params.get("split_alpha", 0.5),
        )

    # Display CDT results
    _render_cdt_results(root, metadata, sim_matrix, cluster_assignments, product_lookup, transactions_df, params)


def _render_similarity_comparison(similarity_matrices: Dict[str, pd.DataFrame]):
    """Render comparison of different similarity methods."""
    methods = list(similarity_matrices.keys())
    if len(methods) < 2:
        return

    cols = st.columns(min(3, len(methods)))
    for i, (name, mat) in enumerate(similarity_matrices.items()):
        with cols[i % len(cols)]:
            st.subheader(name.upper())
            # Heatmap sample (top 20 products)
            sample_products = mat.index[: min(20, len(mat))]
            sample_mat = mat.loc[sample_products, sample_products]
            fig = px.imshow(
                sample_mat.values,
                labels=dict(x="Product", y="Product", color="Similarity"),
                color_continuous_scale="RdBu_r",
                zmin=-1 if name == "phi" else 0,
                zmax=1,
            )
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # Stats
            vals = mat.values[np.triu_indices_from(mat.values, k=1)]
            st.metric("Mean", f"{np.mean(vals):.3f}")
            st.metric("Std", f"{np.std(vals):.3f}")


def _render_community_summary(community_assignments: Dict[str, int], graph: nx.Graph):
    """Render community detection summary."""
    comm_sizes = pd.Series(community_assignments).value_counts().sort_index()
    comm_df = pd.DataFrame({"Community": comm_sizes.index, "Size": comm_sizes.values})
    comm_df["Modularity Contribution"] = [0.0] * len(comm_df)  # placeholder

    st.dataframe(comm_df, use_container_width=True)

    # Community size bar chart
    fig = px.bar(comm_df, x="Community", y="Size", title="Community Sizes")
    st.plotly_chart(fig, use_container_width=True)


def _render_silhouette_plot(silhouette_scores: Dict[int, float], optimal_k: int):
    """Render silhouette score vs k plot."""
    if not silhouette_scores:
        return

    ks = list(silhouette_scores.keys())
    scores = list(silhouette_scores.values())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=scores, mode="lines+markers", name="Silhouette Score"))
    fig.add_vline(
        x=optimal_k, line_dash="dash", line_color="red", annotation_text=f"Optimal k={optimal_k}"
    )
    fig.update_layout(
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_cdt_results(
    root,
    metadata: dict,
    sim_matrix: pd.DataFrame,
    cluster_assignments: dict,
    product_lookup: dict,
    transactions_df: pd.DataFrame,
    params: dict,
):
    """Render CDT results with visualizations and export options."""

    # Quality metrics
    st.subheader("📊 CDT Quality Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Tree Quality", f"{metadata['tree_quality']:.4f}")
    with col2:
        st.metric("Unconstrained Baseline", f"{metadata['unconstrained_baseline']:.4f}")
    with col3:
        st.metric("Quality Ratio", f"{metadata['quality_ratio']:.1%}")
    with col4:
        st.metric("Nodes", metadata["n_nodes"])
    with col5:
        st.metric("Leaves / Depth", f"{metadata['n_leaves']} / {metadata['max_depth']}")

    status = "✅ PASSED" if metadata["passed_threshold"] else "⚠️ BELOW THRESHOLD"
    st.info(f"Quality threshold ({metadata['quality_threshold']:.0%}): {status}")

    # Visualization tabs
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(
        ["🌞 Sunburst", "📦 Treemap", "🌲 Dendrogram", "📋 Tree Table"]
    )

    with viz_tab1:
        fig = plot_sunburst(root)
        st.plotly_chart(fig, use_container_width=True)

    with viz_tab2:
        fig = plot_treemap(root)
        st.plotly_chart(fig, use_container_width=True)

    with viz_tab3:
        st.info("Dendrogram requires linkage matrix from clustering step")
        fig = create_dendrogram_plot(np.array([]))
        st.plotly_chart(fig, use_container_width=True)

    with viz_tab4:
        tree_df = tree_to_dataframe(root)
        st.dataframe(tree_df, use_container_width=True)
        render_export_buttons(tree_df, product_lookup, prefix="cdt")

    # Behavioral matrices
    with st.expander(
        "🔄 Behavioral Matrices (Switching / Substitution / Bundling)", expanded=False
    ):
        _render_behavioral_matrices(sim_matrix, cluster_assignments, product_lookup, params, transactions_df)


def _render_behavioral_matrices(sim_matrix, cluster_assignments, product_lookup, params, transactions_df):
    """Render switching, substitution, and bundling matrices."""
    switching_df = compute_switching_matrix(
        transactions_df,
        product_col="stockcode",
        customer_col="customer_id",
        date_col="date",
        window_days=params.get("window_days", 90),
        min_transactions=params.get("min_switching_transactions", 2),
    )
    substitution_df = get_substitution_matrix(sim_matrix)

    # Build affinity matrix for bundling
    with st.spinner("Building affinity matrix..."):
        affinity_matrix = compute_affinity_matrix(
            transactions_df,
            min_support=params.get("min_support", 0.005),
            min_lift=params.get("min_lift", 1.0),
            top_n_products=params.get("top_n_products", 50),
        )

    if affinity_matrix.empty:
        bundling_df = pd.DataFrame(columns=["product_a", "product_b", "lift", "substitution", "bundle_score"])
    else:
        bundling_df = compute_bundling_matrix(
            affinity_matrix=affinity_matrix,
            substitution_matrix=substitution_df,
            top_n_products=params.get("top_n_products", 50),
            min_lift=params.get("min_lift", 1.2),
            max_substitution=params.get("max_sub", 0.3),
        )

    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Switching", "Substitution", "Bundling"])

    with sub_tab1:
        if not switching_df.empty:
            st.dataframe(switching_df.head(20), use_container_width=True)
        else:
            st.info("No switching data available")

    with sub_tab2:
        if not substitution_df.empty:
            st.dataframe(
                get_top_substitution_pairs(substitution_df, top_n=20), use_container_width=True
            )
        else:
            st.info("No substitution data available")

    with sub_tab3:
        if not bundling_df.empty:
            st.dataframe(get_top_bundling_pairs(bundling_df, top_n=20), use_container_width=True)
        else:
            st.info("No bundling data available")


# ============================================================================
# DEMAND TRANSFERENCE MODE
# ============================================================================


def _render_demand_transference(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render Demand Transference / Delist Simulation."""

    st.header("🔄 Demand Transference & Delist Simulation")

    # Need to run CDT pipeline first to get similarity, clusters, etc.
    with st.spinner("Building CDT pipeline for demand transference..."):
        # Build similarity
        sim_matrix = build_similarity_matrix(
            transactions_df,
            method=params.get("similarity_methods", ["phi"])[0],
            min_cooccurrence=params.get("min_cooccurrence", 5),
        )

        # Build sequences and switching
        sequences = build_customer_sequences(transactions_df)
        switching_df = compute_switching_matrix(sequences)

        # Build CDT
        cluster_assignments = {}  # placeholder - would run clustering
        # ... run clustering to get assignments ...

        # Build demand transference matrix
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

    # Product selector
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

    # Node-level SDP
    with st.expander("🎯 Node-Level Substitutable Demand % (SDP)", expanded=False):
        if cluster_assignments:
            node_sdp = node_delist_impact(transactions_df, dt_matrix, cluster_assignments)
            st.dataframe(node_sdp, use_container_width=True)

            # SDP bar chart
            fig = px.bar(
                node_sdp.sort_values("node_sdp", ascending=True),
                y="node_id",
                x="node_sdp",
                orientation="h",
                color="total_node_revenue",
                title="SDP by CDT Node",
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_delist_waterfall(impact_df: pd.DataFrame, product_names: dict):
    """Render waterfall chart for delist impact."""
    fig = go.Figure()

    # Starting revenue (positive)
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

    # Build required inputs
    with st.spinner("Preparing assortment optimization inputs..."):
        # Similarity
        sim_matrix = build_similarity_matrix(
            transactions_df,
            method=params.get("similarity_methods", ["phi"])[0],
            min_cooccurrence=params.get("min_cooccurrence", 5),
        )

        # Switching for substitution
        sequences = build_customer_sequences(transactions_df)
        switching_df = compute_switching_matrix(sequences)

        # Demand transference
        dt_matrix = compute_demand_transference_matrix(transactions_df, switching_df)

        # Revenue per product
        revenue_per_product = transactions_df.groupby("stockcode").apply(
            lambda x: (x["price"] * x["quantity"]).sum()
        )

        # Cost per product (if available)
        cost_col = params.get("cost_col")
        cost_per_product = None
        if cost_col and cost_col in transactions_df.columns:
            cost_per_product = transactions_df.groupby("stockcode")[cost_col].median()

    # Parameters
    max_skus = params.get("max_skus", 100)
    min_coverage = params.get("min_coverage", 0.80)
    objective = params.get("objective", "revenue")
    solver = params.get("solver", "heuristic")

    # Solver info
    if solver == "milp":
        try:
            from ortools.linear_solver import pywraplp  # noqa: F401

            st.info("🔧 Using OR-Tools MILP solver")
        except ImportError:
            st.warning("OR-Tools not installed. Falling back to heuristic solver.")
            solver = "heuristic"
    else:
        st.info("⚡ Using greedy/SA heuristic solver")

    # Run optimization
    if st.button("🚀 Optimize Assortment", type="primary"):
        with st.spinner(
            f"Optimizing assortment (max {max_skus} SKUs, {min_coverage:.0%} coverage)..."
        ):
            if solver == "milp":
                selected_skus, metrics = optimize_assortment_milp(
                    transactions_df,
                    dt_matrix,
                    dt_matrix,  # sim_matrix as affinity
                    revenue_per_product,
                    None,  # cost_per_product
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
                    cost_per_product,
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

    # Metrics
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
    render_export_buttons(sku_df, product_lookup, prefix="assortment")


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
# HELPER: Export Buttons
# ============================================================================


def render_export_buttons(df: pd.DataFrame, product_lookup: dict, prefix: str = "export"):
    """Render export buttons for DataFrame."""
    col1, col2, col3 = st.columns(3)

    with col1:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            f"{prefix}.csv",
            "text/csv",
            key=f"{prefix}_csv",
        )

    with col2:
        json_str = df.to_json(orient="records", indent=2)
        st.download_button(
            "📥 Download JSON",
            json_str,
            f"{prefix}.json",
            "application/json",
            key=f"{prefix}_json",
        )

    with col3:
        # HTML with plotly if applicable
        st.button(
            "📥 Export Chart (PNG)",
            key=f"{prefix}_png",
            help="Use camera icon on charts for PNG download",
        )


# ============================================================================
# Viz Functions (imported from src.viz.cdt_viz)
# ============================================================================


def create_dendrogram_plot(linkage_matrix: np.ndarray, labels: Optional[list] = None):
    """Create dendrogram plot using the shared plot_dendrogram implementation."""
    if linkage_matrix.size == 0 or len(linkage_matrix) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No clustering data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    labels = labels or [str(i) for i in range(linkage_matrix.shape[0] + 1)]
    return _plot_dendrogram(linkage_matrix, labels)
