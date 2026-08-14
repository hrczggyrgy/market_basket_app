"""Customer Decision Tree (CDT) tab."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.cdt import (
    build_cdt,
    build_similarity_matrix,
    build_transaction_derived_attributes,
    get_cluster_assignments,
    perform_hierarchical_clustering,
    tree_to_dataframe,
)
from src.ui.plots import PALETTE, empty_state, new_fig, show
from src.ui.registry import ModeSpec


def _render_similarity_heatmap(sim: pd.DataFrame, top_n: int = 25) -> None:
    st.subheader(":material/table_chart: Product Similarity Heatmap")
    if sim.empty:
        show(empty_state("No similarity matrix"))
        return

    # Order by hierarchical clustering for better visualization
    linkage_matrix, _ = perform_hierarchical_clustering(sim, method="ward")
    from scipy.cluster.hierarchy import leaves_list

    order = leaves_list(linkage_matrix)
    ordered_products = sim.index[order].tolist()

    top = ordered_products[:top_n]
    sub = sim.loc[top, top]

    fig = go.Figure(
        data=go.Heatmap(
            z=sub.to_numpy(),
            x=[str(c) for c in sub.columns],
            y=[str(i) for i in sub.index],
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "Similarity"},
            text=[[f"{v:.2f}" for v in row] for row in sub.to_numpy()],
            texttemplate="%{text}",
            textfont={"size": 8},
        )
    )
    fig.update_layout(
        xaxis={"tickangle": -45, "side": "top"},
        yaxis={"tickangle": 0, "autorange": "reversed"},
        height=max(400, 18 * top_n),
    )
    show(fig)
    st.caption(
        f"Top {top_n} products by dendrogram order. Red = high similarity, Blue = dissimilar."
    )


def _render_dendrogram(sim: pd.DataFrame) -> None:
    st.subheader(":material/account_tree: Hierarchical Clustering Dendrogram")
    if sim.empty:
        show(empty_state("No similarity matrix"))
        return

    linkage_matrix, _ = perform_hierarchical_clustering(sim, method="ward")

    # Build dendrogram coordinates using scipy
    from scipy.cluster.hierarchy import dendrogram

    ddata = dendrogram(linkage_matrix, labels=sim.index.tolist(), no_plot=True)

    # Plotly dendrogram: create line segments for branches
    fig = new_fig()
    for _i, (x, y) in enumerate(zip(ddata["icoord"], ddata["dcoord"], strict=False)):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line={"color": PALETTE[0], "width": 1.5},
                hoverinfo="none",
                showlegend=False,
            )
        )
    fig.update_layout(
        xaxis={
            "title": "Product index (clustered)",
            "tickvals": list(range(len(ddata["ivl"]))),
            "ticktext": ddata["ivl"],
            "tickangle": -45,
        },
        yaxis={"title": "Distance (1 - similarity)"},
        height=500,
    )
    show(fig)
    st.caption("Ward linkage on 1 - similarity distance. Vertical height = merge distance.")


def _render_tree_sankey(nodes: pd.DataFrame) -> None:
    st.subheader(":material/hub: CDT Flow (Sankey)")
    if nodes.empty:
        show(empty_state("No CDT nodes"))
        return

    # Build Sankey from parent -> child
    internal = nodes[~nodes["is_leaf"].astype(bool)]
    if internal.empty:
        show(empty_state("No internal nodes in CDT"))
        return

    sources = []
    targets = []
    values = []
    labels = {}
    idx = 0

    for _, row in internal.iterrows():
        node_id = row["node_id"]
        parent_id = row["parent_id"]
        if parent_id and parent_id in labels:
            # Find or create label indices
            src_label = f"{parent_id}: {row['attribute']}={row['attribute_value']}"
            tgt_label = f"{node_id}: {row['attribute']}={row['attribute_value']}"
        else:
            src_label = f"ROOT: {row['attribute']}={row['attribute_value']}"
            tgt_label = f"{node_id}: {row['attribute']}={row['attribute_value']}"

        if src_label not in labels:
            labels[src_label] = idx
            idx += 1
        if tgt_label not in labels:
            labels[tgt_label] = idx
            idx += 1

        sources.append(labels[src_label])
        targets.append(labels[tgt_label])
        values.append(int(row["size"]))

    if not sources:
        show(empty_state("No CDT splits to visualize"))
        return

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node={
                    "label": [k for k, v in sorted(labels.items(), key=lambda x: x[1])],
                    "color": PALETTE[0],
                    "pad": 15,
                    "thickness": 20,
                },
                link={
                    "source": sources,
                    "target": targets,
                    "value": values,
                    "color": [PALETTE[1]] * len(sources),
                },
            )
        ]
    )
    fig.update_layout(height=max(400, 30 * len(labels)), font={"size": 10})
    show(fig)
    st.caption("CDT splits from root to leaves. Node size = number of products in leaf.")


def _render_tree_table(nodes: pd.DataFrame) -> None:
    st.subheader(":material/table_rows: CDT Nodes")
    if nodes.empty:
        show(empty_state("No CDT nodes"))
        return

    display = nodes[
        [
            "node_id",
            "name",
            "attribute",
            "attribute_value",
            "size",
            "is_leaf",
            "similarity_within",
            "parent_id",
        ]
    ].copy()
    display["is_leaf"] = display["is_leaf"].map({1: "Yes", 0: "No"})
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_attribute_importance(nodes: pd.DataFrame) -> None:
    st.subheader(":material/analytics: Split Attribute Importance")
    internal = nodes[~nodes["is_leaf"].astype(bool)]
    if internal.empty:
        show(empty_state("No splits in CDT"))
        return

    attr_counts = internal["attribute"].value_counts().reset_index()
    attr_counts.columns = ["attribute", "count"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=attr_counts["attribute"],
                y=attr_counts["count"],
                marker={"color": PALETTE[0]},
                text=attr_counts["count"],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(yaxis={"title": "Number of splits"}, xaxis={"title": "Attribute"})
    show(fig)
    st.caption("How often each attribute was chosen as the splitting criterion.")


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/account_tree: Customer Decision Tree")

    with st.expander("Parameters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        method = c1.selectbox(
            "Similarity method",
            ["embedding", "phi", "jaccard", "pmi", "cosine_tfidf", "ensemble"],
            index=0,
            help="embedding: latent SVD cosine (fast, scalable). Legacy: pairwise metrics.",
        )
        n_clusters = c2.number_input("Clusters", 2, 15, 6)
        max_depth = c3.number_input("Max tree depth", 2, 6, 3)
        top_n = c4.number_input("Top N for heatmap", 10, 60, 25)

        with st.expander("Advanced", expanded=False):
            a1, a2 = st.columns(2)
            n_components = a1.number_input("Embedding dims", 16, 256, 64, step=16)
            top_n_products = a2.number_input("Max products analyzed", 200, 5000, 2000, step=200)

    attrs = build_transaction_derived_attributes(df)
    sim = build_similarity_matrix(
        df,
        method=method,
        n_components=n_components,
        top_n_products=top_n_products,
    )
    clusters = get_cluster_assignments(df, similarity_matrix=sim, n_clusters=n_clusters)
    tree = build_cdt(
        attrs,
        sim,
        cluster_assignments=clusters.set_index("stockcode")["cluster"].to_dict(),
        max_depth=max_depth,
    )
    nodes, products = tree_to_dataframe(tree)

    st.caption(
        f"Tree: {len(nodes)} nodes ({nodes['is_leaf'].sum()} leaves), {len(nodes[~nodes['is_leaf'].astype(bool)])} splits"
    )

    st.divider()
    _render_similarity_heatmap(sim, top_n=top_n)

    st.divider()
    _render_dendrogram(sim)

    st.divider()
    _render_tree_sankey(nodes)

    st.divider()
    _render_attribute_importance(nodes)

    st.divider()
    _render_tree_table(nodes)


MODE_SPEC: ModeSpec = ModeSpec(
    key="cdt",
    label="CDT",
    icon=":material/account_tree:",
    handler=render,
)
