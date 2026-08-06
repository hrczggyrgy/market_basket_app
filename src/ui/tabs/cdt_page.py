"""Customer Decision Tree (CDT) tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.cdt import (
    build_transaction_derived_attributes,
    build_similarity_matrix,
    get_cluster_assignments,
    build_cdt,
    tree_to_dataframe,
)
from src.ui.registry import ModeSpec


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/account_tree: Customer Decision Tree")

    with st.expander("Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        method = c1.selectbox("Similarity", ["phi", "jaccard", "pmi", "cosine_tfidf", "ensemble"])
        n_clusters = c2.number_input("Clusters", 2, 15, 6)
        max_depth = c3.number_input("Max tree depth", 2, 6, 3)

    attrs = build_transaction_derived_attributes(df)
    sim = build_similarity_matrix(df, method=method)
    clusters = get_cluster_assignments(df, similarity_matrix=sim, n_clusters=n_clusters)
    tree = build_cdt(attrs, sim, cluster_assignments=clusters.set_index("stockcode")["cluster"].to_dict(),
                     max_depth=max_depth)

    nodes, products = tree_to_dataframe(tree)

    st.caption(f"Tree: {len(nodes)} nodes, {nodes['is_leaf'].sum()} leaves")
    st.dataframe(nodes, use_container_width=True, hide_index=True)

    if st.checkbox("Show leaf products"):
        st.dataframe(products, use_container_width=True, hide_index=True)


MODE_SPEC: ModeSpec = ModeSpec(
    key="cdt",
    label="CDT",
    icon=":material/account_tree:",
    handler=render,
)