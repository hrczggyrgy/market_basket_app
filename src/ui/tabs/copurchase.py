"""Co-purchase / Affinity tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.copurchase import get_top_affinity_pairs, get_product_affinity_profile
from src.ui.registry import ModeSpec


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/link: Co-purchase Affinity")

    with st.expander("Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        top_n = c1.number_input("Top N pairs", 5, 200, 20)
        min_cooccurrence = c2.number_input("Min co-occurrence", 1, 50, 5)
        top_n_products = c3.number_input("Candidate pool (top products)", 50, 500, 200)

    pairs = get_top_affinity_pairs(df, top_n=top_n, min_cooccurrence=min_cooccurrence, top_n_products=top_n_products)

    if pairs.empty:
        st.warning("No affinity pairs found.")
        return

    st.dataframe(pairs, use_container_width=True, hide_index=True)

    # Product affinity profile
    st.divider()
    st.subheader(":material/search: Product Affinity Profile")
    product = st.selectbox("Select product", df["stockcode"].unique())
    if product:
        profile = get_product_affinity_profile(df, product, top_n=10)
        st.dataframe(profile, use_container_width=True, hide_index=True)


MODE_SPEC: ModeSpec = ModeSpec(
    key="copurchase",
    label="Co-purchase",
    icon=":material/link:",
    handler=render,
)