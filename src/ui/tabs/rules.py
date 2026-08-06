"""Association Rules tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.rules import create_basket_matrix, run_fpgrowth, generate_rules, filter_rules, rules_to_table
from src.analytics.data import derive_product_lookup
from src.ui.registry import ModeSpec


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/schema: Association Rules (FP-Growth)")

    with st.expander("Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        min_support = c1.number_input("Min Support", 0.001, 0.5, 0.01, 0.001)
        max_len = c2.number_input("Max Itemset Length", 2, 5, 3)
        min_threshold = c3.number_input("Min Confidence", 0.01, 1.0, 0.05, 0.01)

    basket = create_basket_matrix(df)
    st.caption(f"Basket matrix: {basket.shape[0]} transactions × {basket.shape[1]} products")

    freq = run_fpgrowth(basket, min_support=min_support, max_len=max_len)
    st.caption(f"Frequent itemsets: {len(freq)}")

    if freq.empty:
        st.warning("No frequent itemsets found with current parameters.")
        return

    rules = generate_rules(freq, min_threshold=min_threshold)
    st.caption(f"Rules generated: {len(rules)}")

    if rules.empty:
        st.warning("No rules meet the confidence threshold.")
        return

    filtered = filter_rules(rules, min_lift=1.0, min_confidence=min_threshold)
    st.caption(f"Rules after filtering (lift ≥ 1.0): {len(filtered)}")

    if not filtered.empty:
        lookup = derive_product_lookup(df)
        table = rules_to_table(filtered, lookup)
        st.dataframe(table, use_container_width=True, hide_index=True)

        csv = table.to_csv(index=False)
        st.download_button(
            ":material/download: Download Rules CSV",
            csv,
            "association_rules.csv",
            "text/csv",
        )


MODE_SPEC: ModeSpec = ModeSpec(
    key="rules",
    label="Association Rules",
    icon=":material/schema:",
    handler=render,
)