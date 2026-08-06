"""Product Switching tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.switching import compute_switching_matrix, get_top_switching_paths, get_customer_loyalty_metrics
from src.ui.registry import ModeSpec


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/swap_horiz: Product Switching")

    with st.expander("Parameters", expanded=True):
        c1, c2 = st.columns(2)
        window_days = c1.number_input("Window (days)", 30, 180, 90)
        min_txns = c2.number_input("Min transactions per customer", 2, 10, 3)

    switching = compute_switching_matrix(df, window_days=window_days, min_transactions=min_txns)

    if switching.empty:
        st.warning("No switching patterns found.")
        return

    st.metric("Total transitions", len(switching))

    top_paths = get_top_switching_paths(df, top_n=20, window_days=window_days, min_transactions=min_txns)
    st.dataframe(top_paths, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader(":material/person: Customer Loyalty Metrics")
    loyalty = get_customer_loyalty_metrics(df)
    st.dataframe(loyalty, use_container_width=True, hide_index=True)


MODE_SPEC: ModeSpec = ModeSpec(
    key="switching",
    label="Switching",
    icon=":material/swap_horiz:",
    handler=render,
)