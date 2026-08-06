"""Customer Segmentation tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.segmentation import (
    compute_rfm_features,
    rfm_segmentation,
    behavioral_segmentation,
    value_based_segmentation,
)
from src.ui.registry import ModeSpec


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/groups: Customer Segmentation")

    tab1, tab2, tab3 = st.tabs(["RFM", "Behavioral", "Value-Based"])

    with tab1:
        rfm = compute_rfm_features(df)
        seg = rfm_segmentation(rfm, method="kmeans", n_segments=5)
        st.dataframe(seg[["customer_id", "recency_days", "frequency", "monetary", "segment"]],
                     use_container_width=True, hide_index=True)

        st.caption("Segment Distribution")
        st.bar_chart(seg["segment"].value_counts())

    with tab2:
        behav = behavioral_segmentation(df, n_clusters=4)
        if isinstance(behav, tuple):
            behav = behav[0]
        st.dataframe(behav[["customer_id", "cluster", "segment", "cluster_confidence"]],
                     use_container_width=True, hide_index=True)

        st.caption("Segment Distribution")
        st.bar_chart(behav["segment"].value_counts())

    with tab3:
        val = value_based_segmentation(df)
        if isinstance(val, tuple):
            val = val[0]
        st.dataframe(val[["customer_id", "recency", "frequency", "monetary", "predicted_clv", "value_segment"]],
                     use_container_width=True, hide_index=True)

        st.caption("Segment Distribution")
        st.bar_chart(val["value_segment"].value_counts())


MODE_SPEC: ModeSpec = ModeSpec(
    key="segmentation",
    label="Segmentation",
    icon=":material/groups:",
    handler=render,
)