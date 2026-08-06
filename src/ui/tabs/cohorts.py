"""Cohort Analysis tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.cohort import compute_cohorts
from src.ui.registry import ModeSpec


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/timeline: Cohort Retention")

    with st.expander("Parameters", expanded=True):
        period = st.selectbox("Cohort period", ["W", "M"], index=0)
        show_heatmap = st.checkbox("Show heatmap", value=True)

    cohort_table = compute_cohorts(df, cohort_period=period)

    if cohort_table.empty:
        st.warning("No cohort data available.")
        return

    # Pivot for heatmap
    pivot = cohort_table.pivot(index="cohort", columns="period_index", values="retention_rate")
    pivot = pivot * 100  # percentage

    if show_heatmap:
        st.caption("Retention Rate (%)")
        st.dataframe(
            pivot.style.format("{:.1f}").background_gradient(cmap="RdYlGn", axis=None),
            use_container_width=True,
        )

    st.caption("Raw Cohort Data")
    st.dataframe(cohort_table, use_container_width=True, hide_index=True)


MODE_SPEC: ModeSpec = ModeSpec(
    key="cohorts",
    label="Cohorts",
    icon=":material/timeline:",
    handler=render,
)