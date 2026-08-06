"""Product Performance tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.performance import (
    compute_product_metrics,
    abc_analysis,
    xyz_analysis,
    product_lifecycle_stage,
    compute_velocity,
    compute_repeat_rate,
)
from src.ui.registry import ModeSpec


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/insights: Product Performance")

    pm = compute_product_metrics(df)
    abc = abc_analysis(df)
    xyz = xyz_analysis(df)
    lifecycle = product_lifecycle_stage(df)
    velocity = compute_velocity(df)
    repeat = compute_repeat_rate(df)

    # Merge all
    perf = pm.merge(abc[["stockcode", "abc_class"]], on="stockcode", how="left")
    perf = perf.merge(xyz[["stockcode", "xyz_class"]], on="stockcode", how="left")
    perf = perf.merge(lifecycle[["stockcode", "stage"]], on="stockcode", how="left")
    perf = perf.merge(velocity[["stockcode", "velocity"]], on="stockcode", how="left")
    perf = perf.merge(repeat[["stockcode", "repeat_rate"]], on="stockcode", how="left")

    # Filters
    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        abc_filter = c1.multiselect("ABC Class", ["A", "B", "C"], default=["A", "B", "C"])
        xyz_filter = c2.multiselect("XYZ Class", ["X", "Y", "Z"], default=["X", "Y", "Z"])
        stage_filter = c3.multiselect("Lifecycle", ["growth", "mature", "decline"], default=["growth", "mature", "decline"])

    filtered = perf[
        perf["abc_class"].isin(abc_filter) &
        perf["xyz_class"].isin(xyz_filter) &
        perf["stage"].isin(stage_filter)
    ]

    # Display
    display_cols = ["stockcode", "revenue", "units", "transactions", "customers",
                    "abc_class", "xyz_class", "stage", "velocity", "repeat_rate"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].sort_values("revenue", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    # Summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Products", len(filtered))
    c2.metric("Total Revenue", f"${filtered['revenue'].sum():,.0f}")
    c3.metric("Avg Velocity", f"{filtered['velocity'].mean():.2f}")


MODE_SPEC: ModeSpec = ModeSpec(
    key="performance",
    label="Performance",
    icon=":material/insights:",
    handler=render,
)