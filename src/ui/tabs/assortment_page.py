"""Assortment Optimization tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.assortment import (
    optimize_assortment_heuristic,
    evaluate_assortment,
    compare_assortment_scenarios,
    build_solution_table,
)
from src.analytics.performance import compute_product_metrics
from src.ui.registry import ModeSpec


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/inventory_2: Assortment Optimization")

    with st.expander("Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        max_skus = c1.number_input("Max SKUs", 10, 200, 50)
        min_coverage = c2.number_input("Min Coverage", 0.1, 1.0, 0.8, 0.05)
        objective = c3.selectbox("Objective", ["revenue", "margin"])

    selected, metrics = optimize_assortment_heuristic(
        df, max_skus=max_skus, min_coverage=min_coverage, objective=objective
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SKUs Selected", len(selected))
    col2.metric("Coverage", f"{metrics.get('coverage', 0):.1%}")
    col3.metric("Kept Revenue", f"${metrics.get('kept_revenue', 0):,.0f}")
    col4.metric("Recovery Rate", f"{metrics.get('recovery_rate', 0):.1%}")

    # Scenario comparison
    st.divider()
    st.subheader(":material/compare_arrows: Scenario Comparison")
    scenarios = compare_assortment_scenarios(df, [])
    if not scenarios.empty:
        st.dataframe(scenarios, use_container_width=True, hide_index=True)

    # Selected SKU table
    st.divider()
    st.subheader(":material/table_rows: Selected Assortment")
    from src.analytics.performance import compute_product_metrics
    revenue = compute_product_metrics(df).set_index("stockcode")["revenue"]
    table = build_solution_table(selected, revenue)
    st.dataframe(table, use_container_width=True, hide_index=True)


MODE_SPEC: ModeSpec = ModeSpec(
    key="assortment",
    label="Assortment",
    icon=":material/inventory_2:",
    handler=render,
)