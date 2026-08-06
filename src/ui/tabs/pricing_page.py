"""Pricing & Elasticity tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.pricing import (
    estimate_loglog_elasticity,
    estimate_hierarchical_elasticity,
    compute_kvi_score,
    diagnose_price_curves_1d,
)
from src.ui.registry import ModeSpec


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/price_check: Pricing & Elasticity")

    tab1, tab2, tab3 = st.tabs(["Elasticity", "KVI Scores", "Price Curves"])

    with tab1:
        elast = estimate_loglog_elasticity(df, min_periods=5)
        if not elast.empty:
            st.dataframe(elast.sort_values("elasticity"), use_container_width=True, hide_index=True)

            hier = estimate_hierarchical_elasticity(df, min_periods=5)
            if not hier.empty:
                st.caption("Hierarchical (Shrunk) Elasticity")
                st.dataframe(hier[["stockcode", "category", "elasticity_ols", "elasticity_shrunk", "shrink_weight"]],
                             use_container_width=True, hide_index=True)
        else:
            st.warning("Insufficient data for elasticity estimation.")

    with tab2:
        kvi = compute_kvi_score(df, method="heuristic")
        if not kvi.empty:
            st.dataframe(kvi.sort_values("kvi_score", ascending=False).head(20),
                         use_container_width=True, hide_index=True)

    with tab3:
        curves = diagnose_price_curves_1d(df, n_tiers=3)
        if not curves.empty:
            st.dataframe(curves[["stockcode", "category", "median_price", "pack_size_numeric",
                                 "price_per_unit", "tier_label", "has_violation"]],
                         use_container_width=True, hide_index=True)
            violations = curves[curves["has_violation"]]
            if not violations.empty:
                st.warning(f"Price curve violations: {len(violations)}")
                st.dataframe(violations, use_container_width=True, hide_index=True)


MODE_SPEC: ModeSpec = ModeSpec(
    key="pricing",
    label="Pricing",
    icon=":material/price_check:",
    handler=render,
)