"""Promotional Analytics tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.promo import (
    detect_promotions,
    compute_promo_baseline,
    calculate_promotional_lift,
    compute_incrementality_waterfall,
    promo_roi_analysis,
)
from src.ui.registry import ModeSpec


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/local_offer: Promotional Analytics")

    # First detect promotions
    promos = detect_promotions(df)

    tab1, tab2, tab3, tab4 = st.tabs(["Promo Periods", "Lift Analysis", "Waterfall", "ROI"])

    with tab1:
        if not promos.empty:
            st.dataframe(promos, use_container_width=True, hide_index=True)
        else:
            st.info("No promotional periods detected.")

    with tab2:
        if not promos.empty:
            lift = calculate_promotional_lift(df, promo_periods=promos)
            if not lift.empty:
                st.dataframe(lift, use_container_width=True, hide_index=True)
            else:
                st.info("No significant promotional lift detected.")
        else:
            st.info("No promotional periods detected.")

    with tab3:
        if not promos.empty:
            baseline_df = compute_promo_baseline(df, promo_periods=promos)
            if not baseline_df.empty:
                waterfall = compute_incrementality_waterfall(baseline_df)
                if not waterfall.empty:
                    st.dataframe(waterfall, use_container_width=True, hide_index=True)
                else:
                    st.info("No incrementality waterfall available.")
            else:
                st.info("No baseline data available.")
        else:
            st.info("No promotional periods detected.")

    with tab4:
        if not promos.empty:
            roi = promo_roi_analysis(df, promo_periods=promos)
            if not roi.empty:
                st.dataframe(roi, use_container_width=True, hide_index=True)
            else:
                st.info("No ROI data available.")
        else:
            st.info("No promotional periods detected.")


MODE_SPEC: ModeSpec = ModeSpec(
    key="promo",
    label="Promotions",
    icon=":material/local_offer:",
    handler=render,
)