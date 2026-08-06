"""Customer Lifetime Value (CLV) tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.clv import predict_clv_bg_nbd, compute_clv_customer_df
from src.ui.registry import ModeSpec


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/account_balance_wallet: Customer Lifetime Value")

    with st.expander("Parameters", expanded=True):
        horizon = st.number_input("Prediction Horizon (days)", 30, 365, 90)

    predictions, diagnostics = predict_clv_bg_nbd(df, prediction_horizon_days=horizon)

    st.caption("Model Diagnostics")
    st.json(diagnostics.set_index("metric")["value"].to_dict())

    customers = compute_clv_customer_df(df)

    # Filters
    c1, c2 = st.columns(2)
    segment_filter = c1.multiselect(
        "CLV Segment",
        customers["clv_segment"].unique().tolist(),
        default=customers["clv_segment"].unique().tolist()
    )
    top_n = c2.number_input("Top N by CLV", 10, 200, 20)

    filtered = customers[customers["clv_segment"].isin(segment_filter)].head(top_n)

    display_cols = ["customer_id", "frequency", "recency_days", "total_revenue",
                    "avg_order_value", "p_alive", "predicted_purchases",
                    "expected_avg_value", "predicted_clv", "clv_12m", "clv_segment"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].sort_values("predicted_clv", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    # Summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", len(customers))
    c2.metric("Total Predicted CLV", f"${customers['predicted_clv'].sum():,.0f}")
    c3.metric("Avg CLV", f"${customers['predicted_clv'].mean():,.0f}")


MODE_SPEC: ModeSpec = ModeSpec(
    key="clv",
    label="CLV",
    icon=":material/account_balance_wallet:",
    handler=render,
)