"""Customer Lifetime Value (CLV) tab."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.clv import compute_clv_customer_df, predict_clv_bg_nbd
from src.ui.plots import PALETTE, empty_state, new_fig, show
from src.ui.registry import ModeSpec


def _render_clv_distribution(customers: pd.DataFrame) -> None:
    st.subheader(":material/bar_chart: CLV Distribution")
    if customers.empty:
        show(empty_state("No CLV data"))
        return

    fig = px.histogram(
        customers,
        x="predicted_clv",
        nbins=40,
        color="clv_segment",
        color_discrete_map={"Platinum": PALETTE[0], "Gold": PALETTE[2], "Silver": PALETTE[4], "Bronze": PALETTE[1]},
        marginal="box",
        hover_data=["customer_id"],
    )
    fig.update_layout(xaxis={"title": "Predicted CLV ($)"}, yaxis={"title": "Count"})
    show(fig)
    st.caption("Histogram of predicted CLV by segment. Box = IQR/median.")


def _render_clv_segments(customers: pd.DataFrame) -> None:
    st.subheader(":material/emoji_events: CLV Segment Summary")
    seg = (
        customers.groupby("clv_segment")
        .agg(
            n_customers=("customer_id", "count"),
            avg_clv=("predicted_clv", "mean"),
            total_clv=("predicted_clv", "sum"),
            avg_freq=("frequency", "mean"),
            avg_p_alive=("p_alive", "mean"),
        )
        .reset_index()
        .sort_values("avg_clv", ascending=False)
    )

    fig = px.bar(
        seg,
        x="clv_segment",
        y="avg_clv",
        color="clv_segment",
        color_discrete_map={"Platinum": PALETTE[0], "Gold": PALETTE[2], "Silver": PALETTE[4], "Bronze": PALETTE[1]},
        text=seg["avg_clv"].apply(lambda x: f"${x:,.0f}"),
        hover_data=["n_customers", "avg_freq", "avg_p_alive"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis={"title": "Segment"}, yaxis={"title": "Avg Predicted CLV ($)"})
    show(fig)
    st.caption("Average predicted CLV per segment. Higher segments = higher future value.")

    st.dataframe(seg, use_container_width=True, hide_index=True)


def _render_frequency_recency(customers: pd.DataFrame) -> None:
    st.subheader(":material/scatter_plot: Frequency vs Recency (BG/NBD space)")
    if customers.empty:
        show(empty_state("No data"))
        return

    fig = px.scatter(
        customers,
        x="frequency",
        y="recency_days",
        size="predicted_clv",
        color="clv_segment",
        color_discrete_map={"Platinum": PALETTE[0], "Gold": PALETTE[2], "Silver": PALETTE[4], "Bronze": PALETTE[1]},
        hover_data=["customer_id", "predicted_clv", "p_alive"],
        log_y=True,
    )
    fig.update_layout(xaxis={"title": "Frequency (repeat purchases)"}, yaxis={"title": "Recency (days)"})
    show(fig)
    st.caption("Top-right = high frequency, high recency (active loyalists). Bottom-right = high freq, long ago (at-risk).")


def _render_p_alive_vs_clv(customers: pd.DataFrame) -> None:
    st.subheader(":material/analytics: P(Alive) vs Predicted CLV")
    fig = px.scatter(
        customers,
        x="p_alive",
        y="predicted_clv",
        size="frequency",
        color="clv_segment",
        color_discrete_map={"Platinum": PALETTE[0], "Gold": PALETTE[2], "Silver": PALETTE[4], "Bronze": PALETTE[1]},
        hover_data=["customer_id", "recency_days"],
        log_y=True,
    )
    fig.update_layout(xaxis={"title": "Probability Alive"}, yaxis={"title": "Predicted CLV ($)"})
    show(fig)
    st.caption("P(Alive) from BG/NBD. Top-right = alive + high value (retention priority).")


def _render_clv_diagnostics(diagnostics: pd.DataFrame) -> None:
    st.subheader(":material/construction: Model Diagnostics")
    if diagnostics.empty:
        show(empty_state("No diagnostics"))
        return

    st.dataframe(diagnostics, use_container_width=True, hide_index=True)

    # Key parameters
    key_params = diagnostics[diagnostics["metric"].isin(["bgf_r", "bgf_alpha", "bgf_a", "bgf_b", "ggf_p", "ggf_q", "ggf_v"])]
    if not key_params.empty:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=key_params["metric"],
                    y=key_params["value"],
                    marker={"color": PALETTE[0]},
                    text=key_params["value"].apply(lambda v: f"{v:.3f}"),
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(yaxis={"title": "Parameter Value", "type": "log"}, xaxis={"title": "Parameter"})
        show(fig)
        st.caption("BG/NBD: r=shape, alpha=scale; GG: p=shape, q=scale, v=avg profit. Higher r/alpha = lower frequency/recency variability.")


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/account_balance_wallet: Customer Lifetime Value")

    with st.expander("Parameters", expanded=True):
        c1, c2 = st.columns(2)
        horizon = c1.number_input("Prediction Horizon (days)", 30, 365, 90)
        freq = c2.selectbox("Frequency", ["D", "W"], index=0)

    try:
        predictions, diagnostics = predict_clv_bg_nbd(df, prediction_horizon_days=horizon, freq=freq)
    except ValueError as e:
        st.error(f"CLV model failed: {e}")
        return

    st.divider()
    _render_clv_diagnostics(diagnostics)

    st.divider()
    customers = compute_clv_customer_df(df, prediction_horizon_days=horizon, freq=freq)

    # Filters
    with st.expander("Filters", expanded=True):
        c1, c2 = st.columns(2)
        segment_filter = c1.multiselect(
            "CLV Segment",
            customers["clv_segment"].unique().tolist(),
            default=customers["clv_segment"].unique().tolist(),
        )
        top_n = c2.number_input("Top N by CLV", 10, 200, 50)

    filtered = customers[customers["clv_segment"].isin(segment_filter)].head(top_n)

    st.divider()
    _render_clv_distribution(filtered)

    st.divider()
    _render_clv_segments(filtered)

    st.divider()
    _render_frequency_recency(filtered)

    st.divider()
    _render_p_alive_vs_clv(filtered)

    st.divider()
    st.subheader(":material/table_rows: Customer CLV Detail")
    display_cols = [
        "customer_id", "frequency", "recency_days", "total_revenue",
        "avg_order_value", "p_alive", "predicted_purchases",
        "expected_avg_value", "predicted_clv", "clv_12m", "clv_segment",
        "entropy", "normalized_entropy",
    ]
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