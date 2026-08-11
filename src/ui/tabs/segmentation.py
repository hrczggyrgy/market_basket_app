"""Customer Segmentation tab — the customer-intelligence hub.

Page pattern: segment scorecard -> segment landscape (value x frequency) ->
differentiation (radar) -> migration -> full tables.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.segmentation import (
    behavioral_segmentation,
    compute_rfm_features,
    compute_segment_migration,
    compute_segment_radar,
    rfm_segmentation,
    value_based_segmentation,
)
from src.ui.plots import PALETTE, empty_state, new_fig, show
from src.ui.registry import ModeSpec


def _segment_landscape(seg: pd.DataFrame, value_col: str, label_col: str) -> None:
    """Value x frequency landscape sized by customer count."""
    agg = (
        seg.groupby(label_col)
        .agg(
            customers=(seg.columns[0], "count"),
            total_value=(value_col, "sum"),
            avg_frequency=(value_col, "mean"),
        )
        .reset_index()
    )
    if agg.empty or len(agg) < 2:
        show(empty_state("Not enough segments to map"))
        return

    fig = px.scatter(
        agg,
        x="avg_frequency",
        y="total_value",
        size="customers",
        color=label_col,
        hover_data=["customers"],
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(
        xaxis={"title": "Avg value per customer"},
        yaxis={"title": "Total segment value"},
        height=420,
    )
    show(fig)
    st.caption(
        "Segment landscape: x = per-customer value, y = total segment value, "
        "size = customer count. Big + far-right segments are the value engine; "
        "large + far-left segments are the growth pool."
    )


def _segment_revenue_share(seg: pd.DataFrame, value_col: str, label_col: str) -> None:
    """Share of total value per segment."""
    shares = (
        seg.groupby(label_col)[value_col]
        .sum()
        .sort_values(ascending=False)
        .pipe(lambda s: s / s.sum() * 100)
    )
    fig = go.Figure(
        data=[
            go.Pie(
                labels=shares.index,
                values=shares.values,
                hole=0.4,
                marker={"colors": PALETTE[: len(shares)]},
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(height=320)
    show(fig)
    st.caption("Share of total customer value per segment. A concentrated pie = the base is fragile.")


def _segment_radar(df: pd.DataFrame) -> None:
    st.subheader(":material/radar: Segment Differentiation (Radar)")
    radar = compute_segment_radar(df, n_clusters=4)
    if radar.empty:
        st.caption("Not enough distinct segments to profile (fewer than 2 non-trivial clusters).")
        return

    features = list(radar["feature"].unique())
    segments = list(radar["segment"].unique())

    fig = go.Figure()
    for i, seg in enumerate(segments):
        row = radar[radar["segment"] == seg].set_index("feature")["normalized_value"]
        values = [row[f] for f in features]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=features + [features[0]],
                fill="toself",
                opacity=0.25,
                name=seg,
                line={"color": PALETTE[i % len(PALETTE)]},
            )
        )

    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        height=440,
        margin={"t": 30, "b": 30},
    )
    show(fig)
    st.caption(
        "Segment profile vs all segments (min-max normalized per axis, 0-1). "
        "Wide petals = the segment over-indexes on that behavior."
    )


def _segment_migration(df: pd.DataFrame) -> None:
    st.subheader(":material/swap_horiz: Segment Migration (first vs second half)")
    migration = compute_segment_migration(df, n_clusters=4)
    if migration.empty:
        st.caption("Not enough stable segments to trace migration (customers must be present in both halves).")
        return

    stay = migration[migration["segment_from"] == migration["segment_to"]]
    move = migration[migration["segment_from"] != migration["segment_to"]]

    top = move.nlargest(20, "customers")
    sources = top["segment_from"].tolist()
    targets = top["segment_to"].tolist()
    values = top["customers"].tolist()

    nodes = list(dict.fromkeys(sources + targets))
    label_to_idx = {s: i for i, s in enumerate(nodes)}
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node={
                    "label": [f"First half: {n}" for n in nodes],
                    "color": PALETTE[0],
                    "pad": 15,
                    "thickness": 20,
                },
                link={
                    "source": [label_to_idx[s] for s in sources],
                    "target": [label_to_idx[t] for t in targets],
                    "value": values,
                    "color": "rgba(255, 140, 0, 0.4)",
                },
            )
        ]
    )
    fig.update_layout(height=max(400, 30 * len(nodes)), font={"size": 10})
    show(fig)

    moved = int(move["customers"].sum())
    stayed = int(stay["customers"].sum())
    st.caption(
        f"{moved:,} customers changed segment between halves vs {stayed:,} who stayed. "
        "Migration toward higher-value segments = retention working; the reverse = value leaking."
    )


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/groups: Customer Segmentation")
    st.caption(
        "Segmentation is the customer-intelligence hub: it explains WHO your "
        "revenue depends on, and WHERE value is being won or lost."
    )

    tab1, tab2, tab3 = st.tabs(["RFM", "Behavioral", "Value-Based"])

    with tab1:
        st.subheader(":material/table_chart: RFM Segments")
        rfm = compute_rfm_features(df)
        seg = rfm_segmentation(rfm, method="kmeans", n_segments=5)
        _segment_revenue_share(seg, "monetary", "segment")
        _segment_landscape(seg, "monetary", "segment")
        st.dataframe(seg[["customer_id", "recency_days", "frequency", "monetary", "segment"]],
                     use_container_width=True, hide_index=True)

    with tab2:
        st.subheader(":material/psychology: Behavioral Segments")
        behav = behavioral_segmentation(df, n_clusters=4)
        if isinstance(behav, tuple):
            behav = behav[0]
        _segment_revenue_share(behav, "total_revenue", "segment")
        _segment_landscape(behav, "total_revenue", "segment")
        _segment_radar(df)
        _segment_migration(df)
        st.dataframe(behav[["customer_id", "cluster", "segment", "cluster_confidence"]],
                     use_container_width=True, hide_index=True)

    with tab3:
        st.subheader(":material/attach_money: Value-Based Segments")
        val = value_based_segmentation(df)
        if isinstance(val, tuple):
            val = val[0]
        _segment_revenue_share(val, "predicted_clv", "value_segment")
        _segment_landscape(val, "predicted_clv", "value_segment")
        st.dataframe(val[["customer_id", "recency", "frequency", "monetary", "predicted_clv", "value_segment"]],
                     use_container_width=True, hide_index=True)


MODE_SPEC: ModeSpec = ModeSpec(
    key="segmentation",
    label="Segmentation",
    icon=":material/groups:",
    handler=render,
)
