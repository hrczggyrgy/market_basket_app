"""Customer Segmentation tab."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.segmentation import (
    compute_rfm_features,
    compute_segment_migration,
    compute_segment_radar,
    rfm_segmentation,
    behavioral_segmentation,
    value_based_segmentation,
)
from src.ui.plots import PALETTE, empty_state, new_fig, show
from src.ui.registry import ModeSpec


def _segment_radar(df: pd.DataFrame) -> None:
    st.subheader(":material/radar: Segment Radar")
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
        "Wide petals = the segment over-indexes on that behavior (e.g. purchase "
        "frequency, revenue, basket size)."
    )


def _segment_migration(df: pd.DataFrame) -> None:
    st.subheader(":material/swap_horiz: Segment Migration (first vs second half)")
    migration = compute_segment_migration(df, n_clusters=4)
    if migration.empty:
        st.caption("Not enough stable segments to trace migration (customers must be present in both halves).")
        return

    stay = migration[migration["segment_from"] == migration["segment_to"]]
    move = migration[migration["segment_from"] != migration["segment_to"]]

    # Sankey for the off-diagonal flows
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
        "Thickness = number of customers migrating."
    )

    st.dataframe(
        migration.sort_values("customers", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


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

        _segment_radar(df)
        _segment_migration(df)

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