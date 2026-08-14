"""Co-purchase / Affinity tab."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.copurchase import (
    compute_cooccurrence_matrix,
    compute_pair_centrality,
    compute_pair_trend,
    get_product_affinity_profile,
    get_top_affinity_pairs,
)
from src.ui.features import get_segment_maps
from src.ui.plots import PALETTE, empty_state, new_fig, show
from src.ui.registry import ModeSpec


def _get_available_segments(df: pd.DataFrame) -> dict[str, list[str]]:
    """Detect available segment columns and their unique values."""
    segment_cols = [c for c in df.columns if c.endswith("_segment") or c == "basket_mission"]
    available = {}
    for col in segment_cols:
        values = sorted(df[col].dropna().unique().tolist())
        if values:
            available[col] = values
    return available


def _render_heatmap(
    df: pd.DataFrame,
    top_n: int,
    segment_col: str | None = None,
    segment_val: str | None = None,
    mission_col: str | None = None,
    mission_val: str | None = None,
) -> None:
    st.subheader(":material/table_chart: Co-occurrence Heatmap")
    cooccurrence = compute_cooccurrence_matrix(
        df,
        top_n_products=top_n,
        segment_col=segment_col,
        segment_val=segment_val,
        mission_col=mission_col,
        mission_val=mission_val,
    )
    if cooccurrence.empty:
        show(empty_state("No co-occurrence data"))
        return

    top = cooccurrence.index[:top_n]
    matrix = cooccurrence.loc[top, top]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.to_numpy(),
            x=[str(c) for c in matrix.columns],
            y=[str(i) for i in matrix.index],
            colorscale="Blues",
            zmin=0,
            showscale=False,
        )
    )
    fig.update_layout(
        xaxis={"tickangle": -45},
        yaxis={"tickangle": -45},
        height=max(420, 18 * top_n),
    )
    show(fig)
    filter_desc = []
    if segment_col and segment_val:
        filter_desc.append(f"{segment_col}={segment_val}")
    if mission_col and mission_val:
        filter_desc.append(f"{mission_col}={mission_val}")
    filter_str = " | ".join(filter_desc) if filter_desc else "all"
    st.caption(
        f"Shared transactions between the top {top_n} products by purchase frequency ({filter_str})."
    )


def _render_centrality_network(
    df: pd.DataFrame,
    top_n: int,
    min_cooccurrence: int,
    segment_col: str | None = None,
    segment_val: str | None = None,
    mission_col: str | None = None,
    mission_val: str | None = None,
) -> None:
    st.subheader(":material/hub: Co-purchase Network Centrality")
    centrality = compute_pair_centrality(
        df,
        top_n_products=top_n,
        min_cooccurrence=min_cooccurrence,
        segment_col=segment_col,
        segment_val=segment_val,
        mission_col=mission_col,
        mission_val=mission_val,
    )
    if centrality.empty:
        show(empty_state("No connected pairs at current thresholds"))
        return

    cooccurrence = compute_cooccurrence_matrix(
        df,
        top_n_products=top_n,
        segment_col=segment_col,
        segment_val=segment_val,
        mission_col=mission_col,
        mission_val=mission_val,
    )
    graph = nx.Graph()
    values = cooccurrence.to_numpy()
    for i in range(len(cooccurrence.index)):
        for j in range(i + 1, len(cooccurrence.index)):
            if values[i, j] >= min_cooccurrence:
                graph.add_edge(
                    cooccurrence.index[i], cooccurrence.columns[j], weight=float(values[i, j])
                )

    pos = nx.spring_layout(graph, seed=42, k=0.6)
    edge_x: list[float] = []
    edge_y: list[float] = []
    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, float("nan")]
        edge_y += [y0, y1, float("nan")]

    pr = centrality.set_index("stockcode")["pagerank"]
    node_sizes = [10 + 40 * pr.get(n, 0.0) for n in graph.nodes()]

    fig = new_fig()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"color": "#B0B0B0", "width": 1},
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[pos[n][0] for n in graph.nodes()],
            y=[pos[n][1] for n in graph.nodes()],
            mode="markers+text",
            text=[str(n) for n in graph.nodes()],
            textposition="bottom center",
            marker={
                "size": node_sizes,
                "color": PALETTE[4],
                "line": {"color": "white", "width": 1},
            },
            hoverinfo="text",
            hovertext=[f"{n}<br>PageRank: {pr.get(n, 0.0):.4f}" for n in graph.nodes()],
        )
    )
    fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False}, showlegend=False)
    show(fig)
    st.caption("Node size = PageRank centrality. Larger nodes are basket 'anchors'.")


def _render_pair_trends(
    df: pd.DataFrame,
    pairs: pd.DataFrame,
    top_n: int,
    segment_col: str | None = None,
    segment_val: str | None = None,
    mission_col: str | None = None,
    mission_val: str | None = None,
) -> None:
    st.subheader(":material/show_chart: Top Pair Trends")
    top = pairs.head(top_n)
    fig = new_fig()
    added = 0
    for _, row in top.iterrows():
        trend = compute_pair_trend(
            df,
            row["product_a"],
            row["product_b"],
            segment_col=segment_col,
            segment_val=segment_val,
            mission_col=mission_col,
            mission_val=mission_val,
        )
        if trend.empty:
            continue
        label = f"{row['product_a']} + {row['product_b']}"
        fig.add_trace(
            go.Scatter(
                x=trend["period"],
                y=trend["cooccurrence"],
                mode="lines",
                name=label,
                line={"width": 1.5},
            )
        )
        added += 1
    if added == 0:
        show(empty_state("No pair trends available"))
        return
    fig.update_layout(yaxis={"title": "Co-occurring transactions"}, xaxis={"title": "Period"})
    show(fig)


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/link: Co-purchase Affinity")

    # Add segment/mission columns (cached - RFM clustering + mission are expensive)
    customer_segments, baskets = get_segment_maps(df)
    if not customer_segments.empty:
        df = df.merge(customer_segments, on="customer_id", how="left")
    if not baskets.empty:
        df = df.merge(baskets, on="transaction_id", how="left")

    # Detect available filters
    available_segments = _get_available_segments(df)

    with st.expander("Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        top_n = c1.number_input("Top N pairs", 5, 200, 20)
        min_cooccurrence = c2.number_input("Min co-occurrence", 1, 50, 5)
        top_n_products = c3.number_input("Candidate pool (top products)", 50, 500, 200)

        # Segment filter
        segment_col = None
        segment_val = None
        if available_segments:
            segment_col = st.selectbox(
                "Customer Segment",
                options=[None] + list(available_segments.keys()),
                format_func=lambda x: "None" if x is None else x,
                key="copurchase_segment_col",
            )
            if segment_col:
                segment_val = st.selectbox(
                    f"{segment_col} value",
                    options=available_segments[segment_col],
                    key="copurchase_segment_val",
                )

        # Mission filter
        mission_col = None
        mission_val = None
        if "basket_mission" in available_segments:
            mission_col = "basket_mission"
            mission_val = st.selectbox(
                "Basket Mission",
                options=["All"] + available_segments["basket_mission"],
                key="copurchase_mission_val",
            )
            if mission_val == "All":
                mission_col = None
                mission_val = None

    pairs = get_top_affinity_pairs(
        df,
        top_n=top_n,
        min_cooccurrence=min_cooccurrence,
        top_n_products=top_n_products,
        segment_col=segment_col,
        segment_val=segment_val,
        mission_col=mission_col,
        mission_val=mission_val,
    )

    if pairs.empty:
        st.warning("No affinity pairs found.")
        return

    st.divider()
    _render_heatmap(
        df,
        top_n=min(top_n_products, 25),
        segment_col=segment_col,
        segment_val=segment_val,
        mission_col=mission_col,
        mission_val=mission_val,
    )

    st.divider()
    _render_centrality_network(
        df,
        top_n=top_n_products,
        min_cooccurrence=min_cooccurrence,
        segment_col=segment_col,
        segment_val=segment_val,
        mission_col=mission_col,
        mission_val=mission_val,
    )

    st.divider()
    _render_pair_trends(
        df,
        pairs,
        top_n=min(top_n, 10),
        segment_col=segment_col,
        segment_val=segment_val,
        mission_col=mission_col,
        mission_val=mission_val,
    )

    st.divider()
    st.subheader(":material/table_rows: Top Pairs")
    st.dataframe(pairs, use_container_width=True, hide_index=True)

    # Product affinity profile
    st.divider()
    st.subheader(":material/search: Product Affinity Profile")
    product = st.selectbox("Select product", df["stockcode"].unique())
    if product:
        profile = get_product_affinity_profile(
            df,
            product,
            top_n=10,
            segment_col=segment_col,
            segment_val=segment_val,
            mission_col=mission_col,
            mission_val=mission_val,
        )
        st.dataframe(profile, use_container_width=True, hide_index=True)


MODE_SPEC: ModeSpec = ModeSpec(
    key="copurchase",
    label="Co-purchase",
    icon=":material/link:",
    handler=render,
    requires=("sufficient_baskets_200", "sufficient_skus_20"),
)
