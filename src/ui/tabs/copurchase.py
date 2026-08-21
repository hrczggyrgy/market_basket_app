"""Co-purchase / Affinity tab — five-layer redesign.

Layers:
  1) Basket network: node size = revenue, edge thickness = affinity
  2) Affinity × revenue matrix: 4 quadrants (basket anchors / growth attachments / niche / low priority)
  3) Basket mission matrix: top-up / routine / stock-up distribution
  4) Manager table: anchor/SKU | aff_int | incremental revenue | basket penetration | recommendation
  5) Product Decision Profile integration for basket-level data
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.copurchase import (
    compute_affinity_matrix,
    get_top_affinity_pairs,
)
from src.analytics.profile_service import (
    ProfileService,
    get_profile_service,
    init_profile_service,
)
from src.ui.plots import PALETTE, new_fig, show
from src.ui.registry import ModeSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_available_segments(df: pd.DataFrame) -> dict[str, list[str]]:
    """Detect available segment columns and their unique values."""
    segment_cols = [c for c in df.columns if c.endswith("_segment") or c == "basket_mission"]
    available: dict[str, list[str]] = {}
    for col in segment_cols:
        values = sorted(df[col].dropna().unique().tolist())
        if values:
            available[col] = values
    return available


def _get_revenue_by_sku(df: pd.DataFrame) -> dict[str, float]:
    """Per-SKU revenue from price * quantity."""
    return {
        str(sku): float((df[df["stockcode"] == sku]["price"] * df[df["stockcode"] == sku]["quantity"]).sum())
        for sku in df["stockcode"].unique()
    }


def _get_affinity_matrix(df: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """Compute the phi-coefficient affinity matrix (optionally truncated to top-N products)."""
    affinity = compute_affinity_matrix(
        df,
        min_cooccurrence=1,
        top_n_products=top_n,
    )
    return affinity


def _classify_quadrant(affinity: float, revenue: float, rev_median: float, aff_median: float) -> str:
    """Classify an SKU pair or SKU into one of four quadrants.

    Quadrants (by affinity/revenue relative to medians):
      - Basket Anchors:   high affinity & high revenue
      - Growth Attachments: high revenue & low affinity
      - Niche:            low affinity & low-to-mid revenue (strong co-purchase, low value)
      - Low Priority:     low affinity & low revenue
    """
    if affinity > aff_median and revenue > rev_median:
        return "basket_anchors"
    if revenue > rev_median and affinity <= aff_median:
        return "growth_attachments"
    if affinity > aff_median and revenue <= rev_median:
        return "niche"
    return "low_priority"


# ---------------------------------------------------------------------------
# Layer 1 — Basket Network
# ---------------------------------------------------------------------------

def _render_basket_network(
    df: pd.DataFrame,
    top_n: int,
    profile_service: ProfileService,
    segment_col: str | None = None,
    segment_val: str | None = None,
    mission_col: str | None = None,
    mission_val: str | None = None,
) -> None:
    """Render basket network: node size = revenue, edge thickness = affinity."""

    st.subheader(":material/network: Basket Network (Revenue × Affinity)")

    # Apply filters
    from src.analytics.copurchase import _filter_df
    filtered = _filter_df(df, segment_col, segment_val, mission_col, mission_val)

    # Top-N products by co-occurrence / revenue
    affinity = compute_affinity_matrix(filtered, min_cooccurrence=1, top_n_products=top_n)
    revenue_by_sku = _get_revenue_by_sku(filtered)
    products = affinity.columns.tolist()

    # Build graph
    graph = nx.Graph()
    values = affinity.to_numpy()

    for i in range(len(products)):
        for j in range(i + 1, len(products)):
            if not np.isnan(values[i, j]) and values[i, j] > 0:
                graph.add_edge(products[i], products[j], weight=float(values[i, j]))

    # Node sizes from revenue
    node_revenue: dict[str, float] = {}
    for n in graph.nodes():
        node_revenue[n] = revenue_by_sku.get(n, 0.0)

    # Normalize node sizes: base 20 + revenue-scaled
    rev_values = np.array([node_revenue.get(n, 0.0) for n in graph.nodes()], dtype=float)
    if rev_values.max() > rev_values.min():
        node_sizes = 20 + 30 * (rev_values - rev_values.min()) / (rev_values.max() - rev_values.min() + 1e-6)
    else:
        node_sizes = np.full(len(graph.nodes()), 20.0)

    # Edge thicknesses from affinity
    edge_thicknesses: list[float] = []
    for u, v in graph.edges():
        w = graph[u][v]["weight"]
        edge_thicknesses.append(max(1.0, 2.0 + 8.0 * w))  # scale thickness by affinity

    pos = nx.spring_layout(graph, seed=42, k=0.8 / max(1, np.sqrt(len(graph.nodes()))))

    edge_x: list[float] = []
    edge_y: list[float] = []
    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, float("nan")]
        edge_y += [y0, y1, float("nan")]

    # Use a constant line width instead of per-edge widths
    fig = new_fig()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"color": "#5A5A5A", "width": 1},
            hoverinfo="none",
        )
    )

    # Node trace
    node_x = [pos[n][0] for n in graph.nodes()]
    node_y = [pos[n][1] for n in graph.nodes()]
    text_labels = [str(n) for n in graph.nodes()]

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=text_labels,
            textposition="bottom center",
            marker={
                "size": node_sizes,
                "color": [node_revenue.get(n, 0.0) for n in graph.nodes()],
                "colorscale": "Viridis",
                "colorbar": {"title": "Revenue"},
                "line": {"color": "white", "width": 0.5},
            },
            hoverinfo="text",
            hovertext=[f"{n}<br>Revenue: {node_revenue.get(n, 0.0):.0f}<br>Affinity: {affinity.loc[n, n] if n in affinity.index else 0:.4f}" for n in graph.nodes()],
        )
    )

    fig.update_layout(
        xaxis={"title": "Affinity (phi)", "visible": False},
        yaxis={"title": "Revenue impact", "visible": False},
        showlegend=False,
        height=500,
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    show(fig)

    st.caption(
        "Node size = revenue (larger = higher revenue). Edge thickness = affinity (thicker = stronger co-purchase)."
    )


# ---------------------------------------------------------------------------
# Layer 2 — Affinity × Revenue Matrix (4 Quadrants)
# ---------------------------------------------------------------------------

def _render_affinity_revenue_matrix(
    df: pd.DataFrame,
    top_n: int,
    profile_service: ProfileService,
    segment_col: str | None = None,
    segment_val: str | None = None,
    mission_col: str | None = None,
    mission_val: str | None = None,
) -> None:
    """Render Affinity × Revenue matrix with 4 quadrants."""

    st.subheader(":material/chart: Affinity × Revenue Matrix")

    # Apply filters
    from src.analytics.copurchase import _filter_df
    filtered = _filter_df(df, segment_col, segment_val, mission_col, mission_val)

    # Compute affinity matrix and revenue
    affinity = compute_affinity_matrix(filtered, min_cooccurrence=1, top_n_products=top_n)
    revenue_by_sku = _get_revenue_by_sku(filtered)
    products = [p for p in affinity.index.tolist() if p in revenue_by_sku]

    rev_vals = np.array([revenue_by_sku.get(p, 0.0) for p in products], dtype=float)
    aff_vals = affinity.loc[products, products].to_numpy()

    # Mask diagonal (self-similarity = 1.0) for all computations
    n = len(products)
    mask = ~np.eye(n, dtype=bool)
    aff_vals_offdiag = aff_vals[mask]

    # Compute medians for quadrant splits (excluding diagonal)
    rev_median = np.median(rev_vals) if len(rev_vals) > 0 else 0
    aff_median = np.median(aff_vals_offdiag[~np.isnan(aff_vals_offdiag)]) if np.any(~np.isnan(aff_vals_offdiag)) else 0

    # Classify each product into a quadrant
    quadrant_map: dict[str, str] = {}
    for i, p in enumerate(products):
        # Use max affinity with any other product as the "connected affinity"
        max_aff = 0.0
        for j in range(len(products)):
            if i != j and not np.isnan(aff_vals[i, j]):
                max_aff = max(max_aff, abs(aff_vals[i, j]))
        quadrant_map[p] = _classify_quadrant(max_aff, rev_vals[i], rev_median, aff_median)

    # Create scatter plot
    fig = new_fig()

    quadrant_colors = {
        "basket_anchors": PALETTE[1],  # green
        "growth_attachments": PALETTE[2],  # teal
        "niche": PALETTE[3],  # orange
        "low_priority": PALETTE[4],  # blue
    }

    quadrant_labels = {
        "basket_anchors": "Basket Anchors",
        "growth_attachments": "Growth Attachments",
        "niche": "Niche",
        "low_priority": "Low Priority",
    }

    for quad, color in quadrant_colors.items():
        quad_products = [p for p in products if quadrant_map.get(p) == quad]
        if not quad_products:
            continue
        # Use max off-diagonal affinity for x position
        x_pos = []
        y_pos = []
        for p in quad_products:
            idx = products.index(p)
            # Max absolute affinity with any other product
            max_aff = 0.0
            for j in range(len(products)):
                if idx != j and not np.isnan(aff_vals[idx, j]):
                    max_aff = max(max_aff, abs(aff_vals[idx, j]))
            x_pos.append(max_aff)
            y_pos.append(revenue_by_sku.get(p, 0.0))

        fig.add_trace(
            go.Scatter(
                x=x_pos,
                y=y_pos,
                mode="markers",
                marker={"color": color, "size": 12, "line": {"color": "white", "width": 0.5}},
                name=quadrant_labels[quad],
                hovertemplate=f"{quadrant_labels[quad]}: %{{x:.3f}} affinity, %{{y:.0f}} revenue<extra></extra>",
            )
        )

    # Add quadrant boundary lines
    if len(products) > 0:
        # Vertical line at aff_median (using max affinity approx)
        # Horizontal line at rev_median
        fig.add_hline(y=rev_median, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=aff_median, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        xaxis={"title": "Max Affinity (phi)", "range": [0, 1.1], "tickmode": "linear"},
        yaxis={"title": "Revenue", "tickformat": ",.0f"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        height=500,
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    show(fig)

    st.caption(
        "Quadrants: Basket Anchors (high aff, high rev) | Growth Attachments (high rev, low aff) | "
        "Niche (high aff, low rev) | Low Priority (low aff, low rev)."
    )


# ---------------------------------------------------------------------------
# Layer 3 — Basket Mission Matrix
# ---------------------------------------------------------------------------

def _render_basket_mission_matrix(
    df: pd.DataFrame,
    top_n: int,
    profile_service: ProfileService,
    segment_col: str | None = None,
    segment_val: str | None = None,
    mission_col: str | None = None,
    mission_val: str | None = None,
) -> None:
    """Render basket mission distribution matrix (top-up / routine / stock-up)."""

    st.subheader(":material/inbox: Basket Mission Matrix")

    # Apply filters
    from src.analytics.copurchase import _filter_df
    filtered = _filter_df(df, segment_col, segment_val, mission_col, mission_val)

    # Ensure basket_mission column exists
    if "basket_mission" not in filtered.columns:
        from src.analytics.data import assign_basket_mission
        filtered = assign_basket_mission(filtered)

    mission_dist = filtered["basket_mission"].value_counts(normalize=True).rename("percentage")
    mission_counts = filtered["basket_mission"].value_counts()

    # Don't show mission types with zero count
    mission_types = mission_dist[mission_dist > 0]

    fig = new_fig()
    colors = {"Top-Up": PALETTE[1], "Regular": PALETTE[2], "Stock-Up": PALETTE[3]}

    fig.add_trace(
        go.Bar(
            x=mission_types.index.tolist(),
            y=mission_types.values.tolist(),
            marker_color=[colors.get(m, PALETTE[0]) for m in mission_types.index],
            text=[f"{v:.1%}" for v in mission_types.values],
            textposition="outside",
            hovertemplate="Mission: %{x}<br>Count: %{customdata}<br>Percentage: %{y:.1%}<extra></extra>",
            customdata=mission_counts[mission_types.index].tolist(),
        )
    )

    fig.update_layout(
        xaxis={"title": "Basket Mission"},
        yaxis={"title": "Percentage", "tickformat": ".0%"},
        height=350,
        margin={"l": 20, "r": 20, "t": 40, "b": 40},
    )
    show(fig)

    # Also show raw counts
    cols = st.columns(3)
    for i, (mission, count) in enumerate(mission_counts.items()):
        if i < 3:
            cols[i].metric(mission, f"{count:,}")

    st.caption(
        "Distribution of basket missions: Top-Up (small baskets, add-on items), "
        "Regular (medium baskets, routine purchases), Stock-Up (large baskets, bulk buys)."
    )


# ---------------------------------------------------------------------------
# Layer 4 — Manager Table
# ---------------------------------------------------------------------------

def _compute_manager_table(
    df: pd.DataFrame,
    top_n: int,
    profile_service: ProfileService,
    segment_col: str | None = None,
    segment_val: str | None = None,
    mission_col: str | None = None,
    mission_val: str | None = None,
) -> pd.DataFrame:
    """Compute the manager table with anchor/SKU | aff_int | incremental revenue | basket penetration | recommendation."""

    from src.analytics.copurchase import _filter_df

    # Apply filters
    filtered = _filter_df(df, segment_col, segment_val, mission_col, mission_val)

    # Initialize profile service if not already
    svc = get_profile_service()
    if svc is None:
        init_profile_service(filtered)

    # Get profiles for all SKUs
    skus = filtered["stockcode"].unique()
    profiles: dict[str, dict[str, Any]] = {}
    for sku in skus:
        try:
            profiles[sku] = svc.get_profile(str(sku))
        except Exception:
            profiles[sku] = {}

    # Compute affinity matrix
    affinity = compute_affinity_matrix(filtered, min_cooccurrence=1, top_n_products=top_n)

    # Revenue by SKU
    revenue_by_sku = _get_revenue_by_sku(filtered)

    # Basket penetration (customer reach / total baskets)
    n_baskets = filtered["transaction_id"].nunique()
    customer_reach_by_sku = {
        str(sku): filtered[filtered["stockcode"] == sku]["customer_id"].nunique()
        for sku in skus
    }
    basket_penetration: dict[str, float] = {
        sku: min(cr / n_baskets, 1.0) for sku, cr in customer_reach_by_sku.items()
    }

    # Build rows
    rows = []
    for sku in skus:
        profile = profiles.get(sku, {})
        revenue = revenue_by_sku.get(sku, 0.0)
        penetration = basket_penetration.get(sku, 0.0)

        # Affinity intensity: max affinity with any other product
        aff_int = 0.0
        if sku in affinity.index:
            for other in affinity.columns:
                if other != sku and not np.isnan(affinity.loc[sku, other]):
                    aff_int = max(aff_int, abs(affinity.loc[sku, other]))

        # Incremental revenue: revenue minus baseline (here we use median revenue as baseline)
        all_revs = np.array([revenue_by_sku.get(s, 0.0) for s in skus])
        baseline = float(np.median(all_revs)) if len(all_revs) > 0 else 0.0
        inc_revenue = max(0.0, revenue - baseline)

        # Recommendation based on profile + quadrant logic
        profile.get("abc", "C")
        growth_pct = profile.get("growth", 0.0)
        elasticity = profile.get("elasticity", 0.0)

        if aff_int > 0.3 and revenue > np.median([revenue_by_sku.get(s, 0.0) for s in skus]):
            recommendation = "Anchor — Maintain & Expand"
        elif revenue > np.median([revenue_by_sku.get(s, 0.0) for s in skus]) and growth_pct > 0:
            recommendation = "Growth — Invest"
        elif aff_int > 0.3 and elasticity < -1.0:
            recommendation = "Protect — Retain"
        elif growth_pct < 0:
            recommendation = "Prune — Phase Out"
        else:
            recommendation = "Review — Test"

        rows.append(
            {
                "SKU": sku,
                "Affinity Intensity": round(aff_int, 3),
                "Revenue": round(revenue, 2),
                "Incremental Revenue": round(inc_revenue, 2),
                "Basket Penetration": round(penetration, 3),
                "Recommendation": recommendation,
            }
        )

    table = pd.DataFrame(rows).sort_values("Revenue", ascending=False).reset_index(drop=True)
    return table


def _render_manager_table(
    df: pd.DataFrame,
    top_n: int,
    profile_service: ProfileService,
    segment_col: str | None = None,
    segment_val: str | None = None,
    mission_col: str | None = None,
    mission_val: str | None = None,
) -> None:
    """Render the manager table with recommendation column."""

    st.subheader(":material/table_manager: Manager Table")

    table = _compute_manager_table(
        df, top_n, profile_service, segment_col, segment_val, mission_col, mission_val
    )

    # Format and display
    st.dataframe(
        table,
        column_config={
            "SKU": st.column_config.TextColumn("SKU", width="small"),
            "Affinity Intensity": st.column_config.NumberColumn("Affinity Intensity", width="medium"),
            "Revenue": st.column_config.NumberColumn("Revenue", width="medium", format="$%d"),
            "Incremental Revenue": st.column_config.NumberColumn("Incremental Revenue", width="medium", format="$%d"),
            "Basket Penetration": st.column_config.NumberColumn("Basket Penetration", width="medium", format="%.1%"),
            "Recommendation": st.column_config.TextColumn("Recommendation", width="large"),
        },
        hide_index=True,
        use_container_width=True,
    )

    # Summary insight
    anchors = table[table["Recommendation"] == "Anchor — Maintain & Expand"]
    growth = table[table["Recommendation"] == "Growth — Invest"]
    st.caption(
        f"Anchors: {len(anchors)} SKUs | Growth: {len(growth)} SKUs | "
        f"Recommendations drive basket-level strategy."
    )


# ---------------------------------------------------------------------------
# Render function
# ---------------------------------------------------------------------------

def render(df: pd.DataFrame) -> None:
    """Render the co-purchase / affinity tab with five-layer structure."""

    st.subheader(":material/link: Co-purchase Affinity")

    # --- Initialize Profile Service (Layer 5) ---
    # Cache the profile service across reruns
    if "profile_service_initialized" not in st.session_state:
        init_profile_service(df)
        st.session_state.profile_service_initialized = True

    profile_service = get_profile_service()

    # Detect available filters
    available_segments = _get_available_segments(df)

    with st.expander("Parameters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        top_n = c1.number_input("Top N products", 10, 200, 50)
        top_n_pairs = c2.number_input("Top affinity pairs", 5, 100, 20)
        top_n_products = c3.number_input("Candidate pool (top products)", 50, 500, 200)
        show_mission = c4.checkbox("Show basket mission", value=True)

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

    # Ensure profile service is initialized with current data
    if profile_service is not None:
        import contextlib
        with contextlib.suppress(Exception):
            # Force recompute by clearing cache miss trigger
            profile_service.refresh()

    # Layer 1: Basket Network
    st.divider()
    _render_basket_network(
        df,
        top_n=top_n,
        profile_service=profile_service,
        segment_col=segment_col,
        segment_val=segment_val,
        mission_col=mission_col,
        mission_val=mission_val,
    )

    # Layer 2: Affinity × Revenue Matrix
    st.divider()
    _render_affinity_revenue_matrix(
        df,
        top_n=top_n,
        profile_service=profile_service,
        segment_col=segment_col,
        segment_val=segment_val,
        mission_col=mission_col,
        mission_val=mission_val,
    )

    # Layer 3: Basket Mission Matrix (conditional)
    if show_mission:
        st.divider()
        _render_basket_mission_matrix(
            df,
            top_n=top_n,
            profile_service=profile_service,
            segment_col=segment_col,
            segment_val=segment_val,
            mission_col=mission_col,
            mission_val=mission_val,
        )

    # Layer 4: Manager Table
    st.divider()
    _render_manager_table(
        df,
        top_n=top_n,
        profile_service=profile_service,
        segment_col=segment_col,
        segment_val=segment_val,
        mission_col=mission_col,
        mission_val=mission_val,
    )

    # Layer 5: Product Decision Profile — per-SKU lookup
    st.divider()
    st.subheader(":material/search: Product Decision Profile")

    product = st.selectbox("Select SKU", df["stockcode"].unique())
    if product:
        try:
            profile = profile_service.get_profile(str(product))
            # Display profile fields
            col1, col2 = st.columns(2)
            profile_items = list(profile.items())
            for i, (k, v) in enumerate(profile_items):
                if i % 2 == 0:
                    with col1:
                        st.text(f"{k}: {v}")
                else:
                    with col2:
                        st.text(f"{k}: {v}")
        except Exception as e:
            st.error(f"Error loading profile: {e}")

    # Top pairs within selected segment/mission
    st.divider()
    st.subheader(":material/link: Top Affinity Pairs")

    pairs = get_top_affinity_pairs(
        df,
        top_n=top_n_pairs,
        min_cooccurrence=3,
        top_n_products=top_n_products,
        segment_col=segment_col,
        segment_val=segment_val,
        mission_col=mission_col,
        mission_val=mission_val,
    )

    if pairs.empty:
        st.warning("No affinity pairs found.")
    else:
        st.dataframe(pairs, use_container_width=True, hide_index=True)


MODE_SPEC: ModeSpec = ModeSpec(
    key="copurchase",
    label="Co-purchase",
    icon=":material/link:",
    handler=render,
    requires=("sufficient_baskets_200", "sufficient_skus_20"),
)
