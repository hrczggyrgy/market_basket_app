"""Product Switching tab - redesigned with 5-layer structure.

Follows the app-wide page pattern: switching flows -> revenue at risk ->
delist safety -> top insights -> actions. Money-framed so "switches" become
"€ at risk" and "€ recoverable".

New 5-layer structure (Waves 8-9):
1. Substitution network (node size=revenue, edge thickness=strength, color by role)
2. Revenue-at-risk matrix (4 quadrants: safe/protect/investigate/unique)
3. Sankey diagram (delisted→substitutes→lost demand revenue flows)
4. Customer switching matrix (source→destination with high-value filter)
5. Manager table (revenue/switching out/main substitute/recovery potential/risk/uniqueness/delist recommendation)

Integrates with Product Decision Profile for switching data.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.profile_service import get_profile_service, init_profile_service
from src.analytics.switching import (
    compute_substitution_strength,
    compute_switch_in_out_rates,
    compute_switching_matrix,
    get_customer_loyalty_metrics,
    get_top_switching_paths,
)
from src.analytics.transference import (
    compute_demand_transference_matrix,
    compute_substitutable_demand_percentage,
    delist_impact_analysis,
)
from src.ui.components_utils import (
    render_insight_cards,
    render_opportunity_table,
)
from src.ui.plots import PALETTE, empty_state, show
from src.ui.registry import ModeSpec


def _revenue_by_product(df: pd.DataFrame) -> pd.Series:
    """Total revenue per product (price * quantity)."""
    return (df["price"] * df["quantity"]).groupby(df["stockcode"]).sum()


def _classify_quadrant(revenue: float, sdp: float, rev_median: float, sdp_median: float) -> str:
    """Classify a product into one of 4 revenue-at-risk quadrants."""
    if revenue >= rev_median and sdp >= sdp_median:
        return "safe"
    elif revenue >= rev_median and sdp < sdp_median:
        return "protect"
    elif revenue < rev_median and sdp < sdp_median:
        return "unique"
    else:
        return "investigate"


def _render_substitution_network(
    substitution_df: pd.DataFrame,
    revenue_by_product: pd.Series,
    product_roles: dict[str, str] | None = None,
) -> None:
    """Render substitution network: node size=revenue, edge thickness=strength, color by role.

    Uses Plotly Sankey where:
    - Node thickness ∝ product revenue
    - Link value ∝ substitution strength (weak=1, moderate=2, strong=3)
    - Node color by role classification
    """
    st.subheader(":material/network: Substitution Network")

    if substitution_df.empty:
        show(empty_state("No substitution data"))
        return

    # Build product list from substitution links
    all_products = list(
        dict.fromkeys(
            substitution_df["from_product"].tolist() + substitution_df["to_product"].tolist()
        )
    )

    # Revenue-based thickness for nodes: map revenue to thickness scale
    rev_series = revenue_by_product.reindex(all_products).fillna(0.0)
    min_rev = float(rev_series.min())
    max_rev = float(rev_series.max())
    rev_range = max_rev - min_rev if max_rev > min_rev else 1.0

    # Thickness mapping: min 20, max 200
    node_thickness = {
        p: 20 + (rev_series[p] - min_rev) / rev_range * 180
        for p in all_products
    }

    # Strength mapping for links
    strength_map = {"weak": 1, "moderate": 2, "strong": 3}
    link_values = []
    link_colors = []

    # Color mapping by role - use palette extended by role
    if product_roles:
        role_colors = {
            "high_value": "#E15759",
            "growth": "#59A14F",
            "mature": "#F28E2B",
            "cash_cow": "#EDF6F9",
            "question_mark": "#FFDDD2",
            "default": "#76B7B2",
        }
        # Build per-link color based from_product role
        from_product_roles = {
            p: product_roles.get(p, "default") for p in all_products
        }
    else:
        from_product_roles = {p: "default" for p in all_products}
        role_colors = {"default": PALETTE[3]}

    # Build Sankey links from substitution data
    sources = substitution_df["from_product"].tolist()
    targets = substitution_df["to_product"].tolist()
    # Use substitution_strength as width; default to 1 if missing
    raw_values = substitution_df["substitution_strength"].map(strength_map).fillna(1).tolist()
    # Scale values for visual thickness
    max_val = max(raw_values) if raw_values else 1
    link_values = [v / max_val * 10 for v in raw_values]

    # Link color by from_product role
    for src in sources:
        role = from_product_roles.get(src, "default")
        link_colors.append(role_colors.get(role, PALETTE[3]))

    # Map product names to indices
    label_to_idx = {p: i for i, p in enumerate(all_products)}
    source_idx = [label_to_idx.get(p, 0) for p in sources]
    target_idx = [label_to_idx.get(p, 0) for p in targets]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node={
                    "label": all_products,
                    "color": [
                        role_colors.get(product_roles.get(p, "default"), PALETTE[3])
                        for p in all_products
                    ],
                    "pad": 15,
                    "thickness": [node_thickness.get(p, 50) for p in all_products],
                    "line": {"color": "black", "width": 0.5},
                },
                link={
                    "source": source_idx,
                    "target": target_idx,
                    "value": link_values,
                    "color": link_colors,
                    "hovertemplate": "%{label}<extra></extra>",
                },
            )
        ]
    )

    fig.update_layout(
        height=max(500, 30 * len(all_products)),
        font={"size": 10},
        title="Substitution Network: node thickness ∝ revenue, link value ∝ substitution strength, color by role",
    )
    show(fig)
    st.caption(
        "Node thickness ∝ product revenue; link value ∝ substitution strength (weak/moderate/strong); "
        "color by product role (from Product Decision Profile)."
    )


def _render_revenue_at_risk_matrix(
    df: pd.DataFrame,
    demand_transference_df: pd.DataFrame,
    top_n: int = 20,
) -> None:
    """Render revenue-at-risk matrix with 4 quadrants: safe/protect/investigate/unique.

    X-axis: Substitutability (SDP, 0-1 scale)
    Y-axis: Revenue (€)
    Four quadrants colored by classification:
    - Safe: high revenue, high substitutability
    - Protect: high revenue, low substitutability
    - Investigate: medium revenue, medium substitutability
    - Unique: low revenue, low substitutability
    """
    st.subheader(":material/water_drop: Revenue-at-Risk Matrix")

    if demand_transference_df is None or demand_transference_df.empty:
        show(empty_state("No switching data"))
        return

    rev = _revenue_by_product(df)

    # Compute SDP
    sdp_df = compute_substitutable_demand_percentage(demand_transference_df, df)
    sdp_lookup = dict(zip(sdp_df["stockcode"], sdp_df["sdp"], strict=False))

    # Build per-product data
    products = rev.index.tolist()
    revenue_vals = [rev.get(p, 0.0) for p in products]
    sdp_vals = [sdp_lookup.get(p, 0.5) for p in products]

    # Compute medians for quadrant split
    rev_median = pd.Series(revenue_vals).median()
    sdp_median = pd.Series(sdp_vals).median()

    # Classify each product into quadrant
    quadrant_labels = []
    for r, s in zip(revenue_vals, sdp_vals, strict=False):
        quadrant_labels.append(_classify_quadrant(r, s, rev_median, sdp_median))

    # Create scatter plot with quadrant coloring
    fig = go.Figure()

    quadrant_colors = {
        "safe": "#59A14F",    # green - safe to manage
        "protect": "#E15759", # red - protect from delisting
        "investigate": "#F28E2B", # orange - investigate
        "unique": "#EDF6F9",  # blue - unique demand
    }

    # Add scatter points colored by quadrant
    for quad in ["safe", "protect", "investigate", "unique"]:
        mask = [q == quad for q in quadrant_labels]
        if any(mask):
            fig.add_trace(
                go.Scatter(
                    x=[sdp_vals[i] for i, m in enumerate(mask) if m],
                    y=[revenue_vals[i] for i, m in enumerate(mask) if m],
                    mode="markers",
                    marker={
                        "color": quadrant_colors[quad],
                        "size": 12,
                        "opacity": 0.7,
                        "line": {"width": 1, "color": "white"},
                    },
                    name=quad,
                    showlegend=True,
                    hovertemplate=f"{quad}: Revenue=%{{y:,.0f}} SDP=%{{x:.1f}}<extra></extra>",
                )
            )

    # Add quadrant boundary lines
    fig.add_vline(x=sdp_median, line_dash="dash", line_color="#888888", opacity=0.5)
    fig.add_hline(y=rev_median, line_dash="dash", line_color="#888888", opacity=0.5)

    # Add quadrant annotations
    fig.add_annotation(
        x=sdp_median,
        y=rev_median,
        text=f"Median:\nRevenue: €{rev_median:,.0f}\nSDP: {sdp_median:.1f}",
        showarrow=False,
        font={"size": 10, "color": "#555555"},
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#888888",
        borderwidth=1,
    )

    fig.update_layout(
        xaxis={"title": "Substitutability (SDP)", "range": [0, 1], "dtick": 0.2},
        yaxis={"title": "Revenue (€)", "tickformat": ",.0f"},
        height=500,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )

    # Add quadrant labels in corners
    fig.add_annotation(
        x=0.1, y=0.9, xref="paper", yref="paper",
        text="Unique", showarrow=False,
        font={"size": 12, "color": quadrant_colors["unique"]},
    )
    fig.add_annotation(
        x=0.9, y=0.9, xref="paper", yref="paper",
        text="Safe", showarrow=False,
        font={"size": 12, "color": quadrant_colors["safe"]},
    )
    fig.add_annotation(
        x=0.1, y=0.1, xref="paper", yref="paper",
        text="Investigate", showarrow=False,
        font={"size": 12, "color": quadrant_colors["investigate"]},
    )
    fig.add_annotation(
        x=0.9, y=0.1, xref="paper", yref="paper",
        text="Protect", showarrow=False,
        font={"size": 12, "color": quadrant_colors["protect"]},
    )

    show(fig)
    st.caption(
        f"4 quadrants split at median Revenue (€{rev_median:,.0f}) and SDP. "
        f"Safe (high rev/high sdp): revenue manageable through substitution. "
        f"Protect (high rev/low sdp): unique demand, protect from delisting. "
        f"Investigate (med rev/med sdp): needs more analysis. "
        f"Unique (low rev/low sdp): unique demand driver, low revenue impact."
    )


def _render_sankey_revenue_flows(
    df: pd.DataFrame,
    demand_transference_df: pd.DataFrame,
    delist_impact_df: pd.DataFrame,
    top_n: int = 15,
) -> None:
    """Render Sankey diagram showing revenue flows: delisted→substitutes→lost demand.

    Shows the revenue flow when products are delisted, how demand transfers to
    substitutes, and the net lost demand after recovery.
    """
    st.subheader(":material/account_tree: Revenue Flow Sankey (Delisted → Substitutes → Lost Demand)")

    if demand_transference_df is None or demand_transference_df.empty:
        show(empty_state("No demand transference data"))
        return

    rev = _revenue_by_product(df)

    # Get delist impact products (bottom by net revenue impact, i.e., most at-risk for delist)
    if delist_impact_df is not None and not delist_impact_df.empty:
        delist_products = delist_impact_df.nlsmallest(top_n, "net_revenue_impact")["stockcode"].tolist()
    else:
        # Fallback: products with lowest SDP (most unique = delist candidates)
        sdp_df = compute_substitutable_demand_percentage(demand_transference_df, df)
        if sdp_df.empty:
            show(empty_state("No SDP data for sankey"))
            return
        delist_products = sdp_df.nsmallest(top_n, "sdp")["stockcode"].tolist()

    # Filter demand transference to delisted products
    dt_delisted = demand_transference_df[
        demand_transference_df["from_product"].isin(delist_products)
    ].copy()

    if dt_delisted.empty:
        show(empty_state("No switching flows from delisted products"))
        return

    # Build Sankey: delisted products → substitutes → lost demand
    # We'll create a 3-level Sankey:
    # Level 1: Delisted products (source)
    # Level 2: Substitute products (middle)
    # Level 3: Lost demand (sink) = delisted revenue - recovered revenue

    # Get recovered revenue per delisted product
    recovered = {}
    for prod in delist_products:
        row = delist_impact_df[delist_impact_df["stockcode"] == prod]
        if not row.empty:
            recovered[prod] = float(row.iloc[0]["estimated_revenue_recovered"])
        else:
            # Compute from DT matrix
            dt_prod = dt_delisted[dt_delisted["from_product"] == prod]
            recovered[prod] = float(dt_prod["observed_switching_transfer_revenue"].sum()) if not dt_prod.empty else 0.0

    # Level 1: Delisted products - use their revenue as size
    # Level 2: Substitute products - aggregate inflow
    # Level 3: Lost demand = revenue - recovered

    all_nodes = list(dict.fromkeys(delist_products))

    # Add substitute products that receive flow
    substitute_products = set()
    for _, row in dt_delisted.iterrows():
        substitute_products.add(row["to_product"])
    all_nodes.extend(list(substitute_products))

    # Level 3: lost demand products (could be same as delisted or new)
    # For simplicity, show lost demand as the net flow

    # Compute values
    # Source nodes: delisted product revenue
    [float(rev.get(p, 0.0)) for p in delist_products]
    # Target nodes: substitute revenue inflow
    target_values = {}
    for p in substitute_products:
        inflow = dt_delisted[dt_delisted["to_product"] == p][
            "observed_switching_transfer_revenue"
        ].sum()
        target_values[p] = float(inflow)
    # Lost demand nodes
    lost_values = {}
    for p in delist_products:
        lost_values[p] = float(rev.get(p, 0.0)) - recovered.get(p, 0.0)

    # Build node index mapping
    unique_nodes = list(dict.fromkeys(all_nodes))
    node_to_idx = {p: i for i, p in enumerate(unique_nodes)}

    # Sankey links:
    # Link 1: delisted → substitute (flow = observed switching transfer revenue)
    # Link 2: delisted → lost demand (flow = delisted revenue - recovered)
    # Link 3: substitute → lost demand (if needed, simplified)

    sources = []
    targets = []
    values = []

    # Link delisted → substitutes
    for _, row in dt_delisted.iterrows():
        frm = row["from_product"]
        to = row["to_product"]
        val = float(row["observed_switching_transfer_revenue"])
        if val > 0:
            sources.append(node_to_idx.get(frm, 0))
            targets.append(node_to_idx.get(to, 0))
            values.append(val)

    # Link delisted → lost demand (net lost)
    for prod in delist_products:
        net_lost = float(rev.get(prod, 0.0)) - recovered.get(prod, 0.0)
        if net_lost != 0:
            sources.append(node_to_idx.get(prod, 0))
            # Find or create a lost demand sink node
            lost_label = f"lost_{prod}"
            if lost_label not in node_to_idx:
                node_to_idx[lost_label] = len(unique_nodes)
                unique_nodes.append(lost_label)
            targets.append(node_to_idx[lost_label])
            values.append(net_lost if net_lost > 0 else -net_lost)  # positive = lost

    # Ensure we have enough colors
    n_palette = len(PALETTE)
    n_nodes = len(unique_nodes)
    node_colors = PALETTE[: (n_nodes - 1) % n_palette + 1]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node={
                    "label": unique_nodes,
                    "color": node_colors,
                    "pad": 15,
                    "thickness": 20,
                    "line": {"color": "black", "width": 0.5},
                },
                link={
                    "source": sources,
                    "target": targets,
                    "value": values,
                    "color": [PALETTE[1]] * len(values),
                },
            )
        ]
    )

    fig.update_layout(
        height=max(500, 35 * len(unique_nodes)),
        font={"size": 10},
    )
    show(fig)
    st.caption(
        f"Sankey showing revenue flows from {len(delist_products)} delisted products to substitutes "
        f"and lost demand. Thickness = revenue flow (€)."
    )


def _render_customer_switching_matrix(
    df: pd.DataFrame,
    switching_matrix: pd.DataFrame,
    top_n: int = 30,
) -> None:
    """Render customer switching matrix with high-value customer filter.

    Shows source→destination switching pairs, filtered to high-value customers
    (top decile by predicted CLV or by revenue contribution).
    """
    st.subheader(":material/swap_horiz: Customer Switching Matrix (High-Value Filter)")

    if switching_matrix.empty:
        show(empty_state("No switching matrix data"))
        return

    # Compute high-value customer switching
    # Use CLV-based filtering: identify top customers and filter matrix
    from src.analytics.switching import get_customer_loyalty_metrics

    get_customer_loyalty_metrics(df)

    # Identify high-value customers (top 20% by total spend)
    customer_revenue = (
        df.groupby("customer_id")
        .apply(lambda x: float((x["price"] * x["quantity"]).sum()), include_groups=False)
        .rename("customer_revenue")
    )
    customer_revenue.sum()
    rev_threshold = customer_revenue.quantile(0.8)  # top 20%
    high_value_customers = set(customer_revenue[customer_revenue >= rev_threshold].index)

    if not high_value_customers:
        show(empty_state("No high-value customers identified"))
        return

    # Filter switching matrix to high-value customers
    # Recompute switching matrix from high-value customer transactions
    hv_df = df[df["customer_id"].isin(high_value_customers)].copy()

    if hv_df.empty or len(hv_df["customer_id"].unique()) < 2:
        show(empty_state("No switching data from high-value customers"))
        return

    hv_matrix = compute_switching_matrix(hv_df, window_days=90, min_transactions=2)

    if hv_matrix.empty:
        show(empty_state("No switching patterns among high-value customers"))
        return

    # Top N by total switching volume
    total_switches = (
        hv_matrix.groupby("from_product")["count"].sum()
        + hv_matrix.groupby("to_product")["count"].sum()
    ).sort_values(ascending=False)
    top_products = total_switches.head(top_n).index.tolist()

    sub = hv_matrix[
        hv_matrix["from_product"].isin(top_products)
        & hv_matrix["to_product"].isin(top_products)
    ]

    if sub.empty:
        show(empty_state("No significant switching among filtered products"))
        return

    # Pivot for heatmap
    pivot = sub.pivot(
        index="from_product", columns="to_product", values="count"
    ).fillna(0)

    # Sort by total volume
    pivot = pivot.loc[top_products, top_products]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=[str(c) for c in pivot.columns],
            y=[str(r) for r in pivot.index],
            colorscale="RdYlGn",
            zmid=0,
            colorbar={"title": "Switch count (high-value customers)"},
            hovertemplate="from: %{y}<br>to: %{x}<br>count: %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis={"title": "Destination Product"},
        yaxis={"title": "Source Product"},
        height=max(400, 20 * len(top_products)),
    )
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(tickangle=0)

    show(fig)
    st.caption(
        f"Switching matrix filtered to {len(high_value_customers)} high-value customers "
        f"(top {rev_threshold:.0%} by revenue). Rows = source product, Columns = destination product, "
        f"Color = switch count. Only products with top {top_n} total switch volume shown."
    )


def _render_manager_table(
    df: pd.DataFrame,
    matrix: pd.DataFrame,
    demand_transference_df: pd.DataFrame,
    substitution_df: pd.DataFrame,
    delist_impact_df: pd.DataFrame,
    profile_service,
    top_n: int = 20,
) -> None:
    """Render manager table with all required columns.

    Columns:
    - revenue: product revenue
    - switching_out: switch-out rate
    - main_substitute: product most customers switch to
    - recovery_potential: recoverable revenue from substitution
    - risk: switching risk status from profile
    - uniqueness: 1 - SDP (high uniqueness = low substitutability)
    - delist_recommendation: delist recommendation based on all factors
    """
    st.subheader(":material/table_rows: Manager Decision Table")

    rev = _revenue_by_product(df)

    # Compute switching in/out rates using pre-computed matrix
    switch_rates = compute_switch_in_out_rates(matrix, df)

    # Compute substitution strength (already computed, reuse)
    if not demand_transference_df.empty:
        sdp_df = compute_substitutable_demand_percentage(demand_transference_df, df)
        sub_df = compute_substitution_strength(
            demand_transference_df, df, sdp_df
        )
    else:
        sub_df = pd.DataFrame()

    # Get profiles for all products (profile service already initialized)
    ps = profile_service

    # Build per-product manager data
    products = rev.index.tolist()
    manager_data = []

    for prod in products:
        # Revenue
        product_rev = float(rev.get(prod, 0.0))

        # Switching out rate
        prod_str = str(prod)
        sw_row = switch_rates[switch_rates["stockcode"] == prod_str] if not switch_rates.empty else pd.DataFrame()
        switch_out_rate = float(sw_row["switch_out_rate"].iloc[0]) if not sw_row.empty else 0.0
        n_switchers_out = int(sw_row["n_switchers_out"].iloc[0]) if not sw_row.empty else 0

        # Main substitute: the to_product with highest substitution strength
        main_substitute = "—"
        recovery_potential = 0.0
        if not sub_df.empty:
            sub_row = sub_df[sub_df["from_product"] == prod_str]
            if not sub_row.empty:
                # strongest strength
                strength_order = {"strong": 3, "moderate": 2, "weak": 1, "dominant": 4}
                best = sub_row.loc[
                    sub_row["substitution_strength"].map(strength_order).idxmax()
                ] if not sub_row["substitution_strength"].map(strength_order).empty else None
                if best is not None:
                    main_substitute = str(best["to_product"].iloc[0])
                    recovery_potential = float(best["recovery_proxy"].iloc[0])

        # Risk from profile
        risk = "unknown"
        uniqueness = 1.0
        if ps is not None:
            try:
                profile = ps.get_profile(prod_str)
                risk = profile.get("switching_risk", "unknown")
                sdp = profile.get("substitutability", 0.5)
                uniqueness = 1.0 - sdp
            except Exception:
                risk = "unknown"
                uniqueness = 1.0

        # Delist recommendation
        delist_rec = "monitor"
        if not delist_impact_df.empty:
            di_row = delist_impact_df[delist_impact_df["stockcode"] == prod_str]
            if not di_row.empty:
                net_impact = float(di_row.iloc[0]["net_revenue_impact"])
                if net_impact > 0:
                    delist_rec = "delist — positive impact"
                elif net_impact > -product_rev * 0.2:
                    delist_rec = "delist — limited impact"
                else:
                    delist_rec = "protect — negative impact"
            else:
                # No delist data but low SDP + high switching risk → delist candidate
                sdp_val = profile.get("substitutability", 0.5) if ps else 0.5
                if sdp_val < 0.3 and risk in ("estimated", "insufficient_transitions"):
                    delist_rec = "delist candidate — low substitutability"
                else:
                    delist_rec = "protect — strategic product"

        # If high switching risk + low substitutability → protect
        if risk == "estimated" and uniqueness > 0.7:
            delist_rec = "protect — unique demand driver"

        manager_data.append(
            {
                "product": prod,
                "revenue": product_rev,
                "switching_out": round(switch_out_rate, 3),
                "n_switchers_out": n_switchers_out,
                "main_substitute": main_substitute,
                "recovery_potential": round(recovery_potential, 2),
                "risk": risk,
                "uniqueness": round(uniqueness, 3),
                "delist_recommendation": delist_rec,
            }
        )

    # Sort by revenue descending
    manager_df = pd.DataFrame(manager_data).sort_values("revenue", ascending=False).reset_index(drop=True)

    # Render as Streamlit table with formatting
    st.write(f"**Manager Decision Table** — showing {len(manager_df)} products sorted by revenue")

    # Format the display
    display_df = manager_df.copy()
    display_df["revenue"] = display_df["revenue"].apply(lambda x: f"€{x:,.0f}")
    display_df["recovery_potential"] = display_df["recovery_potential"].apply(
        lambda x: f"€{x:,.0f}"
    )
    display_df["uniqueness"] = display_df["uniqueness"].apply(lambda x: f"{x:.1%}")
    display_df["switching_out"] = display_df["switching_out"].apply(lambda x: f"{x:.1%}")

    # Format delist recommendation with color coding
    def format_rec(rec):
        if "delist — positive" in rec:
            return f":green[{rec}]"
        elif "delist candidate" in rec:
            return f":red[{rec}]"
        elif "protect" in rec:
            return f":orange[{rec}]"
        else:
            return rec

    display_df["delist_recommendation"] = display_df["delist_recommendation"].apply(format_rec)

    # Render table
    st.dataframe(
        display_df[
            ["product", "revenue", "switching_out", "n_switchers_out", "main_substitute",
             "recovery_potential", "risk", "uniqueness", "delist_recommendation"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Columns: product SKU, revenue, switch-out rate & count, main substitute product, "
        "recoverable revenue from substitution, switching risk status (from Profile), "
        "uniqueness (1-SDP = uniqueness score), delist recommendation."
    )


def render(df: pd.DataFrame) -> None:
    """Render the Product Switching tab with 5-layer structure.

    Layers (Waves 8-9):
    1. Substitution network (node size=revenue, edge thickness=strength, color by role)
    2. Revenue-at-risk matrix (4 quadrants: safe/protect/investigate/unique)
    3. Sankey diagram (delisted→substitutes→lost demand revenue flows)
    4. Customer switching matrix (source→destination with high-value filter)
    5. Manager table (revenue/switching out/main substitute/recovery potential/risk/uniqueness/delist recommendation)

    Integrates with Product Decision Profile for switching data.
    """

    st.subheader(":material/swap_horiz: Product Switching")

    with st.expander("Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        window_days = c1.number_input("Window (days)", 30, 180, 90)
        min_txns = c2.number_input("Min transactions per customer", 2, 10, 3)
        top_n = c3.number_input("Top N paths / products", 5, 50, 20)

    # Initialize Profile Service
    try:
        init_profile_service(df)
        ps = get_profile_service()
    except Exception:
        ps = None
        st.info("Profile service unavailable — some features may have limited data.")

    # Core computations
    matrix = compute_switching_matrix(df, window_days=window_days, min_transactions=min_txns)
    demand_transference = compute_demand_transference_matrix(df, matrix)
    sdp_df = compute_substitutable_demand_percentage(demand_transference, df)
    delist_impact = delist_impact_analysis(
        df,
        demand_transference,
        sdp_df.sort_values("sdp", ascending=False)["stockcode"].head(10).tolist(),
    )

    # --- Layer 1: Substitution Network ---
    st.divider()
    st.header("1. Substitution Network")

    # Compute substitution strength
    substitution_df = compute_substitution_strength(
        demand_transference, df, sdp_df
    )

    # Get product roles from Profile Service
    product_roles = {}
    if ps is not None:
        # Extract role-related info from profiles
        for stockcode in df["stockcode"].unique():
            try:
                profile = ps.get_profile(str(stockcode))
                # Use a simple role mapping based on profile fields
                # Could be enhanced with explicit role field
                sdp = profile.get("substitutability", 0.5)
                switching_risk = profile.get("switching_risk", "unknown")
                # Derive role from SDP and other factors
                if sdp >= 0.8:
                    product_roles[str(stockcode)] = "high_substitutable"
                elif sdp < 0.3 and switching_risk == "estimated":
                    product_roles[str(stockcode)] = "unique_risk"
                elif switching_risk == "estimated":
                    product_roles[str(stockcode)] = "high_value"
                else:
                    product_roles[str(stockcode)] = "default"
            except Exception:
                product_roles[str(stockcode)] = "default"

    _render_substitution_network(
        substitution_df, _revenue_by_product(df), product_roles
    )

    # --- Layer 2: Revenue-at-Risk Matrix ---
    st.divider()
    st.header("2. Revenue-at-Risk Matrix")
    _render_revenue_at_risk_matrix(df, demand_transference, top_n)

    # --- Layer 3: Sankey Revenue Flows ---
    st.divider()
    st.header("3. Revenue Flow Sankey (Delisted → Substitutes → Lost Demand)")
    _render_sankey_revenue_flows(df, demand_transference, delist_impact, top_n)

    # --- Layer 4: Customer Switching Matrix ---
    st.divider()
    st.header("4. Customer Switching Matrix (High-Value Filter)")
    _render_customer_switching_matrix(df, matrix, top_n)

    # --- Layer 5: Manager Table ---
    st.divider()
    st.header("5. Manager Decision Table")
    _render_manager_table(
        df, demand_transference, substitution_df, delist_impact, ps, top_n
    )

    # --- Existing: Top Switching Paths ---
    st.divider()
    st.subheader(":material/task_alt: Top Switching Paths")
    top_paths = get_top_switching_paths(
        df, top_n=top_n, window_days=window_days, min_transactions=min_txns
    )
    if not top_paths.empty:
        st.dataframe(top_paths, use_container_width=True, hide_index=True)
    else:
        st.warning("No switching patterns found with current parameters.")

    st.divider()
    st.subheader(":material/radar: Customer Loyalty Metrics")
    loyalty = get_customer_loyalty_metrics(df)
    st.dataframe(loyalty, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader(":material/insights: Top Insights")
    from src.analytics.insights import generate_switching_insights

    insights = generate_switching_insights(demand_transference, sdp_df, delist_impact)
    render_insight_cards(insights)

    st.divider()
    st.subheader(":material/task_alt: Ranked Decisions")
    from src.analytics.opportunities import generate_switching_opportunities

    opportunities = generate_switching_opportunities(
        sdp_df, delist_impact, _revenue_by_product(df), top_n=10
    )
    render_opportunity_table(opportunities)


# ── Mode spec ─────────────────────────────────────────────────────
MODE_SPEC: ModeSpec = ModeSpec(
    key="switching",
    label="Switching",
    icon=":material/swap_horiz:",
    handler=render,
    requires=("sufficient_customers_100", "sufficient_baskets_500"),
)
