"""Product Switching tab.

Follows the app-wide page pattern: switching flows -> revenue at risk ->
delist safety -> top insights -> actions. Money-framed so "switches" become
"€ at risk" and "€ recoverable".
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.insights import generate_switching_insights
from src.analytics.opportunities import generate_switching_opportunities
from src.analytics.switching import (
    build_event_slices,
    compute_category_switching_by_phase,
    compute_category_switching_matrix,
    compute_switching_matrix,
    compute_transition_matrix,
    get_customer_loyalty_metrics,
    get_top_switching_paths,
)
from src.analytics.transference import (
    compute_demand_transference_matrix,
    compute_recovery_hhi,
    compute_substitutable_demand_percentage,
    delist_impact_analysis,
)
from src.ui.components import render_insight_cards, render_metric_row, render_opportunity_table
from src.ui.features import get_detected_promotions, get_product_lookup
from src.ui.plots import PALETTE, empty_state, new_fig, show
from src.ui.registry import ModeSpec


def _revenue_by_product(df: pd.DataFrame) -> pd.Series:
    return (df["price"] * df["quantity"]).groupby(df["stockcode"]).sum()


def _render_revenue_at_risk(df: pd.DataFrame, demand_transference_df: pd.DataFrame, top_n: int = 12) -> None:
    st.subheader(":material/water_drop: Revenue at Risk (Net Switching)")
    if demand_transference_df is None or demand_transference_df.empty:
        show(empty_state("No switching data"))
        return

    out = (
        demand_transference_df.groupby("from_product")["observed_switching_recovery_proxy"]
        .sum()
        .rename("outflow")
    )
    inflow = (
        demand_transference_df.groupby("to_product")["observed_switching_recovery_proxy"]
        .sum()
        .rename("inflow")
    )
    net = pd.concat([out, inflow], axis=1).fillna(0.0)
    net["net"] = net["inflow"] - net["outflow"]
    net["revenue"] = net.index.map(_revenue_by_product(df)).fillna(0.0)
    top = net.reindex(net["net"].abs().sort_values(ascending=False).index).head(top_n)

    fig = new_fig()
    fig.add_trace(
        go.Bar(
            x=top.index.astype(str),
            y=top["net"],
            marker={"color": ["#59A14F" if v >= 0 else "#E15759" for v in top["net"]]},
            hovertemplate="%{x}<br>Net switching revenue: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis={"title": "Net switching revenue (in - out)"},
        xaxis={"title": "", "tickangle": -45},
    )
    show(fig)

    total_out = float(out.sum())
    total_in = float(inflow.sum())
    total_rev = float(_revenue_by_product(df).sum())
    render_metric_row(
        [
            {"label": "Switching revenue out", "value": f"€{total_out:,.0f}", "help": "Observed revenue-weighted switches away."},
            {"label": "Switching revenue in", "value": f"€{total_in:,.0f}", "help": "Observed revenue-weighted switches toward."},
            {"label": "Share of total revenue", "value": f"{total_out / total_rev:.1%}" if total_rev else "—", "help": "Revenue that moved between products at least once."},
        ]
    )
    st.caption(
        "Net switching revenue per product (recovery-weighted). Green = net inflow "
        "(customers substituting toward it), red = net outflow. Observed correlation, "
        "not a causal delist estimate."
    )


def _render_delist_safety(df: pd.DataFrame, sdp_df: pd.DataFrame, top_n: int = 12) -> None:
    st.subheader(":material/verified: Delist Safety (Substitutable Demand)")
    if sdp_df is None or sdp_df.empty:
        show(empty_state("No SDP data"))
        return

    rev = _revenue_by_product(df)
    work = sdp_df.copy()
    work["revenue"] = work["stockcode"].map(rev).fillna(0.0)
    work = work.sort_values("revenue", ascending=False).head(top_n)

    fig = new_fig()
    fig.add_trace(
        go.Bar(
            x=work["stockcode"].astype(str),
            y=work["sdp"] * 100,
            marker={"color": ["#59A14F" if v >= 80 else "#F28E2B" if v >= 20 else "#E15759" for v in work["sdp"] * 100]},
            hovertemplate="%{x}<br>SDP: %{y:.0f}%<br>Revenue: €%{customdata:,.0f}<extra></extra>",
            customdata=work["revenue"],
        )
    )
    fig.add_hline(y=80, line_dash="dash", line_color="#59A14F", annotation_text="Highly substitutable (safe to cut)")
    fig.add_hline(y=20, line_dash="dash", line_color="#E15759", annotation_text="Unique demand (do not cut)")
    fig.update_layout(
        yaxis={"title": "Substitutable demand % (SDP)", "range": [0, 105]},
        xaxis={"title": "", "tickangle": -45},
    )
    show(fig)
    st.caption(
        "SDP = share of a product's demand that is recoverable inside the assortment "
        "when it goes missing. Green = delisting is low-risk; red = revenue leaks away."
    )


def _render_sankey(df: pd.DataFrame, matrix: pd.DataFrame, top_n: int) -> None:
    st.subheader(":material/account_tree: Switching Flow (Sankey)")
    if matrix.empty:
        show(empty_state("No switching transitions"))
        return

    top = matrix.nlargest(top_n, "count")
    if top.empty:
        show(empty_state("No significant switching paths"))
        return

    # Build Sankey data
    sources = top["from_product"].tolist()
    targets = top["to_product"].tolist()
    values = top["count"].tolist()

    # Map products to indices
    all_products = list(dict.fromkeys(sources + targets))
    label_to_idx = {p: i for i, p in enumerate(all_products)}
    source_idx = [label_to_idx[p] for p in sources]
    target_idx = [label_to_idx[p] for p in targets]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node={
                    "label": all_products,
                    "color": PALETTE[0],
                    "pad": 15,
                    "thickness": 20,
                },
                link={
                    "source": source_idx,
                    "target": target_idx,
                    "value": values,
                    "color": [PALETTE[1]] * len(values),
                },
            )
        ]
    )
    fig.update_layout(height=max(400, 30 * len(all_products)), font={"size": 10})
    show(fig)
    st.caption(f"Top {top_n} product-to-product switches. Thickness = transition count.")


def _render_category_sankey(
    df: pd.DataFrame,
    window_days: int,
    min_txns: int,
    top_categories: int = 15,
) -> None:
    st.subheader(":material/account_tree: Category Switching Flow (Sankey)")
    lookup = get_product_lookup(df)
    cat_matrix = compute_category_switching_matrix(
        df,
        window_days=window_days,
        min_transactions=min_txns,
        product_lookup=lookup,
    )
    if cat_matrix.empty:
        show(empty_state("No category switching transitions"))
        return

    top = cat_matrix.nlargest(top_categories, "count")
    if top.empty:
        show(empty_state("No significant category switching paths"))
        return

    sources = top["from_category"].tolist()
    targets = top["to_category"].tolist()
    values = top["count"].tolist()

    all_cats = list(dict.fromkeys(sources + targets))
    label_to_idx = {c: i for i, c in enumerate(all_cats)}
    source_idx = [label_to_idx[c] for c in sources]
    target_idx = [label_to_idx[c] for c in targets]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(all_cats))]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node={
                    "label": all_cats,
                    "color": colors,
                    "pad": 15,
                    "thickness": 20,
                },
                link={
                    "source": source_idx,
                    "target": target_idx,
                    "value": values,
                    "color": [PALETTE[1]] * len(values),
                },
            )
        ]
    )
    fig.update_layout(height=max(400, 30 * len(all_cats)), font={"size": 10})
    show(fig)
    st.caption(
        f"Top {top_categories} category-to-category switches. "
        "Rolled up from product-level switching; thickness = transition count."
    )


def _render_switcher_loyalist(df: pd.DataFrame, matrix: pd.DataFrame) -> None:
    st.subheader(":material/group: Switcher vs Loyalist per Product")
    if matrix.empty:
        show(empty_state("No switching data"))
        return

    # Per product: out-switches vs in-switches vs loyal (no switch)
    out_counts = matrix.groupby("from_product")["count"].sum().rename("switches_out")
    in_counts = matrix.groupby("to_product")["count"].sum().rename("switches_in")
    all_products = set(out_counts.index) | set(in_counts.index)

    # Total purchases per product as proxy for "loyal" transactions
    total_purchases = df.groupby("stockcode")["transaction_id"].nunique().rename("total_purchases")

    summary = pd.DataFrame(
        {
            "product": list(all_products),
            "switches_out": [out_counts.get(p, 0) for p in all_products],
            "switches_in": [in_counts.get(p, 0) for p in all_products],
            "total_purchases": [total_purchases.get(p, 0) for p in all_products],
        }
    )
    summary["loyal"] = (summary["total_purchases"] - summary["switches_out"]).clip(lower=0)

    fig = new_fig()
    fig.add_trace(
        go.Bar(
            x=summary["product"],
            y=summary["loyal"],
            name="Loyal (no switch out)",
            marker={"color": PALETTE[2]},
        )
    )
    fig.add_trace(
        go.Bar(
            x=summary["product"],
            y=summary["switches_out"],
            name="Switched away from",
            marker={"color": PALETTE[1]},
        )
    )
    fig.add_trace(
        go.Bar(
            x=summary["product"],
            y=summary["switches_in"],
            name="Switched to",
            marker={"color": PALETTE[3]},
        )
    )
    fig.update_layout(barmode="stack", yaxis={"title": "Transaction count"}, xaxis={"tickangle": -45})
    show(fig)
    st.caption("Stacked: loyal transactions, switches away, switches to. Products with high 'switched to' are acquisition drivers.")


def _render_monthly_net_switching(df: pd.DataFrame, window_days: int) -> None:
    st.subheader(":material/trending_up: Monthly Net Switching Direction")
    from src.analytics.switching import _customer_sequences

    seq = _customer_sequences(df, window_days, 2)
    if seq.empty:
        show(empty_state("No sequences for monthly trend"))
        return

    # Identify switches per sequence
    seq = seq.copy()
    seq["switched"] = seq.apply(
        lambda row: set(row["prev_products"].split(",")) != set(row["products"].split(",")),
        axis=1,
    )
    switched = seq[seq["switched"]].copy()
    if switched.empty:
        show(empty_state("No switches in period"))
        return

    # Expand switches to from/to pairs
    rows = []
    for _, row in switched.iterrows():
        prev_set = set(row["prev_products"].split(","))
        cur_set = set(row["products"].split(","))
        for frm in prev_set - cur_set:
            for to in cur_set - prev_set:
                rows.append({"date": row["date"], "from_product": frm, "to_product": to})
    if not rows:
        show(empty_state("No switch pairs"))
        return

    switch_df = pd.DataFrame(rows)
    switch_df["period"] = switch_df["date"].dt.to_period("M").astype(str)

    # Net flow per product per month: in - out
    # Aggregate out and in separately then combine
    out_monthly = switch_df.groupby(["period", "from_product"]).size().rename("out").reset_index()
    out_monthly = out_monthly.rename(columns={"from_product": "product"})
    
    in_monthly = switch_df.groupby(["period", "to_product"]).size().rename("in").reset_index()
    in_monthly = in_monthly.rename(columns={"to_product": "product"})

    # Outer join to get all product-period combinations
    monthly = out_monthly.merge(in_monthly, on=["period", "product"], how="outer").fillna(0)
    monthly["net"] = monthly["in"] - monthly["out"]

    # Top products by total absolute net flow
    product_vol = monthly.groupby("product")["net"].apply(lambda s: s.abs().sum()).sort_values(ascending=False)
    top_products = product_vol.head(10).index.tolist()
    monthly_top = monthly[monthly["product"].isin(top_products)]

    if monthly_top.empty:
        show(empty_state("No monthly trend data"))
        return

    fig = new_fig()
    for product in top_products:
        sub = monthly_top[monthly_top["product"] == product].sort_values("period")
        fig.add_trace(
            go.Bar(
                x=sub["period"],
                y=sub["net"],
                name=product,
                marker={"color": PALETTE[top_products.index(product) % len(PALETTE)]},
            )
        )
    fig.update_layout(barmode="group", yaxis={"title": "Net switches (in - out)"}, xaxis={"title": "Month"})
    show(fig)
    st.caption("Positive = net acquisition; negative = net defection. Shows seasonal switching dynamics.")


def _render_transition_heatmap(df: pd.DataFrame, matrix: pd.DataFrame, top_n: int, window_days: int, min_txns: int) -> None:
    st.subheader(":material/table_chart: Transition Probability Matrix")
    pivot = compute_transition_matrix(df, window_days=window_days, min_transactions=min_txns)
    if pivot.empty:
        show(empty_state("No transition matrix"))
        return

    # Filter to top products by total transitions
    total_out = matrix.groupby("from_product")["count"].sum()
    total_in = matrix.groupby("to_product")["count"].sum()
    product_score = (total_out + total_in).sort_values(ascending=False)
    top_products = product_score.head(top_n).index.tolist()

    sub = pivot.loc[pivot.index.intersection(top_products), pivot.columns.intersection(top_products)]
    if sub.empty:
        show(empty_state("No transitions among top products"))
        return

    fig = go.Figure(
        data=go.Heatmap(
            z=sub.to_numpy(),
            x=[str(c) for c in sub.columns],
            y=[str(i) for i in sub.index],
            colorscale="RdYlGn",
            zmid=0,
            colorbar={"title": "Probability"},
        )
    )
    fig.update_layout(xaxis={"tickangle": -45}, yaxis={"tickangle": 0}, height=max(400, 20 * len(top_products)))
    show(fig)
    st.caption("Row-normalized: each row sums to 1. Green = high probability of switching TO column from row.")


def _render_phase_switch_comparison(
    df: pd.DataFrame,
    window_days: int,
    min_txns: int,
    top_n: int,
) -> None:
    """Compare category switching across pre-event / event / post-event windows."""
    st.subheader(":material/swap_horiz: Time-Sliced Category Switching (Pre / Event / Post)")
    lookup = get_product_lookup(df)

    # Detect promo periods as the event source (existing promo analytics)
    events = get_detected_promotions(df)
    if events.empty:
        st.info("No promotional periods detected — set an event window manually below.")
        return

    st.caption(f"Events detected: {len(events)} promotional periods (earliest {events['start_date'].min().date()} to {events['end_date'].max().date()}).")

    c1, c2, c3 = st.columns(3)
    pre_days = int(c1.number_input("Pre window (days)", 7, 180, 30))
    post_days = int(c2.number_input("Post window (days)", 7, 180, 30))

    phases = compute_category_switching_by_phase(
        df,
        events,
        pre_days=pre_days,
        post_days=post_days,
        window_days=window_days,
        min_transactions=min_txns,
        product_lookup=lookup,
    )

    if not phases:
        st.info("No switching data within the selected pre/event/post windows.")
        return

    for phase in ("pre", "event", "post"):
        if phase not in phases:
            continue
        phase_matrix = phases[phase]
        if phase_matrix.empty:
            continue

        # Show top transitions for phase
        st.markdown(f"**{phase.capitalize()} event**")
        top = phase_matrix.nlargest(top_n, "count")
        if top.empty:
            st.caption("No switches in this phase.")
            continue

        # Horizontal bar: top category transitions
        labels = [f"{r.from_category} → {r.to_category}" for _, r in top.iterrows()]
        fig = go.Figure(
            data=go.Bar(
                x=top["count"],
                y=labels,
                orientation="h",
                marker={"color": PALETTE[(list(phases.keys()).index(phase)) % len(PALETTE)]},
                hovertemplate="%{y}: %{x} switches<extra></extra>",
            )
        )
        fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": labels[::-1]}, height=280)
        show(fig)


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/swap_horiz: Product Switching")

    with st.expander("Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        window_days = c1.number_input("Window (days)", 30, 180, 90)
        min_txns = c2.number_input("Min transactions per customer", 2, 10, 3)
        top_n = c3.number_input("Top N paths / products", 5, 50, 20)

    matrix = compute_switching_matrix(df, window_days=window_days, min_transactions=min_txns)

    if matrix.empty:
        st.warning("No switching patterns found with current parameters.")
        return

    demand_transference = compute_demand_transference_matrix(df, matrix)
    sdp = compute_substitutable_demand_percentage(demand_transference, df)
    delist_impact = delist_impact_analysis(
        df, demand_transference, sdp.sort_values("sdp", ascending=False)["stockcode"].head(10).tolist()
    )
    recovery_hhi = compute_recovery_hhi(demand_transference)

    st.divider()
    _render_revenue_at_risk(df, demand_transference, top_n)

    st.divider()
    _render_delist_safety(df, sdp, top_n)

    st.divider()
    _render_sankey(df, matrix, top_n)

    st.divider()
    _render_category_sankey(df, window_days, min_txns)

    st.divider()
    _render_phase_switch_comparison(df, window_days, min_txns, top_n)

    st.divider()
    _render_switcher_loyalist(df, matrix)

    st.divider()
    _render_monthly_net_switching(df, window_days)

    st.divider()
    _render_transition_heatmap(df, matrix, top_n, window_days, min_txns)

    st.divider()
    st.subheader(":material/table_rows: Top Switching Paths")
    top_paths = get_top_switching_paths(df, top_n=top_n, window_days=window_days, min_transactions=min_txns)
    st.dataframe(top_paths, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader(":material/person: Customer Loyalty Metrics")
    loyalty = get_customer_loyalty_metrics(df)
    st.dataframe(loyalty, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader(":material/radar: Top Insights")
    insights = generate_switching_insights(demand_transference, sdp, delist_impact)
    render_insight_cards(insights)

    st.divider()
    st.subheader(":material/task_alt: Ranked Decisions")
    opportunities = generate_switching_opportunities(
        sdp, delist_impact, _revenue_by_product(df), top_n=10
    )
    render_opportunity_table(opportunities)


MODE_SPEC: ModeSpec = ModeSpec(
    key="switching",
    label="Switching",
    icon=":material/swap_horiz:",
    handler=render,
)