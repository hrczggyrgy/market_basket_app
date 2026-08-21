"""Overview / Dashboard tab — 5-layer decision intelligence structure.

Follows the app-wide page pattern with enhanced decision-intelligence layout:
  Layer 1: KPI Row — 6 decision KPIs
  Layer 2: Primary Strategic Matrix — Category Growth × Strategic Importance (4 quadrants)
  Layer 3: Revenue Decomposition Waterfall — 6 driver contributions
  Layer 4: Growth Driver Matrix — Emerging/Growth/Draggers/Critical risks quadrants
  Layer 5: Decision Table + Executive Agenda — sortable table + dynamic top-5 priorities
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.data import build_dataset_capabilities, get_data_summary
from src.analytics.data_quality import generate_quality_summary
from src.ui.components_utils import (
    render_metric_row,
)
from src.ui.plots import PALETTE, empty_state, new_fig, show
from src.ui.registry import ModeSpec

_DRIVER_LABELS = {
    "customers": "Customer count",
    "frequency": "Shopping frequency",
    "basket_size": "Basket size",
    "price_per_unit": "Price per unit",
    "product_mix": "Product mix",
    "lost_customers": "Lost customers",
}

# ── Layer 1: KPI Row ──────────────────────────────────────────────────────

def _render_kpi_row(df: pd.DataFrame, capabilities: dict) -> None:
    """Render the 6-card KPI row at the top of the Overview tab."""
    work = df.copy()
    work["revenue"] = work["price"] * work["quantity"]
    total_revenue = work["revenue"].sum()
    n_customers = work["customer_id"].nunique()
    len(work)
    work["stockcode"].nunique()

    # Category Revenue
    category_revenue = 0.0
    if "category" in work.columns:
        category_revenue = float(
            (work["price"] * work["quantity"]).groupby(work["category"]).sum().sum()
        )

    # Revenue Growth (WoW)
    weekly = work.copy()
    weekly["date"] = pd.to_datetime(weekly["date"])
    weekly["week"] = weekly["date"].dt.to_period("W").dt.start_time
    weekly_agg = (
        weekly.groupby("week")
        .agg(revenue=("revenue", "sum"), customers=("customer_id", "nunique"))
        .sort_index()
    )
    revenue_growth = 0.0
    if len(weekly_agg) >= 2:
        rev_curr = weekly_agg.iloc[-1]["revenue"]
        rev_prev = weekly_agg.iloc[-2]["revenue"]
        if rev_prev > 0:
            revenue_growth = float(rev_curr / rev_prev - 1)

    # Customer Reach
    customer_reach = float(n_customers)

    # Revenue at Risk — estimate from low-confidence drivers
    revenue_at_risk = 0.0
    if (capabilities.get("sufficient_customers_500") and capabilities.get("sufficient_skus_50")
            and "category" in work.columns):
        # Simple heuristic: revenue from products with low evidence confidence
        # Placeholder: use concentration risk as proxy
        cat_rev = (
            (work["price"] * work["quantity"])
            .groupby(work["category"])
            .sum()
            .sort_values(ascending=False)
        )
        top5_share = cat_rev.head(5).sum() / cat_rev.sum() if len(cat_rev) > 0 else 0
        revenue_at_risk = total_revenue * min(top5_share * 0.3, 0.2)

    # Opportunity Value — residual growth potential
    opportunity_value = 0.0
    if len(weekly_agg) >= 2:
        # Simple: difference between current growth and driver-identified growth
        opportunity_value = total_revenue * abs(revenue_growth) * 0.4

    # Evidence Coverage — fraction of drivers with >= medium confidence
    evidence_coverage = 0.6  # default heuristic; will be refined by actual data

    # Render 6 KPI cards
    render_metric_row(
        [
            {
                "label": "Category Revenue",
                "value": f"€{category_revenue:,.0f}",
                "help": "Revenue attributed to primary category",
            },
            {
                "label": "Revenue Growth",
                "value": f"{revenue_growth:+.1%}",
                "help": "Week-over-week revenue change",
            },
            {
                "label": "Customer Reach",
                "value": f"{int(customer_reach):,}",
                "help": "Unique customers in period",
            },
            {
                "label": "Revenue at Risk",
                "value": f"€{revenue_at_risk:,.0f}",
                "help": "Revenue exposed to low-confidence drivers",
            },
            {
                "label": "Opportunity Value",
                "value": f"€{opportunity_value:,.0f}",
                "help": "Residual growth potential",
            },
            {
                "label": "Evidence Coverage",
                "value": f"{evidence_coverage:.0%}",
                "help": "Drivers with medium+ confidence evidence",
            },
        ]
    )


# ── Layer 2: Primary Strategic Matrix ─────────────────────────────────────

def _render_primary_matrix(df: pd.DataFrame, capabilities: dict) -> None:
    """Render Category Growth × Strategic Importance matrix (4 quadrants)."""
    st.subheader(":material/strategic: Category Growth × Strategic Importance")

    work = df.copy()
    work["revenue"] = work["price"] * work["quantity"]
    n_customers = work["customer_id"].nunique()

    # Build per-category metrics
    if "category" in work.columns:
        cat_metrics = (
            work.groupby("category")
            .agg(
                revenue=("revenue", "sum"),
                customers=("customer_id", "nunique"),
                transactions=("transaction_id", "nunique"),
                n_products=("stockcode", "nunique"),
            )
            .reset_index()
        )
        # Derive growth proxy: revenue per customer (avg revenue per customer)
        cat_metrics["revenue_per_customer"] = cat_metrics["revenue"] / cat_metrics["customers"].replace(
            0, np.nan
        )
        cat_metrics["customer_reach_pct"] = cat_metrics["customers"] / cat_metrics["customers"].sum() * 100

        # Strategic importance: blend revenue per customer and customer reach
        # Normalize for quadrant assignment
        rev_max = cat_metrics["revenue_per_customer"].max()
        rev_min = cat_metrics["revenue_per_customer"].min()
        reach_max = cat_metrics["customer_reach_pct"].max()
        reach_min = cat_metrics["customer_reach_pct"].min()

        if rev_max > rev_min:
            cat_metrics["rev_norm"] = (
                (cat_metrics["revenue_per_customer"] - rev_min) / (rev_max - rev_min)
            )
        else:
            cat_metrics["rev_norm"] = 0.5

        if reach_max > reach_min:
            cat_metrics["reach_norm"] = (
                (cat_metrics["customer_reach_pct"] - reach_min) / (reach_max - reach_min)
            )
        else:
            cat_metrics["reach_norm"] = 0.5

        # Quadrant: Grow (high rev, high reach), Defend (high rev, low reach),
        # Build (low rev, high reach), Review (low rev, low reach)
        median_rev = cat_metrics["rev_norm"].median()
        median_reach = cat_metrics["reach_norm"].median()

        cat_metrics["quadrant"] = pd.cut(
            cat_metrics["rev_norm"],
            bins=[0, median_rev, 1],
            labels=["Build", "Grow"],
            include_lowest=True,
        ).astype(str) + "|" + pd.cut(
            cat_metrics["reach_norm"],
            bins=[0, median_reach, 1],
            labels=["Review", "Defend"],
            include_lowest=True,
        ).astype(str)

        cat_metrics["quadrant"] = cat_metrics["quadrant"].apply(
            lambda s: s.split("|")[0] if s.split("|")[0] == "Grow" or s.split("|")[0] == "Build"
            else s.split("|")[1]
            if s.split("|")[1] == "Defend" or s.split("|")[1] == "Review"
            else "Grow"
        )

        # Simpler approach: assign based on median splits
        cat_metrics["quadrant"] = ""
        for idx, row in cat_metrics.iterrows():
            high_rev = row["rev_norm"] >= median_rev
            high_reach = row["reach_norm"] >= median_reach
            if high_rev and high_reach:
                cat_metrics.at[idx, "quadrant"] = "Grow"
            elif high_rev and not high_reach:
                cat_metrics.at[idx, "quadrant"] = "Defend"
            elif not high_rev and high_reach:
                cat_metrics.at[idx, "quadrant"] = "Build"
            else:
                cat_metrics.at[idx, "quadrant"] = "Review"

    else:
        # Fallback without category column
        st.info("No category column — showing overall growth/importance matrix")
        cat_metrics = pd.DataFrame(
            {
                "quadrant": ["Grow"],
                "revenue_per_customer": [float(work["revenue"].sum()) / max(n_customers, 1)],
                "customer_reach_pct": [float(n_customers) / max(n_customers, 1) * 100],
            }
        )

    # Render bubble matrix with 4 quadrants
    fig = new_fig(height=500)

    # Calculate actual data ranges for quadrant backgrounds
    x_min = cat_metrics["revenue_per_customer"].min() if not cat_metrics.empty else 0
    x_max = cat_metrics["revenue_per_customer"].max() if not cat_metrics.empty else 1
    y_min = cat_metrics["customer_reach_pct"].min() if not cat_metrics.empty else 0
    y_max = cat_metrics["customer_reach_pct"].max() if not cat_metrics.empty else 1

    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    if not cat_metrics.empty:
        # Add quadrant background rectangles using shapes
        fig.add_shape(
            type="rect", x0=x_mid, x1=x_max, y0=y_mid, y1=y_max,
            fillcolor="rgba(78, 121, 167, 0.1)", line_width=0, layer="below"
        )
        fig.add_shape(
            type="rect", x0=x_mid, x1=x_max, y0=y_min, y1=y_mid,
            fillcolor="rgba(225, 87, 89, 0.1)", line_width=0, layer="below"
        )
        fig.add_shape(
            type="rect", x0=x_min, x1=x_mid, y0=y_min, y1=y_mid,
            fillcolor="rgba(89, 161, 79, 0.1)", line_width=0, layer="below"
        )
        fig.add_shape(
            type="rect", x0=x_min, x1=x_mid, y0=y_mid, y1=y_max,
            fillcolor="rgba(242, 142, 43, 0.1)", line_width=0, layer="below"
        )

        # Size bubbles by customer reach, color by quadrant
        size_max = cat_metrics["customers"].max()
        color_map = {
            "Grow": "#4E79A7",
            "Defend": "#E15759",
            "Build": "#59A14F",
            "Review": "#F28E2B",
        }

        for quad in ["Grow", "Defend", "Build", "Review"]:
            quad_df = cat_metrics[cat_metrics["quadrant"] == quad]
            if not quad_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=quad_df["revenue_per_customer"],
                        y=quad_df["customer_reach_pct"],
                        mode="markers+text",
                        marker={
                            "size": (quad_df["customers"] / size_max * 60) + 10,
                            "color": color_map.get(quad, PALETTE[0]),
                            "opacity": 0.8,
                            "line": {"width": 1, "color": "white"},
                        },
                        text=quad_df["category"] if "category" in cat_metrics.columns else quad_df.index,
                        textposition="top center",
                        textfont={"size": 8},
                        name=quad,
                        hovertemplate="<b>%{text}</b><br>" +
                                      "Revenue per customer: €%{x:,.0f}<br>" +
                                      "Customer reach: %{y:.1f}%<br>" +
                                      "<extra></extra>",
                    )
                )

    # Add quadrant labels using actual data coordinates
    fig.add_annotation(
        x=(x_mid + x_max) / 2, y=(y_mid + y_max) / 2,
        text="GROW", showarrow=False,
        font={"size": 16, "color": "#4E79A7", "weight": "bold"}
    )
    fig.add_annotation(
        x=(x_mid + x_max) / 2, y=(y_min + y_mid) / 2,
        text="DEFEND", showarrow=False,
        font={"size": 16, "color": "#E15759", "weight": "bold"}
    )
    fig.add_annotation(
        x=(x_min + x_mid) / 2, y=(y_min + y_mid) / 2,
        text="BUILD", showarrow=False,
        font={"size": 16, "color": "#59A14F", "weight": "bold"}
    )
    fig.add_annotation(
        x=(x_min + x_mid) / 2, y=(y_mid + y_max) / 2,
        text="REVIEW", showarrow=False,
        font={"size": 16, "color": "#F28E2B", "weight": "bold"}
    )

    fig.update_layout(
        xaxis={"title": "Revenue per Customer (€)", "tickformat": ".0f"},
        yaxis={"title": "Customer Reach (%)", "tickformat": ".0f"},
        margin={"l": 80, "r": 30, "t": 80, "b": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )

    show(fig)

    # Summary insight
    len(cat_metrics)
    grow_count = int((cat_metrics["quadrant"] == "Grow").sum()) if "quadrant" in cat_metrics.columns else 0
    defend_count = int((cat_metrics["quadrant"] == "Defend").sum()) if "quadrant" in cat_metrics.columns else 0
    build_count = int((cat_metrics["quadrant"] == "Build").sum()) if "quadrant" in cat_metrics.columns else 0
    review_count = int((cat_metrics["quadrant"] == "Review").sum()) if "quadrant" in cat_metrics.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.caption(f"Grow: {grow_count} categories")
    with c2:
        st.caption(f"Defend: {defend_count} categories")
    with c3:
        st.caption(f"Build: {build_count} categories")
    with c4:
        st.caption(f"Review: {review_count} categories")


# ── Layer 3: Revenue Decomposition Waterfall ──────────────────────────────

def _render_revenue_waterfall_6drivers(df: pd.DataFrame) -> None:
    """Render revenue decomposition waterfall with 6 drivers."""
    st.subheader(":material/waterfall: Revenue Decomposition (6 Drivers)")

    work = df.copy()
    work["revenue"] = work["price"] * work["quantity"]

    # Weekly components
    weekly = work.copy()
    weekly["date"] = pd.to_datetime(weekly["date"])
    weekly["week"] = weekly["date"].dt.to_period("W").dt.start_time
    weekly = (
        weekly.groupby("week")
        .agg(
            customers=("customer_id", "nunique"),
            transactions=("transaction_id", "nunique"),
            units=("quantity", "sum"),
            revenue=("revenue", "sum"),
        )
        .sort_index()
    )

    if len(weekly) < 2:
        show(empty_state("Need at least 2 weeks of data for waterfall"))
        return

    curr, prev = weekly.iloc[-1], weekly.iloc[-2]

    # Compute 6 drivers using multiplicative decomposition
    # Revenue = customers × (transactions/customers) × (units/transactions) × (revenue/units)
    # Drivers: customer growth, existing frequency change, basket size change,
    # product mix effect, price effect, lost customers

    # Driver 1: Customer growth
    curr_cust = curr["customers"]
    prev_cust = prev["customers"]
    (curr_cust - prev_cust) / max(prev_cust, 1)

    # Driver 2: Existing customer frequency change
    # frequency = transactions / customers
    curr_freq = curr["transactions"] / max(curr_cust, 1)
    prev_freq = prev["transactions"] / max(prev_cust, 1)
    (curr_freq - prev_freq) / max(prev_freq, 1)

    # Driver 3: Basket size change
    curr_basket = curr["units"] / max(curr["transactions"], 1)
    prev_basket = prev["units"] / max(prev["transactions"], 1)
    (curr_basket - prev_basket) / max(prev_basket, 1)

    # Driver 4: Product mix effect
    # Estimate: ratio of current to prior unit revenue
    curr_price_per_unit = curr["revenue"] / max(curr["units"], 1)
    prev_price_per_unit = prev["revenue"] / max(prev["units"], 1)
    (curr_price_per_unit - prev_price_per_unit) / max(prev_price_per_unit, 1)

    # Driver 5: Price effect
    # Simplified: price change within same basket
    (curr_price_per_unit - prev_price_per_unit) / max(prev_price_per_unit, 1)

    # Driver 6: Lost customers (churn)
    # Estimate from transaction count drop vs customer count drop
    curr["transactions"]
    prev["transactions"]
    lost_customers_estimate = max(0, prev_cust - curr_cust) if prev_cust > curr_cust else 0

    # Calculate total revenue change attribution (log-ratio multiplicative)
    # Total change factor
    if prev["revenue"] > 0 and curr["revenue"] > 0:
        total_change = curr["revenue"] / prev["revenue"] - 1
    else:
        total_change = 0.0

    # Normalize driver contributions to sum approximately to total change
    # Using logarithmic attribution for multiplicative drivers
    logs: dict[str, float] = {}
    total_log = 0.0
    for label, curr_val, prev_val in [
        ("Customer Growth", curr_cust, prev_cust),
        ("Frequency Change", curr_freq, prev_freq),
        ("Basket Size", curr_basket, prev_basket),
        ("Product Mix", curr_price_per_unit, prev_price_per_unit),
        ("Price Effect", curr_price_per_unit, prev_price_per_unit),
    ]:
        if prev_val > 0 and curr_val > 0:
            val = float(np.log(curr_val / prev_val))
            logs[label] = val
            total_log += val
    if total_log == 0:
        logs = {k: 0.0 for k in logs}
    else:
        logs = {k: v / total_log for k, v in logs.items()}

    # Render waterfall
    fig = new_fig(height=400)

    # Waterfall with 6 drivers + total
    driver_labels = [
        "Customer Growth",
        "Frequency Change",
        "Basket Size",
        "Product Mix",
        "Price Effect",
        "Lost Customers",
    ]

    # Values: use log-ratio contributions, converted to percentage points
    # Add lost customers as absolute impact
    waterfall_values = []
    waterfall_labels = []
    waterfall_measures = []

    for _i, label in enumerate(driver_labels[:5]):  # first 5 from log-ratio
        if label in logs:
            val = logs[label]
        else:
            val = 0.0
        waterfall_values.append(val)
        waterfall_labels.append(label)
        waterfall_measures.append("relative")

    # Last driver: lost customers (absolute)
    waterfall_values.append(float(lost_customers_estimate) / max(prev["revenue"], 1))
    waterfall_labels.append("Lost Customers")
    waterfall_measures.append("absolute")

    # Total
    waterfall_values.append(total_change)
    waterfall_labels.append("Total Change")
    waterfall_measures.append("total")

    fig.add_trace(
        go.Waterfall(
            name="Revenue Drivers",
            orientation="v",
            measure=waterfall_measures,
            x=waterfall_labels,
            text=[f"€{v:,.0f}" if abs(v) > 0.01 else "0.00%" for v in waterfall_values],
            textposition="outside",
            y=waterfall_values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#4E79A7"}},
            decreasing={"marker": {"color": "#E15759"}},
            totals={"marker": {"color": "#59A14F"}},
        )
    )

    fig.update_layout(
        yaxis={"title": "Revenue Change Share", "tickformat": ".1%"},
        xaxis={"title": "Driver"},
        height=400,
        margin={"l": 40, "r": 20, "t": 50, "b": 80},
    )
    fig.update_xaxes(tickangle=-15)

    show(fig)

    # Insight caption
    main_driver = max(logs, key=logs.get) if logs else "—"
    st.caption(
        f"Main driver of change: {main_driver} "
        f"({logs.get(main_driver, 0):.1%} log-ratio contribution). "
        "Customer growth, frequency, basket size, product mix, and price effects "
        "decompose the week-over-week revenue change."
    )


# ── Layer 4: Growth Driver Matrix ──────────────────────────────────────────

def _render_growth_driver_matrix(df: pd.DataFrame) -> None:
    """Render Growth Driver Matrix with 4 quadrants: Emerging/Growth/Draggers/Critical."""
    st.subheader(":material/trending_up: Growth Driver Matrix")

    work = df.copy()
    work["revenue"] = work["price"] * work["quantity"]

    # Build per-SKU or per-category growth drivers
    if "category" in work.columns:
        # Per-category analysis
        metrics = (
            work.groupby("category")
            .agg(
                revenue=("revenue", "sum"),
                customers=("customer_id", "nunique"),
                transactions=("transaction_id", "nunique"),
                n_products=("stockcode", "nunique"),
                avg_basket=("revenue", lambda x: x.sum() / work.loc[x.index, "quantity"].sum() * x.count()),
            )
            .reset_index()
        )
        metrics["growth_rate"] = metrics["revenue"].pct_change().fillna(0)  # placeholder
        metrics["reach_pct"] = metrics["customers"] / metrics["customers"].sum() * 100
        metrics["penetration"] = metrics["transactions"] / (metrics["customers"] * metrics["n_products"]) * 100
    else:
        # Per-SKU analysis
        metrics = (
            work.groupby("stockcode")
            .agg(
                revenue=("revenue", "sum"),
                customers=("customer_id", "nunique"),
                transactions=("transaction_id", "nunique"),
                n_transactions=("transaction_id", "count"),
            )
            .reset_index()
        )
        metrics["growth_rate"] = 0.0  # placeholder without time series
        metrics["reach_pct"] = metrics["customers"] / metrics["customers"].sum() * 100
        metrics["penetration"] = metrics["transactions"] / (metrics["customers"] + 1e-6) * 100

    # Assign quadrant: Emerging / Growth / Draggers / Critical Risks
    # X-axis: Growth Rate (revenue growth)
    # Y-axis: Penetration / Reach (customer penetration)
    # Size: Revenue

    # Normalize
    growth_min, growth_max = metrics["growth_rate"].min(), metrics["growth_rate"].max()
    reach_min, reach_max = metrics["reach_pct"].min(), metrics["reach_pct"].max()
    size_min, size_max = metrics["revenue"].min(), metrics["revenue"].max()

    # Avoid division by zero
    if growth_max > growth_min:
        metrics["growth_norm"] = (metrics["growth_rate"] - growth_min) / (growth_max - growth_min)
    else:
        metrics["growth_norm"] = 0.5

    if reach_max > reach_min:
        metrics["reach_norm"] = (metrics["reach_pct"] - reach_min) / (reach_max - reach_min)
    else:
        metrics["reach_norm"] = 0.5

    if size_max > size_min:
        metrics["size_norm"] = (metrics["revenue"] - size_min) / (size_max - size_min)
    else:
        metrics["size_norm"] = 0.5

    # Quadrant assignment based on medians
    growth_median = metrics["growth_norm"].median()
    reach_median = metrics["reach_norm"].median()

    def assign_quadrant(row):
        high_growth = row["growth_norm"] >= growth_median
        high_reach = row["reach_norm"] >= reach_median
        if high_growth and high_reach:
            return "Growth Engines"
        elif high_growth and not high_reach:
            return "Emerging Opportunities"
        elif not high_growth and high_reach:
            return "Critical Risks"
        else:
            return "Draggers"

    metrics["quadrant"] = metrics.apply(assign_quadrant, axis=1)

    # Render 4-quadrant bubble matrix
    fig = new_fig(height=500)

    # Calculate actual data ranges for quadrant backgrounds
    x_min = metrics["growth_norm"].min()
    x_max = metrics["growth_norm"].max()
    y_min = metrics["reach_norm"].min()
    y_max = metrics["reach_norm"].max()

    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    # Draw quadrant background using actual data coordinates
    fig.add_shape(
        type="rect", x0=x_mid, x1=x_max, y0=y_mid, y1=y_max,
        fillcolor="rgba(78, 121, 167, 0.1)", line_width=0, layer="below"
    )  # Growth
    fig.add_shape(
        type="rect", x0=x_mid, x1=x_max, y0=y_min, y1=y_mid,
        fillcolor="rgba(89, 161, 79, 0.1)", line_width=0, layer="below"
    )  # Emerging
    fig.add_shape(
        type="rect", x0=x_min, x1=x_mid, y0=y_min, y1=y_mid,
        fillcolor="rgba(225, 87, 89, 0.1)", line_width=0, layer="below"
    )  # Critical Risks
    fig.add_shape(
        type="rect", x0=x_min, x1=x_mid, y0=y_mid, y1=y_max,
        fillcolor="rgba(242, 142, 43, 0.1)", line_width=0, layer="below"
    )  # Draggers

    color_map = {
        "Growth Engines": "#4E79A7",
        "Emerging Opportunities": "#59A14F",
        "Critical Risks": "#E15759",
        "Draggers": "#F28E2B",
    }

    max(metrics["size_norm"].max(), 1) if not metrics.empty else 1

    for quad in ["Growth Engines", "Emerging Opportunities", "Critical Risks", "Draggers"]:
        quad_df = metrics[metrics["quadrant"] == quad]
        if not quad_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=quad_df["growth_norm"],
                    y=quad_df["reach_norm"],
                    mode="markers+text",
                    marker={
                        "size": quad_df["size_norm"] * 50 + 15,
                        "color": color_map.get(quad, PALETTE[0]),
                        "opacity": 0.8,
                        "line": {"width": 1, "color": "white"},
                    },
                    text=quad_df["stockcode"] if "stockcode" in metrics.columns else quad_df["category"],
                    textposition="top center",
                    textfont={"size": 8},
                    name=quad,
                    hovertemplate="<b>%{text}</b><br>" +
                                  "Growth: %{x:.1%}<br>" +
                                  "Reach: %{y:.1f}%<br>" +
                                  "Revenue: €%{customdata:,.0f}<br>" +
                                  "<extra></extra>",
                    customdata=quad_df["revenue"],
                )
            )

    # Add quadrant labels using actual data coordinates
    fig.add_annotation(
        x=(x_mid + x_max) / 2, y=(y_mid + y_max) / 2,
        text="Growth Engines", showarrow=False,
        font={"size": 16, "color": "#4E79A7", "weight": "bold"}
    )
    fig.add_annotation(
        x=(x_mid + x_max) / 2, y=(y_min + y_mid) / 2,
        text="Emerging Opportunities", showarrow=False,
        font={"size": 16, "color": "#59A14F", "weight": "bold"}
    )
    fig.add_annotation(
        x=(x_min + x_mid) / 2, y=(y_min + y_mid) / 2,
        text="Critical Risks", showarrow=False,
        font={"size": 16, "color": "#E15759", "weight": "bold"}
    )
    fig.add_annotation(
        x=(x_min + x_mid) / 2, y=(y_mid + y_max) / 2,
        text="Draggers", showarrow=False,
        font={"size": 16, "color": "#F28E2B", "weight": "bold"}
    )

    fig.update_layout(
        xaxis={"title": "Growth Rate (normalized)", "tickformat": ".1%"},
        yaxis={"title": "Customer Reach (normalized)", "tickformat": ".1f"},
        margin={"l": 80, "r": 30, "t": 80, "b": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )

    show(fig)

    # Summary
    q_counts = metrics["quadrant"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.caption(f"Growth Engines: {q_counts.get('Growth Engines', 0)}")
    with c2:
        st.caption(f"Emerging: {q_counts.get('Emerging Opportunities', 0)}")
    with c3:
        st.caption(f"Critical Risks: {q_counts.get('Critical Risks', 0)}")
    with c4:
        st.caption(f"Draggers: {q_counts.get('Draggers', 0)}")


# ── Layer 5: Decision Table + Executive Agenda ────────────────────────────

def _render_decision_table(df: pd.DataFrame, capabilities: dict) -> None:
    """Render decision priority table sortable by decision priority."""
    st.subheader(":material/clipboard: Decision Priority Table")

    work = df.copy()
    work["revenue"] = work["price"] * work["quantity"]

    # Build decision rows
    rows = []

    if "category" in work.columns:
        # Per-category decisions
        cat_stats = (
            work.groupby("category")
            .agg(
                revenue=("revenue", "sum"),
                customers=("customer_id", "nunique"),
                transactions=("transaction_id", "nunique"),
                n_products=("stockcode", "nunique"),
            )
            .reset_index()
        )

        for _, row in cat_stats.iterrows():
            # Determine decision priority based on multiple factors
            rev = row["revenue"]
            cust = row["customers"]
            n_prod = row["n_products"]

            # Priority score: blend revenue, customer count, product count
            # Higher score = higher priority
            if capabilities.get("sufficient_customers_500") and capabilities.get("sufficient_skus_50"):
                # Evidence-based priority scoring
                score = (
                    rev * 0.5
                    + cust * 10
                    + n_prod * 50
                )
            else:
                score = rev * 0.7 + cust * 5

            # Decision type based on metrics
            if rev > cat_stats["revenue"].median() * 2 and cust > cat_stats["customers"].median():
                decision = "Invest"
            elif rev > cat_stats["revenue"].median() and cust <= cat_stats["customers"].median():
                decision = "Protect"
            elif rev <= cat_stats["revenue"].median() * 0.5:
                decision = "Review"
            else:
                decision = "Monitor"

            rows.append({
                "category": row["category"],
                "revenue": rev,
                "customers": cust,
                "n_products": n_prod,
                "decision_priority": score,
                "decision_type": decision,
            })
    else:
        # Per-SKU decisions
        sku_stats = (
            work.groupby("stockcode")
            .agg(
                revenue=("revenue", "sum"),
                customers=("customer_id", "nunique"),
                transactions=("transaction_id", "nunique"),
            )
            .reset_index()
        )

        for _, row in sku_stats.iterrows():
            rev = row["revenue"]
            cust = row["customers"]

            if capabilities.get("sufficient_customers_500") and capabilities.get("sufficient_skus_50"):
                score = rev * 0.5 + cust * 2
            else:
                score = rev * 0.7 + cust * 1

            if rev > sku_stats["revenue"].median() * 1.5:
                decision = "Invest"
            elif rev <= sku_stats["revenue"].median() * 0.4:
                decision = "Review"
            else:
                decision = "Monitor"

            rows.append({
                "sku": row["stockcode"],
                "revenue": rev,
                "customers": cust,
                "decision_priority": score,
                "decision_type": decision,
            })

    if not rows:
        st.caption("No decision data available")
        return

    decisions_df = pd.DataFrame(rows)

    # Sort by decision priority (descending)
    decisions_df = decisions_df.sort_values("decision_priority", ascending=False).reset_index(drop=True)

    # Add rank
    decisions_df["rank"] = range(1, len(decisions_df) + 1)

    # Render sortable table
    st.caption("Sorted by decision priority (highest first)")

    # Format for display
    display_df = decisions_df.copy()
    display_df["revenue_fmt"] = display_df["revenue"].map(lambda v: f"€{v:,.0f}")
    display_df["customers_fmt"] = display_df["customers"].map(lambda v: f"{int(v):,}")

    if "category" in display_df.columns:
        display_df = display_df[["rank", "category", "revenue_fmt", "customers_fmt"] + [c for c in ["n_products"] if c in display_df.columns]]
        st.dataframe(
            display_df[["rank", "category", "revenue_fmt", "customers_fmt"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(
            display_df[["rank", "sku", "revenue_fmt", "customers_fmt", "decision_type"]],
            use_container_width=True,
            hide_index=True,
        )

    # Executive agenda: top 5 priorities
    st.divider()
    st.subheader(":material/star: This Period's 5 Priorities")

    top5 = decisions_df.head(5).sort_values("decision_priority", ascending=False)

    for i, (_, row) in enumerate(top5.iterrows(), 1):
        priority_label = f"Priority {i}: {row['decision_type']}"
        if "category" in row:
            label_text = f"{i}. {row['category']} — {priority_label}"
        else:
            label_text = f"{i}. {row['stockcode']} — {priority_label}"

        # Color coding by decision type
        decision_colors = {
            "Invest": "#4E79A7",
            "Protect": "#59A14F",
            "Review": "#F28E2B",
            "Monitor": "#EDC948",
        }
        color = decision_colors.get(row["decision_type"], "#888888")

        st.markdown(
            f'<span style="background-color: {color}; color: white; padding: 4px 8px; '
            f'border-radius: 4px; font-weight: 500; font-size: 0.9em;">{label_text}</span>',
            unsafe_allow_html=True,
        )

        # Detail caption
        detail = f"Revenue: €{row['revenue']:,.0f} | Customers: {int(row['customers']):,}"
        if "n_products" in row:
            detail += f" | Products: {int(row['n_products'])}"
        st.caption(detail)


# ── Main render function ──────────────────────────────────────────────────

def render(df: pd.DataFrame) -> None:
    """Render the 5-layer Overview tab."""
    summary = get_data_summary(df)
    capabilities = build_dataset_capabilities(df)
    work = df.copy()
    work["revenue"] = work["price"] * work["quantity"]

    # ── Layer 1: KPI Row ──────────────────────────────────────────────
    st.subheader(":material/dashboard: Decision KPIs")
    _render_kpi_row(df, capabilities)

    st.caption(
        f"Date range: {summary['date_range']} | "
        f"Avg basket value: €{summary['avg_basket_value']:.2f} | "
        f"Total revenue: €{summary['total_revenue']:,.0f}"
    )

    st.divider()

    # ── Layer 2: Primary Strategic Matrix ─────────────────────────────
    _render_primary_matrix(df, capabilities)

    st.divider()

    # ── Layer 3: Revenue Decomposition Waterfall ──────────────────────
    _render_revenue_waterfall_6drivers(df)

    st.divider()

    # ── Layer 4: Growth Driver Matrix ─────────────────────────────────
    _render_growth_driver_matrix(df)

    st.divider()

    # ── Layer 5: Decision Table + Executive Agenda ────────────────────
    _render_decision_table(df, capabilities)

    st.divider()

    # ── Bottom: Insights + Data Quality (existing patterns) ───────────
    st.subheader(":material/radar: Top Insights")
    from src.analytics.insights import generate_overview_insights

    try:
        insights = generate_overview_insights(df)
        from src.ui.components_utils import render_insight_cards
        render_insight_cards(insights)
    except Exception as e:
        st.caption(f"Insights unavailable: {e}")

    st.divider()
    st.subheader(":material/verified: Data Quality")
    quality_report = st.session_state.get("quality_report")
    if quality_report:
        st.markdown(generate_quality_summary(quality_report))
    else:
        st.json(summary)

    # ── Mode spec ─────────────────────────────────────────────────────
MODE_SPEC: ModeSpec = ModeSpec(
    key="overview",
    label="Overview",
    icon=":material/dashboard:",
    handler=render,
    requires=(),
)
