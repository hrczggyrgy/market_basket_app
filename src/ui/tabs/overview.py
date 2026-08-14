"""Overview / Dashboard tab.

Follows the app-wide page pattern:
    Scorecard -> Hero visual -> Drivers -> Concentration -> Top insights -> Actions

Every visual is connected to a decision: the revenue decomposition explains
WHERE growth/decline comes from, the Pareto explains concentration risk, and
the insight cards carry evidence + recommended actions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.basket_metrics import spc_revenue_trend
from src.analytics.data import get_data_summary
from src.analytics.data_quality import generate_quality_summary
from src.ui.components import render_insight_cards, render_metric_row
from src.ui.plots import PALETTE, empty_state, new_fig, show
from src.ui.registry import ModeSpec

_DRIVER_LABELS = {
    "customers": "Customer count",
    "frequency": "Shopping frequency",
    "basket_size": "Basket size",
    "price_per_unit": "Price per unit",
}


def _weekly_components(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly revenue = customers x frequency x basket_size x price_per_unit."""
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["revenue"] = work["price"] * work["quantity"]
    work["week"] = work["date"].dt.to_period("W").dt.start_time
    weekly = (
        work.groupby("week")
        .agg(
            customers=("customer_id", "nunique"),
            transactions=("transaction_id", "nunique"),
            units=("quantity", "sum"),
            revenue=("revenue", "sum"),
        )
        .sort_index()
    )
    if weekly.empty:
        return weekly
    weekly["frequency"] = weekly["transactions"] / weekly["customers"].replace(0, np.nan)
    weekly["basket_size"] = weekly["units"] / weekly["transactions"].replace(0, np.nan)
    weekly["price_per_unit"] = weekly["revenue"] / weekly["units"].replace(0, np.nan)
    return weekly


def _growth_attribution(comps: pd.DataFrame) -> dict[str, float]:
    """Multiplicative log-ratio attribution between the last two weeks."""
    if len(comps) < 2:
        return {}
    curr, prev = comps.iloc[-1], comps.iloc[-2]
    logs: dict[str, float] = {}
    total_log = 0.0
    for f in ("customers", "frequency", "basket_size", "price_per_unit"):
        if prev[f] > 0 and curr[f] > 0:
            val = float(np.log(curr[f] / prev[f]))
            logs[f] = val
            total_log += val
    if total_log == 0:
        return {f: 0.0 for f in logs}
    return {f: v / total_log for f, v in logs.items()}


def _render_revenue_decomposition(df: pd.DataFrame) -> None:
    st.subheader(":material/account_tree: Revenue Decomposition")
    comps = _weekly_components(df)
    if len(comps) < 2:
        show(empty_state("Need at least 2 weeks of data"))
        return

    curr, prev = comps.iloc[-1], comps.iloc[-2]
    growth = float(curr["revenue"] / prev["revenue"] - 1) if prev["revenue"] > 0 else 0.0
    attribution = _growth_attribution(comps)

    st.caption(
        "Revenue = customers × shopping frequency × basket size × price per unit. "
        f"Week over week: {growth:+.1%}. The bars show each driver's share of that change."
    )

    fig = new_fig()
    drivers = ["customers", "frequency", "basket_size", "price_per_unit"]
    shares = [attribution.get(d, 0.0) for d in drivers]
    colors = ["#E15759" if s < 0 else "#4E79A7" for s in shares]
    fig.add_trace(
        go.Bar(
            x=[_DRIVER_LABELS[d] for d in drivers],
            y=shares,
            marker={"color": colors},
            hovertemplate="%{x}: %{y:+.0%} of the change<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis={"title": "Share of week-over-week change", "tickformat": ".0%"},
        xaxis={"title": ""},
    )
    show(fig)

    if attribution:
        driver = max(attribution, key=attribution.get)
        st.caption(
            f"**Main driver: {_DRIVER_LABELS[driver]}** ({attribution[driver]:+.0%} of the change). "
            + {
                "customers": "Grow acquisition/win-back or fix retention.",
                "frequency": "Shoppers are visiting more/less often — work repurchase drivers.",
                "basket_size": "Shoppers put more/fewer units in the basket — work cross-sell and add-ons.",
                "price_per_unit": "Price mix shifted — review promotions, tiering and price changes.",
            }[driver]
        )


def _render_spc_trend(df: pd.DataFrame) -> None:
    st.subheader(":material/trending_up: Revenue Trend (SPC)")
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["revenue"] = work["price"] * work["quantity"]
    series = (
        work["revenue"].groupby(work["date"].dt.to_period("W").dt.start_time).sum().sort_index()
    )
    series.index = pd.to_datetime(series.index)

    try:
        spc = spc_revenue_trend(series)
    except Exception as e:
        st.error(f"Failed to compute SPC analysis: {e}")
        return

    fig = new_fig()
    fig.add_trace(
        go.Scatter(
            x=spc["period"],
            y=spc["ucl"],
            mode="lines",
            line={"color": PALETTE[2], "width": 1, "dash": "dash"},
            name="UCL",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spc["period"],
            y=spc["lcl"],
            mode="lines",
            line={"color": PALETTE[2], "width": 1, "dash": "dash"},
            name="LCL",
            fill="tonexty",
            fillcolor="rgba(255, 0, 0, 0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spc["period"],
            y=spc["center"],
            mode="lines",
            line={"color": PALETTE[1], "width": 2, "dash": "dot"},
            name="Center (trailing mean)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spc["period"],
            y=spc["revenue"],
            mode="lines",
            line={"color": PALETTE[0], "width": 2},
            name="Revenue",
        )
    )
    anomalies = spc[spc["anomaly"]]
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["period"],
                y=anomalies["revenue"],
                mode="markers",
                name="Anomaly",
                marker={"color": "red", "size": 10, "symbol": "x"},
                customdata=anomalies["rule"],
                hovertemplate="Anomaly (rule: %{customdata})<br>Revenue: %{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(yaxis={"title": "Revenue"}, xaxis={"title": "Week"})
    show(fig)
    st.caption(
        "SPC control limits: trailing rolling mean ± 2σ (Rule 1: outside limits; "
        "Rule 3: 7 consecutive points on one side). Red X = anomaly — investigate "
        "before extrapolating the trend."
    )


def _render_spc_trend_cached(spc: pd.DataFrame) -> None:
    """Render SPC trend from precomputed data."""
    st.subheader(":material/trending_up: Revenue Trend (SPC)")

    fig = new_fig()
    fig.add_trace(
        go.Scatter(
            x=spc["period"],
            y=spc["ucl"],
            mode="lines",
            line={"color": PALETTE[2], "width": 1, "dash": "dash"},
            name="UCL",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spc["period"],
            y=spc["lcl"],
            mode="lines",
            line={"color": PALETTE[2], "width": 1, "dash": "dash"},
            name="LCL",
            fill="tonexty",
            fillcolor="rgba(255, 0, 0, 0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spc["period"],
            y=spc["center"],
            mode="lines",
            line={"color": PALETTE[1], "width": 2, "dash": "dot"},
            name="Center (trailing mean)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spc["period"],
            y=spc["revenue"],
            mode="lines",
            line={"color": PALETTE[0], "width": 2},
            name="Revenue",
        )
    )
    anomalies = spc[spc["anomaly"]]
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["period"],
                y=anomalies["revenue"],
                mode="markers",
                name="Anomaly",
                marker={"color": "red", "size": 10, "symbol": "x"},
                customdata=anomalies["rule"],
                hovertemplate="Anomaly (rule: %{customdata})<br>Revenue: %{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(yaxis={"title": "Revenue"}, xaxis={"title": "Week"})
    show(fig)
    st.caption(
        "SPC control limits: trailing rolling mean ± 2σ (Rule 1: outside limits; "
        "Rule 3: 7 consecutive points on one side). Red X = anomaly — investigate "
        "before extrapolating the trend."
    )


def _render_calendar_heatmap(df: pd.DataFrame) -> None:
    st.subheader(":material/calendar_month: Daily Revenue Calendar")
    work = df.copy()
    work["revenue"] = work["price"] * work["quantity"]
    daily = work["revenue"].groupby(work["date"].dt.date).sum().reset_index()
    daily.columns = ["date", "revenue"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily["week_of_year"] = daily["date"].dt.isocalendar().week
    daily["day_of_week"] = daily["date"].dt.dayofweek
    if daily.empty:
        show(empty_state("No revenue data"))
        return
    pivot = daily.pivot_table(
        index="week_of_year", columns="day_of_week", values="revenue", fill_value=0
    )
    for d in range(7):
        if d not in pivot.columns:
            pivot[d] = 0
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            y=pivot.index.astype(str),
            colorscale="Blues",
            colorbar={"title": "Revenue"},
            hovertemplate="Week %{y}<br>%{x}<br>Revenue: %{z:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis={"title": "Day of Week"},
        yaxis={"title": "Week of Year", "autorange": "reversed"},
        height=max(300, len(pivot) * 20 + 100),
    )
    show(fig)


def _render_concentration(df: pd.DataFrame) -> None:
    st.subheader(":material/pie_chart: Revenue Concentration (Pareto)")
    work = df.copy()
    revenue = (
        (work["price"] * work["quantity"])
        .groupby(work["stockcode"])
        .sum()
        .sort_values(ascending=False)
    )
    total = revenue.sum()
    if total <= 0:
        show(empty_state("No revenue data"))
        return
    pareto = pd.DataFrame(
        {
            "product": revenue.index,
            "revenue_share_pct": revenue.values / total * 100,
        }
    )
    pareto["cumulative_share_pct"] = pareto["revenue_share_pct"].cumsum()
    top10 = float(pareto["revenue_share_pct"].head(10).sum())
    hhi = float(((revenue / total) ** 2).sum())

    fig = new_fig()
    fig.add_trace(
        go.Bar(
            x=pareto["product"],
            y=pareto["revenue_share_pct"],
            name="Revenue share (%)",
            marker={"color": PALETTE[4]},
            hovertemplate="%{x}<br>Share: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pareto["product"],
            y=pareto["cumulative_share_pct"],
            name="Cumulative (%)",
            yaxis="y2",
            mode="lines+markers",
            line={"color": PALETTE[1], "width": 2},
        )
    )
    fig.update_layout(
        yaxis={"title": "Revenue share (%)"},
        yaxis2={"title": "Cumulative (%)", "overlaying": "y", "side": "right", "range": [0, 105]},
        xaxis={"tickangle": -45, "nticks": 20},
        bargap=0.35,
    )
    show(fig)
    st.caption(
        f"Top-10 products = {top10:.0%} of revenue (HHI {hhi:.2f}). "
        + (
            "Concentrated: protect the top products' availability and pricing."
            if top10 >= 50
            else "Healthy spread: no single product dominates."
        )
    )


def _render_new_vs_returning(df: pd.DataFrame) -> None:
    st.subheader(":material/groups: New vs. Returning Customers")
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    first_date = work.groupby("customer_id")["date"].transform("min")
    split = (
        pd.DataFrame(
            {
                "week": work["date"].dt.to_period("W").astype(str),
                "is_new": work["date"] == first_date,
            }
        )
        .groupby(["week", "is_new"])
        .size()
        .unstack(fill_value=0)
    )
    split = split.rename(columns={False: "returning", True: "new"})
    for col in ("new", "returning"):
        if col not in split.columns:
            split[col] = 0
    split = split[["new", "returning"]]
    if split.empty or split.sum().sum() == 0:
        show(empty_state("No customer activity"))
        return
    fig = new_fig()
    fig.add_trace(
        go.Scatter(
            x=split.index,
            y=split["new"],
            mode="lines",
            stackgroup="one",
            name="New customers",
            line={"color": PALETTE[0]},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=split.index,
            y=split["returning"],
            mode="lines",
            stackgroup="one",
            name="Returning customers",
            line={"color": PALETTE[2]},
        )
    )
    fig.update_layout(yaxis={"title": "Transactions"}, xaxis={"title": "Week"})
    show(fig)
    st.caption(
        "New = first-ever transaction week. If returning is flat while new grows, retention is the lever."
    )


def _render_basket_distribution(df: pd.DataFrame) -> None:
    st.subheader(":material/shopping_basket: Basket Size Distribution")
    sizes = df.groupby("transaction_id")["stockcode"].nunique()
    if len(sizes) == 0:
        show(empty_state("No transactions"))
        return
    fig = new_fig()
    fig.add_trace(
        go.Histogram(
            x=sizes,
            nbinsx=max(5, int(sizes.max())),
            marker={"color": PALETTE[0]},
            opacity=0.9,
        )
    )
    fig.update_layout(
        yaxis={"title": "Transactions"}, xaxis={"title": "Distinct products per basket"}
    )
    show(fig)
    st.caption(
        "Distribution shape tells you where to push: raising the tail (cross-sell) beats raising the mode."
    )


@st.cache_data(show_spinner="Computing SPC trend...", max_entries=5)
def _cached_spc_trend(df: pd.DataFrame) -> pd.DataFrame:
    from src.analytics.basket_metrics import spc_revenue_trend
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["revenue"] = work["price"] * work["quantity"]
    series = work["revenue"].groupby(work["date"].dt.to_period("W").dt.start_time).sum().sort_index()
    series.index = pd.to_datetime(series.index)
    return spc_revenue_trend(series)


@st.cache_data(show_spinner="Generating insights...", max_entries=5)
def _cached_overview_insights(df: pd.DataFrame) -> pd.DataFrame:
    from src.analytics.insights import generate_overview_insights
    return generate_overview_insights(df)


def render(df: pd.DataFrame) -> None:
    summary = get_data_summary(df)
    work = df.copy()
    work["revenue"] = work["price"] * work["quantity"]

    render_metric_row(
        [
            {"label": "Transactions", "value": f"{len(work):,}"},
            {"label": "Customers", "value": f"{work['customer_id'].nunique():,}"},
            {"label": "Products", "value": f"{work['stockcode'].nunique():,}"},
            {"label": "Revenue", "value": f"€{work['revenue'].sum():,.0f}"},
        ]
    )
    st.caption(
        f"Date range: {summary['date_range']} | Avg basket value: €{summary['avg_basket_value']:.2f}"
    )

    st.divider()
    _render_revenue_decomposition(df)

    st.divider()
    with st.spinner("Loading trend analysis..."):
        spc_data = _cached_spc_trend(df)
    # Render SPC trend from cached data
    _render_spc_trend_cached(spc_data)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        _render_new_vs_returning(df)
    with c2:
        _render_basket_distribution(df)

    st.divider()
    _render_calendar_heatmap(df)

    st.divider()
    _render_concentration(df)

    st.divider()
    st.subheader(":material/radar: Top Insights")
    with st.spinner("Loading insights..."):
        insights = _cached_overview_insights(df)
    render_insight_cards(insights)

    st.divider()
    st.subheader(":material/verified: Data Quality")
    quality_report = st.session_state.get("quality_report")
    if quality_report:
        st.markdown(generate_quality_summary(quality_report))
        if quality_report.low_freq_products:
            with st.expander(
                f"Low-frequency products ({len(quality_report.low_freq_products)})", expanded=False
            ):
                st.dataframe(
                    pd.DataFrame(
                        {
                            "stockcode": quality_report.low_freq_products,
                            "transactions": [
                                quality_report.low_freq_counts.get(p, 0)
                                for p in quality_report.low_freq_products
                            ],
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        if quality_report.basket_outlier_txn_ids:
            with st.expander("Basket size outliers", expanded=False):
                st.write(
                    f"Threshold: {quality_report.basket_outlier_threshold} items "
                    f"(above {quality_report.basket_size_percentile:.0%} percentile)"
                )
                st.write(
                    f"Outlier transaction IDs: {', '.join(quality_report.basket_outlier_txn_ids[:50])}"
                )
                if len(quality_report.basket_outlier_txn_ids) > 50:
                    st.caption(f"... and {len(quality_report.basket_outlier_txn_ids) - 50} more")
        if quality_report.duplicate_count > 0:
            with st.expander(
                f"Duplicate transactions ({quality_report.duplicate_count})", expanded=False
            ):
                st.write(
                    f"Duplicate transaction IDs: {', '.join(quality_report.duplicate_txn_ids[:50])}"
                )
                if len(quality_report.duplicate_txn_ids) > 50:
                    st.caption(f"... and {len(quality_report.duplicate_txn_ids) - 50} more")
        if quality_report.incomplete_rows > 0:
            with st.expander(f"Incomplete rows ({quality_report.incomplete_rows})", expanded=False):
                for col, cnt in quality_report.incomplete_row_details.items():
                    st.write(f"- {col}: {cnt} missing")
    else:
        st.json(
            {
                "date_range": summary["date_range"],
                "avg_basket_value": f"€{summary['avg_basket_value']:.2f}",
                "avg_items_per_basket": f"{summary['avg_basket_size']:.2f}",
            }
        )


MODE_SPEC: ModeSpec = ModeSpec(
    key="overview",
    label="Overview",
    icon=":material/dashboard:",
    handler=render,
)
