"""Overview / Dashboard tab."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.data import get_data_summary
from src.analytics.data_quality import generate_quality_summary
from src.ui.plots import PALETTE, empty_state, new_fig, show
from src.ui.registry import ModeSpec


def _revenue_trend(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Daily/weekly/monthly revenue with a rolling average overlay."""
    revenue = df["price"] * df["quantity"]
    if period == "daily":
        key = df["date"].dt.date
        window = 7
    elif period == "weekly":
        key = df["date"].dt.to_period("W").dt.start_time.dt.date
        window = 4
    else:
        key = df["date"].dt.to_period("M").dt.start_time.dt.date
        window = 3
    trend = revenue.groupby(key).sum().sort_index()
    table = pd.DataFrame(
        {
            "revenue": trend,
            "rolling": trend.rolling(window, min_periods=1).mean(),
        }
    )
    table.index = pd.to_datetime(table.index)
    return table


def _new_vs_returning(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly count of first-time (new) vs. repeat (returning) customer transactions."""
    first_date = df.groupby("customer_id")["date"].transform("min")
    split = (
        pd.DataFrame(
            {
                "week": df["date"].dt.to_period("W").astype(str),
                "is_new": df["date"] == first_date,
            }
        )
        .groupby(["week", "is_new"])
        .size()
        .unstack(fill_value=0)
    )
    table = split.rename(columns={False: "returning", True: "new"})
    for col in ("new", "returning"):
        if col not in table.columns:
            table[col] = 0
    return table[["new", "returning"]]


def _revenue_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Per-product revenue share and cumulative share, sorted descending."""
    revenue = (df["price"] * df["quantity"]).groupby(df["stockcode"]).sum().sort_values(ascending=False)
    total = revenue.sum()
    table = pd.DataFrame(
        {
            "product": revenue.index,
            "revenue_share_pct": revenue.values / total * 100,
        }
    )
    table["cumulative_share_pct"] = table["revenue_share_pct"].cumsum()
    return table


def _render_revenue_trend(df: pd.DataFrame) -> None:
    st.subheader(":material/trending_up: Revenue Trend")
    period = st.radio("Period", ["daily", "weekly", "monthly"], horizontal=True, key="ov_trend_period")
    trend = _revenue_trend(df, period)
    window = {"daily": 7, "weekly": 4, "monthly": 3}[period]

    fig = new_fig()
    fig.add_trace(
        go.Scatter(
            x=trend.index,
            y=trend["revenue"],
            mode="lines",
            name="Revenue",
            line={"color": PALETTE[0], "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trend.index,
            y=trend["rolling"],
            mode="lines",
            name=f"{window}-period avg",
            line={"color": PALETTE[1], "width": 2, "dash": "dash"},
        )
    )
    fig.update_layout(yaxis={"title": "Revenue"})
    show(fig)


def _render_customer_split(df: pd.DataFrame) -> None:
    st.subheader(":material/groups: New vs. Returning Customers")
    split = _new_vs_returning(df)
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
    fig.update_layout(yaxis={"title": "Transactions"}, xaxis={"title": "Distinct products per basket"})
    show(fig)


def _render_pareto(df: pd.DataFrame) -> None:
    st.subheader(":material/pie_chart: Revenue Concentration (Pareto)")
    pareto = _revenue_pareto(df)
    if pareto.empty:
        st.info("No revenue data.")
        return

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


def render(df: pd.DataFrame) -> None:
    """Render the overview dashboard."""
    summary = get_data_summary(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions", f"{summary['n_transactions']:,}")
    col2.metric("Customers", f"{summary['n_customers']:,}")
    col3.metric("Products", f"{summary['n_products']:,}")
    col4.metric("Revenue", f"${summary['total_revenue']:,.2f}")

    st.caption(f"Date range: {summary['date_range']}")

    st.divider()

    _render_revenue_trend(df)

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        _render_customer_split(df)
    with c2:
        _render_basket_distribution(df)

    st.divider()
    _render_pareto(df)

    # Data quality
    st.divider()
    st.subheader(":material/verified: Data Quality")

    # Check if quality report is in session state
    quality_report = st.session_state.get("quality_report")
    if quality_report:
        st.markdown(generate_quality_summary(quality_report))

        # Show details in expanders
        if quality_report.low_freq_products:
            with st.expander(f"Low-frequency products ({len(quality_report.low_freq_products)})", expanded=False):
                freq_df = pd.DataFrame({
                    "stockcode": quality_report.low_freq_products,
                    "transactions": [quality_report.low_freq_counts.get(p, 0) for p in quality_report.low_freq_products]
                })
                st.dataframe(freq_df, use_container_width=True, hide_index=True)

        if quality_report.basket_outlier_txn_ids:
            with st.expander(f"Basket size outliers ({len(quality_report.basket_outlier_txn_ids)})", expanded=False):
                st.write(f"Threshold: {quality_report.basket_outlier_threshold} items (above {quality_report.basket_size_percentile:.0%} percentile)")
                st.write(f"Outlier transaction IDs: {', '.join(quality_report.basket_outlier_txn_ids[:50])}")
                if len(quality_report.basket_outlier_txn_ids) > 50:
                    st.caption(f"... and {len(quality_report.basket_outlier_txn_ids) - 50} more")

        if quality_report.duplicate_count > 0:
            with st.expander(f"Duplicate transactions ({quality_report.duplicate_count})", expanded=False):
                st.write(f"Duplicate transaction IDs: {', '.join(quality_report.duplicate_txn_ids[:50])}")
                if len(quality_report.duplicate_txn_ids) > 50:
                    st.caption(f"... and {len(quality_report.duplicate_txn_ids) - 50} more")

        if quality_report.incomplete_rows > 0:
            with st.expander(f"Incomplete rows ({quality_report.incomplete_rows})", expanded=False):
                for col, cnt in quality_report.incomplete_row_details.items():
                    st.write(f"- {col}: {cnt} missing")
    else:
        st.json({
            "date_range": summary['date_range'],
            "avg_basket_value": f"${summary['avg_basket_value']:.2f}",
            "avg_items_per_basket": f"{summary['avg_basket_size']:.2f}",
        })


MODE_SPEC: ModeSpec = ModeSpec(
    key="overview",
    label="Overview",
    icon=":material/dashboard:",
    handler=render,
)
