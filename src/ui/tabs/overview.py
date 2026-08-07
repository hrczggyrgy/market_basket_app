"""Overview / Dashboard tab."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.basket_metrics import spc_revenue_trend
from src.analytics.data import get_data_summary
from src.analytics.data_quality import generate_quality_summary
from src.ui.plots import PALETTE, empty_state, new_fig, show
from src.ui.registry import ModeSpec


def _period_over_period_delta(df: pd.DataFrame, metric: str, period: str = "weekly") -> tuple[float, float, str]:
    """
    Compute period-over-period delta for a metric.
    Returns (current_value, previous_value, delta_color) where delta_color is "normal", "inverse", or "off".
    """
    df = df.copy()
    if period == "weekly":
        df["period"] = df["date"].dt.to_period("W").dt.start_time
    elif period == "monthly":
        df["period"] = df["date"].dt.to_period("M").dt.start_time
    else:
        df["period"] = df["date"].dt.date

    periods = sorted(df["period"].unique())
    if len(periods) < 2:
        return 0.0, 0.0, "off"

    curr_period = periods[-1]
    prev_period = periods[-2]

    curr_df = df[df["period"] == curr_period]
    prev_df = df[df["period"] == prev_period]

    if metric == "transactions":
        curr = len(curr_df)
        prev = len(prev_df)
        color = "normal"  # more transactions = good
    elif metric == "customers":
        curr = curr_df["customer_id"].nunique()
        prev = prev_df["customer_id"].nunique()
        color = "normal"  # more customers = good
    elif metric == "products":
        curr = curr_df["stockcode"].nunique()
        prev = prev_df["stockcode"].nunique()
        color = "off"  # neutral
    elif metric == "revenue":
        curr = (curr_df["price"] * curr_df["quantity"]).sum()
        prev = (prev_df["price"] * prev_df["quantity"]).sum()
        color = "normal"  # more revenue = good
    elif metric == "avg_basket_value":
        curr_baskets = curr_df.groupby("transaction_id")["price"].apply(lambda x: (x * curr_df.loc[x.index, "quantity"]).sum())
        prev_baskets = prev_df.groupby("transaction_id")["price"].apply(lambda x: (x * prev_df.loc[x.index, "quantity"]).sum())
        curr = curr_baskets.mean() if len(curr_baskets) > 0 else 0
        prev = prev_baskets.mean() if len(prev_baskets) > 0 else 0
        color = "normal"
    elif metric == "return_rate":
        # Negative metric - higher returns = bad
        curr = 0.0
        prev = 0.0
        color = "inverse"
    else:
        return 0.0, 0.0, "off"

    return curr, prev, color


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
    st.subheader(":material/trending_up: Revenue Trend (SPC)")
    period = st.radio("Period", ["daily", "weekly", "monthly"], horizontal=True, key="ov_trend_period")

    # Compute daily revenue series for SPC
    revenue = df["price"] * df["quantity"]
    if period == "daily":
        key = df["date"].dt.date
    elif period == "weekly":
        key = df["date"].dt.to_period("W").dt.start_time.dt.date
    else:
        key = df["date"].dt.to_period("M").dt.start_time.dt.date
    series = revenue.groupby(key).sum().sort_index()
    series.index = pd.to_datetime(series.index)

    # SPC analysis
    spc = spc_revenue_trend(series)

    fig = new_fig()
    # UCL/LCL bands
    fig.add_trace(
        go.Scatter(
            x=spc["period"],
            y=spc["ucl"],
            mode="lines",
            line={"color": PALETTE[2], "width": 1, "dash": "dash"},
            name="UCL",
            hovertemplate="UCL: %{y:.2f}<extra></extra>",
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
            hovertemplate="LCL: %{y:.2f}<extra></extra>",
        )
    )
    # Center line
    fig.add_trace(
        go.Scatter(
            x=spc["period"],
            y=spc["center"],
            mode="lines",
            name="Center (trailing mean)",
            line={"color": PALETTE[1], "width": 2, "dash": "dot"},
            hovertemplate="Center: %{y:.2f}<extra></extra>",
        )
    )
    # Revenue line
    fig.add_trace(
        go.Scatter(
            x=spc["period"],
            y=spc["revenue"],
            mode="lines",
            name="Revenue",
            line={"color": PALETTE[0], "width": 2},
            hovertemplate="Revenue: %{y:.2f}<extra></extra>",
        )
    )
    # Anomaly markers
    anomalies = spc[spc["anomaly"]]
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["period"],
                y=anomalies["revenue"],
                mode="markers",
                name="Anomaly",
                marker={"color": "red", "size": 10, "symbol": "x"},
                hovertemplate="Anomaly (%{customdata})<br>Revenue: %{y:.2f}<br>Rule: %{customdata}<extra></extra>",
                customdata=anomalies["rule"],
            )
        )
    fig.update_layout(yaxis={"title": "Revenue"}, xaxis={"title": "Period"})
    show(fig)

    st.caption(
        "SPC control limits: trailing rolling mean ± 2σ (Rule 1: outside limits; "
        "Rule 3: 7 consecutive points on same side of center). Red X = anomaly."
    )


def _render_calendar_heatmap(df: pd.DataFrame) -> None:
    st.subheader(":material/calendar_month: Daily Revenue Calendar Heatmap")
    revenue = df["price"] * df["quantity"]
    daily = revenue.groupby(df["date"].dt.date).sum().reset_index()
    daily.columns = ["date", "revenue"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily["week_of_year"] = daily["date"].dt.isocalendar().week
    daily["day_of_week"] = daily["date"].dt.dayofweek  # 0=Mon

    if daily.empty:
        show(empty_state("No revenue data"))
        return

    # Pivot for heatmap: rows=week, cols=day
    pivot = daily.pivot_table(index="week_of_year", columns="day_of_week", values="revenue", fill_value=0)
    # Ensure all 7 days present
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
            hovertemplate="Week %{y}<br>%{x}<br>Revenue: $%{z:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis={"title": "Day of Week"},
        yaxis={"title": "Week of Year", "autorange": "reversed"},
        height=max(300, len(pivot) * 20 + 100),
    )
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

    # Period-over-period deltas (weekly)
    t_curr, t_prev, t_color = _period_over_period_delta(df, "transactions", "weekly")
    c_curr, c_prev, c_color = _period_over_period_delta(df, "customers", "weekly")
    p_curr, p_prev, p_color = _period_over_period_delta(df, "products", "weekly")
    r_curr, r_prev, r_color = _period_over_period_delta(df, "revenue", "weekly")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions", f"{t_curr:,}", delta=f"{t_curr - t_prev:+,}" if t_prev else None, delta_color=t_color)
    col2.metric("Customers", f"{c_curr:,}", delta=f"{c_curr - c_prev:+,}" if c_prev else None, delta_color=c_color)
    col3.metric("Products", f"{p_curr:,}", delta=f"{p_curr - p_prev:+,}" if p_prev else None, delta_color=p_color)
    col4.metric("Revenue", f"${r_curr:,.2f}", delta=f"${r_curr - r_prev:+,.2f}" if r_prev else None, delta_color=r_color)

    st.caption(f"Date range: {summary['date_range']} | Deltas vs prior week")

    st.divider()

    _render_revenue_trend(df)

    st.divider()

    _render_calendar_heatmap(df)

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
