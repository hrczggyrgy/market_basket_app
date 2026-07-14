"""Cohort Analysis Tab."""

import numpy as np
import pandas as pd
import streamlit as st

from src.analytics.cohort import (
    cohort_comparison_summary,
    compute_cohorts,
)
from src.ui.export import render_analytics_export


@st.cache_data
def _cached_compute_cohorts(
    transactions_df: pd.DataFrame, cohort_period: str, metric: str
) -> pd.DataFrame:
    """Cached wrapper for compute_cohorts."""
    return compute_cohorts(transactions_df, cohort_period=cohort_period, metric=metric)


@st.cache_data
def _cached_cohort_comparison_summary(
    transactions_df: pd.DataFrame, cohort_period: str, max_periods: int
) -> dict:
    """Cached wrapper for cohort_comparison_summary."""
    return cohort_comparison_summary(
        transactions_df, cohort_period=cohort_period, max_periods=max_periods
    )


def render_cohort_tab(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render cohort analysis tab."""
    # product_lookup is available but not used in cohort analysis
    # Kept for consistency with other tab functions
    st.header("📊 Cohort Analysis")

    if transactions_df.empty:
        st.warning("No transaction data available")
        return

    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        cohort_period = st.selectbox(
            "Cohort Period",
            ["Weekly", "Monthly", "Quarterly"],
            index=1,
            key="cohort_period",
        )
    with col2:
        metric = st.selectbox(
            "Metric",
            [
                "Retention Rate",
                "Revenue per Customer",
                "Number of Customers",
                "Average Order Value",
            ],
            index=0,
            key="cohort_metric",
        )
    with col3:
        max_periods = st.slider("Max Periods to Show", 3, 24, 12, key="cohort_max_periods")

    # Map metric to internal names
    metric_map = {
        "Retention Rate": "retention",
        "Revenue per Customer": "revenue",
        "Number of Customers": "orders",
        "Average Order Value": "avg_order_value",
    }
    metric_internal = metric_map[metric]

    # Cohort period mapping
    period_map = {"Weekly": "W", "Monthly": "M", "Quarterly": "Q"}
    period_code = period_map[cohort_period]

    # Compute cohorts
    with st.spinner("Computing cohorts..."):
        cohort_matrix = _cached_compute_cohorts(
            transactions_df, cohort_period=period_code, metric=metric_internal
        )

    if cohort_matrix.empty:
        st.warning("Insufficient data for cohort analysis")
        return

    # Limit to max_periods
    period_cols = [c for c in cohort_matrix.columns if c.startswith("Period ")]
    if len(period_cols) > max_periods:
        cohort_matrix = cohort_matrix[period_cols[:max_periods]]

    # Reset index to get Cohort as column
    cohort_data = cohort_matrix.reset_index()
    # Bug 7 fix: safety check for Cohort column name
    if "Cohort" not in cohort_data.columns:
        cohort_data = cohort_data.rename(columns={cohort_data.columns[0]: "Cohort"})
    else:
        cohort_data = cohort_data.rename(columns={"index": "Cohort"})

    # Summary metrics
    st.subheader("Cohort Summary")

    summary = _cached_cohort_comparison_summary(
        transactions_df, cohort_period=period_code, max_periods=max_periods
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Number of Cohorts", summary.get("n_cohorts", 0))
    with col2:
        st.metric(
            f"Avg {cohort_period} 1 Retention",
            f"{summary.get('avg_retention_period_1', 0):.1%}",
        )
    with col3:
        st.metric(
            f"Avg {cohort_period} 3 Retention",
            f"{summary.get('avg_retention_period_3', 0):.1%}",
        )
    with col4:
        st.metric(
            f"Avg {cohort_period} 6 Retention",
            f"{summary.get('avg_retention_period_6', 0):.1%}",
        )

    col5, col6 = st.columns(2)
    with col5:
        best_key = f"best_cohort_{metric_internal}"
        best_val = summary.get(best_key)
        if best_val is None:
            best_val = "N/A (only retention/revenue supported)"
        st.metric(
            f"Best Cohort ({metric})",
            best_val,
        )
    with col6:
        st.metric("Best Cohort Revenue", summary.get("best_cohort_revenue", "N/A"))

    # Cohort heatmap
    st.subheader(f"Cohort {metric} Heatmap")

    if not cohort_matrix.empty:
        import plotly.express as px

        fig = px.imshow(
            cohort_matrix.values,
            x=cohort_matrix.columns,
            y=cohort_matrix.index,
            color_continuous_scale="RdYlGn",
            labels={"x": "Period", "y": "Cohort", "color": metric},
            title=f"Cohort {metric.capitalize()} Heatmap",
            aspect="auto",
        )
        st.plotly_chart(fig, width="stretch")

        # Show raw data
        with st.expander("View Raw Data"):
            st.dataframe(
                cohort_matrix.style.format(
                    "{:.2%}" if metric_internal == "retention" else "${:,.2f}"
                ).background_gradient(cmap="RdYlGn", axis=None),
                width="stretch",
            )

    # Cohort trends over time
    st.subheader("Cohort Trends Over Time")

    # Line chart of each cohort's metric over periods
    if not cohort_matrix.empty:
        # Prepare data for line chart
        line_data = cohort_matrix.reset_index().melt(
            id_vars="index", var_name="Period", value_name=metric
        )
        line_data = line_data.rename(columns={"index": "Cohort"})
        line_data["Period_Num"] = line_data["Period"].str.extract(r"(\d+)").astype(int)

        fig = px.line(
            line_data,
            x="Period_Num",
            y=metric,
            color="Cohort",
            title=f"{metric} by Cohort Over Time",
            labels={"Period_Num": f"{cohort_period} Number", metric: metric},
        )
        st.plotly_chart(fig, width="stretch")

    # Cohort comparison table
    st.subheader("Cohort Comparison")

    display_cols = ["Cohort"] + period_cols[:max_periods]

    st.dataframe(
        cohort_data[display_cols]
        .style.format("{:.2%}" if metric_internal == "retention" else "${:,.2f}")
        .background_gradient(cmap="RdYlGn", axis=None),
        width="stretch",
    )

    render_analytics_export(cohort_data[display_cols], f"Cohort_{metric}_{cohort_period}")

    # Cohort size vs performance scatter - not available from compute_cohorts
    # Skip this section as we don't have cohort sizes in the matrix format

    # Period-over-period comparison
    st.subheader("Period-over-Period Comparison")

    if len(period_cols) >= 2:
        p1, p2 = st.columns(2)
        with p1:
            period_1 = st.selectbox("Period 1", period_cols, index=0, key="pop_period_1")
        with p2:
            period_2 = st.selectbox(
                "Period 2",
                period_cols,
                index=min(1, len(period_cols) - 1),
                key="pop_period_2",
            )

        if period_1 in cohort_data.columns and period_2 in cohort_data.columns:
            # Bug 4 fix: replace 0 with NaN before division to avoid inf/NaN
            period_1_vals = cohort_data[period_1].replace(0, np.nan)
            comparison = pd.DataFrame(
                {
                    "Cohort": cohort_data["Cohort"],
                    period_1: cohort_data[period_1],
                    period_2: cohort_data[period_2],
                    "Change": cohort_data[period_2] - cohort_data[period_1],
                    "Pct Change": (
                        (cohort_data[period_2] - cohort_data[period_1]) / period_1_vals * 100
                    ).round(1),
                }
            )

            st.dataframe(
                comparison.style.format(
                    {
                        period_1: ("{:.2%}" if metric_internal == "retention" else "${:,.2f}"),
                        period_2: ("{:.2%}" if metric_internal == "retention" else "${:,.2f}"),
                        "Change": ("{:.2%}" if metric_internal == "retention" else "${:,.2f}"),
                        "Pct Change": "{:.1f}%",
                    }
                ).background_gradient(cmap="RdYlGn", subset=["Change", "Pct Change"]),
                width="stretch",
            )
