"""Cohort visualization module."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_ltv_power_law(
    ltv_df: pd.DataFrame,
    fit_params: dict,
    height: int = 600,
    width: int = 800,
) -> go.Figure:
    """
    Plot Cohort LTV curves with power-law fits.

    Shows fitted LTV curves overlaid on actual cumulative revenue.
    """
    fig = go.Figure()

    # Plot fitted curves for each cohort
    for cohort_idx in ltv_df.index:
        ltv_curve = ltv_df.loc[cohort_idx].values
        periods = np.arange(1, len(ltv_curve) + 1)

        fig.add_trace(
            go.Scatter(
                x=periods,
                y=ltv_curve,
                mode="lines",
                name=f"Cohort {cohort_idx}",
                line=dict(width=2),
            )
        )

    # Add annotations for fit parameters
    for i, (cohort_idx, params) in enumerate(fit_params.items()):
        if params["r2"] > 0.5:
            fig.add_annotation(
                x=len(periods) * 0.9,
                y=ltv_df.iloc[i].values[-1] if i < len(ltv_df) else 0,
                text=f"{cohort_idx}: a={params['a']:.2f}, b={params['b']:.2f}, R²={params['r2']:.2f}",
                showarrow=False,
                font=dict(size=9),
            )

    fig.update_layout(
        title="Cohort LTV Curves (Power-Law Fit)",
        xaxis_title="Period",
        yaxis_title="Cumulative LTV ($)",
        height=height,
        width=width,
        hovermode="x unified",
    )

    return fig


def plot_decay_rate(
    decay_df: pd.DataFrame,
    height: int = 400,
    width: int = 800,
) -> go.Figure:
    """
    Plot cohort decay rates with exponential fits.
    """
    fig = go.Figure()

    for cohort_idx in decay_df.index:
        _lambda_val = decay_df.loc[cohort_idx, "lambda"]
        _r0 = decay_df.loc[cohort_idx, "r0"]
        r2 = decay_df.loc[cohort_idx, "r2"]

        fitted = decay_df.loc[cohort_idx, "r0"] * np.exp(
            -decay_df.loc[cohort_idx, "lambda"] * np.arange(13)
        )

        fig.add_trace(
            go.Scatter(
                x=np.arange(13),
                y=fitted,
                mode="lines",
                name=f"Cohort {cohort_idx} (λ={decay_df.loc[cohort_idx, 'lambda']:.3f}, R²={r2:.2f})",
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title="Cohort Retention Decay Rates (Exponential Fit)",
        xaxis_title="Period",
        yaxis_title="Retention Rate",
        height=height,
        width=width,
    )

    return fig


def plot_waterfall_decomposition(
    waterfall_df: pd.DataFrame,
    height: int = 500,
    width: int = 800,
) -> go.Figure:
    """
    Plot customer waterfall decomposition.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="New Customers",
            x=waterfall_df["period"],
            y=waterfall_df["new_customers"],
            marker_color="green",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Retained",
            x=waterfall_df["period"],
            y=waterfall_df["retained_customers"],
            marker_color="blue",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Reactivated",
            x=waterfall_df["period"],
            y=waterfall_df["reactivated_customers"],
            marker_color="orange",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Churned",
            x=waterfall_df["period"],
            y=-waterfall_df["churned_customers"],
            marker_color="red",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=waterfall_df["period"],
            y=waterfall_df["net_change"],
            mode="lines+markers",
            name="Net Change",
            line=dict(color="black", width=3, dash="dash"),
        )
    )

    fig.update_layout(
        title="Customer Waterfall Decomposition",
        xaxis_title="Period",
        yaxis_title="Customers",
        barmode="relative",
        height=500,
        width=800,
    )

    return fig


def plot_ltv_power_law_comparison(
    ltv_df: pd.DataFrame,
    fit_params: dict,
    actual_revenue: pd.DataFrame,
    height: int = 600,
    width: int = 900,
) -> go.Figure:
    """
    Compare fitted LTV power-law curves with actual data.
    """
    fig = go.Figure()

    for i, cohort_idx in enumerate(ltv_df.index):
        # Actual data
        actual = ltv_df.loc[cohort_idx].values

        if len(actual) == 0:
            continue

        periods = np.arange(1, len(actual) + 1)

        fig.add_trace(
            go.Scatter(
                x=periods,
                y=actual,
                mode="markers",
                name=f"Cohort {cohort_idx} (Actual)",
                marker=dict(size=8),
                showlegend=True,
            )
        )

        # Fitted curve
        if cohort_idx in fit_params:
            params = fit_params[cohort_idx]
            a, b = params["a"], params["b"]
            t = np.linspace(1, len(actual), 50)
            fitted = a * t**b

            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=fitted,
                    mode="lines",
                    name=f"Cohort {cohort_idx} (Fit: a={fit_params[cohort_idx]['a']:.2f}, b={fit_params[cohort_idx]['b']:.2f})",
                    line=dict(dash="dash"),
                    showlegend=True,
                )
            )

    fig.update_layout(
        title="Cohort LTV: Actual vs Power-Law Fit",
        xaxis_title="Period",
        yaxis_title="Cumulative Revenue per Customer ($)",
        height=600,
        width=900,
    )

    return fig
