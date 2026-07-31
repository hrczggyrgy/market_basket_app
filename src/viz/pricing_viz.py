"""Pricing & Elasticity visualization module."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_cross_elasticity_heatmap(
    cross_elasticity_df: pd.DataFrame,
    product_lookup: dict,
    height: int = 700,
    width: int = 900,
) -> go.Figure:
    """
    Plot cross-price elasticity matrix as heatmap.

    Shows cross-price elasticity ε_ij where rows = product A (affected),
    columns = product B (price change driver).
    """
    if cross_elasticity_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No cross-elasticity data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        return fig

    # Pivot to matrix
    pivot = cross_elasticity_df.pivot(
        index="product_a", columns="product_b", values="cross_elasticity"
    )

    # Use product names for labels
    products = pivot.index.tolist()
    labels = [f"{p} ({product_lookup.get(p, p)[:20]})" for p in products]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=labels,
            y=labels,
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(pivot.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Cross Elasticity (εᵢⱼ)"),
        )
    )

    fig.update_layout(
        title="Cross-Price Elasticity Matrix (εᵢⱼ)",
        xaxis_title="Product B (Price Driver)",
        yaxis_title="Product A (Quantity Respondent)",
        height=700,
        width=900,
    )

    return fig


def plot_price_volume_scatter(
    elasticity_df: pd.DataFrame,
    product_lookup: dict,
    height: int = 600,
    width: int = 800,
) -> go.Figure:
    """
    Plot price-volume scatter with elasticity iso-curves.
    """
    fig = go.Figure()

    # Color by elasticity interpretation
    def get_color(e):
        if e < -1:
            return "green"  # Elastic
        elif e < -0.1:
            return "yellow"  # Inelastic
        elif abs(e) <= 0.1:
            return "gray"  # Unit elastic
        else:
            return "red"  # Positive (perverse)

    colors = elasticity_df["elasticity"].apply(get_color)

    fig.add_trace(
        go.Scatter(
            x=elasticity_df["avg_price"],
            y=elasticity_df["avg_weekly_qty"],
            mode="markers",
            marker=dict(
                size=10,
                color=colors,
                line=dict(width=1, color="white"),
                showscale=False,
            ),
            text=[
                f"{product_lookup.get(row['stockcode'], row['stockcode'])}<br>"
                f"Elasticity: {row['elasticity']:.3f}<br>"
                f"R²: {row['r_squared']:.3f}"
                for _, row in elasticity_df.iterrows()
            ],
            hoverinfo="text",
        )
    )

    # Add iso-elasticity curves
    x_range = np.logspace(
        np.log10(elasticity_df["avg_price"].min() * 0.5),
        np.log10(elasticity_df["avg_price"].max() * 2),
        50,
    )

    for elasticity_val in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]:
        a = (
            elasticity_df["avg_weekly_qty"]
            / elasticity_df["avg_price"] ** elasticity_df["elasticity"]
        ).median()
        y_vals = a * x_range**elasticity_val

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_vals,
                mode="lines",
                line=dict(dash="dash", width=1, color="gray"),
                name=f"ε = {elasticity_val}",
                showlegend=elasticity_val == -1.0,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Price-Volume Scatter with Elasticity Iso-Curves",
        xaxis_title="Average Price ($)",
        yaxis_title="Average Weekly Quantity",
        xaxis_type="log",
        yaxis_type="log",
        height=600,
        width=800,
    )

    return fig


def plot_revenue_waterfall(
    price_effect: float,
    volume_effect: float,
    mix_effect: float,
    total_change: float,
    height: int = 500,
    width: int = 800,
) -> go.Figure:
    """
    Revenue decomposition waterfall chart.

    Shows revenue change decomposed into:
    - Price effect (price change × base volume)
    - Volume effect (quantity change × base price)
    - Mix effect (interaction)
    """
    fig = go.Figure(
        go.Waterfall(
            name="Revenue Change",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Base Revenue", "Price Effect", "Volume Effect", "Mix Effect", "New Revenue"],
            textposition="outside",
            y=[0, price_effect, volume_effect, mix_effect, total_change],
            text=[f"${v:,.0f}" for v in [0, price_effect, volume_effect, mix_effect, total_change]],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "green"}},
            decreasing={"marker": {"color": "red"}},
            totals={"marker": {"color": "blue"}},
        )
    )

    fig.update_layout(
        title="Revenue Decomposition Waterfall",
        yaxis_title="Revenue ($)",
        height=500,
        width=800,
    )

    return fig


def plot_elasticity_distribution(
    elasticity_df: pd.DataFrame,
    height: int = 400,
    width: int = 700,
) -> go.Figure:
    """Plot elasticity distribution histogram."""
    fig = px.histogram(
        elasticity_df,
        x="elasticity",
        nbins=30,
        title="Elasticity Distribution",
        labels={"elasticity": "Elasticity (β)", "count": "Number of Products"},
        color_discrete_sequence=["#1f77b4"],
    )

    # Add reference lines
    for x, label, color in [
        (-1, "Elastic (β < -1)", "green"),
        (-0.1, "Inelastic (-1 < β < -0.1)", "yellow"),
        (0, "Unit Elastic", "gray"),
        (0.1, "Positive β", "red"),
    ]:
        fig.add_vline(
            x=x,
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_position="top",
        )

    fig.update_layout(height=400, width=700)
    return fig
