"""Heatmap visualizations for market basket analysis."""

from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_heatmap(
    rules_df: pd.DataFrame,
    x_metric: str = "support",
    y_metric: str = "confidence",
    color_metric: str = "lift",
    title: str = "Association Rules Heatmap",
    height: int = 600,
) -> go.Figure:
    """
    Create 2D density heatmap of rules.

    X-axis: support
    Y-axis: confidence
    Color: lift (or other metric)
    """
    if rules_df.empty:
        return _empty_figure("No rules to display")

    # Sample if too many rules
    if len(rules_df) > 10000:
        rules = rules_df.sample(10000, random_state=42)
    else:
        rules = rules_df.copy()

    fig = px.density_heatmap(
        rules,
        x=x_metric,
        y=y_metric,
        z=color_metric,
        nbinsx=30,
        nbinsy=30,
        color_continuous_scale="Viridis",
        title=title,
        height=height,
    )

    fig.update_layout(
        xaxis_title=x_metric.capitalize(),
        yaxis_title=y_metric.capitalize(),
        coloraxis_colorbar=dict(title=color_metric.capitalize()),
    )

    return fig


def create_scatter_heatmap(
    rules_df: pd.DataFrame,
    x_metric: str = "support",
    y_metric: str = "confidence",
    color_metric: str = "lift",
    size_metric: str = "support",
    hover_data: List[str] = None,
    title: str = "Rules Scatter Plot",
    height: int = 600,
) -> go.Figure:
    """
    Create scatter plot of rules with color and size encoding.
    """
    if rules_df.empty:
        return _empty_figure("No rules to display")

    if hover_data is None:
        hover_data = ["antecedents", "consequents", "lift", "leverage", "conviction"]

    # Limit points for performance
    if len(rules_df) > 5000:
        rules = rules_df.nlargest(5000, "lift")
    else:
        rules = rules_df.copy()

    # Format antecedents/consequents for hover
    rules["ant_str"] = rules["antecedents"].apply(
        lambda x: ", ".join(str(i) for i in x)
    )
    rules["cons_str"] = rules["consequents"].apply(
        lambda x: ", ".join(str(i) for i in x)
    )

    fig = px.scatter(
        rules,
        x=x_metric,
        y=y_metric,
        color=color_metric,
        size=size_metric,
        hover_data={
            "ant_str": True,
            "cons_str": True,
            x_metric: ":.4f",
            y_metric: ":.4f",
            color_metric: ":.4f",
            "leverage": ":.4f",
            "conviction": ":.4f",
        },
        color_continuous_scale="Viridis",
        size_max=20,
        title=title,
        height=height,
    )

    fig.update_layout(
        xaxis_title=x_metric.capitalize(), yaxis_title=y_metric.capitalize()
    )

    return fig


def create_affinity_heatmap(
    affinity_matrix: pd.DataFrame,
    product_lookup: dict = None,
    min_lift: float = 1.0,
    max_products: int = 30,
    title: str = "Product Affinity Heatmap",
    height: int = 700,
) -> go.Figure:
    """
    Create heatmap of product affinity matrix (lift values).
    """
    if affinity_matrix.empty:
        return _empty_figure("No affinity data")

    # Select top products by total affinity
    total_affinity = affinity_matrix.sum(axis=1)
    top_products = total_affinity.nlargest(max_products).index.tolist()

    matrix = affinity_matrix.loc[top_products, top_products].copy()

    # Apply min lift filter for display
    matrix_display = matrix.copy()
    matrix_display[matrix_display < min_lift] = 1.0

    # Format labels
    if product_lookup:
        labels = [product_lookup.get(p, p)[:30] for p in top_products]
    else:
        labels = [str(p)[:30] for p in top_products]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_display.values,
            x=labels,
            y=labels,
            colorscale="RdYlBu_r",
            zmid=1.0,
            zmin=1.0,
            zmax=(
                matrix_display.values.max() if matrix_display.values.max() > 1 else 2.0
            ),
            colorbar=dict(title="Lift"),
            hoverongaps=False,
            hovertemplate="Product A: %{y}<br>Product B: %{x}<br>Lift: %{z:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(tickangle=45, side="bottom"),
        yaxis=dict(autorange="reversed"),
        height=height,
        width=height,
    )

    return fig


def create_metric_comparison_heatmap(
    rules_df: pd.DataFrame,
    metrics: List[str] = None,
    title: str = "Rules Metric Comparison",
    height: int = 500,
) -> go.Figure:
    """
    Create heatmap comparing multiple metrics across rules.
    Rows = rules, Columns = metrics
    """
    if rules_df.empty:
        return _empty_figure("No rules to display")

    if metrics is None:
        metrics = [
            "support",
            "confidence",
            "lift",
            "leverage",
            "conviction",
            "zhangs_metric",
        ]

    # Filter available metrics
    available = [m for m in metrics if m in rules_df.columns]
    if not available:
        return _empty_figure("No matching metrics found")

    # Take top rules by lift
    rules = (
        rules_df.nlargest(100, "lift")
        if "lift" in rules_df.columns
        else rules_df.head(100)
    )

    # Normalize metrics for comparison
    data = rules[available].copy()
    for col in data.columns:
        col_max = data[col].max()
        col_min = data[col].min()
        if col_max > col_min:
            data[col] = (data[col] - col_min) / (col_max - col_min)

    # Format rule labels
    rule_labels = rules.apply(
        lambda r: (
            f"{', '.join(map(str, r['antecedents']))} → {', '.join(map(str, r['consequents']))}"
        ),
        axis=1,
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=data.values,
            x=available,
            y=rule_labels,
            colorscale="RdYlBu_r",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Normalized Value"),
            hoverongaps=False,
            hovertemplate="Rule: %{y}<br>Metric: %{x}<br>Value: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
        height=height,
    )

    return fig


def create_support_confidence_lift_3d(
    rules_df: pd.DataFrame, title: str = "Support-Confidence-Lift 3D"
) -> go.Figure:
    """Create 3D scatter plot of support, confidence, lift."""
    if rules_df.empty:
        return _empty_figure("No rules to display")

    rules = rules_df.nlargest(500, "lift") if len(rules_df) > 500 else rules_df

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=rules["support"],
                y=rules["confidence"],
                z=rules["lift"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=rules["lift"],
                    colorscale="Viridis",
                    opacity=0.8,
                    colorbar=dict(title="Lift"),
                ),
                text=[
                    f"A: {', '.join(map(str, a))}<br>C: {', '.join(map(str, c))}"
                    for a, c in zip(rules["antecedents"], rules["consequents"])
                ],
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(xaxis_title="Support", yaxis_title="Confidence", zaxis_title="Lift"),
        height=600,
    )

    return fig


def _empty_figure(message: str) -> go.Figure:
    """Create empty figure with message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
    )
    return fig
