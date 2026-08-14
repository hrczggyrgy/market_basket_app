"""Shared plotly chart helpers for UI tabs."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PALETTE: tuple[str, ...] = (
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#76B7B2",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
)


def new_fig(height: int = 380) -> go.Figure:
    """Create a consistently styled empty plotly figure."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 40, "r": 20, "t": 45, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return fig


def show(fig: go.Figure) -> None:
    """Render a plotly figure full-width."""
    st.plotly_chart(fig, width="stretch")


def empty_state(title: str, detail: str | None = None) -> go.Figure:
    """Figure rendering a centered 'no data' message."""
    fig = go.Figure()
    fig.add_annotation(
        text=title,
        x=0.5,
        y=0.6,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15},
    )
    if detail:
        fig.add_annotation(
            text=detail,
            x=0.5,
            y=0.4,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 12, "color": "#888888"},
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        template="plotly_white",
        height=200,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    return fig


def render_bar_with_ci(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ci_lower_col: str,
    ci_upper_col: str,
    title: str = "",
    x_title: str = "",
    y_title: str = "",
    color: str | None = None,
    color_discrete_map: dict | None = None,
    hover_data: list | None = None,
    height: int = 380,
) -> go.Figure:
    """
    Render a bar chart with error bars (confidence intervals).

    Args:
        df: DataFrame with columns for x, y, ci_lower, ci_upper
        x_col: Column for x-axis (categories)
        y_col: Column for y-axis values (point estimates)
        ci_lower_col: Column for lower CI bound
        ci_upper_col: Column for upper CI bound
        title: Chart title
        x_title: X-axis title
        y_title: Y-axis title
        color: Column name for bar colors
        color_discrete_map: Mapping from color values to palette colors
        hover_data: Additional columns to show on hover
        height: Figure height

    Returns:
        Plotly Figure object
    """
    fig = new_fig(height=height)

    if df.empty:
        return empty_state("No data available")

    # Sort by y value for better readability
    df_sorted = df.sort_values(y_col, ascending=True).reset_index(drop=True)

    # Create error arrays (Plotly expects arrays of [lower_error, upper_error])
    # For asymmetric error bars, we pass dict with 'minus' and 'plus'
    error_y = dict(
        type="data",
        symmetric=False,
        array=(df_sorted[ci_upper_col] - df_sorted[y_col]).clip(lower=0).values,
        arrayminus=(df_sorted[y_col] - df_sorted[ci_lower_col]).clip(lower=0).values,
        visible=True,
        thickness=1.5,
        width=3,
        color="#333333",
    )

    if color and color in df_sorted.columns:
        marker_color = (
            df_sorted[color].map(color_discrete_map) if color_discrete_map else PALETTE[0]
        )
    else:
        marker_color = PALETTE[0]

    hover_template = "%{x}: %{y:.2f}"
    if hover_data:
        for h in hover_data:
            if h in df_sorted.columns:
                hover_template += f"<br>{h}: %{{customdata[{hover_data.index(h)}]}}"

    fig.add_trace(
        go.Bar(
            x=df_sorted[x_col],
            y=df_sorted[y_col],
            error_y=error_y,
            marker_color=marker_color,
            hovertemplate=hover_template + "<extra></extra>",
            customdata=df_sorted[hover_data].values if hover_data else None,
            name=y_col,
        )
    )

    fig.update_layout(
        title=title,
        xaxis={"title": x_title, "tickangle": -45},
        yaxis={"title": y_title},
        hovermode="x unified",
    )
    return fig


def render_line_with_ci(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ci_lower_col: str,
    ci_upper_col: str,
    title: str = "",
    x_title: str = "",
    y_title: str = "",
    color: str | None = None,
    color_discrete_map: dict | None = None,
    height: int = 380,
) -> go.Figure:
    """
    Render a line chart with confidence interval band.

    Args:
        df: DataFrame with columns for x, y, ci_lower, ci_upper
        x_col: Column for x-axis
        y_col: Column for y-axis values (point estimates)
        ci_lower_col: Column for lower CI bound
        ci_upper_col: Column for upper CI bound
        title: Chart title
        x_title: X-axis title
        y_title: Y-axis title
        color: Column name for line colors
        color_discrete_map: Mapping from color values to palette colors
        height: Figure height

    Returns:
        Plotly Figure object
    """
    fig = new_fig(height=height)

    if df.empty:
        return empty_state("No data available")

    df_sorted = df.sort_values(x_col).reset_index(drop=True)

    if color and color in df_sorted.columns:
        groups = df_sorted.groupby(color)
        for i, (name, group) in enumerate(groups):
            line_color = (
                color_discrete_map.get(name, PALETTE[i % len(PALETTE)])
                if color_discrete_map
                else PALETTE[i % len(PALETTE)]
            )
            group = group.sort_values(x_col)

            # CI band
            fig.add_trace(
                go.Scatter(
                    x=group[x_col].tolist() + group[x_col].tolist()[::-1],
                    y=group[ci_upper_col].tolist() + group[ci_lower_col].tolist()[::-1],
                    fill="toself",
                    fillcolor=f"rgba({_hex_to_rgb(line_color)}, 0.2)",
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{name} CI",
                )
            )
            # Main line
            fig.add_trace(
                go.Scatter(
                    x=group[x_col],
                    y=group[y_col],
                    mode="lines+markers",
                    line={"color": line_color, "width": 2},
                    marker={"size": 4},
                    name=str(name),
                    hovertemplate=f"{name}: %{{x}}: %{{y:.2f}}<extra></extra>",
                )
            )
    else:
        # Single line with CI band
        line_color = PALETTE[0]
        fig.add_trace(
            go.Scatter(
                x=df_sorted[x_col].tolist() + df_sorted[x_col].tolist()[::-1],
                y=df_sorted[ci_upper_col].tolist() + df_sorted[ci_lower_col].tolist()[::-1],
                fill="toself",
                fillcolor=f"rgba({_hex_to_rgb(line_color)}, 0.2)",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                name="CI",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_sorted[x_col],
                y=df_sorted[y_col],
                mode="lines+markers",
                line={"color": line_color, "width": 2},
                marker={"size": 4},
                name=y_col,
                hovertemplate="%{{x}}: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis={"title": x_title},
        yaxis={"title": y_title},
        hovermode="x unified",
    )
    return fig


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to 'r,g,b' string for rgba usage."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"{r},{g},{b}"
