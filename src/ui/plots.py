"""Shared plotly chart helpers for UI tabs."""

from __future__ import annotations

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
