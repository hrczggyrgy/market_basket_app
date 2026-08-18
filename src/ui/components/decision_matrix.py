"""Decision Matrix Component.

A heatmap-style matrix with numeric backing for proper color scaling
and text annotations for actions. Supports 2D strategic matrices.

Decision Confidence System:
- Bubble position  = analytical result value (x/y axis)
- Bubble size      = economic impact (revenue units)
- Bubble opacity   = evidence confidence (HIGH=opaque, MEDIUM=semi-transparent, LOW=faint)
- Evidence class   = OBSERVED/ESTIMATED/CAUSAL (integrated with evidence badge convention)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui.components.strategic_table import EvidenceClass
from src.ui.plots import PALETTE, new_fig, show


@dataclass
class MatrixConfig:
    """Configuration for a decision matrix."""

    x_axis: str  # Column name for X axis
    y_axis: str  # Column name for Y axis
    value: str  # Column name for cell values (numeric)
    text: str  # Column name for cell text (annotations)
    x_labels: dict  # Mapping from x values to labels
    y_labels: dict  # Mapping from y values to labels
    x_order: list  # Order of x categories
    y_order: list  # Order of y categories
    colorscale: Optional[list] = None  # Custom colorscale
    colorbar_title: str = "Value"
    height: int = 400
    size_col: Optional[str] = None  # Optional bubble size column (economic impact/revenue)
    color_col: Optional[str] = None  # Optional color category column
    color_map: Optional[dict] = None  # Mapping for color_col
    evidence_col: Optional[str] = None  # Column for evidence class (OBSERVED/ESTIMATED/CAUSAL)
    confidence_col: Optional[str] = None  # Column for confidence level (HIGH/MEDIUM/LOW)


def _confidence_opacity(confidence: str) -> float:
    """Map confidence level to bubble opacity alpha.

    Uses the same scale as strategic_table._get_confidence_rgba():
    HIGH   -> 1.0 (opaque)
    MEDIUM -> 0.6 (semi-transparent)
    LOW    -> 0.3 (faint/transparent)

    Args:
        confidence: Confidence level string ("high", "medium", "low", or abbreviations "h", "m", "l")

    Returns:
        Alpha value in [0.0, 1.0] for Plotly marker opacity
    """
    conf_str = str(confidence).lower().strip()
    if conf_str in ("high", "h"):
        return 1.0
    elif conf_str in ("medium", "m"):
        return 0.6
    elif conf_str in ("low", "l"):
        return 0.3
    else:
        return 0.6


def _apply_confidence_mixin(
    fig: go.Figure,
    df: pd.DataFrame,
    config: MatrixConfig,
    opacity_col: str,
    default_opacity: float = 0.6,
    customdata_col_names: Optional[list[str]] = None,
) -> go.Figure:
    """Apply confidence-based opacity to matrix figure traces.

    Modifies marker opacity on Scatter traces based on a confidence column.
    If opacity_col is set and present in df, each trace's markers receive
    per-point opacity; otherwise all markers use default_opacity.

    Args:
        fig: Plotly Figure to modify
        df: DataFrame with matrix data
        config: MatrixConfig with column mappings
        opacity_col: Column name containing confidence levels
        default_opacity: Fallback opacity if opacity_col is unavailable
        customdata_col_names: Optional list of customdata column names
            (e.g., hover_cols) for index-based lookup

    Returns:
        Modified Plotly Figure with confidence-based opacity applied
    """
    if not opacity_col or opacity_col not in df.columns:
        for trace in fig.data:
            if hasattr(trace, "marker") and trace.marker:
                trace.marker.opacity = default_opacity
        return fig

    for trace in fig.data:
        if not hasattr(trace, "marker") or not trace.marker:
            continue

        customdata = trace.customdata
        if customdata is None:
            trace.marker.opacity = default_opacity
            continue

        # Find the column index of opacity_col within customdata
        col_idx: Optional[int] = None
        if customdata_col_names:
            try:
                col_idx = customdata_col_names.index(opacity_col)
            except ValueError:
                col_idx = None
        if col_idx is None:
            try:
                col_idx = list(df.columns).index(opacity_col)
            except ValueError:
                col_idx = -1

        if col_idx is not None and 0 <= col_idx < customdata.shape[1]:
            conf_values = customdata[:, col_idx]
            opacities: list[float] = []
            for cv in conf_values:
                try:
                    cv_str = str(cv).lower().strip()
                    opacities.append(_confidence_opacity(cv_str))
                except Exception:
                    opacities.append(default_opacity)
            trace.marker.opacity = opacities  # type: ignore[attr-mis]
        else:
            trace.marker.opacity = default_opacity

    return fig


def _render_with_confidence(
    fig: go.Figure,
    df: pd.DataFrame,
    config: MatrixConfig,
) -> go.Figure:
    """Render-time confidence system mixin.

    Applies confidence-based opacity and evidence class color mapping
    to a matrix figure. Integrates with the evidence badge convention
    from strategic_table.py.

    Args:
        fig: Plotly Figure from the matrix renderer
        df: DataFrame with matrix data
        config: MatrixConfig with column mappings

    Returns:
        Modified Plotly Figure with confidence system applied
    """
    # Apply confidence-based opacity
    if config.confidence_col and config.confidence_col in df.columns:
        fig = _apply_confidence_mixin(fig, df, config, config.confidence_col)
    else:
        for trace in fig.data:
            if hasattr(trace, "marker") and trace.marker:
                trace.marker.opacity = 0.6

    # Apply evidence class color mapping if evidence_col is set
    if config.evidence_col and config.evidence_col in df.columns:
        evidence_color_map = {
            EvidenceClass.OBSERVED: "#59A14F",
            EvidenceClass.ESTIMATED: "#F28E2B",
            EvidenceClass.CAUSAL: "#E15759",
        }
        for trace in fig.data:
            if not hasattr(trace, "marker") or not trace.marker:
                continue
            if hasattr(trace, "customdata") and trace.customdata is not None:
                ev_colors: list[str] = []
                for i in range(min(trace.customdata.shape[0], len(df))):
                    try:
                        ev_val = str(trace.customdata[i]).strip()
                        ev_colors.append(evidence_color_map.get(ev_val, PALETTE[0]))
                    except Exception:
                        ev_colors.append(PALETTE[0])
                trace.marker.color = ev_colors  # type: ignore[attr-mis]

    return fig


def render_decision_matrix(
    df: pd.DataFrame,
    config: MatrixConfig,
    key: str = "matrix",
) -> None:
    """Render a decision matrix as a Plotly heatmap.

    Args:
        df: DataFrame with matrix data
        config: MatrixConfig with axis/value/text mappings
        key: Unique key for the component
    """
    if df.empty:
        st.info("No data for matrix")
        return

    # Create pivot tables
    pivot_value = df.pivot_table(
        index=config.y_axis,
        columns=config.x_axis,
        values=config.value,
        aggfunc="first",
        fill_value=0,
    )
    pivot_text = df.pivot_table(
        index=config.y_axis,
        columns=config.x_axis,
        values=config.text,
        aggfunc="first",
        fill_value="",
    )

    # Reorder according to specified order
    y_order = [y for y in config.y_order if y in pivot_value.index]
    x_order = [x for x in config.x_order if x in pivot_value.columns]
    pivot_value = pivot_value.reindex(index=y_order, columns=x_order, fill_value=0)
    pivot_text = pivot_text.reindex(index=y_order, columns=x_order, fill_value="")

    # Build hover text with additional info
    hover_text = []
    for y in y_order:
        row = []
        for x in x_order:
            row_text = (
                str(pivot_text.loc[y, x])
                if y in pivot_text.index and x in pivot_text.columns
                else ""
            )
            row_val = (
                pivot_value.loc[y, x] if y in pivot_value.index and x in pivot_value.columns else 0
            )
            row.append(
                f"{config.y_labels.get(y, y)} / {config.x_labels.get(x, x)}<br>Value: {row_val:,.0f}<br>{row_text}"
            )
        hover_text.append(row)

    # Colorscale
    if config.colorscale is None:
        colorscale = [
            [0.0, "#E15759"],
            [0.35, "#F28E2B"],
            [0.7, "#4E79A7"],
            [1.0, "#59A14F"],
        ]
    else:
        colorscale = config.colorscale

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_value.values,
            x=[config.x_labels.get(x, x) for x in x_order],
            y=[config.y_labels.get(y, y) for y in y_order],
            text=pivot_text.values,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorscale=colorscale,
            zmid=0,
            colorbar={"title": config.colorbar_title, "thickness": 15, "len": 0.75},
            hoverongaps=False,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_text,
            showscale=True,
        )
    )

    fig.update_layout(
        height=config.height,
        margin={"l": 100, "r": 50, "t": 50, "b": 100},
        xaxis={"side": "top", "tickangle": -45},
        yaxis={"autorange": "reversed"},
    )

    show(fig)


def render_bubble_matrix(
    df: pd.DataFrame,
    x: str,
    y: str,
    size: str,
    color: str,
    text: str,
    hover_cols: list[str],
    color_map: Optional[dict] = None,
    x_title: str = "",
    y_title: str = "",
    title: str = "",
    height: int = 450,
    key: str = "bubble_matrix",
    size_max: int = 50,
    add_quadrant_lines: bool = False,
    x_ref: Optional[float] = None,
    y_ref: Optional[float] = None,
    config: Optional[MatrixConfig] = None,
) -> None:
    """Render a bubble scatter matrix (like BCG matrix).

    Args:
        df: DataFrame with data
        x: X-axis column
        y: Y-axis column
        size: Bubble size column (economic impact / revenue units)
        color: Color category column
        text: Text label column
        hover_cols: Additional columns for hover
        color_map: Color mapping for categories
        x_title: X-axis title
        y_title: Y-axis title
        title: Chart title
        height: Figure height
        key: Unique key
        size_max: Maximum bubble size
        add_quadrant_lines: Add median reference lines
        x_ref: X reference line value
        y_ref: Y reference line value
        config: Optional MatrixConfig for confidence/evidence integration
    """
    if df.empty:
        st.info("No data for matrix")
        return

    fig = new_fig(height=height)

    # Determine opacity base value
    base_opacity = 0.8
    if config and config.confidence_col:
        # Per-point opacity will be applied by the mixin after trace creation
        base_opacity = 0.8

    if color_map and color in df.columns:
        for cat, cat_df in df.groupby(color):
            cat_color = color_map.get(cat, PALETTE[0])
            fig.add_trace(
                go.Scatter(
                    x=cat_df[x],
                    y=cat_df[y],
                    mode="markers+text",
                    marker={
                        "size": cat_df[size],
                        "sizemode": "area",
                        "sizeref": 2.0 * max(cat_df[size]) / (size_max**2),
                        "sizemin": 4,
                        "color": cat_color,
                        "opacity": base_opacity,
                        "line": {"width": 1, "color": "white"},
                    },
                    text=cat_df[text],
                    textposition="top center",
                    textfont={"size": 9},
                    name=str(cat),
                    customdata=cat_df[hover_cols].values if hover_cols else None,
                    hovertemplate="<br>".join(
                        [
                            "<b>%{text}</b>",
                            f"{x_title}: %{{x:.2f}}",
                            f"{y_title}: %{{y:.2f}}",
                            *[f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(hover_cols)],
                        ]
                    ),
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=df[x],
                y=df[y],
                mode="markers+text",
                marker={
                    "size": df[size],
                    "sizemode": "area",
                    "sizeref": 2.0 * max(df[size]) / (size_max**2),
                    "sizemin": 4,
                    "color": PALETTE[0],
                    "opacity": base_opacity,
                    "line": {"width": 1, "color": "white"},
                },
                text=df[text],
                textposition="top center",
                textfont={"size": 9},
                customdata=df[hover_cols].values if hover_cols else None,
                hovertemplate="<br>".join(
                    [
                        "<b>%{text}</b>",
                        f"{x_title}: %{{x:.2f}}",
                        f"{y_title}: %{{y:.2f}}",
                        *[f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(hover_cols)],
                    ]
                ),
            )
        )

    # Apply confidence mixin after all traces are created
    if config:
        fig = _apply_confidence_mixin(
            fig, df, config, config.confidence_col,
            customdata_col_names=hover_cols,
        )

    if add_quadrant_lines:
        if x_ref is not None:
            fig.add_vline(x=x_ref, line_dash="dash", line_color="#888888", line_width=1)
        if y_ref is not None:
            fig.add_hline(y=y_ref, line_dash="dash", line_color="#888888", line_width=1)

    fig.update_layout(
        title=title,
        xaxis={"title": x_title, "tickformat": ".0%"},
        yaxis={"title": y_title, "tickformat": ".1f"},
        margin={"l": 80, "r": 30, "t": 50, "b": 60},
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )

    show(fig)