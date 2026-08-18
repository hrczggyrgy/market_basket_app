"""Reliability Badge Component.

Displays reliability level (HIGH/MEDIUM/LOW/INSUFFICIENT) with tooltip
showing dimension scores. Integrates with src.analytics.reliability.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.analytics.reliability import ReliabilityLevel, ReliabilityScore

# Color mapping for reliability levels
RELIABILITY_COLORS = {
    ReliabilityLevel.HIGH: "#59A14F",
    ReliabilityLevel.MEDIUM: "#F28E2B",
    ReliabilityLevel.LOW: "#E15759",
    ReliabilityLevel.INSUFFICIENT: "#7F7F7F",
}

RELIABILITY_ICONS = {
    ReliabilityLevel.HIGH: "✅",
    ReliabilityLevel.MEDIUM: "⚠️",
    ReliabilityLevel.LOW: "🔴",
    ReliabilityLevel.INSUFFICIENT: "❓",
}

DIMENSION_LABELS = {
    "sample_size": "Sample Size",
    "coverage": "Coverage",
    "uncertainty": "Uncertainty",
    "stability": "Stability",
    "assumptions": "Assumptions",
    "data_quality": "Data Quality",
}


def render_reliability_badge(
    reliability: ReliabilityScore | dict[str, Any] | ReliabilityLevel,
    show_score: bool = True,
    show_details: bool = True,
    compact: bool = False,
) -> None:
    """Render a reliability badge with optional tooltip.

    Args:
        reliability: ReliabilityScore object, dict, or ReliabilityLevel enum
        show_score: Show overall score (0-1)
        show_details: Show dimension breakdown on hover
        compact: Use compact inline style
    """
    # Parse input
    if isinstance(reliability, ReliabilityLevel):
        level = reliability
        score = (
            1.0
            if level == ReliabilityLevel.HIGH
            else (
                0.5
                if level == ReliabilityLevel.MEDIUM
                else 0.25
                if level == ReliabilityLevel.LOW
                else 0.0
            )
        )
        dims = {}
        flags = []
    elif isinstance(reliability, ReliabilityScore):
        level = reliability.level
        score = reliability.overall_score
        dims = {k.value: v for k, v in reliability.dimension_scores.items()}
        flags = reliability.flags
    else:
        # Dict input
        level = ReliabilityLevel(reliability.get("level", "insufficient"))
        score = reliability.get("overall_score", 0.0)
        dims = reliability.get("dimension_scores", {})
        flags = reliability.get("flags", [])

    color = RELIABILITY_COLORS[level]
    icon = RELIABILITY_ICONS[level]

    if compact:
        # Inline badge
        badge_html = f"""
        <span style="
            background-color: {color};
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        ">
            {icon} {level.value.upper()}
            {f"· {score:.0%}" if show_score else ""}
        </span>
        """
        st.markdown(badge_html, unsafe_allow_html=True)
    else:
        # Full badge with tooltip
        tooltip_parts = []
        if show_score:
            tooltip_parts.append(f"<b>Overall Score:</b> {score:.1%}")
        if show_details and dims:
            tooltip_parts.append("<br><b>Dimensions:</b>")
            for dim, val in dims.items():
                label = DIMENSION_LABELS.get(dim, dim.replace("_", " ").title())
                tooltip_parts.append(f"<br>&nbsp;&nbsp;{label}: {val:.0%}")
        if flags:
            tooltip_parts.append("<br><b>Flags:</b>")
            for flag in flags:
                tooltip_parts.append(f"<br>&nbsp;&nbsp;• {flag}")

        tooltip = "".join(tooltip_parts) if tooltip_parts else "No details available"

        badge_html = f'''
        <div style="display: inline-flex; align-items: center; gap: 8px;">
            <span style="
                background-color: {color};
                color: white;
                padding: 4px 12px;
                border-radius: 16px;
                font-size: 12px;
                font-weight: 600;
                display: inline-flex;
                align-items: center;
                gap: 6px;
            " title="{tooltip.replace('"', '"')}">
                {icon} {level.value.upper()}
                {f"· {score:.0%}" if show_score else ""}
            </span>
        </div>
        '''
        st.markdown(badge_html, unsafe_allow_html=True)


def render_reliability_inline(
    reliability: ReliabilityScore | dict[str, Any] | ReliabilityLevel,
) -> str:
    """Return HTML string for inline reliability badge (for use in tables).

    Args:
        reliability: ReliabilityScore, dict, or ReliabilityLevel

    Returns:
        HTML string
    """
    if isinstance(reliability, ReliabilityLevel):
        level = reliability
        score = (
            1.0
            if level == ReliabilityLevel.HIGH
            else (
                0.5
                if level == ReliabilityLevel.MEDIUM
                else 0.25
                if level == ReliabilityLevel.LOW
                else 0.0
            )
        )
    elif isinstance(reliability, ReliabilityScore):
        level = reliability.level
        score = reliability.overall_score
    else:
        level = ReliabilityLevel(reliability.get("level", "insufficient"))
        score = reliability.get("overall_score", 0.0)

    color = RELIABILITY_COLORS[level]
    icon = RELIABILITY_ICONS[level]

    return f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 600; display: inline-flex; align-items: center; gap: 4px;">{icon} {level.value.upper()} {score:.0%}</span>'


def render_reliability_legend() -> None:
    """Render a legend showing all reliability levels."""
    cols = st.columns(4)
    for i, (level, color) in enumerate(RELIABILITY_COLORS.items()):
        with cols[i]:
            st.markdown(
                f"""
                <div style="text-align: center; padding: 8px;">
                    <div style="
                        background-color: {color};
                        color: white;
                        padding: 6px 12px;
                        border-radius: 16px;
                        font-size: 12px;
                        font-weight: 600;
                        margin: 0 auto 4px;
                        display: inline-block;
                    ">{RELIABILITY_ICONS[level]} {level.value.upper()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
