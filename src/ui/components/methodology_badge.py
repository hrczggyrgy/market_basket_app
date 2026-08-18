"""Methodology Badge Component.

Displays evidence class (Observed/Estimated/Causal) with click-through
to methodology documentation.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import streamlit as st


class EvidenceClass(Enum):
    """Evidence class for analytical outputs."""

    OBSERVED = "observed"
    ESTIMATED = "estimated"
    CAUSAL = "causal"


EVIDENCE_COLORS = {
    EvidenceClass.OBSERVED: "#59A14F",  # Green - directly calculated from POS
    EvidenceClass.ESTIMATED: "#F28E2B",  # Orange - model-derived
    EvidenceClass.CAUSAL: "#E15759",  # Red - requires identification assumptions
}

EVIDENCE_ICONS = {
    EvidenceClass.OBSERVED: "📊",
    EvidenceClass.ESTIMATED: "🤖",
    EvidenceClass.CAUSAL: "🔬",
}

EVIDENCE_DESCRIPTIONS = {
    EvidenceClass.OBSERVED: "Directly calculated from POS transaction data. No modeling assumptions.",
    EvidenceClass.ESTIMATED: "Model-derived estimate (e.g., elasticity, CLV, KVI score). Subject to model assumptions.",
    EvidenceClass.CAUSAL: "Requires causal identification assumptions (e.g., IV, RDD, synthetic control). Not purely observational.",
}


def render_methodology_badge(
    evidence: EvidenceClass | str,
    show_tooltip: bool = True,
    link: Optional[str] = None,
    compact: bool = False,
) -> None:
    """Render a methodology/evidence badge.

    Args:
        evidence: EvidenceClass enum or string
        show_tooltip: Show description on hover
        link: Optional URL to methodology documentation
        compact: Use compact inline style
    """
    if isinstance(evidence, str):
        evidence = EvidenceClass(evidence.lower())

    color = EVIDENCE_COLORS[evidence]
    icon = EVIDENCE_ICONS[evidence]
    description = EVIDENCE_DESCRIPTIONS[evidence]

    if compact:
        badge_html = f"""
        <span style="
            background-color: {color};
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        ">
            {icon} {evidence.value.upper()}
        </span>
        """
        st.markdown(badge_html, unsafe_allow_html=True)
    else:
        tooltip = description
        if link:
            tooltip += f' <a href="{link}" target="_blank" style="color: white; text-decoration: underline;">→ Methodology</a>'

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
                {icon} {evidence.value.upper()}
            </span>
        </div>
        '''
        st.markdown(badge_html, unsafe_allow_html=True)


def render_evidence_legend() -> None:
    """Render a legend showing all evidence classes."""
    cols = st.columns(3)
    for i, (evidence, color) in enumerate(EVIDENCE_COLORS.items()):
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
                    ">{EVIDENCE_ICONS[evidence]} {evidence.value.upper()}</div>
                    <div style="font-size: 11px; color: #666;">{EVIDENCE_DESCRIPTIONS[evidence]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_evidence_inline(evidence: EvidenceClass | str) -> str:
    """Return HTML string for inline evidence badge (for use in tables).

    Args:
        evidence: EvidenceClass or string

    Returns:
        HTML string
    """
    if isinstance(evidence, str):
        evidence = EvidenceClass(evidence.lower())

    color = EVIDENCE_COLORS[evidence]
    icon = EVIDENCE_ICONS[evidence]

    return f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 600; display: inline-flex; align-items: center; gap: 4px;">{icon} {evidence.value.upper()}</span>'
