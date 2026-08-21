"""Evidence Badge Component.

Displays evidence level using either the 1-5 exploratory-to-causal scale
or the 4-tier reliability convention (HIGH/MEDIUM/LOW/INSUFFICIENT),
with bidirectional mapping between the two conventions.

Supports compact inline rendering for use next to matrix/bubble cells
in tab handlers, and full badges with tooltips for detailed view.

Evidence scale (1-5):
  1: Exploratory       - Preliminary analysis, no modeling
  2: Descriptive       - Descriptive statistics, basic patterns
  3: Predictive        - Model-derived predictions
  4: Quasi-causal      - Quasi-experimental causal evidence
  5: Causal            - Definitive causal identification

Reliability tiers (4-level):
  HIGH     - Strong evidence, robust findings
  MEDIUM   - Moderate evidence, some supporting data
  LOW      - Weak evidence, limited data/support
  INSUFFICIENT - Insufficient evidence, cannot conclude
"""

from __future__ import annotations

import streamlit as st

# 1-5 exploratory-to-causal evidence scale
EVIDENCE_LEVELS = {
    1: {
        "label": "Exploratory",
        "color": "#FF6B6B",  # Red - preliminary
        "description": "Preliminary analysis, no modeling assumptions.",
    },
    2: {
        "label": "Descriptive",
        "color": "#FFD93D",  # Yellow - descriptive patterns
        "description": "Descriptive statistics, basic patterns identified.",
    },
    3: {
        "label": "Predictive",
        "color": "#6BCB77",  # Green - model-derived
        "description": "Model-derived estimate (e.g., elasticity, CLV).",
    },
    4: {
        "label": "Quasi-causal",
        "color": "#4D96FF",  # Blue - quasi-experimental
        "description": "Quasi-experimental causal evidence (e.g., IV, RDD).",
    },
    5: {
        "label": "Causal",
        "color": "#9B5DE5",  # Purple - definitive causal
        "description": "Definitive causal identification (full assumptions met).",
    },
}

# 4-tier reliability convention
RELIABILITY_TIERS = {
    "HIGH": {
        "label": "HIGH",
        "color": "#59A14F",  # Green - strong
        "description": "Strong evidence, robust findings.",
    },
    "MEDIUM": {
        "label": "MEDIUM",
        "color": "#F28E2B",  # Orange - moderate
        "description": "Moderate evidence, some supporting data.",
    },
    "LOW": {
        "label": "LOW",
        "color": "#E15759",  # Red - weak
        "description": "Weak evidence, limited data/support.",
    },
    "INSUFFICIENT": {
        "label": "INSUFFICIENT",
        "color": "#7F7F7F",  # Gray - insufficient
        "description": "Insufficient evidence, cannot conclude.",
    },
}

# Bidirectional mapping: evidence level -> reliability tier
EVIDENCE_TO_RELIABILITY = {
    1: "LOW",
    2: "LOW",
    3: "MEDIUM",
    4: "HIGH",
    5: "HIGH",
}

# Bidirectional mapping: reliability tier -> evidence levels
RELIABILITY_TO_EVIDENCE = {
    "INSUFFICIENT": [1],
    "LOW": [1, 2],
    "MEDIUM": [3],
    "HIGH": [4, 5],
}


def map_evidence_to_reliability(evidence_level: int) -> str:
    """Map a 1-5 evidence level to a reliability tier.

    Args:
        evidence_level: Evidence level (1-5)

    Returns:
        Reliability tier string (HIGH/MEDIUM/LOW/INSUFFICIENT)
    """
    return EVIDENCE_TO_RELIABILITY.get(evidence_level, "LOW")


def map_reliability_to_evidence(reliability_tier: str) -> list[int]:
    """Map a reliability tier to possible evidence levels.

    Args:
        reliability_tier: Reliability tier (HIGH/MEDIUM/LOW/INSUFFICIENT)

    Returns:
        List of evidence levels (1-5) that map to this tier
    """
    return RELIABILITY_TO_EVIDENCE.get(reliability_tier, [3])


def render_evidence_badge(
    level: int | str | None = None,
    convention: str = "evidence",
    show_mapping: bool = True,
    compact: bool = False,
) -> None:
    """Render an evidence badge with dual convention support.

    Accepts either a 1-5 exploratory-to-causal evidence level or a
    4-tier reliability convention value, and renders with appropriate
    color coding. Can also display the mapped counterpart from the
    other convention.

    Args:
        level: Evidence level (1-5 integer for "evidence" convention,
               or reliability tier string for "reliability" convention)
        convention: Input convention type ("evidence" for 1-5 scale,
                    "reliability" for 4-tier reliability)
        show_mapping: Show the mapped counterpart from the other convention
        compact: Use compact inline style (for matrix/bubble cells)
    """
    # ── Normalize input ────────────────────────────────────────────
    if convention == "evidence":
        # level should be int 1-5
        if level is None or level not in EVIDENCE_LEVELS:
            level = 3  # default to Predictive
        level_data = EVIDENCE_LEVELS[level]
        reliability_tier = map_evidence_to_reliability(level)

        # Determine display text
        if compact:
            display_text = level_data["label"]
            # In compact mode, show mapped reliability as small suffix
            if show_mapping:
                display_text += f" · {reliability_tier[0]}"
            badge_color = level_data["color"]
            badge_label = level_data["label"]
        else:
            badge_label = level_data["label"]
            badge_color = level_data["color"]
            reliability_label = RELIABILITY_TIERS[reliability_tier]["label"]

            # Build tooltip/description
            tooltip_parts = [
                f"Evidence level: {badge_label} (Level {level})",
                f"Mapped reliability: {reliability_tier} ({reliability_label})",
                f"Description: {level_data['description']}",
            ]
            tooltip = "<br>".join(tooltip_parts)

            # Full badge HTML
            badge_html = f'''
            <div style="display: inline-flex; align-items: center; gap: 8px;">
                <span style="
                    background-color: {badge_color};
                    color: white;
                    padding: 4px 12px;
                    border-radius: 16px;
                    font-size: 12px;
                    font-weight: 600;
                    display: inline-flex;
                    align-items: center;
                    gap: 6px;
                " title="{tooltip.replace('"', '"')}">
                    {badge_label}
                </span>
            </div>
            '''
            st.markdown(badge_html, unsafe_allow_html=True)
            return

    elif convention == "reliability":
        # level should be "HIGH"/"MEDIUM"/"LOW"/"INSUFFICIENT"
        if level is None or level not in RELIABILITY_TIERS:
            level = "MEDIUM"
        level_data = RELIABILITY_TIERS[level]

        # Map to evidence levels
        evidence_levels = RELIABILITY_TO_EVIDENCE.get(level, [3])
        primary_evidence_level = evidence_levels[0]
        evidence_data = EVIDENCE_LEVELS[primary_evidence_level]
        evidence_label = evidence_data["label"]

        if compact:
            display_text = level_data["label"]
            if show_mapping:
                display_text += f" · L{primary_evidence_level}"
            badge_color = level_data["color"]
            badge_label = level_data["label"]
        else:
            badge_label = level_data["label"]
            badge_color = level_data["color"]
            evidence_data["label"]

            # Build tooltip/description
            tooltip_parts = [
                f"Reliability tier: {badge_label}",
                f"Mapped evidence level: {evidence_label} (Level {primary_evidence_level})",
                f"Description: {level_data['description']}",
            ]
            tooltip = "<br>".join(tooltip_parts)

            # Full badge HTML
            badge_html = f'''
            <div style="display: inline-flex; align-items: center; gap: 8px;">
                <span style="
                    background-color: {badge_color};
                    color: white;
                    padding: 4px 12px;
                    border-radius: 16px;
                    font-size: 12px;
                    font-weight: 600;
                    display: inline-flex;
                    align-items: center;
                    gap: 6px;
                " title="{tooltip.replace('"', '"')}">
                    {badge_label}
                </span>
            </div>
            '''
            st.markdown(badge_html, unsafe_allow_html=True)
            return

    else:
        # Unknown convention, default
        level = 3
        convention = "evidence"
        level_data = EVIDENCE_LEVELS[level]
        badge_color = level_data["color"]
        badge_label = level_data["label"]
        reliability_tier = "MEDIUM"

    # ── Compact inline badge (default/fallback path) ────────────────
    if compact:
        badge_html = f'''
        <span style="
            background-color: {badge_color};
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        ">
            {badge_label}
        </span>
        '''
        st.markdown(badge_html, unsafe_allow_html=True)
        return

    # ── Full badge fallback ──────────────────────────────────────────
    badge_html = f'''
    <div style="display: inline-flex; align-items: center; gap: 8px;">
        <span style="
            background-color: {badge_color};
            color: white;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 12px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        " title="{badge_label}">
            {badge_label}
        </span>
    </div>
    '''
    st.markdown(badge_html, unsafe_allow_html=True)
