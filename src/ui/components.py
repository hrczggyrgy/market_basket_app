"""Shared decision-intelligence UI components.

Render Insight cards, ranked opportunity tables and KPI scorecards consistently
across tabs so every page follows the Signal -> Evidence -> Interpretation ->
Impact -> Action pattern.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

KIND_META: dict[str, dict[str, str]] = {
    "opportunity": {"icon": ":material/trending_up:", "color": "#59A14F"},
    "risk": {"icon": ":material/error:", "color": "#E15759"},
    "growth": {"icon": ":material/trending_up:", "color": "#4E79A7"},
    "leakage": {"icon": ":material/water_drop:", "color": "#B07AA1"},
    "anomaly": {"icon": ":material/pulse_alert:", "color": "#F28E2B"},
    "efficiency": {"icon": ":material/check_circle:", "color": "#76B7B2"},
    "watch": {"icon": ":material/visibility:", "color": "#EDC948"},
}


def render_evidence_badge(evidence_level: int | None = None) -> None:
    """Render an evidence level badge (1-5) with appropriate color and tooltip.

    Evidence levels:
    1: Exploratory
    2: Descriptive
    3: Predictive
    4: Quasi-causal
    5: Causal
    """
    if evidence_level is None:
        level_label = "No evidence"
        color = "#888888"  # Gray
    elif evidence_level == 1:
        level_label = "Exploratory"
        color = "#FF6B6B"  # Red
    elif evidence_level == 2:
        level_label = "Descriptive"
        color = "#FFD93D"  # Yellow
    elif evidence_level == 3:
        level_label = "Predictive"
        color = "#6BCB77"  # Green
    elif evidence_level == 4:
        level_label = "Quasi-causal"
        color = "#4D96FF"  # Blue
    else:  # evidence_level == 5
        level_label = "Causal"
        color = "#9B5DE5"  # Purple

    st.markdown(
        f'<span style="background-color: {color}; color: white; padding: 2px 8px; '
        f'border-radius: 10px; font-size: 0.8em; font-weight: 500;">{level_label}</span>',
        unsafe_allow_html=True,
    )


def render_delta_badge(
    delta_value: float | None = None, is_percent: bool = False, positive_good: bool = True
) -> None:
    """Render a delta/change badge with appropriate coloring.

    Args:
        delta_value: The change value (can be positive or negative)
        is_percent: Whether the value is a percentage
        positive_good: Whether positive values are good (True) or bad (False)
                      For revenue: positive_good=True (more is better)
                      For costs: positive_good=False (less is better)
    """
    if delta_value is None or (isinstance(delta_value, float) and pd.isna(delta_value)):
        st.markdown(
            '<span style="background-color: #888888; color: white; padding: 2px 8px; '
            'border-radius: 10px; font-size: 0.8em; font-weight: 500;">—</span>',
            unsafe_allow_html=True,
        )
        return

    # Format the value
    if is_percent:
        formatted_value = f"{delta_value:+.1%}"
    else:
        formatted_value = f"€{delta_value:+,.0f}"

    # Determine if the delta is "good" or "bad"
    is_good = (delta_value > 0) if positive_good else (delta_value < 0)
    color = "#6BCB77" if is_good else "#FF6B6B"  # Green for good, Red for bad

    st.markdown(
        f'<span style="background-color: {color}; color: white; padding: 2px 8px; '
        f'border-radius: 10px; font-size: 0.8em; font-weight: 500;">{formatted_value}</span>',
        unsafe_allow_html=True,
    )


def render_metric_row(metrics: list[dict[str, Any]]) -> None:
    """Render a row of KPI metric tiles (label, value, help)."""
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics, strict=False):
        col.metric(
            label=metric.get("label", ""),
            value=metric.get("value", ""),
            help=metric.get("help"),
        )


def render_insight_cards(insights_df: pd.DataFrame) -> None:
    """Render structured Insight rows as decision cards."""
    if insights_df is None or insights_df.empty:
        st.caption("No insights available for this data.")
        return
    for _, row in insights_df.iterrows():
        kind = str(row.get("kind", "watch"))
        meta = KIND_META.get(kind, KIND_META["watch"])
        title = str(row.get("title", ""))
        evidence = str(row.get("evidence", ""))
        action = str(row.get("action", ""))
        confidence = str(row.get("confidence", "medium"))
        evidence_level = row.get("evidence_level")
        impact_value = row.get("impact_value")
        sample_size = row.get("sample_size")
        stability = row.get("stability")
        n_transition_pairs = row.get("n_transition_pairs")
        n_unique_products = row.get("n_unique_products")
        confidence_gate = row.get("confidence_gate")

        with st.container(border=True):
            # Header with icon and title
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{meta['icon']} {title}**")
            with col2:
                # Confidence indicator
                confidence_colors = {
                    "high": "#2ECC71",
                    "medium": "#F39C12",
                    "low": "#E74C3C",
                    "insufficient": "#95A5A6",
                }
                conf_color = confidence_colors.get(confidence.lower(), "#95A5A6")
                st.markdown(
                    f'<span style="background-color: {conf_color}; color: white; padding: 2px 6px; '
                    f'border-radius: 3px; font-size: 0.8em;">{confidence.upper()}</span>',
                    unsafe_allow_html=True,
                )

            # Evidence section
            st.markdown(evidence)

            # Evidence level badge
            render_evidence_badge(evidence_level)

            # Action section
            if action:
                st.markdown(f"**Recommended action:** {action}")

            # Metrics section in an organized layout
            metric_cols = st.columns(4)

            with metric_cols[0]:
                # Display impact as delta badge for visual indication of positive/negative impact
                if impact_value is not None and pd.notna(impact_value):
                    # For impact, positive is generally good (revenue increase, cost savings, etc.)
                    render_delta_badge(impact_value, as_pct=False, suffix="€", positive_good=True)
                    st.caption(f"Impact: €{impact_value:,.0f}")
                else:
                    st.metric("Impact", "—")

            with metric_cols[1]:
                if sample_size is not None and pd.notna(sample_size):
                    st.metric("Sample Size", f"{int(sample_size):,}")
                else:
                    st.metric("Sample Size", "—")

            with metric_cols[2]:
                if stability is not None and pd.notna(stability):
                    st.metric("Stability", f"{stability:.0%}")
                else:
                    st.metric("Stability", "—")

            with metric_cols[3]:
                # Show switching-specific metrics if available
                if n_transition_pairs is not None and n_unique_products is not None:
                    st.metric("Switching Pairs", f"{n_transition_pairs}")
                elif confidence_gate is not None:
                    gate_status = "������✓ Pass" if confidence_gate else "������✗ Fail"
                    st.metric("Evidence Gate", gate_status)
                else:
                    st.metric("Details", "—")


def render_opportunity_table(opps_df: pd.DataFrame) -> None:
    """Render ranked Opportunities as an actionable table."""
    if opps_df is None or opps_df.empty:
        st.caption("No actionable opportunities identified from this data.")
        return
    work = opps_df.copy()
    work = work.sort_values("value", ascending=False, na_position="last")
    display = work[["entity", "title", "value", "confidence", "action", "source"]].copy()
    display["value"] = display["value"].map(lambda v: f"€{float(v):,.0f}" if pd.notna(v) else "—")
    display.columns = ["Entity", "Opportunity", "Value", "Confidence", "Action", "Source"]
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_pricing_decision_card(
    stockcode: str,
    kvi_data: pd.DataFrame,
    decision_data: pd.DataFrame,
    elasticity_data: pd.DataFrame,
    status_data: pd.DataFrame,
    confidence_data: pd.DataFrame,
) -> None:
    """Render a comprehensive pricing decision card for a single SKU.

    Displays commercial role, value metrics, price sensitivity, reliability,
    and recommended action in a consolidated view.
    """
    # Get SKU data
    kvi_row = (
        kvi_data[kvi_data["stockcode"] == stockcode].iloc[0]
        if not kvi_data.empty and (kvi_data["stockcode"] == stockcode).any()
        else None
    )
    decision_row = (
        decision_data[decision_data["stockcode"] == stockcode].iloc[0]
        if not decision_data.empty and (decision_data["stockcode"] == stockcode).any()
        else None
    )
    elast_row = (
        elasticity_data[elasticity_data["stockcode"] == stockcode].iloc[0]
        if not elasticity_data.empty and (elasticity_data["stockcode"] == stockcode).any()
        else None
    )
    status_row = (
        status_data[status_data["stockcode"] == stockcode].iloc[0]
        if not status_data.empty and (status_data["stockcode"] == stockcode).any()
        else None
    )
    conf_row = (
        confidence_data[confidence_data["stockcode"] == stockcode].iloc[0]
        if not confidence_data.empty and (confidence_data["stockcode"] == stockcode).any()
        else None
    )

    if kvi_row is None:
        st.error(f"SKU {stockcode} not found in KVI data.")
        return

    # Commercial role
    decision = decision_row["decision"] if decision_row is not None else "unknown"
    decision_labels = {
        "invest": "Invest (high-KVI, elastic traffic driver)",
        "protect": "Protect (high-KVI, inelastic margin carrier)",
        "price_lever": "Price lever (low-KVI, elastic)",
        "review": "Review (low-KVI, inelastic)",
        "insufficient_evidence": "Insufficient evidence for pricing decision",
    }

    with st.container(border=True):
        st.markdown(f"### SKU: {stockcode}")

        # Commercial role badge
        role_color = {
            "invest": "#59A14F",
            "protect": "#4E79A7",
            "price_lever": "#F28E2B",
            "review": "#EDC948",
            "insufficient_evidence": "#E15759",
        }.get(decision, "#888888")

        st.markdown(
            f'<span style="background-color: {role_color}; color: white; padding: 4px 12px; '
            f'border-radius: 4px; font-weight: bold;">{decision_labels.get(decision, decision)}</span>',
            unsafe_allow_html=True,
        )

        # Value metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Revenue", f"€{float(kvi_row['total_revenue']):,.0f}")
        with col2:
            penetration = float(kvi_row.get("basket_penetration", 0))
            st.metric("Basket Penetration", f"{penetration:.1%}")
        with col3:
            kvi_score = float(kvi_row.get("kvi_score", 0))
            st.metric("KVI Score", f"{kvi_score:.2f}")

        st.divider()

        # Price sensitivity section
        st.markdown("#### Price Sensitivity")
        if elast_row is not None and pd.notna(elast_row.get("elasticity")):
            elasticity = float(elast_row["elasticity"])
            conf = conf_row["confidence"] if conf_row is not None else "unknown"
            n_obs = int(elast_row.get("n_obs", 0))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Elasticity", f"{elasticity:.2f}")
            with col2:
                st.metric("Confidence", conf.upper())
            with col3:
                st.metric("Observations", n_obs)
        else:
            status = status_row["elasticity_status"] if status_row is not None else "unknown"
            st.warning(f"Elasticity not estimable: {status}")

        # Reliability section
        st.markdown("#### Reliability")
        if status_row is not None:
            status = status_row["elasticity_status"]
            n_obs = int(status_row.get("n_obs", 0))
            price_cv = status_row.get("price_cv")

            col1, col2 = st.columns(2)
            with col1:
                st.text(f"Status: {status}")
                st.text(f"Observations: {n_obs}")
            with col2:
                if price_cv is not None and pd.notna(price_cv):
                    st.text(f"Price CV: {price_cv:.3f}")

        # Recommendation
        st.divider()
        st.markdown("#### Recommended Action")
        if decision_row is not None and pd.notna(decision_row.get("rationale")):
            st.info(decision_row["rationale"])
        else:
            st.info("Collect more price variation data to enable evidence-based pricing decisions.")


def render_opportunity_cards(opps_df: pd.DataFrame) -> None:
    """Render structured Opportunity rows as cards."""
    if opps_df is None or opps_df.empty:
        st.caption("No opportunities available for this data.")
        return
    for _, row in opps_df.iterrows():
        title = str(row.get("title", ""))
        rationale = str(row.get("rationale", ""))
        action = str(row.get("action", ""))
        confidence = str(row.get("confidence", "medium"))
        value = row.get("value")
        source = str(row.get("source", ""))

        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.markdown(rationale)
            if action:
                st.markdown(f"**Recommended action:** {action}")
            parts: list[str] = []
            if value is not None and pd.notna(value):
                parts.append(f"Value: €{float(value):,.0f}")
            if source:
                parts.append(f"Source: {source}")
            parts.append(f"Confidence: {confidence.capitalize()}")
            st.caption(" · ".join(parts))
