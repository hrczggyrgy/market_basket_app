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

        with st.container(border=True):
            st.markdown(f"**{meta['icon']} {title}**")
            st.markdown(evidence)
            if action:
                st.markdown(f"**Recommended action:** {action}")
            parts: list[str] = []
            impact = row.get("impact_value")
            if impact is not None and pd.notna(impact):
                parts.append(f"Impact: €{float(impact):,.0f}")
            sample = row.get("sample_size")
            if sample is not None and pd.notna(sample):
                parts.append(f"Sample: {int(sample):,}")
            stability = row.get("stability")
            if stability is not None and pd.notna(stability):
                parts.append(f"Stability: {float(stability):.0%}")
            parts.append(f"Confidence: {confidence.capitalize()}")
            st.caption(" · ".join(parts))


def render_opportunity_table(opps_df: pd.DataFrame) -> None:
    """Render ranked Opportunities as an actionable table."""
    if opps_df is None or opps_df.empty:
        st.caption("No actionable opportunities identified from this data.")
        return
    work = opps_df.copy()
    work = work.sort_values("value", ascending=False, na_position="last")
    display = work[["entity", "title", "value", "confidence", "action", "source"]].copy()
    display["value"] = display["value"].map(
        lambda v: f"€{float(v):,.0f}" if pd.notna(v) else "—"
    )
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
    kvi_row = kvi_data[kvi_data["stockcode"] == stockcode].iloc[0] if not kvi_data.empty else None
    decision_row = decision_data[decision_data["stockcode"] == stockcode].iloc[0] if not decision_data.empty else None
    elast_row = elasticity_data[elasticity_data["stockcode"] == stockcode].iloc[0] if not elasticity_data.empty else None
    status_row = status_data[status_data["stockcode"] == stockcode].iloc[0] if not status_data.empty else None
    conf_row = confidence_data[confidence_data["stockcode"] == stockcode].iloc[0] if not confidence_data.empty else None

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
            unsafe_allow_html=True
        )

        # Value metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Revenue", f"€{float(kvi_row['total_revenue']):,.0f}")
        with col2:
            penetration = float(kvi_row.get('basket_penetration', 0))
            st.metric("Basket Penetration", f"{penetration:.1%}")
        with col3:
            kvi_score = float(kvi_row.get('kvi_score', 0))
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
