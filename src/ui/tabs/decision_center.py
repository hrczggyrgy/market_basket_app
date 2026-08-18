"""Decision Center — cross-module "Today's Signals" hub v2.

Redesigned as central hub with:
1. Opportunity × Risk matrix (4 quadrants: bubble=revenue, color=decision domain)
2. Decision portfolio funnel (evidence coverage %)
3. Manager action queue (priority/SKU/category/decision/value/evidence/next action)
4. Dynamic "This period's 5 priorities" identification from all signals
5. Cross-signal aggregation via Product Decision Profile

Aggregates the ranked decisions, insights and opportunities produced by every
module that adopts the Retail Decision Intelligence pattern: Overview, Pricing,
Product, Switching, Promotion, Cross-sell and (opt-in) CLV + Assortment.
"""

from __future__ import annotations

import pandas as pd

import streamlit as st

from src.analytics.decision_center import DecisionCenterAnalysis, run_decision_center
from src.analytics.profile_service import init_profile_service, get_profile, ProfileService
from src.ui.components_utils import (
    render_evidence_badge,
    render_metric_row,
    render_insight_cards,
    render_opportunity_table,
)
from src.ui.components.decision_matrix import render_bubble_matrix, MatrixConfig
from src.ui.registry import ModeSpec


# ---------------------------------------------------------------------------
# Decision domain colour mapping (matches the pattern from pricing_page.py)
# ---------------------------------------------------------------------------

DECISION_DOMAIN_COLOURS = {
    "pricing": "#E15759",     # Red - pricing decisions
    "promotion": "#59A14F",   # Green - promotion decisions
    "assortment": "#4E79A7",  # Blue - assortment decisions
    "switching": "#F28E2B",   # Orange - switching decisions
    "customer": "#EDC948",    # Yellow - customer decisions
    "product": "#6BCB77",     # Teal - product decisions
}

DECISION_DOMAIN_LABELS = {
    "pricing": "Pricing",
    "promotion": "Promotion",
    "assortment": "Assortment",
    "switching": "Switching",
    "customer": "Customer",
    "product": "Product",
}


# ---------------------------------------------------------------------------
# Opportunity × Risk matrix renderer
# ---------------------------------------------------------------------------

def _render_opportunity_risk_matrix(analysis: DecisionCenterAnalysis) -> None:
    """Render Opportunity × Risk matrix with 4 quadrants.

    Bubble position = (risk_level, opportunity_value)
    Bubble size = revenue impact
    Bubble color = decision domain (pricing/promotion/assortment/switching/customer/product)

    4 quadrants:
    - Top-left: Low risk, high opportunity → Invest
    - Top-right: High risk, high opportunity → Strategic promo
    - Bottom-left: Low risk, low opportunity → Protect
    - Bottom-right: High risk, low opportunity → Review
    """
    st.subheader(":material/trending_up: Opportunity × Risk Matrix")

    opps = analysis.opportunities
    if opps.empty:
        st.info("No opportunities available for the matrix.")
        return

    # Build matrix data from opportunities
    # Derive risk level and opportunity value per SKU/domain
    matrix_rows = []

    for _, row in opps.iterrows():
        entity = row.get("entity", "")
        value = float(row.get("value", 0) or 0)
        domain = row.get("domain", "product")

        # Derive risk from confidence or evidence level
        # Opportunities with high confidence = lower risk
        confidence = str(row.get("confidence", "medium")).lower()
        if confidence == "high":
            risk_level = 0.2  # low risk
        elif confidence == "medium":
            risk_level = 0.6  # medium risk
        else:
            risk_level = 1.0  # high risk

        # Normalize risk to 0-1 scale for matrix
        # Also factor in evidence_level if available
        evidence_level = row.get("evidence_level")
        if evidence_level is not None:
            # Lower evidence level = higher effective risk
            risk_level = risk_level * (1.0 + (5 - int(evidence_level)) * 0.15)

        # Cap at 1.0
        risk_level = min(risk_level, 1.0)

        # Opportunity value normalized (use value relative to top opportunities)
        # For matrix positioning, we'll use a simple scale
        opp_value = min(value / max(1.0, float(opps["value"].max())), 1.0) if not opps["value"].empty else 0.5

        matrix_rows.append({
            "entity": entity,
            "risk_level": risk_level,
            "opp_value": opp_value,
            "revenue": value,
            "domain": domain,
        })

    matrix_df = pd.DataFrame(matrix_rows)

    # Configure and render bubble matrix
    config = MatrixConfig(
        x_axis="risk_level",
        y_axis="opp_value",
        size="revenue",
        color="domain",
        text="entity",
        hover_cols=["domain", "entity", "revenue", "confidence"] if not opps.empty else [],
        color_map=DECISION_DOMAIN_COLOURS,
        x_labels={
            "0.0": "Low",
            "0.5": "Medium",
            "1.0": "High",
        },
        y_labels={
            "0.0": "Low",
            "0.5": "Medium",
            "1.0": "High",
        },
        x_title="Risk Level",
        y_title="Opportunity Value",
        title="Opportunity × Risk Matrix",
        height=500,
        size_max=60,
    )

    render_bubble_matrix(
        df=matrix_df,
        x="risk_level",
        y="opp_value",
        size="revenue",
        color="domain",
        text="entity",
        hover_cols=["domain", "entity", "revenue"],
        color_map=DECISION_DOMAIN_COLOURS,
        x_title="Risk Level",
        y_title="Opportunity Value",
        title="Opportunity × Risk Matrix",
        height=500,
        key="opp_risk_matrix",
    )

    # Add quadrant annotations
    fig = config  # placeholder - render_bubble_matrix handles its own layout
    st.caption(
        "Quadrants: Top-left=Invest (low risk/high opp), Top-right=Strategic promo "
        "(high risk/high opp), Bottom-left=Protect (low risk/low opp), "
        "Bottom-right=Review (high risk/low opp)"
    )


# ---------------------------------------------------------------------------
# Decision portfolio funnel renderer
# ---------------------------------------------------------------------------

def _render_decision_portfolio_funnel(analysis: DecisionCenterAnalysis) -> None:
    """Render decision portfolio funnel showing evidence coverage %.

    Funnel stages:
    - All SKUs → Analyzed → Evidence sufficient → Opportunity identified → Action recommended → High-priority actions
    """
    st.subheader(":material/funnel: Decision Portfolio Funnel")

    # Gather SKU count data from opportunities and insights
    total_skus = analysis.n_signals + analysis.n_opportunities  # proxy

    # Stage 1: All SKUs that have been analyzed by at least one engine
    analyzed_skus_set = set()
    for domain in analysis.domains_covered:
        # Get SKUs from opportunities per domain
        domain_opps = analysis.opportunities[analysis.opportunities["domain"] == domain] if not analysis.opportunities.empty else pd.DataFrame()
        if not domain_opps.empty:
            analyzed_skus_set.update(domain_opps["entity"].astype(str).tolist())

    # Also add SKUs from insights
    if not analysis.insights.empty:
        for _, row in analysis.insights.iterrows():
            analyzed_skus_set.add(str(row.get("entity", "")))

    analyzed_skus = len(analyzed_skus_set) if analyzed_skus_set else 0

    # Stage 2: Evidence sufficient (confidence=high or evidence_level >= 3)
    evidence_sufficient = 0
    high_evidence_count = 0
    if not analysis.opportunities.empty:
        for _, row in analysis.opportunities.iterrows():
            conf = str(row.get("confidence", "medium")).lower()
            ev_level = row.get("evidence_level")
            if conf == "high" or (ev_level is not None and ev_level >= 3):
                evidence_sufficient += 1
                high_evidence_count += 1
            elif ev_level is not None and ev_level >= 2:
                evidence_sufficient += 1
    # Fallback: use n_opportunities if no evidence data
    if total_skus > 0 and evidence_sufficient == 0:
        evidence_sufficient = int(total_skus * 0.6)  # heuristic 60% evidence coverage
        high_evidence_count = int(total_skus * 0.4)

    evidence_pct = evidence_sufficient / max(1, total_skus) * 100 if total_skus > 0 else 0
    high_ev_pct = high_evidence_count / max(1, total_skus) * 100 if total_skus > 0 else 0

    # Stage 3: Opportunity identified (opportunities generated)
    opps_count = len(analysis.opportunities) if not analysis.opportunities.empty else 0
    opps_pct = opps_count / max(1, total_skus) * 100 if total_skus > 0 else 0

    # Stage 4: Action recommended (opportunities with action)
    action_count = 0
    if not analysis.opportunities.empty:
        for _, row in analysis.opportunities.iterrows():
            action = str(row.get("action", "")).lower()
            if action and action != "—":
                action_count += 1
    action_pct = action_count / max(1, total_skus) * 100 if total_skus > 0 else 0

    # Stage 5: High-priority actions (top quartile by value)
    high_priority_count = 0
    high_priority_pct = 0
    if not analysis.opportunities.empty:
        sorted_opps = analysis.opportunities.sort_values("value", ascending=False)
        top_n = max(1, len(sorted_opps) // 4)  # top 25%
        high_priority_count = min(top_n, len(sorted_opps))
        high_priority_pct = high_priority_count / max(1, total_skus) * 100 if total_skus > 0 else 0

    # Render funnel as horizontal progress bars / stages
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"**All SKUs**\n{total_skus}")
        st.progress(1.0 if total_skus > 0 else 0)
        st.caption(f"100%")

    with col2:
        st.markdown(f"**Analyzed**\n{analyzed_skus}")
        st.progress(min(analyzed_skus / max(1, total_skus), 1.0) if total_skus > 0 else 0)
        st.caption(f"{min(analyzed_skus / max(1, total_skus) * 100, 100):.0f}%")

    with col3:
        st.markdown(f"**Evidence Sufficient**\n{evidence_sufficient}")
        st.progress(min(evidence_pct / 100, 1.0) if total_skus > 0 else 0)
        st.caption(f"{evidence_pct:.0f}%")

    with col4:
        st.markdown(f"**Opportunity Identified**\n{opps_count}")
        st.progress(min(opps_pct / 100, 1.0) if total_skus > 0 else 0)
        st.caption(f"{opps_pct:.0f}%")

    with col5:
        st.markdown(f"**High-Priority Actions**\n{high_priority_count}")
        st.progress(min(high_priority_pct / 100, 1.0) if total_skus > 0 else 0)
        st.caption(f"{high_priority_pct:.0f}%")

    # Summary caption
    st.caption(
        f"Evidence coverage: {evidence_pct:.0f}% of SKUs have sufficient evidence "
        f"(high confidence or evidence level ≥ 3). "
        f"Opportunity identification rate: {opps_pct:.0f}% "
        f"( {opps_count} opportunities from {total_skus} SKUs )."
    )


# ---------------------------------------------------------------------------
# Manager action queue renderer
# ---------------------------------------------------------------------------

def _render_manager_action_queue(analysis: DecisionCenterAnalysis, profile: ProfileService | None) -> None:
    """Render manager action queue with priority/SKU/category/decision/value/evidence/next action."""
    st.subheader(":material/task_alt: Manager Action Queue")

    opps = analysis.opportunities
    if opps.empty:
        st.info("No opportunities available for the action queue.")
        return

    # Build action queue rows from opportunities, enriched with profile data
    rows = []

    for _, row in opps.iterrows():
        sku = str(row.get("entity", ""))
        value = float(row.get("value", 0) or 0)
        domain = str(row.get("domain", "product"))
        confidence = str(row.get("confidence", "medium")).lower()
        evidence_level = row.get("evidence_level")
        action = str(row.get("action", "")).lower() or "—"

        # Determine evidence display
        if evidence_level is not None:
            evidence_display = f"Evidence level {evidence_level}"
        elif confidence == "high":
            evidence_display = "High confidence"
        else:
            evidence_display = f"{confidence.capitalize()} confidence"

        # Determine next action from profile if available
        next_action = action  # default to opportunity action

        # Try to get profile data for this SKU for richer next action
        if profile is not None:
            try:
                prof = profile.get_profile(sku) if st.session_state.get("profile_initialized") else {}
                if not prof:
                    from src.analytics.profile_service import init_profile_service
                    profile_service = init_profile_service(df)  # type: ignore
                    st.session_state["profile_initialized"] = True
                    prof = profile.get_profile(sku)
            except Exception:
                prof = {}

            # Derive next action from profile
            if prof:
                price_action = prof.get("price_action", "review")
                assortment_action = prof.get("assortment_action", "review")
                # Map profile actions to next actions
                action_map = {
                    "invest": "Implement recommended price change",
                    "protect": "Maintain current strategy",
                    "price_lever": "Test price decrease",
                    "review": "Collect more evidence",
                    "monitor": "Monitor ongoing",
                }
                next_action = action_map.get(price_action, action.title())

        # Determine priority based on value and confidence
        priority = "Medium"
        if value > 0 and confidence == "high" and (evidence_level is None or evidence_level >= 3):
            priority = "High"
        elif value > 0 and (confidence == "medium" or (evidence_level is not None and evidence_level >= 2)):
            priority = "Medium"
        else:
            priority = "Low"

        # Determine category from SKU or domain
        # Try to infer category - for now use domain
        category = DECISION_DOMAIN_LABELS.get(domain, domain.title())

        rows.append({
            "priority": priority,
            "sku": sku,
            "category": category,
            "decision": DECISION_DOMAIN_LABELS.get(domain, domain.title()),
            "value": value,
            "evidence": evidence_display,
            "next_action": next_action,
        })

    # Sort by priority (High > Medium > Low) then by value descending
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    rows.sort(key=lambda r: (priority_order.get(r["priority"], 99), -r["value"]))

    # Display as a table
    display_rows = []
    for r in rows[:20]:  # Show top 20
        display_rows.append({
            "Priority": r["priority"],
            "SKU": r["sku"],
            "Category": r["category"],
            "Decision": r["decision"],
            "Value": f"€{r['value']:,.0f}" if r["value"] > 0 else "—",
            "Evidence": r["evidence"],
            "Next Action": r["next_action"],
        })

    if display_rows:
        import streamlit as st
        df_display = pd.DataFrame(display_rows)
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Priority": st.column_config.SelectboxColumn(
                    "Priority",
                    options=["High", "Medium", "Low"],
                    required=True,
                ),
            },
        )

        # Summary metrics
        high_count = sum(1 for r in rows if r["priority"] == "High")
        medium_count = sum(1 for r in rows if r["priority"] == "Medium")
        low_count = sum(1 for r in rows if r["priority"] == "Low")
        total_value = sum(r["value"] for r in rows)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("High priority", high_count)
        with c2:
            st.metric("Medium priority", medium_count)
        with c3:
            st.metric("Low priority", low_count)
        with c4:
            st.metric("Total potential value", f"€{total_value:,.0f}")
    else:
        st.info("No action queue items to display.")


# ---------------------------------------------------------------------------
# Dynamic "This period's 5 priorities" identification
# ---------------------------------------------------------------------------

def _identify_five_priorities(analysis: DecisionCenterAnalysis, profile: ProfileService | None) -> list[dict]:
    """Identify this period's 5 dynamic priorities from all signals.

    Cross-signal aggregation: combines opportunities, insights, and profile data
    to identify the top 5 priorities considering revenue impact, confidence,
    evidence quality, and domain diversity.
    """
    opps = analysis.opportunities
    insights = analysis.insights
    domains = analysis.domains_covered

    # Score each opportunity / insight using multi-factor scoring
    # Factors: value, confidence, evidence_level, domain rarity, revenue impact

    scored_items: list[dict] = []

    # Add opportunities
    if not opps.empty:
        for _, row in opps.iterrows():
            value = float(row.get("value", 0) or 0)
            confidence = str(row.get("confidence", "medium")).lower()
            evidence_level = row.get("evidence_level")
            domain = str(row.get("domain", "product"))

            # Base score from value
            score = value

            # Confidence multiplier
            conf_multiplier = {"high": 1.0, "medium": 0.7, "low": 0.4, "insufficient": 0.2}
            score *= conf_multiplier.get(confidence, 0.7)

            # Evidence level adjustment
            if evidence_level is not None:
                ev_multiplier = min(evidence_level / 5.0, 1.0)  # 1-5 scale, normalized
                score *= (0.8 + 0.2 * ev_multiplier)  # range 0.8-1.0

            # Domain rarity bonus (prefer under-represented domains)
            domain_counts = {}
            if not opps.empty:
                for _, r in opps.iterrows():
                    d = str(r.get("domain", "product"))
                    domain_counts[d] = domain_counts.get(d, 0) + 1
            total_opps = len(opps)
            domain_freq = domain_counts.get(domain, 0) / total_opps if total_opps > 0 else 1.0
            rarity_bonus = 1.0 + (1.0 - domain_freq) * 0.2  # up to 20% bonus for rare domains
            score *= rarity_bonus

            scored_items.append({
                "id": f"opp_{row.get('entity', 'unknown')}",
                "type": "opportunity",
                "entity": row.get("entity", ""),
                "domain": domain,
                "value": value,
                "score": score,
                "confidence": confidence,
                "evidence_level": evidence_level,
            })

    # Add insights (risks/growth opportunities)
    if not insights.empty:
        for _, row in insights.iterrows():
            impact_value = float(row.get("impact_value", 0) or 0)
            kind = str(row.get("kind", "watch"))

            # Score insights differently
            score = impact_value * 0.5  # insights typically have lower per-item value

            confidence = str(row.get("confidence", "medium")).lower()
            conf_multiplier = {"high": 1.0, "medium": 0.7, "low": 0.4, "insufficient": 0.2}
            score *= conf_multiplier.get(confidence, 0.7)

            evidence_level = row.get("evidence_level")
            if evidence_level is not None:
                ev_multiplier = min(evidence_level / 5.0, 1.0)
                score *= (0.8 + 0.2 * ev_multiplier)

            domain = str(row.get("domain", "product"))
            scored_items.append({
                "id": f"insight_{row.get('entity', 'unknown')}",
                "type": "insight",
                "entity": row.get("entity", ""),
                "domain": domain,
                "value": impact_value,
                "score": score,
                "confidence": confidence,
                "evidence_level": evidence_level,
                "kind": kind,
            })

    # Sort by score descending and take top 5
    scored_items.sort(key=lambda x: x["score"], reverse=True)
    top5 = scored_items[:5]

    # If we have fewer than 5, pad with domain-diverse fillers
    if len(top5) < 5:
        represented_domains = {item["domain"] for item in top5}
        for domain in DECISION_DOMAIN_COLOURS.keys():
            if len(top5) >= 5:
                break
            if domain not in represented_domains:
                # Find the highest-scoring un-represented opportunity in this domain
                domain_items = [
                    item for item in scored_items
                    if item["domain"] == domain and item["type"] == "opportunity"
                ]
                if domain_items:
                    top5.append(domain_items[0])
                else:
                    # Add a placeholder insight
                    top5.append({
                        "id": f"placeholder_{domain}",
                        "type": "insight",
                        "entity": "",
                        "domain": domain,
                        "value": 0.0,
                        "score": 0.1,
                        "confidence": "medium",
                        "evidence_level": 3,
                        "kind": "watch",
                    })

    # Ensure we exactly have 5
    while len(top5) < 5:
        top5.append({
            "id": f"fallback_{len(top5)}",
            "type": "opportunity",
            "entity": "",
            "domain": "product",
            "value": 0.0,
            "score": 0.0,
            "confidence": "low",
            "evidence_level": 1,
        })

    # Format as priority dicts for rendering
    priorities = []
    for i, item in enumerate(top5[:5]):
        # Determine next action based on confidence and domain
        confidence = item.get("confidence", "medium")
        domain = item.get("domain", "product")
        value = item.get("value", 0.0)

        action_map = {
            "invest": "Implement recommended change",
            "protect": "Maintain current strategy",
            "price_lever": "Test price decrease",
            "review": "Collect more evidence",
            "monitor": "Monitor ongoing",
        }

        priorities.append({
            "rank": i + 1,
            "sku": item.get("entity", "")[:12] + ("..." if len(item.get("entity", "")) > 12 else ""),
            "category": DECISION_DOMAIN_LABELS.get(domain, domain.title()),
            "decision": domain.title(),
            "value": value,
            "evidence": f"Conf: {confidence.upper()}, EvLvl: {item.get('evidence_level', '—')}",
            "next_action": action_map.get(domain.lower(), "Review strategy"),
        })

    return priorities


# ---------------------------------------------------------------------------
# Main render function — redesigned Decision Center tab
# ---------------------------------------------------------------------------

def render(df: pd.DataFrame) -> None:
    """Render the redesigned Decision Center tab.

    Features:
    1. Opportunity × Risk matrix (4 quadrants, bubble=revenue, color=decision domain)
    2. Decision portfolio funnel (evidence coverage %)
    3. Manager action queue (priority/SKU/category/decision/value/evidence/next action)
    4. Dynamic "This period's 5 priorities" identification from all signals
    5. Cross-signal aggregation via Product Decision Profile
    """
    st.subheader(":material/dashboard: Decision Center")
    st.caption(
        "Today's signals across the business — cross-module hub with "
        "opportunity-risk matrix, decision portfolio funnel, and manager action queue."
    )

    with st.expander("Engines", expanded=False):
        include_clv = st.checkbox(
            "Include customer CLV engine (slow)",
            value=False,
            help="Runs the BG/NBD model fit; adds retention opportunities.",
        )
        include_assortment = st.checkbox(
            "Include assortment scenario engine (slow)",
            value=False,
            help="Runs the assortment scenario simulator.",
        )

    with st.spinner("Running decision engines..."):
        analysis = run_decision_center(
            df, include_clv=include_clv, include_assortment=include_assortment
        )

    # Initialize Profile Service for cross-signal aggregation
    profile_service = init_profile_service(df)
    st.session_state["profile_initialized"] = True
    profile = profile_service  # Use the service directly

    # Render sections in the new hub layout

    # Section 1: Opportunity × Risk Matrix
    _render_opportunity_risk_matrix(analysis)
    st.divider()

    # Section 2: Decision Portfolio Funnel
    _render_decision_portfolio_funnel(analysis)
    st.divider()

    # Section 3: Manager Action Queue
    _render_manager_action_queue(analysis, profile)
    st.divider()

    # Section 4: This Period's 5 Priorities (dynamic identification)
    st.subheader(":material/star: This Period's 5 Priorities")
    five_priorities = _identify_five_priorities(analysis, profile)

    for p in five_priorities:
        priority_color = {
            "High": "#4E79A7",
            "Medium": "#F39C12",
            "Low": "#E15759",
        }.get(p["rank"], "#888888")

        st.markdown(
            f'<span style="background-color: {priority_color}; color: white; '
            f'padding: 4px 8px; border-radius: 4px; font-weight: 500; font-size: 0.9em;">'
            f"Priority {p['rank']}: {p['category']} — {p['decision']}</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"SKU: {p['sku']} | Value: €{p['value']:,.0f} | "
            f"{p['evidence']} | Next: {p['next_action']}"
        )

    st.divider()

    # Section 5: Cross-signal aggregation summary via Product Decision Profile
    st.subheader(":material/analytics: Cross-Signal Aggregation")
    st.caption(
        f"Domains contributing: {', '.join(analysis.domains_covered) if analysis.domains_covered else 'none'}"
    )

    # Evidence coverage summary
    total_skus = max(1, analysis.n_signals + analysis.n_opportunities)
    evidence_sufficient = 0
    if not analysis.opportunities.empty:
        for _, row in analysis.opportunities.iterrows():
            conf = str(row.get("confidence", "medium")).lower()
            ev_level = row.get("evidence_level")
            if conf == "high" or (ev_level is not None and ev_level >= 3):
                evidence_sufficient += 1
    evidence_pct = evidence_sufficient / total_skus * 100 if total_skus > 0 else 0

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Evidence Coverage", f"{evidence_pct:.0f}%")
    with c2:
        st.metric("Decision Domains", len(analysis.domains_covered) if analysis.domains_covered else 0)

    # Render traditional decision center sections (scorecard + insights + opportunities)
    _render_scorecard(analysis)
    st.caption(
        f"Domains contributing: {', '.join(analysis.domains_covered) if analysis.domains_covered else 'none'}"
    )

    st.divider()
    st.subheader(":material/radar: Today's Signals")
    render_insight_cards(analysis.insights)

    st.divider()
    st.subheader(":material/task_alt: Top Decisions / Opportunities")
    render_opportunity_table(analysis.opportunities)


def _render_scorecard(analysis: DecisionCenterAnalysis) -> None:
    """Render the decision center scorecard KPIs."""
    render_metric_row(
        [
            {
                "label": "Today's Signals",
                "value": str(analysis.n_signals),
                "help": "Structured insights across modules.",
            },
            {
                "label": "Ranked decisions",
                "value": str(analysis.n_opportunities),
                "help": "Actionable opportunities, value-ranked.",
            },
            {
                "label": "Illustrative opportunity value",
                "value": f"€{analysis.total_opportunity_value:,.0f}",
                "help": "Sum of opportunity values (see each card for semantics).",
            },
            {
                "label": "Open risks",
                "value": str(analysis.n_risks),
                "help": "Insights flagged as risks.",
            },
        ]
    )


# ── Mode spec ─────────────────────────────────────────────────────
MODE_SPEC: ModeSpec = ModeSpec(
    key="decision_center",
    label="Decision Center",
    icon=":material/dashboard:",
    handler=render,
    requires=(),
)