"""Reusable insight header component for primary results.

Compact component showing finding, evidence, confidence, and limitation.
Used above the PRIMARY result in major tabs.
"""

import streamlit as st


CONFIDENCE_COLORS = {
    "Strong": "🟢",
    "Directional": "🟡",
    "Exploratory": "🟠",
    "Unavailable": "⚪",
}

CONFIDENCE_HELP = {
    "Strong": "Robust sample, method validated, causal claim supported",
    "Directional": "Adequate sample, method sound, but limitations prevent strong claims",
    "Exploratory": "Small sample, method assumptions unmet, or early-stage analysis",
    "Unavailable": "Insufficient data or method not applicable",
}


def render_result_context(
    title: str,
    finding: str,
    evidence: str,
    confidence: str = "Directional",
    limitation: str = "",
    expanded: bool = False,
):
    """Render a compact result context card above primary output.

    Args:
        title: Section title (e.g., "Top Elasticity Estimate", "Delist Impact Scenario")
        finding: What the selected result means in plain language
        evidence: Relevant metric, denominator, sample size
        confidence: One of "Strong", "Directional", "Exploratory", "Unavailable"
        limitation: Transaction-data or method limitation
        expanded: Whether to show as expander (default False = inline)
    """
    badge = CONFIDENCE_COLORS.get(confidence, "⚪")
    help_text = CONFIDENCE_HELP.get(confidence, "")

    if expanded:
        with st.expander(f"{badge} **{title}** — Confidence: {confidence}", expanded=True):
            _render_context_body(finding, evidence, confidence, limitation, help_text)
    else:
        st.markdown(f"### {badge} {title}")
        st.caption(f"Confidence: **{confidence}** — {help_text}")
        _render_context_body(finding, evidence, confidence, limitation, help_text)


def _render_context_body(
    finding: str,
    evidence: str,
    confidence: str,
    limitation: str,
    help_text: str,
):
    """Render the body of the context card."""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"**Finding:** {finding}")
        st.markdown(f"**Evidence:** {evidence}")

    with col2:
        if limitation:
            st.markdown(f"**⚠️ Limitation:**")
            st.markdown(f"*{limitation}*")

    if confidence in ("Exploratory", "Unavailable"):
        st.warning(
            f"**{confidence} result** — {help_text}. "
            "Do not use for high-stakes decisions without further validation."
        )
    elif confidence == "Directional":
        st.info(
            f"**Directional result** — {help_text}. "
            "Use for hypothesis generation and prioritization."
        )
    else:
        st.success(
            f"**Strong result** — {help_text}. "
            "Suitable for decision support with standard caveats."
        )


def render_metric_context(
    metric_key: str,
    value: float,
    sample_size: int,
    confidence: str = "Directional",
    denominator: str = "",
):
    """Convenience wrapper for single-metric results.

    Uses metric_definitions for canonical labels and caveats.
    """
    from src.analytics.metric_definitions import get_metric_label, get_metric_caveat, get_metric_help

    label = get_metric_label(metric_key)
    caveat = get_metric_caveat(metric_key)
    help_text = get_metric_help(metric_key)

    finding = f"{label} = {value:.3f}" if isinstance(value, float) else f"{label} = {value}"
    evidence = f"n = {sample_size:,}"
    if denominator:
        evidence += f" (denominator: {denominator})"

    limitation = caveat if caveat and confidence in ("Directional", "Exploratory") else ""

    render_result_context(
        title=f"{label} Estimate",
        finding=finding,
        evidence=evidence,
        confidence=confidence,
        limitation=limitation,
    )


def render_scenario_context(
    scenario_name: str,
    selected_sku: str,
    source_revenue: float,
    transfer_rate: float,
    unrecovered: float,
    sample_size: int,
    method: str,
    data_quality: str,
):
    """Specialized context for demand transference / assortment scenarios."""
    finding = (
        f"Delisting **{selected_sku}** (${source_revenue:,.0f} revenue) "
        f"recovers ~{transfer_rate:.0%} via substitution; "
        f"${unrecovered:,.0f} estimated unrecovered."
    )
    evidence = (
        f"Sample: {sample_size:,} transactions | "
        f"Method: {method} | Data quality: {data_quality}"
    )
    limitation = (
        "Historical scenario estimate; not a guarantee of future behavior. "
        "Assumes observed switching = substitution. No causal identification. "
        "Does not account for competitive response, stockouts, or marketing changes."
    )

    render_result_context(
        title=f"Scenario: {scenario_name}",
        finding=finding,
        evidence=evidence,
        confidence="Directional" if data_quality == "good" else "Exploratory",
        limitation=limitation,
    )


def render_elasticity_context(
    sku: str,
    elasticity: float,
    n_obs: int,
    price_cv: float,
    n_price_points: int,
    method: str,
    hdi_lower: float = None,
    hdi_upper: float = None,
):
    """Specialized context for elasticity estimates."""
    if elasticity < -1:
        interp = "Elastic — demand sensitive to price"
    elif elasticity < -0.1:
        interp = "Inelastic — demand not very sensitive"
    else:
        interp = "Positive/near-zero — likely omitted variable bias"

    finding = f"**{sku}**: Elasticity = {elasticity:.3f} ({interp})"
    evidence = (
        f"Observations: {n_obs} weekly periods | "
        f"Price CV: {price_cv:.3f} | Distinct price points: {n_price_points} | "
        f"Method: {method}"
    )
    if hdi_lower is not None and hdi_upper is not None:
        evidence += f" | 94% HDI: [{hdi_lower:.3f}, {hdi_upper:.3f}]"

    limitation = (
        "Observational estimate from transaction data only. "
        "Confounded by promotions, seasonality, stockouts, and competitor actions. "
        "No causal identification without experimental price variation."
    )

    # Confidence based on data adequacy
    if n_obs >= 20 and price_cv >= 0.1 and n_price_points >= 5:
        confidence = "Directional"
    elif n_obs >= 10 and price_cv >= 0.05:
        confidence = "Exploratory"
    else:
        confidence = "Unavailable"

    render_result_context(
        title="Price Elasticity Estimate",
        finding=finding,
        evidence=evidence,
        confidence=confidence,
        limitation=limitation,
    )


def render_uplift_context(
    treatment_n: int,
    control_n: int,
    overlap_pct: float,
    validation_score: float,
    method: str,
    ate: float = None,
):
    """Specialized context for promo uplift modeling."""
    finding = f"Treatment: {treatment_n:,} | Control: {control_n:,} | Overlap: {overlap_pct:.1f}%"
    if ate is not None:
        finding += f" | ATE: {ate:.3f}"

    evidence = (
        f"Method: {method} | Propensity overlap: {overlap_pct:.1f}% | "
        f"Validation score: {validation_score:.3f}"
    )

    limitation = (
        "Observational causal estimate. Requires strong ignorability assumption. "
        "No randomized experiment. Unmeasured confounding likely. "
        "Treatment/control overlap < 80% → high extrapolation risk."
    )

    if overlap_pct >= 80 and validation_score >= 0.7:
        confidence = "Directional"
    elif overlap_pct >= 60:
        confidence = "Exploratory"
    else:
        confidence = "Unavailable"

    render_result_context(
        title="Promo Uplift Estimate",
        finding=finding,
        evidence=evidence,
        confidence=confidence,
        limitation=limitation,
    )