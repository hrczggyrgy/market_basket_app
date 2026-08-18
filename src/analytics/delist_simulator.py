"""Delist Impact Simulator.

Scenario tool where selecting an SKU shows revenue at risk, estimated recovery,
recovery rate, and substitutes with revenue flows; calculates recovery from
substitution network; displays KEEP/DELIST recommendation with confidence.

Integrates with Product Decision Profile for SKU decision fields.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.profile_service import get_profile, get_profile_service, init_profile_service
from src.analytics.transference import (
    compute_demand_transference_matrix,
    delist_impact_analysis,
)
from src.analytics.switching import compute_switching_status


def _get_revenue_at_risk(sku: str, revenue_by_product: pd.Series) -> float:
    """Total revenue at risk if this SKU is removed — the SKU's own revenue."""
    return float(revenue_by_product.get(sku, 0.0))


def _get_estimated_recovery(
    sku: str,
    demand_transference_df: pd.DataFrame,
    revenue_by_product: pd.Series,
) -> float:
    """Estimated recovery revenue from delisted SKU via substitutes.

    Sums observed_switching_transfer_revenue from the SKU to all other products
    that can capture the demand.
    """
    if demand_transference_df is None or demand_transference_df.empty:
        return 0.0
    transferred = float(
        demand_transference_df[demand_transference_df["from_product"] == sku][
            "observed_switching_transfer_revenue"
        ].sum()
    )
    return transferred


def _get_recovery_rate(
    revenue_at_risk: float,
    estimated_recovery: float,
) -> float:
    """Recovery rate = recovery / revenue at risk * 100%."""
    if revenue_at_risk > 0:
        return min(1.0, estimated_recovery / revenue_at_risk)
    return 0.0


def _get_top_substitutes(
    sku: str,
    demand_transference_df: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """Top N substitutes with revenue flows for a delisted SKU."""
    if demand_transference_df is None or demand_transference_df.empty:
        return pd.DataFrame(columns=["to_product", "revenue_flow"])
    sub_df = demand_transference_df[
        demand_transference_df["from_product"] == sku
    ][["to_product", "observed_switching_transfer_revenue"]].copy()
    if sub_df.empty:
        return pd.DataFrame(columns=["to_product", "revenue_flow"])
    sub_df = sub_df.sort_values(
        "observed_switching_transfer_revenue", ascending=False
    ).head(top_n)
    sub_df = sub_df.rename(
        columns={"observed_switching_transfer_revenue": "revenue_flow"}
    )
    return sub_df


def _get_switching_status(sku: str, df: pd.DataFrame) -> str:
    """Get switching status for an SKU, used to determine evidence level."""
    try:
        status_df = compute_switching_status(df)
        ss_rows = status_df[status_df["stockcode"] == sku]
        if len(ss_rows) > 0 and not ss_rows.empty:
            return str(ss_rows.iloc[0]["switching_status"])
    except Exception:
        pass
    return "unavailable"


def _classify_evidence_level(
    switching_status: str,
    revenue_at_risk: float,
    estimated_recovery: float,
    sdp: float | None = None,
) -> str:
    """Classify evidence level as HIGH, MEDIUM, or INSUFFICIENT.

    Falls back to 'Insufficient evidence' when switching data is weak.
    """
    # If switching status indicates insufficient data, return INSUFFICIENT
    if switching_status in ("insufficient_customers", "insufficient_transitions", "insufficient_observations"):
        return "Insufficient evidence"

    # If revenue at risk is very low or recovery is negligible, evidence is weak
    if revenue_at_risk <= 0:
        return "Insufficient evidence"

    # Check if recovery rate is meaningful
    recovery_rate = _get_recovery_rate(revenue_at_risk, estimated_recovery)
    if recovery_rate < 0.1:  # Less than 10% recovery
        return "Insufficient evidence"

    # If we have switching data and meaningful recovery, evidence is sufficient
    if switching_status == "estimated":
        if recovery_rate >= 0.3:
            return "High evidence"
        elif recovery_rate >= 0.1:
            return "Medium evidence"
        else:
            return "Insufficient evidence"

    # Default: if we have some data but not enough for "estimated"
    return "Insufficient evidence"


def _get_keep_delist_recommendation(
    profile: dict[str, Any],
    revenue_at_risk: float,
    estimated_recovery: float,
    recovery_rate: float,
    substitutes: pd.DataFrame,
    switching_status: str,
) -> tuple[str, str, float]:
    """Return (label, action, confidence) for KEEP/DELIST recommendation.

    label: "KEEP" or "DELIST"
    action: human-readable rationale
    confidence: 0.0–1.0 based on data completeness and evidence level
    """
    # Get profile fields
    kvi_score = float(profile.get("kvi_score", 0.5))
    revenue = float(profile.get("revenue", revenue_at_risk))
    substitutability = float(profile.get("substitutability", 0.5))
    elasticity = float(profile.get("elasticity", 0.0))
    reach = float(profile.get("customer_reach", 0.0))
    switching_risk = profile.get("switching_risk", "unknown")

    # Data completeness check
    profile_fields = [
        "revenue", "abc", "xyz", "lifecycle", "velocity", "repeat_rate",
        "customer_reach", "elasticity", "substitutability", "kvi_score",
    ]
    filled = sum(1 for f in profile_fields if profile.get(f) not in (None, 0.0, "unknown", "mature"))
    total = len(profile_fields)
    base_confidence = min(1.0, filled / total) if total > 0 else 0.0

    # Evidence level modifies confidence
    evidence_level = _classify_evidence_level(
        switching_status, revenue_at_risk, estimated_recovery, substitutability
    )

    # If evidence is insufficient, force "KEEP" with low confidence
    if evidence_level == "Insufficient evidence":
        return "KEEP", "Insufficient evidence for delist decision — retain SKU for further analysis", 0.3

    # Decision logic with sufficient evidence
    is_kvi = kvi_score >= 0.6
    high_rev = revenue > 100.0  # threshold can be adjusted

    # KEEP: KVIs protected, or high-revenue high-substitutability anchors
    # Also keep if recovery rate is high (delisting would lose too much)
    if is_kvi or (high_rev and substitutability < 0.3):
        label = "KEEP"
        if recovery_rate >= 0.5:
            action = f"Keep {revenue_at_risk:,.0f} at risk; {recovery_rate:.0%} recoverable from substitutes — KVI protected"
        else:
            action = f"Keep {revenue_at_risk:,.0f} at risk; low recovery ({recovery_rate:.0%}) — KVI protected"
    # DELIST: low revenue, high substitutability, high recovery rate
    elif (not is_kvi) and (substitutability >= 0.5) and (recovery_rate >= 0.3):
        label = "DELIST"
        action = f"Delist {revenue_at_risk:,.0f} — {recovery_rate:.0%} recoverable, substitutability={substitutability:.0%}"
    # Review: medium case
    else:
        label = "KEEP"
        action = f"Keep {revenue_at_risk:,.0f} at risk — review needed, recovery rate {recovery_rate:.0%}"

    # Adjust confidence based on evidence level
    if evidence_level == "High evidence":
        confidence = min(1.0, base_confidence * 1.2)
    elif evidence_level == "Medium evidence":
        confidence = base_confidence
    else:
        confidence = max(0.1, base_confidence * 0.5)

    return label, action, confidence


def simulate_delist_impact(
    sku: str,
    df: pd.DataFrame,
    profile_service: object | None = None,
) -> dict[str, object]:
    """Simulate the delist impact for a given SKU.

    Computes revenue at risk, estimated recovery, recovery rate,
    top substitutes with revenue flows, and KEEP/DELIST recommendation.

    Args:
        sku: The SKU stockcode to analyze.
        df: Transaction DataFrame with canonical columns.
        profile_service: Optional ProfileService instance. If not provided,
            one will be initialized from df.

    Returns:
        Dict with keys:
        - revenue_at_risk: float, total revenue of selected SKU
        - estimated_recovery: float, revenue that can be recovered from substitutes
        - recovery_rate: float, recovery / revenue_at_risk (0-1)
        - recovery_rate_pct: str, recovery rate as percentage string
        - top_substitutes: DataFrame with top 3 substitutes and revenue flows
        - recommendation: "KEEP" or "DELIST"
        - recommendation_action: human-readable rationale
        - confidence: float, 0.0-1.0
        - evidence_level: "High evidence" | "Medium evidence" | "Insufficient evidence"
        - switching_status: per-product switching status
    """
    # Initialize profile service if not provided
    if profile_service is None:
        try:
            profile_service = init_profile_service(df)
        except Exception:
            profile_service = None

    # Get revenue by product
    from src.analytics.data import revenue_column
    revenue_by_product = revenue_column(df).groupby(df["stockcode"]).sum()

    # Get revenue at risk = the SKU's total revenue
    revenue_at_risk = _get_revenue_at_risk(sku, revenue_by_product)

    # Compute demand transference matrix (substitution network)
    demand_transference_df = compute_demand_transference_matrix(df)

    # Get estimated recovery from substitutes
    estimated_recovery = _get_estimated_recovery(sku, demand_transference_df, revenue_by_product)

    # Get recovery rate
    recovery_rate = _get_recovery_rate(revenue_at_risk, estimated_recovery)
    recovery_rate_pct = f"{recovery_rate * 100:.1f}%"

    # Get top 3 substitutes with revenue flows
    top_substitutes = _get_top_substitutes(sku, demand_transference_df, top_n=3)

    # Get switching status for evidence level classification
    switching_status = _get_switching_status(sku, df)

    # Get Product Decision Profile
    profile = {}
    if profile_service is not None:
        try:
            profile = get_profile_service().get_profile(sku) if hasattr(get_profile_service(), 'get_profile') else {}
            if not profile and profile_service is not None:
                # Try direct get_profile if available
                try:
                    profile = profile_service.get_profile(sku) if hasattr(profile_service, 'get_profile') else {}
                except Exception:
                    profile = {}
        except Exception:
            profile = {}

    # Classify evidence level
    evidence_level = _classify_evidence_level(
        switching_status, revenue_at_risk, estimated_recovery,
        float(profile.get("substitutability", 0.5))
    )

    # Get KEEP/DELIST recommendation
    recommendation, recommendation_action, confidence = _get_keep_delist_recommendation(
        profile, revenue_at_risk, estimated_recovery, recovery_rate, top_substitutes, switching_status
    )

    # Format top substitutes for display
    substitutes_display = []
    if not top_substitutes.empty:
        for _, row in top_substitutes.iterrows():
            substitutes_display.append({
                "substitute": row["to_product"],
                "revenue_flow": f"${row['revenue_flow']:,.0f}",
            })

    result = {
        "revenue_at_risk": revenue_at_risk,
        "estimated_recovery": estimated_recovery,
        "recovery_rate": recovery_rate,
        "recovery_rate_pct": recovery_rate_pct,
        "top_substitutes": substitutes_display,
        "recommendation": recommendation,
        "recommendation_action": recommendation_action,
        "confidence": confidence,
        "evidence_level": evidence_level,
        "switching_status": switching_status,
    }

    return result


def render_delist_simulator(
    df: pd.DataFrame,
    sku_selector: str | None = None,
) -> dict[str, object]:
    """Render the Delist Impact Simulator UI in Streamlit.

    Args:
        df: Transaction DataFrame with canonical columns.
        sku_selector: Optional SKU to pre-select. If None, first SKU in data is used.

    Returns:
        Dict with simulation results (same keys as simulate_delist_impact).
    """
    import streamlit as st

    # Initialize profile service
    try:
        profile_service = init_profile_service(df)
    except Exception:
        profile_service = None

    # Get list of SKUs
    from src.analytics.data import revenue_column
    revenue_per_product = revenue_column(df).groupby(df["stockcode"]).sum()
    all_skus = list(revenue_per_product.index)

    # Select SKU
    if sku_selector is None:
        sku_selector = all_skus[0] if all_skus else None

    if sku_selector is None or sku_selector not in all_skus:
        st.error("No SKU data available")
        return {}

    # Run simulation
    result = simulate_delist_impact(sku_selector, df, profile_service)

    # --- UI Layout ---
    st.subheader(f":material/delete_sweep: Delist Impact Simulator — SKU: {sku_selector}")

    # Revenue at risk
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Revenue at Risk", f"${result['revenue_at_risk']:,.0f}")
    with c2:
        st.metric("Estimated Recovery", f"${result['estimated_recovery']:,.0f}")

    # Recovery rate
    st.metric("Recovery Rate", result["recovery_rate_pct"], help=f"Recovery at risk / Revenue at risk")

    # Top substitutes
    st.write("**Top 3 Substitutes with Revenue Flows**")
    if result["top_substitutes"]:
        for sub in result["top_substitutes"]:
            st.write(f"• {sub['substitute']}: {sub['revenue_flow']}")
    else:
        st.write("No substitute data available")

    # Recommendation
    st.write("**KEEP/DELIST Recommendation**")
    rec_color = "normal" if result["recommendation"] == "KEEP" else "inverse"
    st.markdown(f"**{result['recommendation']}** — {result['recommendation_action']}")
    st.caption(f"Confidence: {result['confidence']:.1%} | Evidence: {result['evidence_level']}")

    # Switching status
    st.caption(f"Switching status: {result['switching_status']}")

    return result