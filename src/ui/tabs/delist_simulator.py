"""Delist Impact Simulator tab.

Scenario tool where selecting an SKU shows revenue at risk, estimated recovery,
recovery rate, and substitutes with revenue flows; reuse existing switching/assortment
code; calculate recovery from substitution network; display keep/delist recommendation
with confidence. Integrates with Product Decision Profile for SKU decision fields.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui.registry import ModeSpec

from src.analytics.delist_simulator import simulate_delist_impact


# ---------------------------------------------------------------------------
# Render function
# ---------------------------------------------------------------------------

def render(df: pd.DataFrame) -> None:
    """Render the Delist Impact Simulator tab.

    Five key displays:
      1) SKU selector — choose an SKU from the current dataset
      2) Revenue at risk — total revenue of selected SKU
      3) Estimated recovery — revenue recoverable from substitutes
      4) Recovery rate — recovery / revenue at risk * 100%
      5) Top 3 substitutes with revenue flows
      6) KEEP/DELIST recommendation with confidence level
      7) Product Decision Profile integration
    """

    # Initialize profile service from data
    from src.analytics.profile_service import init_profile_service
    try:
        profile_service = init_profile_service(df)
    except Exception:
        profile_service = None

    # Get SKU list
    from src.analytics.data import revenue_column
    revenue_per_product = revenue_column(df).groupby(df["stockcode"]).sum()
    all_skus = list(revenue_per_product.index)

    # SKU selector in sidebar
    st.sidebar.selectbox(
        "Select SKU to analyze",
        options=all_skus,
        index=all_skus.index(st.session_state.get("delist_sku", all_skus[0])) if all_skus else 0,
        key="delist_sku",
    )
    selected_sku = st.session_state.get("delist_sku", all_skus[0] if all_skus else None)

    if selected_sku is None or selected_sku not in all_skus:
        st.error("No SKU data available")
        return

    # Run simulation
    result = simulate_delist_impact(selected_sku, df, profile_service)

    # --- Display ---
    st.subheader(f":material/delete_sweep: Delist Impact Simulator — SKU: {selected_sku}")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Revenue at Risk", f"${result['revenue_at_risk']:,.0f}")
    with c2:
        st.metric("Estimated Recovery", f"${result['estimated_recovery']:,.0f}")
    with c3:
        st.metric("Recovery Rate", result["recovery_rate_pct"])
    with c4:
        st.metric("Evidence Level", result["evidence_level"])

    st.divider()

    # Top 3 substitutes
    st.write("**Top 3 Substitutes with Revenue Flows**")
    if result["top_substitutes"]:
        for sub in result["top_substitutes"]:
            st.write(f"• **{sub['substitute']}**: {sub['revenue_flow']}")
    else:
        st.write("No substitute data available")

    st.divider()

    # KEEP/DELIST Recommendation
    st.write("**KEEP/DELIST Recommendation**")
    if result["recommendation"] == "KEEP":
        st.success(f"**KEEP** — {result['recommendation_action']}")
    else:
        st.error(f"**DELIST** — {result['recommendation_action']}")

    st.caption(f"Confidence: {result['confidence']:.1%} | Evidence: {result['evidence_level']}")

    st.divider()

    # Product Decision Profile summary
    if profile_service is not None:
        try:
            profile = profile_service.get_profile(selected_sku)
            st.write("**Product Decision Profile (selected SKU)**")
            profile_cols = st.columns(3)
            profile_fields = [
                ("revenue", "Revenue"),
                ("substitutability", "Substitutability"),
                ("kvi_score", "KVI Score"),
                ("customer_reach", "Customer Reach"),
                ("switching_risk", "Switching Risk"),
                ("abc", "ABC Class"),
                ("xyz", "XYZ Class"),
            ]
            for i, (field, label) in enumerate(profile_fields):
                with profile_cols[i % 3]:
                    val = profile.get(field, "N/A")
                    if isinstance(val, float):
                        st.caption(f"{label}: ${val:,.0f}" if "revenue" in label else f"{label}: {val:.2f}")
                    else:
                        st.caption(f"{label}: {val}")
        except Exception as e:
            st.caption(f"Profile unavailable: {e}")

    st.divider()

    # Evidence note
    st.caption(
        f"Switching status: {result['switching_status']} | "
        f"Calculation based on observed switching correlations. "
        f"Fallback to 'Insufficient evidence' when switching data is weak."
    )


# ---------------------------------------------------------------------------
# Mode specification (must come after render function)
# ---------------------------------------------------------------------------

MODE_SPEC: ModeSpec = ModeSpec(
    key="delist_simulator",
    label="Delist Impact Simulator",
    icon=":material/delete_sweep:",
    handler=render,
    requires=("has_stockcode", "sufficient_skus_20"),
)