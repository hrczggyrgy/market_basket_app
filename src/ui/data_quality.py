"""Reusable data quality expander for tab headers.

Shows data health summary and method readiness at top of major tabs.
"""

import streamlit as st

from src.analytics.data_quality import (
    calculate_method_readiness,
    format_readiness_for_ui,
    summarize_data_quality,
    validate_price_quantity,
)
from src.analytics.sufficiency import assess_data_sufficiency, format_sufficiency_summary


def render_data_quality_expander(
    df,
    analysis_name: str,
    params: dict = None,
    show_sufficiency: bool = True,
    show_validation: bool = True,
    show_readiness: bool = True,
    expanded: bool = False,
):
    """Render a comprehensive data quality expander at top of tab.

    Args:
        df: Transaction DataFrame
        analysis_name: Key for method readiness (e.g., 'elasticity', 'promo_uplift', 'cdt')
        params: Analysis parameters for readiness thresholds
        show_sufficiency: Show general data sufficiency badge
        show_validation: Show price/quantity validation
        show_readiness: Show method-specific readiness
        expanded: Start expanded
    """
    if df is None or df.empty:
        st.warning("No data loaded")
        return

    params = params or {}

    with st.expander("📋 Data Quality & Method Readiness", expanded=expanded):
        # --- General Sufficiency ---
        if show_sufficiency:
            sufficiency = assess_data_sufficiency(df)
            st.markdown(format_sufficiency_summary(sufficiency))

            if sufficiency["overall"] == "insufficient":
                st.error("Dataset too small for reliable analysis")
            elif sufficiency["overall"] == "directional":
                st.warning("Results should be treated as directional")

        # --- Price/Quantity Validation ---
        if show_validation:
            validation = validate_price_quantity(df)
            if validation["errors"]:
                for e in validation["errors"]:
                    st.error(f"❌ {e}")
            if validation["warnings"]:
                for w in validation["warnings"]:
                    st.warning(f"⚠️ {w}")
            if validation["info"]:
                for i in validation["info"]:
                    st.info(f"ℹ️ {i}")
            if not validation["errors"] and not validation["warnings"] and not validation["info"]:
                st.success("✅ Price/quantity validation passed")

        # --- Method Readiness ---
        if show_readiness:
            st.divider()
            readiness = calculate_method_readiness(df, analysis_name, params)
            st.markdown(format_readiness_for_ui(readiness))

            status = readiness.get("status", "unknown")
            if status == "blocked":
                st.error("🚫 **Analysis blocked** — requirements not met")
            elif status == "directional":
                st.warning("⚠️ **Directional only** — interpret with caution")
            elif status == "ready":
                st.success("✅ **Ready** — requirements met")

        # --- Quick Summary Stats ---
        st.divider()
        summary = summarize_data_quality(df)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Transactions", f"{summary.get('n_transactions', 0):,}")
        with col2:
            st.metric("Customers", f"{summary.get('n_customers', 0):,}")
        with col3:
            st.metric("Products", f"{summary.get('n_products', 0):,}")
        with col4:
            span = summary.get('date_span_days', 0)
            st.metric("Date Span (days)", f"{span:,}")

        # Missing customer ID alert
        missing_cust = summary.get('missing_customer_id', 0)
        if missing_cust and missing_cust > 0:
            pct = summary.get('missing_customer_id_pct', 0)
            if pct > 20:
                st.error(f"❌ {missing_cust:,} ({pct:.1f}%) rows missing customer_id — segmentation/CDT affected")
            elif pct > 5:
                st.warning(f"⚠️ {missing_cust:,} ({pct:.1f}%) rows missing customer_id")

        # Sparse SKU alert
        sparse = summary.get('sparse_sku_count', 0)
        if sparse and sparse > 0:
            st.info(f"ℹ️ {sparse} SKUs have ≤{summary.get('sparse_sku_threshold', '?')} transactions (sparse)")

        # Attribute coverage
        attr_cov = summary.get('attribute_coverage', {})
        if attr_cov:
            with st.expander("Attribute Coverage", expanded=False):
                for attr, cov in attr_cov.items():
                    st.write(f"**{attr}**: {cov['covered']:,} ({cov['pct']:.1f}%) covered")


def render_method_readiness_badge(readiness: dict) -> str:
    """Render a compact readiness badge for inline use."""
    status = readiness.get("status", "unknown")
    badges = {"ready": "🟢 Ready", "directional": "🟡 Directional", "blocked": "🔴 Blocked"}
    return badges.get(status, "⚪ Unknown")


def render_readiness_inline(df, analysis_name: str, params: dict = None):
    """Inline readiness indicator (single line)."""
    readiness = calculate_method_readiness(df, analysis_name, params or {})
    badge = render_method_readiness_badge(readiness)
    st.caption(f"Method Readiness: {badge}")
    return readiness