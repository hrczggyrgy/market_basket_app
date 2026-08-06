"""Overview / Dashboard tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.data import get_data_summary
from src.analytics.basket_metrics import compute_basket_penetration
from src.analytics.performance import compute_product_metrics
from src.analytics.data_quality import DataQualityReport, generate_quality_summary
from src.ui.registry import ModeSpec


def render(df: pd.DataFrame) -> None:
    """Render the overview dashboard."""
    summary = get_data_summary(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions", f"{summary['n_transactions']:,}")
    col2.metric("Customers", f"{summary['n_customers']:,}")
    col3.metric("Products", f"{summary['n_products']:,}")
    col4.metric("Revenue", f"${summary['total_revenue']:,.2f}")

    st.divider()

    # Quick charts
    c1, c2 = st.columns(2)

    with c1:
        st.subheader(":material/shopping_cart: Basket Penetration")
        bp = compute_basket_penetration(df).head(10)
        st.bar_chart(bp.set_index("stockcode")["penetration"])

    with c2:
        st.subheader(":material/trending_up: Top Products by Revenue")
        pm = compute_product_metrics(df).head(10)
        st.bar_chart(pm.set_index("stockcode")["revenue"])

    # Data quality
    st.divider()
    st.subheader(":material/verified: Data Quality")
    
    # Check if quality report is in session state
    quality_report = st.session_state.get("quality_report")
    if quality_report:
        st.markdown(generate_quality_summary(quality_report))
        
        # Show details in expanders
        if quality_report.low_freq_products:
            with st.expander(f"Low-frequency products ({len(quality_report.low_freq_products)})", expanded=False):
                freq_df = pd.DataFrame({
                    "stockcode": quality_report.low_freq_products,
                    "transactions": [quality_report.low_freq_counts.get(p, 0) for p in quality_report.low_freq_products]
                })
                st.dataframe(freq_df, use_container_width=True, hide_index=True)
        
        if quality_report.basket_outlier_txn_ids:
            with st.expander(f"Basket size outliers ({len(quality_report.basket_outlier_txn_ids)})", expanded=False):
                st.write(f"Threshold: {quality_report.basket_outlier_threshold} items (above {quality_report.basket_size_percentile:.0%} percentile)")
                st.write(f"Outlier transaction IDs: {', '.join(quality_report.basket_outlier_txn_ids[:50])}")
                if len(quality_report.basket_outlier_txn_ids) > 50:
                    st.caption(f"... and {len(quality_report.basket_outlier_txn_ids) - 50} more")
        
        if quality_report.duplicate_count > 0:
            with st.expander(f"Duplicate transactions ({quality_report.duplicate_count})", expanded=False):
                st.write(f"Duplicate transaction IDs: {', '.join(quality_report.duplicate_txn_ids[:50])}")
                if len(quality_report.duplicate_txn_ids) > 50:
                    st.caption(f"... and {len(quality_report.duplicate_txn_ids) - 50} more")
        
        if quality_report.incomplete_rows > 0:
            with st.expander(f"Incomplete rows ({quality_report.incomplete_rows})", expanded=False):
                for col, cnt in quality_report.incomplete_row_details.items():
                    st.write(f"- {col}: {cnt} missing")
    else:
        st.json({
            "date_range": summary['date_range'],
            "avg_basket_value": f"${summary['avg_basket_value']:.2f}",
            "avg_items_per_basket": f"{summary['avg_basket_size']:.2f}",
        })


MODE_SPEC: ModeSpec = ModeSpec(
    key="overview",
    label="Overview",
    icon=":material/dashboard:",
    handler=render,
)