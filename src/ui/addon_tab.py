"""Add-on analysis tab."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.addon import get_addon_recommendations, get_anchor_addon_matrix
from src.analytics.sufficiency import assess_data_sufficiency, format_sufficiency_summary
from src.analytics.basket_metrics import compute_basket_penetration
from src.ui.export import render_analytics_export
from src.ui.insight_header import render_result_context
from src.ui.data_quality import render_data_quality_expander


def render_addon_tab(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict, pipeline: dict = None
):
    """Render add-on analysis tab."""
    st.header("➕ Add-on / Complementary Products")
    st.caption(
        "Finds products that are bought **alongside** an anchor item in the same basket. "
        "**Revenue uplift per anchor** estimates incremental value per transaction. "
        "**Associative only — not causal incrementality.**"
    )

    if transactions_df.empty:
        st.warning("No transaction data available")
        return

    # Data quality & readiness at top
    render_data_quality_expander(transactions_df, "addon", params, expanded=False)

    # Compute basket penetration for enrichment
    basket_pen = None
    with st.spinner("Computing basket penetration..."):
        basket_pen = compute_basket_penetration(transactions_df)

    # Mode selection
    mode = st.radio(
        "Analysis Mode",
        ["Single Anchor Product", "Multiple Anchors (Top Products)"],
        horizontal=True,
        key="addon_mode_radio",
    )

    if mode == "Single Anchor Product":
        render_single_addon(transactions_df, product_lookup, params, basket_pen)
    else:
        render_multi_addon(transactions_df, product_lookup, params, basket_pen)


def render_single_addon(transactions_df: pd.DataFrame, product_lookup: dict, params: dict, basket_pen: pd.DataFrame):
    """Single anchor product add-on analysis."""
    st.subheader("Add-on Recommendations for Anchor Product")

    # Product selector
    products = transactions_df["stockcode"].unique()
    anchor = st.selectbox(
        "Select Anchor Product",
        options=products,
        format_func=lambda x: product_lookup.get(x, x) if product_lookup else x,
        key="addon_anchor_select",
    )

    if anchor:
        with st.spinner(f"Finding add-ons for {product_lookup.get(anchor, anchor)}..."):
            addons = get_addon_recommendations(
                transactions_df,
                anchor,
                min_lift=params.get("min_lift", 1.2),
                top_n=params.get("top_n", 10),
            )

        if not addons.empty:
            addons = _enrich_addons_with_penetration(addons, basket_pen, product_lookup)
            addons["Add-on Name"] = addons["addon_product"].map(product_lookup)
            addons["Anchor Name"] = addons["anchor_product"].map(product_lookup)

            # Insight header for best add-on
            if "revenue_uplift_per_anchor" in addons.columns:
                best_addon = addons.nlargest(1, "revenue_uplift_per_anchor").iloc[0]

                evidence_parts = [
                    f"Uplift: ${best_addon['revenue_uplift_per_anchor']:.2f}/anchor txn",
                    f"Lift: {best_addon['lift']:.2f}",
                    f"P(Add-on|Anchor): {best_addon['p_addon_given_anchor']:.2%}",
                ]
                if "basket_penetration" in best_addon:
                    evidence_parts.append(f"Basket Pen: {best_addon['basket_penetration']:.2%}")

                render_result_context(
                    title="Top Add-on Recommendation",
                    finding=f"`{best_addon['Add-on Name']}` adds ${best_addon['revenue_uplift_per_anchor']:.2f} per `{product_lookup.get(anchor, anchor)}` transaction",
                    evidence=" | ".join(evidence_parts),
                    confidence="Directional",
                    limitation="Associative basket uplift — not causal incrementality. Confounded by trip type, shopper segments, promotions.",
                )

            display_cols = [
                "Add-on Name",
                "p_addon_given_anchor",
                "p_addon_baseline",
                "lift",
                "leverage",
                "conviction",
                "revenue_uplift_per_anchor",
                "addon_price",
            ]
            # Add penetration columns if available
            pen_cols = [c for c in addons.columns if "basket_penetration" in c or "shopper_penetration" in c]
            display_cols.extend(pen_cols)
            available = [c for c in display_cols if c in addons.columns]

            st.dataframe(addons[available].round(4), width="stretch", hide_index=True)

            # Disclaimer
            st.caption(
                "⚠️ **Interpretation**: Lift > 1 indicates co-occurrence above chance. "
                "Revenue uplift is associative (halo effect), not causal incrementality. "
                "No control for confounding factors (trip mission, shopper type, promotions)."
            )

            render_analytics_export(addons, f"AddOns_{anchor}")

            # Visualization
            st.subheader("Add-on Lift vs Confidence")

            fig = px.scatter(
                addons,
                x="p_addon_given_anchor",
                y="lift",
                size="revenue_uplift_per_anchor",
                color="p_addon_given_anchor",
                hover_data=["Add-on Name"],
                title=f"Add-ons for {product_lookup.get(anchor, anchor)}",
                labels={"p_addon_given_anchor": "P(Add-on | Anchor)", "lift": "Lift"},
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No strong add-on products found for this anchor")


def render_multi_addon(transactions_df: pd.DataFrame, product_lookup: dict, params: dict, basket_pen: pd.DataFrame):
    """Multiple anchor products add-on matrix."""
    st.subheader("Add-on Matrix for Top Products")

    @st.cache_data
    def get_addon_matrix_cached(df, min_lift, top_n):
        return get_anchor_addon_matrix(df, min_lift=min_lift, top_n_per_anchor=top_n)

    with st.spinner("Computing add-on matrix..."):
        addon_matrix = get_addon_matrix_cached(
            transactions_df,
            params.get("min_lift", 1.2),
            params.get("top_n", 5),
        )

    if not addon_matrix.empty:
        addon_matrix = _enrich_addons_with_penetration(addon_matrix, basket_pen, product_lookup)
        addon_matrix["Anchor Name"] = addon_matrix["anchor_product"].map(product_lookup)
        addon_matrix["Add-on Name"] = addon_matrix["addon_product"].map(product_lookup)

        # Pivot for heatmap
        pivot = addon_matrix.pivot_table(
            index="Anchor Name", columns="Add-on Name", values="lift", fill_value=1.0
        )

        st.subheader("Add-on Lift Heatmap")
        st.caption("Symmetric lift: Anchor (rows) → Add-on (columns). **Associative only — not causal.**")

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="RdYlGn",
                zmid=1.0,
                zmin=0.5,
                zmax=min(float(pivot.values.max()), 10.0),
                text=pivot.values.round(2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
            )
        )
        fig.update_layout(
            title="Lift: Anchor (rows) → Add-on (columns)",
            height=600,
            xaxis_title="Add-on Product",
            yaxis_title="Anchor Product",
        )
        st.plotly_chart(fig, width="stretch")

        # Table view
        st.subheader("All Add-on Pairs — Ranked Evidence")
        display_cols = [
            "Anchor Name",
            "Add-on Name",
            "p_addon_given_anchor",
            "lift",
            "leverage",
            "revenue_uplift_per_anchor",
        ]
        # Add penetration columns if available
        pen_cols = [c for c in addon_matrix.columns if "basket_penetration" in c or "shopper_penetration" in c]
        display_cols.extend(pen_cols)
        available = [c for c in display_cols if c in addon_matrix.columns]

        st.dataframe(
            addon_matrix[available].round(4),
            width="stretch",
            hide_index=True,
        )

        # Disclaimer
        st.caption(
            "⚠️ **Interpretation**: Lift > 1 indicates co-occurrence above chance. "
            "Revenue uplift is associative (halo effect), not causal incrementality. "
            "No control for confounding factors (trip mission, shopper type, promotions)."
        )

        render_analytics_export(addon_matrix, "AddOn_Matrix")
    else:
        st.info("No add-on relationships found above lift threshold")


def _enrich_addons_with_penetration(addons: pd.DataFrame, basket_pen: pd.DataFrame, product_lookup: dict) -> pd.DataFrame:
    """Add basket penetration metrics to add-ons display."""
    if basket_pen is None or basket_pen.empty:
        return addons

    pen_lookup = basket_pen.set_index("stockcode")

    def get_pen(stockcode, col):
        try:
            return pen_lookup.loc[stockcode, col]
        except (KeyError, IndexError):
            return np.nan

    addons = addons.copy()
    # Add penetration for add-on product
    addons["basket_penetration"] = addons["addon_product"].apply(lambda x: get_pen(x, "basket_penetration"))
    addons["unique_shopper_penetration"] = addons["addon_product"].apply(lambda x: get_pen(x, "unique_shopper_penetration"))

    return addons
