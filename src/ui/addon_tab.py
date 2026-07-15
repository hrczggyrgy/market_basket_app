"""Add-on analysis tab."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.addon import get_addon_recommendations, get_anchor_addon_matrix
from src.ui.export import render_analytics_export


def render_addon_tab(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render add-on analysis tab."""
    st.header("➕ Add-on / Complementary Products")
    st.caption(
        "Finds products that are bought **alongside** an anchor item in the same basket. "
        "**Revenue uplift per anchor** estimates incremental value per transaction."
    )

    if transactions_df.empty:
        st.warning("No transaction data available")
        return

    # Mode selection
    mode = st.radio(
        "Analysis Mode",
        ["Single Anchor Product", "Multiple Anchors (Top Products)"],
        horizontal=True,
        key="addon_mode_radio",
    )

    if mode == "Single Anchor Product":
        render_single_addon(transactions_df, product_lookup, params)
    else:
        render_multi_addon(transactions_df, product_lookup, params)


def render_single_addon(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
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
            addons["Add-on Name"] = addons["addon_product"].map(product_lookup)
            addons["Anchor Name"] = addons["anchor_product"].map(product_lookup)

            # Best revenue add-on callout
            if "revenue_uplift_per_anchor" in addons.columns:
                best_addon = addons.nlargest(1, "revenue_uplift_per_anchor").iloc[0]
                st.success(
                    f"💰 **Highest revenue add-on:** `{best_addon['Add-on Name']}`  \n"
                    f"Expected uplift **${best_addon['revenue_uplift_per_anchor']:.2f}** per anchor transaction · "
                    f"Lift **{best_addon['lift']:.2f}**"
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
            available = [c for c in display_cols if c in addons.columns]

            st.dataframe(addons[available].round(4), width="stretch", hide_index=True)

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


def render_multi_addon(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
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
        addon_matrix["Anchor Name"] = addon_matrix["anchor_product"].map(product_lookup)
        addon_matrix["Add-on Name"] = addon_matrix["addon_product"].map(product_lookup)

        # Pivot for heatmap
        pivot = addon_matrix.pivot_table(
            index="Anchor Name", columns="Add-on Name", values="lift", fill_value=1.0
        )

        st.subheader("Add-on Lift Heatmap")

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
        st.subheader("All Add-on Pairs")
        display_cols = [
            "Anchor Name",
            "Add-on Name",
            "p_addon_given_anchor",
            "lift",
            "leverage",
            "revenue_uplift_per_anchor",
        ]
        st.dataframe(
            addon_matrix[display_cols].round(4),
            width="stretch",
            hide_index=True,
        )

        render_analytics_export(addon_matrix, "AddOn_Matrix")
    else:
        st.info("No add-on relationships found above lift threshold")
