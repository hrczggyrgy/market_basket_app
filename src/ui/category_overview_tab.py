"""Category Overview Tab - NLP-inferred category scorecard with quadrant, heatmap, PoP deltas, and sparklines."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.category_overview import (
    compute_category_scorecard,
    get_category_medians,
)
from src.analytics.sufficiency import assess_data_sufficiency, format_sufficiency_summary
from src.ui.export import render_analytics_export
from src.ui.tabs import persistent_tabs


def render_category_overview_tab(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict, pipeline: dict = None
):
    """Render Category Overview tab with persistent sub-tabs."""
    st.header(" Category Overview")
    st.caption(
        "NLP-inferred category taxonomy  |  Dunnhumby 4-role framework  |  "
        "PoP = Period-over-Period (4-week vs prior 4-week)"
    )

    if transactions_df.empty:
        st.warning("No transaction data available")
        return

    # Data sufficiency gate
    sufficiency = assess_data_sufficiency(transactions_df)
    with st.expander(" Data Sufficiency", expanded=sufficiency["overall"] != "robust"):
        st.markdown(format_sufficiency_summary(sufficiency))
        if sufficiency["overall"] == "insufficient":
            st.warning("Dataset may be too small for reliable category analysis.")
        elif sufficiency["overall"] == "directional":
            st.info("Category results should be treated as directional.")

    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        n_clusters = st.slider(
            "Number of Inferred Categories",
            4,
            15,
            params.get("n_categories", 8),
            key="cat_overview_n_clusters",
            help="KMeans clusters on TF-IDF product names. More = finer granularity.",
        )
    with col2:
        prior_weeks = st.slider(
            "PoP Prior Window (weeks)",
            2,
            8,
            params.get("prior_weeks", 4),
            key="cat_overview_prior_weeks",
            help="Compare last 4 weeks vs this many weeks before.",
        )
    with col3:
        if st.button(" Regenerate Categories", key="cat_overview_regen"):
            st.cache_data.clear()
            st.rerun()

    # Compute category scorecard (cached)
    @st.cache_data
    def get_scorecard_cached(df, n_clust):
        return compute_category_scorecard(df, n_clusters=n_clust)

    with st.spinner("Inferring categories and computing scorecard..."):
        cat_df = get_scorecard_cached(transactions_df, n_clusters)

    if cat_df.empty:
        st.warning("No category data available")
        return

    st.success(f"Inferred {len(cat_df)} categories")

    # Update quadrant medians for role classification
    x_med, y_med = get_category_medians(cat_df)

    # Persistent sub-tabs
    tab_labels = [
        " Role Quadrant",
        " KPI Scorecard",
        " PoP Deltas",
        " Revenue Sparklines",
    ]
    selected = persistent_tabs(tab_labels, "category_overview_main_tabs", default_tab=0)

    if selected == 0:
        _render_role_quadrant(cat_df, x_med, y_med)
    elif selected == 1:
        _render_kpi_scorecard(cat_df)
    elif selected == 2:
        _render_pop_deltas(cat_df, prior_weeks)
    elif selected == 3:
        _render_revenue_sparklines(cat_df)

    # Export
    render_analytics_export(cat_df.drop(columns=["weekly_revenue_series"], errors="ignore"), "Category_Scorecard")


def _render_role_quadrant(cat_df: pd.DataFrame, x_med: float, y_med: float):
    """V1 — Category Role Quadrant (Hero chart)."""
    st.subheader("Category Role Quadrant")
    st.caption(
        f"X = Shopper Penetration % (median: {x_med:.1f}%)  |  "
        f"Y = Basket Attachment % (median: {y_med:.1f}%)  |  "
        "Bubble size = Revenue  |  Color = Suggested Role"
    )

    # Role color mapping
    role_colors = {
        "Destination": "#2E7D32",  # Green
        "Routine": "#1565C0",      # Blue
        "Seasonal": "#FF8F00",     # Amber
        "Convenience": "#C62828",  # Red
    }

    fig = px.scatter(
        cat_df,
        x="shopper_penetration_pct",
        y="basket_attachment_rate_pct",
        size="revenue",
        color="suggested_role",
        text="category",
        title="Category Role Quadrant (Dunnhumby Framework)",
        labels={
            "shopper_penetration_pct": "Shopper Penetration % (Reach)",
            "basket_attachment_rate_pct": "Basket Attachment % (Trip Share)",
            "suggested_role": "Suggested Role",
            "revenue": "Revenue ($)",
        },
        color_discrete_map=role_colors,
        size_max=60,
        hover_data={
            "revenue": ":$,.0f",
            "shopper_penetration_pct": ":.1f",
            "basket_attachment_rate_pct": ":.1f",
            "sku_count": True,
            "promo_dependency_pct": ":.1f",
        },
    )

    # Quadrant lines at medians
    fig.add_hline(
        y=y_med,
        line_dash="dash",
        line_color="gray",
        line_width=1,
        annotation_text="Median Attachment",
        annotation_position="right",
    )
    fig.add_vline(
        x=x_med,
        line_dash="dash",
        line_color="gray",
        line_width=1,
        annotation_text="Median Penetration",
        annotation_position="top",
    )

    # Quadrant labels
    x_max = cat_df["shopper_penetration_pct"].max() * 1.1
    y_max = cat_df["basket_attachment_rate_pct"].max() * 1.1
    x_min = cat_df["shopper_penetration_pct"].min() * 0.9
    y_min = cat_df["basket_attachment_rate_pct"].min() * 0.9

    fig.add_annotation(
        x=x_max * 0.75, y=y_max * 0.9,
        text="<b>Destination</b><br>High Reach, High Trip Share",
        showarrow=False, font=dict(size=11, color="#2E7D32"), bgcolor="rgba(46,125,50,0.1)", borderpad=4,
    )
    fig.add_annotation(
        x=x_max * 0.75, y=y_min * 1.1,
        text="<b>Routine</b><br>High Reach, Low Trip Share",
        showarrow=False, font=dict(size=11, color="#1565C0"), bgcolor="rgba(21,101,192,0.1)", borderpad=4,
    )
    fig.add_annotation(
        x=x_min * 1.1, y=y_max * 0.9,
        text="<b>Seasonal</b><br>Low Reach, High Trip Share",
        showarrow=False, font=dict(size=11, color="#FF8F00"), bgcolor="rgba(255,143,0,0.1)", borderpad=4,
    )
    fig.add_annotation(
        x=x_min * 1.1, y=y_min * 1.1,
        text="<b>Convenience</b><br>Low Reach, Low Trip Share",
        showarrow=False, font=dict(size=11, color="#C62828"), bgcolor="rgba(198,40,40,0.1)", borderpad=4,
    )

    fig.update_layout(
        height=650,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Role summary table
    st.subheader("Role Summary")
    role_summary = (
        cat_df.groupby("suggested_role")
        .agg(
            categories=("category", "count"),
            total_revenue=("revenue", "sum"),
            avg_penetration=("shopper_penetration_pct", "mean"),
            avg_attachment=("basket_attachment_rate_pct", "mean"),
            avg_promo_dep=("promo_dependency_pct", "mean"),
        )
        .reset_index()
    )
    st.dataframe(
        role_summary.style.format(
            {
                "total_revenue": "${:,.0f}",
                "avg_penetration": "{:.1f}%",
                "avg_attachment": "{:.1f}%",
                "avg_promo_dep": "{:.1f}%",
            }
        ).background_gradient(cmap="RdYlGn", subset=["total_revenue"]),
        use_container_width=True,
    )


def _render_kpi_scorecard(cat_df: pd.DataFrame):
    """V2 — KPI Scorecard Heatmap."""
    st.subheader("KPI Scorecard Heatmap")
    st.caption(
        "Cells = normalized KPI (0-1 scale, Green=high, Red=low). "
        "Hover for raw values. Sort by any column."
    )

    kpi_cols = [
        "revenue_share_pct",
        "shopper_penetration_pct",
        "basket_attachment_rate_pct",
        "avg_purchase_frequency",
        "promo_dependency_pct",
        "sku_count",
        "hhi_concentration",
    ]
    kpi_labels = {
        "revenue_share_pct": "Revenue Share %",
        "shopper_penetration_pct": "Shopper Pen %",
        "basket_attachment_rate_pct": "Basket Attach %",
        "avg_purchase_frequency": "Avg Frequency",
        "promo_dependency_pct": "Promo Dep %",
        "sku_count": "SKU Count",
        "hhi_concentration": "HHI",
    }

    # Normalize each column 0-1 (min-max)
    norm_df = cat_df[["category"] + kpi_cols].copy()
    for col in kpi_cols:
        mn, mx = norm_df[col].min(), norm_df[col].max()
        if mx > mn:
            norm_df[col] = (norm_df[col] - mn) / (mx - mn)
        else:
            norm_df[col] = 0.5

    # Heatmap
    fig = go.Figure(go.Heatmap(
        z=norm_df[kpi_cols].values,
        x=[kpi_labels[c] for c in kpi_cols],
        y=norm_df["category"].tolist(),
        colorscale="RdYlGn",
        text=cat_df[kpi_cols].round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="Category: %{y}<br>KPI: %{x}<br>Value: %{text}<extra></extra>",
        colorbar=dict(title="Normalized", tickvals=[0, 0.5, 1], ticktext=["Low", "Med", "High"]),
    ))
    fig.update_layout(
        height=400 + len(cat_df) * 25,
        xaxis=dict(tickangle=30),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw values table
    with st.expander(" View Raw KPI Values", expanded=False):
        display_cols = ["category"] + kpi_cols
        st.dataframe(
            cat_df[display_cols].style.format(
                {
                    "revenue_share_pct": "{:.1f}%",
                    "shopper_penetration_pct": "{:.1f}%",
                    "basket_attachment_rate_pct": "{:.1f}%",
                    "avg_purchase_frequency": "{:.2f}",
                    "promo_dependency_pct": "{:.1f}%",
                    "sku_count": "{:.0f}",
                    "hhi_concentration": "{:.3f}",
                }
            ).background_gradient(cmap="RdYlGn", subset=kpi_cols),
            use_container_width=True,
        )


def _render_pop_deltas(cat_df: pd.DataFrame, prior_weeks: int):
    """V3 — Period-over-Period KPI Delta Panel."""
    st.subheader(f"PoP KPI Deltas (Last 4 Weeks vs Prior {prior_weeks} Weeks)")

    # Top-level metric cards
    col1, col2, col3, col4 = st.columns(4)

    total_revenue = cat_df["revenue"].sum()
    # Recompute total revenue for current vs prior to get overall delta
    # This is a simplified view - in practice would use weekly aggregation
    avg_pen = cat_df["shopper_penetration_pct"].mean()
    avg_freq = cat_df["avg_purchase_frequency"].mean()
    avg_promo = cat_df["promo_dependency_pct"].mean()

    with col1:
        st.metric(
            "Total Revenue",
            f"${total_revenue:,.0f}",
            delta=f"{cat_df['revenue_mom_delta_pct'].mean():.1f}%",
            delta_color="normal",
        )
    with col2:
        st.metric(
            "Avg Shopper Penetration",
            f"{avg_pen:.1f}%",
            delta=None,
        )
    with col3:
        st.metric(
            "Avg Purchase Frequency",
            f"{avg_freq:.2f}",
            delta=None,
        )
    with col4:
        st.metric(
            "Avg Promo Dependency",
            f"{avg_promo:.1f}%",
            delta=None,
        )

    st.divider()

    # Per-category delta table
    st.subheader("Per-Category PoP Deltas")

    delta_cols = [
        "category",
        "revenue",
        "revenue_mom_delta_pct",
        "shopper_penetration_pct",
        "basket_attachment_rate_pct",
        "avg_purchase_frequency",
        "promo_dependency_pct",
        "sku_count",
        "suggested_role",
        "rag_status",
    ]

    # Color mapping for RAG
    def rag_color(val):
        if val == "Green":
            return "background-color: #C8E6C9"
        elif val == "Red":
            return "background-color: #FFCDD2"
        elif val == "Amber":
            return "background-color: #FFE0B2"
        return ""

    styled_df = (
        cat_df[delta_cols]
        .style.format(
            {
                "revenue": "${:,.0f}",
                "revenue_mom_delta_pct": "{:+.1f}%",
                "shopper_penetration_pct": "{:.1f}%",
                "basket_attachment_rate_pct": "{:.1f}%",
                "avg_purchase_frequency": "{:.2f}",
                "promo_dependency_pct": "{:.1f}%",
                "sku_count": "{:.0f}",
            }
        )
        .applymap(rag_color, subset=["rag_status"])
        .background_gradient(cmap="RdYlGn", subset=["revenue_mom_delta_pct"])
    )
    st.dataframe(styled_df, use_container_width=True)

    # Delta waterfall for top categories
    st.subheader("Revenue Delta Waterfall (Top 8 Categories)")
    top_cats = cat_df.nlargest(8, "revenue")[["category", "revenue", "revenue_mom_delta_pct"]].copy()
    top_cats["delta_revenue"] = top_cats["revenue"] * top_cats["revenue_mom_delta_pct"] / 100

    fig = go.Figure(go.Waterfall(
        name="Revenue",
        orientation="v",
        measure=["relative"] * len(top_cats) + ["total"],
        x=top_cats["category"].tolist() + ["Total"],
        y=top_cats["delta_revenue"].tolist() + [top_cats["delta_revenue"].sum()],
        text=[f"{v:+,.0f}" for v in top_cats["delta_revenue"].tolist()] + [f"{top_cats['delta_revenue'].sum():+,.0f}"],
        textposition="outside",
        connector=dict(line=dict(color="rgb(63, 63, 63)")),
        increasing=dict(marker=dict(color="#2E7D32")),
        decreasing=dict(marker=dict(color="#C62828")),
        totals=dict(marker=dict(color="#1565C0")),
    ))
    fig.update_layout(
        title="Revenue Change vs Prior Period",
        yaxis_title="Delta Revenue ($)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_revenue_sparklines(cat_df: pd.DataFrame):
    """V4 — Revenue Trend Sparklines Table."""
    st.subheader("Weekly Revenue Sparklines (Last 12 Weeks)")
    st.caption("Sort by any column. Sparklines show 12-week revenue trend per category.")

    # Prepare display dataframe
    display_df = cat_df[["category", "revenue", "revenue_share_pct", "shopper_penetration_pct",
                          "basket_attachment_rate_pct", "suggested_role", "rag_status",
                          "weekly_revenue_series"]].copy()

    # Sort by revenue by default
    display_df = display_df.sort_values("revenue", ascending=False).reset_index(drop=True)

    # Configure column types for st.dataframe
    column_config = {
        "category": st.column_config.TextColumn("Category", width="medium"),
        "revenue": st.column_config.NumberColumn("Revenue ($)", format="$%,.0f", width="medium"),
        "revenue_share_pct": st.column_config.NumberColumn("Rev Share %", format="%.1f%%", width="small"),
        "shopper_penetration_pct": st.column_config.NumberColumn("Shopper Pen %", format="%.1f%%", width="small"),
        "basket_attachment_rate_pct": st.column_config.NumberColumn("Basket Attach %", format="%.1f%%", width="small"),
        "suggested_role": st.column_config.TextColumn("Role", width="small"),
        "rag_status": st.column_config.TextColumn("RAG", width="small"),
        "weekly_revenue_series": st.column_config.LineChartColumn(
            "12-Week Trend",
            width="large",
            y_min=0,
            y_max=display_df["weekly_revenue_series"].apply(lambda x: max(x) if x else 0).max(),
        ),
    }

    st.dataframe(
        display_df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
    )

    # Detail view for selected category
    st.divider()
    selected_cat = st.selectbox(
        "Drill-down: Select Category for Weekly Detail",
        display_df["category"].tolist(),
        key="cat_overview_drilldown",
    )

    if selected_cat:
        cat_data = cat_df[cat_df["category"] == selected_cat].iloc[0]
        weekly_rev = cat_data["weekly_revenue_series"]

        if weekly_rev:
            weeks = list(range(1, len(weekly_rev) + 1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=weeks, y=weekly_rev,
                mode="lines+markers",
                name="Revenue",
                line=dict(color="#1565C0", width=2),
                marker=dict(size=6),
                fill="tozeroy",
                fillcolor="rgba(21, 101, 192, 0.1)",
            ))
            fig.update_layout(
                title=f"Weekly Revenue: {selected_cat}",
                xaxis_title="Week (1 = 12 weeks ago, 12 = last week)",
                yaxis_title="Revenue ($)",
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Weekly stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Weekly Revenue", f"${sum(weekly_rev)/len(weekly_rev):,.0f}")
            with col2:
                st.metric("Peak Week", f"${max(weekly_rev):,.0f}")
            with col3:
                st.metric("Volatility (CV)", f"{(pd.Series(weekly_rev).std()/pd.Series(weekly_rev).mean()*100):.1f}%")