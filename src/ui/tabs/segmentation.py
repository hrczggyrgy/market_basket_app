"""Customer Segmentation tab — the customer-intelligence hub.

Page pattern: segment scorecard -> segment landscape (value x frequency) ->
differentiation (radar) -> migration -> full tables.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.segmentation import (
    behavioral_segmentation,
    compute_rfm_features,
    compute_segment_migration,
    compute_segment_radar,
    rfm_segmentation,
    value_based_segmentation,
    calculate_segment_value_metrics,
    calculate_segment_engagement_metrics,
    calculate_segment_retention_metrics,
    calculate_segment_basket_metrics,
    calculate_segment_price_behavior_metrics,
    calculate_segment_growth_metrics,
    calculate_segment_concentration_metrics,
    calculate_segment_stability_score,
    calculate_segment_distinctiveness,
)
from src.ui.plots import PALETTE, empty_state, new_fig, show
from src.ui.registry import ModeSpec


def _segment_landscape(seg: pd.DataFrame, value_col: str, label_col: str) -> None:
    """Value x frequency landscape sized by customer count."""
    agg = (
        seg.groupby(label_col)
        .agg(
            customers=(seg.columns[0], "count"),
            total_value=(value_col, "sum"),
            avg_frequency=(value_col, "mean"),
        )
        .reset_index()
    )
    if agg.empty or len(agg) < 2:
        show(empty_state("Not enough segments to map"))
        return

    fig = px.scatter(
        agg,
        x="avg_frequency",
        y="total_value",
        size="customers",
        color=label_col,
        hover_data=["customers"],
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(
        xaxis={"title": "Avg value per customer"},
        yaxis={"title": "Total segment value"},
        height=420,
    )
    show(fig)
    st.caption(
        "Segment landscape: x = per-customer value, y = total segment value, "
        "size = customer count. Big + far-right segments are the value engine; "
        "large + far-left segments are the growth pool."
    )


def _segment_revenue_share(seg: pd.DataFrame, value_col: str, label_col: str) -> None:
    """Share of total value per segment."""
    shares = (
        seg.groupby(label_col)[value_col]
        .sum()
        .sort_values(ascending=False)
        .pipe(lambda s: s / s.sum() * 100)
    )
    fig = go.Figure(
        data=[
            go.Pie(
                labels=shares.index,
                values=shares.values,
                hole=0.4,
                marker={"colors": PALETTE[: len(shares)]},
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(height=320)
    show(fig)
    st.caption("Share of total customer value per segment. A concentrated pie = the base is fragile.")


def _segment_radar(df: pd.DataFrame) -> None:
    st.subheader(":material/radar: Segment Differentiation (Radar)")
    radar = compute_segment_radar(df, n_clusters=4)
    if radar.empty:
        st.caption("Not enough distinct segments to profile (fewer than 2 non-trivial clusters).")
        return

    features = list(radar["feature"].unique())
    segments = list(radar["segment"].unique())

    fig = go.Figure()
    for i, seg in enumerate(segments):
        row = radar[radar["segment"] == seg].set_index("feature")["normalized_value"]
        values = [row[f] for f in features]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=features + [features[0]],
                fill="toself",
                opacity=0.25,
                name=seg,
                line={"color": PALETTE[i % len(PALETTE)]},
            )
        )

    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        height=440,
        margin={"t": 30, "b": 30},
    )
    show(fig)
    st.caption(
        "Segment profile vs all segments (min-max normalized per axis, 0-1). "
        "Wide petals = the segment over-indexes on that behavior."
    )


def _segment_migration(df: pd.DataFrame) -> None:
    st.subheader(":material/swap_horiz: Segment Migration (first vs second half)")
    migration = compute_segment_migration(df, n_clusters=4)
    if migration.empty:
        st.caption("Not enough stable segments to trace migration (customers must be present in both halves).")
        return

    stay = migration[migration["segment_from"] == migration["segment_to"]]
    move = migration[migration["segment_from"] != migration["segment_to"]]

    top = move.nlargest(20, "customers")
    sources = top["segment_from"].tolist()
    targets = top["segment_to"].tolist()
    values = top["customers"].tolist()

    nodes = list(dict.fromkeys(sources + targets))
    label_to_idx = {s: i for i, s in enumerate(nodes)}
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node={
                    "label": [f"First half: {n}" for n in nodes],
                    "color": PALETTE[0],
                    "pad": 15,
                    "thickness": 20,
                },
                link={
                    "source": [label_to_idx[s] for s in sources],
                    "target": [label_to_idx[t] for t in targets],
                    "value": values,
                    "color": "rgba(255, 140, 0, 0.4)",
                },
            )
        ]
    )
    fig.update_layout(height=max(400, 30 * len(nodes)), font={"size": 10})
    show(fig)

    moved = int(move["customers"].sum())
    stayed = int(stay["customers"].sum())
    st.caption(
        f"{moved:,} customers changed segment between halves vs {stayed:,} who stayed. "
        "Migration toward higher-value segments = retention working; the reverse = value leaking."
    )


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/groups: Customer Segmentation")
    st.caption(
        "Segmentation is the customer-intelligence hub: it explains WHO your "
        "revenue depends on, and WHERE value is being won or lost."
    )

    tab1, tab2, tab3, tab4 = st.tabs(["RFM", "Behavioral", "Value-Based", "Strategic Metrics"])

    with tab1:
        st.subheader(":material/table_chart: RFM Segments")
        rfm = compute_rfm_features(df)
        seg = rfm_segmentation(rfm, method="kmeans", n_segments=5)
        _segment_revenue_share(seg, "monetary", "segment")
        _segment_landscape(seg, "monetary", "segment")
        
        # Show enhanced RFM metrics
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(seg[["customer_id", "recency_days", "frequency", "monetary", "segment"]],
                         use_container_width=True, hide_index=True)
        with col2:
            st.subheader("Segment Summary")
            if not seg.empty:
                segment_summary = seg.groupby('segment').agg(
                    count=('customer_id', 'count'),
                    avg_recency=('recency_days', 'mean'),
                    avg_frequency=('frequency', 'mean'),
                    avg_monetary=('monetary', 'mean')
                ).round(2)
                st.dataframe(segment_summary, use_container_width=True)

    with tab2:
        st.subheader(":material/psychology: Behavioral Segments")
        behav = behavioral_segmentation(df, n_clusters=4)
        if isinstance(behav, tuple):
            behav = behav[0]
        _segment_revenue_share(behav, "total_revenue", "segment")
        _segment_landscape(behav, "total_revenue", "segment")
        _segment_radar(df)
        _segment_migration(df)
        
        # Show enhanced behavioral metrics
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(behav[["customer_id", "cluster", "segment", "cluster_confidence"]],
                         use_container_width=True, hide_index=True)
        with col2:
            st.subheader("Segment Summary")
            if not behav.empty:
                segment_summary = behav.groupby('segment').agg(
                    count=('customer_id', 'count'),
                    avg_frequency=('purchase_frequency', 'mean'),
                    avg_revenue=('total_revenue', 'mean'),
                    avg_products=('n_products', 'mean')
                ).round(2)
                st.dataframe(segment_summary, use_container_width=True)

    with tab3:
        st.subheader(":material/attach_money: Value-Based Segments")
        val = value_based_segmentation(df)
        if isinstance(val, tuple):
            val = val[0]
        _segment_revenue_share(val, "predicted_clv", "value_segment")
        _segment_landscape(val, "predicted_clv", "value_segment")
        
        # Show enhanced value-based metrics
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(val[["customer_id", "recency", "frequency", "monetary", "predicted_clv", "value_segment"]],
                         use_container_width=True, hide_index=True)
        with col2:
            st.subsegment("Segment Summary")
            if not val.empty:
                segment_summary = val.groupby('value_segment').agg(
                    count=('customer_id', 'count'),
                    avg_clv=('predicted_clv', 'mean'),
                    avg_frequency=('frequency', 'mean'),
                    avg_monetary=('monetary', 'mean')
                ).round(2)
                st.dataframe(segment_summary, use_container_width=True)

    with tab4:
        st.subheader(":material/insights: Strategic Segment Metrics")
        
        # Calculate all strategic metrics
        try:
            # Get the most recent segmentation for reference
            behav = behavioral_segmentation(df, n_clusters=4)
            if isinstance(behav, tuple):
                behav = behav[0]
            
            if behav.empty or 'segment' not in behav.columns:
                st.warning("No segments available for strategic metrics calculation.")
                return
                
            segments_df = behav[['customer_id', 'segment']].copy()
            
            # Calculate and display strategic metrics in expandable sections
            with st.expander("���💰 Segment Value & Economic Importance", expanded=True):
                value_metrics = calculate_segment_value_metrics(segments_df, df)
                if not value_metrics.empty:
                    # Format for display
                    display_metrics = value_metrics.copy()
                    display_metrics['revenue'] = display_metrics['revenue'].apply(lambda x: f"€{x:,.0f}")
                    display_metrics['revenue_per_customer'] = display_metrics['revenue_per_customer'].apply(lambda x: f"€{x:,.0f}")
                    display_metrics['revenue_per_transaction'] = display_metrics['revenue_per_transaction'].apply(lambda x: f"€{x:,.0f}")
                    st.dataframe(display_metrics[[
                        'segment', 'customers', 'customer_share_pct', 'revenue', 'revenue_share_pct',
                        'value_concentration_index'
                    ]], use_container_width=True)
                else:
                    st.info("No value metrics available")

            with st.expander("���👥 Customer Engagement", expanded=True):
                engagement_metrics = calculate_segment_engagement_metrics(segments_df, df)
                if not engagement_metrics.empty:
                    display_metrics = engagement_metrics.copy()
                    display_metrics['active_customer_rate_pct'] = display_metrics['active_customer_rate_pct'].apply(lambda x: f"{x:.1f}%")
                    display_metrics['dormancy_rate_pct'] = display_metrics['dormancy_rate_pct'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(display_metrics[[
                        'segment', 'purchase_frequency', 'recency_days', 'active_customer_rate_pct', 'dormancy_rate_pct'
                    ]], use_container_width=True)
                else:
                    st.info("No engagement metrics available")

            with st.expander("���🔄 Retention & Lifecycle", expanded=True):
                retention_metrics = calculate_segment_retention_metrics(df, segments_df)
                if not retention_metrics.empty:
                    display_metrics = retention_metrics.copy()
                    display_metrics['repeat_purchase_rate_pct'] = display_metrics['repeat_purchase_rate_pct'].apply(lambda x: f"{x:.1f}%")
                    display_metrics['lapse_rate_pct'] = display_metrics['lapse_rate_pct'].apply(lambda x: f"{x:.1f}%")
                    display_metrics['reactivation_rate_pct'] = display_metrics['reactivation_rate_pct'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(display_metrics[[
                        'segment', 'customer_lifetime_days', 'repeat_purchase_rate_pct', 'lapse_rate_pct', 'reactivation_rate_pct'
                    ]], use_container_width=True)
                else:
                    st.info("No retention metrics available")

            with st.expander("���🛒 Basket Economics", expanded=True):
                basket_metrics = calculate_segment_basket_metrics(segments_df, df)
                if not basket_metrics.empty:
                    display_metrics = basket_metrics.copy()
                    display_metrics['avg_basket_value'] = display_metrics['avg_basket_value'].apply(lambda x: f"€{x:,.2f}")
                    display_metrics['large_basket_rate_pct'] = display_metrics['large_basket_rate_pct'].apply(lambda x: f"{x:.1f}%")
                    display_metrics['small_basket_rate_pct'] = display_metrics['small_basket_rate_pct'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(display_metrics[[
                        'segment', 'avg_basket_value', 'avg_units_per_basket', 'avg_skus_per_basket',
                        'large_basket_rate_pct', 'small_basket_rate_pct'
                    ]], use_container_width=True)
                else:
                    st.info("No basket metrics available")

            with st.expander("���💵 Price Behavior", expanded=True):
                price_metrics = calculate_segment_price_behavior_metrics(segments_df, df)
                if not price_metrics.empty:
                    display_metrics = price_metrics.copy()
                    display_metrics['avg_price_paid'] = display_metrics['avg_price_paid'].apply(lambda x: f"€{x:.2f}")
                    display_metrics['price_orientation_index'] = display_metrics['price_orientation_index'].apply(lambda x: f"{x:+.1f}")
                    display_metrics['premium_product_share_pct'] = display_metrics['premium_product_share_pct'].apply(lambda x: f"{x:.1f}%")
                    display_metrics['value_product_share_pct'] = display_metrics['value_product_share_pct'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(display_metrics[[
                        'segment', 'avg_price_paid', 'price_index_vs_overall', 'price_orientation_index',
                        'premium_product_share_pct', 'value_product_share_pct'
                    ]], use_container_width=True)
                else:
                    st.info("No price behavior metrics available")

            with st.expander("���📈 Growth & Momentum", expanded=True):
                growth_metrics = calculate_segment_growth_metrics(df, segments_df)
                if not growth_metrics.empty:
                    display_metrics = growth_metrics.copy()
                    display_metrics['revenue_growth_pct'] = display_metrics['revenue_growth_pct'].apply(lambda x: f"{x:+.1f}%")
                    display_metrics['customer_growth_pct'] = display_metrics['customer_growth_pct'].apply(lambda x: f"{x:+.1f}%")
                    display_metrics['frequency_growth_pct'] = display_metrics['frequency_growth_pct'].apply(lambda x: f"{x:+.1f}%")
                    display_metrics['avg_order_value_growth_pct'] = display_metrics['avg_order_value_growth_pct'].apply(lambda x: f"{x:+.1f}%")
                    display_metrics['segment_share_change_pct'] = display_metrics['segment_share_change_pct'].apply(lambda x: f"{x:+.1f}%")
                    st.dataframe(display_metrics[[
                        'segment', 'revenue_growth_pct', 'customer_growth_pct', 'frequency_growth_pct',
                        'avg_order_value_growth_pct', 'segment_share_change_pct'
                    ]], use_container_width=True)
                else:
                    st.info("No growth metrics available")

            with st.expander("���🎯 Customer Concentration", expanded=True):
                concentration_metrics = calculate_segment_concentration_metrics(segments_df, df)
                if not concentration_metrics.empty:
                    display_metrics = concentration_metrics.copy()
                    display_metrics['top_10pct_revenue_share'] = display_metrics['top_10pct_revenue_share'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(display_metrics[[
                        'segment', 'revenue_concentration_gini', 'top_10pct_revenue_share'
                    ]], use_container_width=True)
                else:
                    st.info("No concentration metrics available")

            with st.expander("��✨ Segment Quality", expanded=True):
                stability_metrics = calculate_segment_stability_score(segments_df, df)
                distinctiveness_metrics = calculate_segment_distinctiveness(segments_df, df)
                
                if not stability_metrics.empty:
                    display_stability = stability_metrics.copy()
                    display_stability['stability_score'] = display_stability['stability_score'].apply(lambda x: f"{x:.1f}")
                    st.dataframe(display_stability[[
                        'segment', 'customer_count', 'stability_score', 'evidence_level'
                    ]], use_container_width=True)
                
                if not distinctiveness_metrics.empty:
                    display_distinct = distinctiveness_metrics.copy()
                    display_distinct['distinctiveness_score'] = display_distinct['distinctiveness_score'].apply(lambda x: f"{x:.1f}")
                    st.dataframe(display_distinct[[
                        'segment', 'distinctiveness_score', 'defining_characteristics'
                    ]], use_container_width=True)
                    
                if stability_metrics.empty and distinctiveness_metrics.empty:
                    st.info("No quality metrics available")
                    
        except Exception as e:
            st.error(f"Error calculating strategic metrics: {str(e)}")
            st.info("This may be due to insufficient data for some calculations.")


MODE_SPEC: ModeSpec = ModeSpec(
    key="segmentation",
    label="Segmentation",
    icon=":material/groups:",
    handler=render,
)
