"""Customer Segmentation Tab - RFM, Behavioral, Value-based with persistent tab state."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.segmentation import (
    behavioral_segmentation,
    compute_rfm_features,
    rfm_segmentation,
    value_based_segmentation,
)
from src.ui.export import render_analytics_export
from src.ui.tabs import persistent_tabs


def render_segmentation_tab(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict
):
    """Render customer segmentation analysis tab with persistent sub-tabs."""
    # product_lookup is available but not used in segmentation analysis
    # Kept for consistency with other tab functions
    st.header("👥 Customer Segmentation")

    if transactions_df.empty:
        st.warning("No transaction data available")
        return

    # Persistent sub-tabs for different segmentation methods
    tab_labels = ["📊 RFM Segmentation", "🧠 Behavioral", "💰 Value-Based (CLV)"]
    selected = persistent_tabs(tab_labels, "segmentation_main_tabs", default_tab=0)

    if selected == 0:
        render_rfm_segmentation(transactions_df, params)
    elif selected == 1:
        render_behavioral_segmentation(transactions_df, params)
    elif selected == 2:
        render_value_segmentation(transactions_df, params)


def render_rfm_segmentation(transactions_df: pd.DataFrame, params: dict):
    """Render RFM segmentation analysis."""
    st.subheader("RFM-Based Segmentation")

    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        method = st.radio(
            "Segmentation Method",
            ["Quantile (Classic RFM)", "K-Means Clustering"],
            key="seg_tab_rfm_method",
        )
    with col2:
        if method == "K-Means Clustering":
            n_segments = st.slider(
                "Number of Segments", 3, 12, 8, key="seg_tab_n_segments"
            )
        else:
            n_segments = 8

    # Compute RFM
    with st.spinner("Computing RFM features..."):
        rfm = compute_rfm_features(transactions_df)

    st.success(f"Computed RFM for {len(rfm)} customers")

    # Persistent sub-tabs for RFM methods
    rfm_tabs = ["📊 Segment Distribution", "📈 Revenue Analysis", "🎯 3D Visualization", "📋 Profiles"]
    rfm_selected = persistent_tabs(rfm_tabs, "rfm_view_tabs", default_tab=0)

    if method == "Quantile (Classic RFM)":
        # Classic RFM scoring
        rfm_scored = rfm_segmentation(rfm, method="quantile")

        if rfm_selected == 0:
            _render_rfm_segment_distribution(rfm_scored)
        elif rfm_selected == 1:
            _render_rfm_revenue_analysis(rfm_scored)
        elif rfm_selected == 2:
            _render_rfm_3d_visualization(rfm_scored)
        elif rfm_selected == 3:
            _render_rfm_profiles_table(rfm_scored)

        render_analytics_export(rfm_scored, "RFM_Segments")

    else:
        # K-Means clustering
        rfm_clustered = rfm_segmentation(rfm, method="kmeans", n_segments=n_segments)

        if rfm_selected == 0:
            _render_kmeans_segment_distribution(rfm_clustered)
        elif rfm_selected == 1:
            _render_kmeans_revenue_analysis(rfm_clustered)
        elif rfm_selected == 2:
            _render_kmeans_2d_visualization(rfm_clustered)
        elif rfm_selected == 3:
            _render_kmeans_segment_details(rfm_clustered)

        render_analytics_export(rfm_clustered, f"RFM_KMeans_Segments_{n_segments}")


def _render_rfm_segment_distribution(rfm_scored: pd.DataFrame):
    """Render RFM segment distribution."""
    st.subheader("RFM Segment Distribution")

    # Segment counts
    seg_counts = rfm_scored["segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            seg_counts,
            values="Customers",
            names="Segment",
            title="Customer Distribution by RFM Segment",
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        # Revenue by segment
        seg_rev = (
            rfm_scored.groupby("segment")
            .agg(
                customers=("customer_id", "count"),
                avg_recency=("recency_days", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_monetary=("monetary", "mean"),
                total_revenue=("monetary", "sum"),
            )
            .reset_index()
            .sort_values("total_revenue", ascending=False)
        )

        fig = px.bar(
            seg_rev,
            x="segment",
            y="total_revenue",
            color="customers",
            title="Revenue by RFM Segment",
            labels={
                "segment": "Segment",
                "total_revenue": "Revenue ($)",
                "customers": "Customers",
            },
        )
        st.plotly_chart(fig, width="stretch")


def _render_rfm_revenue_analysis(rfm_scored: pd.DataFrame):
    """Render RFM revenue analysis."""
    st.subheader("Revenue by Segment")

    seg_rev = (
        rfm_scored.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )

    fig = px.bar(
        seg_rev,
        x="segment",
        y="total_revenue",
        color="customers",
        title="Revenue by RFM Segment",
        labels={
            "segment": "Segment",
            "total_revenue": "Revenue ($)",
            "customers": "Customers",
        },
    )
    st.plotly_chart(fig, width="stretch")


def _render_rfm_3d_visualization(rfm_scored: pd.DataFrame):
    """Render 3D RFM visualization."""
    st.subheader("RFM 3D Visualization")
    fig = px.scatter_3d(
        rfm_scored,
        x="recency_days",
        y="frequency",
        z="monetary",
        color="segment",
        hover_data=["customer_id", "avg_order_value", "n_unique_products"],
        title="RFM Space Colored by Segment",
        labels={
            "recency_days": "Recency (days)",
            "frequency": "Frequency",
            "monetary": "Monetary ($)",
        },
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, width="stretch")


def _render_rfm_profiles_table(rfm_scored: pd.DataFrame):
    """Render RFM segment profiles table."""
    st.subheader("Segment Profiles")

    seg_rev = (
        rfm_scored.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )

    st.dataframe(
        seg_rev.style.format(
            {
                "avg_recency": "{:.1f}",
                "avg_frequency": "{:.1f}",
                "avg_monetary": "${:,.2f}",
                "total_revenue": "${:,.2f}",
            }
        ).background_gradient(cmap="RdYlGn", subset=["total_revenue"]),
        width="stretch",
    )


def _render_kmeans_segment_distribution(rfm_clustered: pd.DataFrame):
    """Render K-Means segment distribution."""
    st.subheader("K-Means Segment Distribution")

    seg_counts = rfm_clustered["segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            seg_counts,
            values="Customers",
            names="Segment",
            title="Customer Distribution by K-Means Segment",
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        cluster_profiles = (
            rfm_clustered.groupby("segment")
            .agg(
                n_customers=("customer_id", "count"),
                avg_recency=("recency_days", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_monetary=("monetary", "mean"),
                total_revenue=("monetary", "sum"),
                pct_revenue=(
                    "monetary",
                    lambda x: x.sum() / rfm_clustered["monetary"].sum() * 100,
                ),
            )
            .reset_index()
        )

        fig = px.bar(
            cluster_profiles,
            x="segment",
            y="total_revenue",
            color="n_customers",
            title="Revenue by K-Means Segment",
            labels={"segment": "Segment", "total_revenue": "Revenue ($)", "n_customers": "Customers"},
        )
        st.plotly_chart(fig, width="stretch")


def _render_kmeans_revenue_analysis(rfm_clustered: pd.DataFrame):
    """Render K-Means revenue analysis."""
    st.subheader("Segment Profiles")

    cluster_profiles = (
        rfm_clustered.groupby("segment")
        .agg(
            n_customers=("customer_id", "count"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
            pct_revenue=(
                "monetary",
                lambda x: x.sum() / rfm_clustered["monetary"].sum() * 100,
            ),
        )
        .reset_index()
    )

    st.dataframe(
        cluster_profiles.style.format(
            {
                "avg_recency": "{:.1f}",
                "avg_frequency": "{:.1f}",
                "avg_monetary": "${:,.2f}",
                "total_revenue": "${:,.2f}",
                "pct_revenue": "{:.1f}%",
            }
        ).background_gradient(cmap="RdYlGn", subset=["total_revenue", "pct_revenue"]),
        width="stretch",
    )


def _render_kmeans_2d_visualization(rfm_clustered: pd.DataFrame):
    """Render 2D K-Means visualization."""
    st.subheader("Segment Visualization")
    fig = px.scatter(
        rfm_clustered,
        x="recency_days",
        y="monetary",
        color="segment",
        size="frequency",
        hover_data=["customer_id", "frequency", "avg_order_value"],
        title="Customer Segments: Recency vs Monetary",
        labels={"recency_days": "Recency (days)", "monetary": "Monetary ($)"},
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")


def _render_kmeans_segment_details(rfm_clustered: pd.DataFrame):
    """Render K-Means segment details."""
    st.subheader("Segment Details")

    selected_segment = st.selectbox(
        "View Segment Details",
        rfm_clustered["segment"].unique(),
        key="seg_tab_rfm_detail",
    )
    segment_data = rfm_clustered[rfm_clustered["segment"] == selected_segment]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Customers", len(segment_data))
    with col2:
        st.metric("Avg Recency", f"{segment_data['recency_days'].mean():.1f} days")
    with col3:
        st.metric("Avg Frequency", f"{segment_data['frequency'].mean():.1f} orders")
    with col4:
        st.metric("Avg Monetary", f"${segment_data['monetary'].mean():,.2f}")

    col5, col6 = st.columns(2)
    with col5:
        st.metric("Total Revenue", f"${segment_data['monetary'].sum():,.2f}")
    with col6:
        st.metric(
            "Revenue Share",
            f"{segment_data['monetary'].sum() / rfm_clustered['monetary'].sum() * 100:.1f}%",
        )


def render_behavioral_segmentation(transactions_df: pd.DataFrame, params: dict):
    """Render behavioral segmentation analysis with persistent sub-tabs."""
    st.subheader("Behavioral Segmentation")

    n_clusters = st.slider(
        "Number of Clusters", 3, 10, 6, key="seg_tab_behav_n_clusters"
    )

    with st.spinner("Computing behavioral segments..."):
        behavioral = behavioral_segmentation(transactions_df, n_clusters=n_clusters)

    # Persistent sub-tabs
    behav_tabs = ["📊 Profiles", "🎯 Radar Chart", "📦 Box Plots", "📈 Revenue"]
    behav_selected = persistent_tabs(behav_tabs, "behavioral_view_tabs", default_tab=0)

    # Feature columns
    feature_cols = [
        c for c in behavioral.columns if c not in ["customer_id", "cluster", "segment"]
    ]

    # Segment profiles
    profiles = behavioral.groupby("segment")[feature_cols].mean().round(2)
    profiles = profiles.T
    profiles.columns.name = None

    if behav_selected == 0:
        _render_behavioral_profiles(profiles)
    elif behav_selected == 1:
        _render_behavioral_radar(profiles, key_features=feature_cols)
    elif behav_selected == 2:
        _render_behavioral_box_plots(behavioral, feature_cols)
    elif behav_selected == 3:
        _render_behavioral_revenue(transactions_df, behavioral)

    render_analytics_export(behavioral, "Behavioral_Segments")


def _render_behavioral_profiles(profiles: pd.DataFrame):
    """Render behavioral segment profiles."""
    st.subheader("Behavioral Segment Profiles")

    st.dataframe(
        profiles.style.background_gradient(cmap="RdYlGn", axis=1),
        width="stretch",
    )


def _render_behavioral_radar(profiles: pd.DataFrame, key_features: list):
    """Render radar chart for segment comparison."""
    st.subheader("Segment Comparison (Key Metrics)")

    # Select key differentiating features
    selected_features = [
        "total_revenue",
        "purchase_frequency",
        "avg_order_value",
        "n_products",
        "avg_days_between",
        "weekend_ratio",
    ]
    selected_features = [f for f in selected_features if f in profiles.index]

    if selected_features:
        radar_data = profiles.loc[selected_features]

        fig = go.Figure()
        for segment in radar_data.columns:
            fig.add_trace(
                go.Scatterpolar(
                    r=radar_data[segment].values,
                    theta=radar_data.index,
                    fill="toself",
                    name=segment,
                )
            )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, radar_data.max().max()])
            ),
            showlegend=True,
            title="Segment Comparison (Normalized Metrics)",
        )
        st.plotly_chart(fig, width="stretch")


def _render_behavioral_box_plots(behavioral: pd.DataFrame, feature_cols: list):
    """Render box plots for key differentiators."""
    st.subheader("Key Differentiators")

    diff_feature = st.selectbox(
        "Select Feature", feature_cols, key="seg_tab_behav_diff_feature"
    )

    fig = px.box(
        behavioral,
        x="segment",
        y=diff_feature,
        color="segment",
        title=f"{diff_feature} by Segment",
    )
    st.plotly_chart(fig, width="stretch")


def _render_behavioral_revenue(transactions_df: pd.DataFrame, behavioral: pd.DataFrame):
    """Render behavioral revenue analysis."""
    st.subheader("Revenue by Behavioral Segment")

    df = transactions_df.copy()
    df["revenue"] = df["price"] * df["quantity"]
    merged = df.merge(
        behavioral[["customer_id", "segment"]], on="customer_id", how="left"
    )
    seg_rev = (
        merged.groupby("segment")
        .agg(
            customers=("customer_id", "nunique"),
            revenue=("revenue", "sum"),
            avg_order=("revenue", "mean"),
        )
        .reset_index()
    )

    fig = px.bar(
        seg_rev,
        x="segment",
        y="revenue",
        color="customers",
        title="Revenue by Behavioral Segment",
    )
    st.plotly_chart(fig, width="stretch")


def render_value_segmentation(transactions_df: pd.DataFrame, params: dict):
    """Render value-based segmentation analysis with persistent sub-tabs."""
    st.subheader("Value-Based Segmentation (Predicted CLV)")

    horizon = st.slider(
        "Prediction Horizon (days)", 30, 365, 90, key="seg_tab_value_horizon"
    )

    with st.spinner("Computing value segments..."):
        value_segments = value_based_segmentation(
            transactions_df, prediction_horizon_days=horizon
        )

    # Persistent sub-tabs
    value_tabs = ["📊 Distribution", "💰 Revenue", "📋 Profiles", "🎯 CLV Accuracy"]
    value_selected = persistent_tabs(value_tabs, "value_view_tabs", default_tab=0)

    if value_selected == 0:
        _render_value_distribution(value_segments)
    elif value_selected == 1:
        _render_value_revenue(value_segments)
    elif value_selected == 2:
        _render_value_profiles(value_segments)
    elif value_selected == 3:
        _render_value_clv_accuracy(value_segments)

    render_analytics_export(value_segments, "Value_Segments")


def _render_value_distribution(value_segments: pd.DataFrame):
    """Render value segment distribution."""
    st.subheader("Value Segment Distribution")

    seg_counts = value_segments["value_segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            seg_counts,
            values="Customers",
            names="Segment",
            title="Value Segment Distribution",
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        seg_rev = (
            value_segments.groupby("value_segment")
            .agg(
                customers=("customer_id", "count"),
                total_revenue=("monetary", "sum"),
                avg_predicted_clv=("predicted_clv", "mean"),
                avg_recency=("recency", "mean"),
                avg_frequency=("frequency", "mean"),
            )
            .reset_index()
        )

        fig = px.bar(
            seg_rev,
            x="value_segment",
            y="total_revenue",
            color="customers",
            title="Revenue by Value Segment",
            labels={
                "value_segment": "Segment",
                "total_revenue": "Revenue ($)",
                "customers": "Customers",
            },
        )
        st.plotly_chart(fig, width="stretch")


def _render_value_revenue(value_segments: pd.DataFrame):
    """Render value segment revenue."""
    st.subheader("Revenue by Value Segment")

    seg_rev = (
        value_segments.groupby("value_segment")
        .agg(
            customers=("customer_id", "count"),
            total_revenue=("monetary", "sum"),
            avg_predicted_clv=("predicted_clv", "mean"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
        )
        .reset_index()
    )

    fig = px.bar(
        seg_rev,
        x="value_segment",
        y="total_revenue",
        color="customers",
        title="Revenue by Value Segment",
        labels={
            "value_segment": "Segment",
            "total_revenue": "Revenue ($)",
            "customers": "Customers",
        },
    )
    st.plotly_chart(fig, width="stretch")


def _render_value_profiles(value_segments: pd.DataFrame):
    """Render value segment profiles."""
    st.subheader("Segment Profiles")

    profiles = (
        value_segments.groupby("value_segment")
        .agg(
            Customers=("customer_id", "count"),
            Avg_Recency=("recency", "mean"),
            Avg_Frequency=("frequency", "mean"),
            Avg_Monetary=("monetary", "mean"),
            Avg_Predicted_CLV=("predicted_clv", "mean"),
            Total_Revenue=("monetary", "sum"),
        )
        .round(2)
    )

    st.dataframe(
        profiles.style.format(
            {
                "Avg_Recency": "{:.1f}",
                "Avg_Frequency": "{:.1f}",
                "Avg_Monetary": "${:,.2f}",
                "Avg_Predicted_CLV": "${:,.2f}",
                "Total_Revenue": "${:,.2f}",
            }
        ).background_gradient(cmap="RdYlGn"),
        width="stretch",
    )


def _render_value_clv_accuracy(value_segments: pd.DataFrame):
    """Render CLV prediction accuracy."""
    st.subheader("CLV Prediction vs Actual Future Revenue")

    if "future_revenue" in value_segments.columns:
        fig = px.scatter(
            value_segments,
            x="predicted_clv",
            y="future_revenue",
            color="value_segment",
            hover_data=["customer_id", "recency", "frequency"],
            title="Predicted CLV vs Actual Future Revenue",
        )
        # Add diagonal line
        max_val = max(
            value_segments["predicted_clv"].max(),
            value_segments["future_revenue"].max(),
        )
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(dash="dash", color="gray"),
            )
        )
        st.plotly_chart(fig, width="stretch")

        # Accuracy metrics
        valid = value_segments.dropna(subset=["future_revenue"])
        if len(valid) > 10:
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            mae = mean_absolute_error(valid["future_revenue"], valid["predicted_clv"])
            rmse = np.sqrt(
                mean_squared_error(valid["future_revenue"], valid["predicted_clv"])
            )
            col1, col2 = st.columns(2)
            col1.metric("MAE", f"${mae:,.2f}")
            col2.metric("RMSE", f"${rmse:,.2f}")