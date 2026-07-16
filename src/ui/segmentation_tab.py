"""Customer Segmentation Tab - Enhanced with behavioral features, validation, migration, and actionable segment cards."""

import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Additional imports for validation and cluster map
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from src.analytics.segmentation import (
    behavioral_segmentation,
    compute_rfm_features,
    rfm_segmentation,
    value_based_segmentation,
)
from src.analytics.segmentation_enhanced import (
    compute_cluster_stability,
    compute_enhanced_behavioral_features,
    compute_pca_projection,
    compute_segment_migration,
    compute_segment_migration_matrix,
    compute_segment_retention,
    compute_umap_projection,
    get_segment_recommendations,
    label_segment_business,
)
from src.analytics.switching import compute_switching_matrix
from src.ui.export import render_analytics_export
from src.ui.tabs import persistent_tabs


def _normalize_metrics(
    df: pd.DataFrame, method: str = "index", invert_recency: bool = True
) -> pd.DataFrame:
    """Normalize segment metrics to indexed-to-average or 0-1 scale."""
    result = df.copy()
    id_col = result.columns[0]
    value_cols = [c for c in result.columns if c != id_col]

    recency_keywords = ["recency", "recency_days", "days_between", "interval"]

    for col in value_cols:
        col_lower = col.lower()
        is_recency = any(kw in col_lower for kw in recency_keywords)

        if method == "index":
            avg = result[col].mean()
            if avg == 0:
                result[col] = 100.0
            else:
                result[col] = 100.0 * result[col] / avg
                if is_recency and invert_recency:
                    result[col] = 200.0 - result[col]
        elif method == "minmax":
            mn, mx = result[col].min(), result[col].max()
            if mx == mn:
                result[col] = 0.5
            else:
                result[col] = (result[col] - mn) / (mx - mn)
                if is_recency and invert_recency:
                    result[col] = 1.0 - result[col]
    return result


def _render_normalization_toggle(key: str) -> str:
    """Render a raw/normalized toggle and return the selected mode."""
    return st.radio(
        "Display mode",
        ["Raw values", "Indexed to average (100 = avg)"],
        horizontal=True,
        key=key,
    )


# Bug 1: Cached wrappers for heavy computations
@st.cache_data
def _cached_compute_rfm_features(transactions_df):
    return compute_rfm_features(transactions_df)


@st.cache_data
def _cached_rfm_segmentation(rfm_df, method, n_segments):
    return rfm_segmentation(rfm_df, method=method, n_segments=n_segments)


@st.cache_data
def _cached_behavioral_segmentation(transactions_df, n_clusters):
    return behavioral_segmentation(transactions_df, n_clusters=n_clusters)


@st.cache_data
def _cached_value_based_segmentation(transactions_df, prediction_horizon_days):
    return value_based_segmentation(
        transactions_df, prediction_horizon_days=prediction_horizon_days
    )


@st.cache_data
def _cached_compute_enhanced_features(transactions_df):
    return compute_enhanced_behavioral_features(transactions_df)


@st.cache_data
def _cached_compute_segment_migration(transactions_df):
    return compute_segment_migration(transactions_df)


@st.cache_data
def _cached_compute_segment_retention(transactions_df, segment_assignments):
    return compute_segment_retention(transactions_df, segment_assignments)


@st.cache_data
def _cached_compute_umap_projection(features_array, n_components, n_neighbors, min_dist, random_state):
    return compute_umap_projection(features_array, n_components, n_neighbors, min_dist, random_state)


@st.cache_data
def _cached_compute_cluster_stability(transactions_df, n_clusters, n_iterations, method, sample_frac, seed):
    return compute_cluster_stability(transactions_df, n_clusters, n_iterations, method, sample_frac, seed)


def render_segmentation_tab(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render customer segmentation analysis tab with enhanced features."""
    st.header(" Customer Segmentation")

    if transactions_df.empty:
        st.warning("No transaction data available")
        return

    # Enhanced tabs
    tab_labels = [
        " Overview",
        " RFM Segmentation",
        " Behavioral",
        " Value-Based (CLV)",
        " Enhanced Features",
        " Cluster Map",
        " Validation",
        " Migration",
        " Retention",
        " Segment Cards",
    ]
    selected = persistent_tabs(tab_labels, "segmentation_main_tabs", default_tab=0)

    if selected == 0:
        render_overview(transactions_df)
    elif selected == 1:
        render_rfm_segmentation(transactions_df, params)
    elif selected == 2:
        render_behavioral_segmentation(transactions_df, params)
    elif selected == 3:
        render_value_segmentation(transactions_df, params)
    elif selected == 4:
        render_enhanced_features(transactions_df)
    elif selected == 5:
        render_cluster_map(transactions_df)
    elif selected == 6:
        render_validation(transactions_df)
    elif selected == 7:
        render_migration(transactions_df)
    elif selected == 8:
        render_retention(transactions_df)
    elif selected == 9:
        render_segment_cards(transactions_df)


def render_overview(transactions_df: pd.DataFrame):
    """Render segmentation overview with key metrics."""
    st.subheader("Segmentation Overview")

    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)

    with st.spinner("Computing overview metrics..."):
        from src.analytics.segmentation import compute_rfm_features
        rfm = compute_rfm_features(transactions_df)

    with col1:
        st.metric("Total Customers", f"{len(rfm):,}")
    with col2:
        st.metric("Avg Recency", f"{rfm['recency_days'].mean():.0f} days")
    with col3:
        st.metric("Avg Frequency", f"{rfm['frequency'].mean():.1f} orders")
    with col4:
        st.metric("Avg Monetary", f"${rfm['monetary'].mean():,.2f}")

    st.divider()

    # Quick links to detailed views
    st.markdown("""
    **Navigate to detailed analysis:**
    - **RFM Segmentation** — Classic RFM scoring & K-means clustering
    - **Behavioral** — K-means, HDBSCAN, Agglomerative on behavioral features
    - **Value-Based (CLV)** — Predicted lifetime value segments
    - **Enhanced Features** — 20+ behavioral variables beyond RFM
    - **Cluster Map** — UMAP/PCA projection of customer segments
    - **Validation** — Silhouette, Davies-Bouldin, stability scores
    - **Migration** — Segment-to-segment transitions over time
    - **Retention** — Cohort retention curves by segment
    - **Segment Cards** — Business labels, traits, revenue share, actions
    """)


def render_rfm_segmentation(transactions_df: pd.DataFrame, params: dict):
    """Render RFM segmentation analysis."""
    st.subheader("RFM-Based Segmentation")

    col1, col2 = st.columns(2)
    with col1:
        method = st.radio(
            "Segmentation Method",
            ["Quantile (Classic RFM)", "K-Means Clustering"],
            key="seg_tab_rfm_method",
        )
    with col2:
        if method == "K-Means Clustering":
            n_segments = st.slider("Number of Segments", 3, 12, 8, key="seg_tab_n_segments")
        else:
            n_segments = 8

    with st.spinner("Computing RFM features..."):
        rfm = _cached_compute_rfm_features(transactions_df)

    st.success(f"Computed RFM for {len(rfm)} customers")

    rfm_tabs = [" Segment Distribution", " Revenue Analysis", " 3D Visualization", " Profiles"]
    rfm_selected = persistent_tabs(rfm_tabs, "rfm_view_tabs", default_tab=0)

    if method == "Quantile (Classic RFM)":
        rfm_scored = _cached_rfm_segmentation(rfm, method="quantile", n_segments=8)

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
        rfm_clustered = _cached_rfm_segmentation(rfm, method="kmeans", n_segments=n_segments)

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

    seg_counts = rfm_scored["segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(seg_counts, values="Customers", names="Segment", title="Customer Distribution by RFM Segment")
        st.plotly_chart(fig, width="stretch")

    with col2:
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
            labels={"segment": "Segment", "total_revenue": "Revenue ($)", "customers": "Customers"},
        )
        st.plotly_chart(fig, width="stretch")


def _render_rfm_revenue_analysis(rfm_scored: pd.DataFrame):
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
        labels={"segment": "Segment", "total_revenue": "Revenue ($)", "customers": "Customers"},
    )
    st.plotly_chart(fig, width="stretch")


def _render_rfm_3d_visualization(rfm_scored: pd.DataFrame):
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

    mode = _render_normalization_toggle("seg_tab_rfm_profiles_norm")
    if mode == "Indexed to average (100 = avg)":
        display_df = _normalize_metrics(seg_rev, method="index", invert_recency=True)
        st.dataframe(
            display_df.style.format("{:.1f}", subset=display_df.columns[1:]).background_gradient(
                cmap="RdYlGn", axis=0, subset=display_df.columns[1:]
            ),
            width="stretch",
        )
    else:
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
    st.subheader("K-Means Segment Distribution")
    seg_counts = rfm_clustered["segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(seg_counts, values="Customers", names="Segment", title="Customer Distribution by K-Means Segment")
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
                pct_revenue=("monetary", lambda x: x.sum() / rfm_clustered["monetary"].sum() * 100),
            )
            .reset_index()
        )

        fig = px.bar(
            cluster_profiles,
            x="segment",
            y="total_revenue",
            color="n_customers",
            title="Revenue by K-Means Segment",
            labels={
                "segment": "Segment",
                "total_revenue": "Revenue ($)",
                "n_customers": "Customers",
            },
        )
        st.plotly_chart(fig, width="stretch")


def _render_kmeans_revenue_analysis(rfm_clustered: pd.DataFrame):
    st.subheader("Segment Profiles")
    cluster_profiles = (
        rfm_clustered.groupby("segment")
        .agg(
            n_customers=("customer_id", "count"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
            pct_revenue=("monetary", lambda x: x.sum() / rfm_clustered["monetary"].sum() * 100),
        )
        .reset_index()
    )

    mode = _render_normalization_toggle("seg_tab_kmeans_profiles_norm")
    if mode == "Indexed to average (100 = avg)":
        display_df = _normalize_metrics(cluster_profiles, method="index", invert_recency=True)
        st.dataframe(
            display_df.style.format("{:.1f}", subset=display_df.columns[1:]).background_gradient(
                cmap="RdYlGn", axis=0, subset=display_df.columns[1:]
            ),
            width="stretch",
        )
    else:
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
    st.subheader("Behavioral Segmentation")

    n_clusters = st.slider("Number of Clusters", 3, 10, 6, key="seg_tab_behav_n_clusters")
    method = st.selectbox("Clustering Method", ["kmeans", "agglomerative", "gmm", "dbscan", "hdbscan"], key="seg_tab_behav_method")

    with st.spinner("Computing behavioral segments..."):
        from src.analytics.segmentation_enhanced import (
            behavioral_segmentation as enhanced_behavioral_segmentation,
        )
        behavioral, quality_metrics = enhanced_behavioral_segmentation(transactions_df, n_clusters=n_clusters, method=method, return_metrics=True)

    # Show validation metrics
    if quality_metrics:
        st.subheader("Cluster Validation")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Silhouette Score", f"{quality_metrics.get('silhouette_score', 0):.3f}")
        with col2:
            st.metric("Davies-Bouldin", f"{quality_metrics.get('davies_bouldin_score', 0):.3f}")
        with col3:
            st.metric("N Clusters", quality_metrics.get("n_clusters", 0))
        with col4:
            st.metric("Min Cluster Size", quality_metrics.get("cluster_size_min", 0))

    # Persistent sub-tabs
    behav_tabs = [" Profiles", " Radar Chart", " Box Plots", " Revenue", " Switching"]
    behav_selected = persistent_tabs(behav_tabs, "behavioral_view_tabs", default_tab=0)

    feature_cols = [c for c in behavioral.columns if c not in ["customer_id", "cluster", "segment"]]
    profiles = behavioral.groupby("segment")[feature_cols].mean().round(2)
    profiles = profiles.T
    profiles.columns.name = None

    if n_clusters >= 2:
        from src.analytics.segmentation_enhanced import compute_cluster_stability
        stability = compute_cluster_stability(transactions_df, n_clusters=n_clusters, method="kmeans")
        if stability:
            st.info(f"Cluster Stability (ARI): {stability['mean_ari']:.3f} ± {stability['std_ari']:.3f}")

    if " Profiles" in behav_selected:
        _render_behavioral_profiles(profiles)
    elif " Radar Chart" in behav_selected:
        _render_behavioral_radar(profiles, key_features=feature_cols)
    elif " Box Plots" in behav_selected:
        _render_behavioral_box_plots(behavioral, feature_cols)
    elif " Revenue" in behav_selected:
        _render_behavioral_revenue(transactions_df, behavioral)
    elif " Switching" in behav_selected:
        _render_behavioral_switching(transactions_df, behavioral)

    render_analytics_export(behavioral, "Behavioral_Segments")


def _render_behavioral_profiles(profiles: pd.DataFrame):
    st.subheader("Behavioral Segment Profiles")
    mode = _render_normalization_toggle("seg_tab_behav_profiles_norm")
    if mode == "Indexed to average (100 = avg)":
        display_df = _normalize_metrics(profiles.reset_index(), method="index")
        display_df = display_df.set_index(profiles.index.name or "index")
        st.dataframe(
            display_df.style.format("{:.1f}").background_gradient(cmap="RdYlGn", axis=1),
            width="stretch",
        )
    else:
        st.dataframe(
            profiles.style.background_gradient(cmap="RdYlGn", axis=1),
            width="stretch",
        )


def _render_behavioral_radar(profiles: pd.DataFrame, key_features: list):
    st.subheader("Segment Comparison (Key Metrics)")
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
        radar_norm = radar_data.copy()
        for feat in radar_norm.index:
            mn, mx = radar_norm.loc[feat].min(), radar_norm.loc[feat].max()
            if mx > mn:
                radar_norm.loc[feat] = (radar_norm.loc[feat] - mn) / (mx - mn)
            else:
                radar_norm.loc[feat] = 0.5

        fig = go.Figure()
        for segment in radar_norm.columns:
            fig.add_trace(
                go.Scatterpolar(
                    r=radar_norm[segment].values,
                    theta=radar_norm.index,
                    fill="toself",
                    name=segment,
                    hovertemplate="<b>%{theta}</b>: %{customdata:.2f}<extra>%{legend}</extra>",
                    customdata=radar_data[segment].values,
                )
            )
        fig.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 1]}},
            showlegend=True,
            title="Segment Comparison (0-1 Normalized)",
        )
        st.plotly_chart(fig, width="stretch")


def _render_behavioral_box_plots(behavioral: pd.DataFrame, feature_cols: list):
    st.subheader("Key Differentiators")
    diff_feature = st.selectbox("Select Feature", feature_cols, key="seg_tab_behav_diff_feature")
    fig = px.box(
        behavioral,
        x="segment",
        y=diff_feature,
        color="segment",
        title=f"{diff_feature} by Segment",
    )
    st.plotly_chart(fig, width="stretch")


def _render_behavioral_revenue(transactions_df: pd.DataFrame, behavioral: pd.DataFrame):
    st.subheader("Revenue by Behavioral Segment")
    df = transactions_df.copy()
    df["revenue"] = df["price"] * df["quantity"]
    merged = df.merge(behavioral[["customer_id", "segment"]], on="customer_id", how="left")
    seg_rev = (
        merged.groupby("segment")
        .agg(customers=("customer_id", "nunique"), revenue=("revenue", "sum"), avg_order=("revenue", "mean"))
        .reset_index()
    )
    fig = px.bar(seg_rev, x="segment", y="revenue", color="customers", title="Revenue by Behavioral Segment")
    st.plotly_chart(fig, width="stretch")


def _render_behavioral_switching(transactions_df: pd.DataFrame, behavioral: pd.DataFrame):
    st.subheader("Switching Behavior by Segment")
    switch_matrix = compute_switching_matrix(transactions_df)
    if not switch_matrix.empty:
        segment_customers = behavioral.groupby("segment")["customer_id"].apply(set)
        for segment, customers in segment_customers.items():
            if customers:
                seg_switches = switch_matrix[
                    switch_matrix["from_product"].isin(customers) | switch_matrix["to_product"].isin(customers)
                ]
                if not seg_switches.empty:
                    st.write(f"**{segment}** - Top switches:")
                    st.dataframe(seg_switches.head(10), width="stretch")


def render_value_segmentation(transactions_df: pd.DataFrame, params: dict):
    st.subheader("Value-Based Segmentation (Predicted CLV)")

    horizon = st.slider("Prediction Horizon (days)", 30, 365, 90, key="seg_tab_value_horizon")

    with st.spinner("Computing value segments..."):
        value_segments = _cached_value_based_segmentation(transactions_df, horizon)

    value_tabs = [" Distribution", " Revenue", " Profiles", " CLV Accuracy"]
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
    st.subheader("Value Segment Distribution")
    seg_counts = value_segments["value_segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(seg_counts, values="Customers", names="Segment", title="Value Segment Distribution")
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
            labels={"value_segment": "Segment", "total_revenue": "Revenue ($)", "customers": "Customers"},
        )
        st.plotly_chart(fig, width="stretch")


def _render_value_revenue(value_segments: pd.DataFrame):
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
        labels={"value_segment": "Segment", "total_revenue": "Revenue ($)", "customers": "Customers"},
    )
    st.plotly_chart(fig, width="stretch")


def _render_value_profiles(value_segments: pd.DataFrame):
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

    mode = _render_normalization_toggle("seg_tab_value_profiles_norm")
    if mode == "Indexed to average (100 = avg)":
        display_df = _normalize_metrics(profiles.reset_index(), method="index", invert_recency=True)
        display_df = display_df.set_index("value_segment")
        st.dataframe(
            display_df.style.format("{:.1f}").background_gradient(cmap="RdYlGn", axis=0),
            width="stretch",
        )
    else:
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
        max_val = max(value_segments["predicted_clv"].max(), value_segments["future_revenue"].max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Perfect Prediction",
                line={"dash": "dash", "color": "gray"},
            )
        )
        st.plotly_chart(fig, width="stretch")

        valid = value_segments.dropna(subset=["future_revenue"])
        if len(valid) > 10:
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            mae = mean_absolute_error(valid["future_revenue"], valid["predicted_clv"])
            rmse = np.sqrt(mean_squared_error(valid["future_revenue"], valid["predicted_clv"]))
            col1, col2 = st.columns(2)
            col1.metric("MAE", f"${mae:,.2f}")
            col2.metric("RMSE", f"${rmse:,.2f}")


def render_enhanced_features(transactions_df: pd.DataFrame):
    """Render enhanced behavioral features beyond classic RFM."""
    st.subheader("Enhanced Behavioral Features")
    st.caption("20+ transaction-derived features beyond classic RFM")

    with st.spinner("Computing enhanced behavioral features..."):
        features = _cached_compute_enhanced_features(transactions_df)

    if features.empty:
        st.warning("No features computed")
        return

    st.success(f"Computed {len(features.columns)} features for {len(features)} customers")

    # Feature overview
    st.subheader("Feature Overview")

    # Categorize features
    feature_categories = {
        "Classic RFM": [
            "recency_days",
            "frequency",
            "monetary",
            "avg_order_value",
            "customer_lifetime_days",
        ],
        "Basket & Penetration": [
            "n_baskets",
            "basket_penetration",
            "avg_basket_value",
            "avg_basket_depth",
            "avg_basket_categories",
        ],
        "Loyalty & Repeat": [
            "repeat_rate",
            "tt2p_days",
            "median_interpurchase_days",
            "purchase_interval",
        ],
        "Diversity & Concentration": [
            "n_unique_products",
            "n_unique_categories",
            "category_breadth",
            "product_hhi",
        ],
        "Switching & Entropy": [
            "switching_entropy",
            "switch_count",
            "switch_rate",
            "concentration_hhi",
            "loyalty_segment",
        ],
        "Price & Promotion": [
            "avg_price_paid",
            "price_sensitivity",
            "price_cv",
        ],
        "Temporal": [
            "month_of_year",
            "weekend_ratio",
            "avg_days_between",
        ],
    }

    for cat, feats in feature_categories.items():
        available = [f for f in feats if f in features.columns]
        if available:
            with st.expander(f"{cat} ({len(available)} features)"):
                st.dataframe(features[available].describe().T.style.format("{:.2f}"), width="stretch")

    # Feature importance for segmentation
    st.subheader("Feature Correlation with Revenue")
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    if "monetary" in features.columns:
        corrs = features[numeric_cols].corrwith(features["monetary"]).drop("monetary").sort_values(ascending=False)
        fig = px.bar(
            x=corrs.values[:20],
            y=corrs.index[:20],
            orientation="h",
            title="Top 20 Features Correlated with Revenue",
            labels={"x": "Correlation", "y": "Feature"},
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width="stretch")


def render_cluster_map(transactions_df: pd.DataFrame):
    """Render 2D cluster map using UMAP/PCA projection."""
    st.subheader("Cluster Map (UMAP/PCA Projection)")
    st.caption("2D visualization of customer segments in feature space")

    # Get enhanced features
    with st.spinner("Computing enhanced features..."):
        features = _cached_compute_enhanced_features(transactions_df)

    if features.empty:
        st.warning("No features computed")
        return

    # Prepare features for clustering
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in ["customer_id", "month_of_year"]]
    X = features[feature_cols].fillna(0).values

    if len(X) < 10:
        st.warning("Not enough customers for clustering")
        return

    # Scaling
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering controls
    col1, col2, col3 = st.columns(3)
    with col1:
        method = st.selectbox("Clustering Method", ["kmeans", "agglomerative", "gmm", "hdbscan"], key="cluster_map_method")
    with col2:
        n_clusters = st.slider("Number of Clusters", 3, 15, 6, key="cluster_map_n_clusters")
    with col3:
        projection_method = st.selectbox("Projection", ["UMAP", "PCA"], key="cluster_map_projection")

    # Clustering
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
    from sklearn.mixture import GaussianMixture

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)
    elif method == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
    elif method == "hdbscan":
        if not HDBSCAN_AVAILABLE:
            st.warning("HDBSCAN not installed. Using K-means instead.")
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
        else:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
            labels = clusterer.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Projection
    if projection_method == "UMAP":
        if not UMAP_AVAILABLE:
            st.warning("UMAP not available. Using PCA instead.")
            proj = compute_pca_projection(X_scaled)
        else:
            proj = _cached_compute_umap_projection(X_scaled, 2, 15, 0.1, 42)
    else:
        proj = compute_pca_projection(X_scaled)

    # Create plot
    proj_df = pd.DataFrame(proj, columns=["x", "y"])
    proj_df["cluster"] = labels
    proj_df["customer_id"] = features["customer_id"].values

    # Add segment labels
    from src.analytics.segmentation_enhanced import _label_behavioral_clusters
    feature_df = pd.DataFrame(X_scaled, columns=[f"f{i}" for i in range(X_scaled.shape[1])])
    feature_df["cluster"] = labels
    cluster_profiles = feature_df.groupby("cluster").mean()
    cluster_labels = _label_behavioral_clusters(cluster_profiles)

    proj_df["segment_label"] = proj_df["cluster"].map(cluster_labels)

    # Plot
    fig = px.scatter(
        proj_df,
        x="x",
        y="y",
        color="segment_label",
        hover_data=["customer_id"],
        title=f"Customer Segments ({'UMAP' if projection_method == 'UMAP' else 'PCA'} Projection)",
        labels={"x": "Component 1", "y": "Component 2"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, width="stretch")

    # Show cluster sizes
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    st.dataframe(
        pd.DataFrame(
            {
                "Cluster": cluster_counts.index,
                "Label": [cluster_labels.get(i, f"Cluster {i}") for i in cluster_counts.index],
                "Size": cluster_counts.values,
            }
        ),
        width="stretch",
    )


def render_validation(transactions_df: pd.DataFrame):
    """Render cluster validation metrics."""
    st.subheader("Cluster Validation Metrics")
    st.caption("Quality assessment for different clustering configurations")

    # Get enhanced features
    with st.spinner("Computing features..."):
        features = _cached_compute_enhanced_features(transactions_df)

    if features.empty:
        st.warning("No features computed")
        return

    numeric_cols = features.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in ["customer_id", "month_of_year"]]
    X = features[feature_cols].fillna(0).values

    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.metrics import davies_bouldin_score, silhouette_score
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Validation across methods and k values
    st.subheader("Validation Across K Values")

    k_range = st.slider("K Range", 2, 15, (3, 12), key="val_k_range")
    methods = st.multiselect("Methods", ["kmeans", "agglomerative", "gmm"], default=["kmeans", "agglomerative"], key="val_methods")

    if st.button("Run Validation", key="run_validation"):
        results = []
        progress = st.progress(0)

        for method in methods:
            for k in range(k_range[0], k_range[1] + 1):
                try:
                    if method == "kmeans":
                        model = KMeans(n_clusters=k, random_state=42, n_init=10)
                    elif method == "agglomerative":
                        model = AgglomerativeClustering(n_clusters=k)
                    elif method == "gmm":
                        model = GaussianMixture(n_components=k, random_state=42, n_init=10)
                    else:
                        continue

                    labels = model.fit_predict(X_scaled)

                    if len(set(labels)) < 2:
                        continue

                    sil = silhouette_score(X_scaled, labels)
                    db = davies_bouldin_score(X_scaled, labels)

                    results.append({
                        "Method": method,
                        "K": k,
                        "Silhouette": round(sil, 4),
                        "Davies-Bouldin": round(db, 4),
                        "N_Clusters": len(set(labels)),
                    })
                except Exception:
                    pass

                progress.progress(len(results) / ((k_range[1] - k_range[0] + 1) * len(methods)))

        if results:
            results_df = pd.DataFrame(results)

            # Silhouette plot
            fig = px.line(results_df, x="K", y="Silhouette", color="Method", markers=True, title="Silhouette Score by K")
            st.plotly_chart(fig, width="stretch")

            # Davies-Bouldin plot
            fig2 = px.line(results_df, x="K", y="Davies-Bouldin", color="Method", markers=True, title="Davies-Bouldin Index by K")
            st.plotly_chart(fig2, width="stretch")

            # Table
            st.dataframe(
                results_df.style.format({"Silhouette": "{:.4f}", "Davies-Bouldin": "{:.4f}"}).background_gradient(
                    cmap="RdYlGn", subset=["Silhouette"]
                ),
                width="stretch",
            )

            # Best K recommendation
            best = results_df.loc[results_df["Silhouette"].idxmax()]
            st.success(f"Recommended: {best['Method']} with K={best['K']} (Silhouette: {best['Silhouette']:.4f})")

    st.divider()

    # Stability analysis
    st.subheader("Cluster Stability Analysis")
    col1, col2 = st.columns(2)
    with col1:
        n_clusters_stab = st.slider("N Clusters for Stability", 3, 12, 6, key="stab_n_clusters")
    with col2:
        n_iter = st.slider("Iterations", 5, 30, 10, key="stab_n_iter")

    if st.button("Run Stability Analysis", key="run_stability"):
        with st.spinner("Computing stability..."):
            stability = _cached_compute_cluster_stability(transactions_df, n_clusters_stab, n_iter, "kmeans", 0.8, 42)

        if stability:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean ARI", f"{stability['mean_ari']:.4f}")
            with col2:
                st.metric("Std ARI", f"{stability['std_ari']:.4f}")
            with col3:
                st.metric("Min ARI", f"{stability['min_ari']:.4f}")
            with col4:
                st.metric("Max ARI", f"{stability['max_ari']:.4f}")

            if stability["mean_ari"] > 0.8:
                st.success("High stability - clusters are robust")
            elif stability["mean_ari"] > 0.6:
                st.warning("Moderate stability - some cluster variation")
            else:
                st.error("Low stability - clusters may not be reliable")


def render_migration(transactions_df: pd.DataFrame):
    """Render segment migration analysis over time."""
    st.subheader("Segment Migration Analysis")
    st.caption("Track how customers move between segments over time")

    period_freq = st.selectbox(
        "Period Frequency",
        ["M", "W", "Q"],
        format_func=lambda x: {"M": "Monthly", "W": "Weekly", "Q": "Quarterly"}[x],
        key="mig_period",
    )
    n_periods = st.slider("Number of Periods", 3, 12, 6, key="mig_n_periods")

    with st.spinner("Computing segment migrations..."):
        migrations = _cached_compute_segment_migration(transactions_df)

    if migrations.empty:
        st.warning("Insufficient data for migration analysis. Need multiple time periods with sufficient customers.")
        return

    # Migration matrix
    st.subheader("Segment Transition Matrix")
    matrix = compute_segment_migration_matrix(migrations)

    if not matrix.empty:
        fig = px.imshow(
            matrix.values,
            x=matrix.columns,
            y=matrix.index,
            color_continuous_scale="Blues",
            title="Segment Transitions (rows=from, cols=to)",
            labels={"x": "To Segment", "y": "From Segment", "color": "Count"},
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width="stretch")

    # Sankey diagram for flows
    st.subheader("Migration Flows")
    if not migrations.empty:
        agg = migrations.groupby(["segment_from", "segment_to"])["count"].sum().reset_index()
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=agg["segment_from"].unique().tolist()
                        + [s for s in agg["segment_to"].unique() if s not in agg["segment_from"].unique()],
                    ),
                    link=dict(
                        source=[agg["segment_from"].unique().tolist().index(s) for s in agg["segment_from"]],
                        target=[agg["segment_from"].unique().tolist().index(s) for s in agg["segment_to"]],
                        value=agg["count"],
                    ),
)
        ]
    )
    fig.update_layout(title_text="Segment Migration Flows", height=500)
    st.plotly_chart(fig, width="stretch")

    # Migration table
    st.subheader("Detailed Transitions")
    agg = migrations.groupby(["period", "segment_from", "segment_to"])["count"].sum().reset_index()
    st.dataframe(
        agg.pivot_table(index="segment_from", columns=["segment_to", "period"], values="count", fill_value=0),
        width="stretch",
    )


def render_retention(transactions_df: pd.DataFrame):
    """Render segment retention curves."""
    st.subheader("Segment Retention Curves")
    st.caption("Cohort-based retention by segment")

    # First, we need segment assignments
    with st.spinner("Computing segments..."):
        from src.analytics.segmentation_enhanced import behavioral_segmentation
        behavioral = behavioral_segmentation(transactions_df, n_clusters=6, method="kmeans")

    if behavioral.empty:
        st.warning("No segments computed")
        return

    with st.spinner("Computing retention curves..."):
        retention = _cached_compute_segment_retention(transactions_df, behavioral)

    if retention.empty:
        st.warning("Insufficient data for retention analysis")
        return

    # Plot retention curves
    st.subheader("Retention by Segment")
    fig = px.line(
        retention,
        x="period_number",
        y="retention_rate",
        color="segment",
        title="Retention Curves by Segment",
        labels={"period_number": "Periods Since First Purchase", "retention_rate": "Retention Rate", "segment": "Segment"},
        markers=True,
    )
    fig.update_layout(height=500, yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    # Retention table
    st.subheader("Retention Table")
    pivot = retention.pivot_table(index="segment", columns="period_number", values="retention_rate", fill_value=0)
    st.dataframe(pivot.style.format("{:.1%}").background_gradient(cmap="RdYlGn", axis=1), width="stretch")


def render_segment_cards(transactions_df: pd.DataFrame):
    """Render actionable segment cards with business labels, traits, and actions."""
    st.subheader("Actionable Segment Cards")
    st.caption("Each segment gets a business label, key traits, revenue share, and recommended actions")

    # Get behavioral segments
    with st.spinner("Computing segments..."):
        from src.analytics.segmentation_enhanced import behavioral_segmentation
        behavioral = behavioral_segmentation(transactions_df, n_clusters=6, method="kmeans")

    if behavioral.empty:
        st.warning("No segments computed")
        return

    # Get segment profiles
    from src.analytics.segmentation_enhanced import get_segment_profiles, label_segment_business
    profiles = get_segment_profiles(transactions_df, behavioral, segment_col="segment")

    # Add business labels and actions
    for _, row in profiles.iterrows():
        label, action = label_segment_business(row, row["segment"])
        profiles.loc[profiles["segment"] == row["segment"], "business_label"] = label
        profiles.loc[profiles["segment"] == row["segment"], "recommended_action"] = action

    # Render segment cards
    st.subheader("Segment Cards")

    for _, segment in profiles.iterrows():
        with st.expander(
            f"**{segment.get('business_label', segment['segment'])}** ({segment['segment']}) — {segment['n_customers']:,} customers ({segment['customer_share']:.1%})"
        ):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Revenue Share", f"{segment['revenue_share']:.1%}")
                st.metric("Avg Order Value", f"${segment['avg_order_value']:,.2f}")
            with col2:
                st.metric("Customers", f"{segment['n_customers']:,}")
                st.metric("Repeat Rate", f"{segment['repeat_rate']:.1%}")
            with col3:
                st.metric("Avg Recency", f"{segment.get('avg_recency', segment.get('Avg_Recency', 0)):.0f} days")
                st.metric("Avg Frequency", f"{segment.get('avg_frequency', segment.get('Avg_Frequency', 0)):.1f}")
            with col4:
                st.metric("Revenue/Customer", f"${segment.get('revenue_per_customer', segment.get('revenue_per_customer', 0)):,.2f}")
                st.metric("Top Category", segment.get("top_category", segment.get("top_category", "N/A")))

            # Recommended actions
            recs = get_segment_recommendations(segment.get("business_label", segment["segment"]))
            st.markdown("**Recommended Actions:**")
            cols = st.columns(3)
            with cols[0]:
                st.markdown("**Primary:**")
                for action in recs.get("primary", []):
                    st.markdown(f"• {action}")
            with cols[1]:
                st.markdown("**Secondary:**")
                for action in recs.get("secondary", []):
                    st.markdown(f"• {action}")
            with cols[2]:
                st.markdown("**Avoid:**")
                for action in recs.get("avoid", []):
                    st.markdown(f"• {action}")

            # Top products for this segment
            st.markdown("**Top Products in Segment:**")
            segment_customers = set(behavioral[behavioral["segment"] == segment["segment"]]["customer_id"])
            seg_trans = transactions_df[transactions_df["customer_id"].isin(segment_customers)]
            if not seg_trans.empty:
                top_products = (
                    seg_trans.groupby("stockcode")
                    .agg(revenue=("price", "sum"), qty=("quantity", "sum"), customers=("customer_id", "nunique"))
                    .reset_index()
                )
                top_products = top_products.sort_values("revenue", ascending=False).head(5)
                top_products["product_name"] = top_products["stockcode"].map(lambda x: f"Product {x}")
                st.dataframe(
                    top_products[["product_name", "revenue", "qty", "customers"]].style.format({"revenue": "${:,.2f}"}),
                    width="stretch",
                )


# =====================================================================
# EXISTING HELPER FUNCTIONS (preserved from original)
# =====================================================================

def _render_rfm_segment_distribution(rfm_scored: pd.DataFrame):
    """Render RFM segment distribution."""
    st.subheader("RFM Segment Distribution")
    seg_counts = rfm_scored["segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(seg_counts, values="Customers", names="Segment", title="Customer Distribution by RFM Segment")
        st.plotly_chart(fig, width="stretch")
    with col2:
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
            labels={"segment": "Segment", "total_revenue": "Revenue ($)", "customers": "Customers"},
        )
        st.plotly_chart(fig, width="stretch")


def _render_rfm_revenue_analysis(rfm_scored: pd.DataFrame):
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
        labels={"segment": "Segment", "total_revenue": "Revenue ($)", "customers": "Customers"},
    )
    st.plotly_chart(fig, width="stretch")


def _render_rfm_3d_visualization(rfm_scored: pd.DataFrame):
    st.subheader("RFM 3D Visualization")
    fig = px.scatter_3d(
        rfm_scored,
        x="recency_days",
        y="frequency",
        z="monetary",
        color="segment",
        hover_data=["customer_id", "avg_order_value", "n_unique_products"],
        title="RFM Space Colored by Segment",
        labels={"recency_days": "Recency (days)", "frequency": "Frequency", "monetary": "Monetary ($)"},
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, width="stretch")


def _render_rfm_profiles_table(rfm_scored: pd.DataFrame):
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
    mode = _render_normalization_toggle("seg_tab_rfm_profiles_norm")
    if mode == "Indexed to average (100 = avg)":
        display_df = _normalize_metrics(seg_rev, method="index", invert_recency=True)
        st.dataframe(
            display_df.style.format("{:.1f}", subset=display_df.columns[1:]).background_gradient(
                cmap="RdYlGn", axis=0, subset=display_df.columns[1:]
            ),
            width="stretch",
        )
    else:
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
    st.subheader("K-Means Segment Distribution")
    seg_counts = rfm_clustered["segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(seg_counts, values="Customers", names="Segment", title="Customer Distribution by K-Means Segment")
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
                pct_revenue=("monetary", lambda x: x.sum() / rfm_clustered["monetary"].sum() * 100),
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
    st.subheader("Segment Profiles")
    cluster_profiles = (
        rfm_clustered.groupby("segment")
        .agg(
            n_customers=("customer_id", "count"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
            pct_revenue=("monetary", lambda x: x.sum() / rfm_clustered["monetary"].sum() * 100),
        )
        .reset_index()
    )
    mode = _render_normalization_toggle("seg_tab_kmeans_profiles_norm")
    if mode == "Indexed to average (100 = avg)":
        display_df = _normalize_metrics(cluster_profiles, method="index", invert_recency=True)
        st.dataframe(
            display_df.style.format("{:.1f}", subset=display_df.columns[1:]).background_gradient(
                cmap="RdYlGn", axis=0, subset=display_df.columns[1:]
            ),
            width="stretch",
        )
    else:
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


def _render_value_distribution(value_segments: pd.DataFrame):
    st.subheader("Value Segment Distribution")
    seg_counts = value_segments["value_segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(seg_counts, values="Customers", names="Segment", title="Value Segment Distribution")
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
            labels={"value_segment": "Segment", "total_revenue": "Revenue ($)", "customers": "Customers"},
        )
        st.plotly_chart(fig, width="stretch")


def _render_value_revenue(value_segments: pd.DataFrame):
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
        labels={"value_segment": "Segment", "total_revenue": "Revenue ($)", "customers": "Customers"},
    )
    st.plotly_chart(fig, width="stretch")


def _render_value_profiles(value_segments: pd.DataFrame):
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

    mode = _render_normalization_toggle("seg_tab_value_profiles_norm")
    if mode == "Indexed to average (100 = avg)":
        display_df = _normalize_metrics(profiles.reset_index(), method="index", invert_recency=True)
        display_df = display_df.set_index("value_segment")
        st.dataframe(
            display_df.style.format("{:.1f}").background_gradient(cmap="RdYlGn", axis=0),
            width="stretch",
        )
    else:
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
        max_val = max(value_segments["predicted_clv"].max(), value_segments["future_revenue"].max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Perfect Prediction",
                line={"dash": "dash", "color": "gray"},
            )
        )
        st.plotly_chart(fig, width="stretch")

        valid = value_segments.dropna(subset=["future_revenue"])
        if len(valid) > 10:
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            mae = mean_absolute_error(valid["future_revenue"], valid["predicted_clv"])
            rmse = np.sqrt(mean_squared_error(valid["future_revenue"], valid["predicted_clv"]))
            col1, col2 = st.columns(2)
            col1.metric("MAE", f"${mae:,.2f}")
            col2.metric("RMSE", f"${rmse:,.2f}")


def _render_behavioral_profiles(profiles: pd.DataFrame):
    st.subheader("Behavioral Segment Profiles")
    mode = _render_normalization_toggle("seg_tab_behav_profiles_norm")
    if mode == "Indexed to average (100 = avg)":
        display_df = _normalize_metrics(profiles.reset_index(), method="index")
        display_df = display_df.set_index(profiles.index.name or "index")
        st.dataframe(
            display_df.style.format("{:.1f}").background_gradient(cmap="RdYlGn", axis=1),
            width="stretch",
        )
    else:
        st.dataframe(profiles.style.background_gradient(cmap="RdYlGn", axis=1), width="stretch")


def _render_behavioral_radar(profiles: pd.DataFrame, key_features: list):
    st.subheader("Segment Comparison (Key Metrics)")
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
        radar_norm = radar_data.copy()
        for feat in radar_norm.index:
            mn, mx = radar_norm.loc[feat].min(), radar_norm.loc[feat].max()
            if mx > mn:
                radar_norm.loc[feat] = (radar_norm.loc[feat] - mn) / (mx - mn)
            else:
                radar_norm.loc[feat] = 0.5

        fig = go.Figure()
        for segment in radar_norm.columns:
            fig.add_trace(
                go.Scatterpolar(
                    r=radar_norm[segment].values,
                    theta=radar_norm.index,
                    fill="toself",
                    name=segment,
                    hovertemplate="<b>%{theta}</b>: %{customdata:.2f}<extra>%{legend}</extra>",
                    customdata=radar_data[segment].values,
                )
            )
        fig.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 1]}},
            showlegend=True,
            title="Segment Comparison (0-1 Normalized)",
        )
        st.plotly_chart(fig, width="stretch")


def _render_behavioral_box_plots(behavioral: pd.DataFrame, feature_cols: list):
    st.subheader("Key Differentiators")
    diff_feature = st.selectbox("Select Feature", feature_cols, key="seg_tab_behav_diff_feature")
    fig = px.box(behavioral, x="segment", y=diff_feature, color="segment", title=f"{diff_feature} by Segment")
    st.plotly_chart(fig, width="stretch")


def _render_behavioral_revenue(transactions_df: pd.DataFrame, behavioral: pd.DataFrame):
    st.subheader("Revenue by Behavioral Segment")
    df = transactions_df.copy()
    df["revenue"] = df["price"] * df["quantity"]
    merged = df.merge(behavioral[["customer_id", "segment"]], on="customer_id", how="left")
    seg_rev = (
        merged.groupby("segment")
        .agg(customers=("customer_id", "nunique"), revenue=("revenue", "sum"), avg_order=("revenue", "mean"))
        .reset_index()
    )
    fig = px.bar(seg_rev, x="segment", y="revenue", color="customers", title="Revenue by Behavioral Segment")
    st.plotly_chart(fig, width="stretch")


def _render_behavioral_switching(transactions_df: pd.DataFrame, behavioral: pd.DataFrame):
    st.subheader("Switching Behavior by Segment")
    switch_matrix = compute_switching_matrix(transactions_df)
    if not switch_matrix.empty:
        segment_customers = behavioral.groupby("segment")["customer_id"].apply(set)
        for segment, customers in segment_customers.items():
            if customers:
                seg_switches = switch_matrix[
                    switch_matrix["from_product"].isin(customers) | switch_matrix["to_product"].isin(customers)
                ]
                if not seg_switches.empty:
                    st.write(f"**{segment}** - Top switches:")
                    st.dataframe(seg_switches.head(10), width="stretch")


def _render_value_distribution(value_segments: pd.DataFrame):
    st.subheader("Value Segment Distribution")
    seg_counts = value_segments["value_segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(seg_counts, values="Customers", names="Segment", title="Value Segment Distribution")
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
            labels={"value_segment": "Segment", "total_revenue": "Revenue ($)", "customers": "Customers"},
        )
        st.plotly_chart(fig, width="stretch")


def _render_value_revenue(value_segments: pd.DataFrame):
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
        labels={"value_segment": "Segment", "total_revenue": "Revenue ($)", "customers": "Customers"},
    )
    st.plotly_chart(fig, width="stretch")


def _render_value_profiles(value_segments: pd.DataFrame):
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

    mode = _render_normalization_toggle("seg_tab_value_profiles_norm")
    if mode == "Indexed to average (100 = avg)":
        display_df = _normalize_metrics(profiles.reset_index(), method="index", invert_recency=True)
        display_df = display_df.set_index("value_segment")
        st.dataframe(
            display_df.style.format("{:.1f}").background_gradient(cmap="RdYlGn", axis=0),
            width="stretch",
        )
    else:
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
        max_val = max(value_segments["predicted_clv"].max(), value_segments["future_revenue"].max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Perfect Prediction",
                line={"dash": "dash", "color": "gray"},
            )
        )
        st.plotly_chart(fig, width="stretch")

        valid = value_segments.dropna(subset=["future_revenue"])
        if len(valid) > 10:
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            mae = mean_absolute_error(valid["future_revenue"], valid["predicted_clv"])
            rmse = np.sqrt(mean_squared_error(valid["future_revenue"], valid["predicted_clv"]))
            col1, col2 = st.columns(2)
            col1.metric("MAE", f"${mae:,.2f}")
            col2.metric("RMSE", f"${rmse:,.2f}")


def _render_behavioral_profiles(profiles: pd.DataFrame):
    st.subheader("Behavioral Segment Profiles")
    mode = _render_normalization_toggle("seg_tab_behav_profiles_norm")
    if mode == "Indexed to average (100 = avg)":
        display_df = _normalize_metrics(profiles.reset_index(), method="index")
        display_df = display_df.set_index(profiles.index.name or "index")
        st.dataframe(
            display_df.style.format("{:.1f}").background_gradient(cmap="RdYlGn", axis=1),
            width="stretch",
        )
    else:
        st.dataframe(profiles.style.background_gradient(cmap="RdYlGn", axis=1), width="stretch")


def _render_behavioral_radar(profiles: pd.DataFrame, key_features: list):
    st.subheader("Segment Comparison (Key Metrics)")
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
        radar_norm = radar_data.copy()
        for feat in radar_norm.index:
            mn, mx = radar_norm.loc[feat].min(), radar_norm.loc[feat].max()
            if mx > mn:
                radar_norm.loc[feat] = (radar_norm.loc[feat] - mn) / (mx - mn)
            else:
                radar_norm.loc[feat] = 0.5

        fig = go.Figure()
        for segment in radar_norm.columns:
            fig.add_trace(
                go.Scatterpolar(
                    r=radar_norm[segment].values,
                    theta=radar_norm.index,
                    fill="toself",
                    name=segment,
                    hovertemplate="<b>%{theta}</b>: %{customdata:.2f}<extra>%{legend}</extra>",
                    customdata=radar_data[segment].values,
                )
            )
        fig.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 1]}},
            showlegend=True,
            title="Segment Comparison (0-1 Normalized)",
        )
        st.plotly_chart(fig, width="stretch")


def _render_behavioral_box_plots(behavioral: pd.DataFrame, feature_cols: list):
    st.subheader("Key Differentiators")
    diff_feature = st.selectbox("Select Feature", feature_cols, key="seg_tab_behav_diff_feature")
    fig = px.box(behavioral, x="segment", y=diff_feature, color="segment", title=f"{diff_feature} by Segment")
    st.plotly_chart(fig, width="stretch")


def _render_behavioral_revenue(transactions_df: pd.DataFrame, behavioral: pd.DataFrame):
    st.subheader("Revenue by Behavioral Segment")
    df = transactions_df.copy()
    df["revenue"] = df["price"] * df["quantity"]
    merged = df.merge(behavioral[["customer_id", "segment"]], on="customer_id", how="left")
    seg_rev = (
        merged.groupby("segment")
        .agg(customers=("customer_id", "nunique"), revenue=("revenue", "sum"), avg_order=("revenue", "mean"))
        .reset_index()
    )
    fig = px.bar(seg_rev, x="segment", y="revenue", color="customers", title="Revenue by Behavioral Segment")
    st.plotly_chart(fig, width="stretch")


def _render_markov_tab(transition_matrix: pd.DataFrame, product_lookup: dict):
    """Render first-order Markov transition matrix."""
    st.subheader("Markov Transition Matrix")
    st.caption(
        "Each row sums to 1.0 and shows the probability of the next purchase, "
        "conditional on the current product. This is a lightweight sequential model "
        "that strengthens the scientific basis of switching analysis."
    )

    if transition_matrix.empty:
        st.info("Not enough sequential transitions to build the Markov matrix")
        return

    # Heatmap
    fig = px.imshow(
        transition_matrix.values,
        x=transition_matrix.columns,
        y=transition_matrix.index,
        color_continuous_scale="Blues",
        aspect="auto",
        title="Markov Transition Probabilities",
        labels={"x": "Next Product", "y": "Current Product", "color": "P(next | current)"},
    )
    fig.update_layout(height=600, xaxis_tickangle=45)
    st.plotly_chart(fig, width="stretch")

    with st.expander("View Raw Matrix"):
        st.dataframe(transition_matrix.style.format("{:.3f}").background_gradient(cmap="Blues"), width="stretch")


def _render_brand_switching_tab(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render brand switching within categories."""
    st.subheader("Brand Switching Analysis")

    window_days = st.slider("Window (days)", 30, 365, 90, key="brand_switch_window")

    if "brand" not in transactions_df.columns or "category" not in transactions_df.columns:
        st.warning("Brand and Category columns required for brand switching analysis")
        return

    with st.spinner("Computing brand switching..."):
        from src.analytics.switching import compute_brand_switching_matrix, detect_brand_switching
        brand_switches = detect_brand_switching(transactions_df, window_days=window_days)

    if brand_switches.empty:
        st.info("No brand switching detected in the selected window")
        return

    st.subheader("Brand Switching Events")
    brand_matrix = compute_brand_switching_matrix(brand_switches)

    if not brand_matrix.empty:
        st.dataframe(
            brand_matrix.style.format(
                {
                    "switch_count": "{:,}",
                    "unique_customers": "{:,}",
                    "avg_days_between": "{:.1f}",
                    "switch_rate": "{:.2%}",
                }
            ).background_gradient(cmap="RdYlGn", subset=["switch_rate"]),
            width="stretch",
        )

    # Category-level view
    if "category" in brand_switches.columns:
        st.subheader("Switching by Category")
        selected_cat = st.selectbox("Category", ["All"] + sorted(brand_switches["category"].unique()), key="brand_switch_cat")

        cat_switches = brand_switches if selected_cat == "All" else brand_switches[brand_switches["category"] == selected_cat]

        if not cat_switches.empty:
            cat_matrix = compute_brand_switching_matrix(cat_switches)

            if not cat_matrix.empty:
                fig = px.imshow(
                    cat_matrix.set_index(["category", "from_brand"]).loc[:, "switch_rate"].unstack(),
                    color_continuous_scale="RdYlGn",
                    title=f"Brand Switching Rates - {selected_cat}",
                    labels={"x": "To Brand", "y": "From Brand", "color": "Switch Rate"},
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, width="stretch")


def _render_loyalty_tab(loyalty: pd.DataFrame):
    """Render customer loyalty metrics."""
    st.subheader("Customer Loyalty Metrics")

    if loyalty.empty:
        st.info("No loyalty metrics available")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(loyalty))
    with col2:
        st.metric("Avg Transactions", f"{loyalty['transaction_count'].mean():.1f}")
    with col3:
        st.metric("Avg Repeat Rate", f"{loyalty['repeat_rate'].mean():.1%}")
    with col4:
        st.metric("Loyal Customers", f"{(loyalty['loyalty_segment'] == 'Loyal').sum()}")

    # Segment distribution
    st.subheader("Loyalty Segment Distribution")
    seg_counts = loyalty["loyalty_segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(seg_counts, values="Customers", names="Segment", title="Loyalty Segment Distribution")
        st.plotly_chart(fig, width="stretch")

    with col2:
        seg_rev = (
            loyalty.groupby("loyalty_segment")
            .agg(
                customers=("customer_id", "count"),
                avg_revenue=("monetary", "mean"),
                total_revenue=("monetary", "sum"),
            )
            .reset_index()
        )
        fig = px.bar(
            seg_rev,
            x="loyalty_segment",
            y="total_revenue",
            color="customers",
            title="Revenue by Loyalty Segment",
        )
        st.plotly_chart(fig, width="stretch")

    # Scatter: Frequency vs Monetary by segment
    fig = px.scatter(
        loyalty,
        x="frequency",
        y="monetary",
        color="loyalty_segment",
        size="transaction_count",
        hover_data=["customer_id", "switch_rate", "avg_days_between"],
        title="Loyalty Segments: Frequency vs Monetary",
        labels={"frequency": "Frequency", "monetary": "Monetary ($)"},
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")

    # Table
    st.subheader("Loyalty Metrics Detail")
    st.dataframe(
        loyalty.style.format(
            {
                "recency": "{:.0f}",
                "frequency": "{:.0f}",
                "monetary": "${:,.2f}",
                "avg_order_value": "${:,.2f}",
                "avg_days_between": "{:.1f}",
                "n_products": "{:.0f}",
                "n_categories": "{:.0f}",
                "avg_basket_size": "{:.1f}",
                "avg_price": "${:,.2f}",
                "weekend_ratio": "{:.1%}",
                "switch_rate": "{:.2%}",
                "purchase_frequency_per_month": "{:.2f}",
                "concentration_hhi": "{:.3f}",
            }
        ).background_gradient(cmap="RdYlGn", subset=["monetary", "frequency"]),
        width="stretch",
    )


def _render_affinity_tab(affinity_matrix: pd.DataFrame, product_lookup: dict, top_n_products: int):
    """Render the affinity heatmap tab."""
    st.subheader("Product Affinity Heatmap")

    if not affinity_matrix.empty:
        from src.viz.heatmap import create_affinity_heatmap

        fig = create_affinity_heatmap(
            affinity_matrix,
            product_lookup=product_lookup,
            min_lift=1.0,
            max_products=top_n_products,
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Affinity matrix not available")


def _render_sankey_tab(affinity_matrix: pd.DataFrame, product_lookup: dict, top_n_products: int):
    """Render the Sankey flow tab."""
    st.subheader("Co-purchase Flow (Sankey)")

    if not affinity_matrix.empty:
        from src.viz.network import create_sankey_from_matrix

        fig = create_sankey_from_matrix(
            affinity_matrix.head(min(top_n_products, 15)),
            product_lookup=product_lookup,
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Not enough data for Sankey diagram")


def _render_product_profile_tab(
    transactions_df: pd.DataFrame, product_lookup: dict, min_lift: float
):
    """Render the single product affinity profile tab."""
    st.subheader("Single Product Affinity Profile")

    products = transactions_df["stockcode"].unique()
    selected_product = st.selectbox(
        "Select Product",
        options=products,
        format_func=lambda x: product_lookup.get(x, x),
        key="copurchase_tab_product_select",
    )

    if selected_product:
        from src.analytics.copurchase import get_product_affinity_profile

        profile = get_product_affinity_profile(
            transactions_df, selected_product, min_lift=min_lift, top_n=20
        )

        if not profile.empty:
            profile["Co-purchase Product Name"] = profile["co_purchase_product"].map(product_lookup)
            profile["Target Product"] = product_lookup.get(selected_product, selected_product)

            display_cols = [
                "Co-purchase Product Name",
                "support",
                "confidence_target_to_other",
                "confidence_other_to_target",
                "lift",
                "leverage",
            ]

            st.dataframe(
                profile[display_cols].round(4),
                width="stretch",
                hide_index=True,
            )

            render_analytics_export(profile, f"Affinity_{selected_product}")

            # Bar chart
            st.subheader("Top Co-purchases by Lift")
            fig = px.bar(
                profile.head(15),
                x="lift",
                y="Co-purchase Product Name",
                orientation="h",
                color="confidence_target_to_other",
                title=f"Products co-purchased with {product_lookup.get(selected_product, selected_product)}",
                labels={"lift": "Lift", "confidence_target_to_other": "Confidence"},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width="stretch")
        else:
            st.info(f"No strong co-purchases found for {product_lookup.get(selected_product, selected_product)}")


def _render_top_pairs_tab(top_pairs: pd.DataFrame, min_lift: float):
    """Render the top co-purchase pairs tab."""
    st.subheader(f"Top {len(top_pairs)} Co-purchase Pairs (Lift ≥ {min_lift})")

    display_cols = [
        "Product A Name",
        "Product B Name",
        "support",
        "confidence_a_to_b",
        "confidence_b_to_a",
        "lift",
        "leverage",
    ]

    st.dataframe(top_pairs[display_cols].round(4), width="stretch", hide_index=True)

    render_analytics_export(top_pairs, "CoPurchase_Pairs")

    # Scatter plot
    st.subheader("Support vs Lift")
    fig = px.scatter(
        top_pairs,
        x="support",
        y="lift",
        color="confidence_a_to_b",
        size="confidence_a_to_b",
        hover_data=[
            "Product A Name",
            "Product B Name",
            "confidence_a_to_b",
            "confidence_b_to_a",
        ],
        title="Co-purchase Pairs: Support vs Lift",
        labels={
            "support": "Support",
            "lift": "Lift",
            "confidence_a_to_b": "Confidence A→B",
        },
    )
    st.plotly_chart(fig, width="stretch")


def _render_cdt_results_tabs(
    root,
    metadata: dict,
    similarity_matrix: pd.DataFrame,
    switching_df: pd.DataFrame,
    substitution_df: pd.DataFrame,
    bundling_df: pd.DataFrame,
    linkage_matrix: np.ndarray,
    ordered_labels: list[str],
    silhouette_scores: dict[int, float],
    optimal_k: int,
    product_lookup: dict,
    similarity_method: str = "phi",
):
    """Render the CDT results tabs."""
    # Quality summary at top
    render_quality_summary(metadata)

    # Main tabs
    tabs = st.tabs(
        [
            " CDT Sunburst",
            " CDT Treemap",
            " Dendrogram & Clusters",
            " Similarity Heatmap",
            " Switching Analysis",
            " Substitution Analysis",
            " Bundling Opportunities",
            " Export",
        ]
    )

    with tabs[0]:
        st.subheader("Customer Decision Tree - Sunburst View")
        st.caption(
            "Hierarchical tree from bottom-up clustering. Inner rings = higher-level splits (attributes). Outer rings = product leaves."
        )

        fig = plot_sunburst(
            root,
            title=f"CDT: {metadata['n_leaves']} leaf clusters, {metadata['max_depth']} levels",
            height=700,
        )
        st.plotly_chart(fig, width="stretch")

        # Tree statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Nodes", metadata["n_nodes"])
        with col2:
            st.metric("Leaf Clusters", metadata["n_leaves"])
        with col3:
            st.metric("Max Depth", metadata["max_depth"])

    with tabs[1]:
        st.subheader("Customer Decision Tree - Treemap View")
        st.caption("Area proportional to number of products. Color = split attribute.")

        size_metric = st.radio("Size Metric", ["size", "similarity_within"], horizontal=True)
        fig = plot_treemap(root, size_metric=size_metric, height=700)
        st.plotly_chart(fig, width="stretch")

    with tabs[2]:
        st.subheader("Hierarchical Clustering Dendrogram")

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = plot_dendrogram(linkage_matrix, ordered_labels, height=600)
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Cluster Quality")
            st.metric("Optimal k (Silhouette)", optimal_k)
            st.metric("Silhouette Score", f"{silhouette_scores.get(optimal_k, 0):.3f}")

            # Show silhouette chart
            fig_sil = plot_silhouette_scores(silhouette_scores, optimal_k, height=300)
            st.plotly_chart(fig_sil, width="stretch")

    with tabs[3]:
        st.subheader("Product Similarity Matrix")
        st.caption(
            f"Method: {similarity_method.upper()}. Red = dissimilar, Blue = similar (substitutable)."
        )

        top_n = st.slider("Top N Products", 10, min(100, len(similarity_matrix)), 50)
        fig = plot_similarity_heatmap(similarity_matrix, top_n=top_n, height=600)
        st.plotly_chart(fig, width="stretch")

    with tabs[4]:
        st.subheader("Switching Analysis")
        st.caption("Product-to-product switching rates from customer purchase sequences.")

        if not switching_df.empty:
            col1, col2 = st.columns([2, 1])
            with col1:
                # Heatmap
                switch_matrix = switching_matrix_to_heatmap(switching_df, top_n=30)
                if not switch_matrix.empty:
                    fig = plot_behavioral_heatmap(
                        switch_matrix,
                        title="Switching Rate Matrix",
                        height=500,
                        colorscale="Reds",
                    )
                    st.plotly_chart(fig, width="stretch")

            with col2:
                # Top switching paths
                st.write("**Top Switching Paths**")
                top_switches = get_top_switching_paths(switching_df, top_n=15)
                if not top_switches.empty:
                    display_df = top_switches.copy()
                    display_df["from"] = display_df["from_product"].map(
                        lambda x: product_lookup.get(x, x)[:30]
                    )
                    display_df["to"] = display_df["to_product"].map(
                        lambda x: product_lookup.get(x, x)[:30]
                    )
                    st.dataframe(
                        display_df[["from", "to", "switch_count", "switch_rate"]].round(4),
                        hide_index=True,
                        width="stretch",
                    )

            # Network graph
            st.subheader("Switching Network")
            fig_net = plot_switching_network(switching_df, product_lookup, min_rate=0.05)
            st.plotly_chart(fig_net, width="stretch")

            render_analytics_export(switching_df, "CDT_Switching")
        else:
            st.info(
                "No switching data available. Need customers with repeat purchases across products."
            )

    with tabs[5]:
        st.subheader("Substitution Analysis")
        st.caption(
            "High similarity = high substitutability. Derived from co-purchase patterns (Phi coefficient)."
        )

        if not substitution_df.empty:
            # Heatmap
            top_n = st.slider("Top N Products", 10, 100, 50, key="sub_top_n")
            fig = plot_similarity_heatmap(
                substitution_df,
                top_n=top_n,
                title="Substitution Score Matrix",
                height=500,
            )
            st.plotly_chart(fig, width="stretch")

            # Top pairs
            st.write("**Top Substitutable Pairs**")
            top_subs = get_top_substitution_pairs(substitution_df, top_n=20)
            if not top_subs.empty:
                display_df = top_subs.copy()
                display_df["Product A"] = display_df["product_a"].map(
                    lambda x: product_lookup.get(x, x)[:30]
                )
                display_df["Product B"] = display_df["product_b"].map(
                    lambda x: product_lookup.get(x, x)[:30]
                )
                st.dataframe(
                    display_df[["Product A", "Product B", "substitution_score"]].round(4),
                    hide_index=True,
                    width="stretch",
                )
                render_analytics_export(top_subs, "CDT_Substitution")

    with tabs[6]:
        st.subheader("Bundling Opportunities")
        st.caption("High lift + low substitution = true complements (not substitutes).")

        if not bundling_df.empty:
            st.write("**Top Bundling Pairs**")
            top_bundles = get_top_bundling_pairs(bundling_df, top_n=20)
            if not top_bundles.empty:
                display_df = top_bundles.copy()
                display_df["Product A"] = display_df["product_a"].map(
                    lambda x: product_lookup.get(x, x)[:30]
                )
                display_df["Product B"] = display_df["product_b"].map(
                    lambda x: product_lookup.get(x, x)[:30]
                )
                st.dataframe(
                    display_df[
                        [
                            "Product A",
                            "Product B",
                            "lift",
                            "substitution",
                            "bundle_score",
                        ]
                    ].round(4),
                    hide_index=True,
                    width="stretch",
                )
                render_analytics_export(top_bundles, "CDT_Bundling")

            # Scatter: Lift vs Substitution
            st.subheader("Lift vs Substitution Tradeoff")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=bundling_df["substitution"],
                    y=bundling_df["lift"],
                    mode="markers",
                    marker={
                        "size": 8,
                        "color": bundling_df["bundle_score"],
                        "colorscale": "Viridis",
                        "showscale": True,
                        "colorbar": {"title": "Bundle Score"},
                    },
                    text=[
                        f"{product_lookup.get(a, a)} × {product_lookup.get(b, b)}"
                        for a, b in zip(
                            bundling_df["product_a"], bundling_df["product_b"], strict=False
                        )
                    ],
                    hovertemplate="%{text}<br>Substitution: %{x:.3f}<br>Lift: %{y:.3f}<extra></extra>",
                )
            )
            fig.add_vline(
                x=0.3,
                line_dash="dash",
                line_color="red",
                annotation_text="Max Sub for Bundling",
            )
            fig.add_hline(y=1.2, line_dash="dash", line_color="green", annotation_text="Min Lift")
            fig.update_layout(
                title="Bundling Sweet Spot: High Lift + Low Substitution",
                xaxis_title="Substitution Score",
                yaxis_title="Lift",
                height=500,
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, width="stretch")

    with tabs[7]:
        st.subheader("Export Results")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Tree Structure (JSON)**")
            if st.button("Export CDT as JSON"):
                json_str = tree_to_json(root)
                st.download_button(
                    "Download JSON",
                    json_str,
                    file_name="cdt_tree.json",
                    mime="application/json",
                )

        with col2:
            st.write("**Tree Structure (CSV)**")
            if st.button("Export CDT as CSV"):
                df = tree_to_dataframe(root)
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    file_name="cdt_tree.csv",
                    mime="text/csv",
                )

        st.divider()

        # Export behavioral matrices
        st.write("**Behavioral Matrices (Excel)**")
        export_cols = st.columns(3)
        with export_cols[0]:
            if not switching_df.empty:
                csv = switching_df.to_csv(index=False)
                st.download_button("Switching Matrix", csv, "cdt_switching.csv", "text/csv")
        with export_cols[1]:
            if not substitution_df.empty:
                # Export top pairs only
                top_subs = get_top_substitution_pairs(substitution_df, top_n=100)
                csv = top_subs.to_csv(index=False)
                st.download_button("Substitution Pairs", csv, "cdt_substitution.csv", "text/csv")
        with export_cols[2]:
            if not bundling_df.empty:
                top_bundles = get_top_bundling_pairs(bundling_df, top_n=100)
                csv = top_bundles.to_csv(index=False)
                st.download_button("Bundling Pairs", csv, "cdt_bundling.csv", "text/csv")


def detect_attribute_columns(df: pd.DataFrame) -> list[str]:
    """Detect common product attribute columns."""
    candidates = [
        "category",
        "brand",
        "size",
        "flavor",
        "color",
        "variant",
        "type",
        "style",
        "material",
        "collection",
        "line",
        "range",
        "pack_size",
        "unit",
        "weight",
        "volume",
        "scent",
        "design",
        "theme",
        "occasion",
        "target_audience",
        "gender",
        "age_group",
    ]
    return [c for c in candidates if c in df.columns]


def render_quality_summary(metadata: dict):
    """Render CDT quality metrics at top of results."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta = " Pass" if metadata["passed_threshold"] else " Fail"
        st.metric(
            "Tree Quality vs Baseline",
            f"{metadata['quality_ratio']:.1%}",
            delta=delta,
            help="CDT threshold: 60%",
        )

    with col2:
        st.metric("Tree Quality Score", f"{metadata['tree_quality']:.3f}")

    with col3:
        st.metric("Unconstrained Baseline", f"{metadata['unconstrained_baseline']:.3f}")

    with col4:
        st.metric("Threshold", f"{metadata['quality_threshold']:.0%}")

    if not metadata["passed_threshold"]:
        st.warning(
            f" Tree quality ({metadata['quality_ratio']:.1%}) is below the "
            f"{metadata['quality_threshold']:.0%} threshold. "
            f"Consider: lowering min_cluster_size, adding more attributes, "
            f"or using a different similarity method."
        )


def render_cdt_tab(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render Customer Decision Tree & Patterns tab."""
    st.header(" Customer Decision Tree & Patterns (CDT)")

    if transactions_df.empty:
        st.warning("No transaction data available")
        return

    # ============================================================
    # CHECK FOR EXISTING RESULTS IN SESSION STATE
    # ============================================================
    has_results = "cdt_root" in st.session_state

    if has_results:
        # Results exist - render them directly
        root = st.session_state["cdt_root"]
        metadata = st.session_state["cdt_metadata"]
        similarity_matrix = st.session_state["cdt_similarity_matrix"]
        switching_df = st.session_state["cdt_switching_df"]
        substitution_df = st.session_state["cdt_substitution_df"]
        bundling_df = st.session_state["cdt_bundling_df"]
        linkage_matrix = st.session_state["cdt_linkage_matrix"]
        ordered_labels = st.session_state["cdt_ordered_labels"]
        silhouette_scores = st.session_state["cdt_silhouette_scores"]
        optimal_k = st.session_state["cdt_optimal_k"]
        similarity_method = st.session_state.get("cdt_similarity_method", "phi")
        product_lookup = st.session_state.get("cdt_product_lookup", product_lookup)

        # Add a button to clear results and reconfigure
        if st.button(" Reconfigure & Rebuild", type="secondary"):
            # Clear session state
            for key in list(st.session_state.keys()):
                if key.startswith("cdt_"):
                    del st.session_state[key]
            st.rerun()

        # Render results tabs
        _render_cdt_results_tabs(
            root,
            metadata,
            similarity_matrix,
            switching_df,
            substitution_df,
            bundling_df,
            linkage_matrix,
            ordered_labels,
            silhouette_scores,
            optimal_k,
            product_lookup,
            similarity_method,
        )
        return

    # ============================================================
    # NO RESULTS YET - SHOW CONFIGURATION PANEL
    # ============================================================
    _render_cdt_config_panel(transactions_df, product_lookup, params)


def _render_cdt_config_panel(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render the CDT configuration panel and run pipeline when button clicked."""

    # ============================================================
    # CONFIGURATION PANEL
    # ============================================================
    with st.expander(" CDT Configuration", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Similarity")
            similarity_method = st.selectbox(
                "Similarity Method",
                ["phi", "jaccard"],
                index=0,
                help="Phi coefficient ([-1,1]) or Jaccard ([0,1])",
            )
            min_cooccurrence = st.slider(
                "Min Co-occurrence",
                2,
                20,
                params.get("min_cooccurrence", 5),
                help="Min customers buying both products",
            )

        with col2:
            st.subheader("Clustering")
            linkage_method = st.selectbox(
                "Linkage Method",
                ["average", "complete", "single"],
                index=0,
                help="Average = default",
            )
            min_k = st.slider("Min Clusters (k)", 2, 10, params.get("min_k", 2), key="cdt_min_k")
            max_k = st.slider("Max Clusters (k)", 3, 20, params.get("max_k", 15), key="cdt_max_k")

        with col3:
            st.subheader("Tree Building")
            min_cluster_size = st.slider(
                "Min Cluster Size",
                2,
                10,
                params.get("min_cluster_size", 3),
                help="Min products per tree node",
            )
            quality_threshold = (
                st.slider(
                    "Quality Threshold (%)",
                    40,
                    80,
                    params.get("quality_threshold", 60),
                    help="Tree quality vs unconstrained baseline (≥ 60%)",
                )
                / 100.0
            )

        with col4:
            st.subheader("Behavioral")
            top_n_products = st.slider(
                "Top N Products",
                20,
                200,
                params.get("top_n_products", 50),
                help="Limit analysis to top products",
            )
            min_lift = st.slider("Min Lift for Bundling", 1.0, 3.0, 1.2, 0.1)
            max_sub = st.slider("Max Substitution for Bundling", 0.0, 0.5, 0.3, 0.05)

    # Detect attribute columns
    attribute_cols = detect_attribute_columns(transactions_df)
    if attribute_cols:
        st.info(f" Detected attribute columns: {', '.join(attribute_cols)}")
        selected_attrs = st.multiselect(
            "Attributes for Tree Enrichment",
            attribute_cols,
            default=attribute_cols,
            help="These will be tested as split criteria",
        )
    else:
        st.warning(
            "No attribute columns detected (category, brand, size, flavor, etc.). Tree will be built from similarity only."
        )
        selected_attrs = []

    # Category filter
    if "category" in transactions_df.columns:
        categories = ["All"] + sorted(transactions_df["category"].unique().tolist())
        selected_category = st.selectbox("Filter by Category", categories)
        if selected_category != "All":
            transactions_df = transactions_df[transactions_df["category"] == selected_category].copy()
            st.info(
                f"Filtered to category: {selected_category} ({len(transactions_df)} transactions)"
            )

    # ============================================================
    # RUN PIPELINE
    # ============================================================
    run_button = st.button(" Build Customer Decision Tree", type="primary", width="stretch")

    if not run_button:
        st.info("Configure parameters above and click **Build Customer Decision Tree** to start.")
        return

    # Progress tracking
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

    try:
        # Step 1: Build customer sequences
        status_text.text("Step 1/6: Building customer purchase sequences...")
        progress_bar.progress(10)

        sequences = _cached_build_customer_sequences(transactions_df)

        # Step 2: Compute similarity matrix
        status_text.text("Step 2/6: Computing similarity matrix...")
        progress_bar.progress(25)

        similarity_matrix = _cached_build_similarity_matrix(
            transactions_df,
            method=similarity_method,
            min_cooccurrence=min_cooccurrence,
        )

        if similarity_matrix.empty:
            st.error("Could not compute similarity matrix. Check data and parameters.")
            return

        # Limit to top products if specified
        if top_n_products and len(similarity_matrix) > top_n_products:
            # Use average similarity as importance
            avg_sim = similarity_matrix.mean(axis=1).sort_values(ascending=False)
            top_products = avg_sim.head(top_n_products).index.tolist()
            similarity_matrix = similarity_matrix.loc[top_products, top_products]
            sequences = {
                c: [p for p in prods if p in top_products] for c, prods in sequences.items()
            }
            sequences = {c: p for c, p in sequences.items() if p}

        # Step 3: Hierarchical clustering
        status_text.text("Step 3/6: Performing hierarchical clustering...")
        progress_bar.progress(40)

        linkage_matrix, ordered_labels = _cached_perform_hierarchical_clustering(
            similarity_matrix,
            linkage_method=linkage_method,
            distance_method=similarity_method,
        )

        # Find optimal k
        optimal_k, silhouette_scores = _cached_find_optimal_clusters(
            linkage_matrix,
            similarity_matrix,
            distance_method=similarity_method,
            min_clusters=min_k,
            max_clusters=min(max_k, len(similarity_matrix) - 1),
        )

        # Get cluster assignments
        cluster_assignments = _cached_get_cluster_assignments(
            linkage_matrix, similarity_matrix, n_clusters=optimal_k
        )

        # Step 4: Build CDT
        status_text.text("Step 4/6: Building Customer Decision Tree...")
        progress_bar.progress(55)

        # Extract product attributes
        attributes_df = extract_product_attributes(transactions_df, attribute_cols=selected_attrs)

        root, metadata = build_cdt(
            similarity_matrix,
            cluster_assignments,
            attributes_df,
            min_cluster_size=min_cluster_size,
            quality_threshold=quality_threshold,
            candidate_attributes=selected_attrs if selected_attrs else None,
        )

        # Step 5: Compute behavioral matrices
        status_text.text("Step 5/6: Computing behavioral matrices...")
        progress_bar.progress(70)

        # Need affinity matrix for bundling
        basket = create_basket_matrix(transactions_df)
        if len(basket.columns) > top_n_products:
            basket = basket[top_products] if "top_products" in locals() else basket.iloc[:, :top_n_products]

        freq_items = run_fpgrowth(basket, min_support=0.001, max_len=2)
        affinity_matrix = pd.DataFrame(
            1.0, index=basket.columns, columns=basket.columns, dtype=float
        )
        if not freq_items.empty:
            pairs = freq_items[freq_items["length"] == 2]
            product_probs = basket.mean()
            for _, row in pairs.iterrows():
                items = list(row["itemsets"])
                if len(items) == 2:
                    a, b = items[0], items[1]
                    p_a = product_probs.get(a, 0)
                    p_b = product_probs.get(b, 0)
                    if p_a > 0 and p_b > 0:
                        lift = row["support"] / (p_a * p_b)
                        affinity_matrix.loc[a, b] = lift
                        affinity_matrix.loc[b, a] = lift

        switching_df, substitution_df, bundling_df = build_behavioral_matrices(
            transactions_df,
            similarity_matrix,
            affinity_matrix,
            sequences,
            top_n_products=top_n_products,
        )

        # Step 6: Visualizations
        status_text.text("Step 6/6: Generating visualizations...")
        progress_bar.progress(85)

        # Clear progress
        progress_bar.progress(100)
        status_text.text(" Complete!")

        # Store in session for export
        st.session_state["cdt_root"] = root
        st.session_state["cdt_metadata"] = metadata
        st.session_state["cdt_similarity_matrix"] = similarity_matrix
        st.session_state["cdt_switching_df"] = switching_df
        st.session_state["cdt_substitution_df"] = substitution_df
        st.session_state["cdt_bundling_df"] = bundling_df
        st.session_state["cdt_linkage_matrix"] = linkage_matrix
        st.session_state["cdt_ordered_labels"] = ordered_labels
        st.session_state["cdt_silhouette_scores"] = silhouette_scores
        st.session_state["cdt_optimal_k"] = optimal_k
        st.session_state["cdt_sequences"] = sequences
        st.session_state["cdt_product_lookup"] = product_lookup
        st.session_state["cdt_similarity_method"] = similarity_method

    except Exception as e:
        st.error(f"Pipeline failed: {str(e)}")
        st.code(traceback.format_exc())
        return

    # Render results tabs
    _render_cdt_results_tabs(
        root,
        metadata,
        similarity_matrix,
        switching_df,
        substitution_df,
        bundling_df,
        linkage_matrix,
        ordered_labels,
        silhouette_scores,
        optimal_k,
        product_lookup,
        similarity_method,
    )


def _render_cdt_results_tabs(
    root,
    metadata: dict,
    similarity_matrix: pd.DataFrame,
    switching_df: pd.DataFrame,
    substitution_df: pd.DataFrame,
    bundling_df: pd.DataFrame,
    linkage_matrix: np.ndarray,
    ordered_labels: list[str],
    silhouette_scores: dict[int, float],
    optimal_k: int,
    product_lookup: dict,
    similarity_method: str = "phi",
):
    """Render the CDT results tabs."""

    # Quality summary at top
    render_quality_summary(metadata)

    # Main tabs
    tabs = st.tabs(
        [
            " CDT Sunburst",
            " CDT Treemap",
            " Dendrogram & Clusters",
            " Similarity Heatmap",
            " Switching Analysis",
            " Substitution Analysis",
            " Bundling Opportunities",
            " Export",
        ]
    )

    with tabs[0]:
        st.subheader("Customer Decision Tree - Sunburst View")
        st.caption(
            "Hierarchical tree from bottom-up clustering. Inner rings = higher-level splits (attributes). Outer rings = product leaves."
        )

        fig = plot_sunburst(
            root,
            title=f"CDT: {metadata['n_leaves']} leaf clusters, {metadata['max_depth']} levels",
            height=700,
        )
        st.plotly_chart(fig, width="stretch")

        # Tree statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Nodes", metadata["n_nodes"])
        with col2:
            st.metric("Leaf Clusters", metadata["n_leaves"])
        with col3:
            st.metric("Max Depth", metadata["max_depth"])

    with tabs[1]:
        st.subheader("Customer Decision Tree - Treemap View")
        st.caption("Area proportional to number of products. Color = split attribute.")

        size_metric = st.radio("Size Metric", ["size", "similarity_within"], horizontal=True)
        fig = plot_treemap(root, size_metric=size_metric, height=700)
        st.plotly_chart(fig, width="stretch")

    with tabs[2]:
        st.subheader("Hierarchical Clustering Dendrogram")

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = plot_dendrogram(linkage_matrix, ordered_labels, height=600)
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Cluster Quality")
            st.metric("Optimal k (Silhouette)", optimal_k)
            st.metric("Silhouette Score", f"{silhouette_scores.get(optimal_k, 0):.3f}")

            # Show silhouette chart
            fig_sil = plot_silhouette_scores(silhouette_scores, optimal_k, height=300)
            st.plotly_chart(fig_sil, width="stretch")

    with tabs[3]:
        st.subheader("Product Similarity Matrix")
        st.caption(
            f"Method: {similarity_method.upper()}. Red = dissimilar, Blue = similar (substitutable)."
        )

        top_n = st.slider("Top N Products", 10, min(100, len(similarity_matrix)), 50)
        fig = plot_similarity_heatmap(similarity_matrix, top_n=top_n, height=600)
        st.plotly_chart(fig, width="stretch")

    with tabs[4]:
        st.subheader("Switching Analysis")
        st.caption("Product-to-product switching rates from customer purchase sequences.")

        if not switching_df.empty:
            col1, col2 = st.columns([2, 1])
            with col1:
                # Heatmap
                switch_matrix = switching_matrix_to_heatmap(switching_df, top_n=30)
                if not switch_matrix.empty:
                    fig = plot_behavioral_heatmap(
                        switch_matrix,
                        title="Switching Rate Matrix",
                        height=500,
                        colorscale="Reds",
                    )
                    st.plotly_chart(fig, width="stretch")

            with col2:
                # Top switching paths
                st.write("**Top Switching Paths**")
                top_switches = get_top_switching_paths(switching_df, top_n=15)
                if not top_switches.empty:
                    display_df = top_switches.copy()
                    display_df["from"] = display_df["from_product"].map(
                        lambda x: product_lookup.get(x, x)[:30]
                    )
                    display_df["to"] = display_df["to_product"].map(
                        lambda x: product_lookup.get(x, x)[:30]
                    )
                    st.dataframe(
                        display_df[["from", "to", "switch_count", "switch_rate"]].round(4),
                        hide_index=True,
                        width="stretch",
                    )

            # Network graph
            st.subheader("Switching Network")
            fig_net = plot_switching_network(switching_df, product_lookup, min_rate=0.05)
            st.plotly_chart(fig_net, width="stretch")

            render_analytics_export(switching_df, "CDT_Switching")
        else:
            st.info(
                "No switching data available. Need customers with repeat purchases across products."
            )

    with tabs[5]:
        st.subheader("Substitution Analysis")
        st.caption(
            "High similarity = high substitutability. Derived from co-purchase patterns (Phi coefficient)."
        )

        if not substitution_df.empty:
            # Heatmap
            top_n = st.slider("Top N Products", 10, 100, 50, key="sub_top_n")
            fig = plot_similarity_heatmap(
                substitution_df,
                top_n=top_n,
                title="Substitution Score Matrix",
                height=500,
            )
            st.plotly_chart(fig, width="stretch")

            # Top pairs
            st.write("**Top Substitutable Pairs**")
            top_subs = get_top_substitution_pairs(substitution_df, top_n=20)
            if not top_subs.empty:
                display_df = top_subs.copy()
                display_df["Product A"] = display_df["product_a"].map(
                    lambda x: product_lookup.get(x, x)[:30]
                )
                display_df["Product B"] = display_df["product_b"].map(
                    lambda x: product_lookup.get(x, x)[:30]
                )
                st.dataframe(
                    display_df[["Product A", "Product B", "substitution_score"]].round(4),
                    hide_index=True,
                    width="stretch",
                )
                render_analytics_export(top_subs, "CDT_Substitution")

    with tabs[6]:
        st.subheader("Bundling Opportunities")
        st.caption("High lift + low substitution = true complements (not substitutes).")

        if not bundling_df.empty:
            st.write("**Top Bundling Pairs**")
            top_bundles = get_top_bundling_pairs(bundling_df, top_n=20)
            if not top_bundles.empty:
                display_df = top_bundles.copy()
                display_df["Product A"] = display_df["product_a"].map(
                    lambda x: product_lookup.get(x, x)[:30]
                )
                display_df["Product B"] = display_df["product_b"].map(
                    lambda x: product_lookup.get(x, x)[:30]
                )
                st.dataframe(
                    display_df[
                        [
                            "Product A",
                            "Product B",
                            "lift",
                            "substitution",
                            "bundle_score",
                        ]
                    ].round(4),
                    hide_index=True,
                    width="stretch",
                )
                render_analytics_export(top_bundles, "CDT_Bundling")

            # Scatter: Lift vs Substitution
            st.subheader("Lift vs Substitution Tradeoff")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=bundling_df["substitution"],
                    y=bundling_df["lift"],
                    mode="markers",
                    marker={
                        "size": 8,
                        "color": bundling_df["bundle_score"],
                        "colorscale": "Viridis",
                        "showscale": True,
                        "colorbar": {"title": "Bundle Score"},
                    },
                    text=[
                        f"{product_lookup.get(a, a)} × {product_lookup.get(b, b)}"
                        for a, b in zip(
                            bundling_df["product_a"], bundling_df["product_b"], strict=False
                        )
                    ],
                    hovertemplate="%{text}<br>Substitution: %{x:.3f}<br>Lift: %{y:.3f}<extra></extra>",
                )
            )
            fig.add_vline(
                x=0.3,
                line_dash="dash",
                line_color="red",
                annotation_text="Max Sub for Bundling",
            )
            fig.add_hline(y=1.2, line_dash="dash", line_color="green", annotation_text="Min Lift")
            fig.update_layout(
                title="Bundling Sweet Spot: High Lift + Low Substitution",
                xaxis_title="Substitution Score",
                yaxis_title="Lift",
                height=500,
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, width="stretch")

    with tabs[7]:
        st.subheader("Export Results")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Tree Structure (JSON)**")
            if st.button("Export CDT as JSON"):
                json_str = tree_to_json(root)
                st.download_button(
                    "Download JSON",
                    json_str,
                    file_name="cdt_tree.json",
                    mime="application/json",
                )

        with col2:
            st.write("**Tree Structure (CSV)**")
            if st.button("Export CDT as CSV"):
                df = tree_to_dataframe(root)
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    file_name="cdt_tree.csv",
                    mime="text/csv",
                )

        st.divider()

        # Export behavioral matrices
        st.write("**Behavioral Matrices (Excel)**")
        export_cols = st.columns(3)
        with export_cols[0]:
            if not switching_df.empty:
                csv = switching_df.to_csv(index=False)
                st.download_button("Switching Matrix", csv, "cdt_switching.csv", "text/csv")
        with export_cols[1]:
            if not substitution_df.empty:
                # Export top pairs only
                top_subs = get_top_substitution_pairs(substitution_df, top_n=100)
                csv = top_subs.to_csv(index=False)
                st.download_button("Substitution Pairs", csv, "cdt_substitution.csv", "text/csv")
        with export_cols[2]:
            if not bundling_df.empty:
                top_bundles = get_top_bundling_pairs(bundling_df, top_n=100)
                csv = top_bundles.to_csv(index=False)
                st.download_button("Bundling Pairs", csv, "cdt_bundling.csv", "text/csv")


def detect_attribute_columns(df: pd.DataFrame) -> list[str]:
    """Detect common product attribute columns."""
    candidates = [
        "category",
        "brand",
        "size",
        "flavor",
        "color",
        "variant",
        "type",
        "style",
        "material",
        "collection",
        "line",
        "range",
        "pack_size",
        "unit",
        "weight",
        "volume",
        "scent",
        "design",
        "theme",
        "occasion",
        "target_audience",
        "gender",
        "age_group",
    ]
    return [c for c in candidates if c in df.columns]


def render_quality_summary(metadata: dict):
    """Render CDT quality metrics at top of results."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta = " Pass" if metadata["passed_threshold"] else " Fail"
        st.metric(
            "Tree Quality vs Baseline",
            f"{metadata['quality_ratio']:.1%}",
            delta=delta,
            help="CDT threshold: 60%",
        )

    with col2:
        st.metric("Tree Quality Score", f"{metadata['tree_quality']:.3f}")

    with col3:
        st.metric("Unconstrained Baseline", f"{metadata['unconstrained_baseline']:.3f}")

    with col4:
        st.metric("Threshold", f"{metadata['quality_threshold']:.0%}")

    if not metadata["passed_threshold"]:
        st.warning(
            f" Tree quality ({metadata['quality_ratio']:.1%}) is below the "
            f"{metadata['quality_threshold']:.0%} threshold. "
            f"Consider: lowering min_cluster_size, adding more attributes, "
            f"or using a different similarity method."
        )
