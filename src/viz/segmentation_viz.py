"""Segmentation visualization module."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter


def plot_kaplan_meier(
    transactions_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    customer_col: str = "customer_id",
    date_col: str = "date",
    segment_col: str = "segment",
    height: int = 600,
    width: int = 800,
) -> go.Figure:
    """
    Plot Kaplan-Meier survival curves by segment.

    Shows P(still active) vs. days since first purchase.
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df[date_col])

    # Merge segments
    segments = segments_df[[customer_col, segment_col]].drop_duplicates()
    df = df.merge(segments, on=customer_col, how="left")
    df["segment"] = df[segment_col].fillna("Unknown")

    fig = go.Figure()

    for segment in df["segment"].unique():
        seg_df = df[df["segment"] == segment]

        # Compute inter-purchase times
        surv_data = []
        for cust_id, cust_data in seg_df.groupby(customer_col):
            dates = cust_data[date_col].drop_duplicates().sort_values().tolist()
            if len(dates) < 2:
                continue
            gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
            for g in gaps:
                surv_data.append({"duration": g, "event_observed": 1, "segment": segment})

        if not surv_data:
            continue

        surv_df = pd.DataFrame(surv_data)

        # Fit Kaplan-Meier
        kmf = KaplanMeierFitter()
        kmf.fit(surv_df["duration"], event_observed=surv_df["event_observed"])

        # Plot
        fig.add_trace(
            go.Scatter(
                x=kmf.survival_function_.index,
                y=kmf.survival_function_.iloc[:, 0],
                mode="lines",
                name=segment,
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title="Kaplan-Meier Survival Curves by Segment",
        xaxis_title="Days Since Last Purchase",
        yaxis_title="P(Still Active)",
        height=600,
        width=800,
        legend=dict(title="Segment"),
    )

    return fig


def plot_umap_embedding(
    transactions_df: pd.DataFrame,
    product_lookup: dict,
    height: int = 600,
    width: int = 800,
) -> go.Figure:
    """
    Plot UMAP 2D embedding of products colored by CDT tier.
    """
    from umap import UMAP

    from src.analytics.cdt_similarity import build_similarity_matrix

    # Build similarity matrix
    sim_matrix = build_similarity_matrix(transactions_df, method="phi", min_cooccurrence=5)

    # UMAP embedding
    umap = UMAP(n_components=2, random_state=42, min_dist=0.3, n_neighbors=15)
    embedding = umap.fit_transform(sim_matrix.values)

    embed_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"], index=sim_matrix.index)
    embed_df["product_name"] = embed_df.index.map(product_lookup)

    fig = px.scatter(
        embed_df,
        x="UMAP1",
        y="UMAP2",
        hover_data=["product_name"],
        title="Product UMAP Embedding (Phi Similarity)",
        labels={"UMAP1": "UMAP 1", "UMAP2": "UMAP 2"},
        width=width,
        height=height,
    )

    fig.update_traces(marker=dict(size=8, opacity=0.7))
    return fig


def plot_lorenz_curve(
    transactions_df: pd.DataFrame,
    customer_col: str = "customer_id",
    revenue_col: str = "revenue",
    height: int = 500,
    width: int = 700,
) -> go.Figure:
    """
    Plot Lorenz curve with Gini coefficient.
    """
    df = transactions_df.copy()
    df["revenue"] = df["price"] * df["quantity"] if "revenue" not in df.columns else df["revenue"]

    # Revenue per customer
    cust_rev = df.groupby(customer_col)[revenue_col].sum().sort_values()

    # Lorenz curve
    cum_rev = cust_rev.cumsum() / cust_rev.sum()
    cum_cust = np.arange(1, len(cust_rev) + 1) / len(cust_rev)

    # Gini coefficient
    gini = 1 - 2 * np.trapz(cum_rev, cum_cust)

    fig = go.Figure()

    # Perfect equality line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Perfect Equality",
            showlegend=True,
        )
    )

    # Lorenz curve
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([[0], cum_cust]),
            y=np.concatenate([[0], cum_rev]),
            mode="lines",
            name=f"Lorenz Curve (Gini = {gini:.3f})",
            line=dict(width=2, color="blue"),
        )
    )

    fig.update_layout(
        title=f"Lorenz Curve (Gini = {gini:.3f})",
        xaxis_title="Cumulative Share of Customers",
        yaxis_title="Cumulative Share of Revenue",
        height=500,
        width=700,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        showlegend=True,
    )

    return fig


def plot_parallel_coordinates(
    segment_profiles: pd.DataFrame,
    height: int = 600,
    width: int = 900,
) -> go.Figure:
    """
    Parallel coordinates plot for segment profiles.
    """
    fig = px.parallel_coordinates(
        segment_profiles,
        color="total_revenue"
        if "total_revenue" in segment_profiles.columns
        else segment_profiles.columns[-1],
        dimensions=segment_profiles.select_dtypes(include=[np.number]).columns.tolist(),
        color_continuous_scale="Viridis",
        labels={col: col.replace("_", " ").title() for col in segment_profiles.columns},
    )

    fig.update_layout(
        title="Segment Profiles - Parallel Coordinates",
        height=height,
        width=width,
    )

    return fig


def plot_bump_chart(
    rank_df: pd.DataFrame,
    height: int = 600,
    width: int = 900,
) -> go.Figure:
    """
    Bump chart: product rank over time.

    Args:
        rank_df: DataFrame with columns [period, stockcode, rank, product_name]
    """
    fig = px.line(
        rank_df,
        x="period",
        y="rank",
        color="product_name" if "product_name" in rank_df.columns else "stockcode",
        markers=True,
        title="Product Rank Over Time (Bump Chart)",
        labels={"period": "Period", "rank": "Rank"},
    )

    # Reverse y-axis (rank 1 at top)
    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        height=600,
        width=900,
        yaxis_title="Rank (1 = Top)",
        xaxis_title="Period",
    )

    return fig


def plot_segment_migration_sankey(
    migration_matrix: pd.DataFrame,
    segment_names: list = None,
    height: int = 600,
    width: int = 800,
) -> go.Figure:
    """
    Sankey diagram for segment migrations between two periods.

    Args:
        migration_matrix: Square DataFrame (from_segment x to_segment) with counts
        segment_names: List of segment names for display
    """
    if segment_names is None:
        segment_names = migration_matrix.index.tolist()

    # Build Sankey
    n = len(segment_names)
    labels = segment_names + segment_names

    source = []
    target = []
    value = []

    for i, from_seg in enumerate(segment_names):
        for j, to_seg in enumerate(segment_names):
            count = (
                migration_matrix.loc[from_seg, to_seg]
                if from_seg in migration_matrix.index and to_seg in migration_matrix.columns
                else 0
            )
            if count > 0:
                source.append(i)
                target.append(n + j)
                value.append(count)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color="lightblue",
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color="rgba(0, 100, 200, 0.4)",
                ),
            )
        ]
    )

    fig.update_layout(
        title="Customer Segment Migration (Sankey)",
        font_size=12,
        height=height,
        width=width,
    )

    return fig


def plot_segment_radar(
    segment_profiles: pd.DataFrame,
    features: list = None,
    height: int = 600,
    width: int = 600,
) -> go.Figure:
    """
    Radar chart for segment profiles.

    Args:
        segment_profiles: DataFrame with segments as rows, features as columns
        features: List of feature columns to plot
    """
    if features is None:
        features = segment_profiles.select_dtypes(include=[np.number]).columns.tolist()

    # Normalize features to 0-1 for radar
    normalized = segment_profiles[features].copy()
    for col in features:
        mn, mx = normalized[col].min(), normalized[col].max()
        if mx > mn:
            normalized[col] = (normalized[col] - mn) / (mx - mn)
        else:
            normalized[col] = 0.5

    fig = go.Figure()

    for segment in segment_profiles.index:
        values = normalized.loc[segment, features].tolist()
        values += values[:1]  # Close the polygon

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=features + features[:1],
                fill="toself",
                name=segment,
                line=dict(width=2),
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        title="Segment Profiles Radar Chart",
        height=height,
        width=width,
    )

    return fig


def plot_ltv_power_law(
    ltv_df: pd.DataFrame,
    fit_params: dict,
    height: int = 600,
    width: int = 800,
) -> go.Figure:
    """
    Plot Cohort LTV curves with power-law fits.
    """
    fig = go.Figure()

    for cohort_idx in ltv_df.index:
        ltv_curve = ltv_df.loc[cohort_idx].values
        periods = np.arange(1, len(ltv_curve) + 1)

        fig.add_trace(
            go.Scatter(
                x=periods,
                y=ltv_curve,
                mode="lines",
                name=f"Cohort {cohort_idx}",
                line=dict(width=2),
            )
        )

    # Add fit parameters as annotations
    for i, (cohort_idx, params) in enumerate(fit_params.items()):
        if params["r2"] > 0.5:
            fig.add_annotation(
                x=len(periods) * 0.9,
                y=ltv_df.iloc[i].values[-1] if i < len(ltv_df) else 0,
                text=f"{cohort_idx}: a={params['a']:.2f}, b={params['b']:.2f}, R²={params['r2']:.2f}",
                showarrow=False,
                font=dict(size=9),
            )

    fig.update_layout(
        title="Cohort LTV Curves (Power-Law Fit)",
        xaxis_title="Period",
        yaxis_title="Cumulative LTV ($)",
        height=height,
        width=width,
        hovermode="x unified",
    )

    return fig
