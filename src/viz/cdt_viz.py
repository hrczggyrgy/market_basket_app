"""CDT: Visualization Module.

Provides Plotly visualizations for:
- Dendrogram (cluster hierarchy)
- Sunburst (CDT hierarchy - default view)
- Treemap (CDT hierarchy with size encoding)
- Similarity Heatmap (pairwise relationships)
- Behavioral matrices (switching, substitution, bundling)
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

from src.analytics.cdt_tree_builder import tree_to_dataframe

# Matplotlib color cycle -> Plotly color mapping
MPL_TO_PLOTLY = {
    "C0": "#1f77b4",
    "C1": "#ff7f0e",
    "C2": "#2ca02c",
    "C3": "#d62728",
    "C4": "#9467bd",
    "C5": "#8c564b",
    "C6": "#e377c2",
    "C7": "#7f7f7f",
    "C8": "#bcbd22",
    "C9": "#17becf",
}


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels: List[str],
    height: int = 600,
    width: int = 900,
    color_threshold: Optional[float] = None,
    orientation: str = "bottom",
) -> go.Figure:
    """
    Create interactive dendrogram using scipy's dendrogram + Plotly.
    """
    dendro = scipy_dendrogram(
        linkage_matrix,
        labels=labels,
        no_plot=True,
        color_threshold=color_threshold,
    )

    icoord = np.array(dendro["icoord"])
    dcoord = np.array(dendro["dcoord"])
    ivl = dendro["ivl"]
    color_list = dendro["color_list"]

    fig = go.Figure()

    for i in range(len(icoord)):
        x_vals = icoord[i]
        y_vals = dcoord[i]
        color = color_list[i] if i < len(color_list) else "gray"
        color = MPL_TO_PLOTLY.get(color, color)

        if orientation == "bottom":
            fig.add_trace(
                go.Scatter(
                    x=y_vals,
                    y=x_vals,
                    mode="lines",
                    line=dict(color=color, width=2),
                    hoverinfo="none",
                    showlegend=False,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    line=dict(color=color, width=2),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

    leaf_x = [icoord[i][1] for i in range(len(icoord))]

    if orientation == "bottom":
        fig.add_trace(
            go.Scatter(
                x=[0] * len(leaf_x),
                y=leaf_x,
                mode="text",
                text=ivl,
                textposition="middle right",
                textfont=dict(size=10, color="black"),
                hoverinfo="text",
                hovertext=ivl,
                showlegend=False,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=leaf_x,
                y=[0] * len(leaf_x),
                mode="text",
                text=ivl,
                textposition="top center",
                textfont=dict(size=10, color="black"),
                hoverinfo="text",
                hovertext=ivl,
                showlegend=False,
            )
        )

    if color_threshold is not None:
        if orientation == "bottom":
            fig.add_vline(
                x=color_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {color_threshold:.2f}",
            )
        else:
            fig.add_hline(
                y=color_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {color_threshold:.2f}",
            )

    fig.update_layout(
        title=dict(text="Hierarchical Clustering Dendrogram", x=0.5),
        xaxis_title="Distance" if orientation == "bottom" else "Product",
        yaxis_title="Product" if orientation == "bottom" else "Distance",
        height=height,
        width=width,
        showlegend=False,
        hovermode="closest",
        margin=dict(l=150, r=50, t=80, b=50),
        plot_bgcolor="white",
    )

    if orientation == "bottom":
        fig.update_xaxes(showgrid=True, gridcolor="lightgray")
        fig.update_yaxes(showgrid=False, autorange="reversed")
    else:
        fig.update_yaxes(showgrid=True, gridcolor="lightgray")
        fig.update_xaxes(showgrid=False)

    return fig


def plot_silhouette_scores(
    silhouette_scores: Dict[int, float],
    optimal_k: int,
    height: int = 400,
) -> go.Figure:
    """Plot silhouette scores across different k values."""
    k_values = sorted(silhouette_scores.keys())
    scores = [silhouette_scores[k] for k in k_values]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=k_values,
            y=scores,
            marker_color=["red" if k == optimal_k else "steelblue" for k in k_values],
            text=[f"{s:.3f}" for s in scores],
            textposition="auto",
            name="Silhouette Score",
        )
    )

    if optimal_k in silhouette_scores:
        fig.add_annotation(
            x=optimal_k,
            y=silhouette_scores[optimal_k],
            text=f"Optimal k={optimal_k}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            ax=0,
            ay=-40,
            bgcolor="white",
            bordercolor="red",
        )

    fig.update_layout(
        title=dict(text="Silhouette Score vs Number of Clusters", x=0.5),
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score",
        height=height,
        plot_bgcolor="white",
        showlegend=False,
    )
    return fig


def tree_to_sunburst_data(
    root,
) -> Tuple[List[str], List[str], List[str], List[float], List[str]]:
    """Convert TreeNode hierarchy to sunburst data format."""
    ids: List[str] = []
    labels: List[str] = []
    parents: List[str] = []
    values: List[float] = []
    colors: List[str] = []

    _COLOR_MAP = [
        "lightblue", "lightcoral", "lightyellow", "lightpink", "lightcyan",
        "lightsalmon", "lightgreen", "lightgray", "lavender", "wheat",
    ]

    def traverse(node, parent_id: str = "") -> None:
        ids.append(node.node_id)
        labels.append(node.name)
        parents.append(parent_id)
        values.append(node.size)
        if node.is_leaf:
            colors.append("lightgreen")
        elif node.attribute:
            colors.append(_COLOR_MAP[hash(node.attribute) % 10])
        else:
            colors.append("lightgray")
        for child in node.children:
            traverse(child, node.node_id)

    traverse(root)
    return ids, labels, parents, values, colors


def plot_sunburst(
    root,
    title: str = "Customer Decision Tree",
    height: int = 700,
    width: int = 900,
) -> go.Figure:
    """Create interactive Sunburst chart of CDT hierarchy."""
    ids, labels, parents, values, colors = tree_to_sunburst_data(root)

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            maxdepth=-1,
            insidetextorientation="radial",
            marker=dict(colors=colors, line=dict(width=1, color="white")),
            hovertemplate=(
                "<b>%{label}</b><br>Products: %{value}<br>Path: %{id}<extra></extra>"
            ),
            textinfo="label+value",
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        height=height,
        width=width,
        margin=dict(t=60, l=10, r=10, b=10),
        sunburstcolorway=["lightblue", "lightgreen", "lightcoral", "lightyellow"],
    )
    return fig


def tree_to_treemap_data(
    root, size_metric: str = "size"
) -> Tuple[List[str], List[str], List[str], List[float], List[str]]:
    """Convert TreeNode hierarchy to treemap data format."""
    ids: List[str] = []
    labels: List[str] = []
    parents: List[str] = []
    values: List[float] = []
    colors: List[str] = []

    _COLOR_MAP = [
        "lightblue", "lightcoral", "lightyellow", "lightpink", "lightcyan",
        "lightsalmon", "lightgreen", "lightgray", "lavender", "wheat",
    ]

    def traverse(node, parent_id: str = "") -> None:
        ids.append(node.node_id)
        labels.append(node.name)
        parents.append(parent_id)
        if size_metric == "similarity_within":
            values.append(max(node.similarity_within, 0.01) * 100)
        else:
            values.append(node.size)
        if node.is_leaf:
            colors.append("lightgreen")
        elif node.attribute:
            colors.append(_COLOR_MAP[hash(node.attribute) % 10])
        else:
            colors.append("lightgray")
        for child in node.children:
            traverse(child, node.node_id)

    traverse(root)
    return ids, labels, parents, values, colors


def plot_treemap(
    root,
    size_metric: str = "size",
    title: str = "Customer Decision Tree (Treemap)",
    height: int = 700,
    width: int = 900,
) -> go.Figure:
    """Create interactive Treemap of CDT hierarchy."""
    ids, labels, parents, values, colors = tree_to_treemap_data(root, size_metric)

    fig = go.Figure(
        go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(colors=colors, line=dict(width=1, color="white")),
            hovertemplate=(
                "<b>%{label}</b><br>Value: %{value}<br>Path: %{id}<extra></extra>"
            ),
            textinfo="label+value",
            textfont=dict(size=11),
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        height=height,
        width=width,
        margin=dict(t=60, l=10, r=10, b=10),
    )
    return fig


def plot_similarity_heatmap(
    similarity_matrix: pd.DataFrame,
    top_n: int = 50,
    title: str = "Product Similarity Matrix (Phi)",
    height: int = 600,
    colorscale: str = "RdBu",
) -> go.Figure:
    """Plot similarity matrix as heatmap."""
    if similarity_matrix.empty:
        return go.Figure().add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)

    if top_n and len(similarity_matrix) > top_n:
        avg_sim = similarity_matrix.mean(axis=1).sort_values(ascending=False)
        top_products = avg_sim.head(top_n).index.tolist()
        matrix = similarity_matrix.loc[top_products, top_products]
    else:
        matrix = similarity_matrix

    fig = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale=colorscale,
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Similarity"),
            hovertemplate="%{y} vs %{x}<br>Similarity: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=height,
        xaxis=dict(tickangle=45, side="bottom"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=100, r=50, t=80, b=100),
    )
    return fig


def plot_behavioral_heatmap(
    matrix_df: pd.DataFrame,
    title: str = "Behavioral Matrix",
    height: int = 500,
    colorscale: str = "Reds",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> go.Figure:
    """
    Plot square behavioral matrix (switching, substitution, bundling) as heatmap.

    Bug-fix: always set zmin=0 and zmax=data_max so that small switch_rate
    values (e.g. 0.001-0.03) are not collapsed to a flat colour by Plotly
    autoscaling.
    """
    if matrix_df.empty:
        return go.Figure().add_annotation(
            text="No data", x=0.5, y=0.5, showarrow=False
        )

    data_max = float(matrix_df.values.max())
    resolved_zmin = 0.0 if zmin is None else zmin
    # Use a small positive zmax floor so the scale isn't degenerate when
    # all values are zero (shouldn't happen, but defensive)
    resolved_zmax = max(data_max, 1e-6) if zmax is None else zmax

    fig = go.Figure(
        go.Heatmap(
            z=matrix_df.values,
            x=matrix_df.columns.tolist(),
            y=matrix_df.index.tolist(),
            colorscale=colorscale,
            zmin=resolved_zmin,
            zmax=resolved_zmax,
            colorbar=dict(title=title),
            hovertemplate="%{y} \u2192 %{x}<br>Switch Rate: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=height,
        xaxis=dict(tickangle=45, side="bottom"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=100, r=50, t=80, b=100),
    )
    return fig


def plot_switching_network(
    switching_df: pd.DataFrame,
    product_lookup: Dict[str, str],
    min_rate: float = 0.05,
    max_edges: int = 100,
    height: int = 600,
) -> go.Figure:
    """
    Plot switching flows as directed network graph.
    """
    if switching_df.empty:
        return go.Figure().add_annotation(
            text="No switching data", x=0.5, y=0.5, showarrow=False
        )

    df = switching_df[switching_df["switch_rate"] >= min_rate].copy()

    # If no edges survive the min_rate filter, relax to show top-50 by rate
    if df.empty:
        df = switching_df.sort_values("switch_rate", ascending=False).head(50)

    df = df.sort_values("switch_rate", ascending=False).head(max_edges)

    nodes = list(set(df["from_product"]) | set(df["to_product"]))
    node_pos = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    x_coords = np.cos(angles)
    y_coords = np.sin(angles)

    fig = go.Figure()

    max_rate = float(df["switch_rate"].max()) or 1.0
    for _, row in df.iterrows():
        fi = node_pos[row["from_product"]]
        ti = node_pos[row["to_product"]]
        norm = row["switch_rate"] / max_rate
        fig.add_trace(
            go.Scatter(
                x=[x_coords[fi], x_coords[ti]],
                y=[y_coords[fi], y_coords[ti]],
                mode="lines",
                line=dict(
                    width=1 + norm * 6,
                    color=f"rgba(200, 0, 0, {0.25 + norm * 0.75})",
                ),
                hoverinfo="text",
                hovertext=(
                    f"{product_lookup.get(row['from_product'], row['from_product'])}"
                    f" \u2192 "
                    f"{product_lookup.get(row['to_product'], row['to_product'])}"
                    f"<br>Rate: {row['switch_rate']:.1%}"
                ),
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers+text",
            marker=dict(
                size=20,
                color="lightblue",
                line=dict(width=2, color="darkblue"),
            ),
            text=[product_lookup.get(n, n)[:15] for n in nodes],
            textposition="top center",
            textfont=dict(size=9),
            hoverinfo="text",
            hovertext=[product_lookup.get(n, n) for n in nodes],
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(text="Product Switching Network", x=0.5),
        height=height,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"
        ),
        margin=dict(t=60, b=20, l=20, r=20),
        plot_bgcolor="white",
    )
    return fig


def export_tree_json(root) -> str:
    """Export CDT tree as JSON string."""
    return json.dumps(root.to_dict(), indent=2)


def export_tree_csv(root) -> pd.DataFrame:
    """Export CDT tree as flat CSV DataFrame."""
    return tree_to_dataframe(root)
