"""Oracle-style CDT: Visualization Module.

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

# Matplotlib color cycle to Plotly color mapping
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

    Args:
        linkage_matrix: Output from hierarchical clustering
        labels: Product labels for leaves
        height: Figure height
        width: Figure width
        color_threshold: Distance threshold for coloring clusters
        orientation: 'bottom' (horizontal) or 'left' (vertical)

    Returns:
        Plotly Figure
    """
    # Generate dendrogram data using scipy
    dendro = scipy_dendrogram(
        linkage_matrix,
        labels=labels,
        no_plot=True,
        color_threshold=color_threshold,
    )

    icoord = np.array(dendro["icoord"])
    dcoord = np.array(dendro["dcoord"])
    ivl = dendro["ivl"]  # leaf labels in order
    color_list = dendro["color_list"]

    fig = go.Figure()

    # Plot horizontal lines (cluster branches)
    for i in range(len(icoord)):
        x_vals = icoord[i]
        y_vals = dcoord[i]

        # Color based on cluster - convert matplotlib colors to Plotly
        color = color_list[i] if i < len(color_list) else "gray"
        color = MPL_TO_PLOTLY.get(color, color)  # Convert C0, C1, etc. to hex

        if orientation == "bottom":
            # Horizontal dendrogram (distance on x-axis)
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
            # Vertical dendrogram (distance on y-axis)
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

    # Add leaf labels
    leaf_x = [icoord[i][1] for i in range(len(icoord))]  # middle of each leaf

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

    # Add color threshold line if specified
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
    """
    Plot silhouette scores across different k values.

    Args:
        silhouette_scores: Dict of k -> score
        optimal_k: Recommended optimal k
        height: Figure height

    Returns:
        Plotly Figure
    """
    k_values = sorted(silhouette_scores.keys())
    scores = [silhouette_scores[k] for k in k_values]

    fig = go.Figure()

    # Bar chart
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

    # Highlight optimal
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
    """
    Convert TreeNode hierarchy to sunburst data format.

    Returns:
        (ids, labels, parents, values, colors)
    """
    ids = []
    labels = []
    parents = []
    values = []
    colors = []

    def traverse(node, parent_id=""):
        node_id = node.node_id
        ids.append(node_id)
        labels.append(node.name)
        parents.append(parent_id)

        # Value = number of products (for sizing)
        values.append(node.size)

        # Color by attribute or depth
        if node.is_leaf:
            colors.append("lightgreen")
        elif node.attribute:
            # Hash attribute name to consistent color
            attr_hash = hash(node.attribute) % 10
            color_map = [
                "lightblue",
                "lightcoral",
                "lightyellow",
                "lightpink",
                "lightcyan",
                "lightsalmon",
                "lightgreen",
                "lightgray",
                "lavender",
                "wheat",
            ]
            colors.append(color_map[attr_hash])
        else:
            colors.append("lightgray")

        for child in node.children:
            traverse(child, node_id)

    traverse(root)
    return ids, labels, parents, values, colors


def plot_sunburst(
    root,
    title: str = "Customer Decision Tree",
    height: int = 700,
    width: int = 900,
) -> go.Figure:
    """
    Create interactive Sunburst chart of CDT hierarchy.

    Args:
        root: TreeNode root of CDT
        title: Chart title
        height: Figure height
        width: Figure width

    Returns:
        Plotly Figure
    """
    ids, labels, parents, values, colors = tree_to_sunburst_data(root)

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            maxdepth=-1,  # Show all levels
            insidetextorientation="radial",
            marker=dict(colors=colors, line=dict(width=1, color="white")),
            hovertemplate="<b>%{label}</b><br>Products: %{value}<br>Path: %{id}<extra></extra>",
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
    """
    Convert TreeNode hierarchy to treemap data format.

    Args:
        root: TreeNode root
        size_metric: 'size' (product count) or 'similarity_within' or custom

    Returns:
        (ids, labels, parents, values, colors)
    """
    ids = []
    labels = []
    parents = []
    values = []
    colors = []

    def traverse(node, parent_id=""):
        node_id = node.node_id
        ids.append(node_id)
        labels.append(node.name)
        parents.append(parent_id)

        # Value for area sizing
        if size_metric == "size":
            values.append(node.size)
        elif size_metric == "similarity_within":
            values.append(
                max(node.similarity_within, 0.01) * 100
            )  # Scale for visibility
        else:
            values.append(node.size)

        # Color coding
        if node.is_leaf:
            colors.append("lightgreen")
        elif node.attribute:
            attr_hash = hash(node.attribute) % 10
            color_map = [
                "lightblue",
                "lightcoral",
                "lightyellow",
                "lightpink",
                "lightcyan",
                "lightsalmon",
                "lightgreen",
                "lightgray",
                "lavender",
                "wheat",
            ]
            colors.append(color_map[attr_hash])
        else:
            colors.append("lightgray")

        for child in node.children:
            traverse(child, node_id)

    traverse(root)
    return ids, labels, parents, values, colors


def plot_treemap(
    root,
    size_metric: str = "size",
    title: str = "Customer Decision Tree (Treemap)",
    height: int = 700,
    width: int = 900,
) -> go.Figure:
    """
    Create interactive Treemap of CDT hierarchy.

    Args:
        root: TreeNode root of CDT
        size_metric: 'size' (product count), 'similarity_within', or 'revenue'
        title: Chart title
        height: Figure height
        width: Figure width

    Returns:
        Plotly Figure
    """
    ids, labels, parents, values, colors = tree_to_treemap_data(root, size_metric)

    fig = go.Figure(
        go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(colors=colors, line=dict(width=1, color="white")),
            hovertemplate="<b>%{label}</b><br>Value: %{value}<br>Path: %{id}<extra></extra>",
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
    title: str = "Product Similarity Matrix (Yule's Q)",
    height: int = 600,
    colorscale: str = "RdBu",
) -> go.Figure:
    """
    Plot similarity matrix as heatmap.

    Args:
        similarity_matrix: Square similarity matrix
        top_n: Limit to top N products by average similarity
        title: Chart title
        height: Figure height
        colorscale: Plotly colorscale name

    Returns:
        Plotly Figure
    """
    if similarity_matrix.empty:
        return go.Figure().add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)

    # Select top N products by average similarity
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
    colorscale: str = "Viridis",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> go.Figure:
    """
    Plot square behavioral matrix (switching, substitution, bundling) as heatmap.

    Args:
        matrix_df: Square matrix DataFrame
        title: Chart title
        height: Figure height
        colorscale: Plotly colorscale
        zmin, zmax: Color scale bounds

    Returns:
        Plotly Figure
    """
    if matrix_df.empty:
        return go.Figure().add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)

    fig = go.Figure(
        go.Heatmap(
            z=matrix_df.values,
            x=matrix_df.columns.tolist(),
            y=matrix_df.index.tolist(),
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=title),
            hovertemplate="%{y} vs %{x}<br>Value: %{z:.3f}<extra></extra>",
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

    Args:
        switching_df: DataFrame with from_product, to_product, switch_rate
        product_lookup: Dict mapping product_id -> product_name
        min_rate: Minimum switch rate to show edge
        max_edges: Maximum number of edges to display
        height: Figure height

    Returns:
        Plotly Figure
    """
    if switching_df.empty:
        return go.Figure().add_annotation(
            text="No switching data", x=0.5, y=0.5, showarrow=False
        )

    # Filter and limit edges
    df = switching_df[switching_df["switch_rate"] >= min_rate].copy()
    df = df.sort_values("switch_rate", ascending=False).head(max_edges)

    # Get unique nodes
    nodes = set(df["from_product"]) | set(df["to_product"])
    node_list = list(nodes)
    node_pos = {node: i for i, node in enumerate(node_list)}

    # Circular layout
    n_nodes = len(node_list)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    x_coords = np.cos(angles)
    y_coords = np.sin(angles)

    fig = go.Figure()

    # Add edges
    for _, row in df.iterrows():
        from_idx = node_pos[row["from_product"]]
        to_idx = node_pos[row["to_product"]]

        fig.add_trace(
            go.Scatter(
                x=[x_coords[from_idx], x_coords[to_idx]],
                y=[y_coords[from_idx], y_coords[to_idx]],
                mode="lines",
                line=dict(
                    width=1 + row["switch_rate"] * 10,
                    color=f"rgba(200, 0, 0, {0.3 + row['switch_rate'] * 0.7})",
                ),
                hoverinfo="text",
                hovertext=f"{product_lookup.get(row['from_product'], row['from_product'])} → "
                f"{product_lookup.get(row['to_product'], row['to_product'])}<br>"
                f"Rate: {row['switch_rate']:.1%}",
                showlegend=False,
            )
        )

    # Add nodes
    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers+text",
            marker=dict(
                size=20, color="lightblue", line=dict(width=2, color="darkblue")
            ),
            text=[product_lookup.get(n, n)[:15] for n in node_list],
            textposition="top center",
            textfont=dict(size=9),
            hoverinfo="text",
            hovertext=[product_lookup.get(n, n) for n in node_list],
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
