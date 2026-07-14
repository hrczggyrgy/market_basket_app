"""Network graph visualization for association rules."""

from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def create_network_graph(
    rules_df: pd.DataFrame,
    product_lookup: Dict[str, str] = None,
    min_lift: float = 1.0,
    max_nodes: int = 50,
    max_edges: int = 100,
    node_size_metric: str = "support",
    edge_width_metric: str = "lift",
    edge_color_metric: str = "confidence",
    layout: str = "spring",
    title: str = "Association Rules Network",
    height: int = 700,
) -> go.Figure:
    """
    Create interactive network graph of association rules.

    Nodes = Products (items in antecedents/consequents)
    Edges = Rules (antecedent -> consequent)
    """
    if rules_df.empty:
        return _empty_figure("No rules to display")

    # Filter rules
    filtered = rules_df[rules_df["lift"] >= min_lift].copy()
    if filtered.empty:
        return _empty_figure(f"No rules with lift >= {min_lift}")

    # Get unique items from top rules
    top_rules = filtered.nlargest(max_edges, edge_width_metric)

    # Rank items by degree (how many rules they appear in)
    item_degree = {}
    for _, row in top_rules.iterrows():
        for item in row["antecedents"]:
            item_degree[item] = item_degree.get(item, 0) + 1
        for item in row["consequents"]:
            item_degree[item] = item_degree.get(item, 0) + 1

    items = sorted(item_degree, key=item_degree.get, reverse=True)[:max_nodes]

    # Build graph
    G = nx.DiGraph()

    # Add nodes
    for item in items:
        label = product_lookup.get(str(item), str(item)) if product_lookup else str(item)
        G.add_node(item, label=label[:30])

    # Add edges from rules
    edge_data = []
    for _, row in top_rules.iterrows():
        for ant in row["antecedents"]:
            for cons in row["consequents"]:
                if ant in items and cons in items:
                    G.add_edge(
                        ant,
                        cons,
                        support=row["support"],
                        confidence=row["confidence"],
                        lift=row["lift"],
                        leverage=row.get("leverage", 0),
                        conviction=row.get("conviction", 0),
                    )
                    edge_data.append(
                        {
                            "source": ant,
                            "target": cons,
                            "support": row["support"],
                            "confidence": row["confidence"],
                            "lift": row["lift"],
                        }
                    )

    if len(G.nodes()) == 0:
        return _empty_figure("No valid nodes in graph")

    # Compute layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=2 / np.sqrt(len(G.nodes())), iterations=50, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Node sizes based on metric
    node_metrics = {}
    for node in G.nodes():
        # Aggregate metric for node (sum of incident edge metrics)
        val = 0
        for _, _, data in G.edges(node, data=True):
            val += data.get(node_size_metric, 0)
        node_metrics[node] = val if val > 0 else 0.01

    node_sizes = [node_metrics[n] for n in G.nodes()]
    min_size, max_size = 10, 50
    if max(node_sizes) > min(node_sizes):
        node_sizes = [
            min_size
            + (max_size - min_size) * (s - min(node_sizes)) / (max(node_sizes) - min(node_sizes))
            for s in node_sizes
        ]

    # Node colors - based on degree or centrality
    try:
        centrality = nx.betweenness_centrality(G)
    except Exception:
        centrality = {n: G.degree(n) for n in G.nodes()}

    node_colors = [centrality[n] for n in G.nodes()]

    # Edge traces - batch all edges into single trace for performance
    edge_x, edge_y = [], []
    edge_hover = []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        # Hover info
        hover_text = (
            f"{product_lookup.get(str(u), str(u))} → {product_lookup.get(str(v), str(v))}"
            f"<br>Support: {data['support']:.4f}"
            f"<br>Confidence: {data['confidence']:.4f}"
            f"<br>Lift: {data['lift']:.4f}"
        )

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_hover.extend([hover_text, hover_text, ""])

    # Single edge trace for all edges
    edge_traces = [
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color="rgba(100,100,100,0.5)"),
            hoverinfo="text",
            hovertext=edge_hover,
            showlegend=False,
        )
    ]

    # Node trace
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_labels = [G.nodes[n]["label"] for n in G.nodes()]
    node_text = [f"{G.nodes[n]['label']}<br>Connections: {G.degree(n)}" for n in G.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        textfont=dict(size=9, color="black"),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Centrality", thickness=10, x=1.02),
            line=dict(width=2, color="white"),
        ),
        hoverinfo="text",
        hovertext=node_text,
        showlegend=False,
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=height,
        plot_bgcolor="white",
    )

    # Add annotations for edge/node counts
    fig.add_annotation(
        text=f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | Min Lift: {min_lift}",
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.01,
        showarrow=False,
        font=dict(size=10, color="gray"),
        xanchor="left",
        yanchor="bottom",
    )

    return fig


def create_bipartite_network(
    rules_df: pd.DataFrame,
    product_lookup: Dict[str, str] = None,
    min_lift: float = 1.2,
    top_n_rules: int = 30,
    height: int = 600,
) -> go.Figure:
    """
    Create bipartite network: Antecedents on left, Consequents on right.
    """
    if rules_df.empty:
        return _empty_figure("No rules to display")

    filtered = rules_df[rules_df["lift"] >= min_lift].nlargest(top_n_rules, "lift")

    if filtered.empty:
        return _empty_figure(f"No rules with lift >= {min_lift}")

    G = nx.DiGraph()

    # Add nodes with bipartite attribute
    antecedents = set()
    consequents = set()

    for _, row in filtered.iterrows():
        for a in row["antecedents"]:
            antecedents.add(a)
        for c in row["consequents"]:
            consequents.add(c)

    for a in antecedents:
        label = product_lookup.get(str(a), str(a)) if product_lookup else str(a)
        G.add_node(a, label=label[:30], bipartite=0)

    for c in consequents:
        label = product_lookup.get(str(c), str(c)) if product_lookup else str(c)
        G.add_node(c, label=label[:30], bipartite=1)

    # Add edges
    for _, row in filtered.iterrows():
        for a in row["antecedents"]:
            for c in row["consequents"]:
                if a in antecedents and c in consequents:
                    G.add_edge(
                        a,
                        c,
                        lift=row["lift"],
                        confidence=row["confidence"],
                        support=row["support"],
                    )

    # Bipartite layout
    pos = nx.bipartite_layout(G, antecedents, align="vertical", scale=2)

    # Create traces
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="rgba(100,100,100,0.5)"),
        hoverinfo="none",
        showlegend=False,
    )

    # Antecedent nodes
    ant_x = [pos[n][0] for n in antecedents]
    ant_y = [pos[n][1] for n in antecedents]
    ant_labels = [G.nodes[n]["label"] for n in antecedents]

    ant_trace = go.Scatter(
        x=ant_x,
        y=ant_y,
        mode="markers+text",
        text=ant_labels,
        textposition="middle left",
        marker=dict(size=20, color="lightblue", line=dict(width=2, color="darkblue")),
        hoverinfo="text",
        hovertext=[f"Antecedent: {label}" for label in ant_labels],
        showlegend=False,
        name="Antecedents",
    )

    # Consequent nodes
    cons_x = [pos[n][0] for n in consequents]
    cons_y = [pos[n][1] for n in consequents]
    cons_labels = [G.nodes[n]["label"] for n in consequents]

    cons_trace = go.Scatter(
        x=cons_x,
        y=cons_y,
        mode="markers+text",
        text=cons_labels,
        textposition="middle right",
        marker=dict(size=20, color="lightcoral", line=dict(width=2, color="darkred")),
        hoverinfo="text",
        hovertext=[f"Consequent: {label}" for label in cons_labels],
        showlegend=False,
        name="Consequents",
    )

    fig = go.Figure(data=[edge_trace, ant_trace, cons_trace])

    fig.update_layout(
        title=dict(text="Bipartite Rule Network (Antecedents → Consequents)", x=0.5),
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=height,
        plot_bgcolor="white",
    )

    return fig


def create_sankey_from_matrix(
    matrix: pd.DataFrame,
    product_lookup: dict = None,
    title: str = "Product Co-purchase Flow",
) -> go.Figure:
    """Create Sankey diagram from affinity matrix."""
    products = matrix.index.tolist()

    # Create source/target for upper triangle only
    sources = []
    targets = []
    values = []

    for i, src in enumerate(products):
        for j, tgt in enumerate(products):
            if i < j and matrix.iloc[i, j] > 1.0:
                sources.append(i)
                targets.append(j)
                values.append(matrix.iloc[i, j] - 1.0)  # excess over independence

    if not values:
        return _empty_figure("No strong relationships")

    def _label(p):
        if product_lookup:
            name = product_lookup.get(p, p)
            if not isinstance(name, str):
                name = str(name)
        else:
            name = str(p)
        return name[:20] + "..." if len(name) > 20 else name

    labels = [_label(p) for p in products]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=20,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color="lightblue",
                    hovertemplate="%{label}<extra></extra>",
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color="rgba(100, 100, 100, 0.3)",
                    hovertemplate="%{source.label} → %{target.label}: %{value:.2f}<extra></extra>",
                ),
            )
        ]
    )

    fig.update_layout(
        title_text=title,
        font=dict(size=13, family="Arial, sans-serif"),
        height=600,
        margin=dict(l=150, r=50, t=50, b=50),
    )

    return fig


def _empty_figure(message: str) -> go.Figure:
    """Create empty figure with message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        plot_bgcolor="white",
    )
    return fig
