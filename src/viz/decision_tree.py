"""Decision tree visualization using Plotly."""

from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier, _tree


def plot_decision_tree(
    model: DecisionTreeClassifier,
    feature_names: List[str],
    class_names: List[str] = None,
    max_depth: int = 3,
    title: str = "Decision Tree",
    height: int = 700,
) -> go.Figure:
    """
    Create interactive Plotly visualization of decision tree.

    Args:
        model: Fitted DecisionTreeClassifier
        feature_names: List of feature names
        class_names: Target class names
        max_depth: Maximum depth to display
        title: Chart title
        height: Figure height
    """
    if model is None:
        return _empty_figure("No model provided")

    if class_names is None:
        class_names = ["Not Buy", "Buy"]

    tree = model.tree_
    n_nodes = tree.node_count

    if n_nodes == 0:
        return _empty_figure("Empty tree")

    # Extract tree structure
    nodes = []
    edges = []

    def traverse(node_id, depth, parent_id=None, condition=""):
        if depth > max_depth:
            return

        is_leaf = tree.feature[node_id] == _tree.TREE_UNDEFINED

        n_samples = tree.n_node_samples[node_id]
        value = tree.value[node_id][0]
        total = value.sum()
        probs = value / total if total > 0 else [0, 0]
        predicted_class = np.argmax(value)

        if is_leaf:
            label = (
                f"Leaf {node_id}<br>"
                f"Samples: {n_samples}<br>"
                f"Class: {class_names[predicted_class]}<br>"
                f"Prob: {probs[predicted_class]:.2%}<br>"
                f"Distribution: {dict(zip(class_names, value.astype(int)))}"
            )
            color = "lightgreen" if predicted_class == 1 else "lightcoral"
        else:
            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feature_name = feature_names[feature_idx]

            label = (
                f"Node {node_id}<br>"
                f"{feature_name} ≤ {threshold:.2f}<br>"
                f"Samples: {n_samples}<br>"
                f"Class: {class_names[predicted_class]}<br>"
                f"Prob: {probs[predicted_class]:.2%}"
            )
            color = "lightblue"

        nodes.append(
            {
                "id": node_id,
                "label": label,
                "color": color,
                "depth": depth,
                "is_leaf": is_leaf,
                "samples": n_samples,
                "prob_buy": probs[1] if len(probs) > 1 else 0,
                "predicted_class": predicted_class,
            }
        )

        if parent_id is not None:
            edges.append({"source": parent_id, "target": node_id, "condition": condition})

        if not is_leaf:
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            feature_name = feature_names[tree.feature[node_id]]
            threshold = tree.threshold[node_id]

            traverse(left_child, depth + 1, node_id, f"≤ {threshold:.2f}")
            traverse(right_child, depth + 1, node_id, f"> {threshold:.2f}")

    traverse(0, 0)

    if not nodes:
        return _empty_figure("No nodes to display")

    # Create layout positions using hierarchical layout
    node_positions = _hierarchical_layout(nodes, edges)

    # Create edge traces
    edge_x, edge_y = [], []
    edge_text = []

    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        if src in node_positions and tgt in node_positions:
            x0, y0 = node_positions[src]
            x1, y1 = node_positions[tgt]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(edge["condition"])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=2, color="gray"),
        hoverinfo="text",
        text=edge_text,
        showlegend=False,
    )

    # Create node traces by type
    node_traces = []

    for node_type in ["internal", "leaf_buy", "leaf_not_buy"]:
        type_nodes = [
            n
            for n in nodes
            if (node_type == "internal" and not n["is_leaf"])
            or (node_type == "leaf_buy" and n["is_leaf"] and n["predicted_class"] == 1)
            or (node_type == "leaf_not_buy" and n["is_leaf"] and n["predicted_class"] == 0)
        ]

        if not type_nodes:
            continue

        x = [node_positions[n["id"]][0] for n in type_nodes]
        y = [node_positions[n["id"]][1] for n in type_nodes]
        text = [f"N{n['id']}" for n in type_nodes]
        hover = [n["label"] for n in type_nodes]

        if node_type == "internal":
            color = "lightblue"
            size = 30
        elif node_type == "leaf_buy":
            color = "lightgreen"
            size = 25
        else:
            color = "lightcoral"
            size = 25

        node_traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="markers+text",
                text=text,
                textposition="middle center",
                textfont=dict(size=10, color="black"),
                marker=dict(size=size, color=color, line=dict(width=2, color="darkgray")),
                hoverinfo="text",
                hovertext=hover,
                showlegend=False,
                name=node_type,
            )
        )

    fig = go.Figure(data=[edge_trace] + node_traces)

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange="reversed"),
        height=height,
        plot_bgcolor="white",
    )

    # Add legend
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=20, color="lightblue"),
            name="Decision Node",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=15, color="lightgreen"),
            name="Predict: Buy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=15, color="lightcoral"),
            name="Predict: Not Buy",
        )
    )
    fig.update_layout(showlegend=True, legend=dict(x=0.01, y=0.99))

    return fig


def _hierarchical_layout(nodes: List[Dict], edges: List[Dict]) -> Dict[int, tuple]:
    """Compute hierarchical positions for tree nodes."""
    # Group nodes by depth
    depth_groups = {}
    for node in nodes:
        d = node["depth"]
        if d not in depth_groups:
            depth_groups[d] = []
        depth_groups[d].append(node["id"])

    positions = {}

    # Assign x positions based on in-order traversal
    # For each depth, spread nodes horizontally
    max_depth = max(depth_groups.keys()) if depth_groups else 0

    for depth in range(max_depth + 1):
        nodes_at_depth = depth_groups.get(depth, [])
        n = len(nodes_at_depth)

        if n == 1:
            x_positions = [0]
        else:
            x_positions = np.linspace(-1, 1, n)

        y = -depth  # Negative so root is at top

        for i, node_id in enumerate(nodes_at_depth):
            positions[node_id] = (x_positions[i], y)

    return positions


def plot_tree_rules(
    rules: List[Dict],
    feature_names: List[str],
    class_names: List[str] = None,
    title: str = "Decision Rules",
    height: int = 500,
) -> go.Figure:
    """
    Visualize extracted tree rules as horizontal bar chart.
    """
    if not rules:
        return _empty_figure("No rules extracted")

    if class_names is None:
        class_names = ["Not Buy", "Buy"]

    # Filter to leaf rules only
    leaf_rules = [r for r in rules if r.get("is_leaf", False)]

    if not leaf_rules:
        leaf_rules = rules

    # Sort by probability of positive class
    leaf_rules = sorted(leaf_rules, key=lambda x: x.get("probability", 0), reverse=True)

    # Take top 15
    leaf_rules = leaf_rules[:15]

    # Create rule strings
    rule_strings = []
    probs = []
    samples = []
    predictions = []
    colors = []

    for r in leaf_rules:
        conditions = r.get("conditions", [])
        rule_str = " ∧ ".join(conditions) if conditions else "Root"
        rule_strings.append(rule_str)
        probs.append(r.get("probability", 0))
        samples.append(r.get("samples", 0))
        predictions.append(r.get("prediction", "Unknown"))
        colors.append("green" if r.get("prediction") == "Buy" else "red")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=rule_strings[::-1],  # Reverse for top-down
            x=probs[::-1],
            orientation="h",
            marker=dict(color=colors[::-1], opacity=0.7),
            text=[f"P(Buy)={p:.1%}<br>N={s}" for p, s in zip(probs[::-1], samples[::-1])],
            textposition="auto",
            hoverinfo="text",
            hovertext=[
                f"Rule: {rs}<br>Samples: {s}<br>Prediction: {pred}<br>P(Buy): {p:.2%}"
                for rs, s, pred, p in zip(
                    rule_strings[::-1], samples[::-1], predictions[::-1], probs[::-1]
                )
            ],
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Probability of Purchase",
        yaxis_title="Decision Path",
        height=height,
        margin=dict(l=200),
        xaxis=dict(range=[0, 1]),
        plot_bgcolor="white",
    )

    return fig


def plot_feature_importance(
    model: DecisionTreeClassifier,
    feature_names: List[str],
    top_n: int = 15,
    title: str = "Feature Importance",
    height: int = 400,
) -> go.Figure:
    """Plot feature importance from decision tree."""
    if model is None:
        return _empty_figure("No model provided")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig = go.Figure(
        go.Bar(
            x=[importances[i] for i in indices],
            y=[feature_names[i] for i in indices],
            orientation="h",
            marker=dict(color="steelblue"),
            text=[f"{importances[i]:.4f}" for i in indices],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=height,
        margin=dict(l=200),
        plot_bgcolor="white",
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
