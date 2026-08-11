"""Community detection on the product co-purchase graph.

Builds a weighted networkx graph with phi affinity as edge weight, then finds
communities with Louvain or label propagation (both shipped in networkx).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from src.analytics.copurchase import compute_affinity_matrix


def build_product_graph(
    transactions_df: pd.DataFrame,
    min_cooccurrence: int = 5,
    min_weight: float = 0.0,
) -> nx.Graph:
    """Weighted product graph; edge weight = phi affinity (NaN pairs dropped).

    Vectorized: the O(N^2) pair scan is performed with numpy triu indices
    instead of a Python double loop, so a 2,000-product affinity matrix builds
    in a fraction of a second.
    """
    affinity = compute_affinity_matrix(transactions_df, min_cooccurrence=min_cooccurrence)
    graph = nx.Graph()
    products = affinity.index.tolist()
    graph.add_nodes_from(products)

    values = affinity.to_numpy(dtype=float)
    iu = np.triu_indices(values.shape[0], k=1)
    weights = values[iu]
    valid = ~np.isnan(weights) & (weights >= min_weight)
    if not valid.any():
        return graph
    src = np.asarray(iu[0])[valid].tolist()
    dst = np.asarray(iu[1])[valid].tolist()
    edges = [
        (products[s], products[d], {"weight": float(w)})
        for s, d, w in zip(src, dst, weights[valid].tolist(), strict=True)
    ]
    graph.add_edges_from(edges)
    return graph


def detect_communities_louvain(
    graph: nx.Graph, random_seed: int | None = None, resolution: float = 1.0
) -> dict[str, int]:
    """Best-partition Louvain communities."""
    from networkx.algorithms.community import louvain_communities

    communities = louvain_communities(
        graph, weight="weight", seed=random_seed, resolution=resolution
    )
    return _communities_to_dict(communities)


def detect_communities_label_propagation(graph: nx.Graph) -> dict[str, int]:
    """Label propagation communities (fast, deterministic order)."""
    from networkx.algorithms.community import asyn_lpa_communities, label_propagation_communities

    communities = list(asyn_lpa_communities(graph, weight="weight"))
    if not communities:
        communities = list(label_propagation_communities(graph))
    return _communities_to_dict(communities)


def _communities_to_dict(communities: list[set[str]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for cid, community in enumerate(communities):
        for product in community:
            mapping[product] = cid
    return mapping


def detect_communities(
    transactions_df: pd.DataFrame,
    method: str = "louvain",
    min_cooccurrence: int = 5,
    random_seed: int | None = None,
) -> tuple[dict[str, int], float]:
    """Community detection pipeline; returns (product->community, modularity)."""
    graph = build_product_graph(transactions_df, min_cooccurrence=min_cooccurrence)
    if method == "louvain":
        mapping = detect_communities_louvain(graph, random_seed=random_seed)
    elif method in {"label_propagation", "asyn_lpa"}:
        mapping = detect_communities_label_propagation(graph)
    else:
        raise ValueError(f"unknown community method {method!r}")
    modularity = compute_community_modularity(graph, mapping)
    return mapping, modularity


def compute_community_modularity(graph: nx.Graph, communities: dict[str, int]) -> float:
    """Modularity of a partition (product->community) on a weighted graph."""
    from networkx.algorithms.community import modularity

    partition: list[set[str]] = []
    for cid in sorted(set(communities.values())):
        partition.append({node for node, c in communities.items() if c == cid})
    return float(modularity(graph, partition, weight="weight"))
