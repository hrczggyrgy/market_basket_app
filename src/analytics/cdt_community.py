"""CDT: Graph Construction & Community Detection.

Builds product similarity graph and detects communities using
Louvain, Leiden, or Label Propagation algorithms.

Dependency notes:
- python-louvain is optional; if missing, Louvain falls back to label_propagation.
- igraph + leidenalg are optional; if missing, Leiden falls back to label_propagation.
"""

import warnings

import warnings
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


def build_product_graph(
    similarity_matrix: pd.DataFrame,
    min_weight: float = 0.1,
    min_joint_customers: int = 5,
    max_edges_per_node: int = 50,
    joint_customers_matrix: Optional[pd.DataFrame] = None,
) -> nx.Graph:
    """
    Create NetworkX graph from similarity matrix.

    Nodes = products, edges weighted by similarity.
    Pruned by minimum weight and degree.

    Args:
        similarity_matrix: Square product x product similarity matrix
        min_weight: Minimum edge weight to keep
        min_joint_customers: Minimum co-purchasing customers (requires joint_customers_matrix)
        max_edges_per_node: Maximum edges per node (keeps top-k by weight)
        joint_customers_matrix: Optional co-occurrence counts for additional filtering

    Returns:
        NetworkX Graph with product nodes and weighted edges
    """
    G = nx.Graph()
    products = similarity_matrix.index.tolist()

    # Add nodes
    G.add_nodes_from(products)

    # Add edges above threshold
    sim_vals = similarity_matrix.values
    n = len(products)

    for i in range(n):
        for j in range(i + 1, n):
            weight = sim_vals[i, j]
            if weight < min_weight:
                continue

            # Optional joint customer filter
            if joint_customers_matrix is not None:
                joint = joint_customers_matrix.iloc[i, j]
                if joint < min_joint_customers:
                    continue

            G.add_edge(products[i], products[j], weight=float(weight))

    # Degree pruning: keep top-k edges per node
    if max_edges_per_node and max_edges_per_node < G.number_of_nodes():
        edges_to_remove = []
        for node in G.nodes():
            edges = list(G.edges(node, data=True))
            if len(edges) > max_edges_per_node:
                edges.sort(key=lambda x: x[2]["weight"], reverse=True)
                for u, v, d in edges[max_edges_per_node:]:
                    edges_to_remove.append((u, v))
        G.remove_edges_from(edges_to_remove)

    # Remove isolated nodes
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(isolated)

    return G


def detect_communities_louvain(
    graph: nx.Graph,
    resolution: float = 1.0,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Detect communities using Louvain algorithm (python-louvain).

    Args:
        graph: NetworkX graph with weighted edges
        resolution: Resolution parameter (higher = more, smaller communities)
        seed: Random seed

    Returns:
        Dict mapping node -> community_id

    Raises:
        ImportError: if python-louvain is not installed
    """
    import community as community_louvain  # noqa: F811

    # Louvain expects dict of node->community
    partition = community_louvain.best_partition(
        graph, weight="weight", resolution=resolution, random_state=seed
    )
    return partition


def detect_communities_leiden(
    graph: nx.Graph,
    resolution: float = 1.0,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Detect communities using Leiden algorithm (igraph + leidenalg).

    More robust than Louvain; guarantees connected communities.

    Args:
        graph: NetworkX graph
        resolution: Resolution parameter
        seed: Random seed

    Returns:
        Dict mapping node -> community_id

    Raises:
        ImportError: if igraph or leidenalg is not installed
    """
    import igraph as ig
    import leidenalg

    # Convert NetworkX to igraph
    edges = list(graph.edges())
    weights = [graph[u][v].get("weight", 1.0) for u, v in edges]

    g = ig.Graph()
    g.add_vertices(list(graph.nodes()))
    g.add_edges(edges)
    g.es["weight"] = weights

    # Run Leiden
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )

    # Map back to original node names
    node_names = list(graph.nodes())
    return {node_names[i]: int(partition.membership[i]) for i in range(len(node_names))}


def detect_communities_label_propagation(
    graph: nx.Graph,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Detect communities using Label Propagation (NetworkX built-in).

    Fast, no external dependencies, but less stable than Louvain/Leiden.

    Args:
        graph: NetworkX graph
        seed: Random seed

    Returns:
        Dict mapping node -> community_id
    """
    communities = nx.algorithms.community.label_propagation.label_propagation_communities(graph)
    partition = {}
    for comm_id, comm in enumerate(communities):
        for node in comm:
            partition[node] = comm_id
    return partition


def detect_communities(
    graph: nx.Graph,
    method: str = "louvain",
    resolution: float = 1.0,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Unified community detection interface.

    Args:
        graph: NetworkX graph
        method: 'louvain', 'leiden', 'label_propagation'
        resolution: Resolution parameter (for Louvain/Leiden)
        seed: Random seed

    Returns:
        Dict mapping node -> community_id
    """
    if method == "louvain":
        try:
            return detect_communities_louvain(graph, resolution, seed)
        except ImportError:
            warnings.warn(
                "python-louvain not installed; falling back to label_propagation"
            )
            return detect_communities_label_propagation(graph, seed)
    elif method == "leiden":
        try:
            return detect_communities_leiden(graph, resolution, seed)
        except ImportError:
            warnings.warn(
                "igraph/leidenalg not installed; falling back to label_propagation"
            )
            return detect_communities_label_propagation(graph, seed)
    elif method == "label_propagation":
        return detect_communities_label_propagation(graph, seed)
    else:
        raise ValueError(f"Unknown community method: {method}")


def compute_community_modularity(
    graph: nx.Graph,
    partition: Dict[str, int],
) -> float:
    """Compute modularity of a partition (requires python-louvain)."""
    try:
        import community as community_louvain
    except ImportError:
        raise ImportError("python-louvain required to compute modularity")
    return community_louvain.modularity(partition, graph, weight="weight")


def hierarchical_clustering_within_communities(
    similarity_matrix: pd.DataFrame,
    community_assignments: Dict[str, int],
    linkage_method: str = "average",
    distance_method: str = "phi",
) -> Dict[int, Tuple[np.ndarray, List[str]]]:
    """
    Run agglomerative clustering separately within each community.

    This produces more stable dendrograms by constraining merges
    within semantically coherent communities.

    Args:
        similarity_matrix: Full product similarity matrix
        community_assignments: node -> community_id mapping
        linkage_method: 'single', 'complete', 'average', 'ward'
        distance_method: 'phi' or 'jaccard' for distance conversion

    Returns:
        Dict: community_id -> (linkage_matrix, ordered_labels)
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    def similarity_to_distance(sim_matrix: pd.DataFrame, method: str) -> np.ndarray:
        sim_vals = sim_matrix.values.copy()
        if method == "phi":
            dist_vals = 1 - sim_vals
        else:
            dist_vals = 1 - sim_vals
        np.fill_diagonal(dist_vals, 0.0)
        dist_vals = (dist_vals + dist_vals.T) / 2
        return dist_vals

    communities = {}
    for node, comm_id in community_assignments.items():
        communities.setdefault(comm_id, []).append(node)

    results = {}
    for comm_id, nodes in communities.items():
        if len(nodes) < 2:
            # Single node - no clustering needed
            results[comm_id] = (np.array([]), nodes)
            continue

        sub_sim = similarity_matrix.loc[nodes, nodes]
        dist_matrix = similarity_to_distance(sub_sim, distance_method)

        # Condensed distance for linkage
        condensed = squareform(dist_matrix, checks=False)

        # Ward requires Euclidean; fallback to average
        if linkage_method == "ward":
            link_method = "average"
        else:
            link_method = linkage_method

        link_matrix = linkage(condensed, method=link_method)

        # Get leaf order
        dendro = dendrogram(link_matrix, labels=nodes, no_plot=True)
        ordered = [nodes[i] for i in dendro["leaves"]]

        results[comm_id] = (link_matrix, ordered)

    return results


def merge_community_dendrograms(
    community_dendrograms: Dict[int, Tuple[np.ndarray, List[str]]],
    community_assignments: Dict[str, int],
) -> Tuple[np.ndarray, List[str]]:
    """
    Merge community-level dendrograms into a global ordering for display.

    The resulting linkage matrix is suitable for dendrogram visualization
    but may not be valid for fcluster (when communities are disconnected).
    Use community_assignments directly for cluster membership when available.

    Args:
        community_dendrograms: comm_id -> (linkage_matrix, ordered_labels)
        community_assignments: node -> comm_id

    Returns:
        (global_linkage_matrix, global_ordered_labels)
    """
    from scipy.cluster.hierarchy import linkage as scipy_linkage  # noqa: F811
    from scipy.spatial.distance import squareform  # noqa: F811

    comm_ids = sorted(community_dendrograms.keys())

    # Collect all labels in order
    all_labels = []
    for comm_id in comm_ids:
        _link_mat, labels = community_dendrograms[comm_id]
        all_labels.extend(labels)

    if len(all_labels) < 2:
        return np.array([]), all_labels

    # Shift and collect each community's internal linkage
    shifted_linkages = []
    offset = 0
    max_internal_dist = 0.0

    for comm_id in comm_ids:
        link_mat, labels = community_dendrograms[comm_id]
        n_items = len(labels)

        if n_items >= 2 and len(link_mat) > 0:
            shifted = link_mat.copy()
            shifted[:, :2] += offset
            shifted_linkages.append(shifted)
            max_internal_dist = max(max_internal_dist, link_mat[:, 2].max())

        offset += n_items

    if not shifted_linkages:
        return np.array([]), all_labels

    if len(shifted_linkages) == 1:
        return shifted_linkages[0], all_labels

    # Multiple communities: we cannot produce a single valid fcluster linkage
    # from disconnected sub-dendrograms. Instead, return the concatenated
    # linkage for display only. fcluster will fall back to community assignments.
    global_linkage = np.vstack(shifted_linkages)
    return global_linkage, all_labels


def community_detection_pipeline(
    similarity_matrix: pd.DataFrame,
    joint_customers_matrix: Optional[pd.DataFrame] = None,
    method: str = "louvain",
    resolution: float = 1.0,
    graph_min_weight: float = 0.1,
    graph_min_joint: int = 5,
    graph_max_degree: int = 50,
    seed: int = 42,
) -> Tuple[nx.Graph, Dict[str, int], Dict[int, Tuple[np.ndarray, List[str]]]]:
    """
    End-to-end community detection pipeline.

    Returns:
        (graph, node_to_community, community_dendrograms)
    """
    # 1. Build graph
    graph = build_product_graph(
        similarity_matrix,
        min_weight=graph_min_weight,
        min_joint_customers=graph_min_joint,
        max_edges_per_node=graph_max_degree,
        joint_customers_matrix=joint_customers_matrix,
    )

    # 2. Detect communities
    partition = detect_communities(graph, method, resolution, seed)

    # 3. Hierarchical clustering within communities
    comm_dendrograms = hierarchical_clustering_within_communities(similarity_matrix, partition)

    return graph, partition, comm_dendrograms
