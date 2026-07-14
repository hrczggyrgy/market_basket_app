"""CDT: Hierarchical Clustering & Dendrogram.

Performs agglomerative clustering on the similarity matrix using
scipy's hierarchical clustering with precomputed distances.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=UserWarning)


def similarity_to_distance(
    similarity_matrix: pd.DataFrame, method: str = "yules_q"
) -> np.ndarray:
    """
    Convert similarity matrix to distance matrix for clustering.

    For Yule's Q (range [-1, 1]): distance = 1 - similarity
    For Jaccard (range [0, 1]): distance = 1 - similarity

    Both map to [0, 2] for Yule's Q, [0, 1] for Jaccard.
    """
    sim_vals = similarity_matrix.values.copy()
    if method == "yules_q":
        # Yule's Q: 1 = identical, -1 = perfectly dissimilar
        # Map to distance: 0 = identical, 2 = perfectly dissimilar
        dist_vals = 1 - sim_vals
    else:
        # Jaccard: 1 = identical, 0 = no overlap
        dist_vals = 1 - sim_vals

    # Ensure diagonal is 0
    np.fill_diagonal(dist_vals, 0.0)

    # Ensure symmetry
    dist_vals = (dist_vals + dist_vals.T) / 2

    return dist_vals


def perform_hierarchical_clustering(
    similarity_matrix: pd.DataFrame,
    linkage_method: str = "average",
    distance_method: str = "yules_q",
) -> Tuple[np.ndarray, List[str]]:
    """
    Perform agglomerative hierarchical clustering.

    Args:
        similarity_matrix: Square similarity matrix (products x products)
        linkage_method: 'single', 'complete', 'average', 'ward'
        distance_method: 'yules_q' or 'jaccard' for distance conversion

    Returns:
        (linkage_matrix, ordered_product_labels)
    """
    dist_matrix = similarity_to_distance(similarity_matrix, distance_method)

    # Convert to condensed distance vector for linkage
    condensed_dist = squareform(dist_matrix, checks=False)

    # Perform linkage
    # Note: 'ward' requires Euclidean distances, so use 'average' for precomputed
    if linkage_method == "ward":
        linkage_method = "average"
        import logging

        logging.warning(
            "Ward linkage requires Euclidean distances; using average instead."
        )

    linkage_matrix = linkage(condensed_dist, method=linkage_method)

    # Get dendrogram leaf order for consistent labeling
    dendro = dendrogram(linkage_matrix, no_plot=True)
    ordered_labels = [similarity_matrix.index[i] for i in dendro["leaves"]]

    return linkage_matrix, ordered_labels


def find_optimal_clusters(
    linkage_matrix: np.ndarray,
    similarity_matrix: pd.DataFrame,
    distance_method: str = "yules_q",
    min_clusters: int = 2,
    max_clusters: int = 20,
    metric: str = "silhouette",
) -> Tuple[int, Dict[int, float]]:
    """
    Find optimal number of clusters using silhouette score.

    Args:
        linkage_matrix: Output from perform_hierarchical_clustering
        similarity_matrix: Original similarity matrix
        distance_method: For distance conversion
        min_clusters: Minimum clusters to test
        max_clusters: Maximum clusters to test
        metric: 'silhouette' (only supported for precomputed distances)

    Returns:
        (optimal_k, dict of k -> score)
    """
    dist_matrix = similarity_to_distance(similarity_matrix, distance_method)

    n_samples = len(similarity_matrix)
    max_clusters = min(max_clusters, n_samples - 1)

    scores = {}
    best_k = min_clusters
    best_score = -1

    for k in range(min_clusters, max_clusters + 1):
        try:
            labels = fcluster(linkage_matrix, k, criterion="maxclust")

            if metric == "silhouette":
                # Silhouette requires precomputed distance matrix
                score = silhouette_score(dist_matrix, labels, metric="precomputed")
            else:
                score = -np.inf

            scores[k] = score

            if score > best_score:
                best_score = score
                best_k = k

        except Exception:
            scores[k] = np.nan
            continue

    return best_k, scores


def get_cluster_assignments(
    linkage_matrix: np.ndarray,
    similarity_matrix: pd.DataFrame,
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
) -> Dict[str, int]:
    """
    Get flat cluster assignments from hierarchical clustering.

    Args:
        linkage_matrix: Output from perform_hierarchical_clustering
        similarity_matrix: Original similarity matrix (for labels)
        n_clusters: Desired number of clusters (uses 'maxclust' criterion)
        distance_threshold: Cut dendrogram at this distance (uses 'distance' criterion)

    Returns:
        Dict mapping product_id -> cluster_id (0-indexed)
    """
    if n_clusters is not None:
        labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
    elif distance_threshold is not None:
        labels = fcluster(linkage_matrix, distance_threshold, criterion="distance")
    else:
        raise ValueError("Must specify either n_clusters or distance_threshold")

    products = similarity_matrix.index.tolist()
    return dict(zip(products, labels - 1))  # Make 0-indexed


def compute_cluster_quality(
    similarity_matrix: pd.DataFrame,
    cluster_assignments: Dict[str, int],
) -> Dict[str, float]:
    """
    Compute clustering quality metrics.

    Args:
        similarity_matrix: Original similarity matrix
        cluster_assignments: Dict product -> cluster_id

    Returns:
        Dict with metrics
    """
    products = list(cluster_assignments.keys())
    clusters = set(cluster_assignments.values())

    within_sims = []
    between_sims = []

    for i, prod_a in enumerate(products):
        cluster_a = cluster_assignments[prod_a]
        for prod_b in products[i + 1 :]:
            cluster_b = cluster_assignments[prod_b]
            sim = similarity_matrix.loc[prod_a, prod_b]
            if cluster_a == cluster_b:
                within_sims.append(sim)
            else:
                between_sims.append(sim)

    avg_within = np.mean(within_sims) if within_sims else 0
    avg_between = np.mean(between_sims) if between_sims else 0

    # Separation ratio: higher = better separated clusters
    separation_ratio = avg_within / avg_between if avg_between != 0 else np.inf

    return {
        "avg_within_similarity": avg_within,
        "avg_between_similarity": avg_between,
        "separation_ratio": separation_ratio,
        "n_clusters": len(clusters),
        "cluster_sizes": {
            c: sum(1 for v in cluster_assignments.values() if v == c) for c in clusters
        },
    }


def get_dendrogram_data(linkage_matrix: np.ndarray, labels: List[str]) -> Dict:
    """
    Get dendrogram data for visualization.

    Returns dict with icoord, dcoord, ivl, leaves, color_list
    """
    dendro = dendrogram(linkage_matrix, labels=labels, no_plot=True)
    return dendro


def cut_dendrogram_at_k(linkage_matrix: np.ndarray, k: int) -> np.ndarray:
    """Cut dendrogram to get k clusters."""
    return fcluster(linkage_matrix, k, criterion="maxclust") - 1


def compute_cophenetic_correlation(
    linkage_matrix: np.ndarray,
    similarity_matrix: pd.DataFrame,
    distance_method: str = "yules_q",
) -> float:
    """
    Compute cophenetic correlation coefficient.

    Measures how faithfully the dendrogram preserves pairwise distances.
    """
    dist_matrix = similarity_to_distance(similarity_matrix, distance_method)
    condensed_dist = squareform(dist_matrix, checks=False)
    coph_corr, _ = cophenet(linkage_matrix, condensed_dist)
    return coph_corr


def compute_weighted_within_similarity(
    similarity_matrix: pd.DataFrame,
    cluster_assignments: Dict[str, int],
) -> float:
    """
    Compute weighted average within-cluster similarity.

    This matches the tree quality metric used in CDT scoring.
    """
    products = list(cluster_assignments.keys())
    clusters = set(cluster_assignments.values())

    total_weight = 0
    weighted_sim = 0.0

    for cluster_id in clusters:
        cluster_products = [p for p in products if cluster_assignments[p] == cluster_id]
        if len(cluster_products) < 2:
            continue

        # Compute average within-cluster similarity
        sims = []
        for i, a in enumerate(cluster_products):
            for b in cluster_products[i + 1 :]:
                if a in similarity_matrix.index and b in similarity_matrix.columns:
                    sims.append(similarity_matrix.loc[a, b])

        if sims:
            avg_sim = np.mean(sims)
            weight = len(cluster_products)
            weighted_sim += avg_sim * weight
            total_weight += weight

    return weighted_sim / total_weight if total_weight > 0 else 0.0


def compute_unconstrained_baseline(
    similarity_matrix: pd.DataFrame,
    linkage_method: str = "average",
    distance_method: str = "yules_q",
) -> float:
    """
    Compute quality of unconstrained (optimal) clustering as baseline.

    Finds the best possible clustering quality without attribute constraints.
    Uses weighted average within-cluster similarity (matches CDT tree quality).

    Returns:
        Best weighted within-cluster similarity achievable
    """
    linkage_matrix, _ = perform_hierarchical_clustering(
        similarity_matrix, linkage_method, distance_method
    )

    best_quality = 0.0
    max_clusters = min(20, len(similarity_matrix) - 1)

    for k in range(2, max_clusters + 1):
        assignments = get_cluster_assignments(
            linkage_matrix, similarity_matrix, n_clusters=k
        )
        quality = compute_weighted_within_similarity(similarity_matrix, assignments)
        if quality > best_quality:
            best_quality = quality

    return best_quality
