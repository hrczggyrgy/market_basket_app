"""Agglomerative clustering over product similarity for CDT leaves."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from sklearn.metrics import silhouette_score

from src.analytics.schemas import CDT_ASSIGNMENTS, CDT_OPTIMAL_K, CDT_QUALITY, check


def similarity_to_distance(
    similarity_matrix: pd.DataFrame, method: str = "phi"
) -> np.ndarray:
    """Convert a similarity matrix to a distance matrix.

    ``phi`` distances are ``1 - sim`` (sim may be negative); ``jaccard`` /
    ``pmi`` / ``cosine`` use ``1 - sim`` clipped to ``[0, 1]``. The diagonal
    is zeroed and the matrix is made symmetric.
    """
    sim = similarity_matrix.to_numpy(dtype=float)
    if method == "phi":
        distance = 1.0 - sim
    else:
        distance = np.clip(1.0 - sim, 0.0, 1.0)
    distance = np.maximum(distance, distance.T)
    np.fill_diagonal(distance, 0.0)
    distance = np.nan_to_num(distance, nan=0.0, posinf=1.0)
    return distance


def perform_hierarchical_clustering(
    similarity_matrix: pd.DataFrame,
    method: str = "ward",
    n_clusters: int | None = None,
) -> tuple[np.ndarray, pd.Series]:
    """Agglomerative clustering, returning linkage and per-product cluster.

    ``method`` is one of scipy's linkage methods. When ``n_clusters`` is
    given, flat clusters are cut at that height; otherwise all products share
    one cluster.
    """
    distance = similarity_to_distance(similarity_matrix, method=_infer_sim_method(method))
    condensed = _square_to_condensed(distance)
    if len(condensed) < 1:
        products = similarity_matrix.index.tolist()
        return np.empty((0, 4)), pd.Series(1, index=products, name="cluster")
    linkage_matrix = linkage(condensed, method=_safe_linkage_method(method))

    products = similarity_matrix.index.tolist()
    if n_clusters is None or n_clusters < 1:
        clusters = np.ones(len(products), dtype=int)
    else:
        n_clusters = min(n_clusters, len(products))
        clusters = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    return linkage_matrix, pd.Series(clusters, index=products, name="cluster")


def _safe_linkage_method(method: str) -> str:
    return "ward" if method in {"phi", "jaccard", "pmi", "cosine_tfidf", "ensemble"} else method


def _infer_sim_method(method: str) -> str:
    return "phi" if method in {"phi", "jaccard", "pmi", "cosine_tfidf", "ensemble"} else "jaccard"


def _square_to_condensed(distance: np.ndarray) -> np.ndarray:
    n = distance.shape[0]
    if n < 2:
        return np.empty(0)
    iu = np.triu_indices(n, k=1)
    return distance[iu]


def find_optimal_clusters_sklearn(
    similarity_matrix: pd.DataFrame,
    max_clusters: int = 10,
) -> pd.DataFrame:
    """Best k by average silhouette score over a k range."""
    distance = similarity_to_distance(similarity_matrix, method="jaccard")
    condensed = _square_to_condensed(distance)
    if len(condensed) < 1 or len(similarity_matrix) < 3:
        return check(pd.DataFrame(columns=list(CDT_OPTIMAL_K.columns)), CDT_OPTIMAL_K, allow_empty=True)

    linkage_matrix = linkage(condensed, method="ward")
    n = len(similarity_matrix)
    upper = min(max_clusters, n - 1)
    rows: list[dict[str, float | int]] = []
    for k in range(2, upper + 1):
        labels = fcluster(linkage_matrix, t=k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(distance, labels, metric="precomputed"))
        rows.append({"n_clusters": k, "silhouette": score})

    table = pd.DataFrame(rows, columns=list(CDT_OPTIMAL_K.columns))
    return check(table, CDT_OPTIMAL_K, allow_empty=True)


def get_cluster_assignments(
    transactions_df: pd.DataFrame,
    similarity_matrix: pd.DataFrame | None = None,
    *,
    method: str = "phi",
    n_clusters: int = 4,
) -> pd.DataFrame:
    """Per-product cluster assignments from similarity clustering."""
    if similarity_matrix is None:
        from src.analytics.cdt.similarity import build_similarity_matrix

        similarity_matrix = build_similarity_matrix(transactions_df, method=method)
    linkage_matrix, clusters = perform_hierarchical_clustering(
        similarity_matrix, method="ward", n_clusters=n_clusters
    )
    _ = linkage_matrix
    table = clusters.reset_index().rename(
        columns={"index": "stockcode", "cluster": "cluster"}
    )
    return check(table, CDT_ASSIGNMENTS)


def compute_cluster_quality(
    similarity_matrix: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Per-cluster within/across mean similarity."""
    rows: list[dict[str, float | int]] = []
    assignments = cluster_assignments.set_index("stockcode")["cluster"]
    for cluster in sorted(assignments.unique()):
        products = assignments[assignments == cluster].index.tolist()
        valid = [p for p in products if p in similarity_matrix.index]
        if len(valid) < 2:
            within = 1.0 if len(valid) == 1 else 0.0
        else:
            sub = similarity_matrix.loc[valid, valid].to_numpy(dtype=float)
            triu = sub[np.triu_indices(len(valid), k=1)]
            within = float(np.nanmean(triu)) if len(triu) else 1.0

        other_products = [p for p in similarity_matrix.index if p not in valid]
        if not valid or not other_products:
            across = 0.0
        else:
            cross = similarity_matrix.loc[valid, other_products].to_numpy(dtype=float)
            across = float(np.nanmean(cross)) if cross.size else 0.0
        rows.append(
            {"cluster": int(cluster), "size": len(valid), "within_similarity": within, "across_similarity": across}
        )

    return check(pd.DataFrame(rows, columns=list(CDT_QUALITY.columns)), CDT_QUALITY)


def compute_cophenetic_correlation(
    similarity_matrix: pd.DataFrame, linkage_matrix: np.ndarray
) -> float:
    """Cophenetic correlation between the linkage and original distances."""
    distance = similarity_to_distance(similarity_matrix, method="jaccard")
    condensed = _square_to_condensed(distance)
    if len(condensed) < 2 or len(linkage_matrix) < 1:
        return 0.0
    corr, _ = cophenet(linkage_matrix, condensed)
    if not np.isfinite(corr):
        return 0.0
    return float(corr)


def get_dendrogram_data(
    linkage_matrix: np.ndarray, labels: list[str]
) -> dict[str, object]:
    """Serializable dendrogram payload for plotly: linkage + leaf labels."""
    return {
        "linkage": linkage_matrix.tolist(),
        "labels": labels,
        "n_leaves": len(labels),
    }