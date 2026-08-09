"""Latent product embeddings via TruncatedSVD on the sparse customer-product
matrix, followed by top-k cosine nearest-neighbor search.

This replaces the dense N x N similarity matrices as the scalable default:
the full catalog is embedded once (float32) and neighbors are found through a
ball-tree/kd-tree, so memory stays O(n_products * n_components) instead of
O(n_products^2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def build_product_embeddings(
    customer_product: sparse.csr_matrix,
    n_components: int = 64,
    random_seed: int = 42,
) -> np.ndarray:
    """Latent product vectors (float32, unit L2 norm) via TruncatedSVD.

    Args:
        customer_product: Sparse binary (or count) customer x product matrix.
        n_components: Embedding dimensionality.
        random_seed: SVD reproducibility seed.

    Returns:
        Array of shape (n_products, n_components), L2-normalized rows.
    """
    X = customer_product.T.astype("float64")
    if n_components >= min(X.shape):
        n_components = max(1, min(X.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=random_seed)
    embeddings = svd.fit_transform(X)
    embeddings = normalize(embeddings, norm="l2")
    return embeddings.astype("float32")


def build_topk_neighbors(
    embeddings: np.ndarray,
    top_k: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine nearest-neighbor indices and distances for every product.

    Args:
        embeddings: L2-normalized product vectors (n_products x d).
        top_k: Neighbors per product (including self).

    Returns:
        (indices, distances) arrays of shape (n_products, top_k) sorted by
        distance ascending; index 0 is always the product itself.
    """
    n_products = embeddings.shape[0]
    top_k = min(top_k, n_products)
    nn = NearestNeighbors(n_neighbors=top_k, metric="cosine", n_jobs=-1, algorithm="auto")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    return indices.astype(np.int64), distances.astype("float32")


def topk_neighbor_frame(
    product_ids: np.ndarray,
    indices: np.ndarray,
    distances: np.ndarray,
    max_neighbors: int = 20,
) -> pd.DataFrame:
    """Long-format (product, neighbor, distance, rank) table of top-k neighbors.

    Excludes self-edges and ranks by increasing cosine distance.
    """
    rows: list[dict[str, float | int | str]] = []
    for i, pid in enumerate(product_ids):
        taken = 0
        for rank in range(indices.shape[1]):
            j = indices[i, rank]
            if int(j) == int(i):
                continue
            rows.append(
                {
                    "product": str(pid),
                    "neighbor": str(product_ids[int(j)]),
                    "distance": float(distances[i, rank]),
                    "rank": int(taken + 1),
                }
            )
            taken += 1
            if taken >= max_neighbors:
                break
    return pd.DataFrame(rows)
