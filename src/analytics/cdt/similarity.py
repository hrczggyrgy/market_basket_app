"""Product similarity matrices for CDT clustering.

Similarity is computed on the binary customer-product matrix so that all
methods share the same support definition: two products are comparable when
they share customers. Phi is re-used from the affinity module; Jaccard, PMI
and TF-IDF cosine are computed here; an ensemble averages standardized
matrices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.cdt.embedding import build_product_embeddings
from src.analytics.copurchase import compute_affinity_matrix


def _customer_product_matrix(
    transactions_df: pd.DataFrame,
    customer_col: str = "customer_id",
    product_col: str = "stockcode",
    min_product_support: int = 2,
    top_n_products: int | None = None,
) -> pd.DataFrame:
    """Binary customer x product matrix, dropping rare products.

    Args:
        top_n_products: If provided, only keep top-N products by support.
    """
    cust_product = pd.crosstab(transactions_df[customer_col], transactions_df[product_col])
    support = cust_product.sum(axis=0)
    valid = support[support >= min_product_support].index
    cust_product = cust_product[valid]

    if top_n_products is not None and len(cust_product.columns) > top_n_products:
        # Keep only top-N products by support
        top_products = support.sort_values(ascending=False).head(top_n_products).index
        cust_product = cust_product[top_products]
        import warnings

        warnings.warn(
            f"CDT similarity: Product count ({len(valid)}) exceeds top_n_products ({top_n_products}). "
            f"Using top {top_n_products} products by support.",
            UserWarning,
            stacklevel=2,
        )

    return (cust_product > 0).astype(int)


def _pairwise_counts(matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """(co-occurrence, product counts, n_customers) for a binary matrix."""
    cooccur = (matrix.to_numpy().T @ matrix.to_numpy().astype(np.int64)).astype(float)
    counts = matrix.sum(axis=0).to_numpy().astype(float)
    return cooccur, counts, matrix.shape[0]


def compute_phi_matrix(
    transactions_df: pd.DataFrame,
    customer_col: str = "customer_id",
    product_col: str = "stockcode",
    min_cooccurrence: int = 5,
    min_product_support: int = 2,
    top_n_products: int | None = None,
) -> pd.DataFrame:
    """Phi-coefficient matrix on the customer-product matrix (reuses affinity)."""
    if top_n_products is not None:
        import warnings

        warnings.warn(
            "Phi coefficient: top_n_products not directly supported for phi matrix. "
            "Consider using embedding method for large catalogs.",
            UserWarning,
            stacklevel=2,
        )
    matrix = compute_affinity_matrix(transactions_df, min_cooccurrence=min_cooccurrence)
    support = pd.crosstab(transactions_df[customer_col], transactions_df[product_col]).sum(axis=0)
    products = support[support >= min_product_support].index
    keep = [p for p in matrix.index if p in products]
    return matrix.loc[keep, keep].replace(np.nan, 0.0)


def compute_jaccard_matrix(
    transactions_df: pd.DataFrame,
    customer_col: str = "customer_id",
    product_col: str = "stockcode",
    min_cooccurrence: int = 5,
    min_product_support: int = 2,
    top_n_products: int | None = None,
) -> pd.DataFrame:
    """Jaccard similarity: |A∩B| / |A∪B| over shared customers."""
    matrix = _customer_product_matrix(
        transactions_df, customer_col, product_col, min_product_support, top_n_products
    )
    cooccur, counts, _ = _pairwise_counts(matrix)
    union = counts[:, None] + counts[None, :] - cooccur
    sim = np.divide(cooccur, union, out=np.zeros_like(cooccur), where=union > 0)
    sim[cooccur < min_cooccurrence] = 0.0
    np.fill_diagonal(sim, 1.0)
    products = matrix.columns.tolist()
    return pd.DataFrame(sim, index=products, columns=products)


def compute_pmi_matrix(
    transactions_df: pd.DataFrame,
    customer_col: str = "customer_id",
    product_col: str = "stockcode",
    min_cooccurrence: int = 5,
    min_product_support: int = 2,
    top_n_products: int | None = None,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Pointwise mutual information normalized to [0, 1].

    PMI = log(P(a,b) / (P(a) P(b))), rescaled by -log P(a,b) so that the
    result lies in [0, 1] and is symmetric.
    """
    matrix = _customer_product_matrix(
        transactions_df, customer_col, product_col, min_product_support, top_n_products
    )
    cooccur, counts, n = _pairwise_counts(matrix)
    pa = counts / n
    pab = cooccur / n
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = np.log(pab / (pa[:, None] * pa[None, :]) + eps)
        denominator = -np.log(pab + eps)
        pmi = numerator / denominator
    pmi = np.clip(pmi, 0.0, 1.0)
    pmi[cooccur < min_cooccurrence] = 0.0
    np.fill_diagonal(pmi, 1.0)
    products = matrix.columns.tolist()
    return pd.DataFrame(pmi, index=products, columns=products)


def compute_cosine_tfidf_matrix(
    transactions_df: pd.DataFrame,
    customer_col: str = "customer_id",
    product_col: str = "stockcode",
    min_product_support: int = 2,
    top_n_products: int | None = None,
) -> pd.DataFrame:
    """TF-IDF weighted cosine similarity between products over baskets.

    Each customer is a document; products with few customers receive higher
    IDF weight, so niche but consistent products are not drowned out by
    best-sellers.
    """
    matrix = _customer_product_matrix(
        transactions_df, customer_col, product_col, min_product_support, top_n_products
    )
    tf = matrix.to_numpy().astype(float)
    n_customers = tf.shape[0]
    df_count = (tf > 0).sum(axis=0)
    idf = np.log((n_customers + 1) / (df_count + 1)) + 1.0
    weighted = tf * idf[None, :]
    norms = np.linalg.norm(weighted, axis=0)
    sim = weighted.T @ weighted
    sim = np.divide(sim, np.outer(norms, norms), out=np.zeros_like(sim), where=norms > 0)
    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    products = matrix.columns.tolist()
    return pd.DataFrame(sim, index=products, columns=products)


def compute_embedding_matrix(
    transactions_df: pd.DataFrame,
    customer_col: str = "customer_id",
    product_col: str = "stockcode",
    min_product_support: int = 2,
    n_components: int = 64,
    top_n_products: int = 2000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Latent SVD cosine similarity, restricted to the top-N supported products.

    Builds the sparse customer-product matrix once, derives latent product
    vectors via TruncatedSVD and returns the dense pairwise cosine similarity
    (bounded to the largest ``top_n_products`` supported products so memory
    stays O(top_n^2) instead of O(n_products^2)).
    """
    full = _customer_product_matrix(transactions_df, customer_col, product_col, min_product_support)
    if full.shape[1] <= top_n_products:
        matrix = full
    else:
        support = full.sum(axis=0)
        keep = support.sort_values(ascending=False).index[:top_n_products]
        matrix = full[keep]

    from scipy import sparse as sp

    embeddings = build_product_embeddings(
        sp.csr_matrix(matrix.to_numpy().astype(np.float32)),
        n_components=n_components,
        random_seed=random_seed,
    )
    sim = embeddings @ embeddings.T
    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    products = matrix.columns.tolist()
    return pd.DataFrame(sim, index=products, columns=products)


SIMILARITY_METHODS = ("phi", "jaccard", "pmi", "cosine_tfidf", "embedding", "ensemble")


def build_similarity_matrix(
    transactions_df: pd.DataFrame,
    method: str = "phi",
    min_cooccurrence: int = 5,
    min_product_support: int = 2,
    n_components: int = 64,
    top_n_products: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Dispatch to the requested similarity method."""
    if method == "phi":
        return compute_phi_matrix(
            transactions_df,
            min_cooccurrence=min_cooccurrence,
            min_product_support=min_product_support,
            top_n_products=top_n_products,
        )
    if method == "jaccard":
        return compute_jaccard_matrix(
            transactions_df,
            min_cooccurrence=min_cooccurrence,
            min_product_support=min_product_support,
            top_n_products=top_n_products,
        )
    if method == "pmi":
        return compute_pmi_matrix(
            transactions_df,
            min_cooccurrence=min_cooccurrence,
            min_product_support=min_product_support,
            top_n_products=top_n_products,
        )
    if method == "cosine_tfidf":
        return compute_cosine_tfidf_matrix(
            transactions_df, min_product_support=min_product_support, top_n_products=top_n_products
        )
    if method == "embedding":
        return compute_embedding_matrix(
            transactions_df,
            min_product_support=min_product_support,
            n_components=n_components,
            top_n_products=top_n_products or 2000,
            random_seed=random_seed,
        )
    if method == "ensemble":
        return build_similarity_matrix_ensemble(
            transactions_df,
            min_cooccurrence=min_cooccurrence,
            min_product_support=min_product_support,
        )
    raise ValueError(f"unknown similarity method {method!r}; expected one of {SIMILARITY_METHODS}")


def build_similarity_matrix_ensemble(
    transactions_df: pd.DataFrame,
    methods: tuple[str, ...] = ("phi", "jaccard", "pmi", "cosine_tfidf"),
    weights: tuple[float, ...] | None = None,
    min_cooccurrence: int = 5,
    min_product_support: int = 2,
    top_n_products: int | None = None,
) -> pd.DataFrame:
    """Weighted average of z-standardized similarity matrices on shared products."""
    matrices: list[pd.DataFrame] = []
    for method in methods:
        try:
            matrices.append(
                build_similarity_matrix(
                    transactions_df,
                    method=method,
                    min_cooccurrence=min_cooccurrence,
                    min_product_support=min_product_support,
                    top_n_products=top_n_products,
                )
            )
        except ValueError:
            continue
    if not matrices:
        return pd.DataFrame()

    common = matrices[0].index
    for m in matrices[1:]:
        common = common.intersection(m.index)
    common = sorted(common)
    if len(common) < 2:
        return pd.DataFrame()

    scaled = []
    for m in matrices:
        sub = m.loc[common, common]
        mean = float(np.nanmean(sub.to_numpy()))
        std = float(np.nanstd(sub.to_numpy())) or 1.0
        scaled.append((sub - mean) / std)
    if weights is None:
        weights = tuple([1.0 / len(scaled)] * len(scaled))
    ensemble = sum(w * s for w, s in zip(weights, scaled, strict=True))
    ensemble = ensemble - ensemble.to_numpy().min()
    span = float(ensemble.to_numpy().max()) or 1.0
    ensemble = ensemble / span
    np.fill_diagonal(ensemble.to_numpy(), 1.0)
    return ensemble


def bootstrap_similarity_ci(
    transactions_df: pd.DataFrame,
    product_a: str,
    product_b: str,
    method: str = "phi",
    *,
    n_resamples: int = 100,
    ci_level: float = 0.95,
    random_seed: int | None = None,
    min_cooccurrence: int = 5,
    min_product_support: int = 2,
) -> dict[str, float]:
    """Percentile bootstrap CI for a single similarity pair (customer resampling).

    Builds the per-customer payload once, then each bootstrap draw resamples
    customer indices and computes the pair statistic directly from the
    contingency tables - no DataFrame concatenation, no matrix rebuild per
    draw. Phi is basket-based (as in ``compute_affinity_matrix``), Jaccard/PMI/
    TF-IDF are customer-based (as in the pairwise matrix methods).
    """
    rng = np.random.default_rng(random_seed)
    customers = transactions_df["customer_id"].unique().tolist()

    # Per-customer payload: products bought (counts for the customer basis)
    # and the baskets (transaction-level product sets) for the phi basis.
    cust_products: dict[str, frozenset] = {}
    cust_baskets: dict[str, list[frozenset]] = {}
    for c, g in transactions_df.groupby("customer_id"):
        baskets = [frozenset(b) for _, b in g.groupby("transaction_id")["stockcode"]]
        cust_products[c] = frozenset().union(*baskets) if baskets else frozenset()
        cust_baskets[c] = baskets

    product_sets = [cust_products[c] for c in customers]
    basket_lists = [cust_baskets[c] for c in customers]

    def _pair_value(cust_idx: np.ndarray) -> float:
        if method == "phi":
            baskets = [b for i in cust_idx for b in basket_lists[i]]
            n = len(baskets)
            both = sum(1 for b in baskets if product_a in b and product_b in b)
            only_a = sum(1 for b in baskets if product_a in b)
            only_b = sum(1 for b in baskets if product_b in b)
            if both < min_cooccurrence or n == 0 or only_a == 0 or only_b == 0:
                return 0.0
            numerator = both * n - only_a * only_b
            denominator = np.sqrt(only_a * (n - only_a) * only_b * (n - only_b))
            return float(numerator / denominator) if denominator > 0 else 0.0

        sets = [product_sets[i] for i in cust_idx]
        n = len(sets)
        both = sum(1 for s in sets if product_a in s and product_b in s)
        only_a = sum(1 for s in sets if product_a in s)
        only_b = sum(1 for s in sets if product_b in s)
        if both < min_cooccurrence or n == 0 or min(only_a, only_b) == 0:
            return 0.0
        if method == "jaccard":
            return float(both / (only_a + only_b - both))
        if method == "pmi":
            pa, pb, pab = only_a / n, only_b / n, both / n
            with np.errstate(divide="ignore", invalid="ignore"):
                numerator = np.log(pab / (pa * pb) + 1e-6)
                denominator = -np.log(pab + 1e-6)
                return float(np.clip(numerator / denominator, 0.0, 1.0)) if denominator > 0 else 0.0
        if method == "cosine_tfidf":
            idf_a = np.log((n + 1) / (only_a + 1)) + 1.0
            idf_b = np.log((n + 1) / (only_b + 1)) + 1.0
            dot = both * idf_a * idf_b
            norm = np.sqrt(only_a) * idf_a * np.sqrt(only_b) * idf_b
            return float(min(max(dot / norm, 0.0), 1.0)) if norm > 0 else 0.0
        if method == "embedding":
            raise ValueError("bootstrap_similarity_ci does not support method='embedding'")
        raise ValueError(f"unknown similarity method {method!r}")

    point = _pair_value(np.arange(len(customers)))
    replicates: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(customers), size=len(customers))
        replicates.append(_pair_value(idx))
    if not replicates:
        return {
            "estimate": point,
            "lower": point,
            "upper": point,
            "std_error": 0.0,
            "n_resamples": 0,
        }
    arr = np.asarray(replicates)
    alpha = 1.0 - ci_level
    return {
        "estimate": point,
        "lower": float(np.percentile(arr, 100 * alpha / 2)),
        "upper": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        "std_error": float(arr.std(ddof=1)),
        "n_resamples": len(arr),
    }
