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

from src.analytics.copurchase import compute_affinity_matrix


def _customer_product_matrix(
    transactions_df: pd.DataFrame,
    customer_col: str = "customer_id",
    product_col: str = "stockcode",
    min_product_support: int = 2,
) -> pd.DataFrame:
    """Binary customer x product matrix, dropping rare products."""
    cust_product = pd.crosstab(transactions_df[customer_col], transactions_df[product_col])
    support = cust_product.sum(axis=0)
    valid = support[support >= min_product_support].index
    return (cust_product[valid] > 0).astype(int)


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
) -> pd.DataFrame:
    """Phi-coefficient matrix on the customer-product matrix (reuses affinity)."""
    matrix = compute_affinity_matrix(
        transactions_df, min_cooccurrence=min_cooccurrence
    )
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
) -> pd.DataFrame:
    """Jaccard similarity: |A∩B| / |A∪B| over shared customers."""
    matrix = _customer_product_matrix(transactions_df, customer_col, product_col, min_product_support)
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
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Pointwise mutual information normalized to [0, 1].

    PMI = log(P(a,b) / (P(a) P(b))), rescaled by -log P(a,b) so that the
    result lies in [0, 1] and is symmetric.
    """
    matrix = _customer_product_matrix(transactions_df, customer_col, product_col, min_product_support)
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
) -> pd.DataFrame:
    """TF-IDF weighted cosine similarity between products over baskets.

    Each customer is a document; products with few customers receive higher
    IDF weight, so niche but consistent products are not drowned out by
    best-sellers.
    """
    matrix = _customer_product_matrix(transactions_df, customer_col, product_col, min_product_support)
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


SIMILARITY_METHODS = ("phi", "jaccard", "pmi", "cosine_tfidf", "ensemble")


def build_similarity_matrix(
    transactions_df: pd.DataFrame,
    method: str = "phi",
    min_cooccurrence: int = 5,
    min_product_support: int = 2,
) -> pd.DataFrame:
    """Dispatch to the requested similarity method."""
    if method == "phi":
        return compute_phi_matrix(transactions_df, min_cooccurrence=min_cooccurrence, min_product_support=min_product_support)
    if method == "jaccard":
        return compute_jaccard_matrix(transactions_df, min_cooccurrence=min_cooccurrence, min_product_support=min_product_support)
    if method == "pmi":
        return compute_pmi_matrix(transactions_df, min_cooccurrence=min_cooccurrence, min_product_support=min_product_support)
    if method == "cosine_tfidf":
        return compute_cosine_tfidf_matrix(transactions_df, min_product_support=min_product_support)
    if method == "ensemble":
        return build_similarity_matrix_ensemble(
            transactions_df, min_cooccurrence=min_cooccurrence, min_product_support=min_product_support
        )
    raise ValueError(f"unknown similarity method {method!r}; expected one of {SIMILARITY_METHODS}")


def build_similarity_matrix_ensemble(
    transactions_df: pd.DataFrame,
    methods: tuple[str, ...] = ("phi", "jaccard", "pmi", "cosine_tfidf"),
    weights: tuple[float, ...] | None = None,
    min_cooccurrence: int = 5,
    min_product_support: int = 2,
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
    ensemble = sum(w * s for w, s in zip(weights, scaled))
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
) -> dict[str, float]:
    """Percentile bootstrap CI for a single similarity pair (customer resampling)."""
    rng = np.random.default_rng(random_seed)
    customers = transactions_df["customer_id"].unique()
    cust_groups = {c: g for c, g in transactions_df.groupby("customer_id")}

    def _pair_value(d: pd.DataFrame) -> float:
        matrix = build_similarity_matrix(d, method=method, min_cooccurrence=min_cooccurrence)
        if product_a not in matrix.index or product_b not in matrix.index:
            return 0.0
        return float(matrix.loc[product_a, product_b])

    point = _pair_value(transactions_df)
    replicates: list[float] = []
    for _ in range(n_resamples):
        cust_idx = rng.integers(0, len(customers), size=len(customers))
        frames = [cust_groups[c] for c in customers[cust_idx]]
        resample = pd.concat(frames, ignore_index=True)
        if resample.empty:
            continue
        replicates.append(_pair_value(resample))
    if not replicates:
        return {"estimate": point, "lower": point, "upper": point, "std_error": 0.0, "n_resamples": 0}
    arr = np.asarray(replicates)
    alpha = 1.0 - ci_level
    return {
        "estimate": point,
        "lower": float(np.percentile(arr, 100 * alpha / 2)),
        "upper": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        "std_error": float(arr.std(ddof=1)),
        "n_resamples": len(arr),
    }