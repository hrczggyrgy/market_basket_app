"""Co-purchase affinity analysis.

Affinity = phi coefficient (via sklearn.metrics.matthews_corrcoef, which is
exactly phi for binary vectors) between product purchase vectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.rules import create_basket_matrix
from src.analytics.schemas import AFFINITY_PAIRS, check


def _affinity_and_cooccurrence(
    df: pd.DataFrame,
    min_cooccurrence: int,
    top_n_products: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    basket = create_basket_matrix(df)
    if top_n_products is not None:
        counts = basket.sum().sort_values(ascending=False)
        basket = basket[counts.head(top_n_products).index]
    M = basket.to_numpy(dtype=bool)
    n = M.shape[0]
    cooccur = (M.T @ M.astype(np.int64)).astype(int)
    counts = M.sum(axis=0).astype(float)
    numerator = cooccur * n - np.outer(counts, counts)
    denominator = np.sqrt(
        np.outer(counts, n - counts) * np.outer(n - counts, counts)
    )
    phi = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )
    phi = np.where(cooccur >= min_cooccurrence, phi, np.nan)
    np.fill_diagonal(phi, 1.0)
    products = basket.columns.tolist()
    support = counts / n
    return (
        pd.DataFrame(phi, index=products, columns=products),
        pd.DataFrame(cooccur / n, index=products, columns=products),
        support,
    )


def compute_affinity_matrix(
    df: pd.DataFrame,
    min_cooccurrence: int = 5,
    top_n_products: int | None = None,
) -> pd.DataFrame:
    """Product x product phi-coefficient affinity matrix (NaN below min co-occurrence)."""
    affinity, _, _ = _affinity_and_cooccurrence(df, min_cooccurrence, top_n_products)
    return affinity


def get_top_affinity_pairs(
    df: pd.DataFrame,
    top_n: int = 20,
    min_cooccurrence: int = 5,
    min_affinity: float = 0.0,
    top_n_products: int | None = 200,
) -> pd.DataFrame:
    """Highest-affinity product pairs with co-occurrence rates.

    `top_n_products` limits the candidate pool to the most-purchased products,
    which keeps pair enumeration tractable on large catalogs.
    """
    affinity, cooccur, support = _affinity_and_cooccurrence(df, min_cooccurrence, top_n_products)
    products = affinity.columns.tolist()
    rows = []
    for i in range(len(products)):
        for j in range(i + 1, len(products)):
            value = affinity.iloc[i, j]
            if np.isnan(value):
                continue
            rows.append(
                {
                    "product_a": products[i],
                    "product_b": products[j],
                    "affinity": float(value),
                    "cooccurrence": float(cooccur.iloc[i, j]),
                    "support_a": float(support[i]),
                    "support_b": float(support[j]),
                }
            )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return check(pd.DataFrame(columns=list(AFFINITY_PAIRS.columns)), AFFINITY_PAIRS, allow_empty=True)
    pairs = pairs[pairs["affinity"].ge(min_affinity)]
    pairs = pairs.sort_values("affinity", ascending=False).head(top_n).reset_index(drop=True)
    return check(pairs, AFFINITY_PAIRS)


def get_product_affinity_profile(df: pd.DataFrame, product: str, top_n: int = 10) -> pd.DataFrame:
    """Top co-purchase partners for a single product."""
    affinity = compute_affinity_matrix(df, min_cooccurrence=2)
    if product not in affinity.index:
        return check(pd.DataFrame(columns=list(AFFINITY_PAIRS.columns)), AFFINITY_PAIRS, allow_empty=True)
    row = affinity.loc[product].drop(labels=[product]).dropna().sort_values(ascending=False)
    partners = row.head(top_n).index.tolist()
    pairs = get_top_affinity_pairs(df, top_n=10_000, min_cooccurrence=2)
    mask = ((pairs["product_a"] == product) & pairs["product_b"].isin(partners)) | (
        (pairs["product_b"] == product) & pairs["product_a"].isin(partners)
    )
    return check(pairs.loc[mask].reset_index(drop=True), AFFINITY_PAIRS, allow_empty=True)
