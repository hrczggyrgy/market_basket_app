"""Customer Decision Tree: Similarity Engine.

Builds pairwise product similarity from customer purchase sequences using
the Phi coefficient derived from co-purchase patterns.
"""

import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def build_customer_sequences(
    transactions_df: pd.DataFrame,
    customer_col: str = "customer_id",
    product_col: str = "stockcode",
    date_col: str = "date",
) -> Dict[str, List[str]]:
    """
    Build chronological product sequences per customer.

    Args:
        transactions_df: Transaction DataFrame with customer, product, date columns
        customer_col: Customer identifier column
        product_col: Product identifier column
        date_col: Transaction date column

    Returns:
        Dict mapping customer_id -> ordered list of product_ids (chronological)
    """
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([customer_col, date_col])

    # Group by transaction to avoid within-basket consecutive pairs
    if "transaction_id" in df.columns:
        baskets = (
            df.groupby([customer_col, "transaction_id", date_col])[product_col]
            .apply(list)
            .reset_index()
        )
        baskets = baskets.sort_values([customer_col, date_col])
        sequences = (
            baskets.groupby(customer_col)[product_col]
            .apply(lambda x: [p for items in x for p in items])
            .to_dict()
        )
    else:
        sequences = df.groupby(customer_col)[product_col].apply(list).to_dict()
    return sequences


def detect_switches(
    sequences: Dict[str, List[str]],
    max_gap_days: int = 90,
    transactions_df: Optional[pd.DataFrame] = None,
    customer_col: str = "customer_id",
    product_col: str = "stockcode",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Detect product-to-product switches in customer sequences.

    A switch occurs when a customer buys product A then product B
    within max_gap_days (or consecutively in sequence if no dates).

    Args:
        sequences: Customer sequences from build_customer_sequences
        max_gap_days: Maximum days between purchases to count as switch
        transactions_df: Optional original DF for date-based gap calculation
        customer_col, product_col, date_col: Column names if using transactions_df

    Returns:
        DataFrame with columns: from_product, to_product, customer_id, days_between
    """
    switches = []

    if transactions_df is not None:
        # Date-aware switching using original transaction dates
        df = transactions_df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([customer_col, date_col])

        for customer, group in df.groupby(customer_col):
            group = group.sort_values(date_col)
            products = group[product_col].values
            dates = group[date_col].values
            transaction_ids = group.get("transaction_id", pd.Series([None] * len(group))).values

            for i in range(len(products) - 1):
                days_diff = (pd.Timestamp(dates[i + 1]) - pd.Timestamp(dates[i])).days
                same_basket = (
                    transaction_ids[i] is not None
                    and transaction_ids[i + 1] is not None
                    and transaction_ids[i] == transaction_ids[i + 1]
                )
                if days_diff <= max_gap_days and products[i] != products[i + 1] and not same_basket:
                    switches.append(
                        {
                            "from_product": products[i],
                            "to_product": products[i + 1],
                            "customer_id": customer,
                            "days_between": days_diff,
                        }
                    )
    else:
        # Sequence-only switching (consecutive different products)
        for customer, products in sequences.items():
            for i in range(len(products) - 1):
                if products[i] != products[i + 1]:
                    switches.append(
                        {
                            "from_product": products[i],
                            "to_product": products[i + 1],
                            "customer_id": customer,
                            "days_between": None,
                        }
                    )

    if not switches:
        return pd.DataFrame(columns=["from_product", "to_product", "customer_id", "days_between"])

    return pd.DataFrame(switches)


def build_copurchase_tables(
    transactions_df: pd.DataFrame,
    customer_col: str = "customer_id",
    product_col: str = "stockcode",
    min_cooccurrence: int = 5,
    min_product_support: int = 2,
) -> Dict[Tuple[str, str], Dict[str, int]]:
    """
    Build 2x2 co-purchase contingency tables for all product pairs.

    For each pair (A, B), counts across all customers:
    - both: customers who bought both A and B
    - a_only: customers who bought A but not B
    - b_only: customers who bought B but not A
    - neither: customers who bought neither

    Args:
        transactions_df: Transaction DataFrame
        customer_col: Customer identifier column
        product_col: Product identifier column
        min_cooccurrence: Minimum customers buying both to include pair
        min_product_support: Minimum customers buying a single product to include it

    Returns:
        Dict mapping (prod_a, prod_b) -> {both, a_only, b_only, neither}
    """
    # Build binary customer-product matrix
    cust_product = pd.crosstab(transactions_df[customer_col], transactions_df[product_col])
    cust_product = (cust_product > 0).astype(int)

    # Filter rare products
    product_support = cust_product.sum(axis=0)
    valid_products = product_support[product_support >= min_product_support].index
    cust_product = cust_product[valid_products]

    products = cust_product.columns.tolist()
    n_customers = len(cust_product)

    # Co-occurrence via matrix multiplication (vectorized)
    cooccurrence = cust_product.T @ cust_product

    tables = {}
    for i, prod_a in enumerate(products):
        for prod_b in products[i + 1 :]:
            both = cooccurrence.loc[prod_a, prod_b]
            if both < min_cooccurrence:
                continue

            prod_a_count = product_support[prod_a]
            prod_b_count = product_support[prod_b]
            a_only = prod_a_count - both
            b_only = prod_b_count - both
            neither = n_customers - (prod_a_count + prod_b_count - both)

            tables[(prod_a, prod_b)] = {
                "both": both,
                "a_only": a_only,
                "b_only": b_only,
                "neither": neither,
            }

    return tables


def compute_phi_coefficient(table: Dict[str, int]) -> float:
    """
    Compute Phi coefficient from 2x2 co-purchase table.

    φ = (ad - bc) / sqrt((a+b)(a+c)(b+d)(c+d)) where:
    - a = both, b = a_only, c = b_only, d = neither
    - Range: [-1, 1], where 1 = perfect association, -1 = perfect dissociation

    Args:
        table: Dict with keys 'both', 'a_only', 'b_only', 'neither'

    Returns:
        Phi coefficient in [-1, 1]
    """
    a = table["both"]
    b = table["a_only"]
    c = table["b_only"]
    d = table["neither"]

    numerator = a * d - b * c
    denominator = np.sqrt((a + b) * (a + c) * (b + d) * (c + d))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def compute_jaccard(table: Dict[str, int]) -> float:
    """
    Compute Jaccard similarity from 2x2 co-purchase table.

    J = both / (a_only + b_only + both) = intersection / union

    Args:
        table: Dict with keys 'both', 'a_only', 'b_only', 'neither'

    Returns:
        Jaccard similarity in [0, 1]
    """
    both = table["both"]
    a_only = table["a_only"]
    b_only = table["b_only"]

    union = both + a_only + b_only
    if union == 0:
        return 0.0

    return both / union


def compute_pmi_matrix(
    transactions_df: pd.DataFrame,
    customer_col: str = "customer_id",
    product_col: str = "stockcode",
    min_cooccurrence: int = 5,
    min_product_support: int = 2,
    smoothing: float = 1.0,
    ppmi: bool = True,
) -> pd.DataFrame:
    """
    Compute Pointwise Mutual Information (PMI) matrix on customer-product co-occurrence.

    PMI(A, B) = log( P(A,B) / (P(A) * P(B)) )
    PPMI = max(PMI, 0)

    Args:
        transactions_df: Transaction DataFrame
        customer_col: Customer identifier column
        product_col: Product identifier column
        min_cooccurrence: Minimum co-occurrence count
        min_product_support: Minimum product support
        smoothing: Additive smoothing constant
        ppmi: If True, return Positive PMI (max(0, PMI))

    Returns:
        Square DataFrame (products x products) with PMI/PPMI scores
    """
    # Binary customer-product matrix
    cust_product = pd.crosstab(transactions_df[customer_col], transactions_df[product_col])
    cust_product = (cust_product > 0).astype(int)

    # Filter rare products
    product_support = cust_product.sum(axis=0)
    valid_mask = product_support >= min_product_support
    products = product_support.index[valid_mask].tolist()
    cust_product = cust_product[products]

    n_customers = len(cust_product)

    # Co-occurrence matrix (vectorized)
    cooccurrence = cust_product.T @ cust_product
    both = cooccurrence.values.astype(float)

    # Marginal probabilities with smoothing
    p_a = (product_support[products].values + smoothing) / (n_customers + 2 * smoothing)
    p_b = p_a.copy()

    # Joint probabilities
    p_ab = (both + smoothing) / (n_customers + smoothing)

    # PMI: log(p_ab / (p_a * p_b))
    p_a_outer = p_a[:, np.newaxis]
    p_b_outer = p_b[np.newaxis, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(p_ab / (p_a_outer * p_b_outer))
        pmi = np.nan_to_num(pmi, nan=0.0, posinf=0.0, neginf=0.0)

    if ppmi:
        pmi = np.maximum(pmi, 0.0)

    # Apply min_cooccurrence mask
    pmi[both < min_cooccurrence] = 0.0
    np.fill_diagonal(pmi, 1.0)

    return pd.DataFrame(pmi, index=products, columns=products)


def compute_cosine_tfidf_matrix(
    transactions_df: pd.DataFrame,
    customer_col: str = "customer_id",
    product_col: str = "stockcode",
    min_product_support: int = 2,
) -> pd.DataFrame:
    """
    Compute cosine similarity on TF-IDF weighted customer-product matrix.

    TF: binary purchase indicator per customer
    IDF: log(n_customers / n_customers_buying_product)
    
    Cosine similarity = (TF-IDF vector_A) dot (TF-IDF vector_B) / (|A| * |B|)

    Args:
        transactions_df: Transaction DataFrame
        customer_col: Customer identifier column
        product_col: Product identifier column
        min_product_support: Minimum customers buying a product

    Returns:
        Square DataFrame (products x products) with cosine similarity in [0, 1]
    """
    # Binary customer-product matrix
    cust_product = pd.crosstab(transactions_df[customer_col], transactions_df[product_col])
    cust_product = (cust_product > 0).astype(int)

    # Filter rare products
    product_support = cust_product.sum(axis=0)
    valid_mask = product_support >= min_product_support
    products = product_support.index[valid_mask].tolist()
    cust_product = cust_product[products]

    n_customers = len(cust_product)

    # IDF weights
    idf = np.log(n_customers / (product_support[products].values + 1e-10))

    # TF-IDF matrix (sparse-friendly: multiply rows by IDF)
    tfidf = cust_product.values * idf[np.newaxis, :]

    # Cosine similarity via normalized matrix multiplication
    norms = np.linalg.norm(tfidf, axis=0)
    norms[norms == 0] = 1.0
    tfidf_norm = tfidf / norms[np.newaxis, :]

    sim = tfidf_norm.T @ tfidf_norm
    np.fill_diagonal(sim, 1.0)

    return pd.DataFrame(sim, index=products, columns=products)


def _build_similarity_matrix_vectorized(
    transactions_df: pd.DataFrame,
    customer_col: str = "customer_id",
    product_col: str = "stockcode",
    method: str = "phi",
    min_cooccurrence: int = 5,
    min_product_support: int = 2,
) -> pd.DataFrame:
    """
    Build product similarity matrix using fully vectorized matrix operations.

    Uses binary customer-product matrix multiplication to compute all pairwise
    co-occurrence counts in O(n^2) numpy, avoiding Python loops over pairs.
    """
    # Binary customer-product matrix
    cust_product = pd.crosstab(transactions_df[customer_col], transactions_df[product_col])
    cust_product = (cust_product > 0).astype(int)

    # Filter rare products
    product_support = cust_product.sum(axis=0)
    valid_mask = product_support >= min_product_support
    products = product_support.index[valid_mask].tolist()
    cust_product = cust_product[products]

    n_customers = len(cust_product)

    # Co-occurrence matrix (vectorized): count of shared customers per product pair
    cooccurrence = cust_product.T @ cust_product
    both = cooccurrence.values

    product_counts = product_support[products].values  # 1D array

    # a_only[i,j] = customers who bought i but not j
    a_only = product_counts[:, np.newaxis] - both
    # b_only[i,j] = customers who bought j but not i
    b_only = product_counts[np.newaxis, :] - both
    # neither[i,j] = customers who bought neither
    neither = n_customers - (product_counts[:, np.newaxis] + product_counts[np.newaxis, :] - both)

    if method == "phi":
        numerator = both * neither - a_only * b_only
        denominator = np.sqrt(
            (both + a_only) * (both + b_only) * (a_only + neither) * (b_only + neither)
        )
        sim = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator != 0,
        )
    elif method == "jaccard":
        # Jaccard: intersection / union
        union = both + a_only + b_only
        sim = np.divide(both, union, out=np.zeros_like(both, dtype=float), where=union != 0)
    else:
        raise ValueError(f"Unknown method for vectorized: {method}")

    # Apply min_cooccurrence mask: zero out pairs below threshold
    sim[both < min_cooccurrence] = 0.0
    np.fill_diagonal(sim, 1.0)

    return pd.DataFrame(sim, index=products, columns=products)


def build_similarity_matrix(
    transactions_df: pd.DataFrame,
    customer_col: str = "customer_id",
    product_col: str = "stockcode",
    method: str = "phi",
    min_cooccurrence: int = 5,
    min_product_support: int = 2,
) -> pd.DataFrame:
    """
    Build symmetric product similarity matrix.

    Args:
        transactions_df: Transaction DataFrame
        customer_col: Customer identifier column
        product_col: Product identifier column
        method: 'phi', 'jaccard', 'pmi', 'cosine_tfidf'
        min_cooccurrence: Minimum co-purchase count to compute similarity
        min_product_support: Minimum customers buying a product to include it

    Returns:
        Square DataFrame (products x products) with similarity scores.
        Diagonal = 1.0. Values in [-1, 1] for Phi, [0, 1] for Jaccard/PMI/Cosine.
    """
    if method == "phi":
        return _build_similarity_matrix_vectorized(
            transactions_df, customer_col, product_col, "phi", min_cooccurrence, min_product_support
        )
    elif method == "jaccard":
        return _build_similarity_matrix_vectorized(
            transactions_df, customer_col, product_col, "jaccard", min_cooccurrence, min_product_support
        )
    elif method == "pmi":
        return compute_pmi_matrix(
            transactions_df, customer_col, product_col, min_cooccurrence, min_product_support
        )
    elif method == "cosine_tfidf":
        return compute_cosine_tfidf_matrix(
            transactions_df, customer_col, product_col, min_product_support
        )
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def build_similarity_matrix_ensemble(
    transactions_df: pd.DataFrame,
    customer_col: str = "customer_id",
    product_col: str = "stockcode",
    methods: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    min_cooccurrence: int = 5,
    min_product_support: int = 2,
) -> Dict[str, pd.DataFrame]:
    """
    Build multiple similarity matrices and optionally return weighted ensemble.

    Args:
        transactions_df: Transaction DataFrame
        customer_col: Customer identifier column
        product_col: Product identifier column
        methods: List of methods to compute (default: ['phi', 'jaccard', 'pmi', 'cosine_tfidf'])
        weights: Dict mapping method -> weight for ensemble (default: equal)
        min_cooccurrence: Minimum co-purchase count
        min_product_support: Minimum product support

    Returns:
        Dict with individual matrices + 'ensemble' key if weights provided
    """
    if methods is None:
        methods = ["phi", "jaccard", "pmi", "cosine_tfidf"]

    matrices = {}
    for method in methods:
        matrices[method] = build_similarity_matrix(
            transactions_df, customer_col, product_col, method, min_cooccurrence, min_product_support
        )

    if weights:
        # Align on common products
        common_products = set(matrices[methods[0]].index)
        for m in methods[1:]:
            common_products &= set(matrices[m].index)
        common_products = sorted(common_products)

        ensemble = pd.DataFrame(0.0, index=common_products, columns=common_products)
        total_weight = sum(weights.get(m, 1.0) for m in methods)

        for method in methods:
            weight = weights.get(method, 1.0)
            mat = matrices[method].loc[common_products, common_products]
            ensemble += weight * mat

        ensemble /= total_weight
        np.fill_diagonal(ensemble.values, 1.0)
        matrices["ensemble"] = ensemble

    return matrices


def compute_switching_matrix_from_sequences(
    sequences: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Compute product-to-product switching rates from sequences.

    Args:
        sequences: Customer sequences from build_customer_sequences

    Returns:
        DataFrame with from_product, to_product, switch_count, switch_rate
    """
    switch_counts = defaultdict(lambda: defaultdict(int))
    from_totals = defaultdict(int)

    for customer, products in sequences.items():
        for i in range(len(products) - 1):
            if products[i] != products[i + 1]:
                switch_counts[products[i]][products[i + 1]] += 1
                from_totals[products[i]] += 1

    rows = []
    for from_prod, targets in switch_counts.items():
        total = from_totals.get(from_prod, 0)
        for to_prod, count in targets.items():
            rows.append(
                {
                    "from_product": from_prod,
                    "to_product": to_prod,
                    "switch_count": count,
                    "switch_rate": count / total if total > 0 else 0.0,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["from_product", "to_product", "switch_count", "switch_rate"])

    df = pd.DataFrame(rows)
    df = df.sort_values("switch_count", ascending=False).reset_index(drop=True)
    return df


def _hash_dataframe(df: pd.DataFrame, cols: List[str]) -> str:
    """Create hash of dataframe for caching."""
    subset = df[cols].copy()
    subset = subset.sort_values(cols).reset_index(drop=True)
    return hashlib.md5(subset.to_json().encode()).hexdigest()[:16]


def get_cached_similarity_key(
    transactions_df: pd.DataFrame,
    customer_col: str,
    product_col: str,
    method: str,
    min_cooccurrence: int,
) -> str:
    """Generate cache key for similarity matrix."""
    data_hash = _hash_dataframe(transactions_df, [customer_col, product_col, "date"])
    param_str = f"{customer_col}_{product_col}_{method}_{min_cooccurrence}"
    return f"sim_{data_hash}_{param_str}"
