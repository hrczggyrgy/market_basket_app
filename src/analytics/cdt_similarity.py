"""Customer Decision Tree: Similarity Engine.

Builds pairwise product similarity from customer purchase sequences using
Yule's Q coefficient derived from co-purchase patterns.
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

            for i in range(len(products) - 1):
                days_diff = (pd.Timestamp(dates[i + 1]) - pd.Timestamp(dates[i])).days
                if days_diff <= max_gap_days and products[i] != products[i + 1]:
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

    Returns:
        Dict mapping (prod_a, prod_b) -> {both, a_only, b_only, neither}
    """
    # Build customer-product matrix (one-hot)
    cust_product = pd.crosstab(transactions_df[customer_col], transactions_df[product_col])
    cust_product = (cust_product > 0).astype(int)

    products = cust_product.columns.tolist()
    n_customers = len(cust_product)

    # Precompute customer sets for each product
    product_customers = {
        prod: set(cust_product.index[cust_product[prod] == 1]) for prod in products
    }

    tables = {}
    for i, prod_a in enumerate(products):
        cust_a = product_customers[prod_a]
        for prod_b in products[i + 1 :]:
            cust_b = product_customers[prod_b]

            both = len(cust_a & cust_b)
            if both < min_cooccurrence:
                continue

            a_only = len(cust_a - cust_b)
            b_only = len(cust_b - cust_a)
            neither = n_customers - len(cust_a | cust_b)

            tables[(prod_a, prod_b)] = {
                "both": both,
                "a_only": a_only,
                "b_only": b_only,
                "neither": neither,
            }

    return tables


def compute_yules_q(table: Dict[str, int]) -> float:
    """
    Compute Yule's Q coefficient from 2x2 co-purchase table.

    Q = (ad - bc) / (ad + bc) where:
    - a = both, b = a_only, c = b_only, d = neither
    - Range: [-1, 1], where 1 = perfect association, -1 = perfect dissociation

    Args:
        table: Dict with keys 'both', 'a_only', 'b_only', 'neither'

    Returns:
        Yule's Q coefficient in [-1, 1]
    """
    a = table["both"]
    b = table["a_only"]
    c = table["b_only"]
    d = table["neither"]

    numerator = a * d - b * c
    denominator = a * d + b * c

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


def build_similarity_matrix(
    transactions_df: pd.DataFrame,
    customer_col: str = "customer_id",
    product_col: str = "stockcode",
    method: str = "yules_q",
    min_cooccurrence: int = 5,
) -> pd.DataFrame:
    """
    Build symmetric product similarity matrix.

    Args:
        transactions_df: Transaction DataFrame
        customer_col: Customer identifier column
        product_col: Product identifier column
        method: 'yules_q' or 'jaccard'
        min_cooccurrence: Minimum co-purchase count to compute similarity

    Returns:
        Square DataFrame (products x products) with similarity scores.
        Diagonal = 1.0. Values in [-1, 1] for Yule's Q, [0, 1] for Jaccard.
    """
    tables = build_copurchase_tables(transactions_df, customer_col, product_col, min_cooccurrence)

    products = sorted(transactions_df[product_col].unique())
    sim_matrix = pd.DataFrame(0.0, index=products, columns=products, dtype=float)
    np.fill_diagonal(sim_matrix.to_numpy(), 1.0)

    compute_fn = compute_yules_q if method == "yules_q" else compute_jaccard

    for (prod_a, prod_b), table in tables.items():
        sim = compute_fn(table)
        sim_matrix.loc[prod_a, prod_b] = sim
        sim_matrix.loc[prod_b, prod_a] = sim

    return sim_matrix


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
        total = from_totals[from_prod]
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
