"""Frequent Itemset Mining Algorithms: FP-Growth, Apriori, Eclat."""

import time
from collections import namedtuple
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse
from mlxtend.frequent_patterns import apriori, fpgrowth

# Named tuple returned by create_basket_matrix when sparse=True.
SparseBasket = namedtuple("SparseBasket", ["matrix", "index", "columns"])


def _postprocess_itemsets(freq_items: pd.DataFrame) -> pd.DataFrame:
    """Shared postprocessing: add itemset length, sort by support descending."""
    if freq_items.empty:
        return pd.DataFrame(columns=["support", "itemsets"])

    freq_items["length"] = freq_items["itemsets"].apply(len)
    freq_items = freq_items.sort_values("support", ascending=False).reset_index(drop=True)
    return freq_items


def run_fpgrowth(
    basket_df: pd.DataFrame,
    min_support: float = 0.01,
    max_len: int = 3,
    use_colnames: bool = True,
    verbose: int = 0,
) -> pd.DataFrame:
    """
    Run FP-Growth algorithm to find frequent itemsets.

    Args:
        basket_df: One-hot encoded transaction matrix (transactions x items)
        min_support: Minimum support threshold (0-1)
        max_len: Maximum length of itemsets
        use_colnames: Use column names as item names
        verbose: Verbosity level

    Returns:
        DataFrame with columns: support, itemsets (and length)
    """
    if basket_df.empty:
        return pd.DataFrame(columns=["support", "itemsets"])

    freq_items = fpgrowth(
        basket_df,
        min_support=min_support,
        max_len=max_len,
        use_colnames=use_colnames,
        verbose=verbose,
    )

    return _postprocess_itemsets(freq_items)


def run_apriori(
    basket_df: pd.DataFrame,
    min_support: float = 0.01,
    max_len: int = 3,
    use_colnames: bool = True,
    low_memory: bool = False,
    verbose: int = 0,
) -> pd.DataFrame:
    """
    Run Apriori algorithm to find frequent itemsets.

    Args:
        basket_df: One-hot encoded transaction matrix (transactions x items)
        min_support: Minimum support threshold (0-1)
        max_len: Maximum length of itemsets
        use_colnames: Use column names as item names
        low_memory: Use low memory mode (slower but less memory)
        verbose: Verbosity level

    Returns:
        DataFrame with columns: support, itemsets (and length)
    """
    if basket_df.empty:
        return pd.DataFrame(columns=["support", "itemsets"])

    freq_items = apriori(
        basket_df,
        min_support=min_support,
        max_len=max_len,
        use_colnames=use_colnames,
        low_memory=low_memory,
        verbose=verbose,
    )

    return _postprocess_itemsets(freq_items)


def run_eclat(
    basket_df: pd.DataFrame,
    min_support: float = 0.01,
    max_len: int = 3,
    min_combination: int = 1,
) -> pd.DataFrame:
    """
    Run Eclat algorithm to find frequent itemsets using vertical data format.

    Uses boolean numpy arrays (one per item) for tidsets so that intersection
    is a C-level bitwise AND (~100x faster than Python set intersection).

    Args:
        basket_df: One-hot encoded transaction matrix (transactions x items)
        min_support: Minimum support threshold (0-1)
        max_len: Maximum length of itemsets
        min_combination: Minimum combination size

    Returns:
        DataFrame with columns: support, itemsets (and length)
    """
    if basket_df.empty:
        return pd.DataFrame(columns=["support", "itemsets"])

    n_transactions = len(basket_df)
    min_support_count = int(min_support * n_transactions)

    # Vertical format: item -> boolean numpy array of length n_transactions
    item_tidsets = {}
    for item in basket_df.columns:
        arr = basket_df[item].values.astype(bool)
        if arr.sum() >= min_support_count:
            item_tidsets[item] = arr

    if not item_tidsets:
        return pd.DataFrame(columns=["support", "itemsets"])

    freq_itemsets = []

    def eclat_recursive(prefix_items, prefix_tids, items_list, start_idx):
        for i in range(start_idx, len(items_list)):
            item = items_list[i]
            tids = item_tidsets[item]
            new_tids = prefix_tids & tids  # C-level bitwise AND
            support_count = int(new_tids.sum())

            if support_count >= min_support_count:
                new_prefix = prefix_items + [item]
                support = support_count / n_transactions
                freq_itemsets.append((support, frozenset(new_prefix)))

                if len(new_prefix) < max_len:
                    eclat_recursive(new_prefix, new_tids, items_list, i + 1)

    # Seed prefix_tids as all-True array (every transaction)
    all_tids = np.ones(n_transactions, dtype=bool)
    items_list = list(item_tidsets.keys())
    eclat_recursive([], all_tids, items_list, 0)

    if not freq_itemsets:
        return pd.DataFrame(columns=["support", "itemsets"])

    df = pd.DataFrame(freq_itemsets, columns=["support", "itemsets"])
    df["length"] = df["itemsets"].apply(len)
    df = df.sort_values("support", ascending=False).reset_index(drop=True)

    return df


def run_algorithm(
    basket_df: pd.DataFrame,
    algorithm: Literal["fpgrowth", "apriori", "eclat"] = "fpgrowth",
    min_support: float = 0.01,
    max_len: int = 3,
    **kwargs,
) -> pd.DataFrame:
    """
    Run specified frequent itemset mining algorithm.

    Args:
        basket_df: One-hot encoded transaction matrix
        algorithm: Algorithm to use ('fpgrowth', 'apriori', 'eclat')
        min_support: Minimum support threshold
        max_len: Maximum itemset length
        **kwargs: Additional algorithm-specific parameters

    Returns:
        DataFrame with frequent itemsets
    """
    if algorithm == "fpgrowth":
        return run_fpgrowth(basket_df, min_support, max_len, **kwargs)
    elif algorithm == "apriori":
        return run_apriori(basket_df, min_support, max_len, **kwargs)
    elif algorithm == "eclat":
        return run_eclat(basket_df, min_support, max_len, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def create_basket_matrix(
    transactions_df: pd.DataFrame,
    transaction_col: str = "transaction_id",
    item_col: str = "stockcode",
    quantity_col: str = "quantity",
    min_quantity: int = 1,
    sparse: bool = False,
) -> "pd.DataFrame | SparseBasket":
    """
    Create one-hot encoded basket matrix from transaction data.

    Args:
        transactions_df: DataFrame with transaction_id, item, quantity columns
        transaction_col: Column name for transaction ID
        item_col: Column name for item identifier
        quantity_col: Column name for quantity
        min_quantity: Minimum quantity to consider item as present
        sparse: If True, return a SparseBasket(matrix, index, columns) namedtuple
                instead of a dense DataFrame. Callers must unpack index/columns
                separately — scipy CSR matrices do not support pandas-style metadata.

    Returns:
        Dense boolean DataFrame (transactions x items), or SparseBasket namedtuple
        when sparse=True.
    """
    df = transactions_df[transactions_df[quantity_col] >= min_quantity].copy()

    items = df[item_col].unique()
    transactions = df[transaction_col].unique()

    item_to_idx = {item: i for i, item in enumerate(items)}
    txn_to_idx = {txn: i for i, txn in enumerate(transactions)}

    n_transactions = len(transactions)
    n_items = len(items)

    row_indices = df[transaction_col].map(txn_to_idx).values
    col_indices = df[item_col].map(item_to_idx).values
    data = np.ones(len(df), dtype=bool)

    if sparse:
        coo = scipy.sparse.coo_matrix(
            (data, (row_indices, col_indices)), shape=(n_transactions, n_items), dtype=bool
        )
        csr = coo.tocsr()
        # Return metadata alongside the matrix — CSR does not support .index/.columns
        return SparseBasket(matrix=csr, index=transactions, columns=items)
    else:
        coo = scipy.sparse.coo_matrix(
            (data, (row_indices, col_indices)), shape=(n_transactions, n_items), dtype=bool
        )
        csr = coo.tocsr()
        basket = pd.DataFrame.sparse.from_spmatrix(csr, index=transactions, columns=items)
        basket = basket.astype(bool)
        return basket


def get_product_lookup(
    transactions_df: pd.DataFrame,
    code_col: str = "stockcode",
    name_col: str = "product",
) -> dict:
    """Map stockcode to product name; first occurrence wins on duplicate codes."""
    deduped = transactions_df.drop_duplicates(subset=[code_col])
    return dict(zip(deduped[code_col].astype(str), deduped[name_col]))


def compare_algorithms(
    basket_df: pd.DataFrame,
    min_support: float = 0.01,
    max_len: int = 3,
) -> pd.DataFrame:
    """
    Compare results and wall-clock execution time from all three algorithms.

    Returns:
        DataFrame with comparison metrics including elapsed_seconds per algorithm.
    """
    results = {}

    for algo in ["fpgrowth", "apriori", "eclat"]:
        try:
            t0 = time.perf_counter()
            freq = run_algorithm(basket_df, algo, min_support, max_len)
            elapsed = time.perf_counter() - t0
            results[algo] = {
                "n_itemsets": len(freq),
                "max_support": freq["support"].max() if not freq.empty else 0,
                "avg_support": freq["support"].mean() if not freq.empty else 0,
                "max_length": freq["length"].max() if not freq.empty else 0,
                "elapsed_seconds": round(elapsed, 4),
                "error": None,
            }
        except Exception as e:
            results[algo] = {
                "n_itemsets": 0,
                "max_support": 0,
                "avg_support": 0,
                "max_length": 0,
                "elapsed_seconds": None,
                "error": str(e),
            }

    return pd.DataFrame(results).T
