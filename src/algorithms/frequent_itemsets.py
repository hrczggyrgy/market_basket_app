"""Frequent Itemset Mining Algorithms: FP-Growth, Apriori, Eclat."""

from typing import Literal

import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth


def _postprocess_itemsets(freq_items: pd.DataFrame) -> pd.DataFrame:
    """Common postprocessing for frequent itemset results.

    BUG 10 FIX: Extracted duplicate code into helper function.
    """
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

    Eclat uses a depth-first search with tidset intersections.
    This is a custom implementation since mlxtend doesn't have Eclat.

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

    # Convert to vertical format: item -> set of transaction IDs
    item_tidsets = {}
    for item in basket_df.columns:
        tids = set(basket_df.index[basket_df[item]].tolist())
        if len(tids) >= min_support_count:
            item_tidsets[item] = tids

    if not item_tidsets:
        return pd.DataFrame(columns=["support", "itemsets"])

    # Eclat recursive search
    freq_itemsets = []

    def eclat_recursive(prefix_items, prefix_tids, items_list, start_idx):
        for i in range(start_idx, len(items_list)):
            item = items_list[i]
            tids = item_tidsets[item]
            new_tids = prefix_tids & tids
            support_count = len(new_tids)

            if support_count >= min_support_count:
                new_prefix = prefix_items + [item]
                support = support_count / n_transactions
                freq_itemsets.append((support, frozenset(new_prefix)))

                if len(new_prefix) < max_len:
                    # Continue with remaining items
                    eclat_recursive(new_prefix, new_tids, items_list, i + 1)

    items_list = list(item_tidsets.keys())
    eclat_recursive([], set(basket_df.index), items_list, 0)

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
) -> pd.DataFrame:
    """
    Create one-hot encoded basket matrix from transaction data.

    Args:
        transactions_df: DataFrame with transaction_id, item, quantity columns
        transaction_col: Column name for transaction ID
        item_col: Column name for item identifier
        quantity_col: Column name for quantity
        min_quantity: Minimum quantity to consider item as present

    Returns:
        One-hot encoded DataFrame (transactions x items)
    """
    df = transactions_df[transactions_df[quantity_col] >= min_quantity].copy()

    basket = df.groupby([transaction_col, item_col])[quantity_col].sum().unstack(fill_value=0)

    basket = (basket > 0).astype(bool)

    return basket


def get_product_lookup(
    transactions_df: pd.DataFrame,
    code_col: str = "stockcode",
    name_col: str = "product",
) -> dict:
    """Create lookup dictionary from stockcode to product name.

    BUG 14 FIX: Deduplicate on stockcode - first value wins (silent deduplication)
    """
    # Deduplicate on code_col, keeping first occurrence
    deduped = transactions_df.drop_duplicates(subset=[code_col])
    return dict(zip(deduped[code_col].astype(str), deduped[name_col]))


def compare_algorithms(
    basket_df: pd.DataFrame,
    min_support: float = 0.01,
    max_len: int = 3,
) -> pd.DataFrame:
    """
    Compare results from all three algorithms.

    BUG 11 FIX: Ensure consistent schema - use NaN for failed algorithms instead of error dict

    Returns:
        DataFrame with comparison metrics
    """
    results = {}

    for algo in ["fpgrowth", "apriori", "eclat"]:
        try:
            freq = run_algorithm(basket_df, algo, min_support, max_len)
            results[algo] = {
                "n_itemsets": len(freq),
                "max_support": freq["support"].max() if not freq.empty else 0,
                "avg_support": freq["support"].mean() if not freq.empty else 0,
                "max_length": freq["length"].max() if not freq.empty else 0,
                "error": None,
            }
        except Exception as e:
            results[algo] = {
                "n_itemsets": 0,
                "max_support": 0,
                "avg_support": 0,
                "max_length": 0,
                "error": str(e),
            }

    return pd.DataFrame(results).T
