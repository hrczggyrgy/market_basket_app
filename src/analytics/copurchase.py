"""Co-purchase / Affinity analysis."""

from typing import Optional

import numpy as np
import pandas as pd

from src.algorithms.fpgrowth import create_basket_matrix, run_fpgrowth


def compute_affinity_matrix(
    transactions_df: pd.DataFrame,
    min_support: float = 0.005,
    max_len: int = 2,
    min_lift: float = 1.0,
    top_n_products: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute product affinity matrix using lift.

    Returns:
        Square matrix (products x products) with lift values
    """
    # Filter to top N products if specified
    if top_n_products:
        top_products = transactions_df["stockcode"].value_counts().head(top_n_products).index
        transactions_df = transactions_df[transactions_df["stockcode"].isin(top_products)]

    # Create basket matrix
    basket = create_basket_matrix(transactions_df)

    # Run FP-Growth for pairs
    freq_items = run_fpgrowth(basket, min_support=min_support, max_len=max_len)

    if freq_items.empty:
        return pd.DataFrame()

    # Extract 2-itemsets
    pairs = freq_items[freq_items["length"] == 2].copy()
    if pairs.empty:
        return pd.DataFrame()

    # Build affinity matrix
    products = sorted(basket.columns.tolist())
    affinity = pd.DataFrame(1.0, index=products, columns=products, dtype=float)

    # Calculate individual product probabilities
    product_probs = basket.mean()

    for _, row in pairs.iterrows():
        items = list(row["itemsets"])
        if len(items) == 2:
            a, b = items[0], items[1]
            p_a = product_probs.get(a, 0)
            p_b = product_probs.get(b, 0)
            p_ab = row["support"]

            if p_a > 0 and p_b > 0:
                lift = p_ab / (p_a * p_b)
                if lift >= min_lift:
                    affinity.loc[a, b] = lift
                    affinity.loc[b, a] = lift

    return affinity


def get_top_affinity_pairs(
    transactions_df: pd.DataFrame,
    min_support: float = 0.005,
    min_lift: float = 1.2,
    top_n: int = 50,
    top_n_products: Optional[int] = None,
) -> pd.DataFrame:
    """
    Get top co-purchase pairs ranked by lift.

    Returns:
        DataFrame with product_a, product_b, support, confidence, lift, leverage
        plus academically useful symmetric association metrics.
    """
    empty_cols = [
        "product_a",
        "product_b",
        "support",
        "confidence_a_to_b",
        "confidence_b_to_a",
        "lift",
        "leverage",
        "conviction_a_to_b",
        "conviction_b_to_a",
        "jaccard",
        "kulczynski",
        "cosine",
        "phi_coefficient",
    ]

    if top_n_products:
        top_products = transactions_df["stockcode"].value_counts().head(top_n_products).index
        transactions_df = transactions_df[transactions_df["stockcode"].isin(top_products)]

    basket = create_basket_matrix(transactions_df)
    freq_items = run_fpgrowth(basket, min_support=min_support, max_len=2)

    if freq_items.empty:
        return pd.DataFrame(columns=empty_cols)

    pairs = freq_items[freq_items["length"] == 2].copy()
    if pairs.empty:
        return pd.DataFrame(columns=empty_cols)

    product_probs = basket.mean()
    results = []

    for _, row in pairs.iterrows():
        items = list(row["itemsets"])
        a, b = items[0], items[1]

        support = row["support"]
        p_a = product_probs.get(a, 0)
        p_b = product_probs.get(b, 0)

        if p_a > 0 and p_b > 0:
            lift = support / (p_a * p_b)

            if lift >= min_lift:
                conf_a_to_b = support / p_a
                conf_b_to_a = support / p_b
                leverage = support - (p_a * p_b)
                conv_a_to_b = (1 - p_b) / (1 - conf_a_to_b) if conf_a_to_b < 1 else 1e6
                conv_b_to_a = (1 - p_a) / (1 - conf_b_to_a) if conf_b_to_a < 1 else 1e6

                denom_j = p_a + p_b - support
                jaccard = support / denom_j if denom_j > 0 else 0.0
                kulczynski = 0.5 * (conf_a_to_b + conf_b_to_a)
                cosine = support / np.sqrt(p_a * p_b) if p_a > 0 and p_b > 0 else 0.0
                denom_phi = np.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))
                phi = (support - p_a * p_b) / denom_phi if denom_phi > 0 else 0.0

                results.append(
                    {
                        "product_a": a,
                        "product_b": b,
                        "support": support,
                        "confidence_a_to_b": conf_a_to_b,
                        "confidence_b_to_a": conf_b_to_a,
                        "lift": lift,
                        "leverage": leverage,
                        "conviction_a_to_b": conv_a_to_b,
                        "conviction_b_to_a": conv_b_to_a,
                        "jaccard": jaccard,
                        "kulczynski": kulczynski,
                        "cosine": cosine,
                        "phi_coefficient": phi,
                    }
                )

    if not results:
        return pd.DataFrame(columns=empty_cols)

    df = pd.DataFrame(results)
    df = df.sort_values(["lift", "kulczynski", "jaccard"], ascending=False).head(top_n)
    df = df.reset_index(drop=True)

    return df


def get_product_affinity_profile(
    transactions_df: pd.DataFrame,
    target_product: str,
    min_lift: float = 1.0,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Get affinity profile for a specific product - what other products co-purchase with it.
    """
    basket = create_basket_matrix(transactions_df)

    if target_product not in basket.columns:
        return pd.DataFrame()

    freq_items = run_fpgrowth(basket, min_support=0.001, max_len=2)

    if freq_items.empty:
        return pd.DataFrame()

    pairs = freq_items[freq_items["length"] == 2].copy()
    if pairs.empty:
        return pd.DataFrame()

    product_probs = basket.mean()
    p_target = product_probs.get(target_product, 0)

    if p_target == 0:
        return pd.DataFrame()

    results = []

    for _, row in pairs.iterrows():
        items = list(row["itemsets"])
        if target_product in items:
            other = items[0] if items[1] == target_product else items[1]

            support = row["support"]
            p_other = product_probs.get(other, 0)

            if p_other > 0:
                lift = support / (p_target * p_other)

                if lift >= min_lift:
                    conf_target_to_other = support / p_target
                    conf_other_to_target = support / p_other
                    denom_j = p_target + p_other - support
                    jaccard = support / denom_j if denom_j > 0 else 0.0
                    kulczynski = 0.5 * (conf_target_to_other + conf_other_to_target)

                    results.append(
                        {
                            "co_purchase_product": other,
                            "support": support,
                            "confidence_target_to_other": conf_target_to_other,
                            "confidence_other_to_target": conf_other_to_target,
                            "lift": lift,
                            "leverage": support - (p_target * p_other),
                            "jaccard": jaccard,
                            "kulczynski": kulczynski,
                        }
                    )

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values(["lift", "kulczynski"], ascending=False).head(top_n).reset_index(
        drop=True
    )

    return df
