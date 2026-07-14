"""Add-on / Complementary product analysis."""

from typing import List, Optional

import numpy as np
import pandas as pd

from src.algorithms.fpgrowth import create_basket_matrix


def get_addon_recommendations(
    transactions_df: pd.DataFrame,
    anchor_product: str,
    min_support: float = 0.001,
    min_lift: float = 1.2,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Get add-on product recommendations for a given anchor product.

    Args:
        transactions_df: Transaction data
        anchor_product: The anchor product (stockcode)
        min_support: Minimum support for co-purchase
        min_lift: Minimum lift threshold
        top_n: Number of recommendations to return

    Returns:
        DataFrame with add-on products ranked by lift/confidence
    """
    basket = create_basket_matrix(transactions_df)

    if anchor_product not in basket.columns:
        return pd.DataFrame()

    # Get transactions containing anchor
    anchor_transactions = basket[basket[anchor_product] == 1]

    if len(anchor_transactions) == 0:
        return pd.DataFrame()

    # Find other products in those transactions
    other_products = anchor_transactions.drop(columns=[anchor_product])

    # Calculate conditional probabilities
    results = []
    p_anchor = basket[anchor_product].mean()

    for product in other_products.columns:
        if product == anchor_product:
            continue

        # P(Add-on | Anchor)
        p_both = (anchor_transactions[product] == 1).sum() / len(basket)
        p_addon = basket[product].mean()

        if p_anchor > 0 and p_addon > 0:
            confidence = p_both / p_anchor  # P(Addon | Anchor)
            lift = p_both / (p_anchor * p_addon)
            leverage = p_both - (p_anchor * p_addon)
            conviction = (1 - p_addon) / (1 - confidence) if confidence < 1 else np.inf

            if lift >= min_lift and p_both >= min_support:
                # Revenue uplift estimate
                anchor_price = transactions_df[
                    transactions_df["stockcode"] == anchor_product
                ]["price"].median()
                addon_price = transactions_df[transactions_df["stockcode"] == product][
                    "price"
                ].median()
                revenue_uplift = (
                    addon_price * confidence
                )  # Expected additional revenue per anchor sale

                results.append(
                    {
                        "anchor_product": anchor_product,
                        "addon_product": product,
                        "p_addon_given_anchor": confidence,
                        "p_addon_baseline": p_addon,
                        "lift": lift,
                        "leverage": leverage,
                        "conviction": conviction,
                        "support": p_both,
                        "revenue_uplift_per_anchor": revenue_uplift,
                        "anchor_price": anchor_price,
                        "addon_price": addon_price,
                    }
                )

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = (
        df.sort_values(["lift", "p_addon_given_anchor"], ascending=[False, False])
        .head(top_n)
        .reset_index(drop=True)
    )

    return df


def get_anchor_addon_matrix(
    transactions_df: pd.DataFrame,
    anchor_products: Optional[List[str]] = None,
    min_lift: float = 1.2,
    top_n_per_anchor: int = 5,
) -> pd.DataFrame:
    """
    Get add-on recommendations for multiple anchor products.

    Returns:
        DataFrame with anchor, addon, lift, confidence for each pair
    """
    if anchor_products is None:
        # Use top 20 products by frequency
        anchor_products = (
            transactions_df["stockcode"].value_counts().head(20).index.tolist()
        )

    all_results = []

    for anchor in anchor_products:
        recs = get_addon_recommendations(
            transactions_df, anchor, min_lift=min_lift, top_n=top_n_per_anchor
        )
        all_results.append(recs)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def get_addon_by_category(
    transactions_df: pd.DataFrame,
    anchor_product: str,
    category_col: str = "category",
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Get add-on recommendations grouped by category (if category column exists).
    """
    if category_col not in transactions_df.columns:
        return pd.DataFrame()

    recs = get_addon_recommendations(transactions_df, anchor_product, top_n=top_n * 3)

    if recs.empty:
        return recs

    # Get category for each addon
    product_categories = (
        transactions_df.drop_duplicates("stockcode")
        .set_index("stockcode")[category_col]
        .to_dict()
    )

    recs["addon_category"] = recs["addon_product"].map(product_categories)

    # Group by category and get top per category
    grouped = recs.groupby("addon_category").head(top_n).reset_index(drop=True)

    return grouped
