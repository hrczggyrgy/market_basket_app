"""CDT: Behavioral Metrics Engine.

Computes switching, substitution, and bundling matrices from transaction data.
These are the three core behavioral patterns CDT identifies.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_switching_matrix(
    sequences: Dict[str, List[str]],
    top_n_products: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute product-to-product switching rates from customer sequences.

    Switching rate from A to B = (customers who bought A then B) / (customers who bought A)

    Args:
        sequences: Customer sequences from build_customer_sequences
        top_n_products: Limit to top N products by frequency

    Returns:
        DataFrame with from_product, to_product, switch_count, switch_rate
    """
    # Count transitions
    switch_counts = defaultdict(lambda: defaultdict(int))
    from_counts = defaultdict(int)

    for customer, products in sequences.items():
        for i in range(len(products) - 1):
            if products[i] != products[i + 1]:
                switch_counts[products[i]][products[i + 1]] += 1
                from_counts[products[i]] += 1

    if top_n_products:
        # Filter to top products
        all_products = set()
        for d in switch_counts.values():
            all_products.update(d.keys())
        all_products.update(switch_counts.keys())

        product_freq = defaultdict(int)
        for products in sequences.values():
            for p in products:
                product_freq[p] += 1
        top_products = set(
            sorted(all_products, key=lambda x: product_freq.get(x, 0), reverse=True)[
                :top_n_products
            ]
        )

        switch_counts = {
            k: {k2: v2 for k2, v2 in v.items() if k2 in top_products}
            for k, v in switch_counts.items()
            if k in top_products
        }
        from_counts = {k: v for k, v in from_counts.items() if k in top_products}

    rows = []
    for from_prod, targets in switch_counts.items():
        total = from_counts.get(from_prod, 0)
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


def switching_matrix_to_heatmap(
    switching_df: pd.DataFrame,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Convert switching DataFrame to square matrix for heatmap.

    Args:
        switching_df: Output from compute_switching_matrix
        top_n: Limit to top N products by total switches

    Returns:
        Square matrix (from x to) with switch_rate values
    """
    if switching_df.empty:
        return pd.DataFrame()

    product_activity = defaultdict(int)

    for _, row in switching_df.iterrows():
        product_activity[row["from_product"]] += row["switch_count"]
        product_activity[row["to_product"]] += row["switch_count"]

    top_products = sorted(product_activity.keys(), key=lambda x: product_activity[x], reverse=True)[
        :top_n
    ]

    matrix = pd.DataFrame(0.0, index=top_products, columns=top_products, dtype=float)

    for _, row in switching_df.iterrows():
        if row["from_product"] in top_products and row["to_product"] in top_products:
            matrix.loc[row["from_product"], row["to_product"]] = row["switch_rate"]

    return matrix


def get_substitution_matrix(
    similarity_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get substitution scores directly from similarity matrix.

    In CDT, substitution score = similarity coefficient (Phi).
    High similarity = high substitutability.

    Args:
        similarity_matrix: Output from build_similarity_matrix

    Returns:
        Square matrix of substitution scores (same as similarity matrix)
    """
    return similarity_matrix.copy()


def compute_bundling_matrix(
    affinity_matrix: pd.DataFrame,
    substitution_matrix: pd.DataFrame,
    top_n_products: Optional[int] = None,
    min_lift: float = 1.0,
    max_substitution: float = 0.3,
) -> pd.DataFrame:
    """
    Compute bundling scores: high lift (co-purchase) + low substitution = true complements.

    Bundle Score = Lift * (1 - Substitution)  [normalized to 0-1]

    Args:
        affinity_matrix: Lift-based co-purchase matrix from copurchase.py
        substitution_matrix: Similarity/substitution matrix
        top_n_products: Limit to top N products
        min_lift: Minimum lift to consider
        max_substitution: Maximum substitution to consider complementary

    Returns:
        DataFrame with product_a, product_b, lift, substitution, bundle_score
    """
    # Ensure matrices aligned
    common_products = list(set(affinity_matrix.index) & set(substitution_matrix.index))

    if top_n_products:
        # Use affinity sum as importance
        importance = affinity_matrix.loc[common_products].sum(axis=1)
        common_products = importance.nlargest(top_n_products).index.tolist()

    rows = []
    for i, prod_a in enumerate(common_products):
        for prod_b in common_products[i + 1 :]:
            lift = affinity_matrix.loc[prod_a, prod_b]
            sub = substitution_matrix.loc[prod_a, prod_b]

            if lift >= min_lift and sub <= max_substitution:
                # Normalized bundle score: high lift, low substitution
                # Scale lift (typically 1-10+) to [0,1] via log
                lift_score = np.log1p(lift) / np.log1p(affinity_matrix.values.max())
                sub_penalty = 1 - sub  # sub in [-1,1] for Phi, so 1-sub in [0,2]
                bundle_score = lift_score * sub_penalty

                rows.append(
                    {
                        "product_a": prod_a,
                        "product_b": prod_b,
                        "lift": lift,
                        "substitution": sub,
                        "bundle_score": bundle_score,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=["product_a", "product_b", "lift", "substitution", "bundle_score"]
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("bundle_score", ascending=False).reset_index(drop=True)
    return df


def build_behavioral_matrices(
    transactions_df: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    affinity_matrix: pd.DataFrame,
    sequences: Dict[str, List[str]],
    top_n_products: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build all three behavioral matrices in one call.

    Returns:
        (switching_df, substitution_df, bundling_df)
    """
    switching_df = compute_switching_matrix(sequences, top_n_products)
    substitution_df = get_substitution_matrix(similarity_matrix)
    bundling_df = compute_bundling_matrix(affinity_matrix, substitution_df, top_n_products)

    return switching_df, substitution_df, bundling_df


def get_top_substitution_pairs(
    substitution_matrix: pd.DataFrame,
    top_n: int = 20,
    min_similarity: float = 0.0,
) -> pd.DataFrame:
    """
    Get top substitutable product pairs from substitution matrix.

    Args:
        substitution_matrix: Square similarity matrix
        top_n: Number of pairs to return
        min_similarity: Minimum similarity threshold

    Returns:
        DataFrame with product_a, product_b, substitution_score
    """
    products = substitution_matrix.index.tolist()
    pairs = []

    for i, prod_a in enumerate(products):
        for prod_b in products[i + 1 :]:
            score = substitution_matrix.loc[prod_a, prod_b]
            if score >= min_similarity:
                pairs.append(
                    {
                        "product_a": prod_a,
                        "product_b": prod_b,
                        "substitution_score": score,
                    }
                )

    if not pairs:
        return pd.DataFrame(columns=["product_a", "product_b", "substitution_score"])

    df = pd.DataFrame(pairs)
    df = df.sort_values("substitution_score", ascending=False).head(top_n).reset_index(drop=True)
    return df


def get_top_bundling_pairs(
    bundling_df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Get top bundling pairs from bundling DataFrame.

    Args:
        bundling_df: Output from compute_bundling_matrix
        top_n: Number of pairs to return

    Returns:
        Top N bundling pairs
    """
    if bundling_df.empty:
        return bundling_df
    return bundling_df.head(top_n).reset_index(drop=True)


def get_top_switching_paths(
    switching_df: pd.DataFrame,
    top_n: int = 20,
    min_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Get top switching paths.

    Args:
        switching_df: Output from compute_switching_matrix
        top_n: Number of paths to return
        min_rate: Minimum switch rate threshold

    Returns:
        Top N switching paths
    """
    if switching_df.empty:
        return switching_df

    df = switching_df[switching_df["switch_rate"] >= min_rate]
    return df.head(top_n).reset_index(drop=True)


def compute_customer_switching_profiles(
    sequences: Dict[str, List[str]],
    top_n_products: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute switching behavior profile per customer.

    Useful for segmentation: loyalists vs switchers.

    Returns:
        DataFrame with customer_id, total_purchases, unique_products,
        switch_count, switch_rate, top_from_product, top_to_product
    """
    profiles = []

    for customer, products in sequences.items():
        if len(products) < 2:
            continue

        switches = []
        for i in range(len(products) - 1):
            if products[i] != products[i + 1]:
                switches.append((products[i], products[i + 1]))

        if not switches:
            profiles.append(
                {
                    "customer_id": customer,
                    "total_purchases": len(products),
                    "unique_products": len(set(products)),
                    "switch_count": 0,
                    "switch_rate": 0.0,
                    "top_from_product": None,
                    "top_to_product": None,
                }
            )
            continue

        switch_counts = defaultdict(int)
        for from_p, to_p in switches:
            switch_counts[(from_p, to_p)] += 1

        top_switch = max(switch_counts.items(), key=lambda x: x[1])[0]

        profiles.append(
            {
                "customer_id": customer,
                "total_purchases": len(products),
                "unique_products": len(set(products)),
                "switch_count": len(switches),
                "switch_rate": len(switches) / (len(products) - 1),
                "top_from_product": top_switch[0],
                "top_to_product": top_switch[1],
            }
        )

    df = pd.DataFrame(profiles)

    if top_n_products and not df.empty:
        # Filter customers who bought top products
        product_freq = defaultdict(int)
        for products in sequences.values():
            for p in products:
                product_freq[p] += 1
        top_products = set(
            sorted(product_freq, key=product_freq.get, reverse=True)[:top_n_products]
        )

        df = df[df["top_from_product"].isin(top_products) | df["top_to_product"].isin(top_products)]

    return df


def detect_brand_switching(
    transactions_df: pd.DataFrame,
    brand_col: str = "brand",
    category_col: str = "category",
    customer_col: str = "customer_id",
    date_col: str = "date",
    window_days: int = 90,
) -> pd.DataFrame:
    """
    Detect brand switching within the same category.

    Returns brand-to-brand switching events with customer and timing.
    """
    if brand_col not in transactions_df.columns or category_col not in transactions_df.columns:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "category",
                "from_brand",
                "to_brand",
                "from_date",
                "to_date",
                "days_between",
            ]
        )

    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([customer_col, date_col])

    switches = []

    for customer, group in df.groupby(customer_col):
        group = group.sort_values(date_col)

        for i in range(len(group) - 1):
            curr = group.iloc[i]
            nxt = group.iloc[i + 1]

            days_diff = (nxt[date_col] - curr[date_col]).days

            if (
                days_diff <= window_days
                and curr[category_col] == nxt[category_col]
                and curr[brand_col] != nxt[brand_col]
            ):
                switches.append(
                    {
                        "customer_id": customer,
                        "category": curr[category_col],
                        "from_brand": curr[brand_col],
                        "to_brand": nxt[brand_col],
                        "from_date": curr[date_col],
                        "to_date": nxt[date_col],
                        "days_between": days_diff,
                    }
                )

    return pd.DataFrame(switches)


def compute_brand_switching_matrix(
    brand_switches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate brand switching events to brand-to-brand matrix per category.
    """
    if brand_switches.empty:
        return pd.DataFrame()

    matrix = (
        brand_switches.groupby(["category", "from_brand", "to_brand"])
        .agg(
            switch_count=("customer_id", "count"),
            unique_customers=("customer_id", "nunique"),
            avg_days_between=("days_between", "mean"),
        )
        .reset_index()
    )

    # Add rate per category
    from_totals = matrix.groupby(["category", "from_brand"])["switch_count"].sum().reset_index()
    from_totals.columns = ["category", "from_brand", "total_switches_from"]

    matrix = matrix.merge(from_totals, on=["category", "from_brand"])
    matrix["switch_rate"] = matrix["switch_count"] / matrix["total_switches_from"]

    return matrix.sort_values(["category", "switch_count"], ascending=[True, False]).reset_index(
        drop=True
    )
