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

    Switching rate from A->B = customers who bought A then B / customers who bought A.

    Args:
        sequences: {customer_id: [product_id, ...]} from build_customer_sequences
        top_n_products: Limit to top-N products by purchase frequency

    Returns:
        DataFrame with columns: from_product, to_product, switch_count, switch_rate
    """
    switch_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    from_counts: Dict[str, int] = defaultdict(int)

    for _customer, products in sequences.items():
        for i in range(len(products) - 1):
            if products[i] != products[i + 1]:
                switch_counts[products[i]][products[i + 1]] += 1
                from_counts[products[i]] += 1

    if top_n_products:
        product_freq: Dict[str, int] = defaultdict(int)
        for prods in sequences.values():
            for p in prods:
                product_freq[p] += 1
        top_set = set(
            sorted(product_freq, key=product_freq.get, reverse=True)[:top_n_products]  # type: ignore[arg-type]
        )
        switch_counts = {
            k: {k2: v2 for k2, v2 in v.items() if k2 in top_set}
            for k, v in switch_counts.items()
            if k in top_set
        }
        from_counts = {k: v for k, v in from_counts.items() if k in top_set}

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
        return pd.DataFrame(
            columns=["from_product", "to_product", "switch_count", "switch_rate"]
        )

    df = pd.DataFrame(rows)
    return df.sort_values("switch_count", ascending=False).reset_index(drop=True)


def switching_matrix_to_heatmap(
    switching_df: pd.DataFrame,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Convert switching DataFrame to a square (from x to) pivot matrix for heatmap.

    Uses vectorised pivot_table instead of iterrows to ensure all cells are filled.
    top_products is a set for O(1) membership tests.

    Args:
        switching_df: Output from compute_switching_matrix
        top_n: Limit to top-N products by total switch activity

    Returns:
        Square DataFrame (from_product x to_product) with switch_rate values, or
        empty DataFrame if no data.
    """
    if switching_df.empty:
        return pd.DataFrame()

    # Rank products by total switch activity (sum of counts as source + destination)
    from_totals = switching_df.groupby("from_product")["switch_count"].sum()
    to_totals = switching_df.groupby("to_product")["switch_count"].sum()
    activity = from_totals.add(to_totals, fill_value=0).sort_values(ascending=False)
    top_products: set = set(activity.head(top_n).index.tolist())

    filtered = switching_df[
        switching_df["from_product"].isin(top_products)
        & switching_df["to_product"].isin(top_products)
    ]

    if filtered.empty:
        return pd.DataFrame()

    # Vectorised pivot — fills all cells, NaN -> 0
    matrix = filtered.pivot_table(
        index="from_product",
        columns="to_product",
        values="switch_rate",
        aggfunc="mean",
        fill_value=0.0,
    )

    # Make square: ensure all top products appear on both axes
    ordered = sorted(top_products & set(matrix.index) | top_products & set(matrix.columns))
    matrix = matrix.reindex(index=ordered, columns=ordered, fill_value=0.0)
    return matrix


def get_substitution_matrix(
    similarity_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get substitution scores directly from similarity matrix.

    In CDT, substitution score = similarity coefficient (Phi or Jaccard).
    High similarity == high substitutability.
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

    Bundle Score = log(1+lift) / log(1+max_lift)  *  (1 - substitution)

    Args:
        affinity_matrix: Lift-based co-purchase matrix
        substitution_matrix: Similarity/substitution matrix
        top_n_products: Limit to top-N products
        min_lift: Minimum lift to consider
        max_substitution: Maximum substitution to consider complementary

    Returns:
        DataFrame with product_a, product_b, lift, substitution, bundle_score
    """
    common_products = list(set(affinity_matrix.index) & set(substitution_matrix.index))

    if top_n_products:
        importance = affinity_matrix.loc[common_products].sum(axis=1)
        common_products = importance.nlargest(top_n_products).index.tolist()

    max_lift = float(affinity_matrix.values.max())
    log_max = np.log1p(max_lift) if max_lift > 0 else 1.0

    rows = []
    for i, prod_a in enumerate(common_products):
        for prod_b in common_products[i + 1:]:
            lift = float(affinity_matrix.loc[prod_a, prod_b])
            sub = float(substitution_matrix.loc[prod_a, prod_b])

            if lift >= min_lift and sub <= max_substitution:
                lift_score = np.log1p(lift) / log_max
                bundle_score = lift_score * (1.0 - sub)
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
    return df.sort_values("bundle_score", ascending=False).reset_index(drop=True)


def build_behavioral_matrices(
    transactions_df: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    affinity_matrix: pd.DataFrame,
    sequences: Dict[str, List[str]],
    top_n_products: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build all three behavioral matrices in one call.

    Args:
        transactions_df: Raw transaction DataFrame (unused here, kept for API compat)
        similarity_matrix: Product similarity matrix from cdt_similarity
        affinity_matrix: Lift-based co-purchase matrix from fpgrowth step
        sequences: {customer_id: [product_id, ...]} from build_customer_sequences
        top_n_products: Limit matrices to top-N products by frequency

    Returns:
        (switching_df, substitution_df, bundling_df)
    """
    # Bug-fix: pass sequences dict (not transactions_df) to compute_switching_matrix
    switching_df = compute_switching_matrix(sequences, top_n_products)
    substitution_df = get_substitution_matrix(similarity_matrix)
    bundling_df = compute_bundling_matrix(
        affinity_matrix, substitution_df, top_n_products
    )
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
    pairs = [
        {
            "product_a": prod_a,
            "product_b": prod_b,
            "substitution_score": substitution_matrix.loc[prod_a, prod_b],
        }
        for i, prod_a in enumerate(products)
        for prod_b in products[i + 1:]
        if substitution_matrix.loc[prod_a, prod_b] >= min_similarity
    ]

    if not pairs:
        return pd.DataFrame(columns=["product_a", "product_b", "substitution_score"])

    df = pd.DataFrame(pairs)
    return df.sort_values("substitution_score", ascending=False).head(top_n).reset_index(drop=True)


def get_top_bundling_pairs(
    bundling_df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return top bundling pairs (already sorted by bundle_score)."""
    if bundling_df.empty:
        return bundling_df
    return bundling_df.head(top_n).reset_index(drop=True)


def get_top_switching_paths(
    switching_df: pd.DataFrame,
    top_n: int = 20,
    min_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Get top switching paths by switch_count.

    Args:
        switching_df: Output from compute_switching_matrix
        top_n: Number of paths to return
        min_rate: Minimum switch_rate to include

    Returns:
        Top-N paths sorted by switch_count descending
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
    Compute switching behaviour profile per customer.

    Useful for segmentation: loyalists vs switchers.

    Returns:
        DataFrame with customer_id, total_purchases, unique_products,
        switch_count, switch_rate, top_from_product, top_to_product
    """
    profiles = []

    for customer, products in sequences.items():
        if len(products) < 2:
            continue

        switches = [
            (products[i], products[i + 1])
            for i in range(len(products) - 1)
            if products[i] != products[i + 1]
        ]

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

        switch_freq: Dict[Tuple[str, str], int] = defaultdict(int)
        for pair in switches:
            switch_freq[pair] += 1
        top_switch = max(switch_freq, key=switch_freq.get)  # type: ignore[arg-type]

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
        product_freq_: Dict[str, int] = defaultdict(int)
        for prods in sequences.values():
            for p in prods:
                product_freq_[p] += 1
        top_set = set(
            sorted(product_freq_, key=product_freq_.get, reverse=True)[:top_n_products]  # type: ignore[arg-type]
        )
        df = df[
            df["top_from_product"].isin(top_set) | df["top_to_product"].isin(top_set)
        ]

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

    from_totals = (
        matrix.groupby(["category", "from_brand"])["switch_count"]
        .sum()
        .reset_index()
        .rename(columns={"switch_count": "total_switches_from"})
    )
    matrix = matrix.merge(from_totals, on=["category", "from_brand"])
    matrix["switch_rate"] = matrix["switch_count"] / matrix["total_switches_from"]

    return matrix.sort_values(
        ["category", "switch_count"], ascending=[True, False]
    ).reset_index(drop=True)
