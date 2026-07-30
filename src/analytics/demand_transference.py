"""Demand Transference (DT) — Oracle Retail CDT companion science.

Estimates the fraction of revenue that transfers from a removed/delisted
product to its substitutes, using switching rates weighted by historical
revenue. Enables assortment optimization directly from transaction data.

References
----------
- Oracle Retail Modeling Engine 14.0 Release Notes
- Oracle Retail AI Foundation Cloud Service Implementation Guide 23.2
- Oracle Retail Science Cloud Services 19.1 Implementation Guide
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_demand_transference_matrix(
    transactions_df: pd.DataFrame,
    switching_df: pd.DataFrame,
    product_col: str = "stockcode",
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """Build revenue-weighted demand transference matrix.

    For each ordered product pair (A -> B):
        DT(A->B) = switch_rate(A->B) * revenue_share(A)

    where revenue_share(A) = revenue(A) / total_category_revenue.

    Parameters
    ----------
    transactions_df : Raw transaction DataFrame.
    switching_df    : Output of compute_switching_matrix() with columns
                      from_product, to_product, switch_rate.
    product_col     : Product identifier column.
    top_n           : Limit to top-N products by revenue.

    Returns
    -------
    DataFrame with columns:
        from_product, to_product, switch_rate, revenue_share_from,
        demand_transference, revenue_at_risk
    """
    df = transactions_df.copy()
    df["revenue"] = df["price"] * df["quantity"]

    product_revenue = df.groupby(product_col)["revenue"].sum()
    total_revenue = product_revenue.sum()
    revenue_share = (product_revenue / total_revenue).rename("revenue_share")

    if top_n:
        top_products = product_revenue.nlargest(top_n).index
        switching_df = switching_df[
            switching_df["from_product"].isin(top_products)
            & switching_df["to_product"].isin(top_products)
        ]

    result = switching_df.copy()
    result["revenue_share_from"] = result["from_product"].map(revenue_share).fillna(0)
    result["demand_transference"] = result["switch_rate"] * result["revenue_share_from"]

    # Revenue at risk: absolute revenue that transfers if product is delisted
    result["revenue_from"] = result["from_product"].map(product_revenue).fillna(0)
    result["revenue_at_risk"] = result["switch_rate"] * result["revenue_from"]

    return (
        result[
            ["from_product", "to_product", "switch_rate",
             "revenue_share_from", "demand_transference", "revenue_at_risk"]
        ]
        .sort_values("demand_transference", ascending=False)
        .reset_index(drop=True)
    )


def compute_substitutable_demand_percentage(
    demand_transference_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
) -> Dict[str, float]:
    """Compute Substitutable Demand Percentage (SDP) per product.

    SDP(A) = total revenue transferable away from A to all substitutes
              / total category revenue

    Oracle uses SDP as a key assortment optimization input:
    - SDP > 0.8 : product is highly substitutable — potential delist candidate
    - SDP < 0.2 : unique demand driver — must-stock

    Returns dict: {product_id: sdp_score}
    """
    df = transactions_df.copy()
    df["revenue"] = df["price"] * df["quantity"]
    total_revenue = (df["price"] * df["quantity"]).sum()

    if total_revenue == 0:
        return {}

    sdp = (
        demand_transference_df
        .groupby("from_product")["revenue_at_risk"]
        .sum()
        / total_revenue
    )
    return sdp.to_dict()


def delist_impact_analysis(
    transactions_df: pd.DataFrame,
    demand_transference_df: pd.DataFrame,
    products_to_delist: List[str],
    product_col: str = "stockcode",
) -> pd.DataFrame:
    """Estimate revenue impact of delisting a set of products.

    For each delisted product:
    - revenue_lost    : its own historical revenue
    - revenue_recovered: revenue estimated to transfer to substitutes
                         (sum of revenue_at_risk for that product's DT rows)
    - net_revenue_impact: revenue_recovered - revenue_lost  (negative = net loss)
    - recovery_rate   : revenue_recovered / revenue_lost

    Parameters
    ----------
    products_to_delist : List of product IDs to simulate delisting.

    Returns
    -------
    DataFrame with one row per delisted product.
    """
    df = transactions_df.copy()
    df["revenue"] = df["price"] * df["quantity"]
    product_revenue = df.groupby(product_col)["revenue"].sum()

    rows = []
    for prod in products_to_delist:
        rev = product_revenue.get(prod, 0.0)
        transferred = demand_transference_df[
            demand_transference_df["from_product"] == prod
        ]["revenue_at_risk"].sum()

        rows.append({
            product_col: prod,
            "product_revenue": rev,
            "estimated_revenue_recovered": transferred,
            "net_revenue_impact": transferred - rev,
            "recovery_rate": transferred / rev if rev > 0 else np.nan,
        })

    return pd.DataFrame(rows).sort_values("net_revenue_impact").reset_index(drop=True)


def node_delist_impact(
    transactions_df: pd.DataFrame,
    demand_transference_df: pd.DataFrame,
    cluster_assignments: Dict[str, int],
    product_col: str = "stockcode",
) -> pd.DataFrame:
    """Per-CDT-node delist impact summary.

    For each cluster node, computes:
    - total_revenue       : sum of all product revenues in node
    - internal_recovery   : DT within the node (intra-node transfers)
    - external_leakage    : DT flowing out of the node to other nodes
    - node_sdp            : internal_recovery / total_revenue
    """
    df = transactions_df.copy()
    df["revenue"] = df["price"] * df["quantity"]
    product_revenue = df.groupby(product_col)["revenue"].sum()

    product_to_node = pd.Series(cluster_assignments)
    nodes = sorted(set(cluster_assignments.values()))

    rows = []
    for node_id in nodes:
        node_products = [p for p, n in cluster_assignments.items() if n == node_id]
        node_rev = sum(product_revenue.get(p, 0) for p in node_products)

        node_dt = demand_transference_df[
            demand_transference_df["from_product"].isin(node_products)
        ]
        internal = node_dt[
            node_dt["to_product"].isin(node_products)
        ]["revenue_at_risk"].sum()
        external = node_dt[
            ~node_dt["to_product"].isin(node_products)
        ]["revenue_at_risk"].sum()

        rows.append({
            "node_id": node_id,
            "n_products": len(node_products),
            "total_node_revenue": round(node_rev, 2),
            "internal_recovery": round(internal, 2),
            "external_leakage": round(external, 2),
            "node_sdp": round(internal / node_rev, 4) if node_rev > 0 else np.nan,
        })

    return pd.DataFrame(rows).sort_values("total_node_revenue", ascending=False).reset_index(drop=True)


# ============================================================================
# ADVANCED: Markov Chain & MNL Simulation
# ============================================================================

def build_substitution_matrix_markov(
    switching_df: pd.DataFrame,
    max_iterations: int = 10,
    convergence_threshold: float = 1e-6,
) -> pd.DataFrame:
    """
    Build substitution matrix via multi-step Markov chain on switching rates.

    P^(k) = P^k where P is the one-step switching matrix.
    Steady-state substitution probability = lim_{k->inf} P^k.

    This captures indirect substitution chains (A->B->C) that simple
    one-step switching misses.

    Parameters
    ----------
    switching_df : Output from compute_switching_matrix()
    max_iterations : Maximum power iterations
    convergence_threshold : Stop when max change < threshold

    Returns
    -------
    Square DataFrame (products x products) with steady-state substitution probs.
    """
    # Build transition matrix
    products = sorted(set(switching_df["from_product"].unique()) | set(switching_df["to_product"].unique()))
    n = len(products)
    product_idx = {p: i for i, p in enumerate(products)}

    P = np.zeros((n, n))
    for _, row in switching_df.iterrows():
        i = product_idx[row["from_product"]]
        j = product_idx[row["to_product"]]
        P[i, j] = row["switch_rate"]

    # Normalize rows to sum to 1 (stochastic matrix)
    row_sums = P.sum(axis=1)
    P[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

    # Power iteration for steady state
    P_k = P.copy()
    for _ in range(max_iterations):
        P_next = P_k @ P
        diff = np.abs(P_next - P_k).max()
        P_k = P_next
        if diff < convergence_threshold:
            break

    return pd.DataFrame(P_k, index=products, columns=products)


def build_substitution_matrix_mnl(
    transactions_df: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    price_col: str = "price",
    utility_weight_price: float = -0.5,
    utility_weight_similarity: float = 1.0,
    utility_weight_revenue_share: float = 0.5,
) -> pd.DataFrame:
    """
    Multinomial Logit (MNL) substitution model.

    Utility of product j when i is removed:
        U_j = w_price * price_j + w_sim * sim(i,j) + w_rev * log(revenue_j)

    P(j | i_removed) = exp(U_j) / sum_k exp(U_k)

    Parameters
    ----------
    transactions_df : Transaction data for revenue and price stats.
    similarity_matrix : Product similarity (Phi/Jaccard/PMI) matrix.
    utility_weight_price : Weight for price (negative = prefer cheaper substitutes).
    utility_weight_similarity : Weight for similarity to removed product.
    utility_weight_revenue_share : Weight for product popularity.

    Returns
    -------
    Square DataFrame with MNL substitution probabilities.
    """
    df = transactions_df.copy()
    df["revenue"] = df["price"] * df["quantity"]

    product_revenue = df.groupby("stockcode")["revenue"].sum()
    product_price = df.groupby("stockcode")["price"].median()

    products = similarity_matrix.index.tolist()
    n = len(products)

    # Utility matrix
    U = np.zeros((n, n))
    for i, prod_i in enumerate(products):
        for j, prod_j in enumerate(products):
            if i == j:
                U[i, j] = -np.inf  # Can't substitute with itself
                continue

            price_j = product_price.get(prod_j, product_price.mean())
            sim_ij = similarity_matrix.loc[prod_i, prod_j]
            rev_j = product_revenue.get(prod_j, product_revenue.mean())

            U[i, j] = (
                utility_weight_price * price_j +
                utility_weight_similarity * sim_ij +
                utility_weight_revenue_share * np.log(rev_j + 1)
            )

    # Softmax per row
    exp_U = np.exp(U - U.max(axis=1, keepdims=True))
    P = exp_U / exp_U.sum(axis=1, keepdims=True)
    P[~np.isfinite(P)] = 0

    return pd.DataFrame(P, index=products, columns=products)


def simulate_assortment_change(
    transactions_df: pd.DataFrame,
    delist_products: List[str],
    substitution_matrix: pd.DataFrame,
    demand_transference_matrix: pd.DataFrame,
    revenue_per_product: pd.Series,
    cdt_tree: Optional[object] = None,
    constraint_max_recovery: float = 1.0,
) -> Dict[str, float]:
    """
    Simulate removing SKUs and compute full demand reallocation.

    Uses either:
    - Markov steady-state substitution matrix (multi-step chains)
    - MNL choice model (price/similarity utilities)

    Returns detailed per-SKU and aggregate metrics.
    """
    all_products = revenue_per_product.index.tolist()
    kept_products = [p for p in all_products if p not in delist_products]

    # Volume at risk
    lost_volume = revenue_per_product[delist_products].sum()

    # Recovery via substitution matrix
    recovered = 0.0
    recovery_detail = {}

    for delisted in delist_products:
        if delisted not in substitution_matrix.index:
            continue
        # For each kept product, recovery = substitution_prob * lost_volume
        for kept in kept_products:
            if kept not in substitution_matrix.columns:
                continue
            prob = substitution_matrix.loc[delisted, kept]
            amt = prob * revenue_per_product.get(delisted, 0)
            recovered += amt
            recovery_detail.setdefault(delisted, {})[kept] = amt

    # Apply CDT constraints if tree provided
    if cdt_tree is not None:
        recovered = _apply_cdt_recovery_constraints(
            recovered, recovery_detail, cdt_tree, kept_products, constraint_max_recovery
        )

    return {
        "lost_revenue": lost_volume,
        "recovered_revenue": recovered,
        "net_impact": recovered - lost_volume,
        "recovery_rate": recovered / lost_volume if lost_volume > 0 else 0,
        "recovery_detail": recovery_detail,
    }


def _apply_cdt_recovery_constraints(
    recovered: float,
    recovery_detail: Dict,
    cdt_tree,
    kept_products: List[str],
    max_recovery: float,
) -> float:
    """Apply CDT-based constraints on demand recovery.

    - Can only recover to products in same CDT leaf (high substitutability)
    - Maximum recovery capped by max_recovery fraction
    - External leakage penalty for cross-node recovery
    """
    # Simplified: cap total recovery
    return min(recovered, max_recovery * sum(recovery_detail.get(d, {}).values() for d in recovery_detail))
