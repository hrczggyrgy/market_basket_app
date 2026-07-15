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
