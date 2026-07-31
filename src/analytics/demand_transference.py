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
import statsmodels.api as sm


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
            [
                "from_product",
                "to_product",
                "switch_rate",
                "revenue_share_from",
                "demand_transference",
                "revenue_at_risk",
            ]
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

    sdp = demand_transference_df.groupby("from_product")["revenue_at_risk"].sum() / total_revenue
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
        transferred = demand_transference_df[demand_transference_df["from_product"] == prod][
            "revenue_at_risk"
        ].sum()

        rows.append(
            {
                product_col: prod,
                "product_revenue": rev,
                "estimated_revenue_recovered": transferred,
                "net_revenue_impact": transferred - rev,
                "recovery_rate": transferred / rev if rev > 0 else np.nan,
            }
        )

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
    nodes = sorted(set(cluster_assignments.values()))

    rows = []
    for node_id in nodes:
        node_products = [p for p, n in cluster_assignments.items() if n == node_id]
        node_rev = sum(product_revenue.get(p, 0) for p in node_products)

        node_dt = demand_transference_df[demand_transference_df["from_product"].isin(node_products)]
        internal = node_dt[node_dt["to_product"].isin(node_products)]["revenue_at_risk"].sum()
        external = node_dt[~node_dt["to_product"].isin(node_products)]["revenue_at_risk"].sum()

        rows.append(
            {
                "node_id": node_id,
                "n_products": len(node_products),
                "total_node_revenue": round(node_rev, 2),
                "internal_recovery": round(internal, 2),
                "external_leakage": round(external, 2),
                "node_sdp": round(internal / node_rev, 4) if node_rev > 0 else np.nan,
            }
        )

    return (
        pd.DataFrame(rows).sort_values("total_node_revenue", ascending=False).reset_index(drop=True)
    )


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
    products = sorted(
        set(switching_df["from_product"].unique()) | set(switching_df["to_product"].unique())
    )
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
                utility_weight_price * price_j
                + utility_weight_similarity * sim_ij
                + utility_weight_revenue_share * np.log(rev_j + 1)
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
    return min(
        recovered, max_recovery * sum(recovery_detail.get(d, {}).values() for d in recovery_detail)
    )


def compute_category_leakage_rate(
    switching_df: pd.DataFrame,
    cluster_assignments: Dict[str, int],
    min_cooccurrence: int = 5,
) -> Dict[str, float]:
    """
    Compute category-level leakage rate: fraction of demand from a product
    that leaks OUT of the category entirely (no within-category substitute chosen).

    Uses switching rates instead of DT revenue_at_risk.

    Args:
        switching_df: Output from compute_switching_matrix with columns
                      from_product, to_product, switch_rate
        cluster_assignments: Dict mapping product -> category
        min_cooccurrence: Minimum co-occurrence threshold (ignored, kept for API compatibility)

    Returns:
        Dict: {category: leakage_rate}
    """
    dt = switching_df.copy()
    dt["from_category"] = dt["from_product"].map(dict(cluster_assignments.items()))
    dt["to_category"] = dt["to_product"].map(dict(cluster_assignments.items()))

    # Leakage = switching rate flowing to different category
    leakage = dt[dt["from_category"] != dt["to_category"]]
    total_switch = dt.groupby("from_category")["switch_rate"].sum()
    leakage_sum = leakage.groupby("from_category")["switch_rate"].sum()

    leakage_rate = (leakage_sum / total_switch).fillna(0).to_dict()
    return leakage_rate


def compute_cross_price_elasticity(
    transactions_df: pd.DataFrame,
    product_pairs: List[Tuple[str, str]],
    price_col: str = "price",
    qty_col: str = "quantity",
    date_col: str = "date",
    freq: str = "W",
    min_periods: int = 10,
    min_price_variation: float = 0.05,
) -> pd.DataFrame:
    """
    Estimate cross-price elasticities for specified product pairs.

    For each pair (A, B), runs bivariate log-log OLS:
        log(qty_A) = alpha + beta_own * log(price_A) + beta_cross * log(price_B) + error

    beta_cross > 0 -> B is a substitute for A
    beta_cross < 0 -> B is a complement to A

    Args:
        transactions_df: Transaction data
        product_pairs: List of (product_a, product_b) tuples to estimate cross-elasticity
        freq: Time frequency for aggregation
        min_periods: Minimum weeks of data
        min_price_variation: Minimum price CV

    Returns:
        DataFrame with cross-elasticity estimates per pair
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    results = []

    for prod_a, prod_b in product_pairs:
        # Get weekly data for both products
        prod_a_df = df[df["stockcode"] == prod_a].copy()
        prod_b_df = df[df["stockcode"] == prod_b].copy()

        if len(prod_a_df) < min_periods or len(prod_b_df) < min_periods:
            continue

        weekly_a = (
            prod_a_df.set_index("date")
            .groupby(pd.Grouper(freq=freq))
            .agg(avg_price_a=("price", "mean"), total_qty_a=("quantity", "sum"))
            .dropna()
        )
        weekly_b = (
            prod_b_df.set_index("date")
            .groupby(pd.Grouper(freq=freq))
            .agg(avg_price_b=("price", "mean"), total_qty_b=("quantity", "sum"))
            .dropna()
        )

        # Align on date index
        weekly = weekly_a.join(weekly_b, how="inner")
        if len(weekly) < min_periods:
            continue

        # Check price variation for both
        cv_a = weekly["avg_price_a"].std() / weekly["avg_price_a"].mean()
        cv_b = weekly["avg_price_b"].std() / weekly["avg_price_b"].mean()
        if cv_a < min_price_variation or cv_b < min_price_variation:
            continue

        # Log-log regression: log(qty_a) ~ log(price_a) + log(price_b)
        log_price_a = np.log(weekly["avg_price_a"].replace(0, np.nan).dropna())
        log_price_b = np.log(
            weekly.loc[log_price_a.index, "avg_price_b"].replace(0, np.nan).dropna()
        )
        log_qty_a = np.log(weekly.loc[log_price_a.index, "total_qty_a"].replace(0, np.nan).dropna())

        if len(log_price_a) < min_periods:
            continue

        # Bivariate OLS
        X = np.column_stack([log_price_a.values, log_price_b.values])
        X = sm.add_constant(X)
        y = log_qty_a.values

        try:
            model = sm.OLS(y, X).fit(cov_type="HC3")
            own_elasticity = model.params[1]
            cross_elasticity = model.params[2]
            own_se = model.bse[1]
            cross_se = model.bse[2]
            own_p = model.pvalues[1]
            cross_p = model.pvalues[2]
            r2 = model.rsquared
        except Exception:
            continue

        results.append(
            {
                "product_a": prod_a,
                "product_b": prod_b,
                "own_elasticity": own_elasticity,
                "own_elasticity_se": own_se,
                "own_elasticity_p": own_p,
                "cross_elasticity": cross_elasticity,
                "cross_elasticity_se": cross_se,
                "cross_elasticity_p": cross_p,
                "r_squared": r2,
                "n_obs": len(log_price_a),
                "avg_price_a": weekly["avg_price_a"].mean(),
                "avg_price_b": weekly["avg_price_b"].mean(),
            }
        )

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def compute_recovery_hhi(
    demand_transference_df: pd.DataFrame,
    cluster_assignments: Dict[str, int],
) -> pd.DataFrame:
    """
    Compute Herfindahl-Hirschman Index (HHI) on demand recovery for each delisted product.

    HHI = sum(s_i^2) where s_i = share of revenue recovered by substitute i
    HHI -> 1: concentrated recovery (fragile, one substitute captures most demand)
    HHI -> 0: diversified recovery (robust, many substitutes share demand)

    Returns DataFrame with HHI per delisted product.
    """
    dt = demand_transference_df.copy()
    dt["from_category"] = dt["from_product"].map(dict(cluster_assignments.items()))
    dt["to_category"] = dt["to_product"].map(dict(cluster_assignments.items()))

    rows = []
    for delisted in dt["from_product"].unique():
        delisted_dt = dt[dt["from_product"] == delisted]
        if delisted_dt.empty:
            continue

        # Revenue at risk shares per substitute
        total_risk = delisted_dt["revenue_at_risk"].sum()
        if total_risk == 0:
            continue

        shares = delisted_dt.groupby("to_product")["revenue_at_risk"].sum() / total_risk
        hhi = (shares**2).sum()

        rows.append(
            {
                "delisted_product": delisted,
                "recovery_hhi": hhi,
                "n_substitutes": len(shares),
                "total_revenue_at_risk": total_risk,
                "top_substitute": shares.idxmax() if len(shares) > 0 else None,
                "top_share": shares.max() if len(shares) > 0 else 0,
            }
        )

    return pd.DataFrame(rows).sort_values("recovery_hhi")
