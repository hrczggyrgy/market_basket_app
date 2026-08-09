"""Demand transference: how much demand reallocates when a SKU is delisted.

Models the fraction of revenue that transfers from a removed product to its
substitutes, weighted by observed switching behaviour and historical revenue.
Outputs feed assortment optimization and delist-impact simulation.

Time windows and switching counts reuse :mod:`src.analytics.switching`.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analytics.data import revenue_column
from src.analytics.schemas import (
    CROSS_ELASTICITY,
    DEMAND_TRANSFERENCE,
    DELIST_IMPACT,
    NODE_DELIST_IMPACT,
    RECOVERY_HHI,
    SDP_SCORES,
    TRANSFERENCE_CI,
    check,
)
from src.analytics.switching import compute_switching_matrix

# Scalar metric keys returned by simulate_assortment_change / delist impact.
TRANSFERENCE_METRICS = ("lost_revenue", "recovered_revenue", "net_impact", "recovery_rate")


def _switch_rates(switching_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize switching counts to row probabilities P(to | from)."""
    df = switching_df.copy()
    total = df.groupby("from_product")["count"].transform("sum")
    df["switch_rate"] = np.where(total > 0, df["count"] / total, 0.0)
    return df


def _scalar_value(series: pd.Series, key: object, default: float) -> float:
    """Index a possibly-duplicated series, returning a plain float."""
    value = series.get(key, default)
    if isinstance(value, pd.Series):
        return float(value.iloc[0]) if len(value) else default
    return float(value)


def compute_demand_transference_matrix(
    transactions_df: pd.DataFrame,
    switching_df: pd.DataFrame | None = None,
    product_col: str = "stockcode",
    top_n: int | None = None,
) -> pd.DataFrame:
    """Revenue-weighted observed switching for each ordered product pair.

    observed_switching_transference(A -> B) = P(switch A->B) * revenue_share(A)
    observed_switching_recovery_proxy(A -> B) = P(switch A->B) * revenue(A)

    WHERE P(switch A->B) is the switching count normalized by all switches away
    from A. ``observed_switching_recovery_proxy`` therefore sums across A's substitutes
    to the OBSERVED revenue recovery if A is delisted.

    WARNING: This is an OBSERVED correlation, NOT a causal counterfactual estimate.
    It assumes switching behavior is invariant to delisting (no strategic response).
    """
    df = transactions_df.copy()
    df["revenue"] = revenue_column(df)
    product_revenue = df.groupby(product_col)["revenue"].sum()
    total_revenue = float(product_revenue.sum())
    revenue_share = (product_revenue / total_revenue).rename("revenue_share")

    if switching_df is None:
        switching_df = compute_switching_matrix(df)
    if switching_df.empty:
        return pd.DataFrame(columns=list(DEMAND_TRANSFERENCE.columns))

    result = _switch_rates(switching_df)
    if top_n is not None:
        top_products = product_revenue.nlargest(top_n).index
        result = result[
            result["from_product"].isin(top_products) & result["to_product"].isin(top_products)
        ]

    result["revenue_share_from"] = result["from_product"].map(revenue_share).fillna(0.0)
    result["observed_switching_transference"] = result["switch_rate"] * result["revenue_share_from"]
    result["revenue_from"] = result["from_product"].map(product_revenue).fillna(0.0)
    result["observed_switching_recovery_proxy"] = result["switch_rate"] * result["revenue_from"]

    table = (
        result[
            [
                "from_product",
                "to_product",
                "switch_rate",
                "revenue_share_from",
                "observed_switching_transference",
                "observed_switching_recovery_proxy",
            ]
        ]
        .sort_values("observed_switching_transference", ascending=False)
        .reset_index(drop=True)
    )
    return check(table, DEMAND_TRANSFERENCE)


def compute_substitutable_demand_percentage(
    demand_transference_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
) -> pd.DataFrame:
    """Substitutable Demand Percentage per product.

    SDP(A) = sum of observed switching recovery proxy away from A / total revenue.
    High SDP (>= 0.8) marks a highly substitutable delist candidate; low SDP
    (< 0.2) marks a unique demand driver that must stay in stock.
    
    WARNING: Based on observed switching correlations, NOT causal estimates.
    """
    df = transactions_df.copy()
    df["revenue"] = revenue_column(df)
    total_revenue = float(df["revenue"].sum())

    if total_revenue <= 0:
        return check(pd.DataFrame(columns=list(SDP_SCORES.columns)), SDP_SCORES, allow_empty=True)

    sdp = demand_transference_df.groupby("from_product")["observed_switching_recovery_proxy"].sum() / total_revenue
    table = sdp.reset_index().rename(columns={"from_product": product_col, "observed_switching_recovery_proxy": "sdp"})
    return check(table, SDP_SCORES)


def delist_impact_analysis(
    transactions_df: pd.DataFrame,
    demand_transference_df: pd.DataFrame,
    products_to_delist: Sequence[str],
    product_col: str = "stockcode",
) -> pd.DataFrame:
    """Per-product revenue impact of a simulated delist.

    ``observed_revenue_recovered`` is the sum of ``observed_switching_recovery_proxy`` for the delisted
    product's transfers; ``net_revenue_impact`` = recovered - own revenue.
    
    WARNING: This is based on OBSERVED switching correlations, NOT causal estimates.
    """
    df = transactions_df.copy()
    df["revenue"] = revenue_column(df)
    product_revenue = df.groupby(product_col)["revenue"].sum()

    rows = []
    for prod in products_to_delist:
        rev = _scalar_value(product_revenue, prod, 0.0)
        transferred = float(
            demand_transference_df[
                demand_transference_df["from_product"] == prod
            ]["observed_switching_recovery_proxy"].sum()
        )
        rows.append(
            {
                "stockcode": prod,
                "product_revenue": rev,
                "estimated_revenue_recovered": transferred,
                "net_revenue_impact": transferred - rev,
                "recovery_rate": transferred / rev if rev > 0 else np.nan,
            }
        )

    table = (
        pd.DataFrame(rows, columns=list(DELIST_IMPACT.columns))
        .sort_values("net_revenue_impact")
        .reset_index(drop=True)
    )
    return check(table, DELIST_IMPACT)


def node_delist_impact(
    transactions_df: pd.DataFrame,
    demand_transference_df: pd.DataFrame,
    cluster_assignments: dict[str, int],
    product_col: str = "stockcode",
) -> pd.DataFrame:
    """Per-node summary of intra-node recovery and external leakage.

    ``node_sdp`` = internal recovery / node revenue, a measure of how
    self-contained each cluster (e.g. CDT leaf) is against delists.
    
    WARNING: Based on observed switching correlations, NOT causal estimates.
    """
    df = transactions_df.copy()
    df["revenue"] = revenue_column(df)
    product_revenue = df.groupby(product_col)["revenue"].sum()

    rows = []
    for node_id in sorted(set(cluster_assignments.values())):
        node_products = [p for p in cluster_assignments if cluster_assignments[p] == node_id]
        node_rev = float(sum(product_revenue.get(p, 0.0) for p in node_products))
        node_dt = demand_transference_df[
            demand_transference_df["from_product"].isin(node_products)
        ]
        internal = float(node_dt[node_dt["to_product"].isin(node_products)]["observed_switching_recovery_proxy"].sum())
        external = float(node_dt[~node_dt["to_product"].isin(node_products)]["observed_switching_recovery_proxy"].sum())
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

    table = (
        pd.DataFrame(rows, columns=list(NODE_DELIST_IMPACT.columns))
        .sort_values("total_node_revenue", ascending=False)
        .reset_index(drop=True)
    )
    return check(table, NODE_DELIST_IMPACT)


def build_substitution_matrix_markov(
    switching_df: pd.DataFrame,
    max_iterations: int = 20,
    convergence_threshold: float = 1e-6,
) -> pd.DataFrame:
    """Multi-step substitution probabilities via Markov power iteration.

    Row-normalized one-step switching P is iterated to P^k until convergence,
    capturing indirect substitution chains (A -> B -> C).
    """
    products = sorted(
        set(switching_df["from_product"].unique()) | set(switching_df["to_product"].unique())
    )
    if not products:
        return pd.DataFrame()
    idx = {p: i for i, p in enumerate(products)}
    n = len(products)

    P = np.zeros((n, n))
    for row in switching_df.itertuples(index=False):
        P[idx[row.from_product], idx[row.to_product]] = float(row.count)
    row_sums = P.sum(axis=1)
    P[row_sums > 0] /= P[row_sums > 0].sum(axis=1, keepdims=True)

    P_k = P.copy()
    for _ in range(max_iterations):
        P_next = P_k @ P
        if np.abs(P_next - P_k).max() < convergence_threshold:
            P_k = P_next
            break
        P_k = P_next
    return pd.DataFrame(P_k, index=products, columns=products)


def build_similarity_substitution_score(
    transactions_df: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    price_col: str = "price",
    utility_weight_price: float = -0.5,
    utility_weight_similarity: float = 1.0,
    utility_weight_revenue_share: float = 0.5,
    price_sensitivity_floor: float = 0.0,
) -> pd.DataFrame:
    """Similarity-based substitution score (formerly MNL model).

    WARNING: This is a HEURISTIC similarity-based score, NOT a multinomial logit model.
    - No price coefficient estimation; uses fixed utility weights
    - No choice data; similarity is used as a proxy for substitution
    - Softmax over similarity + price + revenue is a scoring heuristic, NOT a discrete choice model
    - Results are descriptive similarity scores, NOT probability of substitution
    
    USE FOR EXPLORATORY ANALYSIS ONLY.
    
    U(j | i removed) = w_price * price_j + w_sim * sim(i, j)
                      + w_rev * log(revenue_j + 1)
    with a softmax over the products j != i. Uses median price and total
    revenue per product from the transaction snapshot.
    """
    import warnings
    warnings.warn(
        "build_similarity_substitution_score is a heuristic similarity-based score, "
        "NOT a multinomial logit model. No price coefficients are estimated. "
        "Results are descriptive similarity scores, NOT substitution probabilities.",
        UserWarning,
        stacklevel=2
    )
    df = transactions_df.copy()
    df["revenue"] = revenue_column(df)
    product_revenue = df.groupby("stockcode")["revenue"].sum()
    product_price = df.groupby("stockcode")["price"].median()

    products = list(similarity_matrix.index)
    n = len(products)
    U = np.full((n, n), -np.inf)

    for i, prod_i in enumerate(products):
        for j, prod_j in enumerate(products):
            if i == j:
                continue
            price_j = float(product_price.get(prod_j, product_price.mean()))
            if price_j <= price_sensitivity_floor:
                continue
            sim_ij = float(similarity_matrix.loc[prod_i, prod_j])
            rev_j = float(product_revenue.get(prod_j, product_revenue.mean()))
            U[i, j] = (
                utility_weight_price * price_j
                + utility_weight_similarity * sim_ij
                + utility_weight_revenue_share * np.log(rev_j + 1)
            )

    with np.errstate(over="ignore", invalid="ignore"):
        row_max = U.max(axis=1, keepdims=True)
        valid = np.isfinite(row_max)
        exp_U = np.zeros_like(U)
        if valid.any():
            exp_U[valid[:, 0]] = np.exp(U[valid[:, 0]] - row_max[valid[:, 0]])
        P = np.zeros_like(U)
        P[valid[:, 0]] = exp_U[valid[:, 0]] / exp_U[valid[:, 0]].sum(axis=1, keepdims=True)

    return pd.DataFrame(P, index=products, columns=products)


# Backward compatibility alias
build_substitution_matrix_mnl = build_similarity_substitution_score


def build_similarity_matrix(
    transactions_df: pd.DataFrame,
    top_n: int = 100,
    min_cooccurrence: int = 5,
) -> pd.DataFrame:
    """Square symmetric similarity matrix from co-purchase affinity pairs.

    Wraps :func:`src.analytics.copurchase.get_top_affinity_pairs` and pivots
    to a dense matrix, filled with the pair affinity (0 when absent).
    """
    from src.analytics.copurchase import get_top_affinity_pairs

    pairs = get_top_affinity_pairs(
        transactions_df, top_n=top_n, min_cooccurrence=min_cooccurrence
    )
    if pairs.empty:
        return pd.DataFrame()
    products = sorted(set(pairs["product_a"]) | set(pairs["product_b"]))
    matrix = pd.DataFrame(0.0, index=products, columns=products)
    for row in pairs.itertuples(index=False):
        matrix.loc[row.product_a, row.product_b] = float(row.affinity)
        matrix.loc[row.product_b, row.product_a] = float(row.affinity)
    return matrix


def simulate_assortment_change(
    delist_products: Sequence[str],
    substitution_matrix: pd.DataFrame,
    revenue_per_product: pd.Series,
    *,
    extra_kept: Sequence[str] = (),
) -> dict[str, object]:
    """Reallocate delisted revenue through a substitution matrix.

    For each delisted product, every kept product captures
    ``sub_prob * delisted_revenue`` from it (row of the matrix indexed by the
    delisted product, column the kept product).
    """
    from src.analytics.data import safe_divide

    all_products = list(revenue_per_product.index)
    kept = [p for p in all_products if p not in delist_products] + list(extra_kept)
    kept = list(dict.fromkeys(kept))
    kept_set = set(kept)

    lost = float(sum(_scalar_value(revenue_per_product, p, 0.0) for p in delist_products))
    recovered = 0.0
    detail: dict[str, dict[str, float]] = {}
    for delisted in delist_products:
        if delisted not in substitution_matrix.index:
            continue
        delisted_rev = _scalar_value(revenue_per_product, delisted, 0.0)
        for tgt, prob in substitution_matrix.loc[delisted].items():
            if tgt in kept_set and np.isfinite(prob) and prob > 0:
                amount = float(prob) * delisted_rev
                recovered += amount
                detail.setdefault(delisted, {})[tgt] = amount

    return {
        "lost_revenue": lost,
        "recovered_revenue": recovered,
        "net_impact": recovered - lost,
        "recovery_rate": float(safe_divide(recovered, lost)),
        "recovery_detail": detail,
    }


def compute_cross_price_elasticity(
    transactions_df: pd.DataFrame,
    product_pairs: Sequence[tuple[str, str]],
    price_col: str = "price",
    qty_col: str = "quantity",
    date_col: str = "date",
    freq: str = "W",
    min_periods: int = 10,
    min_price_variation: float = 0.05,
) -> pd.DataFrame:
    """Pairwise bivariate log-log OLS cross-price elasticities.

    ``log(qty_A) ~ log(price_A) + log(price_B)`` with HC3 robust SEs.
    A positive cross term marks B as a substitute for A; negative marks a
    complement. Pairs without enough aligned weeks or price variation are
    skipped.
    """
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["revenue"] = revenue_column(df)

    results: list[dict[str, float | int | str]] = []
    for prod_a, prod_b in product_pairs:
        a = df[df["stockcode"] == prod_a].copy()
        b = df[df["stockcode"] == prod_b].copy()
        if a.empty or b.empty:
            continue

        weekly_a = (
            a.set_index(date_col)
            .groupby(pd.Grouper(freq=freq))
            .agg(avg_price_a=(price_col, "mean"), total_qty_a=(qty_col, "sum"))
            .dropna()
        )
        weekly_b = (
            b.set_index(date_col)
            .groupby(pd.Grouper(freq=freq))
            .agg(avg_price_b=(price_col, "mean"), total_qty_b=(qty_col, "sum"))
            .dropna()
        )
        weekly = weekly_a.join(weekly_b, how="inner")
        if len(weekly) < min_periods:
            continue

        cv_a = weekly["avg_price_a"].std() / weekly["avg_price_a"].mean()
        cv_b = weekly["avg_price_b"].std() / weekly["avg_price_b"].mean()
        if cv_a < min_price_variation or cv_b < min_price_variation:
            continue

        # Validate data before log transformation
        if (weekly[["avg_price_a", "avg_price_b", "total_qty_a"]] == 0).any().any():
            continue  # Skip products with zero values
        
        log_price_a = np.log(weekly["avg_price_a"])
        log_price_b = np.log(weekly["avg_price_b"])
        log_qty_a = np.log(weekly["total_qty_a"])
        valid = log_price_a.notna() & log_price_b.notna() & log_qty_a.notna()
        if valid.sum() < min_periods:
            continue

        X = sm.add_constant(log_price_a[valid])
        X["price_b"] = log_price_b[valid]
        y = log_qty_a[valid]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model = sm.OLS(y, X).fit(cov_type="HC3")

        results.append(
            {
                "product_a": prod_a,
                "product_b": prod_b,
                "own_elasticity": float(model.params.iloc[1]),
                "own_elasticity_se": float(model.bse.iloc[1]),
                "own_elasticity_p": float(model.pvalues.iloc[1]),
                "cross_elasticity": float(model.params.iloc[2]),
                "cross_elasticity_se": float(model.bse.iloc[2]),
                "cross_elasticity_p": float(model.pvalues.iloc[2]),
                "r_squared": float(model.rsquared),
                "n_obs": int(valid.sum()),
                "avg_price_a": float(weekly["avg_price_a"].mean()),
                "avg_price_b": float(weekly["avg_price_b"].mean()),
            }
        )

    table = pd.DataFrame(results, columns=list(CROSS_ELASTICITY.columns))
    return check(table, CROSS_ELASTICITY, allow_empty=True)


def compute_recovery_hhi(demand_transference_df: pd.DataFrame) -> pd.DataFrame:
    """Herfindahl concentration of demand recovery per delisted product.

    HHI = sum(s_i^2) over shares of recovered revenue. ~1 indicates a single
    substitute captures most demand (fragile); ~0 indicates diversified,
    robust recovery.
    
    WARNING: Based on observed switching correlations, NOT causal estimates.
    """
    rows = []
    for delisted in demand_transference_df["from_product"].unique():
        dt = demand_transference_df[demand_transference_df["from_product"] == delisted]
        total_risk = float(dt["observed_switching_recovery_proxy"].sum())
        if total_risk <= 0:
            continue
        shares = dt.groupby("to_product")["observed_switching_recovery_proxy"].sum() / total_risk
        rows.append(
            {
                "delisted_product": delisted,
                "recovery_hhi": float((shares**2).sum()),
                "n_substitutes": int(len(shares)),
                "total_revenue_at_risk": total_risk,
                "top_substitute": shares.idxmax(),
                "top_share": float(shares.max()),
            }
        )

    table = (
        pd.DataFrame(rows, columns=list(RECOVERY_HHI.columns))
        .sort_values("recovery_hhi")
        .reset_index(drop=True)
    )
    return check(table, RECOVERY_HHI, allow_empty=True)


def bootstrap_demand_transference_ci(
    transactions_df: pd.DataFrame,
    switching_df: pd.DataFrame | None = None,
    *,
    top_n: int | None = None,
    n_resamples: int = 100,
    ci_level: float = 0.95,
    random_seed: int | None = None,
    max_pairs: int = 10,
) -> pd.DataFrame:
    """Percentile bootstrap CIs for DT estimates, resampling at customer level.

    The full DT matrix is rebuilt per resample (switching included); only the
    top ``max_pairs`` pairs by point estimate are bootstrapped to bound
    runtime.
    """
    rng = np.random.default_rng(random_seed)
    customers = transactions_df["customer_id"].unique()
    cust_groups = {c: g for c, g in transactions_df.groupby("customer_id")}

    def _point(d: pd.DataFrame) -> pd.DataFrame:
        return compute_demand_transference_matrix(
            d, switching_df=switching_df, top_n=top_n
        )

    point = _point(transactions_df)
    if point.empty:
        return check(pd.DataFrame(columns=list(TRANSFERENCE_CI.columns)), TRANSFERENCE_CI, allow_empty=True)

    rows: list[dict[str, float | int | str]] = []
    for pair_row in point.head(max_pairs).itertuples(index=False):
        from_p, to_p = pair_row.from_product, pair_row.to_product
        estimate = float(pair_row.observed_switching_transference)
        replicates: list[float] = []
        for _ in range(n_resamples):
            cust_idx = rng.integers(0, len(customers), size=len(customers))
            frames = [cust_groups[c] for c in customers[cust_idx]]
            resample = pd.concat(frames, ignore_index=True)
            if resample.empty:
                continue
            rt = _point(resample)
            if rt.empty:
                continue
            match = rt[(rt["from_product"] == from_p) & (rt["to_product"] == to_p)]
            replicates.append(float(match["observed_switching_transference"].iloc[0]) if not match.empty else 0.0)

        if not replicates:
            continue
        arr = np.asarray(replicates)
        alpha = 1.0 - ci_level
        rows.append(
            {
                "pair": f"{from_p}->{to_p}",
                "estimate": estimate,
                "lower": float(np.percentile(arr, 100 * alpha / 2)),
                "upper": float(np.percentile(arr, 100 * (1 - alpha / 2))),
                "std_error": float(arr.std(ddof=1)),
                "n_resamples": len(arr),
            }
        )

    table = pd.DataFrame(rows, columns=list(TRANSFERENCE_CI.columns))
    return check(table, TRANSFERENCE_CI, allow_empty=True)