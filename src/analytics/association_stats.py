"""Statistical significance testing for association rules.

Provides Fisher's exact test p-values and Benjamini-Hochberg FDR-adjusted
q-values for pairwise co-purchase associations, elevating market basket
analysis to academic publication standard.

Also adds:
- Collective strength (symmetry-corrected conviction)
- Added value  A->B  (conditional lift minus baseline)
- Temporal recency-weighted support (exponential decay on transaction dates)
- Mutual information per pair (information-theoretic association measure)

References
----------
- Fisher, R.A. (1922). On the interpretation of chi-squared from contingency
  tables. J. Royal Statistical Society.
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate.
  J. Royal Statistical Society Series B, 57(1), 289-300.
- Tan, P., Kumar, V. & Srivastava, J. (2004). Selecting the right objective
  measure for association analysis. Information Systems, 29(4), 293-313.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


# ---------------------------------------------------------------------------
# 1. Statistical significance
# ---------------------------------------------------------------------------

def fisher_pvalue_from_table(
    both: int, a_only: int, b_only: int, neither: int
) -> float:
    """Compute one-sided Fisher exact test p-value for positive association.

    Tests H0: products A and B are purchased independently.
    Alternative: A and B co-occur more than expected by chance.

    The 2x2 contingency table is:
         B=1      B=0
    A=1 [both]  [a_only]
    A=0 [b_only][neither]
    """
    table = np.array([[both, a_only], [b_only, neither]])
    _, pvalue = fisher_exact(table, alternative="greater")
    return float(pvalue)


def benjamini_hochberg_correction(
    pvalues: pd.Series,
    alpha: float = 0.05,
) -> pd.Series:
    """Apply Benjamini-Hochberg FDR correction to a Series of p-values.

    Returns q-values (adjusted p-values). Pairs with q_value <= alpha
    are considered statistically significant at the given FDR level.

    Parameters
    ----------
    pvalues : Series of raw p-values (any index).
    alpha   : FDR level (default 0.05).

    Returns
    -------
    Series of q-values with same index as pvalues.
    """
    n = len(pvalues)
    if n == 0:
        return pvalues.copy()

    sorted_idx = pvalues.argsort()
    ranks = np.arange(1, n + 1)

    sorted_pvals = pvalues.iloc[sorted_idx].values
    q_sorted = sorted_pvals * n / ranks

    # Ensure monotonicity (cumulative minimum from right)
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])

    q_sorted = np.minimum(q_sorted, 1.0)

    q_values = pd.Series(index=pvalues.index, dtype=float)
    q_values.iloc[sorted_idx] = q_sorted
    return q_values


def add_significance_to_pairs(
    pairs_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    alpha: float = 0.05,
    product_col: str = "stockcode",
) -> pd.DataFrame:
    """Enrich a co-purchase pairs DataFrame with statistical significance columns.

    Expects pairs_df to have columns: product_a, product_b, support.
    Adds: p_value, q_value, is_significant.

    The 2x2 table is reconstructed from the basket matrix.
    """
    from src.algorithms.fpgrowth import create_basket_matrix

    basket = create_basket_matrix(transactions_df)
    n_customers = len(basket)
    product_support = basket.mean()

    result = pairs_df.copy()
    pvalues = []

    for _, row in result.iterrows():
        pa = row.get("product_a")
        pb = row.get("product_b")
        if pa not in basket.columns or pb not in basket.columns:
            pvalues.append(1.0)
            continue

        p_a = product_support.get(pa, 0)
        p_b = product_support.get(pb, 0)
        p_ab = row.get("support", 0)

        both = int(round(p_ab * n_customers))
        a_count = int(round(p_a * n_customers))
        b_count = int(round(p_b * n_customers))
        a_only = max(0, a_count - both)
        b_only = max(0, b_count - both)
        neither = max(0, n_customers - a_count - b_count + both)

        pval = fisher_pvalue_from_table(both, a_only, b_only, neither)
        pvalues.append(pval)

    result["p_value"] = pvalues
    result["q_value"] = benjamini_hochberg_correction(
        pd.Series(pvalues, index=result.index)
    ).values
    result["is_significant"] = result["q_value"] <= alpha
    return result


# ---------------------------------------------------------------------------
# 2. Additional association metrics
# ---------------------------------------------------------------------------

def collective_strength(
    support_ab: float, p_a: float, p_b: float
) -> float:
    """Collective strength: symmetry-corrected conviction.

    CS = P(A union B) / (1 - P(A union B))  *  (1 - P(A)P(B)) / (P(A)P(B))

    CS > 1 means A and B violate independence in the positive direction.
    More robust than lift for high-support items.
    Reference: Tan et al. (2004).
    """
    p_union = p_a + p_b - support_ab
    if p_union >= 1.0 or p_union <= 0:
        return np.nan
    expected_union = p_a + p_b - p_a * p_b
    if expected_union <= 0 or expected_union >= 1:
        return np.nan
    cs = (p_union / (1 - p_union)) * ((1 - p_a * p_b) / (p_a * p_b))
    return float(cs)


def added_value(support_ab: float, p_a: float, p_b: float) -> float:
    """Added value of rule A -> B.

    AV(A->B) = P(B|A) - P(B) = support(A,B)/P(A) - P(B)

    Measures the incremental probability of B beyond its base rate.
    AV > 0: positive rule, A increases likelihood of B.
    AV < 0: negative rule, A decreases likelihood of B (repulsion).
    Range: (-1, 1).
    """
    if p_a <= 0:
        return np.nan
    conf = support_ab / p_a
    return float(conf - p_b)


def mutual_information_pair(support_ab: float, p_a: float, p_b: float) -> float:
    """Pointwise mutual information (PMI) for a product pair.

    PMI(A, B) = log2( P(A,B) / (P(A) * P(B)) )
               = log2(lift)

    Positive PMI: words/items appear together more than expected.
    PMI = 0: independence.
    """
    if p_a <= 0 or p_b <= 0 or support_ab <= 0:
        return 0.0
    return float(np.log2(support_ab / (p_a * p_b)))


def temporal_recency_support(
    transactions_df: pd.DataFrame,
    product_a: str,
    product_b: str,
    decay_halflife_days: int = 180,
    product_col: str = "stockcode",
) -> float:
    """Recency-weighted co-purchase support with exponential decay.

    Recent co-occurrences are weighted more heavily than old ones.
    Weight(t) = exp(-lambda * days_ago)  where lambda = ln(2) / halflife.

    Returns weighted support (float in [0, 1]).
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    reference_date = df["date"].max()

    if "transaction_id" in df.columns:
        basket_id_col = "transaction_id"
    else:
        df["_bid"] = (
            df["customer_id"].astype(str) + "_"
            + df["date"].dt.strftime("%Y%m%d")
        )
        basket_id_col = "_bid"

    baskets_with_a = set(df[df[product_col] == product_a][basket_id_col])
    baskets_with_b = set(df[df[product_col] == product_b][basket_id_col])
    both_baskets = baskets_with_a & baskets_with_b

    if not both_baskets:
        return 0.0

    basket_dates = df[df[basket_id_col].isin(both_baskets)].groupby(basket_id_col)["date"].min()
    days_ago = (reference_date - basket_dates).dt.days.clip(lower=0)

    lam = np.log(2) / decay_halflife_days
    weights = np.exp(-lam * days_ago)

    # Normalise by total weighted basket count
    all_basket_dates = df.groupby(basket_id_col)["date"].min()
    all_days_ago = (reference_date - all_basket_dates).dt.days.clip(lower=0)
    total_weight = np.exp(-lam * all_days_ago).sum()

    return float(weights.sum() / total_weight) if total_weight > 0 else 0.0


def enrich_pairs_with_extended_metrics(
    pairs_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add all extended metrics to a co-purchase pairs DataFrame.

    Adds: collective_strength, added_value_a_to_b, added_value_b_to_a,
          mutual_information, p_value, q_value, is_significant.

    Input pairs_df must have: product_a, product_b, support,
    confidence_a_to_b, confidence_b_to_a, p_a (marginal prob of A),
    p_b (marginal prob of B). If p_a/p_b missing, recomputed from transactions.
    """
    from src.algorithms.fpgrowth import create_basket_matrix

    basket = create_basket_matrix(transactions_df)
    product_probs = basket.mean()

    df = pairs_df.copy()

    cs_vals, av_ab, av_ba, mi_vals = [], [], [], []

    for _, row in df.iterrows():
        pa = row["product_a"]
        pb = row["product_b"]
        sup = row["support"]
        p_a = product_probs.get(pa, 0)
        p_b = product_probs.get(pb, 0)

        cs_vals.append(collective_strength(sup, p_a, p_b))
        av_ab.append(added_value(sup, p_a, p_b))
        av_ba.append(added_value(sup, p_b, p_a))
        mi_vals.append(mutual_information_pair(sup, p_a, p_b))

    df["collective_strength"] = cs_vals
    df["added_value_a_to_b"] = av_ab
    df["added_value_b_to_a"] = av_ba
    df["mutual_information"] = mi_vals

    # Add significance
    df = add_significance_to_pairs(df, transactions_df)
    return df
