"""Association rule generation with extended metrics."""

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules


def generate_rules(
    frequent_itemsets: pd.DataFrame,
    metric: str = "confidence",
    min_threshold: float = 0.5,
    support_only: bool = False,
) -> pd.DataFrame:
    """
    Generate association rules from frequent itemsets with extended metrics.

    Args:
        frequent_itemsets: DataFrame from FP-Growth with 'support' and 'itemsets' columns
        metric: Metric to evaluate rules ('confidence', 'lift', 'leverage', 'conviction')
        min_threshold: Minimum threshold for the metric
        support_only: If True, only return rules with antecedent support >= min_threshold

    Returns:
        DataFrame with association rules and extended metrics
    """
    if frequent_itemsets.empty:
        return _empty_rules_df()

    # Ensure itemsets are frozenset for mlxtend
    freq_items = frequent_itemsets.copy()
    freq_items["itemsets"] = freq_items["itemsets"].apply(frozenset)

    # Map string items to integers to avoid mlxtend bug with string items
    all_items = set()
    for itemset in freq_items["itemsets"]:
        all_items.update(itemset)

    item_to_int = {item: idx for idx, item in enumerate(all_items)}
    int_to_item = {idx: item for item, idx in item_to_int.items()}

    # Convert itemsets to integer representation
    freq_items_int = freq_items.copy()
    freq_items_int["itemsets"] = freq_items["itemsets"].apply(
        lambda x: frozenset(item_to_int[item] for item in x)
    )

    try:
        rules = association_rules(
            freq_items_int,
            metric=metric,
            min_threshold=min_threshold,
            support_only=support_only,
        )
    except Exception as e:
        raise RuntimeError(f"Association rules generation failed: {str(e)}")

    if rules.empty:
        return _empty_rules_df()

    # Map integer items back to original strings
    def map_items(frozenset_int):
        return frozenset(int_to_item[i] for i in frozenset_int)

    rules["antecedents"] = rules["antecedents"].apply(map_items)
    rules["consequents"] = rules["consequents"].apply(map_items)

    # Add extended metrics
    rules = add_extended_metrics(rules)

    # Convert frozensets to tuples for display
    rules["antecedents"] = rules["antecedents"].apply(tuple)
    rules["consequents"] = rules["consequents"].apply(tuple)

    # Add rule length columns
    rules["antecedent_len"] = rules["antecedents"].apply(len)
    rules["consequent_len"] = rules["consequents"].apply(len)
    rules["rule_len"] = rules["antecedent_len"] + rules["consequent_len"]

    # Sort by lift descending, then confidence
    rules = rules.sort_values(["lift", "confidence"], ascending=[False, False]).reset_index(
        drop=True
    )

    return rules


def add_extended_metrics(rules: pd.DataFrame) -> pd.DataFrame:
    """Add extended association rule metrics."""
    rules = rules.copy()

    # Leverage: P(A,B) - P(A)P(B)
    rules["leverage"] = rules["support"] - (
        rules["antecedent support"] * rules["consequent support"]
    )

    # Conviction: (1 - P(B)) / (1 - P(B|A)) = (1 - P(B)) / (1 - confidence)
    # Handle division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        rules["conviction"] = np.where(
            rules["confidence"] < 1,
            (1 - rules["consequent support"]) / (1 - rules["confidence"]),
            np.inf,
        )

    # Zhang's Metric: (P(A,B) - P(A)P(B)) / max(P(A,B) - P(A)P(B), P(A)P(not B))
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = rules["support"] - (rules["antecedent support"] * rules["consequent support"])
        denominator = np.maximum(
            numerator, rules["antecedent support"] * (1 - rules["consequent support"])
        )
        rules["zhangs_metric"] = np.where(denominator != 0, numerator / denominator, 0)

    # Collective Strength
    with np.errstate(divide="ignore", invalid="ignore"):
        p_a = rules["antecedent support"]
        p_b = rules["consequent support"]
        p_ab = rules["support"]
        p_not_a = 1 - p_a
        p_not_b = 1 - p_b
        p_not_a_not_b = 1 - p_a - p_b + p_ab

        numerator_cs = p_ab + p_not_a_not_b
        denominator_cs = p_a * p_b + p_not_a * p_not_b

        second_num = 1 - p_a * p_b - p_not_a * p_not_b
        second_den = 1 - p_ab - p_not_a_not_b

        rules["collective_strength"] = np.where(
            (denominator_cs != 0) & (second_den != 0),
            (numerator_cs / denominator_cs) * (second_num / second_den),
            1.0,
        )

    # ============ NEW METRICS ============

    # Chi-squared statistic
    with np.errstate(divide="ignore", invalid="ignore"):
        # Build contingency table values
        a = p_ab  # P(A,B)
        b = p_a - p_ab  # P(A, not B)
        c = p_b - p_ab  # P(not A, B)
        d = p_not_a_not_b  # P(not A, not B)

        # Expected values under independence
        e_a = p_a * p_b
        e_b = p_a * p_not_b
        e_c = p_not_a * p_b
        e_d = p_not_a * p_not_b

        # Chi-squared
        chi2 = np.where(
            (e_a > 0) & (e_b > 0) & (e_c > 0) & (e_d > 0),
            (a - e_a) ** 2 / e_a
            + (b - e_b) ** 2 / e_b
            + (c - e_c) ** 2 / e_c
            + (d - e_d) ** 2 / e_d,
            0,
        )
        rules["chi_squared"] = chi2

        # Phi coefficient (correlation for binary variables)
        # phi = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))
        numerator_phi = a * d - b * c
        denominator_phi = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
        rules["phi_coefficient"] = np.where(
            denominator_phi != 0, numerator_phi / denominator_phi, 0
        )

        # Cosine similarity: P(A,B) / sqrt(P(A)P(B))
        rules["cosine"] = np.where((p_a > 0) & (p_b > 0), p_ab / np.sqrt(p_a * p_b), 0)

        # Kulczynski: 0.5 * (P(B|A) + P(A|B)) = 0.5 * (confidence + P(A|B))
        p_a_given_b = np.where(p_b > 0, p_ab / p_b, 0)
        rules["kulczynski"] = 0.5 * (rules["confidence"] + p_a_given_b)

        # Imbalance Ratio: |P(B|A) - P(A|B)| / (P(B|A) + P(A|B))
        rules["imbalance_ratio"] = np.where(
            (rules["confidence"] + p_a_given_b) > 0,
            np.abs(rules["confidence"] - p_a_given_b) / (rules["confidence"] + p_a_given_b),
            0,
        )

        # Odds Ratio: (a*d) / (b*c)
        rules["odds_ratio"] = np.where((b * c) > 0, (a * d) / (b * c), np.inf)

        # Certainty Factor: (P(B|A) - P(B)) / (1 - P(B)) if P(B|A) > P(B)
        rules["certainty_factor"] = np.where(
            rules["confidence"] > p_b,
            (rules["confidence"] - p_b) / (1 - p_b),
            np.where(rules["confidence"] < p_b, (rules["confidence"] - p_b) / p_b, 0),
        )

        # Added Value: P(B|A) - P(B)
        rules["added_value"] = rules["confidence"] - p_b

        # Sebag-Schoenauer: P(A,B) / P(A, not B)
        rules["sebag_schoenauer"] = np.where(b > 0, a / b, np.inf)

        # Jaccard: P(A,B) / P(A or B)
        rules["jaccard"] = np.where((p_a + p_b - p_ab) > 0, p_ab / (p_a + p_b - p_ab), 0)

        # Gini Index: 1 - sum(P(class|rule)^2)
        # For binary: 2 * P(B|A) * (1 - P(B|A))
        rules["gini_index"] = 2 * rules["confidence"] * (1 - rules["confidence"])

        # Laplace correction: (support + 1) / (antecedent_support + 2)
        # Not directly applicable without counts, approximate:
        rules["laplace"] = (p_ab + 1 / p_a) / (p_a + 2 / p_a) if p_a.all() > 0 else 0

    return rules


def filter_rules(
    rules: pd.DataFrame,
    min_support: float = 0.0,
    min_confidence: float = 0.0,
    min_lift: float = 1.0,
    max_lift: float = np.inf,
    min_leverage: float = -np.inf,
    min_conviction: float = 1.0,
    min_zhangs: float = -1.0,
    max_antecedent_len: int = 10,
    max_consequent_len: int = 10,
) -> pd.DataFrame:
    """Filter rules by multiple criteria."""
    if rules.empty:
        return rules

    filtered = rules[
        (rules["support"] >= min_support)
        & (rules["confidence"] >= min_confidence)
        & (rules["lift"] >= min_lift)
        & (rules["lift"] <= max_lift)
        & (rules["leverage"] >= min_leverage)
        & (rules["conviction"] >= min_conviction)
        & (rules["zhangs_metric"] >= min_zhangs)
        & (rules["antecedent_len"] <= max_antecedent_len)
        & (rules["consequent_len"] <= max_consequent_len)
    ].copy()

    return filtered.reset_index(drop=True)


def get_rules_for_item(
    rules: pd.DataFrame,
    item: str,
    as_antecedent: bool = True,
    as_consequent: bool = True,
) -> pd.DataFrame:
    """Get all rules where item appears in antecedent or consequent."""
    if rules.empty:
        return rules

    mask = pd.Series(False, index=rules.index)

    if as_antecedent:
        mask |= rules["antecedents"].apply(lambda x: item in x)
    if as_consequent:
        mask |= rules["consequents"].apply(lambda x: item in x)

    return rules[mask].copy()


def format_rules_for_display(rules: pd.DataFrame, product_lookup: dict = None) -> pd.DataFrame:
    """Format rules for display with product names."""
    display = rules.copy()

    if product_lookup:

        def format_items(items):
            return ", ".join(product_lookup.get(str(i), str(i)) for i in items)

        display["antecedents_str"] = display["antecedents"].apply(format_items)
        display["consequents_str"] = display["consequents"].apply(format_items)
        display["rule"] = display["antecedents_str"] + " → " + display["consequents_str"]
    else:
        display["antecedents_str"] = display["antecedents"].apply(lambda x: ", ".join(map(str, x)))
        display["consequents_str"] = display["consequents"].apply(lambda x: ", ".join(map(str, x)))
        display["rule"] = display["antecedents_str"] + " → " + display["consequents_str"]

    return display


def _empty_rules_df() -> pd.DataFrame:
    """Return empty DataFrame with correct columns."""
    return pd.DataFrame(
        columns=[
            "antecedents",
            "consequents",
            "antecedent support",
            "consequent support",
            "support",
            "confidence",
            "lift",
            "leverage",
            "conviction",
            "zhangs_metric",
            "collective_strength",
            "chi_squared",
            "phi_coefficient",
            "cosine",
            "kulczynski",
            "imbalance_ratio",
            "odds_ratio",
            "certainty_factor",
            "added_value",
            "sebag_schoenauer",
            "jaccard",
            "gini_index",
            "laplace",
            "antecedent_len",
            "consequent_len",
            "rule_len",
        ]
    )
