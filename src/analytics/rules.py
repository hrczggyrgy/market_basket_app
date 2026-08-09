"""Frequent itemsets and association rules via mlxtend (FP-Growth).

No hand-rolled mining: mlxtend's fpgrowth + association_rules are the
well-tested implementations. This module only orchestrates them and validates
its declared output contracts.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

from src.analytics.schemas import CATEGORY_RULES, FREQUENT_ITEMSETS, RULES, RULES_TABLE, check


@dataclass
class ActionabilityConfig:
    """Configuration for actionability calibration workflow."""
    min_support: float = 0.02
    max_support: float = 0.05
    min_confidence: float = 0.40
    max_confidence: float = 0.60
    min_lift: float = 1.2
    sample_size: int = 100
    random_state: int = 42


@dataclass
class ActionabilityResult:
    """Result of actionability calibration."""
    total_rules_reviewed: int
    actionable_count: int
    non_actionable_count: int
    actionability_rate: float
    locked_thresholds: dict
    review_log: list


DEFAULT_ACTIONABILITY_CONFIG = ActionabilityConfig()


def create_basket_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Wide binary basket matrix: one row per transaction, 1 if item present."""
    matrix = (
        df.groupby(["transaction_id", "stockcode"])
        .size()
        .unstack(fill_value=0)
        .clip(upper=1)
        .astype(bool)
    )
    return matrix


def run_fpgrowth(
    basket: pd.DataFrame,
    min_support: float = 0.01,
    max_len: int = 3,
    max_skus: int = 5000,
) -> pd.DataFrame:
    """Frequent itemsets via mlxtend FP-Growth, with itemset length added.

    Args:
        basket: Binary basket matrix (transaction_id x stockcode)
        min_support: Minimum support threshold
        max_len: Maximum itemset length
        max_skus: Maximum number of SKUs to process (scalability guard)
    """
    n_skus = basket.shape[1]
    if n_skus > max_skus:
        # Sample SKUs by support to reduce dimensionality
        sku_support = basket.mean().sort_values(ascending=False)
        top_skus = sku_support.head(max_skus).index
        basket = basket[top_skus]
        import warnings
        warnings.warn(
            f"FP-Growth: SKU count ({n_skus}) exceeds max_skus ({max_skus}). "
            f"Using top {max_skus} SKUs by support.",
            UserWarning,
            stacklevel=2
        )

    freq = fpgrowth(basket, min_support=min_support, use_colnames=True, max_len=max_len)
    if freq.empty:
        return check(pd.DataFrame(columns=list(FREQUENT_ITEMSETS.columns)), FREQUENT_ITEMSETS, allow_empty=True)
    freq = freq.copy()
    freq["length"] = freq["itemsets"].map(len)
    freq = freq.sort_values(["support", "length"], ascending=False).reset_index(drop=True)
    return check(freq, FREQUENT_ITEMSETS)


def generate_rules(
    freq_items: pd.DataFrame,
    metric: str = "confidence",
    min_threshold: float = 0.1,
) -> pd.DataFrame:
    """Association rules from frequent itemsets (mlxtend)."""
    check(freq_items, FREQUENT_ITEMSETS, allow_empty=True)
    if freq_items.empty:
        return check(pd.DataFrame(columns=list(RULES.columns)), RULES, allow_empty=True)
    rules = association_rules(freq_items, metric=metric, min_threshold=min_threshold)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    
    # Add required columns with defaults
    if "lift_ci_lower" not in rules.columns:
        rules["lift_ci_lower"] = np.nan
    if "lift_ci_upper" not in rules.columns:
        rules["lift_ci_upper"] = np.nan
    if "q_value" not in rules.columns:
        rules["q_value"] = np.nan
    if "is_redundant" not in rules.columns:
        rules["is_redundant"] = False
    
    if rules.empty:
        return check(pd.DataFrame(columns=list(RULES.columns)), RULES, allow_empty=True)
    return check(rules, RULES)


def filter_rules(
    rules: pd.DataFrame,
    min_support: float = 0.0,
    min_confidence: float = 0.0,
    min_lift: float = 0.0,
    max_lift: float = 100.0,
) -> pd.DataFrame:
    """Filter rules by support / confidence / lift bounds."""
    check(rules, RULES, allow_empty=True)
    if rules.empty:
        return rules
    mask = (
        rules["support"].ge(min_support)
        & rules["confidence"].ge(min_confidence)
        & rules["lift"].ge(min_lift)
        & rules["lift"].le(max_lift)
    )
    return check(rules.loc[mask].reset_index(drop=True), RULES, allow_empty=True)


def flag_redundant_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """Mark rules subsumed by a shorter, equally-strong rule (same consequent).

    A rule ``X -> Y`` is flagged redundant when a strict subset of its
    antecedent ``X' ⊂ X`` yields the same consequent with confidence and lift
    that are at least as high.
    """
    check(rules, RULES, allow_empty=True)
    result = rules.copy()
    if result.empty:
        return result

    by_consequent: dict[frozenset, list[int]] = {}
    for idx, row in result.iterrows():
        by_consequent.setdefault(row["consequents"], []).append(idx)

    for idxs in by_consequent.values():
        order = sorted(
            idxs,
            key=lambda i: (len(result.loc[i, "antecedents"]), -result.loc[i, "confidence"]),
        )
        for i in order:
            ante_i = result.loc[i, "antecedents"]
            if len(ante_i) < 2:
                continue
            for j in order:
                ante_j = result.loc[j, "antecedents"]
                if i == j or len(ante_j) >= len(ante_i):
                    continue
                if ante_j.issubset(ante_i) and (
                    result.loc[j, "confidence"] >= result.loc[i, "confidence"]
                    and result.loc[j, "lift"] >= result.loc[i, "lift"]
                ):
                    result.loc[i, "is_redundant"] = True
                    break
    return check(result, RULES)


def benjamini_hochberg_fdr(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg procedure for FDR control.
    
    Returns q-values (FDR-adjusted p-values).
    
    Args:
        p_values: Array of p-values
        alpha: Target FDR level
        
    Returns:
        Array of q-values (same shape as input)
    """
    n = len(p_values)
    if n == 0:
        return np.array([])
    
    # Sort p-values and keep original indices
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    # Compute q-values
    q_values = np.zeros(n)
    for i, p in enumerate(sorted_p):
        # BH: q_i = min(p_i * n / (i+1), 1) for i from 1 to n
        q = p * n / (i + 1)
        q_values[sorted_idx[i]] = min(q, 1.0)
    
    # Monotonicity correction: q_i <= q_{i+1}
    for i in range(n - 2, -1, -1):
        if q_values[i] > q_values[i + 1]:
            q_values[i] = q_values[i + 1]
    
    return q_values


def add_fdr_correction(rules: pd.DataFrame, p_value_col: str = "p_value") -> pd.DataFrame:
    """Add Benjamini-Hochberg FDR corrected q-values to rules DataFrame.
    
    Args:
        rules: DataFrame of association rules
        p_value_col: Column name containing p-values for lift test
        
    Returns:
        Rules DataFrame with added 'q_value' column
    """
    check(rules, RULES, allow_empty=True)
    if rules.empty or p_value_col not in rules.columns:
        return rules
    
    p_values = rules[p_value_col].values
    q_values = benjamini_hochberg_fdr(p_values)
    result = rules.copy()
    result["q_value"] = q_values
    return check(result, RULES)


def bootstrap_lift_ci(
    df: pd.DataFrame,
    rules: pd.DataFrame,
    metric: str = "confidence",
    min_threshold: float = 0.05,
    max_len: int = 3,
    n_resamples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Customer-level bootstrap CI on lift for each rule.

    Resamples the customer base with replacement, re-mines rules from the
    resampled baskets, and records each rule's lift across resamples. Returns a
    copy of ``rules`` with ``lift_ci_lower`` / ``lift_ci_upper`` filled from the
    bootstrap percentiles (NaN when a rule has fewer than 3 valid resamples).
    """
    check(rules, RULES, allow_empty=True)
    result = rules.copy()
    if result.empty or len(df) == 0:
        return result

    rng = np.random.default_rng(seed)
    customers = np.asarray(df["customer_id"].unique())
    cust_groups = {c: g for c, g in df.groupby("customer_id")}

    def rule_key(row: pd.Series) -> tuple[frozenset, frozenset]:
        return (frozenset(row["antecedents"]), frozenset(row["consequents"]))

    target_keys: list[tuple[frozenset, frozenset]] = [rule_key(row) for _, row in rules.iterrows()]

    lifts: dict[int, list[float]] = {i: [] for i in rules.index}
    for _ in range(n_resamples):
        cust_idx = rng.integers(0, len(customers), size=len(customers))
        frames = [cust_groups[c] for c in customers[cust_idx]]
        sample = pd.concat(frames, ignore_index=True)
        if sample.empty:
            continue
        basket = create_basket_matrix(sample)
        if basket.empty:
            continue
        freq = run_fpgrowth(basket, min_support=rules["support"].min(), max_len=max_len)
        if freq.empty:
            continue
        resampled = association_rules(freq, metric=metric, min_threshold=min_threshold)
        if resampled.empty:
            continue
        resampled["_key"] = [rule_key(row) for _, row in resampled.iterrows()]
        lookup = resampled.set_index("_key")["lift"]
        for idx, key in zip(rules.index, target_keys, strict=True):
            if key in lookup.index and np.isfinite(lookup[key]):
                lifts[idx].append(float(lookup[key]))

    for idx, values in lifts.items():
        if len(values) >= 3:
            result.loc[idx, "lift_ci_lower"] = float(np.percentile(values, 5))
            result.loc[idx, "lift_ci_upper"] = float(np.percentile(values, 95))
    return check(result, RULES)


def rules_to_table(rules: pd.DataFrame, product_lookup: pd.DataFrame | None = None) -> pd.DataFrame:
    """Human-readable rule table with joined product names."""
    check(rules, RULES, allow_empty=True)

    def name(ids: frozenset) -> str:
        items = sorted(ids)
        if product_lookup is not None:
            lookup = product_lookup.set_index("stockcode")["product"].to_dict()
            return " + ".join(str(lookup.get(i, i)) for i in items)
        return " + ".join(str(i) for i in items)

    table = pd.DataFrame(
        {
            "antecedent": rules["antecedents"].map(name),
            "consequent": rules["consequents"].map(name),
            "support": rules["support"],
            "confidence": rules["confidence"],
            "lift": rules["lift"],
            "leverage": rules["leverage"],
            "conviction": rules["conviction"],
            "zhangs_metric": rules["zhangs_metric"],
        }
    )
    return check(table, RULES_TABLE, allow_empty=True)


def aggregate_rules_to_categories(
    rules: pd.DataFrame,
    product_lookup: pd.DataFrame,
    transactions_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate SKU-level association rules to category-level rules.
    
    Maps antecedents and consequents to their categories, then aggregates
    by (antecedent_category, consequent_category) computing basket-count-based lift.
    
    Category lift is computed from basket counts: 
    lift = P(antecedent_cat & consequent_cat) / (P(antecedent_cat) * P(consequent_cat))
    where probabilities are estimated from transaction basket frequencies.
    
    Args:
        rules: DataFrame of SKU-level rules (RULES contract)
        product_lookup: DataFrame with stockcode -> category mapping
        transactions_df: Optional transaction data for computing basket-based category lift
        
    Returns:
        DataFrame validated against CATEGORY_RULES contract
    """
    check(rules, RULES, allow_empty=True)
    if rules.empty:
        return check(pd.DataFrame(columns=list(CATEGORY_RULES.columns)), CATEGORY_RULES, allow_empty=True)
    
    # Build stockcode -> category mapping
    cat_map = product_lookup.set_index("stockcode")["category"].to_dict()
    
    def map_to_category(itemset: frozenset) -> str:
        """Map a frozenset of stockcodes to a sorted, joined string of categories."""
        cats = sorted(cat_map.get(item, "Unknown") for item in itemset)
        return " + ".join(cats)
    
    rules_cat = rules.copy()
    rules_cat["antecedent_category"] = rules_cat["antecedents"].map(map_to_category)
    rules_cat["consequent_category"] = rules_cat["consequents"].map(map_to_category)
    
    # Compute basket-based category lift if transaction data provided
    if transactions_df is not None:
        # Create basket matrix for category-level analysis
        from src.analytics.rules import create_basket_matrix
        cat_basket = transactions_df.copy()
        cat_basket["category"] = cat_basket["stockcode"].map(cat_map)
        cat_basket = cat_basket.dropna(subset=["category"])
        
        # Per-basket category presence
        cat_basket_matrix = (
            cat_basket.groupby(["transaction_id", "category"])
            .size()
            .unstack(fill_value=0)
            .clip(upper=1)
            .astype(bool)
        )
        
        n_baskets = len(cat_basket_matrix)
        cat_support = cat_basket_matrix.mean()
        
        def compute_cat_lift(row):
            ante_cats = row["antecedent_category"].split(" + ")
            cons_cats = row["consequent_category"].split(" + ")
            
            # P(antecedent_cat) = all antecedent categories present
            if len(ante_cats) == 1:
                ante_mask = cat_basket_matrix[ante_cats[0]]
            else:
                ante_mask = cat_basket_matrix[ante_cats].all(axis=1)
            
            if len(cons_cats) == 1:
                cons_mask = cat_basket_matrix[cons_cats[0]]
            else:
                cons_mask = cat_basket_matrix[cons_cats].all(axis=1)
            
            both_mask = ante_mask & cons_mask
            p_both = both_mask.mean()
            p_ante = ante_mask.mean()
            p_cons = cons_mask.mean()
            
            if p_ante > 0 and p_cons > 0:
                return float(p_both / (p_ante * p_cons))
            return 1.0
        
        rules_cat["cat_lift"] = rules_cat.apply(compute_cat_lift, axis=1)
    else:
        # Fallback to SKU-level lift mean (but warn)
        import warnings
        warnings.warn(
            "aggregate_rules_to_categories: No transactions_df provided. "
            "Category lift computed as mean of SKU-level lifts (not basket-based).",
            UserWarning,
            stacklevel=2
        )
        rules_cat["cat_lift"] = rules_cat["lift"]
    
    # Aggregate by category pair
    agg = rules_cat.groupby(["antecedent_category", "consequent_category"]).agg(
        rule_count=("lift", "count"),
        support=("support", "mean"),
        confidence=("confidence", "mean"),
        lift=("lift", "mean"),  # SKU-level mean lift
        avg_lift=("cat_lift", "mean"),  # Basket-based category lift (renamed for schema compatibility)
        max_lift=("lift", "max"),
    ).reset_index()
    
    # Sort by basket-based category lift (primary), then rule_count
    agg = agg.sort_values(["avg_lift", "rule_count"], ascending=[False, False]).reset_index(drop=True)
    
    return check(agg, CATEGORY_RULES, allow_empty=True)


def get_default_params() -> dict:
    """Return default starting parameters for actionability calibration."""
    return {
        "min_support": 0.02,
        "min_confidence": 0.40,
        "min_lift": 1.2,
    }


def get_actionability_guidance(actionability_rate: float) -> str:
    """Return guidance text based on actionability rate."""
    if actionability_rate < 0.30:
        return (
            "⚠️ Actionability rate is below 30%. Consider **raising thresholds**: "
            "increase min_support to 0.03-0.05, min_confidence to 0.50-0.60, or min_lift to 1.5-2.0. "
            "Too many rules are being generated but few are actionable."
        )
    elif actionability_rate < 0.40:
        return (
            "⚠️ Actionability rate is below 40%. Consider **raising thresholds slightly**: "
            "increase min_support to 0.025-0.03, min_confidence to 0.45-0.50, or min_lift to 1.3-1.5."
        )
    elif actionability_rate > 0.70:
        return (
            "✅ Actionability rate is above 70%. Consider **lowering thresholds** to discover more rules: "
            "decrease min_support to 0.01-0.015, min_confidence to 0.30-0.35, or min_lift to 1.1."
        )
    elif actionability_rate > 0.60:
        return (
            "✅ Actionability rate is above 60%. You may **lower thresholds slightly** to discover more rules: "
            "decrease min_support to 0.015-0.02, min_confidence to 0.35-0.40."
        )
    else:
        return (
            "ℹ️ Actionability rate is in the 40-60% range. This is a reasonable balance. "
            "Fine-tune thresholds based on domain knowledge if needed."
        )


def sample_rules_for_review(
    rules: pd.DataFrame,
    n: int = 100,
    random_state: int = 42,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Sample a random subset of rules for manual actionability review.
    
    Args:
        rules: DataFrame of association rules (already filtered/sorted by lift)
        n: Number of rules to sample (default 100)
        random_state: Random seed for reproducibility
        seed: Deprecated alias for random_state
    
    Returns:
        Sampled DataFrame with added 'actionable' column (initially None)
    """
    if rules.empty:
        return check(pd.DataFrame(columns=list(RULES.columns) + ["actionable"]), RULES, allow_empty=True)
    
    if seed is not None:
        random_state = seed
    
    rng = random.Random(random_state)
    rules_copy = rules.copy()
    n = min(n, len(rules_copy))
    
    # Use random.sample for unbiased sampling without replacement
    sampled_indices = rng.sample(range(len(rules_copy)), n)
    sampled = rules_copy.iloc[sampled_indices].copy()
    
    # Add actionable column for manual labeling (None = not yet reviewed)
    sampled["actionable"] = None
    sampled["review_notes"] = ""
    
    return check(sampled, RULES, allow_empty=True)


def record_actionability_judgment(
    reviewed_rules: pd.DataFrame,
    rule_indices: list,
    actionable: bool,
    notes: str = "",
) -> pd.DataFrame:
    """Record actionability judgment for specific rules.
    
    Args:
        reviewed_rules: DataFrame with actionable column
        rule_indices: List of row indices to label
        actionable: True if actionable, False if not
        notes: Optional review notes
    
    Returns:
        Updated DataFrame with actionability labels recorded
    """
    updated = reviewed_rules.copy()
    for idx in rule_indices:
        if idx in updated.index:
            updated.loc[idx, "actionable"] = actionable
            if notes:
                updated.loc[idx, "review_notes"] = notes
    return check(updated, RULES, allow_empty=True)


def calculate_actionability_rate(reviewed_rules: pd.DataFrame) -> tuple:
    """Calculate actionability rate from reviewed rules.
    
    Returns:
        (actionable_count, non_actionable_count, pending_count, actionability_rate, guidance_text)
    """
    reviewed = reviewed_rules[reviewed_rules["actionable"].notna()]
    total_reviewed = len(reviewed)
    
    if total_reviewed == 0:
        return (0, 0, 0, 0.0, "No rules reviewed yet. Start reviewing to calculate actionability rate.")
    
    actionable_count = int(reviewed["actionable"].sum())
    non_actionable_count = total_reviewed - actionable_count
    pending_count = len(reviewed_rules) - total_reviewed
    actionability_rate = actionable_count / total_reviewed
    
    guidance = get_actionability_guidance(actionable_count / total_reviewed)
    
    return (actionable_count, non_actionable_count, len(reviewed_rules) - total_reviewed, actionability_rate, guidance)


def export_thresholds(thresholds: dict, filepath: str) -> None:
    """Export locked-in thresholds to a JSON config file."""
    config = {
        "min_support": thresholds.get("min_support", 0.02),
        "min_confidence": thresholds.get("min_confidence", 0.40),
        "min_lift": thresholds.get("min_lift", 1.2),
        "exported_at": pd.Timestamp.now().isoformat(),
    }
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)


def load_thresholds(filepath: str) -> dict:
    """Load locked-in thresholds from a JSON config file."""
    with open(filepath, "r") as f:
        return json.load(f)


def run_actionability_calibration(
    df: pd.DataFrame,
    config: Optional[ActionabilityConfig] = None,
) -> ActionabilityResult:
    """Run the full actionability calibration workflow.
    
    This is the main entry point for the actionability calibration workflow.
    It runs FP-Growth, generates rules, samples for review, and returns
    a result object with the calibration outcome.
    
    Args:
        df: Transaction DataFrame
        config: Optional ActionabilityConfig (uses defaults if None)
    
    Returns:
        ActionabilityResult with calibration outcome
    """
    if config is None:
        config = ActionabilityConfig()
    
    # Run FP-Growth with default parameters
    basket = create_basket_matrix(df)
    freq = run_fpgrowth(basket, min_support=config.min_support, max_len=3)
    
    if freq.empty:
        return ActionabilityResult(
            total_rules_reviewed=0,
            actionable_count=0,
            non_actionable_count=0,
            actionability_rate=0.0,
            locked_thresholds=dict(
                min_support=config.min_support,
                min_confidence=config.min_confidence,
                min_lift=config.min_lift,
            ),
            review_log=[],
        )
    
    # Generate rules with default parameters
    rules = generate_rules(freq, metric="confidence", min_threshold=0.40)
    
    if rules.empty:
        return ActionabilityResult(
            total_rules_reviewed=0,
            actionable_count=0,
            non_actionable_count=0,
            actionability_rate=0.0,
            locked_thresholds=dict(
                min_support=config.min_support,
                min_confidence=0.40,
                min_lift=config.min_lift,
            ),
            review_log=[],
        )
    
    # Filter by minimum lift
    rules = filter_rules(rules, min_lift=1.2)
    
    if rules.empty:
        return ActionabilityResult(
            total_rules_reviewed=0,
            actionable_count=0,
            non_actionable_count=0,
            actionability_rate=0.0,
            locked_thresholds=dict(
                min_support=config.min_support,
                min_confidence=0.40,
                min_lift=config.min_lift,
            ),
            review_log=[],
        )
    
    # Sort by lift (primary) then support/confidence (secondary)
    rules = rules.sort_values(["lift", "support", "confidence"], ascending=[False, False, False]).reset_index(drop=True)
    
    # Sample for review
    sample_size = min(100, len(rules))
    sampled = sample_rules_for_review(rules, n=100, random_state=42)
    
    # Return result with sampled rules for review
    # The actual review would be done in the UI
    return ActionabilityResult(
        total_rules_reviewed=0,  # Will be updated after review
        actionable_count=0,
        non_actionable_count=0,
        actionability_rate=0.0,
        locked_thresholds=dict(
            min_support=config.min_support,
            min_confidence=0.40,
            min_lift=config.min_lift,
        ),
        review_log=[],  # Will be populated during review
    )


# Ensure rules table is always sorted by lift first (primary), then support/confidence (secondary)
def sort_rules_by_lift(rules: pd.DataFrame) -> pd.DataFrame:
    """Sort rules by lift (primary), then support and confidence (secondary)."""
    if rules.empty:
        return rules
    return rules.sort_values(
        ["lift", "support", "confidence"],
        ascending=[False, False, False]
    ).reset_index(drop=True)


def get_actionability_guidance(actionability_rate: float) -> str:
    """Return guidance text based on actionability rate."""
    if actionability_rate < 0.30:
        return (
            "⚠️ Actionability rate is below 30%. Consider **raising thresholds**: "
            "increase min_support to 0.03-0.05, min_confidence to 0.50-0.60, or min_lift to 1.5-2.0. "
            "Too many rules are being generated but few are actionable."
        )
    elif actionability_rate < 0.40:
        return (
            "⚠️ Actionability rate is below 40%. Consider **raising thresholds slightly**: "
            "increase min_support to 0.025-0.03, min_confidence to 0.45-0.50, or min_lift to 1.3-1.5."
        )
    elif actionability_rate > 0.70:
        return (
            "✅ Actionability rate is above 70%. Consider **lowering thresholds** to discover more rules: "
            "decrease min_support to 0.01-0.015, min_confidence to 0.30-0.35, or min_lift to 1.1."
        )
    elif actionability_rate > 0.60:
        return (
            "✅ Actionability rate is above 60%. You may **lower thresholds slightly** to discover more rules: "
            "decrease min_support to 0.015-0.02, min_confidence to 0.35-0.40."
        )
    else:
        return (
            "ℹ️ Actionability rate is in the 40-60% range. This is a reasonable balance. "
            "Fine-tune thresholds based on domain knowledge if needed."
        )


def filter_rules(
    rules: pd.DataFrame,
    min_support: float = 0.0,
    min_confidence: float = 0.0,
    min_lift: float = 0.0,
    max_lift: float = 100.0,
) -> pd.DataFrame:
    """Filter rules by support / confidence / lift bounds."""
    check(rules, RULES, allow_empty=True)
    if rules.empty:
        return rules
    mask = (
        rules["support"].ge(min_support)
        & rules["confidence"].ge(min_confidence)
        & rules["lift"].ge(min_lift)
        & rules["lift"].le(max_lift)
    )
    return check(rules.loc[mask].reset_index(drop=True), RULES, allow_empty=True)


def sort_rules_by_lift(rules: pd.DataFrame) -> pd.DataFrame:
    """Sort rules by lift (primary), then support and confidence (secondary)."""
    if rules.empty:
        return rules
    return rules.sort_values(
        ["lift", "support", "confidence"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
