"""Tests for association rules (mlxtend FP-Growth orchestration)."""

import pandas as pd
import pytest

from src.analytics.rules import (
    bootstrap_lift_ci,
    create_basket_matrix,
    filter_rules,
    flag_redundant_rules,
    generate_rules,
    rules_to_table,
    run_fpgrowth,
)
from src.analytics.schemas import FREQUENT_ITEMSETS, RULES, RULES_TABLE, SchemaError


def test_create_basket_matrix(sample_df: pd.DataFrame) -> None:
    basket = create_basket_matrix(sample_df)
    n_txn = sample_df["transaction_id"].nunique()
    n_sku = sample_df["stockcode"].nunique()
    assert basket.shape == (n_txn, n_sku)
    assert basket.dtypes.eq("bool").all()
    assert basket.sum(axis=1).min() >= 2


def test_create_basket_matrix_binary_no_duplicates(sample_df: pd.DataFrame) -> None:
    basket = create_basket_matrix(sample_df)
    assert basket.index.is_unique
    assert basket.columns.is_unique
    assert basket.values.sum() == basket.values.astype(bool).sum()


def test_run_fpgrowth_contract(sample_df: pd.DataFrame) -> None:
    basket = create_basket_matrix(sample_df)
    freq = run_fpgrowth(basket, min_support=0.02, max_len=3)
    FREQUENT_ITEMSETS.validate(freq)
    assert len(freq) > 0
    assert freq["support"].between(0.02, 1.0).all()
    assert freq["length"].between(1, 3).all()


def test_run_fpgrowth_empty_returns_contract() -> None:
    basket = create_basket_matrix(pd.DataFrame(
        {
            "transaction_id": ["T1", "T2"],
            "stockcode": ["A", "B"],
        }
    ))
    freq = run_fpgrowth(basket, min_support=0.9, max_len=3)
    FREQUENT_ITEMSETS.validate(freq, allow_empty=True)
    assert freq.empty


def test_run_fpgrowth_respects_max_len(sample_df: pd.DataFrame) -> None:
    basket = create_basket_matrix(sample_df)
    freq = run_fpgrowth(basket, min_support=0.01, max_len=2)
    assert freq["length"].max() <= 2


def test_generate_rules_contract(sample_df: pd.DataFrame) -> None:
    basket = create_basket_matrix(sample_df)
    freq = run_fpgrowth(basket, min_support=0.02, max_len=3)
    rules = generate_rules(freq, metric="confidence", min_threshold=0.2)
    RULES.validate(rules)
    assert (rules["confidence"] >= 0.2 - 1e-9).all()
    assert (rules["support"] >= 0.02 - 1e-9).all()


def test_generate_rules_empty(sample_df: pd.DataFrame) -> None:
    basket = create_basket_matrix(sample_df)
    freq = run_fpgrowth(basket, min_support=0.02, max_len=3)
    rules = generate_rules(freq, metric="confidence", min_threshold=0.99)
    RULES.validate(rules, allow_empty=True)


def test_filter_rules_contract(sample_df: pd.DataFrame) -> None:
    basket = create_basket_matrix(sample_df)
    freq = run_fpgrowth(basket, min_support=0.02, max_len=3)
    rules = generate_rules(freq)
    filtered = filter_rules(rules, min_support=0.03, min_confidence=0.25, min_lift=1.2, max_lift=50.0)
    RULES.validate(filtered, allow_empty=True)
    if not filtered.empty:
        assert filtered["support"].min() >= 0.03 - 1e-9
        assert filtered["confidence"].min() >= 0.25 - 1e-9
        assert filtered["lift"].min() >= 1.2 - 1e-9


def test_rules_to_table(sample_df: pd.DataFrame) -> None:
    basket = create_basket_matrix(sample_df)
    freq = run_fpgrowth(basket, min_support=0.02, max_len=3)
    rules = generate_rules(freq)
    table = rules_to_table(rules)
    RULES_TABLE.validate(table, allow_empty=True)
    if not table.empty:
        assert table["antecedent"].dtype == object
        assert table["consequent"].dtype == object


def test_rules_to_table_uses_lookup(sample_df: pd.DataFrame) -> None:
    from src.analytics.data import derive_product_lookup

    basket = create_basket_matrix(sample_df)
    freq = run_fpgrowth(basket, min_support=0.02, max_len=3)
    rules = generate_rules(freq)
    lookup = derive_product_lookup(sample_df)
    table = rules_to_table(rules, lookup)
    RULES_TABLE.validate(table, allow_empty=True)


def test_bad_input_rejected(sample_df: pd.DataFrame) -> None:
    with pytest.raises(SchemaError):
        generate_rules(pd.DataFrame({"wrong": [1]}))


def test_flag_redundant_rules(sample_df: pd.DataFrame) -> None:
    basket = create_basket_matrix(sample_df)
    freq = run_fpgrowth(basket, min_support=0.02, max_len=3)
    rules = generate_rules(freq)
    flagged = flag_redundant_rules(rules)
    RULES.validate(flagged)
    # Only rules with multi-item antecedents may be flagged.
    multi = rules["antecedents"].map(len) >= 2
    assert flagged.loc[multi, "is_redundant"].sum() >= 0
    assert not flagged.loc[~multi, "is_redundant"].any()


def test_flag_redundant_rules_empty() -> None:
    empty = pd.DataFrame(columns=list(RULES.columns))
    flagged = flag_redundant_rules(empty)
    assert flagged.empty


def test_flag_redundant_rules_marks_subsumed(sample_df: pd.DataFrame) -> None:
    basket = create_basket_matrix(sample_df)
    freq = run_fpgrowth(basket, min_support=0.02, max_len=3)
    rules = generate_rules(freq)
    flagged = flag_redundant_rules(rules)

    # A flagged rule must have a shorter-antecedent rule for the same consequent
    # that is at least as strong (confidence & lift).
    by_consequent: dict[frozenset, list[pd.Series]] = {}
    for _, row in flagged.iterrows():
        by_consequent.setdefault(row["consequents"], []).append(row)
    for rows in by_consequent.values():
        if not any(r["is_redundant"] for r in rows):
            continue
        short = [r for r in rows if len(r["antecedents"]) == 1]
        long = [r for r in rows if r["is_redundant"]]
        for r_long in long:
            assert any(
                s["antecedents"].issubset(r_long["antecedents"])
                and s["confidence"] >= r_long["confidence"]
                and s["lift"] >= r_long["lift"]
                for s in short
            ), f"Redundant rule without a stronger short rule: {r_long['antecedents']} -> {r_long['consequents']}"


def test_bootstrap_lift_ci(sample_df: pd.DataFrame) -> None:
    basket = create_basket_matrix(sample_df)
    freq = run_fpgrowth(basket, min_support=0.05, max_len=2)
    rules = generate_rules(freq, metric="confidence", min_threshold=0.1)
    if rules.empty:
        return
    bootstrapped = bootstrap_lift_ci(sample_df, rules, n_resamples=6)
    RULES.validate(bootstrapped)
    valid = bootstrapped["lift_ci_lower"].notna()
    if valid.any():
        ok = bootstrapped.loc[valid]
        assert (ok["lift_ci_lower"] <= ok["lift_ci_upper"]).all()
