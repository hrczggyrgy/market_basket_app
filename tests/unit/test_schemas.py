"""Tests for the output-contract validation machinery (architecture rule 2)."""

import pandas as pd
import pytest

from src.analytics.schemas import (
    DataContract,
    RULES,
    TRANSACTIONS,
    SchemaError,
    check,
    contract,
)


def test_contract_validate_passes() -> None:
    c = DataContract(name="t", columns=("a", "b"))
    df = pd.DataFrame({"a": [1], "b": [2], "extra": [3]})
    c.validate(df)


def test_contract_validate_missing_column() -> None:
    c = DataContract(name="t", columns=("a", "b"))
    with pytest.raises(SchemaError):
        c.validate(pd.DataFrame({"a": [1]}))


def test_contract_validate_not_dataframe() -> None:
    with pytest.raises(SchemaError):
        TRANSACTIONS.validate("not a df")


def test_contract_validate_empty_rejected() -> None:
    with pytest.raises(SchemaError):
        TRANSACTIONS.validate(pd.DataFrame(columns=TRANSACTIONS.columns))


def test_contract_validate_empty_allowed() -> None:
    TRANSACTIONS.validate(pd.DataFrame(columns=TRANSACTIONS.columns), allow_empty=True)


def test_check_returns_df() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert check(df, contract("a", "b")) is df


def test_check_raises_on_bad_output() -> None:
    with pytest.raises(SchemaError):
        check(pd.DataFrame({"a": [1]}), contract("a", "b"))


def test_contract_helper() -> None:
    c = contract("x", "y")
    assert c.columns == ("x", "y")


def test_transactions_contract_matches_required_schema() -> None:
    assert TRANSACTIONS.columns == (
        "date",
        "transaction_id",
        "stockcode",
        "product",
        "customer_id",
        "price",
        "quantity",
    )


def test_rules_contract_declared() -> None:
    df = pd.DataFrame(
        {
            "antecedents": [{1}],
            "consequents": [{2}],
            "antecedent support": [0.5],
            "consequent support": [0.4],
            "support": [0.3],
            "confidence": [0.6],
            "lift": [1.5],
            "lift_ci_lower": [1.1],
            "lift_ci_upper": [1.9],
            "leverage": [0.1],
            "conviction": [1.2],
            "zhangs_metric": [0.8],
            "q_value": [0.95],
            "is_redundant": [False],
        }
    )
    RULES.validate(df)


def test_rules_contract_rejects_out_of_range_support() -> None:
    df = pd.DataFrame(
        {
            "antecedents": [{1}],
            "consequents": [{2}],
            "antecedent support": [0.5],
            "consequent support": [0.4],
            "support": [1.5],
            "confidence": [0.6],
            "lift": [1.5],
            "lift_ci_lower": [1.1],
            "lift_ci_upper": [1.9],
            "leverage": [0.1],
            "conviction": [1.2],
            "zhangs_metric": [0.8],
            "q_value": [0.95],
            "is_redundant": [False],
        }
    )
    with pytest.raises(SchemaError):
        RULES.validate(df)


def test_empty_result_sentinel() -> None:
    from src.analytics.schemas import EmptyResult, is_empty_result, make_empty_result

    er = make_empty_result(TRANSACTIONS, reason="no rows")
    assert is_empty_result(er)
    assert er.reason == "no rows"
    assert not er


def test_value_validator_warning_severity() -> None:
    from src.analytics.schemas import ValueValidator

    c = DataContract(
        name="t",
        columns=("a",),
        validators=(ValueValidator("a", lambda s: s >= 0, "a must be non-negative", severity="warning"),),
    )
    df, warnings = c.validate(pd.DataFrame({"a": [-1]}))
    assert warnings and "a must be non-negative" in warnings[0]
