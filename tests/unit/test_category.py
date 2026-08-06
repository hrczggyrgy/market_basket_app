"""Tests for category analytics."""

import pandas as pd
import pytest

from src.analytics.category import (
    compute_category_kpis,
    compute_category_scorecard,
    infer_categories_nlp,
)
from src.analytics.schemas import CATEGORY_KPIS, CATEGORY_SCORECARD, INFERRED_CATEGORIES, check


def test_category_kpis_contract_and_totals(sample_df: pd.DataFrame) -> None:
    table = compute_category_kpis(sample_df)
    check(table, CATEGORY_KPIS)
    assert len(table) >= 1
    assert (table["revenue"] > 0).all()
    assert table["penetration"].between(0, 1).all()
    assert table["revenue_share"].sum() == pytest.approx(1.0, rel=1e-9)


def test_category_kpis_empty_input() -> None:
    table = compute_category_kpis(pd.DataFrame(columns=["date", "price", "quantity", "transaction_id", "customer_id", "category"]))
    assert table.empty


def test_category_scorecard_contract(sample_df: pd.DataFrame) -> None:
    table = compute_category_scorecard(sample_df)
    check(table, CATEGORY_SCORECARD)
    assert set(table["rag"].unique()) <= {"green", "amber", "red"}
    assert set(table["role"].unique()) <= {"growth", "parity", "traffic_driver", "niche"}


def test_category_scorecard_rows_match_kpis(sample_df: pd.DataFrame) -> None:
    scorecard = compute_category_scorecard(sample_df)
    kpis = compute_category_kpis(sample_df)
    assert set(scorecard["category"]) == set(kpis["category"])
    assert (scorecard["revenue"].values == kpis["revenue"].values).all()


def test_infer_categories_nlp_contract(sample_df: pd.DataFrame) -> None:
    table = infer_categories_nlp(sample_df, n_categories=6)
    check(table, INFERRED_CATEGORIES)
    assert table["stockcode"].nunique() == len(table)
    assert table["inferred_category"].nunique() >= 1


def test_infer_categories_nlp_missing_column() -> None:
    table = infer_categories_nlp(pd.DataFrame({"stockcode": ["A"]}), n_categories=3)
    assert table.empty
