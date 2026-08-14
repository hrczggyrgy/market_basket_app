"""End-to-end smoke tests against the real-life dataset (UCI Online Retail).

The file is not committed; these tests skip when it is absent. They exercise
the real analytics entry points on messier, real-world data to catch
assumptions broken by the synthetic fixture.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analytics.basket_metrics import (
    compute_basket_penetration,
    compute_customer_entropy,
)
from src.analytics.clv import predict_clv_bg_nbd
from src.analytics.cohort import compute_cohorts
from src.analytics.copurchase import get_top_affinity_pairs
from src.analytics.data import (
    build_dataset_capabilities,
    derive_product_lookup,
    get_data_summary,
    load_transactions,
)
from src.analytics.rules import (
    create_basket_matrix,
    filter_rules,
    generate_rules,
    rules_to_table,
    run_fpgrowth,
)
from src.analytics.schemas import (
    AFFINITY_PAIRS,
    BASKET_PENETRATION,
    CLV_DIAGNOSTICS,
    CLV_PREDICTIONS,
    COHORT_RETENTION,
    CUSTOMER_ENTROPY,
    FREQUENT_ITEMSETS,
    RULES,
    RULES_TABLE,
)
from src.analytics.switching import get_customer_loyalty_metrics

REAL_DATA = Path(__file__).resolve().parents[2] / "transactions2.csv"

pytestmark = pytest.mark.skipif(
    not os.path.exists(REAL_DATA), reason="real dataset transactions2.csv not present"
)


@pytest.fixture(scope="module")
def real_df() -> pd.DataFrame:
    df, warning, dropped, _ = load_transactions(REAL_DATA)
    assert not df.empty
    return df


def test_loader_clean_ids(real_df: pd.DataFrame) -> None:
    assert not real_df["stockcode"].str.contains(r"\.0$").any()
    assert not real_df["customer_id"].str.contains(r"\.0$").any()


def test_summary_and_capabilities(real_df: pd.DataFrame) -> None:
    summary = get_data_summary(real_df)
    assert summary["n_transactions"] > 1000
    assert summary["total_revenue"] > 0
    caps = build_dataset_capabilities(real_df)
    assert not any(caps.values()), "real subset has no optional columns"


def test_rules_pipeline(real_df: pd.DataFrame) -> None:
    basket = create_basket_matrix(real_df)
    freq = run_fpgrowth(basket, min_support=0.01, max_len=3)
    FREQUENT_ITEMSETS.validate(freq)
    rules = generate_rules(freq, min_threshold=0.05)
    RULES.validate(rules)
    assert len(rules) > 0
    filtered = filter_rules(rules, min_lift=1.0, min_confidence=0.05)
    RULES.validate(filtered, allow_empty=True)
    lookup = derive_product_lookup(real_df)
    table = rules_to_table(filtered, lookup)
    RULES_TABLE.validate(table, allow_empty=True)


def test_copurchase(real_df: pd.DataFrame) -> None:
    pairs = get_top_affinity_pairs(real_df, top_n=10, min_cooccurrence=20)
    AFFINITY_PAIRS.validate(pairs)


def test_switching_loyalty(real_df: pd.DataFrame) -> None:
    metrics = get_customer_loyalty_metrics(real_df)
    assert not metrics.empty


def test_basket_metrics(real_df: pd.DataFrame) -> None:
    BASKET_PENETRATION.validate(compute_basket_penetration(real_df))
    CUSTOMER_ENTROPY.validate(compute_customer_entropy(real_df))


def test_cohorts(real_df: pd.DataFrame) -> None:
    table = compute_cohorts(real_df, cohort_period="W")
    COHORT_RETENTION.validate(table)


def test_clv_bg_nbd(real_df: pd.DataFrame) -> None:
    predictions, diagnostics = predict_clv_bg_nbd(real_df, prediction_horizon_days=90)
    CLV_PREDICTIONS.validate(predictions)
    CLV_DIAGNOSTICS.validate(diagnostics)
    assert len(predictions) > 100
    assert predictions["p_alive"].between(0, 1).all()
    assert (predictions["predicted_clv"] >= 0).all()
