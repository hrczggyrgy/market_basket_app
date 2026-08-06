"""Tests for the next-product choice model."""

import pandas as pd
import pytest

from src.analytics.choice_model import build_customer_features, train_choice_model
from src.analytics.schemas import CUSTOMER_FEATURES, MODEL_METRICS, check


def test_build_customer_features_contract(sample_df: pd.DataFrame) -> None:
    features = build_customer_features(sample_df)
    check(features, CUSTOMER_FEATURES)
    if features.empty:
        return
    assert features["customer_id"].is_unique
    assert features["recency_days"].between(0, 500).all()
    assert (features["monetary"] >= 0).all()
    assert (features["n_baskets"] >= 1).all()
    assert features["favorite_category"].notna().all()


def test_build_customer_features_all_rows_have_target(sample_df: pd.DataFrame) -> None:
    features = build_customer_features(sample_df)
    if features.empty:
        return
    assert features["target_product"].notna().all()
    assert features["target_product"].isin(sample_df["stockcode"]).all()


def test_train_choice_model_returns_contracts(sample_df: pd.DataFrame) -> None:
    features = build_customer_features(sample_df)
    if len(features) < 20 or features["target_product"].nunique() < 2:
        pytest.skip("not enough labeled customers for training")
    metrics, importance, rules = train_choice_model(features)
    check(metrics, MODEL_METRICS)
    assert metrics.loc[metrics["metric"] == "accuracy", "value"].iloc[0] >= 0.0
    assert metrics.loc[metrics["metric"] == "accuracy", "value"].iloc[0] <= 1.0
    assert list(importance.columns) == ["feature", "importance"]
    assert list(rules.columns) == ["rule_index", "rule_path", "n_samples", "purity", "target_class"]
    assert (rules["purity"] > 0).all()
    assert (rules["purity"] <= 1.0).all()
    assert (rules["n_samples"] >= 1).all()


def test_train_choice_model_importance_sum_to_one(sample_df: pd.DataFrame) -> None:
    features = build_customer_features(sample_df)
    if len(features) < 20 or features["target_product"].nunique() < 2:
        pytest.skip("not enough labeled customers for training")
    _, importance, _ = train_choice_model(features)
    assert importance["importance"].sum() == pytest.approx(1.0, abs=1e-6)


def test_train_choice_model_insufficient_data() -> None:
    df = pd.DataFrame(columns=["stockcode", "customer_id", "price", "quantity", "transaction_id", "date", "category"])
    features = build_customer_features(df, prediction_window_days=7)
    assert features.empty
    metrics, importance, rules = train_choice_model(features)
    assert metrics.empty
    assert importance.empty
    assert rules.empty
