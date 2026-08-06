"""Tests for the seeded sample-data generator."""

import pandas as pd

from src.analytics.sample_data import THEMES, generate_transactions


def test_generate_transactions_schema() -> None:
    df = generate_transactions(n_customers=50, n_products=30, n_days=60, seed=1)
    expected = {
        "date",
        "transaction_id",
        "stockcode",
        "product",
        "customer_id",
        "price",
        "quantity",
        "category",
        "brand",
        "size",
        "flavor",
        "promo_flag",
        "cost",
    }
    assert set(df.columns) == expected
    assert df["price"].min() > 0
    assert df["quantity"].min() >= 1


def test_generate_transactions_deterministic() -> None:
    a = generate_transactions(n_customers=80, n_products=40, n_days=90, seed=42)
    b = generate_transactions(n_customers=80, n_products=40, n_days=90, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_generate_transactions_seed_changes_output() -> None:
    a = generate_transactions(n_customers=80, n_products=40, n_days=90, seed=1)
    b = generate_transactions(n_customers=80, n_products=40, n_days=90, seed=2)
    assert not a.equals(b)


def test_generate_transactions_has_promo_signal() -> None:
    df = generate_transactions(n_customers=120, n_products=60, n_days=180, seed=42)
    assert df["promo_flag"].sum() > 0
    promo_price = df.loc[df["promo_flag"], "price"]
    base_price = df.loc[~df["promo_flag"], "price"]
    assert promo_price.mean() < base_price.mean()


def test_generate_transactions_theme_copurchase_signal() -> None:
    df = generate_transactions(n_customers=120, n_products=60, n_days=180, seed=42)
    baskets = df.groupby("transaction_id")["product"].agg(set)
    n = len(baskets)
    lifts = []
    for theme in THEMES:
        tc = baskets.apply(lambda b: theme[0] in b and theme[1] in b).sum()
        p_a = baskets.apply(lambda b: theme[0] in b).mean()
        p_b = baskets.apply(lambda b: theme[1] in b).mean()
        if p_a * p_b > 0:
            lifts.append((tc / n) / (p_a * p_b))
    assert len(lifts) >= 8
    strong = sum(1 for lift in lifts if lift > 1.5)
    assert strong >= 6, f"only {strong}/10 themes show co-purchase lift > 1.5"


def test_write_sample_fixture(tmp_path) -> None:
    from src.analytics.sample_data import write_sample_fixture

    path = tmp_path / "fixture.csv"
    df = write_sample_fixture(path, n_customers=40, n_products=20, n_days=30, seed=3)
    assert path.exists()
    reread = pd.read_csv(path)
    assert len(reread) == len(df)
