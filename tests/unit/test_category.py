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


"""Tests for category analytics."""

import pandas as pd
import pytest

from src.analytics.category import (
    compute_category_kpis,
    compute_category_roles,
    compute_category_scorecard,
    infer_categories_nlp,
)
from src.analytics.schemas import CATEGORY_KPIS, CATEGORY_ROLES, CATEGORY_SCORECARD, INFERRED_CATEGORIES, check


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


def test_compute_category_roles_contract_and_classification(sample_df: pd.DataFrame) -> None:
    """Test that category roles are computed with correct contract and all 4 roles possible."""
    roles = compute_category_roles(sample_df)
    check(roles, CATEGORY_ROLES)
    assert len(roles) >= 1
    assert set(roles["role"].unique()) <= {"Destination", "Routine", "Seasonal", "Convenience"}
    # All signal columns should be present and in valid ranges
    assert roles["trip_generation_rate"].between(0, 1).all()
    assert (roles["demand_cv"] >= 0).all()
    assert (roles["seasonality_amplitude"] >= 0).all()
    assert roles["attachment_rate"].between(0, 1).all()
    # destination_categories column should be present
    assert "destination_categories" in roles.columns


def test_compute_category_roles_synthetic_fixture() -> None:
    """Test with a synthetic fixture designed to produce all 4 roles."""
    import numpy as np
    
    rng = np.random.default_rng(42)
    n_customers = 200
    n_days = 365  # Full year for proper seasonality
    
    categories = ["Destination_Cat", "Routine_Cat", "Seasonal_Cat", "Convenience_Cat"]
    products = []
    for i, cat in enumerate(categories):
        for j in range(5):
            products.append({
                "stockcode": f"SKU{i:02d}{j:02d}",
                "product": f"{cat} Prod{j}",
                "category": cat,
            })
    
    products_df = pd.DataFrame(products)
    
    # Generate transactions
    rows = []
    txn_id = 0
    days = pd.date_range("2024-01-01", periods=n_days, freq="D")
    
    # Destination: high trip generation, low CV, low seasonality - appears as dominant in many baskets EVERY day
    # Routine: moderate trip gen, low CV, low seasonality - consistent weekly
    # Seasonal: low trip gen, moderate CV, HIGH seasonality (summer only)
    # Convenience: low trip gen, high attachment to Destination - appears WITH Destination
    
    for day_idx, date in enumerate(days):
        for cust in range(n_customers):
            # Destination - appears in many baskets as dominant EVERY day
            if rng.random() < 0.25:
                txn_id += 1
                for cat in ["Destination_Cat"]:
                    for prod in products_df[products_df["category"] == cat]["product"].sample(2, random_state=rng).values:
                        rows.append((
                            date, f"TXN{txn_id}", f"SKU{cat[:3]}{rng.integers(0,5):02d}",
                            prod, f"CUST{cust:04d}",
                            rng.uniform(10, 20), rng.integers(1, 3),
                            cat, "Brand", "M", "Flavor", False, 5.0
                        ))
            
            # Routine - consistent weekly purchases (only on weekdays, consistent) - LOWER frequency
            if rng.random() < 0.08 and date.weekday() < 5:
                txn_id += 1
                for cat in ["Routine_Cat"]:
                    for prod in products_df[products_df["category"] == cat]["product"].sample(1, random_state=rng).values:
                        rows.append((
                            date, f"TXN{txn_id}", f"SKU{cat[:3]}{rng.integers(0,5):02d}",
                            prod, f"CUST{cust:04d}",
                            rng.uniform(5, 15), rng.integers(1, 2),
                            cat, "Brand", "M", "Flavor", False, 3.0
                        ))
            
            # Seasonal - ONLY in summer (June-August), very strong seasonal pattern
            is_summer = 150 < day_idx < 250
            seasonal_mult = 1.0 if is_summer else 0.01  # Almost zero in off-season
            if rng.random() < 0.2 * seasonal_mult:
                txn_id += 1
                for cat in ["Seasonal_Cat"]:
                    for prod in products_df[products_df["category"] == cat]["product"].sample(1, random_state=rng).values:
                        rows.append((
                            date, f"TXN{txn_id}", f"SKU{cat[:3]}{rng.integers(0,5):02d}",
                            prod, f"CUST{cust:04d}",
                            rng.uniform(8, 18), rng.integers(1, 2),
                            cat, "Brand", "M", "Flavor", False, 4.0
                        ))
            
            # Convenience - ONLY appears WITH Destination (high attachment)
            if rng.random() < 0.12:
                txn_id += 1
                # Add a Destination item first
                dest_prod = products_df[products_df["category"] == "Destination_Cat"]["product"].sample(1, random_state=rng).values[0]
                rows.append((
                    date, f"TXN{txn_id}", f"SKUDest{rng.integers(0,5):02d}",
                    dest_prod, f"CUST{cust:04d}",
                    rng.uniform(10, 20), rng.integers(1, 2),
                    "Destination_Cat", "Brand", "M", "Flavor", False, 5.0
                ))
                # Then add Convenience
                for prod in products_df[products_df["category"] == "Convenience_Cat"]["product"].sample(1, random_state=rng).values:
                    rows.append((
                        date, f"TXN{txn_id}", f"SKUConv{rng.integers(0,5):02d}",
                        prod, f"CUST{cust:04d}",
                        rng.uniform(2, 8), rng.integers(1, 2),
                        "Convenience_Cat", "Brand", "M", "Flavor", False, 1.5
                    ))
    
    df = pd.DataFrame(rows, columns=[
        "date", "transaction_id", "stockcode", "product",
        "customer_id", "price", "quantity", "category", "brand",
        "size", "flavor", "promo_flag", "cost"
    ])
    df["date"] = pd.to_datetime(df["date"])
    
    roles = compute_category_roles(df)
    check(roles, CATEGORY_ROLES)
    
    # Should have all 4 categories
    assert set(roles["category"]) == set(categories)
    
    # Print for debugging
    print("Roles:")
    print(roles[["category", "role", "trip_generation_rate", "demand_cv", "seasonality_amplitude", "attachment_rate"]].to_string())
    
    # Check roles are assigned (at least 3 of 4 roles should appear given the synthetic design)
    unique_roles = set(roles["role"].unique())
    assert len(unique_roles) >= 3, f"Expected at least 3 roles, got {unique_roles}"
    
    # Destination should have high trip_generation_rate and low demand_cv
    dest_row = roles[roles["category"] == "Destination_Cat"].iloc[0]
    assert dest_row["role"] == "Destination"
    assert dest_row["trip_generation_rate"] > 0.15
    assert dest_row["demand_cv"] < 0.25
    
    # Seasonal should have high seasonality
    seas_row = roles[roles["category"] == "Seasonal_Cat"].iloc[0]
    assert seas_row["role"] == "Seasonal"
    assert seas_row["seasonality_amplitude"] > 0.50
    
    # Convenience should have high attachment to Destination
    conv_row = roles[roles["category"] == "Convenience_Cat"].iloc[0]
    assert conv_row["role"] == "Convenience"
    assert conv_row["attachment_rate"] > 0.20
    
    # Routine should be the fallback
    routine_row = roles[roles["category"] == "Routine_Cat"].iloc[0]
    assert routine_row["role"] == "Routine"
