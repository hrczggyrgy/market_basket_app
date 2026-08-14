"""Tests for category analytics."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.category import (
    compute_assortment_efficiency,
    compute_category_growth_matrix,
    compute_category_kpis,
    compute_category_manager_scorecard,
    compute_category_scorecard,
    compute_category_trend,
    enrich_with_categories,
    infer_categories_nlp,
)
from src.analytics.schemas import (
    ASSORTMENT_EFFICIENCY,
    CATEGORY_GROWTH_MATRIX,
    CATEGORY_KPIS,
    CATEGORY_SCORECARD,
    CATEGORY_TREND,
    INFERRED_CATEGORIES,
    check,
)


def test_category_kpis_contract_and_totals(sample_df: pd.DataFrame) -> None:
    table = compute_category_kpis(sample_df)
    check(table, CATEGORY_KPIS)
    assert len(table) >= 1
    assert (table["revenue"] > 0).all()
    assert table["penetration"].between(0, 1).all()
    assert table["revenue_share"].sum() == pytest.approx(1.0, rel=1e-9)


def test_category_kpis_empty_input() -> None:
    table = compute_category_kpis(
        pd.DataFrame(
            columns=["date", "price", "quantity", "transaction_id", "customer_id", "category"]
        )
    )
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
from src.analytics.schemas import (
    CATEGORY_KPIS,
    CATEGORY_MANAGER_SCORECARD,
    CATEGORY_ROLES,
    CATEGORY_SCORECARD,
    INFERRED_CATEGORIES,
    check,
)


def test_category_kpis_contract_and_totals(sample_df: pd.DataFrame) -> None:
    table = compute_category_kpis(sample_df)
    check(table, CATEGORY_KPIS)
    assert len(table) >= 1
    assert (table["revenue"] > 0).all()
    assert table["penetration"].between(0, 1).all()
    assert table["revenue_share"].sum() == pytest.approx(1.0, rel=1e-9)


def test_category_kpis_empty_input() -> None:
    table = compute_category_kpis(
        pd.DataFrame(
            columns=["date", "price", "quantity", "transaction_id", "customer_id", "category"]
        )
    )
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


def test_enrich_with_categories_keeps_existing(sample_df: pd.DataFrame) -> None:
    df, was_inferred = enrich_with_categories(sample_df)
    assert not was_inferred
    assert "category" in df.columns
    assert (df["category"] == sample_df["category"]).all()


def test_enrich_with_categories_infers_when_missing(sample_df: pd.DataFrame) -> None:
    no_cat = sample_df.drop(columns=["category"])
    df, was_inferred = enrich_with_categories(no_cat, n_categories=4)
    assert was_inferred
    assert "category" in df.columns
    assert df["category"].notna().all()
    assert df["category"].nunique() >= 1
    assert set(df["stockcode"]) == set(sample_df["stockcode"])


def test_enrich_with_categories_empty_or_no_product() -> None:
    df, was_inferred = enrich_with_categories(pd.DataFrame(), n_categories=3)
    assert not was_inferred
    df2, was2 = enrich_with_categories(
        pd.DataFrame({"stockcode": ["A"], "category": ["X"]}), n_categories=3
    )
    assert not was2
    assert "category" in df2.columns


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
    n_days = 730  # Two full years: seasonality requires >= 2 annual cycles

    categories = ["Destination_Cat", "Routine_Cat", "Seasonal_Cat", "Convenience_Cat"]
    products = []
    for i, cat in enumerate(categories):
        for j in range(5):
            products.append(
                {
                    "stockcode": f"SKU{i:02d}{j:02d}",
                    "product": f"{cat} Prod{j}",
                    "category": cat,
                }
            )

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
        day_of_year = day_idx % 365  # seasonal pattern repeats across years
        for cust in range(n_customers):
            # Destination - appears in many baskets as dominant EVERY day
            if rng.random() < 0.25:
                txn_id += 1
                for cat in ["Destination_Cat"]:
                    for prod in (
                        products_df[products_df["category"] == cat]["product"]
                        .sample(2, random_state=rng)
                        .values
                    ):
                        rows.append(
                            (
                                date,
                                f"TXN{txn_id}",
                                f"SKU{cat[:3]}{rng.integers(0, 5):02d}",
                                prod,
                                f"CUST{cust:04d}",
                                rng.uniform(10, 20),
                                rng.integers(1, 3),
                                cat,
                                "Brand",
                                "M",
                                "Flavor",
                                False,
                                5.0,
                            )
                        )

            # Routine - consistent weekly purchases (only on weekdays, consistent) - LOWER frequency
            if rng.random() < 0.08 and date.weekday() < 5:
                txn_id += 1
                for cat in ["Routine_Cat"]:
                    for prod in (
                        products_df[products_df["category"] == cat]["product"]
                        .sample(1, random_state=rng)
                        .values
                    ):
                        rows.append(
                            (
                                date,
                                f"TXN{txn_id}",
                                f"SKU{cat[:3]}{rng.integers(0, 5):02d}",
                                prod,
                                f"CUST{cust:04d}",
                                rng.uniform(5, 15),
                                rng.integers(1, 2),
                                cat,
                                "Brand",
                                "M",
                                "Flavor",
                                False,
                                3.0,
                            )
                        )

            # Seasonal - ONLY in summer (June-August), very strong seasonal pattern
            is_summer = 150 < day_of_year < 250
            seasonal_mult = 1.0 if is_summer else 0.01  # Almost zero in off-season
            if rng.random() < 0.2 * seasonal_mult:
                txn_id += 1
                for cat in ["Seasonal_Cat"]:
                    for prod in (
                        products_df[products_df["category"] == cat]["product"]
                        .sample(1, random_state=rng)
                        .values
                    ):
                        rows.append(
                            (
                                date,
                                f"TXN{txn_id}",
                                f"SKU{cat[:3]}{rng.integers(0, 5):02d}",
                                prod,
                                f"CUST{cust:04d}",
                                rng.uniform(8, 18),
                                rng.integers(1, 2),
                                cat,
                                "Brand",
                                "M",
                                "Flavor",
                                False,
                                4.0,
                            )
                        )

            # Convenience - ONLY appears WITH Destination (high attachment)
            if rng.random() < 0.12:
                txn_id += 1
                # Add a Destination item first
                dest_prod = (
                    products_df[products_df["category"] == "Destination_Cat"]["product"]
                    .sample(1, random_state=rng)
                    .values[0]
                )
                rows.append(
                    (
                        date,
                        f"TXN{txn_id}",
                        f"SKUDest{rng.integers(0, 5):02d}",
                        dest_prod,
                        f"CUST{cust:04d}",
                        rng.uniform(10, 20),
                        rng.integers(1, 2),
                        "Destination_Cat",
                        "Brand",
                        "M",
                        "Flavor",
                        False,
                        5.0,
                    )
                )
                # Then add Convenience
                for prod in (
                    products_df[products_df["category"] == "Convenience_Cat"]["product"]
                    .sample(1, random_state=rng)
                    .values
                ):
                    rows.append(
                        (
                            date,
                            f"TXN{txn_id}",
                            f"SKUConv{rng.integers(0, 5):02d}",
                            prod,
                            f"CUST{cust:04d}",
                            rng.uniform(2, 8),
                            rng.integers(1, 2),
                            "Convenience_Cat",
                            "Brand",
                            "M",
                            "Flavor",
                            False,
                            1.5,
                        )
                    )

    df = pd.DataFrame(
        rows,
        columns=[
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
        ],
    )
    df["date"] = pd.to_datetime(df["date"])

    roles = compute_category_roles(df)
    check(roles, CATEGORY_ROLES)

    # Should have all 4 categories
    assert set(roles["category"]) == set(categories)

    # Print for debugging
    print("Roles:")
    print(
        roles[
            [
                "category",
                "role",
                "trip_generation_rate",
                "demand_cv",
                "seasonality_amplitude",
                "attachment_rate",
            ]
        ].to_string()
    )

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


def _tiny_two_role_fixture() -> pd.DataFrame:
    """Compact fixture: CatA strongly seasonal, CatB steady all year -> 2 roles."""
    import numpy as np

    rng = np.random.default_rng(11)
    days = pd.date_range("2024-01-01", periods=180, freq="D")
    rows = []
    for d in days:
        if 60 <= d.dayofyear <= 240:
            for _ in range(10):
                rows.append(
                    (
                        d,
                        f"T{rng.integers(0, 99999)}",
                        f"A{rng.integers(0, 5)}",
                        "a",
                        f"C{rng.integers(0, 30)}",
                        round(rng.uniform(3, 9), 2),
                        1,
                        "CatA",
                    )
                )
        if rng.random() < 0.18:
            rows.append(
                (
                    d,
                    f"T{rng.integers(0, 99999)}",
                    f"B{rng.integers(0, 5)}",
                    "b",
                    f"C{rng.integers(0, 30)}",
                    round(rng.uniform(3, 9), 2),
                    1,
                    "CatB",
                )
            )
    df = pd.DataFrame(
        rows,
        columns=[
            "date",
            "transaction_id",
            "stockcode",
            "product",
            "customer_id",
            "price",
            "quantity",
            "category",
        ],
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_manager_scorecard_contract_and_coverage() -> None:
    """All categories appear; scorecard satisfies CATEGORY_MANAGER_SCORECARD."""
    df = _tiny_two_role_fixture()
    sc = compute_category_manager_scorecard(df)
    check(sc, CATEGORY_MANAGER_SCORECARD)
    assert set(sc["category"]) == set(df["category"])
    # shares are well-formed
    assert sc["revenue_share"].sum() == pytest.approx(1.0, rel=1e-6)
    assert sc["sku_share"].sum() == pytest.approx(1.0, rel=1e-6)
    assert sc["basket_penetration"].between(0, 1).all()
    assert sc["repeat_purchase_rate"].between(0, 1).all()


def test_manager_scorecard_two_roles_present() -> None:
    """Configured fixture produces at least two distinct roles."""
    df = _tiny_two_role_fixture()
    sc = compute_category_manager_scorecard(df)
    assert len(set(sc["role"])) >= 2


def test_manager_scorecard_empty_input() -> None:
    sc = compute_category_manager_scorecard(pd.DataFrame(columns=["date", "price", "quantity"]))
    assert sc.empty
    check(sc, CATEGORY_MANAGER_SCORECARD, allow_empty=True)


def test_manager_scorecard_sample_fixture(sample_df: pd.DataFrame) -> None:
    """Runs on the real sample fixture without exceptions; totals balance."""
    sc = compute_category_manager_scorecard(sample_df)
    check(sc, CATEGORY_MANAGER_SCORECARD)
    assert not sc.empty
    assert sc["revenue_share"].sum() == pytest.approx(1.0, rel=1e-6)
    assert sc["sku_share"].sum() == pytest.approx(1.0, rel=1e-6)
    assert sc["kvi_count"].sum() >= 1


def test_category_trend_contract_and_penetration(sample_df: pd.DataFrame) -> None:
    """Weekly trend per category; penetration bounded; revenue positive."""
    trend = compute_category_trend(sample_df)
    check(trend, CATEGORY_TREND)
    assert not trend.empty
    assert set(trend["category"]) == set(sample_df["category"])
    assert trend["basket_penetration"].between(0, 1).all()
    assert (trend["revenue"] >= 0).all()
    # each category has its own weekly series
    assert trend.groupby("category")["period"].nunique().min() >= 2


def test_category_trend_empty_input() -> None:
    trend = compute_category_trend(
        pd.DataFrame(columns=["date", "price", "quantity", "transaction_id", "category"])
    )
    assert trend.empty
    check(trend, CATEGORY_TREND, allow_empty=True)


def test_category_trend_penetration_uses_baskets() -> None:
    """Penetration = share of baskets containing the category in that period."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02"]),
            "transaction_id": ["T1", "T2", "T2"],
            "category": ["Cat A", "Cat A", "Cat B"],
            "price": [1.0, 2.0, 3.0],
            "quantity": [1, 1, 1],
            "customer_id": ["c1", "c2", "c2"],
        }
    )
    trend = compute_category_trend(df)
    row_a = trend[trend["category"] == "Cat A"].iloc[0]
    row_b = trend[trend["category"] == "Cat B"].iloc[0]
    # 2 baskets total that week: A in both, B in one
    assert row_a["basket_penetration"] == pytest.approx(1.0)
    assert row_b["basket_penetration"] == pytest.approx(0.5)
    assert row_a["revenue"] == pytest.approx(3.0)
    assert row_b["revenue"] == pytest.approx(3.0)


def test_assortment_efficiency_contract_and_totals(sample_df: pd.DataFrame) -> None:
    """Shares sum to 1; index matches revenue_share / sku_share."""
    eff = compute_assortment_efficiency(sample_df)
    check(eff, ASSORTMENT_EFFICIENCY)
    assert not eff.empty
    assert eff["sku_share"].sum() == pytest.approx(1.0, rel=1e-6)
    assert eff["revenue_share"].sum() == pytest.approx(1.0, rel=1e-6)
    idx = (eff["revenue_share"] / eff["sku_share"]).where(eff["sku_share"] > 0, 0.0)
    assert np.allclose(eff["efficiency_index"], idx, rtol=1e-4)


def test_assortment_efficiency_labels_sanity(sample_df: pd.DataFrame) -> None:
    """Labels are consistent with the index thresholds."""
    eff = compute_assortment_efficiency(sample_df)
    labels = eff["efficiency_label"].unique()
    assert set(labels) <= {"efficient", "balanced", "under_efficient"}
    for _, row in eff.iterrows():
        idx = row["efficiency_index"]
        if idx > 1.1:
            assert row["efficiency_label"] == "efficient"
        elif idx < 0.9:
            assert row["efficiency_label"] == "under_efficient"
        else:
            assert row["efficiency_label"] == "balanced"


def test_assortment_efficiency_empty_input() -> None:
    eff = compute_assortment_efficiency(
        pd.DataFrame(columns=["date", "price", "quantity", "stockcode"])
    )
    assert eff.empty
    check(eff, ASSORTMENT_EFFICIENCY, allow_empty=True)


def test_growth_matrix_contract_and_quadrants(sample_df: pd.DataFrame) -> None:
    gm = compute_category_growth_matrix(sample_df)
    check(gm, CATEGORY_GROWTH_MATRIX)
    assert not gm.empty
    assert set(gm["quadrant"].unique()) <= {"star", "cash_cow", "question_mark", "dog"}
    assert gm["revenue_share"].between(0, 1).all()


def test_growth_matrix_median_split() -> None:
    """Synthetic data across several weeks exercises the four median-split quadrants."""
    cats_profiles = [
        # (name, weekly_base, weekly_growth)  -> revenue share and growth spread
        ("Cat0", 50, 5),  # high base, rising  -> star-ish
        ("Cat1", 40, 4),  # high base, rising
        ("Cat2", 60, -2),  # high base, falling -> cash cow
        ("Cat3", 45, -3),  # high base, falling
        ("Cat4", 8, 6),  # low base, rising   -> question mark
        ("Cat5", 6, 5),  # low base, rising
        ("Cat6", 7, -4),  # low base, falling  -> dog
        ("Cat7", 5, -5),  # low base, falling
    ]
    weeks = pd.date_range("2024-01-01", periods=26, freq="W")
    tx_id = 0
    rows = []
    for name, base, weekly_growth in cats_profiles:
        for w_i, week in enumerate(weeks):
            week_qty = base + weekly_growth * w_i
            if week_qty <= 0:
                continue
            n_tx = max(1, int(round(week_qty / 8)))
            qty = max(1, int(round(week_qty / n_tx)))
            for _ in range(n_tx):
                rows.append(
                    {
                        "date": week,
                        "transaction_id": f"T{tx_id}",
                        "customer_id": f"C{tx_id % 5}",
                        "stockcode": f"{name}S{tx_id % 3}",
                        "category": name,
                        "price": 1.0,
                        "quantity": qty,
                    }
                )
                tx_id += 1
    df = pd.DataFrame(rows)
    gm = compute_category_growth_matrix(df)
    check(gm, CATEGORY_GROWTH_MATRIX)
    assert not gm.empty
    assert gm["quadrant"].nunique() >= 2
    # median-split correctness
    smed = gm["revenue_share"].median()
    gmed = gm["growth_pct"].median()
    for _, row in gm.iterrows():
        expected = (
            "star"
            if row["revenue_share"] >= smed and row["growth_pct"] >= gmed
            else "cash_cow"
            if row["revenue_share"] >= smed
            else "question_mark"
            if row["growth_pct"] >= gmed
            else "dog"
        )
        assert row["quadrant"] == expected


def test_growth_matrix_empty_input() -> None:
    gm = compute_category_growth_matrix(
        pd.DataFrame(columns=["date", "price", "quantity", "stockcode"])
    )
    assert gm.empty
    check(gm, CATEGORY_GROWTH_MATRIX, allow_empty=True)
