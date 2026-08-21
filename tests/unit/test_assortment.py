"""Unit tests for assortment optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow
from src.analytics.assortment import (
    build_solution_table,
    compare_assortment_scenarios,
    evaluate_assortment,
    evaluate_selected_scenarios,
    optimize_assortment_heuristic,
    optimize_assortment_milp,
)
from src.analytics.schemas import (
    ASSORTMENT_EVALUATION,
    ASSORTMENT_SCENARIO,
    ASSORTMENT_SOLUTION,
)
from src.analytics.transference import compute_demand_transference_matrix


@pytest.fixture(scope="module")
def assortment_inputs(sample_df: pd.DataFrame) -> dict[str, object]:
    revenue = (sample_df["price"] * sample_df["quantity"]).groupby(sample_df["stockcode"]).sum()
    dt = compute_demand_transference_matrix(sample_df)
    return {"df": sample_df, "revenue": revenue, "dt": dt}


def test_milp_solution_contract_and_constraints(assortment_inputs: dict[str, object]) -> None:
    df = assortment_inputs["df"]  # type: ignore[assignment]
    revenue = assortment_inputs["revenue"]  # type: ignore[assignment]
    dt = assortment_inputs["dt"]  # type: ignore[assignment]
    selected, metrics = optimize_assortment_milp(
        df, revenue, dt, max_skus=20, min_coverage=0.5, time_limit_seconds=20
    )
    assert len(selected) <= 20
    assert metrics["solver_status"] in {"optimal", "iteration_limit"}
    total = float(revenue.sum())
    kept = float(revenue.loc[[p for p in selected if p in revenue.index]].sum())
    assert kept / total >= 0.45


def test_milp_accepts_none_transference(assortment_inputs: dict[str, object]) -> None:
    df = assortment_inputs["df"]  # type: ignore[assignment]
    selected, _ = optimize_assortment_milp(df, max_skus=15, time_limit_seconds=20)
    assert 0 < len(selected) <= 15


def test_milp_margin_objective(assortment_inputs: dict[str, object]) -> None:
    df = assortment_inputs["df"]  # type: ignore[assignment]
    revenue = assortment_inputs["revenue"]  # type: ignore[assignment]
    cost = revenue * 0.6
    selected, _ = optimize_assortment_milp(
        df,
        revenue,
        objective="margin",
        cost_per_product=cost,
        max_skus=20,
        min_coverage=0.5,
        time_limit_seconds=20,
    )
    assert len(selected) > 0


@pytest.mark.slow
def test_heuristic_reaches_max_skus(assortment_inputs: dict[str, object]) -> None:
    df = assortment_inputs["df"]  # type: ignore[assignment]
    revenue = assortment_inputs["revenue"]  # type: ignore[assignment]
    dt = assortment_inputs["dt"]  # type: ignore[assignment]
    selected, metrics = optimize_assortment_heuristic(
        df, revenue, dt, max_skus=20, min_coverage=0.5, iterations=60, random_seed=7
    )
    assert len(selected) == 20
    assert metrics["n_skus"] == 20
    assert 0 <= metrics["coverage"] <= 1


def test_evaluate_assortment_metrics(assortment_inputs: dict[str, object]) -> None:
    df = assortment_inputs["df"]  # type: ignore[assignment]
    revenue = assortment_inputs["revenue"]  # type: ignore[assignment]
    dt = assortment_inputs["dt"]  # type: ignore[assignment]
    top20 = revenue.head(20).index.tolist()
    metrics = evaluate_assortment(top20, df, dt, revenue)
    assert metrics["n_skus"] == 20
    assert metrics["kept_revenue"] > 0
    assert metrics["recovered_revenue"] >= 0
    assert metrics["unmet_demand"] >= 0
    assert (
        abs(metrics["expected_revenue"] - (metrics["kept_revenue"] + metrics["recovered_revenue"]))
        < 1e-6
    )
    assert 0 <= metrics["coverage"] <= 1
    assert metrics["n_categories_total"] > 0


def test_compare_scenarios_contract(assortment_inputs: dict[str, object]) -> None:
    df = assortment_inputs["df"]  # type: ignore[assignment]
    table = compare_assortment_scenarios(
        df, [], n_scenarios=4, max_skus_range=(15, 30), random_seed=3
    )
    ASSORTMENT_SCENARIO.validate(table)
    assert set(table["method"]) <= {"greedy", "random", "milp"}
    assert table["n_skus"].ge(1).all()
    assert table["coverage"].between(0, 1).all()


def test_build_solution_table(assortment_inputs: dict[str, object]) -> None:
    revenue = assortment_inputs["revenue"]  # type: ignore[assignment]
    table = build_solution_table(revenue.head(10).index.tolist(), revenue)
    ASSORTMENT_SOLUTION.validate(table)
    assert table["rank"].tolist() == list(range(1, 11))
    assert table["revenue"].is_monotonic_decreasing


def test_evaluate_selected_scenarios(assortment_inputs: dict[str, object]) -> None:
    df = assortment_inputs["df"]  # type: ignore[assignment]
    revenue = assortment_inputs["revenue"]  # type: ignore[assignment]
    scenarios = compare_assortment_scenarios(
        df, [], n_scenarios=3, max_skus_range=(15, 25), random_seed=1
    )
    mapping = {
        int(row.scenario_id): revenue.head(int(row.n_skus)).index.tolist()
        for row in scenarios.itertuples()
    }
    table = evaluate_selected_scenarios(
        [int(s) for s in scenarios["scenario_id"]], scenarios, mapping, df
    )
    ASSORTMENT_EVALUATION.validate(table)
    assert len(table) == len(scenarios)
    assert table["selected_skus"].gt(0).all()
