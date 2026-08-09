"""Assortment optimization: MILP and heuristic range planning.

Selects a subset of SKUs maximizing kept revenue plus demand-recovery from
delisted products (via :mod:`src.analytics.transference`), subject to a max
SKU count, a minimum revenue-coverage, and at least one SKU per category.

The MILP formulates the bilinear recovery term ``(1 - x_i) * x_j`` with the
standard upper-bound relaxation, which is exact for positive weights: for
each transference edge (i -> j) a variable ``z_ij`` is constrained by
``z <= x_j`` and ``z <= 1 - x_i``, and the solver is free to set ``z = 1``
only when ``x_i = 0`` and ``x_j = 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from src.analytics.data import revenue_column
from src.analytics.schemas import (
    ASSORTMENT_EVALUATION,
    ASSORTMENT_SCENARIO,
    ASSORTMENT_SOLUTION,
    check,
)
from src.analytics.transference import compute_demand_transference_matrix

DEFAULT_RECOVERY_MARGIN = 0.30
MILP_STATUS = {0: "optimal", 1: "iteration_limit", 2: "infeasible", 3: "unbounded"}


@dataclass(frozen=True)
class AssortmentMetrics:
    """Evaluation metrics of a candidate assortment."""

    kept_revenue: float
    recovered_revenue: float
    lost_revenue: float
    unmet_demand: float
    expected_revenue: float
    coverage: float
    recovery_rate: float
    n_skus: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "kept_revenue": self.kept_revenue,
            "recovered_revenue": self.recovered_revenue,
            "lost_revenue": self.lost_revenue,
            "unmet_demand": self.unmet_demand,
            "expected_revenue": self.expected_revenue,
            "coverage": self.coverage,
            "recovery_rate": self.recovery_rate,
            "n_skus": self.n_skus,
        }


def _revenue_series(df: pd.DataFrame) -> pd.Series:
    """Per-SKU revenue series (sums duplicated index entries if any)."""
    revenue = revenue_column(df).groupby(df["stockcode"]).sum()
    return revenue.sort_values(ascending=False)


def _transfers_by_from(dt_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group transference edges by their from-product."""
    return {frm: g for frm, g in dt_df.groupby("from_product")}


def _evaluate_solution(
    kept: set[str],
    revenue_per_product: pd.Series,
    transfers: dict[str, pd.DataFrame],
) -> AssortmentMetrics:
    """Score a kept set: kept revenue + recovery toward kept SKUs."""
    present = [p for p in sorted(kept) if p in revenue_per_product.index]
    kept_revenue = float(revenue_per_product.loc[present].sum())
    total_revenue = float(revenue_per_product.sum())

    lost = 0.0
    recovered = 0.0
    for prod, rev in revenue_per_product.items():
        if prod in kept:
            continue
        lost += float(rev)
        edges = transfers.get(prod)
        if edges is None:
            continue
        in_kept = edges["to_product"].isin(kept)
        if in_kept.any():
            recovered += float(edges.loc[in_kept, "observed_switching_recovery_proxy"].sum())

    unmet = lost - recovered
    expected = kept_revenue + recovered
    coverage = min(1.0, expected / total_revenue) if total_revenue > 0 else 0.0
    return AssortmentMetrics(
        kept_revenue=kept_revenue,
        recovered_revenue=recovered,
        lost_revenue=lost,
        unmet_demand=unmet,
        expected_revenue=expected,
        coverage=coverage,
        recovery_rate=min(1.0, recovered / lost) if lost > 0 else 0.0,
        n_skus=len(kept),
    )


def _category_of(df: pd.DataFrame) -> dict[str, str]:
    if "category" not in df.columns:
        return {}
    return dict(df.drop_duplicates("stockcode").set_index("stockcode")["category"])


def optimize_assortment_milp(
    transactions_df: pd.DataFrame,
    revenue_per_product: pd.Series | None = None,
    demand_transference_df: pd.DataFrame | None = None,
    *,
    max_skus: int = 100,
    min_coverage: float = 0.80,
    objective: str = "revenue",
    cost_per_product: pd.Series | None = None,
    top_n: int = 200,
    time_limit_seconds: int = 60,
    recovery_margin: float = DEFAULT_RECOVERY_MARGIN,
) -> tuple[list[str], dict[str, object]]:
    """Select SKUs maximizing revenue (or margin) + recovery via scipy MILP.

    Decision variables ``x_i`` are binary. The objective coefficients are

        revenue_i + recovery_margin * sum_j revenue_at_risk(i -> j)

    when ``objective == "margin"`` the direct term is replaced by
    ``revenue_i - cost_i`` and the recovery term is scaled by
    ``recovery_margin``. Constraints: ``sum x <= max_skus``,
    ``sum revenue_i x_i >= min_coverage * total``, and at least one SKU per
    category. Only the top ``top_n`` products by revenue are candidates.
    """
    if revenue_per_product is None:
        revenue_per_product = _revenue_series(transactions_df)
    revenue = revenue_per_product.head(top_n)

    if demand_transference_df is None:
        demand_transference_df = compute_demand_transference_matrix(
            transactions_df, top_n=top_n
        )
    if demand_transference_df is None or demand_transference_df.empty:
        dt_edges = pd.DataFrame(columns=["from_product", "to_product", "observed_switching_recovery_proxy"])
    else:
        dt_edges = demand_transference_df[
            ["from_product", "to_product", "observed_switching_recovery_proxy"]
        ]
    dt_edges = dt_edges[dt_edges["from_product"].isin(revenue.index) & dt_edges["to_product"].isin(revenue.index)]

    products = list(revenue.index)
    idx = {p: i for i, p in enumerate(products)}
    n = len(products)

    direct = revenue.values.astype(float).copy()
    if objective == "margin" and cost_per_product is not None:
        margin = revenue.values.astype(float) - np.asarray(
            [float(cost_per_product.get(p, 0.0)) for p in products]
        )
        direct = margin

    recovery = np.zeros(n)
    for _, e in dt_edges.iterrows():
        if e["from_product"] in idx and e["to_product"] in idx:
            recovery[idx[e["from_product"]]] += float(e["observed_switching_recovery_proxy"])

    coeff = direct  # Remove recovery term from direct coefficient - handled via z_ij variables

    # Build z_ij variables for each transference edge
    edge_list = []
    edge_recovery = []
    for _, e in dt_edges.iterrows():
        if e["from_product"] in idx and e["to_product"] in idx:
            i = idx[e["from_product"]]
            j = idx[e["to_product"]]
            edge_list.append((i, j))
            edge_recovery.append(float(e["observed_switching_recovery_proxy"]))
    
    m = len(edge_list)
    
    # New variable structure: x_1...x_n, z_1...z_m
    # Total variables: n + m
    total_vars = n + m
    
    # Objective: maximize direct*x + recovery_margin * sum(edge_recovery * z)
    # c is negative because milp minimizes
    c = np.zeros(total_vars)
    c[:n] = -direct
    for k in range(m):
        c[n + k] = -recovery_margin * edge_recovery[k]

    # Integrality: x_i are binary, z_ij are binary
    integrality = np.ones(total_vars, dtype=bool)

    constraints: list[LinearConstraint] = []
    # Max SKUs: sum x_i <= max_skus
    rows_skus = np.zeros((1, total_vars))
    rows_skus[0, :n] = 1.0
    constraints.append(LinearConstraint(rows_skus, -np.inf, max_skus))
    
    # Coverage: sum revenue_i * x_i >= min_coverage * total_revenue
    rows_cov = np.zeros((1, total_vars))
    rows_cov[0, :n] = revenue.values.astype(float)
    constraints.append(LinearConstraint(rows_cov, min_coverage * float(revenue.sum()), np.inf))
    
    # Category: at least one SKU per category
    category_of = _category_of(transactions_df)
    for cat in dict.fromkeys(category_of.get(p) for p in products if category_of.get(p)):
        cat_skus = [i for i, p in enumerate(products) if category_of.get(p) == cat]
        if cat_skus:
            row = np.zeros((1, total_vars))
            row[0, cat_skus] = 1.0
            constraints.append(LinearConstraint(row, 1, np.inf))
    
    # z_ij constraints: z_ij <= x_j and z_ij <= 1 - x_i
    for k, (i, j) in enumerate(edge_list):
        # z_ij <= x_j  =>  z_ij - x_j <= 0
        row1 = np.zeros(total_vars)
        row1[n + k] = 1.0
        row1[j] = -1.0
        constraints.append(LinearConstraint(row1, -np.inf, 0.0))
        # z_ij <= 1 - x_i  =>  z_ij + x_i <= 1
        row2 = np.zeros(total_vars)
        row2[n + k] = 1.0
        row2[i] = 1.0
        constraints.append(LinearConstraint(row2, -np.inf, 1.0))

    relaxed_to: float | None = None
    coverage_attempt = min_coverage
    result = None
    while True:
        constraints[1] = LinearConstraint(
            rows_cov, coverage_attempt * float(revenue.sum()), np.inf
        )
        result = milp(
            c=c,
            integrality=integrality,
            bounds=Bounds(0, 1),
            constraints=constraints,
            options={"time_limit": time_limit_seconds},
        )
        if result.success:
            break
        if coverage_attempt <= 0.05:
            break
        coverage_attempt /= 2.0
        relaxed_to = coverage_attempt

    if result is None or not result.success:
        raise RuntimeError(f"MILP solver failed: {result.message}")

    # Only the first n variables are x_i (product selection)
    selected = [p for i, p in enumerate(products) if result.x[i] > 0.5]
    metrics: dict[str, object] = {
        "solver_status": MILP_STATUS.get(result.status, "unknown"),
        "objective_value": float(result.fun * -1),
        "n_candidates": n,
        "minimum_coverage_enforced": relaxed_to if relaxed_to is not None else min_coverage,
    }
    return selected, metrics


def optimize_assortment_heuristic(
    transactions_df: pd.DataFrame,
    revenue_per_product: pd.Series | None = None,
    demand_transference_df: pd.DataFrame | None = None,
    *,
    max_skus: int = 100,
    min_coverage: float = 0.80,
    objective: str = "revenue",
    cost_per_product: pd.Series | None = None,
    iterations: int = 400,
    random_seed: int | None = None,
    recovery_margin: float = DEFAULT_RECOVERY_MARGIN,
) -> tuple[list[str], dict[str, object]]:
    """Simulated-annealing assortment search over revenue-ranked candidates.

    Starts from the top-``max_skus`` revenue products and explores single
    swap moves with a cooling acceptance rule; a final greedy pass scans all
    single swaps for improvement. Deterministic given ``random_seed``.
    """
    if revenue_per_product is None:
        revenue_per_product = _revenue_series(transactions_df)
    revenue = revenue_per_product.head(max_skus * 4)

    if demand_transference_df is None or demand_transference_df.empty:
        dt_edges = pd.DataFrame(columns=["from_product", "to_product", "revenue_at_risk"])
    else:
        dt_edges = demand_transference_df[
            ["from_product", "to_product", "revenue_at_risk"]
        ]
    dt_edges = dt_edges[dt_edges["from_product"].isin(revenue.index) & dt_edges["to_product"].isin(revenue.index)]
    transfers = _transfers_by_from(dt_edges)

    margin_of: dict[str, float] = {}
    if objective == "margin" and cost_per_product is not None:
        margin_of = {
            p: float(revenue.get(p, 0.0) - cost_per_product.get(p, 0.0)) for p in revenue.index
        }

    def objective_value(kept: set[str]) -> float:
        metrics = _evaluate_solution(kept, revenue, transfers)
        base = metrics.kept_revenue if objective != "margin" else sum(
            margin_of.get(p, 0.0) for p in kept
        )
        value = base + recovery_margin * metrics.recovered_revenue
        if metrics.coverage < min_coverage:
            value -= 1000.0 * (min_coverage - metrics.coverage)
        return value

    rng = np.random.default_rng(random_seed)
    current = set(revenue.index[:max_skus])
    candidates = list(revenue.index)
    best = set(current)
    best_value = objective_value(best)
    temperature = 1.0
    cooling = 0.995

    for _ in range(iterations):
        if not current:
            break
        remove_sku = rng.choice(sorted(current))
        pool = [p for p in candidates if p not in current]
        if not pool:
            break
        add_sku = str(rng.choice(pool))
        neighbor = set(current)
        neighbor.discard(remove_sku)
        neighbor.add(add_sku)
        value = objective_value(neighbor)
        if value > best_value or rng.random() < np.exp((value - best_value) / temperature):
            current = neighbor
            if value > best_value:
                best = set(current)
                best_value = value
        temperature *= cooling

    improved = True
    while improved:
        improved = False
        for remove_sku in list(best):
            for add_sku in candidates:
                if add_sku in best:
                    continue
                neighbor = set(best)
                neighbor.discard(remove_sku)
                neighbor.add(add_sku)
                value = objective_value(neighbor)
                if value > best_value + 1e-9:
                    best = set(neighbor)
                    best_value = value
                    improved = True
                    break
            if improved:
                break

    metrics: dict[str, object] = {**_evaluate_solution(set(best), revenue, transfers).to_dict()}
    metrics["objective_value"] = best_value
    return sorted(best), metrics


def evaluate_assortment(
    selected_skus: list[str],
    transactions_df: pd.DataFrame,
    demand_transference_df: pd.DataFrame | None = None,
    revenue_per_product: pd.Series | None = None,
) -> dict[str, object]:
    """Comprehensive evaluation of a candidate assortment."""
    if revenue_per_product is None:
        revenue_per_product = _revenue_series(transactions_df)
    if demand_transference_df is None:
        demand_transference_df = compute_demand_transference_matrix(transactions_df)
    transfers = (
        _transfers_by_from(demand_transference_df)
        if demand_transference_df is not None and not demand_transference_df.empty
        else {}
    )
    metrics = _evaluate_solution(set(selected_skus), revenue_per_product, transfers)

    category_of = _category_of(transactions_df)
    selected_cats = {category_of[p] for p in selected_skus if p in category_of}
    all_cats = set(category_of.values())
    result: dict[str, object] = {**metrics.to_dict()}
    result["n_categories_covered"] = len(selected_cats)
    result["n_categories_total"] = len(all_cats)
    return result


def _scenario_row(
    scenario_id: int,
    method: str,
    kept: set[str],
    revenue: pd.Series,
    transfers: dict[str, pd.DataFrame],
) -> dict[str, object]:
    m = _evaluate_solution(kept, revenue, transfers)
    row: dict[str, object] = {"scenario_id": scenario_id, "method": method}
    row.update(m.to_dict())
    return row


def compare_assortment_scenarios(
    transactions_df: pd.DataFrame,
    base_assortment: list[str],
    demand_transference_df: pd.DataFrame | None = None,
    *,
    n_scenarios: int = 6,
    max_skus_range: tuple[int, int] = (30, 90),
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Score random, greedy, and MILP-derived candidate assortments.

    Method mix: the first scenario is the revenue-greedy assortment; the
    remaining ones are random draws (even) and MILP solves at the scenario's
    SKU budget (odd). Returns a scored comparison table.
    """
    revenue = _revenue_series(transactions_df)
    if demand_transference_df is None:
        demand_transference_df = compute_demand_transference_matrix(transactions_df)
    transfers = (
        _transfers_by_from(demand_transference_df)
        if demand_transference_df is not None and not demand_transference_df.empty
        else {}
    )

    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    for i in range(1, n_scenarios + 1):
        max_skus = int(rng.integers(*max_skus_range))
        if i == 1:
            kept = set(revenue.head(max_skus).index)
            method = "greedy"
        elif i % 2 == 0:
            kept = set(rng.choice(list(revenue.index), size=min(max_skus, len(revenue)), replace=False))
            method = "random"
        else:
            selected, _ = optimize_assortment_milp(
                transactions_df,
                revenue_per_product=revenue,
                demand_transference_df=demand_transference_df,
                max_skus=max_skus,
                time_limit_seconds=15,
            )
            kept = set(selected)
            method = "milp"
        rows.append(_scenario_row(i, method, kept, revenue, transfers))

    table = pd.DataFrame(rows, columns=list(ASSORTMENT_SCENARIO.columns))
    return check(table, ASSORTMENT_SCENARIO)


def build_solution_table(
    selected_skus: list[str],
    revenue_per_product: pd.Series,
) -> pd.DataFrame:
    """Selected-SKU table with revenue and rank for display."""
    present = [p for p in selected_skus if p in revenue_per_product.index]
    ranked = revenue_per_product.loc[present].sort_values(ascending=False)
    rows = [
        {"stockcode": sku, "selected": 1, "revenue": float(rev), "rank": rank}
        for rank, (sku, rev) in enumerate(ranked.items(), start=1)
    ]
    table = pd.DataFrame(rows, columns=list(ASSORTMENT_SOLUTION.columns))
    return check(table, ASSORTMENT_SOLUTION, allow_empty=True)


def evaluate_selected_scenarios(
    scenario_ids: list[int],
    scenarios_df: pd.DataFrame,
    selected_by_scenario: dict[int, list[str]],
    transactions_df: pd.DataFrame,
    demand_transference_df: pd.DataFrame | None = None,
    revenue_per_product: pd.Series | None = None,
) -> pd.DataFrame:
    """Full evaluation of chosen scenarios (coverage, categories, recovery)."""
    if revenue_per_product is None:
        revenue_per_product = _revenue_series(transactions_df)
    if demand_transference_df is None:
        demand_transference_df = compute_demand_transference_matrix(transactions_df)
    transfers = (
        _transfers_by_from(demand_transference_df)
        if demand_transference_df is not None and not demand_transference_df.empty
        else {}
    )
    category_of = _category_of(transactions_df)
    all_cats = set(category_of.values())

    rows: list[dict[str, object]] = []
    for sid in scenario_ids:
        method_row = scenarios_df[scenarios_df["scenario_id"] == sid]
        method = str(method_row["method"].iloc[0]) if not method_row.empty else "custom"
        kept = set(selected_by_scenario.get(sid, []))
        metrics = _evaluate_solution(kept, revenue_per_product, transfers)
        row: dict[str, object] = {
            "scenario_id": sid,
            "method": method,
            "selected_skus": metrics.n_skus,
        }
        row.update(metrics.to_dict())
        row.pop("n_skus", None)
        selected_cats = {category_of[p] for p in kept if p in category_of}
        row["n_categories_covered"] = len(selected_cats)
        row["n_categories_total"] = len(all_cats)
        rows.append(row)

    table = pd.DataFrame(rows, columns=list(ASSORTMENT_EVALUATION.columns))
    return check(table, ASSORTMENT_EVALUATION)
