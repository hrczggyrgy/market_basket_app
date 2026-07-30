"""Assortment Optimization — Heuristic & MILP solvers for range planning."""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def optimize_assortment_heuristic(
    transactions_df: pd.DataFrame,
    substitution_matrix: pd.DataFrame,
    demand_transference_matrix: pd.DataFrame,
    revenue_per_product: pd.Series,
    cost_per_product: Optional[pd.Series] = None,
    max_skus: int = 100,
    min_coverage: float = 0.80,
    objective: str = "revenue",
    iterations: int = 1000,
    temperature: float = 1.0,
    cooling_rate: float = 0.995,
) -> Tuple[List[str], Dict]:
    """
    Simulated annealing + greedy local search for assortment optimization.

    Objective: maximize revenue (or margin) - penalty * unmet_demand + bonus * recovery
    Subject to: max SKUs, min coverage
    """

    # Initial solution: top products by revenue
    all_products = list(revenue_per_product.index)
    sorted_products = revenue_per_product.sort_values(ascending=False).index.tolist()
    current_solution = sorted_products[:max_skus]

    # Precompute data structures — ensure all values are numeric
    sub_matrix = substitution_matrix.copy().astype(float)
    if demand_transference_matrix is not None:
        dt_matrix = demand_transference_matrix.copy().astype(float)
    else:
        dt_matrix = pd.DataFrame(0.0, index=all_products, columns=all_products)

    # Ensure indices align
    common_products = set(current_solution) & set(sub_matrix.index)
    if len(common_products) < len(current_solution):
        # Filter to common products
        current_solution = [p for p in current_solution if p in common_products]

    total_revenue = revenue_per_product.sum()
    total_demand = revenue_per_product.sum()  # proxy for total demand

    def evaluate(solution: List[str]) -> Tuple[float, Dict]:
        """Evaluate a solution."""
        if not solution:
            return -1e9, {}

        solution_set = set(solution)

        # Revenue from kept SKUs
        kept_revenue = sum(revenue_per_product.get(p, 0) for p in solution)

        # Demand transference from removed SKUs
        removed = [p for p in revenue_per_product.index if p not in solution_set]
        recovered = 0.0
        for r in removed:
            if r in dt_matrix.index:
                # Sum transfers to kept products
                transfers = dt_matrix.loc[r]
                for target, row in transfers.items():
                    if target in solution_set:
                        recovered += float(row) if not isinstance(row, dict) else row.get("revenue_at_risk", 0)

        # Unmet demand (revenue lost - recovered)
        lost = sum(revenue_per_product.get(p, 0) for p in removed)
        unmet = lost - recovered

        # Coverage
        covered_revenue = kept_revenue + recovered
        coverage = covered_revenue / total_demand if total_demand > 0 else 0

        # Objective
        if objective == "margin" and cost_per_product is not None:
            kept_margin = sum(
                (revenue_per_product.get(p, 0) - cost_per_product.get(p, 0)) for p in solution
            )
            recovered_margin = recovered * 0.3  # assume 30% margin on recovered
            obj = kept_margin + recovered_margin - 0.5 * unmet
        else:
            obj = covered_revenue - 0.5 * unmet

        # Penalty for constraint violations
        if len(solution) > max_skus:
            obj -= 1000 * (len(solution) - max_skus)
        if coverage < min_coverage:
            obj -= 1000 * (min_coverage - coverage)

        metrics = {
            "kept_revenue": kept_revenue,
            "recovered": recovered,
            "unmet": unmet,
            "coverage": coverage,
            "expected_revenue": covered_revenue,
            "recovery_rate": recovered / lost if lost > 0 else 0,
            "n_skus": len(solution),
        }

        return obj, metrics

    # Simulated annealing
    current_obj, current_metrics = evaluate(current_solution)
    best_solution = current_solution[:]
    best_obj = current_obj
    best_metrics = current_metrics

    temp = temperature

    for iteration in range(iterations):
        # Generate neighbor: swap one product
        if len(current_solution) >= 2 and np.random.random() < 0.7:
            # Swap
            remove_idx = np.random.randint(len(current_solution))
            # Find candidate not in solution
            candidates = [p for p in all_products if p not in current_solution]
            if not candidates:
                continue
            add_candidate = np.random.choice(candidates)

            new_solution = current_solution[:]
            new_solution[remove_idx] = add_candidate
        else:
            # Add/remove
            if len(current_solution) < max_skus and np.random.random() < 0.5:
                # Add
                candidates = [p for p in all_products if p not in current_solution]
                if candidates:
                    new_solution = current_solution + [np.random.choice(candidates)]
                else:
                    new_solution = current_solution[:]
            else:
                # Remove
                if len(current_solution) > 1:
                    remove_idx = np.random.randint(len(current_solution))
                    new_solution = [p for i, p in enumerate(current_solution) if i != remove_idx]
                else:
                    new_solution = current_solution[:]

        new_obj, new_metrics = evaluate(new_solution)

        # Acceptance
        if new_obj > current_obj or np.random.random() < np.exp((new_obj - current_obj) / temp):
            current_solution = new_solution
            current_obj = new_obj
            current_metrics = new_metrics

            if new_obj > best_obj:
                best_solution = new_solution[:]
                best_obj = new_obj
                best_metrics = new_metrics

        temp *= cooling_rate

    # Final greedy improvement
    best_solution = _greedy_improvement(
        best_solution,
        all_products,
        revenue_per_product,
        dt_matrix,
        max_skus,
        min_coverage,
        objective,
        cost_per_product,
    )

    final_obj, final_metrics = evaluate(best_solution)

    return best_solution, final_metrics


def _greedy_improvement(
    solution: List[str],
    all_products: List[str],
    revenue_per_product: pd.Series,
    dt_matrix: pd.DataFrame,
    max_skus: int,
    min_coverage: float,
    objective: str,
    cost_per_product: Optional[pd.Series],
) -> List[str]:
    """Greedy local search improvement."""
    solution_set = set(solution)
    improved = True

    while improved:
        improved = False
        best_swap = None
        best_gain = 0

        # Try all single swaps
        for remove_sku in solution:
            for add_sku in all_products:
                if add_sku in solution_set:
                    continue

                new_solution = [p for p in solution if p != remove_sku] + [add_sku]
                _, new_metrics = evaluate_solution(
                    new_solution,
                    all_products,
                    revenue_per_product,
                    dt_matrix,
                    max_skus,
                    min_coverage,
                    objective,
                    cost_per_product,
                )
                gain = new_metrics.get("expected_revenue", 0) - new_metrics.get("kept_revenue", 0)

                if gain > best_gain:
                    best_gain = gain
                    best_swap = (remove_sku, add_sku)

        if best_swap:
            solution = [p for p in solution if p != best_swap[0]] + [best_swap[1]]
            solution_set = set(solution)
            improved = True

    return solution


def evaluate_solution(
    solution: List[str],
    all_products: List[str],
    revenue_per_product: pd.Series,
    dt_matrix: pd.DataFrame,
    max_skus: int,
    min_coverage: float,
    objective: str,
    cost_per_product: Optional[pd.Series],
) -> Tuple[float, Dict]:
    """Evaluate a single solution."""
    solution_set = set(solution)

    kept_revenue = sum(revenue_per_product.get(p, 0) for p in solution)

    removed = [p for p in revenue_per_product.index if p not in solution_set]
    recovered = 0.0
    for r in removed:
        if r in dt_matrix.index:
            for target, row in dt_matrix.loc[r].items():
                if target in solution_set:
                    recovered += float(row) if not isinstance(row, dict) else row.get("revenue_at_risk", 0)

    lost = sum(revenue_per_product.get(p, 0) for p in removed)
    covered_revenue = kept_revenue + recovered

    total_demand = revenue_per_product.sum()
    coverage = covered_revenue / total_demand if total_demand > 0 else 0

    if objective == "margin" and cost_per_product is not None:
        kept_margin = sum(
            (revenue_per_product.get(p, 0) - cost_per_product.get(p, 0)) for p in solution
        )
        obj = kept_margin + recovered * 0.3 - 0.5 * (lost - recovered)
    else:
        obj = covered_revenue - 0.5 * (lost - recovered)

    if len(solution) > max_skus:
        obj -= 1000 * (len(solution) - max_skus)
    if coverage < min_coverage:
        obj -= 1000 * (min_coverage - coverage)

    return obj, {
        "kept_revenue": kept_revenue,
        "recovered": recovered,
        "unmet": lost - recovered,
        "coverage": coverage,
        "expected_revenue": covered_revenue,
        "recovery_rate": recovered / lost if lost > 0 else 0,
        "n_skus": len(solution),
    }


def optimize_assortment_milp(
    transactions_df: pd.DataFrame,
    substitution_matrix: pd.DataFrame,
    demand_transference_matrix: pd.DataFrame,
    revenue_per_product: pd.Series,
    cdt_tree: Optional[object] = None,
    max_skus: int = 100,
    min_coverage: float = 0.80,
    objective: str = "revenue",
    cost_per_product: Optional[pd.Series] = None,
    time_limit_seconds: int = 60,
) -> Tuple[List[str], Dict]:
    """
    MILP assortment optimization using OR-Tools.

    maximize sum(revenue_i * x_i + sum_j DT(i->j) * revenue_i * (1-x_i) * x_j)
    s.t. sum(x_i) <= max_skus
         coverage >= min_coverage
         x_i in {0,1}
    """
    try:
        from ortools.linear_solver import pywraplp
    except ImportError:
        raise ImportError("OR-Tools required for MILP. Install: pip install ortools")

    all_products = list(revenue_per_product.index)
    n = len(all_products)

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        raise RuntimeError("SCIP solver not available in OR-Tools")

    # Decision variables
    x = {}
    for i, sku in enumerate(all_products):
        x[i] = solver.IntVar(0, 1, f"x_{sku}")

    # Objective coefficients
    obj_coeffs = []
    for i, sku in enumerate(all_products):
        revenue = revenue_per_product.get(sku, 0)

        # Direct revenue
        coeff = revenue

        # Transference recovery (if removed, how much recovers to kept SKUs)
        # This is approximated: if kept, full revenue; if removed, partial recovery
        # Full MILP would need quadratic terms; we linearize
        if sku in demand_transference_matrix.index:
            # Sum of transfers to all products (upper bound on recovery)
            row = demand_transference_matrix.loc[sku]
            if isinstance(row, pd.DataFrame) and "revenue_at_risk" in row.columns:
                max_recovery = row["revenue_at_risk"].sum()
            elif isinstance(row, pd.Series):
                # Cell-as-dict format: each value is a dict with "revenue_at_risk"
                max_recovery = 0.0
                for val in row.values:
                    if isinstance(val, dict):
                        max_recovery += val.get("revenue_at_risk", 0)
                    else:
                        max_recovery += float(val)
            else:
                max_recovery = 0.0
            # Conservative: assume 50% recovery if kept
            coeff += 0.5 * max_recovery

        obj_coeffs.append(coeff)

    # Maximize
    solver.Maximize(solver.Sum(obj_coeffs[i] * x[i] for i in range(n)))

    # Constraints
    # Max SKUs
    solver.Add(solver.Sum(x[i] for i in range(n)) <= max_skus)

    # Min coverage (linearized)
    # coverage = (sum(revenue_i * x_i) + sum_j DT(i->j) * revenue_i * x_i) / total_demand
    # This is complex; use simpler constraint: revenue_kept >= min_coverage * total_revenue
    total_rev = float(revenue_per_product.sum())
    min_rev = min_coverage * total_rev
    solver.Add(
        solver.Sum(revenue_per_product.get(all_products[i], 0) * x[i] for i in range(n)) >= min_rev
    )

    # Category constraints (if CDT available, at least one per leaf)
    # Simplified: ensure at least 1 SKU per category
    if "category" in transactions_df.columns:
        cat_map = transactions_df.drop_duplicates("stockcode").set_index("stockcode")["category"]
        categories = set(cat_map.values)
        for cat in categories:
            cat_skus = [i for i, sku in enumerate(all_products) if cat_map.get(sku) == cat]
            if cat_skus:
                solver.Add(solver.Sum(x[i] for i in cat_skus) >= 1)

    # Solve
    solver.SetTimeLimit(time_limit_seconds * 1000)
    status = solver.Solve()

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"MILP solver failed with status {status}")

    # Extract solution
    selected = [all_products[i] for i in range(n) if x[i].solution_value() > 0.5]

    # Evaluate
    metrics = {
        "solver_status": "optimal" if status == pywraplp.Solver.OPTIMAL else "feasible",
        "objective_value": solver.Objective().Value(),
    }

    return selected, metrics


def evaluate_assortment(
    selected_skus: List[str],
    transactions_df: pd.DataFrame,
    substitution_matrix: pd.DataFrame,
    demand_transference_matrix: pd.DataFrame,
    cdt_tree: Optional[object] = None,
) -> Dict:
    """Comprehensive evaluation of an assortment."""

    revenue_per_product = transactions_df.groupby("stockcode").apply(
        lambda x: (x["price"] * x["quantity"]).sum()
    )

    total_demand = revenue_per_product.sum()
    selected_set = set(selected_skus)

    # Kept revenue
    kept_revenue = sum(revenue_per_product.get(p, 0) for p in selected_skus)

    # Recovered demand
    removed = [p for p in revenue_per_product.index if p not in selected_set]
    recovered = 0.0
    recovery_detail = {}
    for r in removed:
        if r in demand_transference_matrix.index:
            for target, row in demand_transference_matrix.loc[r].items():
                if target in selected_set:
                    amt = float(row) if not isinstance(row, dict) else row.get("revenue_at_risk", 0)
                    recovered += amt
                    if r not in recovery_detail:
                        recovery_detail[r] = 0
                    recovery_detail[r] += amt

    lost = sum(revenue_per_product.get(p, 0) for p in removed)
    unmet = lost - recovered

    # Coverage
    covered_revenue = kept_revenue + recovered
    coverage = covered_revenue / total_demand if total_demand > 0 else 0

    # Category balance
    cat_balances = {}
    if "category" in transactions_df.columns:
        cat_map = transactions_df.drop_duplicates("stockcode").set_index("stockcode")["category"]
        for cat in cat_map.unique():
            cat_skus = [p for p in selected_skus if cat_map.get(p) == cat]
            cat_rev = sum(revenue_per_product.get(p, 0) for p in cat_skus)
            cat_balances[cat] = cat_rev

    # CDT leaf coverage
    leaf_coverage = 0
    if cdt_tree is not None:
        leaf_coverage = _compute_cdt_leaf_coverage(cdt_tree, selected_set)

    return {
        "selected_skus": len(selected_skus),
        "kept_revenue": kept_revenue,
        "recovered_revenue": recovered,
        "lost_revenue": lost,
        "unmet_demand": unmet,
        "expected_revenue": covered_revenue,
        "coverage": coverage,
        "recovery_rate": recovered / lost if lost > 0 else 0,
        "category_balance": cat_balances,
        "cdt_leaf_coverage": leaf_coverage,
    }


def _compute_cdt_leaf_coverage(cdt_tree, selected_skus: set) -> float:
    """Compute fraction of CDT leaves with at least one SKU selected."""
    leaves = []

    def collect(node):
        if node.is_leaf:
            leaves.append(node)
        else:
            for c in node.children:
                collect(c)

    collect(cdt_tree)

    if not leaves:
        return 0.0

    covered = sum(1 for leaf in leaves if any(p in selected_skus for p in leaf.products))
    return covered / len(leaves)


def generate_assortment_scenarios(
    transactions_df: pd.DataFrame,
    base_assortment: List[str],
    n_scenarios: int = 10,
    max_skus_range: Tuple[int, int] = (50, 150),
) -> List[Dict]:
    """Generate diverse candidate assortments for comparison."""

    revenue_per_product = transactions_df.groupby("stockcode").apply(
        lambda x: (x["price"] * x["quantity"]).sum()
    )

    scenarios = []

    for i in range(n_scenarios):
        if i < n_scenarios // 3:
            # Random sampling
            max_skus = np.random.randint(*max_skus_range)
            selected = np.random.choice(
                base_assortment, min(max_skus, len(base_assortment)), replace=False
            ).tolist()
        elif i < 2 * n_scenarios // 3:
            # Greedy revenue-based
            max_skus = np.random.randint(*max_skus_range)
            selected = revenue_per_product.sort_values(ascending=False).index[:max_skus].tolist()
        else:
            # CDT-guided: keep full leaves, prune low-quality
            max_skus = np.random.randint(*max_skus_range)
            selected = _cdt_guided_selection(transactions_df, base_assortment, max_skus)

        scenarios.append(
            {
                "scenario_id": i + 1,
                "method": ["random", "greedy", "cdt_guided"][min(2, i // (n_scenarios // 3))],
                "skus": selected,
            }
        )

    return scenarios


def _cdt_guided_selection(
    transactions_df: pd.DataFrame,
    base_assortment: List[str],
    max_skus: int,
) -> List[str]:
    """CDT-guided assortment selection (simplified)."""
    # Would build CDT and select based on leaf quality
    # For now, return top revenue
    revenue_per_product = transactions_df.groupby("stockcode").apply(
        lambda x: (x["price"] * x["quantity"]).sum()
    )
    return revenue_per_product.loc[base_assortment].nlargest(max_skus).index.tolist()
