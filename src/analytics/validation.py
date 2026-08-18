"""Validation harness for analytics modules.

Provides a uniform interface to run all analytics functions on a dataset
and collect their outputs + diagnostics for comparison / regression testing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    module: str
    function: str
    success: bool
    output_shape: Optional[tuple[int, int]] = None
    output_columns: Optional[List[str]] = None
    error: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class ValidationHarness:
    """Runs analytics functions and captures their outputs for validation."""

    def __init__(self, transactions_df: pd.DataFrame):
        self.df = transactions_df
        self.results: List[ValidationResult] = []

    def _run(
        self, module: str, function: str, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> ValidationResult:
        """Execute a function and capture results."""
        try:
            output = fn(*args, **kwargs)
            if isinstance(output, tuple):
                main_output = output[0]
                diagnostics = {}
                for i, o in enumerate(output[1:]):
                    if isinstance(o, dict):
                        diagnostics[f"output_{i}"] = o
                    elif isinstance(o, pd.DataFrame):
                        diagnostics[f"output_{i}"] = {
                            "shape": o.shape,
                            "columns": o.columns.tolist(),
                        }
            else:
                main_output = output
                diagnostics = {}

            # Store the main output in diagnostics for chaining
            diagnostics["output"] = main_output

            return ValidationResult(
                module=module,
                function=function,
                success=True,
                output_shape=main_output.shape if isinstance(main_output, pd.DataFrame) else None,
                output_columns=main_output.columns.tolist()
                if isinstance(main_output, pd.DataFrame)
                else None,
                diagnostics=diagnostics,
            )
        except Exception as e:
            return ValidationResult(
                module=module,
                function=function,
                success=False,
                error=str(e),
            )

    def run_all(self) -> List[ValidationResult]:
        """Run all registered analytics functions."""
        self.results = []

        def _run_and_store(
            module: str, function: str, fn: Callable[..., Any], *args: Any, **kwargs: Any
        ) -> ValidationResult:
            """Execute a function, store result, and return it."""
            result = self._run(module, function, fn, *args, **kwargs)
            self.results.append(result)
            return result

        # Core data
        _run_and_store("data", "load_transactions", lambda: self.df)

        # Rules / association
        from src.analytics.rules import create_basket_matrix, generate_rules, run_fpgrowth

        basket_result = _run_and_store(
            "rules", "create_basket_matrix", create_basket_matrix, self.df
        )
        if basket_result.success:
            # create_basket_matrix returns DataFrame directly, not a tuple
            basket = basket_result.diagnostics.get("output", basket_result.output_shape and self.df)
            freq = _run_and_store("rules", "run_fpgrowth", run_fpgrowth, basket, min_support=0.01)
            if freq.success:
                _run_and_store(
                    "rules",
                    "generate_rules",
                    generate_rules,
                    freq.diagnostics.get("output", freq.output_shape and self.df),
                    min_threshold=0.05,
                )

        # Co-purchase / affinity
        from src.analytics.copurchase import compute_affinity_matrix, get_top_affinity_pairs

        _run_and_store(
            "copurchase", "get_top_affinity_pairs", get_top_affinity_pairs, self.df, top_n=20
        )
        _run_and_store(
            "copurchase",
            "compute_affinity_matrix",
            compute_affinity_matrix,
            self.df,
            min_cooccurrence=5,
        )

        # Add-on
        from src.analytics.addon import get_addon_recommendations

        # Use a real product from the data as anchor
        anchor_sku = self.df["stockcode"].iloc[0] if len(self.df) > 0 else "SKU001"
        _run_and_store(
            "addon",
            "get_addon_recommendations",
            get_addon_recommendations,
            self.df,
            anchor=anchor_sku,
            top_n=10,
            min_lift=1.2,
        )

        # Switching
        from src.analytics.switching import compute_switching_matrix, get_customer_loyalty_metrics

        _run_and_store("switching", "compute_switching_matrix", compute_switching_matrix, self.df)
        _run_and_store(
            "switching", "get_customer_loyalty_metrics", get_customer_loyalty_metrics, self.df
        )

        # Basket metrics
        from src.analytics.basket_metrics import (
            compute_basket_penetration,
            compute_customer_entropy,
            compute_ipt_cv,
        )

        _run_and_store(
            "basket_metrics", "compute_basket_penetration", compute_basket_penetration, self.df
        )
        _run_and_store(
            "basket_metrics", "compute_customer_entropy", compute_customer_entropy, self.df
        )
        _run_and_store("basket_metrics", "compute_ipt_cv", compute_ipt_cv, self.df)

        # Cohort
        from src.analytics.cohort import compute_cohorts

        _run_and_store("cohort", "compute_cohorts", compute_cohorts, self.df)

        # Performance / product
        from src.analytics.performance import (
            abc_analysis,
            compute_product_metrics,
            compute_repeat_rate,
            compute_time_to_second_purchase,
            compute_velocity,
            product_lifecycle_stage,
            xyz_analysis,
        )

        _run_and_store("performance", "compute_product_metrics", compute_product_metrics, self.df)
        pm = compute_product_metrics(self.df)
        if not pm.empty:
            _run_and_store("performance", "abc_analysis", abc_analysis, self.df)
            _run_and_store("performance", "xyz_analysis", xyz_analysis, self.df)
            _run_and_store(
                "performance", "product_lifecycle_stage", product_lifecycle_stage, self.df
            )
            _run_and_store("performance", "compute_velocity", compute_velocity, self.df)
            _run_and_store("performance", "compute_repeat_rate", compute_repeat_rate, self.df)
            _run_and_store(
                "performance",
                "compute_time_to_second_purchase",
                compute_time_to_second_purchase,
                self.df,
            )

        # Category
        from src.analytics.category import compute_category_kpis, compute_category_scorecard

        _run_and_store("category", "compute_category_kpis", compute_category_kpis, self.df)
        _run_and_store(
            "category", "compute_category_scorecard", compute_category_scorecard, self.df
        )

        # Choice model
        from src.analytics.choice_model import build_customer_features, train_choice_model

        cf = _run_and_store(
            "choice_model", "build_customer_features", build_customer_features, self.df
        )
        if cf.success:
            _run_and_store(
                "choice_model",
                "train_choice_model",
                train_choice_model,
                cf.diagnostics.get("output", cf.output_shape and self.df),
            )

        # Promotional
        from src.analytics.promo import (
            compute_incrementality_waterfall,
            compute_promo_baseline,
            detect_promotions,
            pre_post_promo_comparison,
            promo_roi_analysis,
        )

        promo_periods = _run_and_store("promo", "detect_promotions", detect_promotions, self.df)
        baseline_df = None
        if promo_periods.success:
            baseline_df = _run_and_store(
                "promo",
                "compute_promo_baseline",
                compute_promo_baseline,
                self.df,
                promo_periods=promo_periods.diagnostics.get(
                    "output", promo_periods.output_shape and self.df
                ),
            )
            _run_and_store(
                "promo",
                "pre_post_promo_comparison",
                pre_post_promo_comparison,
                self.df,
                promo_periods=promo_periods.diagnostics.get(
                    "output", promo_periods.output_shape and self.df
                ),
            )
            if baseline_df.success:
                _run_and_store(
                    "promo",
                    "compute_incrementality_waterfall",
                    compute_incrementality_waterfall,
                    baseline_df.diagnostics.get("output", baseline_df.output_shape and self.df),
                )
            _run_and_store(
                "promo",
                "promo_roi_analysis",
                promo_roi_analysis,
                self.df,
                promo_periods=promo_periods.diagnostics.get(
                    "output", promo_periods.output_shape and self.df
                ),
            )

        # CLV
        from src.analytics.clv import compute_clv_customer_df, predict_clv_bg_nbd

        clv = _run_and_store("clv", "predict_clv_bg_nbd", predict_clv_bg_nbd, self.df)
        if clv.success:
            _run_and_store("clv", "compute_clv_customer_df", compute_clv_customer_df, self.df)

        # Transference
        from src.analytics.cdt.similarity import build_similarity_matrix
        from src.analytics.transference import (
            build_substitution_matrix_mnl,
            compute_demand_transference_matrix,
            compute_recovery_hhi,
            compute_substitutable_demand_percentage,
            delist_impact_analysis,
        )

        dt = _run_and_store(
            "transference",
            "compute_demand_transference_matrix",
            compute_demand_transference_matrix,
            self.df,
        )
        if dt.success:
            dt_output = dt.diagnostics.get("output", dt.output_shape and self.df)
            if dt_output is not None:
                _run_and_store(
                    "transference",
                    "compute_substitutable_demand_percentage",
                    compute_substitutable_demand_percentage,
                    dt_output,
                    self.df,
                )
                _run_and_store(
                    "transference",
                    "delist_impact_analysis",
                    delist_impact_analysis,
                    self.df,
                    dt_output,
                    dt_output["from_product"].unique()[:5],
                )
                sim_matrix = _run_and_store(
                    "transference",
                    "build_similarity_matrix",
                    build_similarity_matrix,
                    self.df,
                    method="phi",
                )
                if sim_matrix.success:
                    _run_and_store(
                        "transference",
                        "build_substitution_matrix_mnl",
                        build_substitution_matrix_mnl,
                        self.df,
                        sim_matrix.diagnostics.get("output", sim_matrix.output_shape and self.df),
                    )
                _run_and_store(
                    "transference", "compute_recovery_hhi", compute_recovery_hhi, dt_output
                )

        # Assortment
        from src.analytics.assortment import (
            compare_assortment_scenarios,
            evaluate_assortment,
            optimize_assortment_heuristic,
        )

        assort = _run_and_store(
            "assortment", "optimize_assortment_heuristic", optimize_assortment_heuristic, self.df
        )
        if assort.success:
            output = assort.diagnostics.get("output")
            if output is None:
                output = assort.output_shape and self.df

            kept_skus = (
                output[0]
                if isinstance(output, tuple)
                else self.df["stockcode"].unique()[:10]
            )
            _run_and_store(
                "assortment", "evaluate_assortment", evaluate_assortment, kept_skus, self.df
            )
            _run_and_store(
                "assortment",
                "compare_assortment_scenarios",
                compare_assortment_scenarios,
                self.df,
                kept_skus,
            )
        else:
            _run_and_store(
                "assortment",
                "evaluate_assortment",
                evaluate_assortment,
                self.df["stockcode"].unique()[:10],
                self.df,
            )
            _run_and_store(
                "assortment",
                "compare_assortment_scenarios",
                compare_assortment_scenarios,
                self.df,
                self.df["stockcode"].unique()[:10],
            )

        # CDT
        from src.analytics.cdt import (
            build_cdt,
            build_similarity_matrix,
            build_transaction_derived_attributes,
            get_cluster_assignments,
            tree_to_dataframe,
        )

        attrs = _run_and_store(
            "cdt",
            "build_transaction_derived_attributes",
            build_transaction_derived_attributes,
            self.df,
        )
        if attrs.success:
            sim = _run_and_store("cdt", "build_similarity_matrix", build_similarity_matrix, self.df)
            if sim.success:
                _run_and_store(
                    "cdt",
                    "get_cluster_assignments",
                    get_cluster_assignments,
                    self.df,
                    sim.diagnostics.get("output", sim.output_shape and self.df),
                )
                tree = _run_and_store(
                    "cdt",
                    "build_cdt",
                    build_cdt,
                    attrs.diagnostics.get("output", attrs.output_shape and self.df),
                    sim.diagnostics.get("output", sim.output_shape and self.df),
                )
                if tree.success:
                    _run_and_store(
                        "cdt",
                        "tree_to_dataframe",
                        tree_to_dataframe,
                        tree.diagnostics.get("output", tree.output_shape and self.df),
                    )

        # Segmentation
        from src.analytics.segmentation import (
            behavioral_segmentation,
            compute_rfm_features,
            rfm_segmentation,
            value_based_segmentation,
        )

        rfm = _run_and_store("segmentation", "compute_rfm_features", compute_rfm_features, self.df)
        if rfm.success:
            _run_and_store(
                "segmentation",
                "rfm_segmentation",
                rfm_segmentation,
                rfm.diagnostics.get("output", rfm.output_shape and self.df),
            )
        _run_and_store("segmentation", "behavioral_segmentation", behavioral_segmentation, self.df)
        _run_and_store(
            "segmentation", "value_based_segmentation", value_based_segmentation, self.df
        )

        # Pricing
        from src.analytics.pricing import (
            compute_kvi_score,
            diagnose_price_curves_1d,
            estimate_hierarchical_elasticity,
            estimate_loglog_elasticity,
            iv_elasticity_manual_2sls,
            local_price_response,
        )

        elast = _run_and_store(
            "pricing", "estimate_loglog_elasticity", estimate_loglog_elasticity, self.df
        )
        if elast.success:
            _run_and_store(
                "pricing",
                "estimate_hierarchical_elasticity",
                estimate_hierarchical_elasticity,
                self.df,
            )
            _run_and_store(
                "pricing",
                "compute_kvi_score",
                compute_kvi_score,
                self.df,
                elast.diagnostics.get("output", elast.output_shape and self.df),
            )
        _run_and_store("pricing", "diagnose_price_curves_1d", diagnose_price_curves_1d, self.df)
        # Only run IV if cost column exists
        if "cost" in self.df.columns:
            _run_and_store(
                "pricing", "iv_elasticity_manual_2sls", iv_elasticity_manual_2sls, self.df, "cost"
            )
        _run_and_store("pricing", "local_price_response", local_price_response, self.df)

        return self.results

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame summary of all validation results."""
        rows = []
        for r in self.results:
            rows.append(
                {
                    "module": r.module,
                    "function": r.function,
                    "success": r.success,
                    "output_shape": str(r.output_shape) if r.output_shape else None,
                    "n_columns": len(r.output_columns) if r.output_columns else None,
                    "error": r.error,
                }
            )
        return pd.DataFrame(rows)

    def outputs(self) -> Dict[str, pd.DataFrame]:
        """Map contract-relevant results to their output DataFrames."""
        collected: Dict[str, pd.DataFrame] = {}
        for r in self.results:
            if not r.success:
                continue
            output = r.diagnostics.get("output")
            if isinstance(output, pd.DataFrame) and not output.empty:
                collected[r.function] = output
        return collected

    def validate_cross_contracts(self) -> List[str]:
        """Run referential-integrity checks across collected outputs."""
        from src.analytics.schemas import validate_referential_integrity

        outputs = self.outputs()
        alias = {
            "compute_demand_transference_matrix": "demand_transference",
            "compute_switching_matrix": "switching_matrix",
            "get_top_affinity_pairs": "affinity_pairs",
            "compute_customer_entropy": "customer_entropy",
            "compute_rfm_features": "rfm_features",
            "compute_clv_customer_df": "clv_customer",
            "score_uplift_by_customer": "uplift_scores",
        }
        named = {}
        for fn_name, df in outputs.items():
            name = alias.get(fn_name, fn_name)
            if name not in named or df is not outputs[name]:
                named[name] = df
        named["transactions"] = self.df
        return validate_referential_integrity(named, {})

    def print_summary(self) -> None:
        """Print a formatted summary to stdout."""
        df = self.summary()
        print(f"Validation Summary: {df['success'].sum()}/{len(df)} passed")
        for _, row in df.iterrows():
            status = "✓" if row["success"] else "✗"
            print(
                f"  {status} {row['module']}.{row['function']}: {row['output_shape']} {row['error'] or ''}"
            )
        cross = self.validate_cross_contracts()
        for msg in cross:
            print(f"  ⚠ cross-contract: {msg}")


def export_baseline(harness: ValidationHarness, path: str) -> None:
    """Write a regression baseline (shapes/columns per function) to JSON."""
    baseline: Dict[str, Any] = {}
    for r in harness.results:
        if r.success and r.output_shape:
            baseline[r.function] = {
                "success": r.success,
                "shape": list(r.output_shape),
                "columns": r.output_columns,
            }
    with open(path, "w") as fh:
        json.dump(baseline, fh, indent=2, sort_keys=True)


def assert_validation(harness: ValidationHarness, baseline_path: str) -> List[str]:
    """Compare harness results against a frozen baseline.

    Returns a list of discrepancies (empty if in compliance). Raises
    AssertionError if the harness itself has failures.
    """
    import os

    failures = [r for r in harness.results if not r.success]
    if failures:
        details = "; ".join(f"{r.module}.{r.function}: {r.error}" for r in failures)
        raise AssertionError(f"Validation harness failures: {details}")

    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"Baseline not found: {baseline_path}. Run export_baseline() first."
        )

    with open(baseline_path) as fh:
        baseline = json.load(fh)

    discrepancies = []
    current = {r.function: r for r in harness.results if r.success and r.output_shape}
    for fn_name, expected in baseline.items():
        result = current.get(fn_name)
        if result is None:
            discrepancies.append(f"{fn_name}: missing from current run")
            continue
        if expected.get("success") and not result.success:
            discrepancies.append(f"{fn_name}: now failing ({result.error})")
            continue
        expected_shape = tuple(expected["shape"]) if expected.get("shape") else None
        if expected_shape and result.output_shape != expected_shape:
            discrepancies.append(
                f"{fn_name}: shape {result.output_shape} != baseline {expected_shape}"
            )
        if expected.get("columns") and result.output_columns != expected["columns"]:
            discrepancies.append(f"{fn_name}: columns changed")
    # Skip check for new functions not in baseline (allow expansion)
    return discrepancies


def run_validation(transactions_df: pd.DataFrame) -> ValidationHarness:
    """Convenience function to run full validation and return harness."""
    harness = ValidationHarness(transactions_df)
    harness.run_all()
    return harness
