from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path (script is at scripts/benchmark.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.analytics.sample_data import generate_transactions
from src.performance.profiler import measure_analysis, get_profiler_stats, profiler_state


def make_dataset(size_name):
    sizes = {
        "10k": dict(n_customers=200, n_products=50, n_days=90, seed=42),
        "10000": dict(n_customers=200, n_products=50, n_days=90, seed=42),
        "50k": dict(n_customers=500, n_products=100, n_days=180, seed=42),
        "100k": dict(n_customers=1000, n_products=200, n_days=365, seed=42),
        "250k": dict(n_customers=2000, n_products=200, n_days=365, seed=42),
        "500k": dict(n_customers=4000, n_products=200, n_days=365, seed=42),
    }
    params = sizes[size_name]
    df = generate_transactions(**params)
    return df


@measure_analysis
def benchmark_basket_penetration(df):
    from src.analytics.basket_metrics import compute_basket_penetration
    return compute_basket_penetration(df)


@measure_analysis
def benchmark_basket_penetration_over_time(df):
    from src.analytics.basket_metrics import basket_penetration_over_time
    return basket_penetration_over_time(df, period="W")


@measure_analysis
def benchmark_basket_composition(df):
    from src.analytics.basket_metrics import compute_basket_composition
    return compute_basket_composition(df)


@measure_analysis
def benchmark_customer_entropy(df):
    from src.analytics.basket_metrics import compute_customer_entropy
    return compute_customer_entropy(df)


@measure_analysis
def benchmark_cohort_sizes(df):
    from src.analytics.cohort import compute_cohort_sizes
    return compute_cohort_sizes(df, cohort_period="M")


@measure_analysis
def benchmark_cohort_retention(df):
    from src.analytics.cohort import compute_cohorts
    return compute_cohorts(df, cohort_period="M")


@measure_analysis
def benchmark_assortment(df):
    from src.analytics.assortment import evaluate_assortment
    skus = df["stockcode"].unique().tolist()[:10]
    result = evaluate_assortment(skus, df)
    return {"n_skus_evaluated": len(skus), "result": result}


@measure_analysis
def benchmark_rules_fpgrowth(df):
    from src.analytics.rules import run_fpgrowth
    basket = df.groupby(["transaction_id", "stockcode"]).size().unstack(fill_value=0).clip(upper=1).astype(bool)
    return run_fpgrowth(basket, min_support=0.05, max_len=3)


@measure_analysis
def benchmark_clv_bg_nbd(df):
    from src.analytics.clv import predict_clv_bg_nbd
    n_customers = min(50, df["customer_id"].nunique())
    customer_ids = df["customer_id"].unique()[:n_customers]
    try:
        result = predict_clv_bg_nbd(df, prediction_horizon_days=30, freq="D")
        return {"n_customers": n_customers, "predictions_shape": result[0].shape if hasattr(result[0], "shape") else len(result[0])}
    except Exception as e:
        return {"n_customers": n_customers, "error": str(e)[:60]}


@measure_analysis
def benchmark_clv_gamma_gamma(df):
    from src.analytics.clv import predict_clv_bg_nbd
    n_customers = min(50, df["customer_id"].nunique())
    try:
        result = predict_clv_bg_nbd(df, prediction_horizon_days=30, freq="D")
        preds = result[0]
        if hasattr(preds, "shape"):
            return {"n_customers": n_customers, "predictions_rows": len(preds)}
        return {"n_customers": n_customers, "predictions_rows": len(preds)}
    except Exception as e:
        return {"n_customers": n_customers, "error": str(e)[:60]}


@measure_analysis
def benchmark_performance_abc(df):
    from src.analytics.performance import abc_analysis
    result = abc_analysis(df)
    return {"n_skus": len(df["stockcode"].unique()), "abc_result": result}


BENCHMARKS = [
    ("basket_penetration", benchmark_basket_penetration),
    ("basket_penetration_over_time", benchmark_basket_penetration_over_time),
    ("basket_composition", benchmark_basket_composition),
    ("customer_entropy", benchmark_customer_entropy),
    ("cohort_sizes", benchmark_cohort_sizes),
    ("cohort_retention", benchmark_cohort_retention),
    ("assortment", benchmark_assortment),
    ("rules_fpgrowth", benchmark_rules_fpgrowth),
    ("clv_bg_nbd", benchmark_clv_bg_nbd),
    ("clv_gamma_gamma", benchmark_clv_gamma_gamma),
    ("performance_abc", benchmark_performance_abc),
]


def run_benchmarks(size_name):
    df = make_dataset(size_name)
    n_rows = len(df)
    n_customers = df["customer_id"].nunique()
    n_skus = df["stockcode"].nunique()
    profiler_state._session.clear()

    results = {
        "dataset_size": size_name,
        "n_rows": n_rows,
        "n_customers": n_customers,
        "n_skus": n_skus,
        "modules": [],
    }

    for name, func in BENCHMARKS:
        try:
            start = time.perf_counter()
            result = func(df)
            duration_ms = (time.perf_counter() - start) * 1000
            stats = get_profiler_stats(func.__name__)
            module_result = {
                "name": name,
                "duration_ms": round(duration_ms, 2),
                "input_rows": n_rows,
                "output_rows": result if isinstance(result, int) else (
                    len(result) if isinstance(result, (list, pd.Series)) else
                    result.shape[0] if hasattr(result, "shape") else 1
                ),
                "cache_hit": False,
            }
            if stats:
                module_result.update({
                    "avg_duration_ms": round(stats.get("avg_duration_ms", duration_ms), 2),
                    "cache_hit_rate": round(stats.get("cache_hit_rate", 0), 3),
                    "peak_memory_mb": round(stats.get("peak_memory_mb", 0), 2),
                })
            results["modules"].append(module_result)
        except Exception as e:
            results["modules"].append({
                "name": name,
                "duration_ms": None,
                "error": str(e),
            })
    return results


def main():
    parser = argparse.ArgumentParser(description="Market Basket App Benchmarks")
    parser.add_argument("--size", choices=["10k", "10000", "50k", "100k", "250k", "500k"], required=True)
    parser.add_argument("--output", default="benchmarks/baseline.json")
    args = parser.parse_args()
    results = run_benchmarks(args.size)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
