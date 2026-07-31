"""Synthetic-data validation suite for elasticity estimators.

Generates data with known ground-truth elasticities and benchmarks
recovery accuracy across all four methods.
"""

import warnings

import numpy as np
import pandas as pd

from .pricing import (
    estimate_bayesian_hierarchical_elasticity,
    estimate_hierarchical_elasticity,
    estimate_loglog_elasticity,
)

warnings.filterwarnings("ignore")


def generate_synthetic_elasticity_data(
    n_skus: int = 20,
    n_weeks: int = 52,
    n_categories: int = 4,
    price_range: tuple = (0.5, 5.0),
    price_cv: float = 0.15,
    noise_scale: float = 0.15,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate weekly-level transaction data with known elasticities.

    For each SKU:  log(Q) = alpha + beta * log(P) + epsilon
    where beta (elasticity) is drawn per-SKU:  N(-1.2, 0.4) clipped to [-2.5, -0.2].

    Returns
    -------
    (transactions_df, ground_truth_df)
        transactions_df has columns: date, transaction_id, stockcode, product,
        customer_id, price, quantity, category
        ground_truth_df has columns: stockcode, category, true_elasticity, true_intercept
    """
    rng = np.random.default_rng(random_seed)

    categories = [f"Cat{chr(65 + i)}" for i in range(n_categories)]
    skus_per_cat = max(1, n_skus // n_categories)

    rows = []
    truth_rows = []

    # Weekly dates
    dates = pd.date_range("2024-01-01", periods=n_weeks, freq="W")

    for cat in categories:
        for i in range(skus_per_cat):
            sku = f"{cat}_{i + 1:03d}"

            # Ground-truth parameters
            true_alpha = rng.normal(4.0, 0.5)
            true_beta = rng.normal(-1.2, 0.4)
            true_beta = np.clip(true_beta, -2.5, -0.2)
            base_price = rng.uniform(*price_range)

            truth_rows.append(
                {
                    "stockcode": sku,
                    "category": cat,
                    "true_elasticity": true_beta,
                    "true_intercept": true_alpha,
                    "base_price": base_price,
                }
            )

            # Generate weekly observations
            for week_idx, d in enumerate(dates):
                p = base_price * (1 + rng.normal(0, price_cv))
                p = max(price_range[0] * 0.5, p)

                log_q = true_alpha + true_beta * np.log(p) + rng.normal(0, noise_scale)
                q = max(1, int(np.exp(log_q)))

                rows.append(
                    {
                        "date": d,
                        "transaction_id": f"{sku}_{week_idx:03d}",
                        "stockcode": sku,
                        "product": f"Product {sku}",
                        "customer_id": f"C_{cat}_{week_idx % 10}",
                        "price": round(p, 2),
                        "quantity": q,
                        "category": cat,
                    }
                )

    transactions_df = pd.DataFrame(rows)
    ground_truth_df = pd.DataFrame(truth_rows)
    return transactions_df, ground_truth_df


def _rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def _bias(actual, predicted):
    return np.mean(predicted - actual)


def _coverage(actual, lower, upper):
    return np.mean((actual >= lower) & (actual <= upper))


def run_validation(
    n_skus: int = 20,
    n_weeks: int = 52,
    n_categories: int = 4,
    n_samples: int = 300,
) -> pd.DataFrame:
    """Run all four elasticity estimators against synthetic ground truth.

    Returns
    -------
    DataFrame with columns:
        method, rmse, bias, coverage_94, n_products, avg_true_elasticity, avg_estimated
    """
    df, truth = generate_synthetic_elasticity_data(
        n_skus=n_skus, n_weeks=n_weeks, n_categories=n_categories
    )

    results = []

    # 1. Log-log OLS
    ols = estimate_loglog_elasticity(df, min_periods=5, min_price_variation=0.01)
    if not ols.empty:
        merged = ols.merge(truth, on="stockcode")
        results.append(
            {
                "method": "loglog_ols",
                "rmse": _rmse(merged["true_elasticity"], merged["elasticity"]),
                "bias": _bias(merged["true_elasticity"], merged["elasticity"]),
                "coverage_94": np.nan,
                "n_products": len(merged),
                "avg_true_elasticity": merged["true_elasticity"].mean(),
                "avg_estimated": merged["elasticity"].mean(),
            }
        )

    # 2. Empirical Bayes (hierarchical_eb)
    heb = estimate_hierarchical_elasticity(df, min_periods=5, min_price_variation=0.01)
    if not heb.empty:
        merged = heb.merge(truth, on="stockcode")
        results.append(
            {
                "method": "hierarchical_eb",
                "rmse": _rmse(merged["true_elasticity"], merged["elasticity_shrunk"]),
                "bias": _bias(merged["true_elasticity"], merged["elasticity_shrunk"]),
                "coverage_94": np.nan,
                "n_products": len(merged),
                "avg_true_elasticity": merged["true_elasticity"].mean(),
                "avg_estimated": merged["elasticity_shrunk"].mean(),
            }
        )

    # 3. Bayesian hierarchical (ADVI)
    bayes_advi = estimate_bayesian_hierarchical_elasticity(
        df,
        min_periods=5,
        min_price_variation=0.01,
        n_samples=n_samples,
        bayesian_mode="fast (ADVI)",
    )
    if not bayes_advi.empty:
        merged = bayes_advi.merge(truth, on="stockcode")
        coverage = _coverage(
            merged["true_elasticity"],
            merged["elasticity_hdi_lower"],
            merged["elasticity_hdi_upper"],
        )
        results.append(
            {
                "method": "bayesian_advi",
                "rmse": _rmse(merged["true_elasticity"], merged["elasticity_mean"]),
                "bias": _bias(merged["true_elasticity"], merged["elasticity_mean"]),
                "coverage_94": coverage,
                "n_products": len(merged),
                "avg_true_elasticity": merged["true_elasticity"].mean(),
                "avg_estimated": merged["elasticity_mean"].mean(),
            }
        )

    # 4. Bayesian hierarchical (NUTS) — fewer samples to keep runtime manageable
    try:
        bayes_nuts, _ = estimate_bayesian_hierarchical_elasticity(
            df,
            min_periods=5,
            min_price_variation=0.01,
            n_samples=n_samples // 2,
            n_tune=n_samples // 2,
            bayesian_mode="full (NUTS)",
            return_trace=True,
        )
        if not bayes_nuts.empty:
            merged = bayes_nuts.merge(truth, on="stockcode")
            coverage = _coverage(
                merged["true_elasticity"],
                merged["elasticity_hdi_lower"],
                merged["elasticity_hdi_upper"],
            )
            results.append(
                {
                    "method": "bayesian_nuts",
                    "rmse": _rmse(merged["true_elasticity"], merged["elasticity_mean"]),
                    "bias": _bias(merged["true_elasticity"], merged["elasticity_mean"]),
                    "coverage_94": coverage,
                    "n_products": len(merged),
                    "avg_true_elasticity": merged["true_elasticity"].mean(),
                    "avg_estimated": merged["elasticity_mean"].mean(),
                }
            )
    except Exception as e:
        # NUTS benchmark failed; log warning and continue
        import streamlit as st

        st.warning(f"NUTS benchmark failed: {e}")

    return pd.DataFrame(results)
