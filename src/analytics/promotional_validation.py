"""Synthetic-data validation suite for promo detection methods.

Simulates a product with known promo weeks (planted price drops) and measures
precision/recall of promo-week detection for both the legacy % threshold
and the new adaptive z-score method.
"""

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def generate_synthetic_promo_data(
    n_weeks: int = 52,
    n_products: int = 3,
    promo_weeks_fraction: float = 0.15,
    price_drop_pct_range: tuple = (0.1, 0.3),
    random_seed: int = 42,
) -> tuple[pd.DataFrame, Dict[str, List[int]]]:
    """Generate weekly transaction data with known promo weeks.

    For each product, a fraction of weeks are designated as "promo" with
    price drops in the given range. The rest are regular price weeks.

    Returns
    -------
    (transactions_df, true_promo_weeks)
        transactions_df has weekly price/quantity per product.
        true_promo_weeks maps stockcode -> list of week indices with promos.
    """
    rng = np.random.default_rng(random_seed)
    dates = pd.date_range("2024-01-01", periods=n_weeks, freq="W")
    date_strs = [str(d.date()).replace("-", "") for d in dates]

    rows = []
    true_promo_weeks: Dict[str, List[int]] = {}

    for pid in range(n_products):
        sku = f"PROD_{pid+1:03d}"
        base_price = rng.uniform(5.0, 20.0)
        promo_weeks = set(
            rng.choice(n_weeks, size=max(1, int(n_weeks * promo_weeks_fraction)), replace=False)
        )
        true_promo_weeks[sku] = sorted(promo_weeks)

        for w in range(n_weeks):
            if w in promo_weeks:
                drop = rng.uniform(*price_drop_pct_range)
                price = base_price * (1 - drop)
            else:
                price = base_price * rng.normal(1, 0.05)

            price = max(0.1, round(price, 2))
            qty = max(1, int(rng.poisson(10 + (20 if w in promo_weeks else 0))))
            tid = f"{sku}_{w:03d}"
            rows.append({
                "date": dates[w],
                "transaction_id": tid,
                "stockcode": sku,
                "product": f"Product {sku}",
                "customer_id": f"CUST_{sku}_{w % 10}",
                "price": price,
                "quantity": qty,
            })

    return pd.DataFrame(rows), true_promo_weeks


def _promo_week_precision_recall(
    detected: pd.DataFrame,
    true_promo_weeks: Dict[str, List[int]],
) -> Dict[str, float]:
    """Compute precision, recall, and F1 of promo week detection."""
    true_weeks: Dict[str, set] = {}
    for sku, weeks in true_promo_weeks.items():
        true_weeks[sku] = set(weeks)

    detected_weeks: Dict[str, set] = {}
    if not detected.empty:
        for sku in true_promo_weeks:
            prod_detected = detected[detected["stockcode"] == sku]
            weeks = set()
            for _, row in prod_detected.iterrows():
                start = pd.Timestamp(row["start_date"])
                end = pd.Timestamp(row["end_date"])
                for d in pd.date_range(start, end, freq="W"):
                    week_num = (d - pd.Timestamp("2024-01-01")).days // 7
                    if 0 <= week_num < 52:
                        weeks.add(week_num)
            detected_weeks[sku] = weeks

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for sku in true_promo_weeks:
        t = true_weeks.get(sku, set())
        d = detected_weeks.get(sku, set())
        total_tp += len(t & d)
        total_fp += len(d - t)
        total_fn += len(t - d)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
    }


def run_promo_detection_validation(
    n_weeks: int = 52,
    n_products: int = 3,
    promo_weeks_fraction: float = 0.15,
    price_drop_pct_range: tuple = (0.1, 0.3),
    methods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Benchmark promo detection methods against synthetic ground truth.

    Parameters
    ----------
    methods : list of str
        Detection methods to benchmark. Defaults to:
        ["fixed_threshold", "adaptive_zscore"]

    Returns
    -------
    DataFrame with columns:
        method, precision, recall, f1, true_positives, false_positives,
        false_negatives, runtime_seconds
    """
    if methods is None:
        methods = ["fixed_threshold", "adaptive_zscore"]

    df, true_promo_weeks = generate_synthetic_promo_data(
        n_weeks=n_weeks,
        n_products=n_products,
        promo_weeks_fraction=promo_weeks_fraction,
        price_drop_pct_range=price_drop_pct_range,
        random_seed=42,
    )

    results = []

    for method in methods:
        t0 = time.time()
        try:
            if method == "fixed_threshold":
                from .promotional import detect_promotions

                detected = detect_promotions(
                    df,
                    price_change_threshold=0.1,
                    min_duration_days=1,
                    max_duration_days=30,
                )
            elif method == "adaptive_zscore":
                from .promotional import detect_promotions_adaptive

                detected = detect_promotions_adaptive(
                    df,
                    baseline_window=12,
                    z_score_threshold=-2.0,
                    min_duration_days=1,
                    max_duration_days=30,
                )
            else:
                continue

            metrics = _promo_week_precision_recall(detected, true_promo_weeks)
            runtime = time.time() - t0

            results.append({
                "method": method,
                **metrics,
                "runtime_seconds": round(runtime, 3),
            })
        except Exception as exc:
            runtime = time.time() - t0
            results.append({
                "method": method,
                "precision": -1.0,
                "recall": -1.0,
                "f1": -1.0,
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "runtime_seconds": round(runtime, 3),
                "error": str(exc),
            })

    return pd.DataFrame(results)
