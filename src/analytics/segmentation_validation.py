"""Synthetic-data validation suite for segmentation methods.

Generates data with known ground-truth customer segments and benchmarks
recovery accuracy across RFM Quantile, RFM K-Means, and Behavioral Clustering.
"""

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def generate_synthetic_customer_segments(
    n_customers: int = 200,
    n_true_segments: int = 4,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Generate synthetic transaction data with known customer segment labels.

    Each customer belongs to one of n_true_segments with distinct purchasing
    behaviors (frequency, basket size, price sensitivity, category breadth).

    Returns
    -------
    (transactions_df, true_segment_labels)
        transactions_df has standard transaction columns.
        true_segment_labels maps customer_id -> segment_id
    """
    rng = np.random.default_rng(random_seed)
    customers = [f"C_{i:04d}" for i in range(n_customers)]
    products = [f"P_{i:03d}" for i in range(60)]
    categories = ["CatA", "CatB", "CatC", "CatD", "CatE"]
    dates = pd.date_range("2024-01-01", periods=52, freq="W")
    date_strs = [str(d.date()).replace("-", "") for d in dates]

    # Segment traits: (avg_freq, avg_basket, price_sensitivity, n_categories)
    segment_traits = {
        0: (12, 5, 0.3, 3),  # High-value: frequent, large baskets, moderate price sensitivity
        1: (6, 3, 0.5, 4),  # Variety seekers: moderate freq, high category breadth
        2: (3, 2, 0.1, 2),  # Loyal: infrequent, small baskets, low price sensitivity
        3: (8, 4, 0.7, 2),  # Deal-driven: frequent, moderate baskets, high price sensitivity
        4: (2, 1, 0.4, 1),  # Light: infrequent, small baskets, narrow focus
        5: (15, 6, 0.2, 5),  # Power shoppers: very frequent, large baskets, broad
    }

    # Assign customers to segments
    customers_per_seg = max(1, n_customers // n_true_segments)
    true_labels: Dict[str, int] = {}
    for i, cust in enumerate(customers):
        cid = min(i // customers_per_seg, n_true_segments - 1)
        true_labels[cust] = cid

    rows = []
    for cust in customers:
        cid = true_labels[cust]
        freq, basket, price_sens, n_cats = segment_traits.get(cid, (4, 3, 0.3, 2))
        n_txns = rng.poisson(freq) + 1

        for _ in range(n_txns):
            idx = rng.integers(len(dates))
            date = dates[idx]
            date_str = date_strs[idx]
            n_items = rng.poisson(basket) + 1
            for _ in range(n_items):
                cat_idx = rng.integers(min(n_cats, len(categories)))
                cat = categories[cat_idx]
                cat_products = [p for p in products if p.endswith(cat[-1])] or products
                chosen = rng.choice(cat_products)
                base_price = rng.uniform(1.0, 15.0)
                if rng.random() < price_sens:
                    price = round(base_price * rng.uniform(0.6, 0.85), 2)
                else:
                    price = round(base_price * rng.uniform(0.9, 1.1), 2)
                qty = rng.poisson(1) + 1
                tid = f"{cust}_{date_str}_{rng.integers(1000)}"
                rows.append({
                    "date": date,
                    "transaction_id": tid,
                    "stockcode": chosen,
                    "product": f"Product {chosen}",
                    "customer_id": cust,
                    "price": price,
                    "quantity": qty,
                    "category": cat,
                })

    return pd.DataFrame(rows), true_labels


def _adjusted_rand_index(true: Dict[str, int], predicted: pd.Series) -> float:
    from sklearn.metrics import adjusted_rand_score

    common = [k for k in true if k in predicted.index]
    if len(common) < 2:
        return 0.0
    true_vec = [true[k] for k in common]
    pred_vec = [predicted[k] for k in common]
    return adjusted_rand_score(true_vec, pred_vec)


def _normalized_mutual_info(true: Dict[str, int], predicted: pd.Series) -> float:
    from sklearn.metrics import normalized_mutual_info_score

    common = [k for k in true if k in predicted.index]
    if len(common) < 2:
        return 0.0
    true_vec = [true[k] for k in common]
    pred_vec = [predicted[k] for k in common]
    return normalized_mutual_info_score(true_vec, pred_vec)


def run_segmentation_validation(
    n_customers: int = 200,
    n_true_segments: int = 4,
    methods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run segmentation methods against synthetic ground-truth segments.

    Parameters
    ----------
    methods : list of str
        Segmentation methods to benchmark. Defaults to:
        ["rfm_quantile", "rfm_kmeans", "behavioral"]

    Returns
    -------
    DataFrame with columns:
        method, adjusted_rand_index, normalized_mutual_info, n_segments_found,
        n_true_segments, runtime_seconds
    """
    if methods is None:
        methods = ["rfm_quantile", "rfm_kmeans", "behavioral"]

    df, true_labels = generate_synthetic_customer_segments(
        n_customers=n_customers,
        n_true_segments=n_true_segments,
        random_seed=42,
    )

    results = []

    for method in methods:
        t0 = time.time()
        try:
            if method == "rfm_quantile":
                from .segmentation import compute_rfm_features, rfm_segmentation

                rfm = compute_rfm_features(df)
                seg = rfm_segmentation(rfm, method="quantile", n_segments=8)
                predicted = seg.set_index("customer_id")["segment"].rank(method="dense")

            elif method == "rfm_kmeans":
                from .segmentation import compute_rfm_features, rfm_segmentation

                n_seg = min(n_true_segments + 2, 8)
                rfm = compute_rfm_features(df)
                seg = rfm_segmentation(rfm, method="kmeans", n_segments=n_seg)
                predicted = seg.set_index("customer_id")["cluster"].astype(int)

            elif method == "behavioral":
                from .segmentation import behavioral_segmentation

                n_seg = min(n_true_segments + 2, 8)
                seg = behavioral_segmentation(df, n_clusters=n_seg)
                predicted = seg.set_index("customer_id")["cluster"].astype(int)

            else:
                continue

            if predicted.dtype == "object":
                predicted = predicted.rank(method="dense")

            ari = _adjusted_rand_index(true_labels, predicted)
            nmi = _normalized_mutual_info(true_labels, predicted)
            n_found = predicted.nunique()
            runtime = time.time() - t0

            results.append({
                "method": method,
                "adjusted_rand_index": round(ari, 4),
                "normalized_mutual_info": round(nmi, 4),
                "n_segments_found": n_found,
                "n_true_segments": n_true_segments,
                "runtime_seconds": round(runtime, 3),
            })
        except Exception as exc:
            runtime = time.time() - t0
            results.append({
                "method": method,
                "adjusted_rand_index": -1.0,
                "normalized_mutual_info": -1.0,
                "n_segments_found": 0,
                "n_true_segments": n_true_segments,
                "runtime_seconds": round(runtime, 3),
                "error": str(exc),
            })

    return pd.DataFrame(results)
