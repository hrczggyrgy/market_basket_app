"""Synthetic-data validation suite for CDT similarity/clustering methods.

Generates data with known ground-truth cluster assignments and benchmarks
recovery accuracy across legacy and ensemble similarity methods.
"""

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .cdt_clustering import (
    find_optimal_clusters,
    get_cluster_assignments,
    perform_hierarchical_clustering,
)
from .cdt_similarity import build_similarity_matrix


def generate_synthetic_cluster_data(
    n_products: int = 30,
    n_true_clusters: int = 3,
    n_customers: int = 200,
    noise_level: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Generate synthetic transaction data with known product cluster structure.

    Each product belongs to one of n_true_clusters. Customers have preferences
    for clusters, and noise_level controls how often they buy outside their
    preferred cluster.

    Returns
    -------
    (transactions_df, true_cluster_labels)
        transactions_df has columns: date, transaction_id, stockcode, product,
        customer_id, price, quantity, category
        true_cluster_labels maps stockcode -> cluster_id
    """
    rng = np.random.default_rng(random_seed)
    products_per_cluster = max(1, n_products // n_true_clusters)
    products = [f"P{i + 1:03d}" for i in range(n_products)]
    categories = [f"Cat{chr(65 + i)}" for i in range(n_true_clusters)]

    true_labels: Dict[str, int] = {}
    product_to_cat: Dict[str, str] = {}
    for i, p in enumerate(products):
        cid = i // products_per_cluster
        cid = min(cid, n_true_clusters - 1)
        true_labels[p] = cid
        product_to_cat[p] = categories[cid]

    dates = pd.date_range("2024-01-01", periods=52, freq="W")
    date_strs = [d.strftime("%Y%m%d") for d in dates]
    rows = []

    for cid in range(n_true_clusters):
        cluster_products = [p for p in products if true_labels[p] == cid]
        cluster_customers = n_customers // n_true_clusters
        start = cid * cluster_customers
        end = start + cluster_customers if cid < n_true_clusters - 1 else n_customers
        customer_ids = [f"C_{cid}_{j}" for j in range(end - start)]

        for cust in customer_ids:
            n_purchases = rng.poisson(8) + 1
            for _ in range(n_purchases):
                if rng.random() > noise_level:
                    chosen = rng.choice(cluster_products)
                else:
                    chosen = rng.choice(products)

                idx = rng.integers(len(dates))
                date = dates[idx]
                date_str = date_strs[idx]
                qty = rng.poisson(2) + 1
                price = round(rng.uniform(1.0, 10.0), 2)
                tid = f"{cust}_{date_str}_{rng.integers(100)}"
                rows.append(
                    {
                        "date": date,
                        "transaction_id": tid,
                        "stockcode": chosen,
                        "product": f"Product {chosen}",
                        "customer_id": cust,
                        "price": price,
                        "quantity": qty,
                        "category": product_to_cat[chosen],
                    }
                )

    return pd.DataFrame(rows), true_labels


def _adjusted_rand_index(true: Dict[str, int], predicted: Dict[str, int]) -> float:
    from sklearn.metrics import adjusted_rand_score

    common = [k for k in true if k in predicted]
    if len(common) < 2:
        return 0.0
    true_vec = [true[k] for k in common]
    pred_vec = [predicted[k] for k in common]
    return adjusted_rand_score(true_vec, pred_vec)


def _normalized_mutual_info(true: Dict[str, int], predicted: Dict[str, int]) -> float:
    from sklearn.metrics import normalized_mutual_info_score

    common = [k for k in true if k in predicted]
    if len(common) < 2:
        return 0.0
    true_vec = [true[k] for k in common]
    pred_vec = [predicted[k] for k in common]
    return normalized_mutual_info_score(true_vec, pred_vec)


def run_cdt_validation(
    n_products: int = 30,
    n_true_clusters: int = 3,
    n_customers: int = 200,
    noise_level: float = 0.2,
    min_cooccurrence: int = 2,
    min_k: int = 2,
    max_k: int = 10,
    methods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run CDT similarity methods against synthetic ground-truth clusters.

    Parameters
    ----------
    methods : list of str
        Similarity methods to benchmark. Defaults to:
        ["legacy_yules_q", "legacy_jaccard", "ensemble_phi_jaccard_pmi_tfidf"]

    Returns
    -------
    DataFrame with columns:
        method, adjusted_rand_index, normalized_mutual_info, n_clusters_found,
        n_true_clusters, runtime_seconds
    """
    if methods is None:
        methods = [
            "legacy_phi",
            "legacy_jaccard",
            "ensemble_phi_jaccard_pmi_tfidf",
        ]

    df, true_labels = generate_synthetic_cluster_data(
        n_products=n_products,
        n_true_clusters=n_true_clusters,
        n_customers=n_customers,
        noise_level=noise_level,
        random_seed=42,
    )

    # Map legacy names to method parameter for build_similarity_matrix
    method_to_param = {
        "legacy_phi": "phi",
        "legacy_jaccard": "jaccard",
        "legacy_yules_q": "phi",
        "ensemble_phi_jaccard_pmi_tfidf": "ensemble",
    }

    results = []

    for method in methods:
        t0 = time.time()
        try:
            sim_param = method_to_param.get(method, method)

            sim = build_similarity_matrix(
                df,
                method=sim_param,
                min_cooccurrence=min_cooccurrence,
            )

            if sim.empty or len(sim) < 2:
                runtime = time.time() - t0
                results.append(
                    {
                        "method": method,
                        "adjusted_rand_index": 0.0,
                        "normalized_mutual_info": 0.0,
                        "n_clusters_found": 1,
                        "n_true_clusters": n_true_clusters,
                        "n_products_matched": 0,
                        "runtime_seconds": round(runtime, 3),
                    }
                )
                continue

            linkage_matrix, _ = perform_hierarchical_clustering(
                sim, linkage_method="average", distance_method="phi"
            )

            optimal_k, _ = find_optimal_clusters(
                linkage_matrix,
                sim,
                distance_method="phi",
                min_clusters=min_k,
                max_clusters=min(max_k, len(sim) - 1),
            )

            predicted = get_cluster_assignments(linkage_matrix, sim, n_clusters=optimal_k)

            ari = _adjusted_rand_index(true_labels, predicted)
            nmi = _normalized_mutual_info(true_labels, predicted)
            runtime = time.time() - t0

            results.append(
                {
                    "method": method,
                    "adjusted_rand_index": round(ari, 4),
                    "normalized_mutual_info": round(nmi, 4),
                    "n_clusters_found": optimal_k,
                    "n_true_clusters": n_true_clusters,
                    "n_products_matched": len([k for k in true_labels if k in predicted]),
                    "runtime_seconds": round(runtime, 3),
                }
            )
        except Exception as exc:
            runtime = time.time() - t0
            results.append(
                {
                    "method": method,
                    "adjusted_rand_index": float("nan"),
                    "normalized_mutual_info": float("nan"),
                    "n_clusters_found": 0,
                    "n_true_clusters": n_true_clusters,
                    "n_products_matched": 0,
                    "runtime_seconds": round(runtime, 3),
                    "error": str(exc),
                }
            )

    return pd.DataFrame(results)
