"""CDT validation harness: synthetic data + ARI/NMI against ground truth."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.analytics.schemas import CDT_VALIDATION, check


def generate_synthetic_cluster_data(
    n_products: int = 30,
    n_true_clusters: int = 3,
    n_customers: int = 200,
    noise_level: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Synthetic transaction data with known product cluster structure.

    Each product belongs to a true cluster. Customers prefer their cluster
    with probability 1 - noise_level; otherwise they buy randomly across
    all products.
    """
    rng = np.random.default_rng(random_seed)
    products_per_cluster = max(1, n_products // n_true_clusters)
    products = [f"P{i + 1:03d}" for i in range(n_products)]
    categories = [f"Cat{chr(65 + i)}" for i in range(n_true_clusters)]

    true_labels: dict[str, int] = {}
    product_to_cat: dict[str, str] = {}
    for i, p in enumerate(products):
        cid = min(i // products_per_cluster, n_true_clusters - 1)
        true_labels[p] = cid
        product_to_cat[p] = categories[cid]

    dates = pd.date_range("2024-01-01", periods=52, freq="W")
    date_strs = [d.strftime("%Y%m%d") for d in dates]
    rows: list[dict[str, object]] = []

    for cid in range(n_true_clusters):
        cluster_products = [p for p in products if true_labels[p] == cid]
        cluster_customers = n_customers // n_true_clusters
        start = cid * cluster_customers
        end = start + cluster_customers if cid < n_true_clusters - 1 else n_customers
        customer_ids = [f"C_{cid}_{j}" for j in range(end - start)]

        for cust in customer_ids:
            n_purchases = int(rng.poisson(8) + 1)
            for _ in range(n_purchases):
                if rng.random() > noise_level:
                    chosen = rng.choice(cluster_products)
                else:
                    chosen = rng.choice(products)
                idx = int(rng.integers(len(dates)))
                date = dates[idx]
                date_str = date_strs[idx]
                qty = int(rng.poisson(2) + 1)
                price = round(float(rng.uniform(1.0, 10.0)), 2)
                tid = f"{cust}_{date_str}_{int(rng.integers(100))}"
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


def run_cdt_validation(
    transactions_df: pd.DataFrame,
    true_labels: dict[str, int],
    similarity_method: str = "phi",
    clustering_method: str = "ward",
    n_clusters: int | None = None,
    min_cooccurrence: int = 5,
) -> pd.DataFrame:
    """End-to-end CDT pipeline + ARI / NMI against ground truth.

    Builds similarity, clusters, then compares recovered clusters to the
    supplied ``true_labels``. Returns a small metrics table.
    """
    from src.analytics.cdt.clustering import (
        find_optimal_clusters_sklearn,
        perform_hierarchical_clustering,
    )
    from src.analytics.cdt.similarity import build_similarity_matrix

    if n_clusters is None:
        opt = find_optimal_clusters_sklearn(
            build_similarity_matrix(
                transactions_df, method=similarity_method, min_cooccurrence=min_cooccurrence
            ),
            max_clusters=min(10, transactions_df["stockcode"].nunique() - 1),
        )
        if opt.empty:
            n_clusters = len(set(true_labels.values()))
        else:
            n_clusters = int(opt.loc[opt["silhouette"].idxmax(), "n_clusters"])

    sim = build_similarity_matrix(
        transactions_df, method=similarity_method, min_cooccurrence=min_cooccurrence
    )
    linkage_matrix, pred_clusters = perform_hierarchical_clustering(
        sim, method=clustering_method, n_clusters=n_clusters
    )
    pred = pred_clusters.to_dict()

    common = [k for k in true_labels if k in pred]
    if len(common) < 2:
        ari = 0.0
        nmi = 0.0
    else:
        true_vec = [true_labels[k] for k in common]
        pred_vec = [pred[k] for k in common]
        ari = float(adjusted_rand_score(true_vec, pred_vec))
        nmi = float(normalized_mutual_info_score(true_vec, pred_vec))

    rows = [
        {"method": "ARI", "metric": "adjusted_rand_index", "value": ari},
        {"method": "NMI", "metric": "normalized_mutual_info", "value": nmi},
        {"method": "pipeline", "metric": "n_clusters_used", "value": float(n_clusters)},
    ]
    return check(pd.DataFrame(rows, columns=list(CDT_VALIDATION.columns)), CDT_VALIDATION)
