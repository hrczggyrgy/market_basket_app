#!/usr/bin/env python3
"""Performance benchmark script.

Run with: ENGAGE_PROFILE=1 python scripts/benchmark_performance.py

Measures end-to-end timings for each major analytical component.
"""

import os
import sys
import time

os.environ["ENGAGE_PROFILE"] = "1"

from pathlib import Path

# Ensure src is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics.sample_data import generate_transactions
from src.analytics.feature_store import build_feature_store
from src.analytics.cdt.embedding import build_product_embeddings, build_topk_neighbors
from src.analytics.cdt.similarity import build_similarity_matrix
from src.analytics.cdt.community import build_product_graph
from src.analytics.performance import compute_product_metrics
from src.analytics.pricing.elasticity import estimate_loglog_elasticity
from src.analytics.pricing.kvi import compute_kvi_score
from src.analytics.promo import detect_promotions
from src.analytics.rules import create_basket_matrix, run_fpgrowth, generate_rules
from src.analytics.segmentation import behavioral_segmentation
from src.analytics.clv import _fit_bg_nbd as fit_bg_nbd, _fit_gamma_gamma as fit_gamma_gamma, predict_clv_bg_nbd as predict_clv


def _timeit(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


def main():
    print("=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Generate test data
    print("\nGenerating test data...")
    df = generate_transactions(n_days=365, n_products=2000, n_customers=5000, seed=42)
    print(f"  Rows: {len(df):,}, Customers: {df['customer_id'].nunique():,}, Products: {df['stockcode'].nunique():,}")
    
    results = {}
    
    # 1. Feature Store
    print("\n1. Feature Store")
    t, fs = _timeit(build_feature_store, df)
    print(f"   build_feature_store: {t:.3f}s  (products={len(fs.products)}, customers={len(fs.customers)}, sparse nnz={fs.customer_product_binary.nnz:,})")
    results["Feature Store"] = t
    
    # 2. CDT Embeddings
    print("\n2. CDT Embeddings")
    t, emb = _timeit(build_product_embeddings, fs.customer_product_binary, n_components=64)
    print(f"   TruncatedSVD: {t:.3f}s  shape={emb.shape}")
    results["TruncatedSVD"] = t
    
    t, (idx, dist) = _timeit(build_topk_neighbors, emb, top_k=50)
    print(f"   NearestNeighbors(top-50): {t:.3f}s")
    results["NearestNeighbors"] = t
    
    # 3. Similarity matrices (embedding method)
    print("\n3. Similarity Matrices")
    t, sim = _timeit(build_similarity_matrix, df, method="embedding", top_n_products=2000)
    print(f"   embedding (top-2000): {t:.3f}s  shape={sim.shape}")
    results["Embedding Similarity"] = t
    
    # Legacy phi (for comparison)
    t, sim_phi = _timeit(build_similarity_matrix, df, method="phi", min_cooccurrence=5)
    print(f"   phi (full): {t:.3f}s  shape={sim_phi.shape}")
    results["Phi Similarity"] = t
    
    # 4. Community Graph
    print("\n4. Community Graph")
    t, graph = _timeit(build_product_graph, df, min_cooccurrence=5)
    print(f"   build_product_graph: {t:.3f}s  nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")
    results["Product Graph"] = t
    
    # 5. Performance metrics
    print("\n5. Performance Metrics")
    t, perf = _timeit(compute_product_metrics, df)
    print(f"   compute_product_metrics: {t:.3f}s  rows={len(perf)}")
    results["Product Metrics"] = t
    
    # 6. Pricing - Elasticity
    print("\n6. Pricing - Elasticity")
    t, elast = _timeit(estimate_loglog_elasticity, df)
    print(f"   estimate_loglog_elasticity: {t:.3f}s  rows={len(elast)}")
    results["Elasticity"] = t
    
    # 7. Pricing - KVI
    print("\n7. Pricing - KVI")
    t, kvi = _timeit(compute_kvi_score, df)
    print(f"   compute_kvi_score: {t:.3f}s  rows={len(kvi)}")
    results["KVI"] = t
    
    # 8. Promo detection
    print("\n8. Promotions")
    t, promos = _timeit(detect_promotions, df)
    print(f"   detect_promotions: {t:.3f}s  periods={len(promos)}")
    results["Promo Detection"] = t
    
    # 9. Market Basket
    print("\n9. Market Basket")
    t, basket = _timeit(create_basket_matrix, df)
    print(f"   create_basket_matrix: {t:.3f}s  shape={basket.shape}")
    results["Basket Matrix"] = t
    
    t, freq = _timeit(run_fpgrowth, basket, min_support=0.01, max_len=3)
    print(f"   run_fpgrowth: {t:.3f}s  itemsets={len(freq)}")
    results["FP-Growth"] = t
    
    if not freq.empty:
        t, rules = _timeit(generate_rules, freq, metric="confidence", min_threshold=0.1)
        print(f"   generate_rules: {t:.3f}s  rules={len(rules)}")
        results["Rule Generation"] = t
    
    # 10. Segmentation
    print("\n10. Segmentation")
    t, seg = _timeit(behavioral_segmentation, df, n_clusters=4)
    print(f"   behavioral_segmentation: {t:.3f}s  segments={seg['segment'].nunique() if 'segment' in seg.columns else 'N/A'}")
    results["Behavioral Segmentation"] = t
    
    # 11. CLV
    print("\n11. CLV")
    t, bg = _timeit(fit_bg_nbd, df)
    print(f"   fit_bg_nbd: {t:.3f}s")
    results["BG-NBD"] = t
    
    t, gg = _timeit(fit_gamma_gamma, df)
    print(f"   fit_gamma_gamma: {t:.3f}s")
    results["Gamma-Gamma"] = t
    
    if bg and gg:
        t, clv = _timeit(predict_clv, df, bg, gg, time_horizon=52)
        print(f"   predict_clv: {t:.3f}s  customers={len(clv)}")
        results["CLV Prediction"] = t
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (sorted by time)")
    print("=" * 60)
    for name, t in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:30s}  {t:8.3f}s")
    print(f"  {'TOTAL':30s}  {sum(results.values()):8.3f}s")


if __name__ == "__main__":
    main()