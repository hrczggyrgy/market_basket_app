"""Test all analytics modules against the new FMCG generator output."""
import sys
import traceback as tb_module

results = []

def try_it(module_name, func_name, callable_ref, args=None, kwargs=None):
    """Try calling a function, record result or full traceback."""
    args = args or ()
    kwargs = kwargs or {}
    try:
        result = callable_ref(*args, **kwargs)
        results.append((module_name, func_name, "OK", None, type(result).__name__))
        return result
    except Exception as e:
        tb = "".join(tb_module.format_exception(*sys.exc_info()))
        results.append((module_name, func_name, "ERROR", str(e), tb))
        return None


def print_results():
    """Print organized results."""
    errors = [r for r in results if r[2] == "ERROR"]
    ok_count = len([r for r in results if r[2] == "OK"])

    print(f"\n{'='*80}")
    print(f"RESULTS: {len(results)} calls, {ok_count} OK, {len(errors)} ERRORS")
    print(f"{'='*80}\n")

    if not errors:
        print("ALL MODULES PASSED - No errors found!\n")
        return

    # Group by module
    from collections import defaultdict
    by_module = defaultdict(list)
    for r in errors:
        by_module[r[0]].append(r)

    for mod_name in sorted(by_module.keys()):
        errs = by_module[mod_name]
        print(f"\n{'#'*80}")
        print(f"### MODULE: {mod_name} ({len(errs)} error(s))")
        print(f"{'#'*80}\n")
        for _, func_name, _, msg, tb in errs:
            print(f"  Function: {func_name}")
            print(f"  Error: {msg}")
            print(f"  Full Traceback:")
            for line in tb.strip().split("\n"):
                print(f"    {line}")
            print()


# ============================================================
# Generate data
# ============================================================
print("Generating sample data (n_customers=200, seed=42)...")
from src.data.generator import generate_transactions
from src.algorithms.fpgrowth import create_basket_matrix, run_fpgrowth, get_product_lookup
import pandas as pd

df = generate_transactions(n_customers=200, seed=42)
product_lookup = get_product_lookup(df)
basket = create_basket_matrix(df)
freq_items = run_fpgrowth(basket, min_support=0.01, max_len=4)

print(f"Dataset: {len(df)} rows, {df['transaction_id'].nunique()} orders, {df['customer_id'].nunique()} customers, {df['stockcode'].nunique()} products")
print(f"Basket: {basket.shape}")
print(f"Freq itemsets: {len(freq_items)}\n")

# ============================================================
# 1. Association Rules
# ============================================================
print("-" * 60)
print("1. Association Rules (src.rules.generator)")
print("-" * 60)
try:
    from src.rules.generator import generate_rules, filter_rules
    print("  Imports OK")
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Association Rules", "import", "ERROR", str(e), tb))

# Test generate_rules with default params
try_it("Association Rules", "generate_rules(default)", generate_rules, [freq_items])
# Test with alternative params
try_it("Association Rules", "generate_rules(lift)", generate_rules, [freq_items], {"metric": "lift", "min_threshold": 1.2})

# If we got rules, test filter_rules
rules = None
for r in results:
    if r[0] == "Association Rules" and r[1] == "generate_rules(default)" and r[2] == "OK":
        # We need to call it again and capture the return
        pass

# Let me re-capture properly
rules_default = try_it("Association Rules", "generate_rules(default)", generate_rules, [freq_items])
rules_lift = try_it("Association Rules", "generate_rules(lift)", generate_rules, [freq_items], {"metric": "lift", "min_threshold": 1.2})

if rules_default is not None and not rules_default.empty:
    try_it("Association Rules", "filter_rules", filter_rules, [rules_default], {"min_confidence": 0.3, "min_lift": 1.0})
    try_it("Association Rules", "filter_rules(strict)", filter_rules, [rules_default], {"min_support": 0.02, "min_confidence": 0.5, "min_lift": 1.5})

# ============================================================
# 2. Co-purchase
# ============================================================
print("\n" + "-" * 60)
print("2. Co-purchase (src.analytics.copurchase)")
print("-" * 60)
try:
    from src.analytics.copurchase import compute_affinity_matrix
    print("  Imports OK")
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Co-purchase", "import", "ERROR", str(e), tb))

try_it("Co-purchase", "compute_affinity_matrix(default)", compute_affinity_matrix, [df])
try_it("Co-purchase", "compute_affinity_matrix(top50)", compute_affinity_matrix, [df], {"min_support": 0.01, "max_len": 2, "top_n_products": 50})

# Also test the cross-sell function from product_performance as the user requested
try:
    from src.analytics.product_performance import cross_sell_opportunity_matrix
    try_it("Co-purchase", "cross_sell_opportunity_matrix", cross_sell_opportunity_matrix, [df])
    try_it("Co-purchase", "cross_sell_opportunity_matrix(top20)", cross_sell_opportunity_matrix, [df], {"top_n": 20})
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Co-purchase", "cross_sell_opportunity_matrix import", "ERROR", str(e), tb))

# Also try the named function user asked for (even if it doesn't exist)
try:
    from src.analytics.copurchase import compute_cross_sell_opportunity_matrix
    try_it("Co-purchase", "compute_cross_sell_opportunity_matrix", compute_cross_sell_opportunity_matrix, [df])
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Co-purchase", "compute_cross_sell_opportunity_matrix", "ERROR", str(e), tb))

# ============================================================
# 3. Add-on
# ============================================================
print("\n" + "-" * 60)
print("3. Add-on (src.analytics.addon)")
print("-" * 60)
try:
    from src.analytics.addon import get_addon_recommendations, get_anchor_addon_matrix, get_addon_by_category
    print("  Imports OK")
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Add-on", "import", "ERROR", str(e), tb))

# Pick a common product as anchor
common_product = df["stockcode"].value_counts().index[0]
print(f"  Using anchor product: {common_product}")

try_it("Add-on", "get_addon_recommendations", get_addon_recommendations, [df, common_product])
try_it("Add-on", "get_addon_recommendations(alt)", get_addon_recommendations, [df, common_product], {"min_lift": 1.5, "top_n": 5})
try_it("Add-on", "get_anchor_addon_matrix", get_anchor_addon_matrix, [df])
try_it("Add-on", "get_anchor_addon_matrix(alt)", get_anchor_addon_matrix, [df], {"min_lift": 1.1, "top_n_per_anchor": 3})
try_it("Add-on", "get_addon_by_category", get_addon_by_category, [df, common_product])

# Try the name the user specified
try:
    from src.analytics.addon import identify_addon_opportunities
    try_it("Add-on", "identify_addon_opportunities", identify_addon_opportunities, [df])
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Add-on", "identify_addon_opportunities", "ERROR", str(e), tb))

# ============================================================
# 4. Switching
# ============================================================
print("\n" + "-" * 60)
print("4. Switching (src.analytics.switching)")
print("-" * 60)
try:
    from src.analytics.switching import compute_switching_matrix, detect_brand_switching, get_customer_loyalty_metrics
    print("  Imports OK")
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Switching", "import", "ERROR", str(e), tb))

try_it("Switching", "compute_switching_matrix(default)", compute_switching_matrix, [df])
try_it("Switching", "compute_switching_matrix(alt)", compute_switching_matrix, [df], {"window_days": 30, "min_transactions": 3})
try_it("Switching", "detect_brand_switching(default)", detect_brand_switching, [df])
try_it("Switching", "detect_brand_switching(alt)", detect_brand_switching, [df], {"window_days": 60})
try_it("Switching", "get_customer_loyalty_metrics", get_customer_loyalty_metrics, [df])

# ============================================================
# 5. Segmentation
# ============================================================
print("\n" + "-" * 60)
print("5. Segmentation (src.analytics.segmentation)")
print("-" * 60)
try:
    from src.analytics.segmentation import compute_rfm_features, rfm_segmentation, behavioral_segmentation, value_based_segmentation, get_segment_profiles
    print("  Imports OK")
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Segmentation", "import", "ERROR", str(e), tb))

rfm = try_it("Segmentation", "compute_rfm_features", compute_rfm_features, [df])
if rfm is not None:
    try_it("Segmentation", "rfm_segmentation(quantile)", rfm_segmentation, [rfm])
    try_it("Segmentation", "rfm_segmentation(kmeans)", rfm_segmentation, [rfm], {"method": "kmeans", "n_segments": 6})
    try_it("Segmentation", "rfm_segmentation(kmeans-4)", rfm_segmentation, [rfm], {"method": "kmeans", "n_segments": 4})

behav_seg = try_it("Segmentation", "behavioral_segmentation(default)", behavioral_segmentation, [df])
try_it("Segmentation", "behavioral_segmentation(alt)", behavioral_segmentation, [df], {"n_clusters": 4})
try_it("Segmentation", "value_based_segmentation", value_based_segmentation, [df])
try_it("Segmentation", "value_based_segmentation(alt)", value_based_segmentation, [df], {"prediction_horizon_days": 30})

if behav_seg is not None and rfm is not None:
    try_it("Segmentation", "get_segment_profiles", get_segment_profiles, [df, behav_seg])

# Try the named function the user specified
try:
    from src.analytics.segmentation import perform_behavioral_segmentation
    try_it("Segmentation", "perform_behavioral_segmentation", perform_behavioral_segmentation, [df])
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Segmentation", "perform_behavioral_segmentation", "ERROR", str(e), tb))

# ============================================================
# 6. Segmentation Enhanced
# ============================================================
print("\n" + "-" * 60)
print("6. Segmentation Enhanced (src.analytics.segmentation_enhanced)")
print("-" * 60)
try:
    from src.analytics.segmentation_enhanced import (
        compute_rfm_features as rfm_enh, rfm_segmentation as rfm_seg_enh,
        behavioral_segmentation as behav_seg_enh, predict_clv,
        survival_analysis, ensemble_segmentation, value_based_segmentation as val_seg_enh,
        get_segment_profiles as get_seg_prof_enh, get_available_models
    )
    print("  Imports OK")
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Segmentation Enhanced", "import", "ERROR", str(e), tb))

rfm_e = try_it("Segmentation Enhanced", "compute_rfm_features", rfm_enh, [df])
if rfm_e is not None:
    try_it("Segmentation Enhanced", "rfm_segmentation(quantile)", rfm_seg_enh, [rfm_e])
    try_it("Segmentation Enhanced", "rfm_segmentation(kmeans)", rfm_seg_enh, [rfm_e], {"method": "kmeans"})
    try_it("Segmentation Enhanced", "rfm_segmentation(gmm)", rfm_seg_enh, [rfm_e], {"method": "gmm"})

try_it("Segmentation Enhanced", "behavioral_segmentation(kmeans)", behav_seg_enh, [df])
try_it("Segmentation Enhanced", "behavioral_segmentation(agglomerative)", behav_seg_enh, [df], {"method": "agglomerative"})
try_it("Segmentation Enhanced", "behavioral_segmentation(gmm)", behav_seg_enh, [df], {"method": "gmm"})
try_it("Segmentation Enhanced", "behavioral_segmentation(dbscan)", behav_seg_enh, [df], {"method": "dbscan"})
try_it("Segmentation Enhanced", "predict_clv", predict_clv, [df])
try_it("Segmentation Enhanced", "predict_clv(rf)", predict_clv, [df], {"model_type": "random_forest"})
try_it("Segmentation Enhanced", "survival_analysis", survival_analysis, [df])
try_it("Segmentation Enhanced", "ensemble_segmentation", ensemble_segmentation, [df])
try_it("Segmentation Enhanced", "value_based_segmentation", val_seg_enh, [df])

if rfm_e is not None:
    quant_seg = try_it("Segmentation Enhanced", "rfm_segmentation(quantile-ref)", rfm_seg_enh, [rfm_e])
    if quant_seg is not None:
        try_it("Segmentation Enhanced", "get_segment_profiles", get_seg_prof_enh, [df, quant_seg])

try_it("Segmentation Enhanced", "get_available_models", get_available_models, [])

# Try the named function the user specified
try:
    from src.analytics.segmentation_enhanced import perform_enhanced_segmentation
    try_it("Segmentation Enhanced", "perform_enhanced_segmentation", perform_enhanced_segmentation, [df])
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Segmentation Enhanced", "perform_enhanced_segmentation", "ERROR", str(e), tb))

# ============================================================
# 7. Cohort
# ============================================================
print("\n" + "-" * 60)
print("7. Cohort (src.analytics.cohort)")
print("-" * 60)
try:
    from src.analytics.cohort import (
        compute_cohorts, compute_cohort_sizes, period_over_period_comparison,
        year_over_year_comparison, cohort_comparison_summary
    )
    print("  Imports OK")
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Cohort", "import", "ERROR", str(e), tb))

try_it("Cohort", "compute_cohorts(retention)", compute_cohorts, [df], {"metric": "retention"})
try_it("Cohort", "compute_cohorts(revenue)", compute_cohorts, [df], {"metric": "revenue"})
try_it("Cohort", "compute_cohorts(orders)", compute_cohorts, [df], {"metric": "orders"})
try_it("Cohort", "compute_cohorts(avg_order_value)", compute_cohorts, [df], {"metric": "avg_order_value"})
try_it("Cohort", "compute_cohort_sizes", compute_cohort_sizes, [df])
try_it("Cohort", "period_over_period_comparison", period_over_period_comparison, [df])
try_it("Cohort", "year_over_year_comparison", year_over_year_comparison, [df])
try_it("Cohort", "cohort_comparison_summary", cohort_comparison_summary, [df])

# Try the named functions the user specified
try:
    from src.analytics.cohort import compute_cohort_analysis
    try_it("Cohort", "compute_cohort_analysis", compute_cohort_analysis, [df])
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Cohort", "compute_cohort_analysis", "ERROR", str(e), tb))

try:
    from src.analytics.cohort import compute_retention_rates
    try_it("Cohort", "compute_retention_rates", compute_retention_rates, [df])
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Cohort", "compute_retention_rates", "ERROR", str(e), tb))

# ============================================================
# 8. Promotional
# ============================================================
print("\n" + "-" * 60)
print("8. Promotional (src.analytics.promotional)")
print("-" * 60)
try:
    from src.analytics.promotional import (
        detect_promotions, calculate_promotional_lift, calculate_incremental_revenue,
        promotion_roi_analysis, halo_effect_analysis, promotion_timing_analysis
    )
    print("  Imports OK")
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Promotional", "import", "ERROR", str(e), tb))

promos = try_it("Promotional", "detect_promotions(default)", detect_promotions, [df])
try_it("Promotional", "detect_promotions(alt)", detect_promotions, [df], {"price_change_threshold": 0.1, "min_duration_days": 2})

if promos is not None and not promos.empty:
    try_it("Promotional", "calculate_promotional_lift", calculate_promotional_lift, [df, promos])
    try_it("Promotional", "calculate_incremental_revenue", calculate_incremental_revenue, [df, promos])
    try_it("Promotional", "promotion_roi_analysis", promotion_roi_analysis, [df, promos])
    try_it("Promotional", "halo_effect_analysis", halo_effect_analysis, [df, promos])
    try_it("Promotional", "promotion_timing_analysis", promotion_timing_analysis, [df, promos])
else:
    # Try with auto-detect
    try_it("Promotional", "calculate_promotional_lift(auto)", calculate_promotional_lift, [df])
    try_it("Promotional", "promotion_timing_analysis(empty)", promotion_timing_analysis, [df, pd.DataFrame()])

# Try the named functions the user specified
try:
    from src.analytics.promotional import compute_promotional_lift
    try_it("Promotional", "compute_promotional_lift", compute_promotional_lift, [df])
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Promotional", "compute_promotional_lift", "ERROR", str(e), tb))

try:
    from src.analytics.promotional import analyze_price_elasticity
    try_it("Promotional", "analyze_price_elasticity", analyze_price_elasticity, [df])
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Promotional", "analyze_price_elasticity", "ERROR", str(e), tb))

try:
    from src.analytics.promotional import compute_halo_effects
    try_it("Promotional", "compute_halo_effects", compute_halo_effects, [df])
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Promotional", "compute_halo_effects", "ERROR", str(e), tb))

# ============================================================
# 9. Product Performance
# ============================================================
print("\n" + "-" * 60)
print("9. Product Performance (src.analytics.product_performance)")
print("-" * 60)
try:
    from src.analytics.product_performance import (
        compute_product_metrics, abc_analysis, xyz_analysis,
        product_lifecycle_stage, product_seasonality, product_affinity_score,
        cross_sell_opportunity_matrix, price_elasticity_analysis
    )
    print("  Imports OK")
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Product Performance", "import", "ERROR", str(e), tb))

prod_metrics = try_it("Product Performance", "compute_product_metrics", compute_product_metrics, [df])
if prod_metrics is not None:
    try_it("Product Performance", "abc_analysis(3)", abc_analysis, [prod_metrics])
    try_it("Product Performance", "abc_analysis(4)", abc_analysis, [prod_metrics], {"n_classes": 4})

try_it("Product Performance", "xyz_analysis", xyz_analysis, [df])
try_it("Product Performance", "xyz_analysis(quarterly)", xyz_analysis, [df], {"period": "Q"})

if prod_metrics is not None:
    try_it("Product Performance", "product_lifecycle_stage", product_lifecycle_stage, [prod_metrics, df])
    try_it("Product Performance", "product_lifecycle_stage(Q)", product_lifecycle_stage, [prod_metrics, df], {"period": "Q"})

common_prod = df["stockcode"].value_counts().index[0]
try_it("Product Performance", "product_seasonality", product_seasonality, [df, common_prod])
try_it("Product Performance", "product_affinity_score", product_affinity_score, [df, common_prod])
try_it("Product Performance", "cross_sell_opportunity_matrix", cross_sell_opportunity_matrix, [df])
try_it("Product Performance", "cross_sell_opportunity_matrix(20)", cross_sell_opportunity_matrix, [df], {"top_n": 20})
try_it("Product Performance", "price_elasticity_analysis", price_elasticity_analysis, [df, common_prod])

# Try the named functions the user specified
try:
    from src.analytics.product_performance import compute_product_performance
    try_it("Product Performance", "compute_product_performance", compute_product_performance, [df])
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Product Performance", "compute_product_performance", "ERROR", str(e), tb))

try:
    from src.analytics.product_performance import compute_product_lifecycle
    try_it("Product Performance", "compute_product_lifecycle", compute_product_lifecycle, [df])
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Product Performance", "compute_product_lifecycle", "ERROR", str(e), tb))

try:
    from src.analytics.product_performance import compute_xyz_variability
    try_it("Product Performance", "compute_xyz_variability", compute_xyz_variability, [df])
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Product Performance", "compute_xyz_variability", "ERROR", str(e), tb))

try:
    from src.analytics.product_performance import compute_market_basket_metrics
    try_it("Product Performance", "compute_market_basket_metrics", compute_market_basket_metrics, [df])
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("Product Performance", "compute_market_basket_metrics", "ERROR", str(e), tb))

# ============================================================
# 10. CDT Similarity
# ============================================================
print("\n" + "-" * 60)
print("10. CDT Similarity (src.analytics.cdt_similarity)")
print("-" * 60)
try:
    from src.analytics.cdt_similarity import (
        build_similarity_matrix, build_copurchase_tables,
        compute_phi_coefficient, compute_jaccard,
        build_customer_sequences, detect_switches,
        compute_switching_matrix_from_sequences
    )
    print("  Imports OK")
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("CDT Similarity", "import", "ERROR", str(e), tb))

try_it("CDT Similarity", "build_similarity_matrix(phi)", build_similarity_matrix, [df], {"method": "phi"})
try_it("CDT Similarity", "build_similarity_matrix(jaccard)", build_similarity_matrix, [df], {"method": "jaccard"})
try_it("CDT Similarity", "build_similarity_matrix(phi,strict)", build_similarity_matrix, [df], {"min_cooccurrence": 10, "min_product_support": 5})
try_it("CDT Similarity", "build_copurchase_tables", build_copurchase_tables, [df])
try_it("CDT Similarity", "build_customer_sequences", build_customer_sequences, [df])
try_it("CDT Similarity", "detect_switches", detect_switches, [{"CUST0001": ["CL001", "W001", "M001", "CL001"]}])

seqs = build_customer_sequences(df)
if seqs:
    try_it("CDT Similarity", "detect_switches(from_df)", detect_switches, [seqs], {"transactions_df": df})
    try_it("CDT Similarity", "compute_switching_matrix_from_sequences", compute_switching_matrix_from_sequences, [seqs])

# Test with a sample table
sample_table = {"both": 50, "a_only": 30, "b_only": 20, "neither": 100}
try_it("CDT Similarity", "compute_phi_coefficient", compute_phi_coefficient, [sample_table])
try_it("CDT Similarity", "compute_jaccard", compute_jaccard, [sample_table])

# Try the named functions the user specified
try:
    from src.analytics.cdt_similarity import compute_cosine_similarity
    try_it("CDT Similarity", "compute_cosine_similarity", compute_cosine_similarity, [df])
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("CDT Similarity", "compute_cosine_similarity", "ERROR", str(e), tb))

try:
    from src.analytics.cdt_similarity import compute_jaccard_similarity
    try_it("CDT Similarity", "compute_jaccard_similarity", compute_jaccard_similarity, [df])
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("CDT Similarity", "compute_jaccard_similarity", "ERROR", str(e), tb))

try:
    from src.analytics.cdt_similarity import compute_yules_q
    try_it("CDT Similarity", "compute_yules_q", compute_yules_q, [df])
except Exception as e:
    tb = "".join(tb_module.format_exception(*sys.exc_info()))
    results.append(("CDT Similarity", "compute_yules_q", "ERROR", str(e), tb))

# ============================================================
# Print final results
# ============================================================
print_results()
