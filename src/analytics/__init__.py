"""Analytics module initialization."""

from .addon import (
    get_addon_by_category,
    get_addon_recommendations,
    get_anchor_addon_matrix,
)
from .assortment_opt import (
    evaluate_assortment,
    generate_assortment_scenarios,
    optimize_assortment_heuristic,
    optimize_assortment_milp,
)
from .basket_metrics import (
    compute_basket_composition,
    compute_basket_penetration,
    compute_basket_value_uplift,
    compute_copurchase_index,
    compute_shopper_loyalty_metrics,
)
from .bootstrap import (
    bootstrap_ci,
    bootstrap_ci_customer,
    bootstrap_transaction_samples,
)
from .cdt_attributes import (
    FUNCTIONAL_FIT_ATTRIBUTES,
    build_transaction_derived_attributes,
    derive_basket_size_affinity,
    derive_price_tier,
    derive_seasonality_class,
    derive_substitution_tier,
    derive_velocity_tier,
    extract_attributes_from_product_text,
    get_candidate_attributes,
    merge_extracted_attributes,
)
from .cdt_behavioral import (
    build_behavioral_matrices,
    compute_brand_switching_matrix,
    compute_bundling_matrix,
    detect_brand_switching,
    get_substitution_matrix,
    get_top_bundling_pairs,
    get_top_substitution_pairs,
    get_top_switching_paths,
)
from .cdt_clustering import (
    compute_cluster_quality,
    compute_unconstrained_baseline,
    find_optimal_clusters,
    get_cluster_assignments,
    get_dendrogram_data,
    perform_hierarchical_clustering,
)
from .cdt_community import (
    build_product_graph,
    community_detection_pipeline,
    detect_communities,
    detect_communities_label_propagation,
    detect_communities_leiden,
    detect_communities_louvain,
    hierarchical_clustering_within_communities,
    merge_community_dendrograms,
)
from .cdt_similarity import (
    build_copurchase_tables,
    build_customer_sequences,
    build_similarity_matrix,
    build_similarity_matrix_ensemble,
    compute_cosine_tfidf_matrix,
    compute_jaccard,
    compute_phi_coefficient,
    compute_pmi_matrix,
    compute_switching_matrix_from_sequences,
    detect_switches,
)
from .cdt_tree_builder import (
    TreeNode,
    build_cdt,
    extract_product_attributes,
    score_tree,
    tree_to_dataframe,
    tree_to_json,
)
from .cdt_validation import (
    generate_synthetic_cluster_data,
    run_cdt_validation,
)
from .cohort import (
    cohort_comparison_summary,
    compute_cohort_sizes,
    compute_cohorts,
    period_over_period_comparison,
    year_over_year_comparison,
)
from .copurchase import (
    compute_affinity_matrix,
    get_product_affinity_profile,
    get_top_affinity_pairs,
)
from .demand_transference import (
    compute_demand_transference_matrix,
    compute_substitutable_demand_percentage,
    delist_impact_analysis,
    node_delist_impact,
)
from .pricing import (
    compute_kvi_score,
    diagnose_price_curves,
    estimate_bayesian_hierarchical_elasticity,
    estimate_elasticity_xgb,
    estimate_hierarchical_elasticity,
    estimate_loglog_elasticity,
)
from .product_performance import (
    compute_product_metrics,
    cross_sell_opportunity_matrix,
    price_elasticity_analysis,
    product_affinity_score,
    product_lifecycle_stage,
    product_seasonality,
)
from .promo_uplift import (
    build_uplift_dataset,
    decompose_promo_lift,
    derive_promo_flag,
    estimate_propensity_score,
    evaluate_uplift_model,
    promo_roi_analysis,
    train_s_learner_uplift,
    train_t_learner_uplift,
    train_xgb_uplift,
)
from .promotional import (
    calculate_incremental_revenue,
    calculate_promotional_lift,
    detect_promotions,
    halo_effect_analysis,
    promotion_timing_analysis,
)
from .promotional_validation import (
    generate_synthetic_promo_data,
    run_promo_detection_validation,
)
from .segmentation import (
    behavioral_segmentation,
    compute_rfm_features,
    get_segment_profiles,
    rfm_segmentation,
    value_based_segmentation,
)
from .segmentation_validation import (
    generate_synthetic_customer_segments,
    run_segmentation_validation,
)
from .sufficiency import (
    assess_data_sufficiency,
    format_sufficiency_summary,
    sufficiency_badge,
)
from .switching import (
    compute_switching_matrix,
    get_customer_loyalty_metrics,
    get_switching_heatmap_data,
)
from .validation import generate_synthetic_elasticity_data, run_validation

__all__ = [
    "compute_affinity_matrix",
    "get_top_affinity_pairs",
    "get_product_affinity_profile",
    "get_addon_recommendations",
    "get_anchor_addon_matrix",
    "get_addon_by_category",
    "compute_switching_matrix",
    "get_customer_loyalty_metrics",
    "detect_brand_switching",
    "get_top_switching_paths",
    "get_switching_heatmap_data",
    "bootstrap_ci",
    "bootstrap_ci_customer",
    "bootstrap_transaction_samples",
    "assess_data_sufficiency",
    "format_sufficiency_summary",
    "sufficiency_badge",
    "compute_rfm_features",
    "rfm_segmentation",
    "behavioral_segmentation",
    "value_based_segmentation",
    "get_segment_profiles",
    "compute_product_metrics",
    "product_lifecycle_stage",
    "product_seasonality",
    "product_affinity_score",
    "cross_sell_opportunity_matrix",
    "price_elasticity_analysis",
    "compute_cohorts",
    "compute_cohort_sizes",
    "period_over_period_comparison",
    "year_over_year_comparison",
    "cohort_comparison_summary",
    "detect_promotions",
    "calculate_promotional_lift",
    "calculate_incremental_revenue",
    "promo_roi_analysis",
    "halo_effect_analysis",
    "promotion_timing_analysis",
    # Basket metrics
    "compute_basket_penetration",
    "compute_basket_value_uplift",
    "compute_basket_composition",
    "compute_copurchase_index",
    "compute_shopper_loyalty_metrics",
    # CDT modules
    "build_customer_sequences",
    "detect_switches",
    "build_copurchase_tables",
    "compute_phi_coefficient",
    "compute_jaccard",
    "build_similarity_matrix",
    "build_similarity_matrix_ensemble",
    "compute_pmi_matrix",
    "compute_cosine_tfidf_matrix",
    "compute_switching_matrix_from_sequences",
    "perform_hierarchical_clustering",
    "find_optimal_clusters",
    "get_cluster_assignments",
    "compute_cluster_quality",
    "get_dendrogram_data",
    "compute_unconstrained_baseline",
    "TreeNode",
    "build_cdt",
    "score_tree",
    "tree_to_dataframe",
    "tree_to_json",
    "extract_product_attributes",
    "FUNCTIONAL_FIT_ATTRIBUTES",
    "build_transaction_derived_attributes",
    "derive_basket_size_affinity",
    "derive_price_tier",
    "derive_seasonality_class",
    "derive_substitution_tier",
    "derive_velocity_tier",
    "extract_attributes_from_product_text",
    "get_candidate_attributes",
    "merge_extracted_attributes",
    "compute_switching_matrix",
    "get_substitution_matrix",
    "compute_bundling_matrix",
    "build_behavioral_matrices",
    "get_top_substitution_pairs",
    "get_top_bundling_pairs",
    "get_top_switching_paths",
    "detect_brand_switching",
    "compute_brand_switching_matrix",
    # Community detection
    "build_product_graph",
    "detect_communities",
    "detect_communities_louvain",
    "detect_communities_leiden",
    "detect_communities_label_propagation",
    "hierarchical_clustering_within_communities",
    "merge_community_dendrograms",
    "community_detection_pipeline",
    # Demand transference
    "compute_demand_transference_matrix",
    "compute_substitutable_demand_percentage",
    "delist_impact_analysis",
    "node_delist_impact",
    # Promo uplift
    "derive_promo_flag",
    "build_uplift_dataset",
    "train_t_learner_uplift",
    "train_s_learner_uplift",
    "train_xgb_uplift",
    "estimate_propensity_score",
    "evaluate_uplift_model",
    "decompose_promo_lift",
    "promo_roi_analysis",
    # Pricing
    "estimate_loglog_elasticity",
    "estimate_hierarchical_elasticity",
    "estimate_bayesian_hierarchical_elasticity",
    "estimate_elasticity_xgb",
    "compute_kvi_score",
    "diagnose_price_curves",
    # Assortment
    "evaluate_assortment",
    "generate_assortment_scenarios",
    "optimize_assortment_heuristic",
    "optimize_assortment_milp",
    # Validation
    "generate_synthetic_elasticity_data",
    "run_validation",
    "generate_synthetic_cluster_data",
    "run_cdt_validation",
    "generate_synthetic_promo_data",
    "run_promo_detection_validation",
    "generate_synthetic_customer_segments",
    "run_segmentation_validation",
]
