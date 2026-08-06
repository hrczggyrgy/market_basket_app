"""Customer Decision Tree (CDT) package.

Public API
----------
- attributes: derive_price_tier, derive_velocity_tier, derive_basket_size_affinity,
  derive_seasonality_class, derive_substitution_tier, build_transaction_derived_attributes,
  get_candidate_attributes
- similarity: compute_phi_matrix, compute_jaccard_matrix, compute_pmi_matrix,
  compute_cosine_tfidf_matrix, build_similarity_matrix, build_similarity_matrix_ensemble,
  bootstrap_similarity_ci
- clustering: similarity_to_distance, perform_hierarchical_clustering,
  find_optimal_clusters_sklearn, get_cluster_assignments, compute_cluster_quality,
  compute_cophenetic_correlation, get_dendrogram_data
- community: build_product_graph, detect_communities_louvain,
  detect_communities_label_propagation, detect_communities, compute_community_modularity
- tree: TreeNode, compute_mutual_information, compute_entropy_gain, compute_gini_gain,
  compute_within_group_similarity, compute_attribute_split_quality,
  find_best_attribute_split, build_cdt, count_nodes, count_leaves, max_depth,
  score_tree, tree_to_dataframe, tree_to_json, prune_tree
- validation: generate_synthetic_cluster_data, run_cdt_validation
"""

from src.analytics.cdt.attributes import (
    build_transaction_derived_attributes,
    derive_basket_size_affinity,
    derive_price_tier,
    derive_seasonality_class,
    derive_substitution_tier,
    derive_velocity_tier,
    get_candidate_attributes,
)
from src.analytics.cdt.clustering import (
    compute_cluster_quality,
    compute_cophenetic_correlation,
    find_optimal_clusters_sklearn,
    get_cluster_assignments,
    get_dendrogram_data,
    perform_hierarchical_clustering,
    similarity_to_distance,
)
from src.analytics.cdt.community import (
    build_product_graph,
    compute_community_modularity,
    detect_communities,
    detect_communities_label_propagation,
    detect_communities_louvain,
)
from src.analytics.cdt.similarity import (
    bootstrap_similarity_ci,
    build_similarity_matrix,
    build_similarity_matrix_ensemble,
    compute_cosine_tfidf_matrix,
    compute_jaccard_matrix,
    compute_pmi_matrix,
    compute_phi_matrix,
)
from src.analytics.cdt.tree import (
    TreeNode,
    build_cdt,
    compute_attribute_split_quality,
    compute_entropy_gain,
    compute_gini_gain,
    compute_mutual_information,
    compute_within_group_similarity,
    count_leaves,
    count_nodes,
    find_best_attribute_split,
    max_depth,
    prune_tree,
    score_tree,
    tree_to_dataframe,
    tree_to_json,
)
from src.analytics.cdt.validation import (
    generate_synthetic_cluster_data,
    run_cdt_validation,
)

__all__ = [
    # attributes
    "build_transaction_derived_attributes",
    "derive_basket_size_affinity",
    "derive_price_tier",
    "derive_seasonality_class",
    "derive_substitution_tier",
    "derive_velocity_tier",
    "get_candidate_attributes",
    # similarity
    "bootstrap_similarity_ci",
    "build_similarity_matrix",
    "build_similarity_matrix_ensemble",
    "compute_cosine_tfidf_matrix",
    "compute_jaccard_matrix",
    "compute_pmi_matrix",
    "compute_phi_matrix",
    # clustering
    "compute_cluster_quality",
    "compute_cophenetic_correlation",
    "find_optimal_clusters_sklearn",
    "get_cluster_assignments",
    "get_dendrogram_data",
    "perform_hierarchical_clustering",
    "similarity_to_distance",
    # community
    "build_product_graph",
    "compute_community_modularity",
    "detect_communities",
    "detect_communities_label_propagation",
    "detect_communities_louvain",
    # tree
    "TreeNode",
    "build_cdt",
    "compute_attribute_split_quality",
    "compute_entropy_gain",
    "compute_gini_gain",
    "compute_mutual_information",
    "compute_within_group_similarity",
    "count_leaves",
    "count_nodes",
    "find_best_attribute_split",
    "max_depth",
    "prune_tree",
    "score_tree",
    "tree_to_dataframe",
    "tree_to_json",
    # validation
    "generate_synthetic_cluster_data",
    "run_cdt_validation",
]