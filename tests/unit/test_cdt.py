"""Unit tests for the CDT package."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.cdt import (
    build_product_graph,
    build_similarity_matrix,
    build_transaction_derived_attributes,
    bootstrap_similarity_ci,
    compute_cophenetic_correlation,
    compute_cluster_quality,
    compute_entropy_gain,
    compute_mutual_information,
    compute_within_group_similarity,
    detect_communities,
    detect_communities_label_propagation,
    detect_communities_louvain,
    find_best_attribute_split,
    find_optimal_clusters_sklearn,
    generate_synthetic_cluster_data,
    get_cluster_assignments,
    get_dendrogram_data,
    perform_hierarchical_clustering,
    run_cdt_validation,
    score_tree,
    similarity_to_distance,
    build_cdt,
    tree_to_dataframe,
    TreeNode,
)
from src.analytics.schemas import (
    CDT_ASSIGNMENTS,
    CDT_ATTRIBUTES,
    CDT_COMMUNITY,
    CDT_OPTIMAL_K,
    CDT_QUALITY,
    CDT_TREE_NODES,
    CDT_TREE_PRODUCTS,
    CDT_TREE_SCORE,
    CDT_VALIDATION,
)


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    from src.analytics.data import load_transactions

    df, _, _, _ = load_transactions("sample_data/sample_transactions.csv")
    return df


def test_derived_attributes_contract(sample_df: pd.DataFrame) -> None:
    table = build_transaction_derived_attributes(sample_df)
    CDT_ATTRIBUTES.validate(table)
    assert set(table.columns) == {
        "stockcode",
        "price_tier",
        "velocity_tier",
        "seasonality_class",
        "basket_size_affinity",
        "substitution_tier",
    }
    assert table["seasonality_class"].isin({"Seasonal", "Steady", "Sporadic"}).all()
    assert table["price_tier"].notna().all()


def test_similarity_dispatch(sample_df: pd.DataFrame) -> None:
    for method in ("phi", "jaccard", "pmi", "cosine_tfidf"):
        sim = build_similarity_matrix(sample_df, method=method, min_product_support=2)
        assert sim.shape[0] == sim.shape[1]
        assert (sim.index == sim.columns).all()
        assert np.allclose(np.diag(sim.to_numpy()), 1.0)
        assert np.isfinite(sim.to_numpy()).all()


def test_similarity_ensemble(sample_df: pd.DataFrame) -> None:
    sim = build_similarity_matrix(sample_df, method="ensemble", min_product_support=2)
    assert sim.shape[0] == sim.shape[1]
    assert np.allclose(np.diag(sim.to_numpy()), 1.0)


def test_bootstrap_similarity_ci(sample_df: pd.DataFrame) -> None:
    products = sample_df["stockcode"].unique()[:2]
    ci = bootstrap_similarity_ci(
        sample_df, products[0], products[1], method="phi", n_resamples=10, random_seed=42
    )
    assert {"estimate", "lower", "upper", "std_error", "n_resamples"} <= set(ci.keys())
    assert ci["n_resamples"] >= 1


def test_distance_and_clustering(sample_df: pd.DataFrame) -> None:
    sim = build_similarity_matrix(sample_df, method="phi", min_product_support=2)
    dist = similarity_to_distance(sim, method="phi")
    assert dist.shape == sim.shape
    assert np.allclose(np.diag(dist), 0.0)
    assert (dist >= 0).all()

    linkage, clusters = perform_hierarchical_clustering(sim, n_clusters=3)
    CDT_ASSIGNMENTS.validate(clusters.reset_index().rename(columns={"index": "stockcode", "cluster": "cluster"}))
    assert len(clusters) == len(sim)


def test_optimal_k(sample_df: pd.DataFrame) -> None:
    sim = build_similarity_matrix(sample_df, method="phi", min_product_support=2)
    table = find_optimal_clusters_sklearn(sim, max_clusters=5)
    CDT_OPTIMAL_K.validate(table, allow_empty=True)
    if not table.empty:
        assert table["n_clusters"].between(2, 5).all()


def test_cluster_assignments(sample_df: pd.DataFrame) -> None:
    table = get_cluster_assignments(sample_df, n_clusters=4, method="phi")
    CDT_ASSIGNMENTS.validate(table)
    assert table["cluster"].between(1, 4).all()


def test_cluster_quality(sample_df: pd.DataFrame) -> None:
    sim = build_similarity_matrix(sample_df, method="phi", min_product_support=2)
    assignments = get_cluster_assignments(sample_df, similarity_matrix=sim, n_clusters=3)
    quality = compute_cluster_quality(sim, assignments)
    CDT_QUALITY.validate(quality)
    assert quality["size"].sum() == len(sim)


def test_cophenetic(sample_df: pd.DataFrame) -> None:
    sim = build_similarity_matrix(sample_df, method="phi", min_product_support=2)
    linkage, _ = perform_hierarchical_clustering(sim, n_clusters=3)
    corr = compute_cophenetic_correlation(sim, linkage)
    assert 0.0 <= corr <= 1.0


def test_dendrogram_data(sample_df: pd.DataFrame) -> None:
    sim = build_similarity_matrix(sample_df, method="phi", min_product_support=2)
    linkage, clusters = perform_hierarchical_clustering(sim, n_clusters=3)
    data = get_dendrogram_data(linkage, sim.index.tolist())
    assert set(data.keys()) == {"linkage", "labels", "n_leaves"}
    assert len(data["labels"]) == len(sim)


def test_community_detection(sample_df: pd.DataFrame) -> None:
    mapping, modularity = detect_communities(sample_df, method="louvain", random_seed=42)
    assert isinstance(mapping, dict)
    assert len(mapping) >= 1
    assert 0.0 <= modularity <= 1.0

    mapping2, mod2 = detect_communities(sample_df, method="label_propagation")
    assert isinstance(mapping2, dict)
    assert len(mapping2) >= 1


def test_community_contracts(sample_df: pd.DataFrame) -> None:
    mapping, _ = detect_communities(sample_df, method="louvain", random_seed=42)
    table = pd.DataFrame([{"stockcode": k, "community": v} for k, v in mapping.items()])
    CDT_COMMUNITY.validate(table)


def test_mutual_information() -> None:
    true = {"A": 1, "B": 1, "C": 2, "D": 2}
    attr = {"A": "x", "B": "x", "C": "y", "D": "y"}
    mi = compute_mutual_information(true, attr)
    assert mi > 0.0


def test_entropy_gain() -> None:
    true = {"A": 1, "B": 1, "C": 2, "D": 2}
    attr = {"A": "x", "B": "x", "C": "y", "D": "y"}
    products = ["A", "B", "C", "D"]
    gain = compute_entropy_gain(true, products, attr)
    assert gain >= 0.0


def test_within_group_similarity(sample_df: pd.DataFrame) -> None:
    sim = build_similarity_matrix(sample_df, method="phi", min_product_support=2)
    products = sim.index.tolist()[:3]
    score = compute_within_group_similarity(products, sim)
    assert 0.0 <= score <= 1.0


def test_find_best_split(sample_df: pd.DataFrame) -> None:
    attrs = build_transaction_derived_attributes(sample_df)
    sim = build_similarity_matrix(sample_df, method="phi", min_product_support=2)
    products = attrs["stockcode"].tolist()
    attr, groups, score = find_best_attribute_split(
        products,
        attrs,
        sim,
        min_cluster_size=3,
        criterion="entropy",
        cluster_assignments=None,  # clusterless → similarity-based
    )
    # may be None if no split found; just verify returns proper types
    assert attr is None or isinstance(attr, str)
    assert isinstance(groups, dict)
    assert isinstance(score, float)


def test_tree_build_and_score(sample_df: pd.DataFrame) -> None:
    attrs = build_transaction_derived_attributes(sample_df)
    sim = build_similarity_matrix(sample_df, method="phi", min_product_support=2)
    root = build_cdt(attrs, sim, max_depth=2, min_cluster_size=3)
    assert isinstance(root, TreeNode)
    nodes, products = tree_to_dataframe(root)
    CDT_TREE_NODES.validate(nodes)
    CDT_TREE_PRODUCTS.validate(products, allow_empty=True)
    assert nodes["node_id"].nunique() == len(nodes)

    scores = score_tree(root, sim)
    CDT_TREE_SCORE.validate(scores)
    assert scores["metric"].isin({"n_nodes", "n_leaves", "depth", "mean_leaf_similarity", "products_covered"}).all()


def test_synthetic_validation() -> None:
    df, true = generate_synthetic_cluster_data(n_products=20, n_true_clusters=3, random_seed=1)
    table = run_cdt_validation(df, true, n_clusters=3)
    CDT_VALIDATION.validate(table)
    ari_row = table[table["metric"] == "adjusted_rand_index"]
    assert len(ari_row) == 1
    assert 0.0 <= float(ari_row["value"].iloc[0]) <= 1.0


def test_invalid_methods_raise(sample_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        build_similarity_matrix(sample_df, method="unknown")
    with pytest.raises(ValueError):
        detect_communities(sample_df, method="unknown")