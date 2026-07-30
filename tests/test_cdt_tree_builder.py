"""Tests for Customer Decision Tree builder."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.cdt_tree_builder import (
    TreeNode,
    build_cdt,
    compute_attribute_split_quality,
    compute_entropy_gain,
    compute_gini_gain,
    compute_mutual_information,
    compute_split_score,
    count_leaves,
    count_nodes,
    extract_product_attributes,
    find_best_attribute_split,
    score_tree,
    tree_to_dataframe,
    tree_to_json,
)


def _make_similarity_matrix(products, seed=42):
    rng = np.random.default_rng(seed)
    n = len(products)
    sim = rng.uniform(0.3, 0.9, size=(n, n))
    sim = (sim + sim.T) / 2
    np.fill_diagonal(sim, 1.0)
    return pd.DataFrame(sim, index=products, columns=products)


class TestMutualInformation:
    def test_perfect_split(self):
        products = ["A", "B", "C", "D"]
        cluster_assignments = {"A": 0, "B": 0, "C": 1, "D": 1}
        attr_values = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
        mi = compute_mutual_information(cluster_assignments, attr_values)
        assert mi > 0

    def test_no_split(self):
        cluster_assignments = {"A": 0, "B": 0, "C": 0, "D": 0}
        attr_values = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
        mi = compute_mutual_information(cluster_assignments, attr_values)
        assert mi == 0.0

    def test_single_product(self):
        mi = compute_mutual_information({"A": 0}, {"A": "X"})
        assert mi == 0.0


class TestGiniGain:
    def test_perfect_split(self):
        cluster_assignments = {"A": 0, "B": 0, "C": 1, "D": 1}
        attr_values = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
        gain = compute_gini_gain(cluster_assignments, attr_values)
        assert gain > 0

    def test_no_split(self):
        cluster_assignments = {"A": 0, "B": 0, "C": 0, "D": 0}
        attr_values = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
        gain = compute_gini_gain(cluster_assignments, attr_values)
        assert gain == 0.0


class TestEntropyGain:
    def test_perfect_split(self):
        cluster_assignments = {"A": 0, "B": 0, "C": 1, "D": 1}
        attr_values = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
        gain = compute_entropy_gain(cluster_assignments, attr_values)
        assert gain > 0

    def test_no_split(self):
        cluster_assignments = {"A": 0, "B": 0, "C": 0, "D": 0}
        attr_values = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
        gain = compute_entropy_gain(cluster_assignments, attr_values)
        assert gain == 0.0


class TestComputeSplitScore:
    def test_mutual_info_criterion(self):
        cluster_assignments = {"A": 0, "B": 0, "C": 1, "D": 1}
        attr_values = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
        score = compute_split_score(
            list(cluster_assignments.keys()),
            cluster_assignments,
            attr_values,
            criterion="mutual_info",
        )
        assert score > 0

    def test_gini_criterion(self):
        cluster_assignments = {"A": 0, "B": 0, "C": 1, "D": 1}
        attr_values = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
        score = compute_split_score(
            list(cluster_assignments.keys()),
            cluster_assignments,
            attr_values,
            criterion="gini",
        )
        assert score > 0

    def test_entropy_criterion(self):
        cluster_assignments = {"A": 0, "B": 0, "C": 1, "D": 1}
        attr_values = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
        score = compute_split_score(
            list(cluster_assignments.keys()),
            cluster_assignments,
            attr_values,
            criterion="entropy",
        )
        assert score > 0

    def test_mixed_criterion(self):
        cluster_assignments = {"A": 0, "B": 0, "C": 1, "D": 1}
        attr_values = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
        score = compute_split_score(
            list(cluster_assignments.keys()),
            cluster_assignments,
            attr_values,
            criterion="mixed",
            alpha=0.5,
        )
        assert score > 0

    def test_unknown_criterion_raises(self):
        cluster_assignments = {"A": 0, "B": 0, "C": 1, "D": 1}
        attr_values = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
        with pytest.raises(ValueError, match="Unknown split criterion"):
            compute_split_score(
                list(cluster_assignments.keys()),
                cluster_assignments,
                attr_values,
                criterion="invalid",
            )

    def test_mixed_criterion_alpha_zero(self):
        """Mixed with alpha=0 should equal pure gini gain."""
        cluster_assignments = {"A": 0, "B": 0, "C": 1, "D": 1}
        attr_values = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
        mixed_score = compute_split_score(
            list(cluster_assignments.keys()),
            cluster_assignments,
            attr_values,
            criterion="mixed",
            alpha=0.0,
        )
        gini_score = compute_split_score(
            list(cluster_assignments.keys()),
            cluster_assignments,
            attr_values,
            criterion="gini",
        )
        assert mixed_score == pytest.approx(gini_score)


class TestComputeAttributeSplitQuality:
    def test_basic_split(self):
        products = ["A", "B", "C", "D"]
        cluster_assignments = {"A": 0, "B": 0, "C": 1, "D": 1}
        attr_values = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
        sim_matrix = _make_similarity_matrix(products)
        score, groups = compute_attribute_split_quality(
            products, cluster_assignments, attr_values, sim_matrix, min_cluster_size=2
        )
        assert score > 0
        assert set(groups.keys()) == {"X", "Y"}

    def test_below_min_cluster_size(self):
        products = ["A", "B", "C", "D"]
        cluster_assignments = {"A": 0, "B": 0, "C": 1, "D": 1}
        attr_values = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
        sim_matrix = _make_similarity_matrix(products)
        score, groups = compute_attribute_split_quality(
            products, cluster_assignments, attr_values, sim_matrix, min_cluster_size=5
        )
        assert score == 0.0
        assert groups == {}


class TestFindBestAttributeSplit:
    def test_best_attribute(self):
        products = ["A", "B", "C", "D"]
        cluster_assignments = {"A": 0, "B": 0, "C": 1, "D": 1}
        attrs_df = pd.DataFrame(
            {
                "category": {"A": "X", "B": "X", "C": "Y", "D": "Y"},
                "brand": {"A": "P", "B": "Q", "C": "P", "D": "Q"},
            }
        ).reindex(products)
        sim_matrix = _make_similarity_matrix(products)
        best_attr, groups, score = find_best_attribute_split(
            products, cluster_assignments, attrs_df, sim_matrix, min_cluster_size=2
        )
        assert best_attr is not None
        assert len(groups) >= 2
        assert score > 0

    def test_no_candidate_attributes(self):
        products = ["A", "B", "C", "D"]
        cluster_assignments = {"A": 0, "B": 0, "C": 1, "D": 1}
        attrs_df = pd.DataFrame(index=products)
        sim_matrix = _make_similarity_matrix(products)
        best_attr, groups, score = find_best_attribute_split(
            products, cluster_assignments, attrs_df, sim_matrix, candidate_attributes=[]
        )
        assert best_attr is None
        assert groups == {}
        assert score == 0.0


class TestTreeNode:
    def test_create_leaf(self):
        node = TreeNode(
            node_id="node_1",
            name="Test Leaf",
            products=["A", "B"],
            similarity_within=0.8,
            size=2,
            is_leaf=True,
        )
        assert node.node_id == "node_1"
        assert node.is_leaf

    def test_to_dict(self):
        node = TreeNode(
            node_id="node_1",
            name="Test Internal",
            products=["A", "B"],
            similarity_within=0.8,
            size=2,
            is_leaf=False,
        )
        child = TreeNode(
            node_id="node_2",
            name="Test Leaf",
            products=["A"],
            similarity_within=1.0,
            size=1,
            is_leaf=True,
        )
        node.children.append(child)
        child.parent = node
        d = node.to_dict()
        assert d["node_id"] == "node_1"
        assert len(d["children"]) == 1


class TestBuildCdtIntegration:
    @pytest.fixture
    def data(self):
        products = [f"P{i}" for i in range(10)]
        cluster_assignments = {p: i % 3 for i, p in enumerate(products)}
        sim_matrix = _make_similarity_matrix(products)
        attrs_df = pd.DataFrame(
            {
                "category": {p: f"Cat{v % 2}" for v, p in enumerate(products)},
                "brand": {p: f"Brand{v % 3}" for v, p in enumerate(products)},
            }
        ).reindex(products)
        return products, cluster_assignments, sim_matrix, attrs_df

    def test_build_cdt_returns_tree_and_metadata(self, data):
        products, cluster_assignments, sim_matrix, attrs_df = data
        root, metadata = build_cdt(
            sim_matrix,
            cluster_assignments,
            attrs_df,
            min_cluster_size=2,
            quality_threshold=0.3,
        )
        assert isinstance(root, TreeNode)
        assert isinstance(metadata, dict)
        assert "n_nodes" in metadata
        assert "n_leaves" in metadata
        assert "max_depth" in metadata
        assert "tree_quality" in metadata
        assert "unconstrained_baseline" in metadata
        assert metadata["n_nodes"] >= 1
        assert metadata["n_leaves"] >= 1

    def test_build_cdt_gini_criterion(self, data):
        products, cluster_assignments, sim_matrix, attrs_df = data
        root, metadata = build_cdt(
            sim_matrix,
            cluster_assignments,
            attrs_df,
            min_cluster_size=2,
            quality_threshold=0.3,
            criterion="gini",
        )
        assert metadata["n_nodes"] >= 1

    def test_build_cdt_mixed_criterion(self, data):
        products, cluster_assignments, sim_matrix, attrs_df = data
        root, metadata = build_cdt(
            sim_matrix,
            cluster_assignments,
            attrs_df,
            min_cluster_size=2,
            quality_threshold=0.3,
            criterion="mixed",
            alpha=0.7,
        )
        assert metadata["n_nodes"] >= 1

    def test_build_cdt_high_threshold_passed_false(self, data):
        products, cluster_assignments, sim_matrix, attrs_df = data
        root, metadata = build_cdt(
            sim_matrix,
            cluster_assignments,
            attrs_df,
            min_cluster_size=2,
            quality_threshold=1.0,
        )
        assert not metadata["passed_threshold"]


class TestTreeUtilities:
    def test_count_nodes_and_leaves(self):
        root = TreeNode("n0", "root", ["A", "B"], 0.5, 2, is_leaf=False)
        c1 = TreeNode("n1", "c1", ["A"], 1.0, 1, is_leaf=True)
        c2 = TreeNode("n2", "c2", ["B"], 1.0, 1, is_leaf=True)
        root.children = [c1, c2]
        assert count_nodes(root) == 3
        assert count_leaves(root) == 2

    def test_score_tree_single_leaf(self):
        root = TreeNode("n0", "leaf", ["A", "B"], 0.8, 2, is_leaf=True)
        sim = _make_similarity_matrix(["A", "B"])
        score = score_tree(root, sim)
        assert score >= 0

    def test_tree_to_dataframe(self):
        root = TreeNode("n0", "root", ["A", "B"], 0.5, 2, is_leaf=False)
        c1 = TreeNode("n1", "leaf1", ["A"], 1.0, 1, is_leaf=True)
        c2 = TreeNode("n2", "leaf2", ["B"], 1.0, 1, is_leaf=True)
        root.children = [c1, c2]
        df = tree_to_dataframe(root)
        assert len(df) == 3
        expected_cols = {"node_id", "parent_id", "depth", "name", "n_products", "is_leaf", "similarity_within", "attribute", "attribute_value", "split_criterion", "split_score", "products", "cluster_id"}
        assert expected_cols.issubset(set(df.columns))

    def test_tree_to_json(self):
        root = TreeNode("n0", "root", ["A"], 1.0, 1, is_leaf=True)
        js = tree_to_json(root)
        assert isinstance(js, str)
        assert "n0" in js

    def test_extract_product_attributes(self):
        df = pd.DataFrame({
            "stockcode": ["A", "B", "C"],
            "category": ["Cat1", "Cat2", "Cat1"],
            "brand": ["B1", "B2", "B1"],
        })
        attrs = extract_product_attributes(df, attribute_cols=["category", "brand"])
        assert isinstance(attrs, pd.DataFrame)
        assert list(attrs.index) == ["A", "B", "C"]
        assert list(attrs.columns) == ["category", "brand"]

    def test_extract_product_attributes_no_attrs(self):
        df = pd.DataFrame({
            "stockcode": ["A", "B"],
            "category": ["Cat1", "Cat2"],
        })
        attrs = extract_product_attributes(df, attribute_cols=[])
        assert attrs.empty


class TestBuildCdtEdgeCases:
    def test_single_product(self):
        products = ["A"]
        cluster_assignments = {"A": 0}
        sim = _make_similarity_matrix(products)
        attrs = pd.DataFrame({"category": {"A": "X"}}).reindex(products)
        root, metadata = build_cdt(sim, cluster_assignments, attrs, min_cluster_size=2)
        assert root.is_leaf

    def test_two_products(self):
        products = ["A", "B"]
        cluster_assignments = {"A": 0, "B": 1}
        sim = _make_similarity_matrix(products)
        attrs = pd.DataFrame({"category": {"A": "X", "B": "Y"}}).reindex(products)
        root, metadata = build_cdt(sim, cluster_assignments, attrs, min_cluster_size=2)
        assert metadata["n_nodes"] >= 1

    def test_no_attributes_provided(self):
        products = ["A", "B", "C"]
        cluster_assignments = {"A": 0, "B": 0, "C": 0}
        sim = _make_similarity_matrix(products)
        attrs = pd.DataFrame(index=products)
        root, metadata = build_cdt(
            sim, cluster_assignments, attrs, min_cluster_size=2, candidate_attributes=[]
        )
        assert root.is_leaf
