"""Tests for the CDT dendrogram visualization and supporting functions."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy.cluster.hierarchy import linkage

from src.analytics.cdt_clustering import (
    get_cluster_assignments,
    perform_hierarchical_clustering,
)
from src.analytics.cdt_community import merge_community_dendrograms
from src.ui.cdt_assortment_tab import create_dendrogram_plot
from src.viz.cdt_viz import plot_dendrogram

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_linkage():
    """Valid 3-item linkage matrix (ward)."""
    X = np.array([[1.0, 2.0], [2.0, 3.0], [5.0, 6.0]])
    return linkage(X, method="ward")


@pytest.fixture
def medium_linkage():
    """Valid 6-item linkage matrix."""
    X = np.array([[1, 2], [2, 3], [3, 4], [7, 8], [8, 9], [9, 10]])
    return linkage(X, method="ward")


@pytest.fixture
def similarity_df():
    """3x3 similarity matrix as DataFrame."""
    return pd.DataFrame(
        [[1.0, 0.8, 0.3], [0.8, 1.0, 0.2], [0.3, 0.2, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )


# ---------------------------------------------------------------------------
# plot_dendrogram
# ---------------------------------------------------------------------------


class TestPlotDendrogram:
    def test_valid_linkage_returns_figure(self, small_linkage):
        fig = plot_dendrogram(small_linkage, ["A", "B", "C"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_medium_linkage_returns_figure(self, medium_linkage):
        labels = [f"P{i}" for i in range(6)]
        fig = plot_dendrogram(medium_linkage, labels)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_empty_linkage_returns_figure_with_annotation(self):
        fig = plot_dendrogram(np.array([]), [])
        assert isinstance(fig, go.Figure)
        annotations = fig.layout.annotations
        assert len(annotations) > 0
        assert "No clustering data" in annotations[0]["text"]

    def test_orientation_bottom(self, small_linkage):
        fig_bottom = plot_dendrogram(small_linkage, ["A", "B", "C"], orientation="bottom")
        assert fig_bottom.layout.xaxis.title["text"] == "Distance"

    def test_orientation_right(self, small_linkage):
        fig_right = plot_dendrogram(small_linkage, ["A", "B", "C"], orientation="right")
        assert fig_right.layout.yaxis.title["text"] == "Distance"

    def test_color_threshold(self, small_linkage):
        fig = plot_dendrogram(small_linkage, ["A", "B", "C"], color_threshold=0.5)
        assert isinstance(fig, go.Figure)

    def test_height_width_params(self, small_linkage):
        fig = plot_dendrogram(small_linkage, ["A", "B", "C"], height=400, width=500)
        assert fig.layout.height == 400
        assert fig.layout.width == 500

    def test_labels_match_items(self):
        Z = linkage(np.array([[1.0], [2.0], [3.0]]), method="ward")
        fig = plot_dendrogram(Z, ["X", "Y", "Z"])
        assert isinstance(fig, go.Figure)

    def test_two_item_linkage(self):
        Z = linkage(np.array([[1.0], [2.0]]), method="ward")
        fig = plot_dendrogram(Z, ["A", "B"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_three_item_linkage_with_labels(self):
        Z = linkage(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), method="ward")
        fig = plot_dendrogram(Z, ["A", "B", "C"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# create_dendrogram_plot (cdt_assortment_tab)
# ---------------------------------------------------------------------------


class TestCreateDendrogramPlot:
    def test_valid_linkage_delegates_to_plot_dendrogram(self, small_linkage):
        fig = create_dendrogram_plot(small_linkage, labels=["A", "B", "C"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_empty_linkage_returns_annotation(self):
        fig = create_dendrogram_plot(np.array([]))
        assert isinstance(fig, go.Figure)
        annotations = fig.layout.annotations
        assert len(annotations) > 0
        assert "No clustering data" in annotations[0]["text"]

    def test_valid_linkage_no_labels_provided(self, small_linkage):
        fig = create_dendrogram_plot(small_linkage)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# perform_hierarchical_clustering → plot_dendrogram integration
# ---------------------------------------------------------------------------


class TestClusteringToDendrogram:
    def test_full_pipeline_on_synthetic_data(self, similarity_df):
        """Run perform_hierarchical_clustering then plot_dendrogram."""
        Z, labels = perform_hierarchical_clustering(similarity_df, linkage_method="average")
        assert Z.size > 0
        assert len(labels) == 3
        fig = plot_dendrogram(Z, labels)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_empty_similarity_matrix(self):
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="cannot be determined on an empty"):
            perform_hierarchical_clustering(empty_df, linkage_method="average")

    def test_full_pipeline_ordered_labels_match(self, similarity_df):
        """Verify that ordered labels returned are a permutation of input."""
        Z, labels = perform_hierarchical_clustering(similarity_df, linkage_method="average")
        assert set(labels) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# merge_community_dendrograms → plot_dendrogram integration
# ---------------------------------------------------------------------------


class TestCommunityMergeToDendrogram:
    @pytest.fixture
    def community_data(self):
        """Two communities with 3 items each."""
        X1 = np.array([[1.0, 1.0], [1.5, 1.5], [2.0, 2.0]])
        X2 = np.array([[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]])
        Z1 = linkage(X1, method="ward")
        Z2 = linkage(X2, method="ward")
        dendrograms = {
            0: (Z1, ["S0", "S1", "S2"]),
            1: (Z2, ["S3", "S4", "S5"]),
        }
        assignments = {"S0": 0, "S1": 0, "S2": 0, "S3": 1, "S4": 1, "S5": 1}
        return dendrograms, assignments

    def test_merge_two_communities(self, community_data):
        dendrograms, assignments = community_data
        merged_Z, merged_labels = merge_community_dendrograms(dendrograms, assignments)
        assert merged_Z.size > 0
        assert len(merged_labels) == 6
        # merged_Z is display-only (may not be valid scipy linkage);
        # skip plot_dendrogram test here

    def test_merge_single_community(self):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        Z = linkage(X, method="ward")
        dendrograms = {0: (Z, ["A", "B", "C"])}
        assignments = {"A": 0, "B": 0, "C": 0}
        merged_Z, merged_labels = merge_community_dendrograms(dendrograms, assignments)
        assert merged_Z.size > 0
        assert merged_labels == ["A", "B", "C"]

    def test_merge_empty_dict(self):
        merged_Z, merged_labels = merge_community_dendrograms({}, {})
        assert merged_Z.size == 0
        assert merged_labels == []
        fig = plot_dendrogram(merged_Z, merged_labels)
        assert isinstance(fig, go.Figure)

    def test_merge_with_empty_linkage(self):
        dendrograms = {0: (np.array([]), []), 1: (np.array([]), [])}
        assignments = {}
        merged_Z, merged_labels = merge_community_dendrograms(dendrograms, assignments)
        assert merged_Z.size == 0


# ---------------------------------------------------------------------------
# get_cluster_assignments (uses linkage output)
# ---------------------------------------------------------------------------


class TestGetClusterAssignments:
    def test_valid_linkage(self, small_linkage, similarity_df):
        assignments = get_cluster_assignments(small_linkage, similarity_df, n_clusters=2)
        assert len(assignments) == 3
        assert all(cluster in (0, 1) for cluster in assignments.values())

    def test_empty_linkage_returns_per_product(self, similarity_df):
        assignments = get_cluster_assignments(np.array([]), similarity_df, n_clusters=2)
        assert len(assignments) == 3
        assert len(set(assignments.values())) == 3  # each its own cluster

    def test_single_cluster(self, small_linkage, similarity_df):
        assignments = get_cluster_assignments(small_linkage, similarity_df, n_clusters=1)
        assert len(assignments) == 3
        assert all(cluster == 0 for cluster in assignments.values())
