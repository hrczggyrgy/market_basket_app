"""Unified CDT Service - merges best of cdt_tab.py and cdt_assortment_tab.py"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.application.pipeline import PipelineStage, get_pipeline_store
from src.config import AppConfig, get_config
from src.domain.dto import PipelineResult
from src.infrastructure.logging import (
    AnalysisLogger,
    get_logger,
    log_dataframe_info,
    log_pipeline_stage,
)


@dataclass
class CDTConfig:
    """Configuration for CDT pipeline."""

    # Similarity
    similarity_methods: List[str] = None
    min_cooccurrence: int = 5

    # Community detection
    community_method: str = "label_propagation"
    community_resolution: float = 1.0
    graph_min_weight: float = 0.1
    graph_max_degree: int = 50

    # Clustering
    linkage_method: str = "average"
    min_k: int = 2
    max_k: int = 15

    # Tree building
    min_cluster_size: int = 3
    quality_threshold: float = 0.6
    split_criterion: str = "mutual_info"
    split_alpha: float = 0.5

    # Behavioral
    top_n_products: int = 50
    min_lift: float = 1.2
    max_sub: float = 0.3

    def __post_init__(self):
        if self.similarity_methods is None:
            self.similarity_methods = ["phi"]


class CDTService:
    """Unified CDT service with pipeline integration."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self.pipeline = get_pipeline_store()
        self.logger = get_logger(self.__class__.__name__)

    def _validate_request(self, transactions_df: pd.DataFrame) -> List[str]:
        """Validate request and return warnings."""
        warnings = []

        if transactions_df.empty:
            warnings.append("Empty transaction DataFrame")

        n_transactions = transactions_df["transaction_id"].nunique()
        n_customers = transactions_df["customer_id"].nunique()
        n_products = transactions_df["stockcode"].nunique()

        if n_transactions < 100:
            warnings.append(f"Only {n_transactions} transactions - results may be unreliable")
        if n_customers < 30:
            warnings.append(f"Only {n_customers} customers - segmentation unreliable")
        if n_products < 10:
            warnings.append(f"Only {n_products} products - limited analysis possible")

        return warnings

    def _run_with_logging(
        self,
        stage: PipelineStage,
        computation: callable,
        *args,
        **kwargs,
    ) -> PipelineResult:
        """Run a pipeline stage with logging."""
        logger = get_logger(f"{self.__class__.__name__}.{stage.value}")
        log_pipeline_stage(logger, stage.value, "started")

        result = self.pipeline.execute_stage(stage, computation, *args, **kwargs)

        if result.success:
            log_pipeline_stage(logger, stage.value, "completed", duration_ms=result.duration_ms)
            if result.data is not None:
                log_dataframe_info(logger, stage.value, result.data)
        else:
            log_pipeline_stage(logger, stage.value, "failed", error=result.error)

        return result

    def execute_cdt(
        self,
        transactions_df: pd.DataFrame,
        product_lookup: Dict[str, str],
        cdt_config: CDTConfig,
    ) -> Dict[str, Any]:
        """Execute full CDT pipeline."""
        import time

        start = time.perf_counter()
        warnings = self._validate_request(transactions_df)

        with AnalysisLogger("cdt", "cdt_build") as _:
            # Stage 1: Similarity Matrix
            sim_result = self._run_with_logging(
                PipelineStage.SIMILARITY_MATRIX,
                self._build_similarity_matrices,
                transactions_df,
                cdt_config.similarity_methods,
                cdt_config.min_cooccurrence,
            )

            if not sim_result.success or sim_result.data.empty:
                return {
                    "success": False,
                    "error": "Failed to build similarity matrix",
                    "warnings": warnings,
                }

            similarity_matrices = sim_result.data

            # Use ensemble or first method
            primary_method = cdt_config.similarity_methods[0]
            sim_matrix = similarity_matrices.get(
                "ensemble", similarity_matrices.get(primary_method)
            )

            # Stage 2: Community Detection (optional)
            community_result = None
            if cdt_config.community_method != "none":
                community_result = self._run_with_logging(
                    PipelineStage.SIMILARITY_MATRIX,  # reuse stage
                    self._detect_communities,
                    sim_matrix,
                    cdt_config,
                )

            # Stage 3: Hierarchical Clustering
            cluster_result = self._run_with_logging(
                PipelineStage.HIERARCHICAL_CLUSTERING,
                self._perform_clustering,
                sim_matrix,
                community_result.data if community_result and community_result.success else None,
                cdt_config,
            )

            # Stage 4: CDT Tree Build
            tree_result = self._run_with_logging(
                PipelineStage.CDT_TREE_BUILD,
                self._build_tree,
                cluster_result.data if cluster_result.success else None,
                sim_matrix,
                cdt_config,
                transactions_df,
            )

            # Stage 5: Behavioral Matrices
            behavioral_result = self._run_with_logging(
                PipelineStage.BEHAVIORAL_MATRICES,
                self._build_behavioral_matrices,
                transactions_df,
                sim_matrix,
                tree_result.data if tree_result.success else None,
                cdt_config,
            )

        execution_time = (time.perf_counter() - start) * 1000

        return {
            "success": True,
            "warnings": warnings,
            "execution_time_ms": execution_time,
            "similarity_matrices": similarity_matrices,
            "similarity_matrix": sim_matrix,
            "community_assignments": community_result.data if community_result else None,
            "linkage_matrix": cluster_result.data[0] if cluster_result.success else None,
            "ordered_labels": cluster_result.data[1] if cluster_result.success else None,
            "cluster_assignments": cluster_result.data[2] if cluster_result.success else None,
            "silhouette_scores": cluster_result.data[3] if cluster_result.success else None,
            "optimal_k": cluster_result.data[4] if cluster_result.success else None,
            "tree_root": tree_result.data if tree_result.success else None,
            "tree_metadata": tree_result.metrics if tree_result.success else None,
            "switching_df": behavioral_result.data[0] if behavioral_result.success else None,
            "substitution_df": behavioral_result.data[1] if behavioral_result.success else None,
            "bundling_df": behavioral_result.data[2] if behavioral_result.success else None,
        }

    def _build_similarity_matrices(
        self,
        transactions_df: pd.DataFrame,
        methods: List[str],
        min_cooccurrence: int,
    ) -> Dict[str, pd.DataFrame]:
        """Build similarity matrices using ensemble or single method."""
        from src.analytics.cdt_similarity import build_similarity_matrix_ensemble

        return build_similarity_matrix_ensemble(
            transactions_df,
            methods=methods,
            min_cooccurrence=min_cooccurrence,
        )

    def _detect_communities(
        self,
        sim_matrix: pd.DataFrame,
        cdt_config: CDTConfig,
    ) -> Optional[Dict[str, int]]:
        """Detect communities in product graph."""
        from src.analytics.cdt_community import (
            build_product_graph,
            detect_communities,
        )

        graph = build_product_graph(
            sim_matrix,
            min_weight=cdt_config.graph_min_weight,
            min_joint_customers=cdt_config.min_cooccurrence,
            max_edges_per_node=cdt_config.graph_max_degree,
        )

        if graph.number_of_nodes() == 0:
            return None

        return detect_communities(
            graph,
            method=cdt_config.community_method,
            resolution=cdt_config.community_resolution,
        )

    def _perform_clustering(
        self,
        sim_matrix: pd.DataFrame,
        community_assignments: Optional[Dict[str, int]],
        cdt_config: CDTConfig,
    ) -> Tuple[np.ndarray, List[str], Dict[str, int], Dict[int, float], int]:
        """Perform hierarchical clustering, optionally within communities."""
        from src.analytics.cdt_clustering import (
            find_optimal_clusters,
            get_cluster_assignments,
            perform_hierarchical_clustering,
        )
        from src.analytics.cdt_community import (
            hierarchical_clustering_within_communities,
            merge_community_dendrograms,
        )

        if community_assignments:
            # Cluster within communities
            comm_dendrograms = hierarchical_clustering_within_communities(
                sim_matrix,
                community_assignments,
                linkage_method=cdt_config.linkage_method,
                distance_method="phi",
            )
            linkage_matrix, ordered_labels = merge_community_dendrograms(
                comm_dendrograms, community_assignments
            )
        else:
            # Global clustering
            linkage_matrix, ordered_labels = perform_hierarchical_clustering(
                sim_matrix,
                linkage_method=cdt_config.linkage_method,
                distance_method="phi",
            )

        optimal_k, silhouette_scores = find_optimal_clusters(
            linkage_matrix,
            sim_matrix,
            distance_method="phi",
            min_clusters=cdt_config.min_k,
            max_clusters=min(cdt_config.max_k, len(sim_matrix) - 1),
        )

        cluster_assignments = get_cluster_assignments(
            linkage_matrix, sim_matrix, n_clusters=optimal_k
        )

        return linkage_matrix, ordered_labels, cluster_assignments, silhouette_scores, optimal_k

    def _build_tree(
        self,
        cluster_assignments: Dict[str, int],
        sim_matrix: pd.DataFrame,
        cdt_config: CDTConfig,
        transactions_df: pd.DataFrame,
    ):
        """Build Customer Decision Tree."""
        from src.analytics.cdt_attributes import build_transaction_derived_attributes
        from src.analytics.cdt_tree_builder import (
            build_cdt,
            extract_product_attributes,
        )

        # Extract attributes
        attr_df = extract_product_attributes(
            transactions_df,
            attribute_cols=["category", "brand", "size", "flavour", "flavor", "variant"],
        )

        # Add transaction-derived attributes
        txn_attrs = build_transaction_derived_attributes(
            transactions_df,
            sim_matrix,
            n_tiers=3,
        )
        attr_df = pd.concat([attr_df, txn_attrs], axis=1)

        # Ensure all products in attr_df
        all_products = list(sim_matrix.index)
        attr_df = attr_df.reindex(all_products)

        root, metadata = build_cdt(
            sim_matrix,
            cluster_assignments,
            attr_df,
            min_cluster_size=cdt_config.min_cluster_size,
            quality_threshold=cdt_config.quality_threshold,
            candidate_attributes=None,
            criterion=cdt_config.split_criterion,
            alpha=cdt_config.split_alpha,
        )

        return root, metadata

    def _build_behavioral_matrices(
        self,
        transactions_df: pd.DataFrame,
        sim_matrix: pd.DataFrame,
        tree_root,
        cdt_config: CDTConfig,
    ):
        """Build switching, substitution, and bundling matrices."""
from src.analytics.cdt_behavioral import (
        build_behavioral_matrices,
        compute_affinity_matrix,
    )

        # Build behavioral matrices (switching, substitution, bundling)
        switching_df, substitution_df, bundling_df = build_behavioral_matrices(
            transactions_df,
            sim_matrix,
            affinity_matrix,
            top_n_products=cdt_config.top_n_products,
        )

        return switching_df, substitution_df, bundling_df


def get_cdt_service(config: Optional[AppConfig] = None) -> CDTService:
    """Get CDT service instance."""
    return CDTService(config)
