"""Analysis service layer - orchestrates pipeline execution."""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.application.pipeline import PipelineStage, get_pipeline_store
from src.config import AppConfig, get_config
from src.domain.dto import (
    AnalysisMode,
    AnalysisRequest,
    AnalysisResponse,
    PipelineResult,
)
from src.infrastructure.logging import (
    AnalysisLogger,
    get_logger,
    log_dataframe_info,
    log_pipeline_stage,
)


class AnalysisService(ABC):
    """Base class for analysis services."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self.pipeline = get_pipeline_store()
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def get_required_stages(self) -> List[PipelineStage]:
        """Return required pipeline stages for this analysis."""
        pass

    @abstractmethod
    def execute(self, request: AnalysisRequest) -> AnalysisResponse:
        """Execute the analysis."""
        pass

    def _validate_request(self, request: AnalysisRequest) -> List[str]:
        """Validate request and return warnings."""
        warnings = []

        if request.transactions_df.empty:
            warnings.append("Empty transaction DataFrame")

        # Check minimum data requirements
        n_transactions = request.transactions_df["transaction_id"].nunique()
        n_customers = request.transactions_df["customer_id"].nunique()
        n_products = request.transactions_df["stockcode"].nunique()

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


class AssociationRulesService(AnalysisService):
    """Association rules analysis service."""

    def get_required_stages(self) -> List[PipelineStage]:
        return [
            PipelineStage.DATA_LOAD,
            PipelineStage.BASKET_MATRIX,
            PipelineStage.FREQUENT_ITEMSETS,
            PipelineStage.ASSOCIATION_RULES,
        ]

    def execute(self, request: AnalysisRequest) -> AnalysisResponse:
        import time

        from src.algorithms.frequent_itemsets import (
            create_basket_matrix,
            run_fpgrowth,
        )
        from src.rules.generator import generate_rules

        start = time.perf_counter()
        warnings = self._validate_request(request)

        with AnalysisLogger("association_rules", request.mode.value) as _:
            # Stage 1: Basket matrix
            basket_result = self._run_with_logging(
                PipelineStage.BASKET_MATRIX,
                create_basket_matrix,
                request.transactions_df,
            )

            # Stage 2: Frequent itemsets
            fp_params = request.config.get("fpgrowth_params", {})
            freq_result = self._run_with_logging(
                PipelineStage.FREQUENT_ITEMSETS,
                run_fpgrowth,
                basket_result.data,
                min_support=fp_params.get("min_support", self.config.fpgrowth.min_support),
                max_len=fp_params.get("max_len", self.config.fpgrowth.max_itemset_len),
            )

            # Stage 3: Generate rules
            rules_params = request.config.get("rules_params", {})
            rules_result = self._run_with_logging(
                PipelineStage.ASSOCIATION_RULES,
                generate_rules,
                freq_result.data,
                metric=rules_params.get("metric", "confidence"),
                min_threshold=rules_params.get(
                    "min_threshold", self.config.fpgrowth.min_confidence
                ),
            )

        execution_time = (time.perf_counter() - start) * 1000

        return AnalysisResponse(
            mode=request.mode,
            results={
                PipelineStage.BASKET_MATRIX: basket_result,
                PipelineStage.FREQUENT_ITEMSETS: freq_result,
                PipelineStage.ASSOCIATION_RULES: rules_result,
            },
            summary={
                "n_transactions": len(basket_result.data) if basket_result.data is not None else 0,
                "n_frequent_itemsets": len(freq_result.data) if freq_result.data is not None else 0,
                "n_rules": len(rules_result.data) if rules_result.data is not None else 0,
            },
            warnings=warnings,
            execution_time_ms=execution_time,
        )


class CDTService(AnalysisService):
    """Customer Decision Tree analysis service."""

    def get_required_stages(self) -> List[PipelineStage]:
        return [
            PipelineStage.DATA_LOAD,
            PipelineStage.SIMILARITY_MATRIX,
            PipelineStage.HIERARCHICAL_CLUSTERING,
            PipelineStage.CDT_TREE_BUILD,
            PipelineStage.BEHAVIORAL_MATRICES,
        ]

    def execute(self, request: AnalysisRequest) -> AnalysisResponse:
        import time

        start = time.perf_counter()
        warnings = self._validate_request(request)

        with AnalysisLogger("cdt", request.mode.value) as _:
            # Stage 1: Similarity matrix
            from src.analytics.cdt_similarity import build_similarity_matrix

            sim_params = request.config.get("similarity_params", {})
            sim_result = self._run_with_logging(
                PipelineStage.SIMILARITY_MATRIX,
                build_similarity_matrix,
                request.transactions_df,
                method=sim_params.get("method", "phi"),
                min_cooccurrence=sim_params.get(
                    "min_cooccurrence", self.config.cdt.min_cooccurrence
                ),
            )

            # Stage 2: Hierarchical clustering
            from src.analytics.cdt_clustering import (
                find_optimal_clusters,
                get_cluster_assignments,
                perform_hierarchical_clustering,
            )

            cluster_params = request.config.get("cluster_params", {})
            link_result = self._run_with_logging(
                PipelineStage.HIERARCHICAL_CLUSTERING,
                perform_hierarchical_clustering,
                sim_result.data,
                linkage_method=cluster_params.get("linkage_method", self.config.cdt.linkage_method),
                distance_method=sim_params.get("method", "phi"),
            )

            opt_result = self._run_with_logging(
                PipelineStage.HIERARCHICAL_CLUSTERING,
                find_optimal_clusters,
                link_result.data[0],
                sim_result.data,
                distance_method=sim_params.get("method", "phi"),
                min_clusters=cluster_params.get("min_k", self.config.cdt.min_k),
                max_clusters=cluster_params.get("max_k", self.config.cdt.max_k),
            )

            assign_result = self._run_with_logging(
                PipelineStage.HIERARCHICAL_CLUSTERING,
                get_cluster_assignments,
                link_result.data[0],
                sim_result.data,
                n_clusters=opt_result.data,
            )

            # Stage 3: CDT Tree build
            from src.analytics.cdt_tree_builder import (
                build_cdt,
                extract_product_attributes,
            )

            tree_params = request.config.get("tree_params", {})
            attr_result = self._run_with_logging(
                PipelineStage.CDT_TREE_BUILD,
                extract_product_attributes,
                request.transactions_df,
            )

            cdt_result = self._run_with_logging(
                PipelineStage.CDT_TREE_BUILD,
                build_cdt,
                sim_result.data,
                assign_result.data,
                attr_result.data,
                min_cluster_size=tree_params.get(
                    "min_cluster_size", self.config.cdt.min_cluster_size
                ),
                quality_threshold=tree_params.get(
                    "quality_threshold", self.config.cdt.quality_threshold
                ),
                criterion=tree_params.get("split_criterion", self.config.cdt.split_criterion),
                alpha=tree_params.get("split_alpha", self.config.cdt.split_alpha),
            )

            # Stage 4: Behavioral matrices
            from src.analytics.cdt_behavioral import build_behavioral_matrices

            beh_result = self._run_with_logging(
                PipelineStage.BEHAVIORAL_MATRICES,
                build_behavioral_matrices,
                request.transactions_df,
                cdt_result.data,
                sim_result.data,
            )

        execution_time = (time.perf_counter() - start) * 1000

        return AnalysisResponse(
            mode=request.mode,
            results={
                PipelineStage.SIMILARITY_MATRIX: sim_result,
                PipelineStage.HIERARCHICAL_CLUSTERING: link_result,
                PipelineStage.CDT_TREE_BUILD: cdt_result,
                PipelineStage.BEHAVIORAL_MATRICES: beh_result,
            },
            summary={
                "n_products": len(sim_result.data) if sim_result.data is not None else 0,
                "n_clusters": opt_result.data,
                "tree_quality": cdt_result.metrics.get("quality_ratio", 0),
                "tree_passed_threshold": cdt_result.metrics.get("passed_threshold", False),
            },
            warnings=warnings,
            execution_time_ms=execution_time,
        )


class SegmentationService(AnalysisService):
    """Customer segmentation service."""

    def get_required_stages(self) -> List[PipelineStage]:
        return [
            PipelineStage.DATA_LOAD,
            PipelineStage.CUSTOMER_FEATURES,
            PipelineStage.SEGMENTATION,
            PipelineStage.CLV_PREDICTION,
        ]

    def execute(self, request: AnalysisRequest) -> AnalysisResponse:
        import time

        from src.analytics.segmentation import (
            behavioral_segmentation,
            compute_rfm_features,
            value_based_segmentation,
        )

        start = time.perf_counter()
        warnings = self._validate_request(request)

        with AnalysisLogger("segmentation", request.mode.value) as _:
            # Stage 1: RFM features
            rfm_result = self._run_with_logging(
                PipelineStage.CUSTOMER_FEATURES,
                compute_rfm_features,
                request.transactions_df,
            )

            # Stage 2: Segmentation
            seg_params = request.config.get("segmentation_params", {})
            method = seg_params.get("method", self.config.segmentation.method)

            if method == "behavioral":
                seg_result = self._run_with_logging(
                    PipelineStage.SEGMENTATION,
                    behavioral_segmentation,
                    request.transactions_df,
                    n_clusters=seg_params.get("n_clusters", self.config.segmentation.n_segments),
                )
            elif method == "rfm_kmeans":
                from src.analytics.segmentation import rfm_segmentation

                seg_result = self._run_with_logging(
                    PipelineStage.SEGMENTATION,
                    rfm_segmentation,
                    rfm_result.data,
                    method="kmeans",
                    n_segments=seg_params.get("n_segments", self.config.segmentation.n_segments),
                )
            else:
                from src.analytics.segmentation import rfm_segmentation

                seg_result = self._run_with_logging(
                    PipelineStage.SEGMENTATION,
                    rfm_segmentation,
                    rfm_result.data,
                    method="quantile",
                )

            # Stage 3: CLV
            clv_params = request.config.get("clv_params", {})
            clv_result = self._run_with_logging(
                PipelineStage.CLV_PREDICTION,
                value_based_segmentation,
                request.transactions_df,
                prediction_horizon_days=clv_params.get(
                    "value_horizon", self.config.segmentation.value_horizon
                ),
            )

        execution_time = (time.perf_counter() - start) * 1000

        return AnalysisResponse(
            mode=request.mode,
            results={
                PipelineStage.CUSTOMER_FEATURES: rfm_result,
                PipelineStage.SEGMENTATION: seg_result,
                PipelineStage.CLV_PREDICTION: clv_result,
            },
            summary={
                "n_segments": seg_result.data["segment"].nunique()
                if seg_result.data is not None
                else 0,
                "n_customers": len(rfm_result.data) if rfm_result.data is not None else 0,
            },
            warnings=warnings,
            execution_time_ms=execution_time,
        )


class ElasticityService(AnalysisService):
    """Price elasticity analysis service."""

    def get_required_stages(self) -> List[PipelineStage]:
        return [
            PipelineStage.DATA_LOAD,
            PipelineStage.ELASTICITY_ESTIMATION,
            PipelineStage.KVI_SCORING,
        ]

    def execute(self, request: AnalysisRequest) -> AnalysisResponse:
        import time

        start = time.perf_counter()
        warnings = self._validate_request(request)

        with AnalysisLogger("elasticity", request.mode.value) as _:
            # Stage 1: Elasticity estimation
            from src.analytics.pricing import estimate_loglog_elasticity

            elast_params = request.config.get("elasticity_params", {})
            elast_result = self._run_with_logging(
                PipelineStage.ELASTICITY_ESTIMATION,
                estimate_loglog_elasticity,
                request.transactions_df,
                min_periods=elast_params.get("min_periods", self.config.elasticity.min_periods),
                min_price_variation=elast_params.get(
                    "min_price_variation", self.config.elasticity.min_price_variation
                ),
            )

            # Stage 2: KVI scoring
            from src.analytics.pricing import compute_kvi_score

            kvi_result = self._run_with_logging(
                PipelineStage.KVI_SCORING,
                compute_kvi_score,
                request.transactions_df,
                elasticity_df=elast_result.data,
            )

        execution_time = (time.perf_counter() - start) * 1000

        return AnalysisResponse(
            mode=request.mode,
            results={
                PipelineStage.ELASTICITY_ESTIMATION: elast_result,
                PipelineStage.KVI_SCORING: kvi_result,
            },
            summary={
                "n_products_analyzed": len(elast_result.data)
                if elast_result.data is not None
                else 0,
                "mean_elasticity": elast_result.data["elasticity"].mean()
                if elast_result.data is not None
                else 0,
            },
            warnings=warnings,
            execution_time_ms=execution_time,
        )


class PromoUpliftService(AnalysisService):
    """Promotional uplift modeling service."""

    def get_required_stages(self) -> List[PipelineStage]:
        return [
            PipelineStage.DATA_LOAD,
            PipelineStage.PROMO_DETECTION,
            PipelineStage.UPLIFT_MODELING,
        ]

    def execute(self, request: AnalysisRequest) -> AnalysisResponse:
        import time

        start = time.perf_counter()
        warnings = self._validate_request(request)

        with AnalysisLogger("promo_uplift", request.mode.value) as _:
            # Stage 1: Promo detection
            from src.analytics.promo_uplift import build_uplift_dataset, detect_promotions

            promo_params = request.config.get("promo_params", {})
            promo_result = self._run_with_logging(
                PipelineStage.PROMO_DETECTION,
                detect_promotions,
                request.transactions_df,
                price_change_threshold=promo_params.get(
                    "drop_threshold", self.config.promo_uplift.drop_threshold
                ),
                baseline_window=promo_params.get(
                    "baseline_window", self.config.promo_uplift.baseline_window
                ),
            )

            # Stage 2: Uplift modeling
            uplift_params = request.config.get("uplift_params", {})
            from src.analytics.promo_uplift import (
                evaluate_uplift_model,
                train_s_learner_uplift,
                train_t_learner_uplift,
            )

            method = uplift_params.get("method", self.config.promo_uplift.uplift_method)

            if method == "t_learner":
                model_result = self._run_with_logging(
                    PipelineStage.UPLIFT_MODELING,
                    train_t_learner_uplift,
                    *build_uplift_dataset(request.transactions_df, promo_result.data),
                    n_estimators=uplift_params.get(
                        "base_n_estimators", self.config.promo_uplift.base_n_estimators
                    ),
                    max_depth=uplift_params.get(
                        "base_max_depth", self.config.promo_uplift.base_max_depth
                    ),
                )
            else:
                model_result = self._run_with_logging(
                    PipelineStage.UPLIFT_MODELING,
                    train_s_learner_uplift,
                    *build_uplift_dataset(request.transactions_df, promo_result.data),
                    n_estimators=uplift_params.get(
                        "base_n_estimators", self.config.promo_uplift.base_n_estimators
                    ),
                    max_depth=uplift_params.get(
                        "base_max_depth", self.config.promo_uplift.base_max_depth
                    ),
                )

            # Stage 3: Evaluation
            eval_result = self._run_with_logging(
                PipelineStage.UPLIFT_MODELING,
                evaluate_uplift_model,
                model_result.data,
                *build_uplift_dataset(request.transactions_df, promo_result.data),
            )

        execution_time = (time.perf_counter() - start) * 1000

        return AnalysisResponse(
            mode=request.mode,
            results={
                PipelineStage.PROMO_DETECTION: promo_result,
                PipelineStage.UPLIFT_MODELING: eval_result,
            },
            summary={
                "n_promos_detected": len(promo_result.data) if promo_result.data is not None else 0,
                "qini": eval_result.metrics.get("qini", 0),
                "auuc": eval_result.metrics.get("auuc", 0),
            },
            warnings=warnings,
            execution_time_ms=execution_time,
        )


# Service registry
SERVICE_REGISTRY = {
    AnalysisMode.ASSOCIATION_RULES: AssociationRulesService,
    AnalysisMode.COPURCHASE: AssociationRulesService,
    AnalysisMode.ADDON: AssociationRulesService,
    AnalysisMode.SWITCHING: AssociationRulesService,
    AnalysisMode.CDT_BUILDER: CDTService,
    AnalysisMode.CDT_BENCHMARK: CDTService,
    AnalysisMode.DEMAND_TRANSFERENCE: CDTService,
    AnalysisMode.ASSORTMENT_OPTIMIZER: CDTService,
    AnalysisMode.CUSTOMER_SEGMENTATION: SegmentationService,
    AnalysisMode.PRODUCT_PERFORMANCE: SegmentationService,
    AnalysisMode.COHORT_ANALYSIS: SegmentationService,
    AnalysisMode.ELASTICITY_ANALYSIS: ElasticityService,
    AnalysisMode.KVI_IDENTIFICATION: ElasticityService,
    AnalysisMode.PRICE_CURVE_DIAGNOSTICS: ElasticityService,
    AnalysisMode.PROMO_UPLIFT_MODELING: PromoUpliftService,
    AnalysisMode.ELASTICITY_BENCHMARK: ElasticityService,
}


def get_analysis_service(mode: AnalysisMode, config: Optional[AppConfig] = None) -> AnalysisService:
    """Get analysis service for a mode."""
    service_class = SERVICE_REGISTRY.get(mode)
    if service_class is None:
        raise ValueError(f"No service registered for mode: {mode}")
    return service_class(config)
