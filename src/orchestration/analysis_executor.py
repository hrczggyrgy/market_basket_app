"""Analysis executor that resolves dependencies, checks cache, runs engine, stores result."""

from __future__ import annotations

from typing import Any, Optional

from src.orchestration.analysis_registry import AnalysisSpec, get
from src.orchestration.result_store import (
    get,
    get_default,
    get_schema_version_func,
    invalidate,
    param_hash,
    set,
)


class AnalysisExecutor:
    """Executes analyses with dependency resolution and caching.

    Responsibilities:
    - Resolve analysis dependency order (topological sort)
    - Check ResultStore cache before computing
    - Run the analysis engine if cache miss
    - Store result in ResultStore with versioned key
    """

    def __init__(self, dataset_id: str, dataset: Any) -> None:
        """Initialize executor with dataset identifier and DataFrame.

        Args:
            dataset_id: Unique identifier for the dataset.
            dataset: The transaction DataFrame or data source.
        """
        self.dataset_id = dataset_id
        self.dataset = dataset
        self._results: dict[str, Any] = {}

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def execute(self, analysis_key: str, params: dict[str, Any] | None = None) -> Any:
        """Execute an analysis with caching and dependency resolution.

        Args:
            analysis_key: The analysis specification key (e.g., "basket", "pricing").
            params: Optional parameter dictionary for the analysis.

        Returns:
            The analysis result.
        """
        params = params or {}
        analysis_spec = get(analysis_key)

        # Compute deterministic parameter hash
        ph = param_hash(params, schema_version=get_schema_version_func())

        # Check cache first
        cached = get(
            self.dataset_id,
            analysis_key,
            analysis_spec.version,
            ph,
        )
        if cached is not None:
            return cached

        # Ensure dependency analyses are executed first
        self._resolve_dependencies(analysis_spec)

        # Run the analysis engine
        result = self._run_engine(analysis_spec, params)

        # Store in result store
        set(
            self.dataset_id,
            analysis_key,
            analysis_spec.version,
            ph,
            result,
        )

        self._results[analysis_key] = result
        return result

    def _resolve_dependencies(self, spec: AnalysisSpec) -> None:
        """Ensure all dependency analyses have been executed.

        Args:
            spec: The AnalysisSpec for the target analysis.
        """
        deps = spec.dependencies
        # Topological sort not needed at this level; just ensure each dep is executed
        # The caller (AnalysisOrchestrator) handles the full ordering
        for dep_key in deps:
            if dep_key not in self._results:
                # Execute dependency with empty params (or default)
                try:
                    self.execute(dep_key, {})
                except Exception:
                    # Dependency might not be runnable without data/context;
                    # let the engine handle it
                    pass

    def _run_engine(self, spec: AnalysisSpec, params: dict[str, Any]) -> Any:
        """Run the analysis engine for the given spec.

        Dispatches to the appropriate analytics function based on the key.

        Args:
            spec: The AnalysisSpec for the analysis.
            params: Parameter dictionary.

        Returns:
            The analysis result.
        """
        from src.analytics.assortment import optimize_assortment_heuristic
        from src.analytics.basket_metrics import compute_basket_metrics
        from src.analytics.clv import predict_clv_bg_nbd
        from src.analytics.insights.cdt import generate_cdt_insights
        from src.analytics.insights.cohort import generate_cohort_insights
        from src.analytics.insights.overview import generate_overview_insights
        from src.analytics.insights.product import generate_product_insights
        from src.analytics.pricing.pipeline import run_pricing_analysis

        dispatch: dict[str, Any] = {
            "overview": lambda: generate_overview_insights(self.dataset),
            "basket": lambda: compute_basket_metrics(self.dataset),
            "cohorts": lambda: generate_cohort_insights(self.dataset),
            "cdt": lambda: generate_cdt_insights(self.dataset),
            "pricing": lambda: run_pricing_analysis(self.dataset, **params),
            "product": lambda: generate_product_insights(self.dataset),
            "switching": lambda: self._run_switching(),
            "promotion": lambda: self._run_promotion(),
            "cross_sell": lambda: self._run_cross_sell(),
            "segmentation": lambda: self._run_segmentation(),
            "rules": lambda: self._run_rules(),
            "clv": lambda: predict_clv_bg_nbd(self.dataset, **params),
            "assortment": lambda: {"solution": optimize_assortment_heuristic(self.dataset)},
            "network": lambda: self._run_network(),
            "markov": lambda: self._run_markov(),
        }

        if spec.key not in dispatch:
            raise ValueError(f"No engine dispatch for analysis key: {spec.key}")

        return dispatch[spec.key]()

    def _run_switching(self) -> dict[str, Any]:
        from src.analytics.switching import (
            compute_switching_matrix,
            compute_switching_status,
            compute_transition_matrix,
        )
        matrix = compute_switching_matrix(self.dataset)
        status = compute_switching_status(self.dataset)
        transition = compute_transition_matrix(self.dataset)
        return {"matrix": matrix, "status": status, "transition": transition}

    def _run_promotion(self) -> dict[str, Any]:
        from src.analytics.promo import (
            compute_promo_baseline,
            detect_promotions,
            promo_roi_analysis,
        )
        promos = detect_promotions(self.dataset)
        baseline = compute_promo_baseline(self.dataset, promos)
        roi = promo_roi_analysis(self.dataset, promos)
        return {"promos": promos, "baseline": baseline, "roi": roi}

    def _run_cross_sell(self) -> dict[str, Any]:
        from src.analytics.copurchase import get_top_affinity_pairs
        from src.analytics.rules import create_basket_matrix, generate_rules, run_fpgrowth
        basket = create_basket_matrix(self.dataset)
        itemsets = run_fpgrowth(basket)
        rules = generate_rules(itemsets)
        affinity = get_top_affinity_pairs(self.dataset)
        return {"itemsets": itemsets, "rules": rules, "affinity": affinity}

    def _run_segmentation(self) -> dict[str, Any]:
        from src.analytics.segmentation import (
            behavioral_segmentation,
            compute_rfm_features,
            rfm_segmentation,
            value_based_segmentation,
        )
        rfm_features = compute_rfm_features(self.dataset)
        rfm_seg = rfm_segmentation(rfm_features)
        beh_seg = behavioral_segmentation(self.dataset)
        val_seg = value_based_segmentation(self.dataset)
        return {"rfm_features": rfm_features, "rfm_segments": rfm_seg, "behavioral": beh_seg, "value_based": val_seg}

    def _run_rules(self) -> dict[str, Any]:
        from src.analytics.rules import create_basket_matrix, generate_rules, run_fpgrowth
        basket = create_basket_matrix(self.dataset)
        itemsets = run_fpgrowth(basket)
        rules = generate_rules(itemsets)
        return {"itemsets": itemsets, "rules": rules}

    def _run_network(self) -> dict[str, Any]:
        # Network analysis placeholder - could use transference or switching
        from src.analytics.transference import (
            compute_demand_transference_matrix,
            compute_switching_matrix,
        )
        matrix = compute_switching_matrix(self.dataset)
        transference = compute_demand_transference_matrix(self.dataset, matrix)
        return {"transference": transference}

    def _run_markov(self) -> dict[str, Any]:
        # Markov analysis placeholder - could use transition matrices
        from src.analytics.switching import compute_transition_matrix
        transition = compute_transition_matrix(self.dataset)
        return {"transition": transition}

    # ---------------------------------------------------------------------
    # Cache management helpers
    # ---------------------------------------------------------------------

    def invalidate_analysis(self, analysis_key: str, param_hash: Optional[str] = None) -> None:
        """Invalidate a specific analysis's cached results.

        Args:
            analysis_key: The analysis specification key.
            param_hash: Optional specific parameter hash to invalidate.
        """
        spec = get(analysis_key)
        invalidate(
            self.dataset_id,
            analysis_key,
            spec.version,
            param_hash=param_hash,
        )

    def refresh_analysis(self, analysis_key: str, params: dict[str, Any] | None = None) -> Any:
        """Force refresh a cached analysis result.

        Args:
            analysis_key: The analysis specification key.
            params: Optional parameter dictionary.

        Returns:
            The refreshed result.
        """
        self.invalidate_analysis(analysis_key)
        return self.execute(analysis_key, params)

    @classmethod
    def status(cls) -> dict[str, Any]:
        """Get cache status."""
        return get_default().status()
