"""Readiness engine computing analysis status (READY/CALCULATING/CACHED/NOT_AVAILABLE/INSUFFICIENT_DATA/ADVANCED)."""

from __future__ import annotations

from typing import Any, Tuple

from src.orchestration.analysis_registry import ANALYSIS_SPECS, AnalysisSpec
from src.orchestration.result_store import (
    get_schema_version_func,
    has,
    param_hash,
)

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

READY = "READY"
CALCULATING = "CALCULATING"
CACHED = "CACHED"
NOT_AVAILABLE = "NOT_AVAILABLE"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
ADVANCED = "ADVANCED"


# Minimum data points required per tier
_MIN_DATA_FOR_TIER = {"A": 10, "B": 100, "C": 1000}


def _min_data_for_tier(tier: str) -> int:
    """Return minimum data points required for a tier."""
    return _MIN_DATA_FOR_TIER.get(tier, 100)


class ReadinessEngine:
    """Computes the readiness status for each analysis.

    Status flow:
    - NOT_AVAILABLE: No data, or analysis not registered
    - INSUFFICIENT_DATA: Data exists but too little for the analysis
    - CALCULATING: Analysis is currently being computed
    - CACHED: Result is available in the result store
    - READY: Result is available and can be displayed
    - ADVANCED: Analysis requires special conditions (high memory, long time)
    """

    @staticmethod
    def compute_status(
        analysis_key: str,
        dataset: Any,
        *,
        dataset_id: str = "default",
    ) -> Tuple[str, dict[str, Any]]:
        """Compute the readiness status for a given analysis.

        Args:
            analysis_key: The analysis specification key.
            dataset: The transaction DataFrame.
            dataset_id: Identifier for the dataset.

        Returns:
            Tuple of (status, metadata) where metadata includes version, ttl, etc.
        """
        # Check if analysis is registered
        if analysis_key not in ANALYSIS_SPECS:
            return NOT_AVAILABLE, {"reason": f"Analysis '{analysis_key}' not registered"}

        spec = ANALYSIS_SPECS[analysis_key]

        # Check data availability
        if dataset is None or (hasattr(dataset, "__len__") and len(dataset) == 0):
            return NOT_AVAILABLE, {"reason": "No data available", "tier": spec.tier}

        # Check minimum data requirements based on tier
        if hasattr(dataset, "__len__"):
            data_len = len(dataset)
            if data_len < _min_data_for_tier(spec.tier):
                return INSUFFICIENT_DATA, {
                    "reason": "Insufficient data",
                    "data_points": data_len,
                    "tier": spec.tier,
                    "min_required": _min_data_for_tier(spec.tier),
                }

        # Check cache
        try:
            hp = param_hash({}, schema_version=get_schema_version_func())
            cached = has(dataset_id, analysis_key, spec.version, hp)
            if cached:
                return CACHED, {"tier": spec.tier, "version": spec.version}
        except Exception:
            pass

        # Determine status based on tier
        if spec.tier == "A":
            return READY, {"tier": spec.tier, "version": spec.version, "instant": True}

        # Check dependencies
        deps_met = ReadinessEngine._check_dependencies(spec, dataset_id)
        if not deps_met:
            return CALCULATING, {"tier": spec.tier, "version": spec.version, "reason": "dependencies_not_met"}

        # Tier C: advanced computation
        if spec.tier == "C":
            return ADVANCED, {
                "tier": spec.tier,
                "version": spec.version,
                "reason": "heavy_computation",
                "max_memory_mb": spec.max_memory_mb,
                "expected_seconds": spec.expected_seconds,
            }

        # Tier B: lazy cached
        return READY, {"tier": spec.tier, "version": spec.version, "cacheable": spec.cacheable}

    @staticmethod
    def _check_dependencies(spec: AnalysisSpec, dataset_id: str = "default") -> bool:
        """Check if all dependency analyses have cached results.

        Args:
            spec: The AnalysisSpec to check.
            dataset_id: The dataset identifier for checking cache.

        Returns:
            True if all dependencies have cached results.
        """
        from src.orchestration.result_store import get_schema_version_func, has, param_hash

        for dep_key in spec.dependencies:
            if dep_key not in ANALYSIS_SPECS:
                continue
            dep_spec = ANALYSIS_SPECS[dep_key]
            # Tier A dependencies are always "met" since they recompute instantly
            if dep_spec.tier == "A":
                continue
            # For other tiers, check if they have cached results
            hp = param_hash({}, schema_version=get_schema_version_func())
            if not has(dataset_id, dep_key, dep_spec.version, hp):
                return False
        return True
