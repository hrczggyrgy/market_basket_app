"""Pipeline state management - replaces session_state + pipeline dict duality."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar

import pandas as pd

from src.domain.dto import PipelineResult, PipelineStage

T = TypeVar("T")


@dataclass
class PipelineState:
    """Immutable pipeline state snapshot."""

    stage_data: Dict[PipelineStage, Any] = field(default_factory=dict)
    stage_results: Dict[PipelineStage, PipelineResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 0

    def get(self, stage: PipelineStage, default: T = None) -> T:
        """Get data from a stage."""
        return self.stage_data.get(stage, default)

    def get_result(self, stage: PipelineStage) -> Optional[PipelineResult]:
        """Get full result for a stage."""
        return self.stage_results.get(stage)

    def has_data(self, stage: PipelineStage) -> bool:
        """Check if stage has non-empty data."""
        val = self.stage_data.get(stage)
        if val is None:
            return False
        if isinstance(val, pd.DataFrame):
            return not val.empty
        if isinstance(val, (list, dict, set)):
            return len(val) > 0
        return True

    def with_data(self, stage: PipelineStage, data: Any) -> "PipelineState":
        """Return new state with updated stage data (immutable)."""
        new_state = PipelineState(
            stage_data={**self.stage_data, stage: data},
            stage_results=self.stage_results.copy(),
            metadata=self.metadata.copy(),
            created_at=self.created_at,
            updated_at=datetime.now(),
            version=self.version + 1,
        )
        return new_state

    def with_result(self, stage: PipelineStage, result: PipelineResult) -> "PipelineState":
        """Return new state with updated stage result (immutable)."""
        new_state = PipelineState(
            stage_data=self.stage_data.copy(),
            stage_results={**self.stage_results, stage: result},
            metadata=self.metadata.copy(),
            created_at=self.created_at,
            updated_at=datetime.now(),
            version=self.version + 1,
        )
        return new_state


class PipelineStore:
    """
    Thread-safe pipeline store with history and invalidation.

    Replaces the dual session_state + pipeline dict approach with a single
    source of truth that supports:
    - Immutable state transitions
    - Stage dependency tracking
    - Cache invalidation
    - History for debugging
    """

    def __init__(self):
        self._current: PipelineState = PipelineState()
        self._history: List[PipelineState] = []
        self._dependencies: Dict[PipelineStage, List[PipelineStage]] = {
            PipelineStage.BASKET_MATRIX: [PipelineStage.DATA_LOAD],
            PipelineStage.FREQUENT_ITEMSETS: [PipelineStage.BASKET_MATRIX],
            PipelineStage.ASSOCIATION_RULES: [PipelineStage.FREQUENT_ITEMSETS],
            PipelineStage.SIMILARITY_MATRIX: [PipelineStage.DATA_LOAD],
            PipelineStage.HIERARCHICAL_CLUSTERING: [PipelineStage.SIMILARITY_MATRIX],
            PipelineStage.CDT_TREE_BUILD: [
                PipelineStage.HIERARCHICAL_CLUSTERING,
                PipelineStage.SIMILARITY_MATRIX,
            ],
            PipelineStage.BEHAVIORAL_MATRICES: [
                PipelineStage.CDT_TREE_BUILD,
                PipelineStage.SIMILARITY_MATRIX,
            ],
            PipelineStage.CUSTOMER_FEATURES: [PipelineStage.DATA_LOAD],
            PipelineStage.SEGMENTATION: [PipelineStage.CUSTOMER_FEATURES],
            PipelineStage.CLV_PREDICTION: [PipelineStage.CUSTOMER_FEATURES],
            PipelineStage.ELASTICITY_ESTIMATION: [PipelineStage.DATA_LOAD],
            PipelineStage.KVI_SCORING: [
                PipelineStage.ELASTICITY_ESTIMATION,
                PipelineStage.DATA_LOAD,
            ],
            PipelineStage.PRICE_TIERS: [
                PipelineStage.DATA_LOAD,
                PipelineStage.ELASTICITY_ESTIMATION,
            ],
            PipelineStage.PROMO_DETECTION: [PipelineStage.DATA_LOAD],
            PipelineStage.UPLIFT_MODELING: [PipelineStage.PROMO_DETECTION, PipelineStage.DATA_LOAD],
            PipelineStage.DEMAND_TRANSFERENCE: [
                PipelineStage.CDT_TREE_BUILD,
                PipelineStage.SIMILARITY_MATRIX,
            ],
            PipelineStage.ASSORTMENT_OPTIMIZATION: [
                PipelineStage.DEMAND_TRANSFERENCE,
                PipelineStage.CDT_TREE_BUILD,
            ],
        }
        self._lock = None  # Use threading.Lock() in production if needed

    @property
    def current(self) -> PipelineState:
        """Get current pipeline state."""
        return self._current

    @property
    def history(self) -> List[PipelineState]:
        """Get state history."""
        return self._history

    def get(self, stage: PipelineStage, default: T = None) -> T:
        """Get data for a stage."""
        return self._current.get(stage, default)

    def get_result(self, stage: PipelineStage) -> Optional[PipelineResult]:
        """Get full result for a stage."""
        return self._current.get_result(stage)

    def has_data(self, stage: PipelineStage) -> bool:
        """Check if stage has data."""
        return self._current.has_data(stage)

    def set_data(self, stage: PipelineStage, data: Any) -> PipelineState:
        """Set data for a stage (returns new state)."""
        # Save current state to history
        self._history.append(self._current)
        if len(self._history) > 100:  # Limit history
            self._history = self._history[-50:]

        # Update state
        self._current = self._current.with_data(stage, data)
        return self._current

    def set_result(self, stage: PipelineStage, result: PipelineResult) -> PipelineState:
        """Set result for a stage (returns new state)."""
        self._history.append(self._current)
        if len(self._history) > 100:
            self._history = self._history[-50:]

        self._current = self._current.with_result(stage, result)
        return self._current

    def execute_stage(
        self,
        stage: PipelineStage,
        computation: callable,
        *args,
        **kwargs,
    ) -> PipelineResult:
        """
        Execute a pipeline stage with timing and error handling.

        Args:
            stage: Pipeline stage being executed
            computation: Callable that performs the computation
            *args, **kwargs: Arguments passed to computation

        Returns:
            PipelineResult with data, metrics, or error
        """
        import time

        start = time.perf_counter()
        try:
            data = computation(*args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000

            result = PipelineResult(
                stage=stage,
                success=True,
                data=data,
                duration_ms=duration_ms,
            )

            self.set_result(stage, result)
            self.set_data(stage, data)

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            result = PipelineResult(
                stage=stage,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )
            self.set_result(stage, result)
            return result

    def invalidate_downstream(self, from_stage: PipelineStage) -> None:
        """Invalidate all stages that depend on the given stage."""
        to_invalidate = self._get_downstream_stages(from_stage)

        for stage in to_invalidate:
            if stage in self._current.stage_data:
                self._history.append(self._current)
                self._current = PipelineState(
                    stage_data={k: v for k, v in self._current.stage_data.items() if k != stage},
                    stage_results={
                        k: v for k, v in self._current.stage_results.items() if k != stage
                    },
                    metadata=self._current.metadata.copy(),
                    created_at=self._current.created_at,
                    updated_at=datetime.now(),
                    version=self._current.version + 1,
                )

    def _get_downstream_stages(self, stage: PipelineStage) -> List[PipelineStage]:
        """Get all stages that transitively depend on the given stage."""
        downstream = []
        visited = set()

        def dfs(s: PipelineStage):
            if s in visited:
                return
            visited.add(s)
            for dependent, deps in self._dependencies.items():
                if s in deps:
                    downstream.append(dependent)
                    dfs(dependent)

        dfs(stage)
        return downstream

    def clear(self) -> None:
        """Clear all pipeline state."""
        self._history.append(self._current)
        self._current = PipelineState()

    def get_summary(self) -> Dict[str, str]:
        """Get human-readable summary of pipeline state."""
        summary = {}
        for stage in PipelineStage:
            if self.has_data(stage):
                val = self._current.stage_data[stage]
                if isinstance(val, pd.DataFrame):
                    summary[stage.value] = f"DataFrame({len(val)} rows × {len(val.columns)} cols)"
                elif isinstance(val, dict):
                    summary[stage.value] = f"dict({len(val)} keys)"
                elif isinstance(val, list):
                    summary[stage.value] = f"list({len(val)} items)"
                else:
                    summary[stage.value] = f"{type(val).__name__}"
            else:
                summary[stage.value] = "empty"
        return summary

    def export_state(self) -> Dict[str, Any]:
        """Export pipeline state for serialization."""
        return {
            "stage_data": {
                k.value: v.to_dict() if isinstance(v, pd.DataFrame) else v
                for k, v in self._current.stage_data.items()
            },
            "stage_results": {
                k.value: {
                    "stage": r.stage.value,
                    "success": r.success,
                    "metrics": r.metrics,
                    "error": r.error,
                    "duration_ms": r.duration_ms,
                    "timestamp": r.timestamp.isoformat(),
                }
                for k, r in self._current.stage_results.items()
            },
            "metadata": self._current.metadata,
            "version": self._current.version,
        }


# Global pipeline store
_store: Optional[PipelineStore] = None


def get_pipeline_store() -> PipelineStore:
    """Get global pipeline store instance."""
    global _store
    if _store is None:
        _store = PipelineStore()
    return _store


def reset_pipeline_store() -> PipelineStore:
    """Reset pipeline store (for testing)."""
    global _store
    _store = PipelineStore()
    return _store
