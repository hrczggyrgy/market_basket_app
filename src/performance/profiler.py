"""Performance profiler for analytics operations.

Provides a `@measure_analysis` decorator that records:
- duration_ms: computation time in milliseconds
- input_rows: number of input rows
- output_rows: number of output rows
- cache_hit: whether result was from cache
- memory_before: RSS memory before computation
- memory_after: RSS memory after computation
- analysis_key: identifier for the analysis
- dataset_id: identifier for the dataset
"""

from __future__ import annotations

import hashlib
import time
from functools import wraps
from typing import Callable, Dict, Optional

import pandas as pd

from src.utils.hashing import df_hash as _df_hash


def _estimate_memory(df: pd.DataFrame) -> int:
    """Estimate DataFrame memory usage in bytes (RSS approximation)."""
    return df.memory_usage(deep=True).sum()


def _make_analysis_key(
    func_name: str,
    dataset_id: str,
    *args,
    **kwargs,
) -> str:
    """Create a deterministic analysis key from function name, dataset_id, and params."""

    key_parts = [func_name, dataset_id]
    # Hash kwargs to keep key deterministic and compact
    kw_str = str(sorted(kwargs.items()))
    key_parts.append(hashlib.md5(kw_str.encode()).hexdigest()[:8])
    return "_".join(key_parts)


class _ProfilerState:
    """Thread-local profiler state (simple, not full Threading)."""

    def __init__(self):
        self._session: Dict[str, Dict] = {}

    def record(
        self,
        analysis_key: str,
        duration_ms: float,
        input_rows: int,
        output_rows: int,
        cache_hit: bool,
        memory_before: int,
        memory_after: int,
    ) -> None:
        if analysis_key not in self._session:
            self._session[analysis_key] = []
        self._session[analysis_key].append(
            {
                "duration_ms": duration_ms,
                "input_rows": input_rows,
                "output_rows": output_rows,
                "cache_hit": cache_hit,
                "memory_before": memory_before,
                "memory_after": memory_after,
            }
        )

    def get_stats(self, analysis_key: str) -> Optional[Dict[str, float]]:
        if analysis_key not in self._session or not self._session[analysis_key]:
            return None
        entries = self._session[analysis_key]
        return {
            "avg_duration_ms": sum(e["duration_ms"] for e in entries)
            / len(entries),
            "avg_input_rows": sum(e["input_rows"] for e in entries)
            / len(entries),
            "avg_output_rows": sum(e["output_rows"] for e in entries) / len(entries),
            "cache_hit_rate": sum(1 for e in entries if e["cache_hit"]) / len(entries),
            "peak_memory_mb": max(e["memory_after"] for e in entries) / (1024 * 1024),
        }


profiler_state = _ProfilerState()


def measure_analysis(
    func: Callable,
) -> Callable:
    """Decorator that profiles analytics function execution.

    Records:
    - duration_ms: wall-clock computation time
    - input_rows: number of rows in input DataFrame(s)
    - output_rows: number of rows in output DataFrame(s) / result size
    - cache_hit: whether result was served from Streamlit cache
    - memory_before: RSS before execution
    - memory_after: RSS after execution
    - analysis_key: deterministic key for this analysis/dataset combo

    The decorated function's return value is unchanged.
    """

    func_name = func.__name__

    @wraps(func)
    def wrapper(*args, **kwargs):
        import streamlit as st

        # Extract dataset_id from DataFrame argument first,
        # then fall back to kwargs or args
        dataset_id = ""
        df_for_hash = None
        # Check for DataFrame in positional args
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                df_for_hash = arg
                break
        # Check for DataFrame in kwargs
        if df_for_hash is None and "df" in kwargs:
            df_for_hash = kwargs["df"]

        if df_for_hash is not None:
            dataset_id = _df_hash(df_for_hash)
        else:
            # Fall back to existing logic: kwargs or args[1]
            dataset_id = kwargs.get("dataset_id", "")
            if not dataset_id and len(args) > 1:
                dataset_id = str(args[1]) if len(args) > 1 else ""

        analysis_key = _make_analysis_key(func_name, dataset_id, *args, **kwargs)

        # Memory before
        import resource

        try:
            memory_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except ImportError:
            # Fallback: estimate from args
            memory_before = 0
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    memory_before += _estimate_memory(arg)
                elif isinstance(arg, dict):
                    for v in arg.values():
                        if isinstance(v, pd.DataFrame):
                            memory_before += _estimate_memory(v)

        # Start timing
        start_time = time.perf_counter()

        # Execute the function
        result = func(*args, **kwargs)

        # End timing
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Memory after
        try:
            memory_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except ImportError:
            memory_after = 0
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    memory_after += _estimate_memory(arg)
                elif isinstance(arg, dict):
                    for v in arg.values():
                        if isinstance(v, pd.DataFrame):
                            memory_after += _estimate_memory(v)

        # Count input/output rows
        input_rows = 0
        output_rows = 0

        # Determine input DataFrame
        input_df = None
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                input_df = arg
                break
        if input_df is None and "df" in kwargs:
            input_df = kwargs["df"]

        if input_df is not None:
            input_rows = len(input_df)

        if isinstance(result, pd.DataFrame):
            output_rows = len(result)
        elif isinstance(result, dict):
            for v in result.values():
                if isinstance(v, pd.DataFrame):
                    output_rows += len(v)

        # Record profiling data
        cache_hit = st.session_state.get("_cache_hit", False)  # type: ignore
        profiler_state.record(
            analysis_key=analysis_key,
            duration_ms=duration_ms,
            input_rows=input_rows,
            output_rows=output_rows,
            cache_hit=cache_hit,
            memory_before=memory_before,
            memory_after=memory_after,
        )

        return result

    return wrapper  # type: ignore


# Convenience: expose stats lookup
def get_profiler_stats(analysis_key: str) -> Optional[Dict[str, float]]:
    """Get aggregated profiler statistics for an analysis key."""
    return profiler_state.get_stats(analysis_key)
