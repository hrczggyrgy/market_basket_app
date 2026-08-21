"""Memory monitoring utilities for performance tracking."""

from __future__ import annotations

import os
from typing import Any, Optional

import pandas as pd
import psutil


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB (RSS)."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_memory_usage_bytes() -> int:
    """Get current process memory usage in bytes (RSS)."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def get_memory_percent() -> float:
    """Get current process memory usage as percentage of system memory."""
    process = psutil.Process(os.getpid())
    return process.memory_percent()


def get_system_memory_info() -> dict:
    """Get system memory information."""
    mem = psutil.virtual_memory()
    return {
        "total_mb": mem.total / (1024 * 1024),
        "available_mb": mem.available / (1024 * 1024),
        "used_mb": mem.used / (1024 * 1024),
        "percent": mem.percent,
    }


class MemoryMonitor:
    """Context manager for tracking memory usage during operations."""

    def __init__(self, label: str = "operation"):
        self.label = label
        self.start_memory_mb: Optional[float] = None
        self.end_memory_mb: Optional[float] = None
        self.peak_memory_mb: Optional[float] = None

    def __enter__(self) -> MemoryMonitor:
        self.start_memory_mb = get_memory_usage_mb()
        self.peak_memory_mb = self.start_memory_mb
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_memory_mb = get_memory_usage_mb()
        if self.peak_memory_mb is not None and self.end_memory_mb > self.peak_memory_mb:
            self.peak_memory_mb = self.end_memory_mb

    def get_delta_mb(self) -> float:
        """Get memory delta in MB."""
        if self.start_memory_mb is None or self.end_memory_mb is None:
            return 0.0
        return self.end_memory_mb - self.start_memory_mb

    def get_peak_mb(self) -> float:
        """Get peak memory in MB."""
        return self.peak_memory_mb or 0.0

    def get_summary(self) -> dict:
        """Get memory usage summary."""
        return {
            "label": self.label,
            "start_mb": self.start_memory_mb,
            "end_mb": self.end_memory_mb,
            "delta_mb": self.get_delta_mb(),
            "peak_mb": self.get_peak_mb(),
        }


def check_memory_threshold(threshold_mb: float) -> bool:
    """Check if current memory usage exceeds threshold."""
    return get_memory_usage_mb() > threshold_mb


def warn_if_memory_exceeds(threshold_mb: float, message: str = "") -> bool:
    """Warn if memory usage exceeds threshold. Returns True if exceeded."""
    current = get_memory_usage_mb()
    if current > threshold_mb:
        import warnings
        warnings.warn(
            f"Memory threshold exceeded: {current:.1f} MB > {threshold_mb:.1f} MB. {message}",
            UserWarning,
            stacklevel=2,
        )
        return True
    return False


def memory_guardrail(threshold_mb: float = 900, analysis_name: str = "analysis"):
    """Decorator that checks memory before and after analysis execution.
    
    Warns if memory exceeds threshold and logs peak memory usage.
    """
    def decorator(func):
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check memory before
            import warnings
            pre_mem = get_memory_usage_mb()
            if pre_mem > threshold_mb:
                warnings.warn(
                    f"{analysis_name}: Memory already high before execution: {pre_mem:.1f} MB > {threshold_mb:.1f} MB",
                    UserWarning,
                    stacklevel=2,
                )

            result = func(*args, **kwargs)

            # Check memory after
            post_mem = get_memory_usage_mb()
            peak_mem = max(pre_mem, post_mem)

            if peak_mem > threshold_mb:
                warnings.warn(
                    f"{analysis_name}: Peak memory {peak_mem:.1f} MB exceeded threshold {threshold_mb:.1f} MB",
                    UserWarning,
                    stacklevel=2,
                )

            # Log memory delta
            import logging
            logging.getLogger(__name__).info(
                f"{analysis_name}: Memory delta={post_mem - pre_mem:.1f} MB, "
                f"peak={peak_mem:.1f} MB"
            )

            return result
        return wrapper
    return decorator


def estimate_dataframe_memory(df: pd.DataFrame) -> float:
    """Estimate DataFrame memory usage in MB."""
    return df.memory_usage(deep=True).sum() / (1024 * 1024)


def check_memory_before_analysis(
    df: pd.DataFrame,
    threshold_mb: float = 900,
    analysis_name: str = "analysis",
) -> tuple[bool, str]:
    """Check if there's enough memory for analysis.
    
    Returns (can_proceed, message).
    """
    import warnings
    current_mem = get_memory_usage_mb()
    df_mem = estimate_dataframe_memory(df)
    estimated_peak = current_mem + df_mem * 3  # Conservative estimate

    if estimated_peak > threshold_mb:
        msg = (
            f"{analysis_name}: Estimated peak memory {estimated_peak:.1f} MB "
            f"exceeds threshold {threshold_mb:.1f} MB "
            f"(current={current_mem:.1f} MB, df={df_mem:.1f} MB)"
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        return False, msg
    return True, f"Memory check passed: estimated peak {estimated_peak:.1f} MB"


class ColdStartHandler:
    """Handler for Streamlit Cloud cold-start scenarios.
    
    Since Streamlit Cloud has ephemeral filesystem, cached data must be
    re-uploaded on each session. This handler manages the cold-start flow.
    """

    def __init__(self, session_state: Any, dataset_id: str | None = None):
        self.session_state = session_state
        self.dataset_id = dataset_id

    def check_cold_start(self) -> bool:
        """Check if this is a cold start (no cached data in session).
        
        Returns True if cold start (requires re-upload).
        """
        # Check if we have a valid dataset_id in session state
        if "dataset_id" not in self.session_state:
            return True
        if self.session_state["dataset_id"] != self.dataset_id:
            return True
        return False

    def handle_cold_start(self) -> None:
        """Handle cold start by clearing cached results and prompting re-upload."""
        # Clear any cached data from previous session
        from src.orchestration.result_store import invalidate_all
        invalidate_all()

        # Clear session state except for config
        keys_to_keep = {"config", "theme", "sidebar_state"}
        for key in list(self.session_state.keys()):
            if key not in keys_to_keep:
                del self.session_state[key]

    def get_cold_start_message(self) -> str:
        """Get user-friendly cold start message."""
        return (
            "Welcome back! Due to Streamlit Cloud's ephemeral filesystem, "
            "your previous session data was not persisted. "
            "Please re-upload your CSV file to continue."
        )
