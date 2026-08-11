"""Timing utilities for performance profiling.

Usage:
    import os
    os.environ["ENGAGE_PROFILE"] = "1"  # or set in shell before running streamlit

    from src.analytics.profiling import timed
    @timed("my_function")
    def my_function(...):
        ...

Then run: ENGAGE_PROFILE=1 streamlit run app.py
Timing logs will appear in the terminal.
"""

from __future__ import annotations

import functools
import os
import time
from typing import Any, Callable

try:
    import streamlit as st
    HAS_ST = True
except ImportError:
    HAS_ST = False


ENGAGE_PROFILE = os.environ.get("ENGAGE_PROFILE", "0") == "1"


def _log(msg: str) -> None:
    if HAS_ST:
        try:
            st.sidebar.caption(msg)
        except Exception:
            print(msg)
    else:
        print(msg)


def timed(name: str | None = None):
    """Decorator to time a function. Outputs to terminal and optionally Streamlit sidebar.

    Args:
        name: Optional custom label. Defaults to function's qualified name.

    Only active when ENGAGE_PROFILE=1 environment variable is set.
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        label = name or fn.__qualname__

        if not ENGAGE_PROFILE:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                _log(f"[PERF] {label}: {elapsed:.3f}s")
        return wrapper
    return decorator


def time_block(label: str):
    """Context manager for timing a code block.

    Usage:
        with time_block("my_operation"):
            do_something()
    """
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start
            _log(f"[PERF] {label}: {elapsed:.3f}s")

    if not ENGAGE_PROFILE:
        class NoOp:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return NoOp()

    return Timer()


def get_profile_flag() -> bool:
    """Check if profiling is enabled."""
    return ENGAGE_PROFILE
