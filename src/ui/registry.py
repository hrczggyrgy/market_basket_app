"""UI registry: single source of truth for sidebar modes and their handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any

import pandas as pd


@dataclass(frozen=True)
class ModeSpec:
    """Sidebar mode specification."""
    key: str              # unique internal key
    label: str            # display label in sidebar
    icon: str             # Material Symbols shortcode (e.g., ":material/analytics:")
    handler: Callable[[pd.DataFrame], None]  # render function
    requires: tuple[str, ...] = ()  # optional: data capability flags required


# Global registry
_MODES: Dict[str, ModeSpec] = {}


def register_mode(spec: ModeSpec) -> None:
    """Register a sidebar mode (idempotent - safe to call multiple times)."""
    if spec.key in _MODES:
        # Mode already registered; skip to allow re-runs
        return
    _MODES[spec.key] = spec


def get_modes() -> Dict[str, ModeSpec]:
    """Return all registered modes (ordered by registration)."""
    return dict(_MODES)


def get_mode(key: str) -> ModeSpec:
    """Get a specific mode by key."""
    return _MODES[key]


def render_sidebar(df: pd.DataFrame) -> str:
    """Render sidebar and return selected mode key."""
    import streamlit as st

    modes = get_modes()
    options = [spec.key for spec in modes.values()]
    labels = {spec.key: f"{spec.icon} {spec.label}" for spec in modes.values()}

    available = options

    selected = st.sidebar.radio(
        "Analysis Mode",
        options=available,
        format_func=lambda k: labels[k],
        index=0,
    )
    return selected


def dispatch(mode_key: str, df: pd.DataFrame) -> None:
    """Dispatch to the selected mode's handler."""
    import streamlit as st

    mode = get_mode(mode_key)
    try:
        mode.handler(df)
    except Exception as e:
        st.error(f"Error in {mode.label}: {e}")
        if st.checkbox("Show traceback", key=f"traceback_{mode_key}"):
            import traceback
            st.code(traceback.format_exc())