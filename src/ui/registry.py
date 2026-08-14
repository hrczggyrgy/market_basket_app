"""UI registry: single source of truth for sidebar modes and their handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class ModeSpec:
    """Sidebar mode specification."""

    key: str  # unique internal key
    label: str  # display label in sidebar
    icon: str  # Material Symbols shortcode (e.g., ":material/analytics:")
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


def _check_capabilities(df: pd.DataFrame, requires: tuple[str, ...]) -> tuple[bool, list[str]]:
    """Check if dataframe meets required capabilities.
    
    Returns (all_met, missing_list)
    """
    from src.analytics.data import build_dataset_capabilities
    
    capabilities = build_dataset_capabilities(df)
    missing = [req for req in requires if not capabilities.get(req, False)]
    return len(missing) == 0, missing


def render_sidebar(df: pd.DataFrame) -> str:
    """Render sidebar and return selected mode key."""
    modes = get_modes()
    
    # Check capabilities for each mode
    mode_status: dict[str, dict[str, object]] = {}
    for key, spec in modes.items():
        all_met, missing = _check_capabilities(df, spec.requires)
        mode_status[key] = {
            "available": all_met,
            "missing": missing,
            "label": f"{spec.icon} {spec.label}",
        }
    
    # Build options - available modes first, then unavailable
    available_options = [k for k, v in mode_status.items() if bool(v["available"])]
    unavailable_options = [k for k, v in mode_status.items() if not bool(v["available"])]
    options = available_options + unavailable_options
    
    labels: dict[str, str] = {}
    for key, status in mode_status.items():
        if status["available"]:
            labels[key] = str(status["label"])
        else:
            missing_list = status["missing"]
            missing_str = ", ".join(missing_list) if isinstance(missing_list, list) else "unknown"
            labels[key] = f"{status['label']}  ⚠️ (missing: {missing_str})"
    
    # Default to first available mode, or first mode if none available
    default_index = 0
    if available_options:
        default_index = options.index(available_options[0])
    
    # First radio call to get selection (without disabled)
    selected = st.sidebar.radio(
        "Analysis Mode",
        options=options,
        format_func=lambda k: labels[k],
        index=default_index,
    )
    
    # Show capability info for selected mode if unavailable
    if selected in mode_status and not mode_status[selected]["available"]:
        missing_obj = mode_status[selected]["missing"]
        missing_list = missing_obj if isinstance(missing_obj, list) else []
        missing_str = ", ".join(missing_list) if missing_list else "unknown"
        st.sidebar.warning(
            f"**{labels[selected]} is not available**\n\n"
            f"Missing capabilities: {missing_str}\n\n"
            f"Please ensure your data includes the required columns and meets minimum volume thresholds."
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
