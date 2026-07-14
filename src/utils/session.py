"""Session state management."""

from typing import Any

import streamlit as st


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "transactions_df": None,
        "rules_df": None,
        "product_lookup": None,
        "basket_matrix": None,
        "frequent_itemsets": None,
        "analysis_results": {},
        "last_params": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_state(key: str, default: Any = None) -> Any:
    """Get value from session state."""
    return st.session_state.get(key, default)


def set_state(key: str, value: Any):
    """Set value in session state."""
    st.session_state[key] = value


def clear_analysis_state():
    """Clear analysis-related session state."""
    keys_to_clear = [
        "rules_df",
        "frequent_itemsets",
        "analysis_results",
        "basket_matrix",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            st.session_state[key] = None
