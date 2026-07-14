"""Persistent tab utilities for Streamlit.

Provides tab components that maintain their selected state across reruns.
"""

from typing import List

import streamlit as st


def persistent_tabs(
    labels: List[str],
    key: str,
    default_tab: int = 0,
) -> int:
    """
    Create persistent tabs that remember selection across reruns.

    Uses st.session_state to track the active tab.

    Args:
        labels: List of tab labels
        key: Unique key for this tab group
        default_tab: Default tab index (0-based)

    Returns:
        Index of currently selected tab
    """
    # Initialize session state for this tab group
    state_key = f"_persistent_tab_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = default_tab

    # Track which tab is active using a radio button (hidden)
    # The radio button maintains state across reruns
    selected = st.radio(
        "",
        options=range(len(labels)),
        index=st.session_state[state_key],
        format_func=lambda i: labels[i],
        key=f"{key}_radio",
        horizontal=True,
        label_visibility="collapsed",
    )

    # Update session state
    st.session_state[state_key] = selected

    # Return the selected index
    return selected


def persistent_tabs_container(
    labels: List[str],
    key: str,
    default_tab: int = 0,
) -> tuple[int, List[st.container]]:
    """
    Create persistent tabs that return containers for each tab.

    Only the selected tab's container will have content rendered into it.
    This mimics the `with tab:` pattern but with persistent state.

    Args:
        labels: List of tab labels
        key: Unique key for this tab group
        default_tab: Default tab index (0-based)

    Returns:
        Tuple of (selected_index, List of containers - one per tab)
    """
    selected = persistent_tabs(labels, key, default_tab)

    # Create containers for all tabs
    containers = [st.container() for _ in labels]

    return selected, containers


def tabbed_view(
    labels: List[str],
    key: str,
    default_tab: int = 0,
) -> int:
    """
    Simple tabbed view that returns selected index.

    Alias for persistent_tabs with simpler name.
    """
    return persistent_tabs(labels, key, default_tab)
