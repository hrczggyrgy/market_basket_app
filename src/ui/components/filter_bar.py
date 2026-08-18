"""Filter Bar Component.

A reusable filter bar with multi-select, search, and top/bottom toggle
for strategic tables and matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import streamlit as st


@dataclass
class FilterConfig:
    """Configuration for a single filter."""

    key: str
    label: str
    options: list[Any]
    default: Optional[list[Any]] = None
    multiselect: bool = True
    help: Optional[str] = None


@dataclass
class FilterBarConfig:
    """Configuration for the filter bar."""

    filters: list[FilterConfig]
    key: str
    enable_search: bool = True
    enable_top_bottom: bool = True
    search_placeholder: str = "Search..."
    collapsed: bool = False


def render_filter_bar(
    df: pd.DataFrame,
    config: FilterBarConfig,
) -> pd.DataFrame:
    """Render filter bar and return filtered DataFrame.

    Args:
        df: Input DataFrame
        config: FilterBarConfig with filter definitions

    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df

    filtered = df.copy()

    # Search across all columns
    search_query = ""
    if config.enable_search:
        search_query = st.text_input(
            "🔍 " + config.search_placeholder,
            key=f"{config.key}_global_search",
            label_visibility="collapsed",
        )

    # Individual filters
    if config.filters:
        # Use columns for layout
        n_cols = min(len(config.filters), 4)
        cols = st.columns(n_cols)
        for i, filter_cfg in enumerate(config.filters):
            with cols[i % n_cols]:
                col_data = (
                    filtered[filter_cfg.key]
                    if filter_cfg.key in filtered.columns
                    else pd.Series(dtype=object)
                )
                if col_data.empty:
                    continue

                unique_vals = col_data.dropna().unique()
                if len(unique_vals) == 0:
                    continue

                # Limit options for performance
                display_options = sorted(unique_vals, key=str)[:100]

                default = filter_cfg.default
                if default is None:
                    default = list(display_options)

                selected = st.multiselect(
                    filter_cfg.label,
                    options=display_options,
                    default=default,
                    key=f"{config.key}_filter_{filter_cfg.key}",
                    help=filter_cfg.help,
                )
                if selected:
                    filtered = filtered[filtered[filter_cfg.key].isin(selected)]

    # Global search
    if search_query:
        mask = filtered.astype(str).apply(
            lambda row: row.str.contains(search_query, case=False, na=False).any(),
            axis=1,
        )
        filtered = filtered[mask]

    # Top/Bottom toggle
    if config.enable_top_bottom and len(filtered) > 10:
        toggle_cols = st.columns([1, 1, 4])
        with toggle_cols[0]:
            sort_col = st.selectbox(
                "Sort by",
                options=[
                    c
                    for c in filtered.columns
                    if filtered[c].dtype in ("int64", "float64", "int32", "float32")
                ],
                key=f"{config.key}_sort_col",
            )
        with toggle_cols[1]:
            sort_order = st.radio(
                "Order",
                options=["Top", "Bottom"],
                horizontal=True,
                key=f"{config.key}_sort_order",
            )
        if sort_col:
            ascending = sort_order == "Bottom"
            filtered = filtered.sort_values(sort_col, ascending=ascending)

    return filtered


def create_filter_configs_from_df(
    df: pd.DataFrame,
    filter_columns: list[str],
    key_prefix: str = "",
) -> list[FilterConfig]:
    """Create FilterConfig list from DataFrame columns.

    Args:
        df: DataFrame to analyze
        filter_columns: List of column names to create filters for
        key_prefix: Prefix for filter keys

    Returns:
        List of FilterConfig objects
    """
    configs = []
    for col in filter_columns:
        if col not in df.columns:
            continue
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 1 or len(unique_vals) > 100:
            continue
        configs.append(
            FilterConfig(
                key=col,
                label=col.replace("_", " ").title(),
                options=sorted(unique_vals, key=str),
            )
        )
    return configs
