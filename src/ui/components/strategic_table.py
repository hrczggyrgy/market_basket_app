"""Strategic Table Component.

A reusable, feature-rich table for strategic decision-making with:
- Sorting, filtering, search
- Conditional formatting
- Evidence badges (Observed/Estimated/Causal)
- Reliability badges (HIGH/MEDIUM/LOW/INSUFFICIENT)
- Action labels
- CSV/Excel export
- Column metadata support
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import pandas as pd
import streamlit as st


class EvidenceClass(Enum):
    """Evidence class for analytical outputs."""

    OBSERVED = "observed"
    ESTIMATED = "estimated"
    CAUSAL = "causal"


class ReliabilityLevel(Enum):
    """Reliability tier for analytical outputs."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


@dataclass
class ColumnConfig:
    """Configuration for a table column."""

    name: str
    label: str
    format: Optional[str] = None  # e.g., ",.0f", ".1%", ".2f"
    conditional_format: Optional[Callable[[Any], str]] = None  # returns CSS color
    evidence_class: Optional[EvidenceClass] = None
    reliability: Optional[ReliabilityLevel] = None
    is_action: bool = False
    sortable: bool = True
    filterable: bool = True
    help: Optional[str] = None


@dataclass
class TableConfig:
    """Configuration for the strategic table."""

    columns: list[ColumnConfig]
    key: str
    height: Optional[int] = None
    page_size: int = 50
    enable_search: bool = True
    enable_filters: bool = True
    enable_export: bool = True
    default_sort: Optional[tuple[str, bool]] = None  # (column, ascending)
    row_selection: bool = False


def _format_value(value: Any, fmt: Optional[str]) -> str:
    """Format a value according to format string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if fmt:
        try:
            return format(value, fmt)
        except (ValueError, TypeError):
            pass
    return str(value)


def _get_evidence_badge(evidence: EvidenceClass) -> str:
    """Get HTML for evidence badge."""
    colors = {
        EvidenceClass.OBSERVED: "#59A14F",  # Green
        EvidenceClass.ESTIMATED: "#F28E2B",  # Orange
        EvidenceClass.CAUSAL: "#E15759",  # Red
    }
    color = colors.get(evidence, "#7F7F7F")
    return f'<span style="background-color: {color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold;">{evidence.value.upper()}</span>'


def _get_reliability_badge(reliability: ReliabilityLevel) -> str:
    """Get HTML for reliability badge."""
    colors = {
        ReliabilityLevel.HIGH: "#59A14F",
        ReliabilityLevel.MEDIUM: "#F28E2B",
        ReliabilityLevel.LOW: "#E15759",
        ReliabilityLevel.INSUFFICIENT: "#7F7F7F",
    }
    color = colors.get(reliability, "#7F7F7F")
    return f'<span style="background-color: {color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold;">{reliability.value.upper()}</span>'


def _get_confidence_rgba(confidence: Any) -> str:
    """Get rgba color string for confidence opacity.

    Maps confidence level to an rgba color where the alpha channel
    represents opacity (0.3-1.0). Used for bubble opacity in matrices
    and conditional formatting in the strategic table.

    Args:
        confidence: Confidence value (e.g., "high", "medium", "low",
                    "h", "m", "l", or numeric 0-10)

    Returns:
        rgba color string like "rgba(0, 0, 0, 0.6)"
    """
    if confidence is None:
        return "rgba(0, 0, 0, 0.5)"
    conf_str = str(confidence).lower().strip()
    if conf_str in ("high", "h"):
        return "rgba(0, 0, 0, 1.0)"
    elif conf_str in ("medium", "m"):
        return "rgba(0, 0, 0, 0.6)"
    elif conf_str in ("low", "l"):
        return "rgba(0, 0, 0, 0.3)"
    else:
        try:
            val = float(confidence)
            if val >= 8:
                return "rgba(0, 0, 0, 1.0)"
            elif val >= 5:
                return "rgba(0, 0, 0, 0.6)"
            else:
                return "rgba(0, 0, 0, 0.3)"
        except (ValueError, TypeError):
            return "rgba(0, 0, 0, 0.5)"


def _get_action_badge(action: str) -> str:
    """Get HTML for action badge."""
    colors = {
        "invest": "#59A14F",
        "protect": "#4E79A7",
        "develop": "#F28E2B",
        "rationalize": "#E15759",
        "review": "#BAB0AC",
        "keep": "#59A14F",
        "delist_candidate": "#E15759",
        "hedge_volatility": "#F28E2B",
        "manage_demand": "#4E79A7",
        "selective_grow": "#8CD17D",
        "price_lever": "#B07AA1",
    }
    color = colors.get(action.lower(), "#7F7F7F")
    return f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 500;">{action.replace("_", " ").title()}</span>'


def render_strategic_table(
    df: pd.DataFrame,
    config: TableConfig,
) -> Optional[pd.DataFrame]:
    """Render a strategic table with all features.

    Args:
        df: DataFrame to display
        config: TableConfig with column configurations and options

    Returns:
        Selected rows DataFrame if row_selection=True, else None
    """
    if df.empty:
        st.info("No data to display")
        return None

    # Create column mapping

    # Build display DataFrame
    display_cols = [c.name for c in config.columns if c.name in df.columns]
    display_df = df[display_cols].copy()

    # Apply default sort
    if config.default_sort:
        sort_col, ascending = config.default_sort
        if sort_col in display_df.columns:
            display_df = display_df.sort_values(sort_col, ascending=ascending).reset_index(
                drop=True
            )

    # Search
    search_query = ""
    if config.enable_search:
        search_query = st.text_input(
            "🔍 Search",
            key=f"{config.key}_search",
            placeholder="Search across all columns...",
            label_visibility="collapsed",
        )
        if search_query:
            mask = display_df.astype(str).apply(
                lambda row: row.str.contains(search_query, case=False, na=False).any(),
                axis=1,
            )
            display_df = display_df[mask].reset_index(drop=True)

    # Filters
    if config.enable_filters:
        filter_cols = st.columns(
            min(
                4, len([c for c in config.columns if c.filterable and c.name in display_df.columns])
            )
        )
        filter_idx = 0
        for col_config in config.columns:
            if not col_config.filterable or col_config.name not in display_df.columns:
                continue
            if filter_idx >= len(filter_cols):
                break
            with filter_cols[filter_idx]:
                unique_vals = display_df[col_config.name].dropna().unique()
                if len(unique_vals) <= 20 and len(unique_vals) > 1:
                    selected = st.multiselect(
                        col_config.label,
                        options=sorted(unique_vals, key=str),
                        default=list(sorted(unique_vals, key=str)),
                        key=f"{config.key}_filter_{col_config.name}",
                    )
                    if selected:
                        display_df = display_df[
                            display_df[col_config.name].isin(selected)
                        ].reset_index(drop=True)
                filter_idx += 1

    # Pagination
    total_rows = len(display_df)
    total_pages = max(1, (total_rows + config.page_size - 1) // config.page_size)

    if total_pages > 1:
        page = (
            st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                key=f"{config.key}_page",
            )
            - 1
        )
    else:
        page = 0

    start_idx = page * config.page_size
    end_idx = min(start_idx + config.page_size, total_rows)
    page_df = display_df.iloc[start_idx:end_idx].copy()

    # Build HTML table for rich formatting
    html_parts = ['<table style="width: 100%; border-collapse: collapse; font-size: 13px;">']

    # Header
    html_parts.append("<thead><tr>")
    for col_config in config.columns:
        if col_config.name not in page_df.columns:
            continue
        help_text = f' title="{col_config.help}"' if col_config.help else ""
        html_parts.append(
            f'<th style="text-align: left; padding: 8px 12px; border-bottom: 2px solid #ddd; background: #f8f9fa; font-weight: 600;"{help_text}>{col_config.label}</th>'
        )
    html_parts.append("</tr></thead>")

    # Body
    html_parts.append("<tbody>")
    for _, row in page_df.iterrows():
        html_parts.append("<tr>")
        for col_config in config.columns:
            if col_config.name not in row.index:
                continue
            value = row[col_config.name]
            cell_html = ""

            # Format value
            formatted = _format_value(value, col_config.format)

            # Apply conditional formatting
            style = "padding: 8px 12px; border-bottom: 1px solid #eee;"
            if col_config.conditional_format:
                try:
                    color = col_config.conditional_format(value)
                    if color:
                        style += f" color: {color}; font-weight: 600;"
                except Exception:
                    pass

            # Evidence badge
            if col_config.evidence_class:
                badge = _get_evidence_badge(col_config.evidence_class)
                cell_html = f"{badge} {formatted}"
            # Reliability badge
            elif col_config.reliability:
                badge = _get_reliability_badge(col_config.reliability)
                cell_html = f"{badge} {formatted}"
            # Action badge
            elif col_config.is_action:
                badge = _get_action_badge(str(value))
                cell_html = badge
            else:
                cell_html = formatted

            html_parts.append(f'<td style="{style}">{cell_html}</td>')
        html_parts.append("</tr>")
    html_parts.append("</tbody></table>")

    st.markdown("".join(html_parts), unsafe_allow_html=True)

    # Pagination info
    if total_pages > 1:
        st.caption(
            f"Showing {start_idx + 1}–{end_idx} of {total_rows} rows (page {page + 1} of {total_pages})"
        )

    # Export
    if config.enable_export:
        export_cols = st.columns([1, 1, 4])
        with export_cols[0]:
            csv = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 CSV",
                csv,
                f"{config.key}_export.csv",
                "text/csv",
                key=f"{config.key}_csv",
                use_container_width=True,
            )
        with export_cols[1]:
            try:
                import io

                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    display_df.to_excel(writer, index=False, sheet_name="Data")
                st.download_button(
                    "📊 Excel",
                    buffer.getvalue(),
                    f"{config.key}_export.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"{config.key}_xlsx",
                    use_container_width=True,
                )
            except ImportError:
                st.caption("Excel export requires openpyxl")

    # Row selection
    if config.row_selection:
        selected_indices = st.multiselect(
            "Select rows",
            options=range(len(page_df)),
            format_func=lambda i: (
                f"{page_df.iloc[i].get('stockcode', page_df.iloc[i].get('category', f'Row {i}'))}"
            ),
            key=f"{config.key}_selection",
        )
        if selected_indices:
            return page_df.iloc[selected_indices].copy()

    return None


def create_strategic_table_config(
    columns: list[dict[str, Any]],
    key: str,
    **kwargs: Any,
) -> TableConfig:
    """Create TableConfig from simplified column definitions.

    Available column configuration keys:
        - name: Column name (must match DataFrame column)
        - label: Display label (defaults to name)
        - format: Format string for values (e.g., ",.0f", ".1%", ".2f")
        - conditional_format: Callable[[Any], str] returning CSS color.
          For confidence column, use _get_confidence_rgba() to map confidence
          to text opacity (rgba(r,g,b,a) where a=1.0/0.6/0.3 for high/med/low).
        - evidence_class: EvidenceClass enum (OBSERVED/ESTIMATED/CAUSAL).
          When set, displays an evidence badge with color coding.
        - reliability: ReliabilityLevel enum (HIGH/MEDIUM/LOW/INSUFFICIENT).
          When set, displays a reliability badge with color coding.
        - action: bool or str. When True or a valid action key,
          displays an action badge with color coding.
        - sortable: Whether the column is sortable (default True).
        - filterable: Whether the column is filterable (default True).
        - help: Help text displayed on hover.

    Args:
        columns: List of dicts with keys: name, label, format, evidence, reliability, action, etc.
        key: Unique key for the table
        **kwargs: Additional TableConfig options

    Returns:
        TableConfig object
    """
    col_configs = []
    for col in columns:
        evidence = col.get("evidence")
        reliability = col.get("reliability")
        col_configs.append(
            ColumnConfig(
                name=col["name"],
                label=col.get("label", col["name"]),
                format=col.get("format"),
                conditional_format=col.get("conditional_format"),
                evidence_class=EvidenceClass(evidence) if evidence else None,
                reliability=ReliabilityLevel(reliability) if reliability else None,
                is_action=col.get("action", False),
                sortable=col.get("sortable", True),
                filterable=col.get("filterable", True),
                help=col.get("help"),
            )
        )
    return TableConfig(columns=col_configs, key=key, **kwargs)
