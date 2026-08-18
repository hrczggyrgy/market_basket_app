"""Download Component.

Utilities for CSV/Excel export with metadata.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import streamlit as st


def render_download_buttons(
    df: pd.DataFrame,
    base_name: str,
    key: str,
    include_metadata: bool = True,
    metadata: Optional[dict[str, Any]] = None,
    csv_label: str = "📥 Download CSV",
    excel_label: str = "📊 Download Excel",
) -> None:
    """Render download buttons for CSV and Excel.

    Args:
        df: DataFrame to export
        base_name: Base filename (without extension)
        key: Unique key for the component
        include_metadata: Include metadata sheet in Excel
        metadata: Additional metadata to include
        csv_label: Label for CSV button
        excel_label: Label for Excel button
    """
    if df.empty:
        st.info("No data to export")
        return

    cols = st.columns([1, 1, 4])

    with cols[0]:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            csv_label,
            csv,
            f"{base_name}.csv",
            "text/csv",
            key=f"{key}_csv",
            use_container_width=True,
        )

    with cols[1]:
        try:
            import io

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Data")
                if include_metadata and metadata:
                    meta_df = pd.DataFrame(list(metadata.items()), columns=["Property", "Value"])
                    meta_df.to_excel(writer, index=False, sheet_name="Metadata")
            excel_bytes = buffer.getvalue()
            st.download_button(
                excel_label,
                excel_bytes,
                f"{base_name}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"{key}_xlsx",
                use_container_width=True,
            )
        except ImportError:
            st.caption("Excel export requires openpyxl")


def create_metadata(
    source: str,
    filters: Optional[dict[str, Any]] = None,
    reliability: Optional[dict[str, Any]] = None,
    evidence_class: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create standardized metadata dict for exports.

    Args:
        source: Source module/analysis name
        filters: Applied filters
        reliability: Reliability info
        evidence_class: Evidence class
        **kwargs: Additional metadata

    Returns:
        Metadata dictionary
    """
    import json
    from datetime import datetime

    meta = {
        "Generated": datetime.now().isoformat(),
        "Source": source,
        "Rows": kwargs.get("rows", "N/A"),
        "Columns": kwargs.get("columns", "N/A"),
    }

    if filters:
        meta["Filters"] = json.dumps(filters)
    if reliability:
        meta["Reliability"] = json.dumps(reliability)
    if evidence_class:
        meta["Evidence Class"] = evidence_class

    meta.update(kwargs)
    return meta
