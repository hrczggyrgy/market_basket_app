"""Export UI component with metadata."""

import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st


def _build_export_metadata(
    analysis_name: str,
    date_range: tuple = None,
    filters: Dict[str, Any] = None,
    method_params: Dict[str, Any] = None,
    sample_sizes: Dict[str, int] = None,
    readiness_status: str = None,
    limitation: str = None,
) -> Dict[str, Any]:
    """Build metadata dict for export."""
    return {
        "analysis": analysis_name,
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "date_range": {
            "start": date_range[0].isoformat() if date_range and date_range[0] else None,
            "end": date_range[1].isoformat() if date_range and date_range[1] else None,
        } if date_range else None,
        "filters": filters or {},
        "method_parameters": method_params or {},
        "sample_sizes": sample_sizes or {},
        "readiness_confidence": readiness_status or "unknown",
        "limitation_caveat": limitation or "",
    }


def _inject_metadata_csv(df: pd.DataFrame, metadata: Dict[str, Any]) -> str:
    """Prepend metadata as commented lines to CSV."""
    meta_lines = ["# Metadata"]
    for key, value in metadata.items():
        if isinstance(value, dict):
            meta_lines.append(f"# {key}: {json.dumps(value)}")
        else:
            meta_lines.append(f"# {key}: {value}")
    meta_lines.append("# End Metadata")
    meta_lines.append("")  # blank line
    csv_data = df.to_csv(index=False)
    return "\n".join(meta_lines) + csv_data


def _inject_metadata_json(df: pd.DataFrame, metadata: Dict[str, Any]) -> str:
    """Wrap DataFrame and metadata in JSON envelope."""
    envelope = {
        "metadata": metadata,
        "data": df.to_dict(orient="records"),
    }
    return json.dumps(envelope, indent=2, default=str)


def export_plotly_chart(fig, filename: str, metadata: Dict[str, Any] = None):
    """Export Plotly chart as HTML with embedded metadata."""
    if metadata is None:
        metadata = _build_export_metadata(analysis_name=filename)
    
    # Embed metadata as a script tag in the HTML
    metadata_json = json.dumps(metadata, indent=2, default=str)
    
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    
    # Inject metadata as a script tag
    metadata_script = f'<script id="chart-metadata" type="application/json">{metadata_json}</script>'
    html = html.replace('</head>', f'{metadata_script}\n</head>')
    
    st.download_button(
        label=f"📥 Download {filename} (HTML with Metadata)",
        data=html,
        file_name=f"{filename}.html",
        mime="text/html",
        key=f"export_chart_{filename}",
        width="stretch",
    )


def render_export_buttons(
    rules_df: pd.DataFrame,
    product_lookup: dict = None,
    prefix: str = "export",
    metadata: Optional[Dict[str, Any]] = None,
):
    """Render download buttons for rules with metadata."""
    if rules_df.empty:
        st.info("No data to export")
        return

    col1, col2, col3 = st.columns(3)

    # Prepare export data
    export_df = rules_df.copy()

    if product_lookup:

        def format_items(items):
            return ", ".join(product_lookup.get(str(i), str(i)) for i in items)

        export_df["antecedents_str"] = export_df["antecedents"].apply(format_items)
        export_df["consequents_str"] = export_df["consequents"].apply(format_items)
        export_df["rule"] = export_df["antecedents_str"] + " → " + export_df["consequents_str"]

    # Build default metadata if not provided
    if metadata is None:
        metadata = _build_export_metadata(
            analysis_name="association_rules",
        )

    # CSV export with metadata
    csv = _inject_metadata_csv(export_df, metadata)
    col1.download_button(
        label=" Download CSV",
        data=csv,
        file_name="association_rules.csv",
        mime="text/csv",
        key=f"{prefix}_csv",
        width="stretch",
    )

    # JSON export with metadata
    json_str = _inject_metadata_json(export_df, metadata)
    col2.download_button(
        label=" Download JSON",
        data=json_str,
        file_name="association_rules.json",
        mime="application/json",
        key=f"{prefix}_json",
        width="stretch",
    )

    # Excel export (if openpyxl available)
    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Rules")
            # Add metadata sheet
            meta_df = pd.DataFrame([(k, str(v)) for k, v in metadata.items()], columns=["Key", "Value"])
            meta_df.to_excel(writer, index=False, sheet_name="Metadata")
        col3.download_button(
            label=" Download Excel",
            data=buffer.getvalue(),
            file_name="association_rules.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{prefix}_xlsx",
            width="stretch",
        )
    except ImportError:
        col3.info("Install openpyxl for Excel export")


def render_analytics_export(
    df: pd.DataFrame,
    name: str,
    prefix: str = "analytics",
    metadata: Optional[Dict[str, Any]] = None,
    date_range: tuple = None,
    filters: Dict[str, Any] = None,
    method_params: Dict[str, Any] = None,
    sample_sizes: Dict[str, int] = None,
    readiness_status: str = None,
    limitation: str = None,
):
    """Generic export for analytics dataframes with metadata."""
    if df.empty:
        return

    # Build metadata if not provided
    if metadata is None:
        metadata = _build_export_metadata(
            analysis_name=name,
            date_range=date_range,
            filters=filters,
            method_params=method_params,
            sample_sizes=sample_sizes,
            readiness_status=readiness_status,
            limitation=limitation,
        )

    col1, col2 = st.columns(2)

    csv = _inject_metadata_csv(df, metadata)
    col1.download_button(
        f" {name} (CSV)",
        data=csv,
        file_name=f"{name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        key=f"{prefix}_{name}_csv",
        width="stretch",
    )

    json_str = _inject_metadata_json(df, metadata)
    col2.download_button(
        f" {name} (JSON)",
        data=json_str,
        file_name=f"{name.lower().replace(' ', '_')}.json",
        mime="application/json",
        key=f"{prefix}_{name}_json",
        width="stretch",
    )
    
    # Also offer chart export if we have a chart (this would be used by passing a fig)
    # Note: Caller should call export_plotly_chart separately if they have a figure
