"""Export UI component."""

import pandas as pd
import streamlit as st


def render_export_buttons(
    rules_df: pd.DataFrame, product_lookup: dict = None, prefix: str = "export"
):
    """Render download buttons for rules."""
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

    # CSV export
    csv = export_df.to_csv(index=False)
    col1.download_button(
        label=" Download CSV",
        data=csv,
        file_name="association_rules.csv",
        mime="text/csv",
        key=f"{prefix}_csv",
        width="stretch",
    )

    # JSON export
    json_str = export_df.to_json(orient="records", indent=2)
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
        import io

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Rules")
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


def render_analytics_export(df: pd.DataFrame, name: str, prefix: str = "analytics"):
    """Generic export for analytics dataframes."""
    if df.empty:
        return

    col1, col2 = st.columns(2)

    csv = df.to_csv(index=False)
    col1.download_button(
        f" {name} (CSV)",
        data=csv,
        file_name=f"{name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        key=f"{prefix}_{name}_csv",
        width="stretch",
    )

    json_str = df.to_json(orient="records", indent=2)
    col2.download_button(
        f" {name} (JSON)",
        data=json_str,
        file_name=f"{name.lower().replace(' ', '_')}.json",
        mime="application/json",
        key=f"{prefix}_{name}_json",
        width="stretch",
    )
