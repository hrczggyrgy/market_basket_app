"""Streamlit app entry point."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

# Ensure project root is in Python path for src package imports (via src.__init__)
import src  # noqa: F401
from src.analytics.data import load_transactions
from src.analytics.simulation import SCENARIOS, config_for, generate_sample_transactions
from src.ui import registry
from src.ui.registry import ModeSpec
from src.ui.tabs import (
    assortment_page,
    category_page,
    cdt_page,
    clv_page,
    cohorts,
    copurchase,
    decision_center,
    overview,
    performance,
    pricing_page,
    promo_page,
    rules,
    segmentation,
    switching,
)

# Page config
st.set_page_config(
    page_title="Market Basket Intelligence",
    page_icon=":material/shopping_cart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Register all modes
def _register_modes() -> None:
    modes: tuple[ModeSpec, ...] = (
        decision_center.MODE_SPEC,
        overview.MODE_SPEC,
        rules.MODE_SPEC,
        copurchase.MODE_SPEC,
        switching.MODE_SPEC,
        cohorts.MODE_SPEC,
        performance.MODE_SPEC,
        category_page.MODE_SPEC,
        cdt_page.MODE_SPEC,
        segmentation.MODE_SPEC,
        pricing_page.MODE_SPEC,
        promo_page.MODE_SPEC,
        assortment_page.MODE_SPEC,
        clv_page.MODE_SPEC,
    )
    for mode in modes:
        registry.register_mode(mode)


@st.cache_data(show_spinner="Loading transactions...")
def _load_data(path: str, _version: int = 2) -> tuple:
    df, warning, dropped, quality_report = load_transactions(path)
    return df, warning, dropped, quality_report


@st.cache_data(show_spinner="Simulating sample data...")
def _generate_sample_csv(scenario_key: str, _version: int = 1) -> bytes:
    config = config_for(scenario_key)
    df = generate_sample_transactions(config)
    return df.to_csv(index=False).encode()


def _load_sample_scenario(scenario_key: str) -> tuple:
    return load_transactions(io.BytesIO(_generate_sample_csv(scenario_key)))


def main() -> None:
    _register_modes()

    # Sidebar: data source
    st.sidebar.header(":material/database: Data Source")

    # Data source selection
    data_source_option = st.sidebar.radio(
        "Select data source",
        options=["Upload CSV", "Use sample data"],
        index=0,
        help="Upload your own transaction CSV or use the built-in sample data to explore the app",
    )

    uploaded = None
    if data_source_option == "Upload CSV":
        uploaded = st.sidebar.file_uploader(
            "Upload CSV", type=["csv"], help="Upload your transaction CSV file"
        )
        if uploaded is not None:
            # Security validation
            if uploaded.size > 100 * 1024 * 1024:  # 100MB limit
                st.error("File size exceeds 100MB limit. Please upload a smaller file.")
                st.stop()

            # Validate filename
            filename = uploaded.name.lower()
            if not filename.endswith(".csv"):
                st.error("Only CSV files are allowed.")
                st.stop()

            # Check for potentially malicious patterns in filename
            if any(char in filename for char in ["..", "/", "\\", "\0"]):
                st.error("Invalid filename. Please use a simple filename with .csv extension.")
                st.stop()

            data_source: str | io.BytesIO = io.BytesIO(uploaded.getvalue())
            source_label = uploaded.name
        else:
            data_source = "sample_data/sample_transactions.csv"
            source_label = "Sample data"
    else:
        scenario_labels = {k: v.label for k, v in SCENARIOS.items()}
        scenario_key = st.sidebar.selectbox(
            "Sample Scenario",
            options=list(SCENARIOS.keys()),
            format_func=lambda k: scenario_labels[k],
            help="Pick a simulated retail regime to explore the app.",
        )
        scenario = SCENARIOS[scenario_key]
        st.sidebar.caption(scenario.description)
        data_source = scenario_key
        source_label = f"Sample ({scenario.label})"
        uploaded = None

    # Load data
    with st.spinner("Loading data..."):
        try:
            if isinstance(data_source, str) and data_source in SCENARIOS:
                df, warning, dropped, quality_report = _load_sample_scenario(data_source)
            else:
                df, warning, dropped, quality_report = _load_data(data_source, _version=2)
        except ValueError as e:
            st.error(f"Data validation error: {e}")
            st.info(
                "Please check your CSV file format and ensure all required columns are present."
            )
            st.info(
                "Required columns: date, transaction_id, stockcode, product, customer_id, price, quantity"
            )
            st.stop()
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty or contains no data.")
            st.info("Please upload a valid CSV file with transaction data.")
            st.stop()
        except pd.errors.ParserError:
            st.error("Failed to parse the CSV file. Please check the file format.")
            st.info("Ensure your CSV is properly formatted with consistent delimiters and quoting.")
            st.stop()
        except MemoryError:
            st.error("The file is too large to process in memory.")
            st.info("Try uploading a smaller file or using the sample data.")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error loading data: {e}")
            st.info("Please try again or contact support if the issue persists.")
            st.stop()

    # Store quality report in session state for overview tab
    if quality_report:
        st.session_state["quality_report"] = quality_report

    if warning:
        st.sidebar.warning(warning)
    if dropped:
        st.sidebar.info(f"Dropped {dropped} rows during cleaning")

    # Enhanced data quality visualization
    if quality_report and quality_report.has_issues():
        with st.sidebar.expander("Data Quality Issues", expanded=False):
            if quality_report.low_freq_products:
                st.warning(
                    f"**Low-frequency products:** {len(quality_report.low_freq_products)} products with insufficient transactions"
                )
                if st.checkbox("Show low-frequency products"):
                    st.write(
                        pd.DataFrame(
                            list(quality_report.low_freq_counts.items()),
                            columns=["Product", "Transaction Count"],
                        )
                    )

            if quality_report.basket_outlier_txn_ids:
                st.warning(
                    f"**Basket outliers:** {len(quality_report.basket_outlier_txn_ids)} unusually large baskets"
                )

            if quality_report.duplicate_count > 0:
                st.warning(
                    f"**Duplicate transactions:** {quality_report.duplicate_count} potential duplicates detected"
                )

            if quality_report.incomplete_rows > 0:
                st.warning(
                    f"**Incomplete rows:** {quality_report.incomplete_rows} rows with missing required fields"
                )

            if quality_report.volume_warning:
                st.error(quality_report.volume_warning)

    st.sidebar.caption(
        f"{len(df):,} rows \u2022 {df['customer_id'].nunique()} customers \u2022 {df['stockcode'].nunique()} products"
    )
    st.sidebar.caption(f"Source: {source_label}")

    # Sidebar: mode selection
    selected = registry.render_sidebar(df)

    # Main area
    st.title(":material/shopping_cart: Market Basket Intelligence")
    st.caption("Customer Decision Intelligence for Retail")

    # Dispatch
    registry.dispatch(selected, df)


if __name__ == "__main__":
    main()
