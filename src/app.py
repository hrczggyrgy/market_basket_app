"""Streamlit app entry point."""

from __future__ import annotations

import io
import os
import sys
import streamlit as st

# Ensure project root is in Python path for src package imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from src.ui import registry
from src.ui.registry import ModeSpec
from src.ui.tabs import (
    overview, rules, copurchase, switching, cohorts,
    performance, category_page, cdt_page, segmentation, pricing_page,
    promo_page, assortment_page, clv_page
)
from src.analytics.data import load_transactions
from src.analytics.config import get_config
from src.analytics.simulation import SCENARIOS, config_for, generate_sample_transactions


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
def _load_data_from_path(path: str, _version: int = 2) -> tuple:
    """Load transactions from a file path. Cached for performance."""
    df, warning, dropped, quality_report = load_transactions(path)
    return df, warning, dropped, quality_report


def _load_data_from_bytes(data: bytes) -> tuple:
    """Load transactions from bytes (for uploaded files). Not cached since uploads are unique."""
    df, warning, dropped, quality_report = load_transactions(io.BytesIO(data))
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
        help="Upload your own transaction CSV or use the built-in sample data to explore the app"
    )

    uploaded = None
    if data_source_option == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Upload your transaction CSV file")
        if uploaded is not None:
            data = uploaded.getvalue()
            source_label = uploaded.name
        else:
            data = None
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
        data = f"scenario:{scenario_key}"
        source_label = f"Sample ({scenario.label})"

    # Load data
    with st.spinner("Loading data..."):
        try:
            if uploaded is not None:
                df, warning, dropped, quality_report = _load_data_from_bytes(uploaded.getvalue())
            elif isinstance(data, str) and data.startswith("scenario:"):
                df, warning, dropped, quality_report = _load_sample_scenario(data.split(":", 1)[1])
            else:
                df, warning, dropped, quality_report = _load_data_from_path("sample_data/sample_transactions.csv", _version=2)
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()

    # Store quality report in session state for overview tab
    if quality_report:
        st.session_state["quality_report"] = quality_report

    if warning:
        st.sidebar.warning(warning)
    if dropped:
        st.sidebar.info(f"Dropped {dropped} rows during cleaning")

    st.sidebar.caption(f"{len(df):,} rows \u2022 {df['customer_id'].nunique()} customers \u2022 {df['stockcode'].nunique()} products")
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