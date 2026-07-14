"""Main Streamlit application."""

import traceback

import pandas as pd
import streamlit as st

from src.algorithms.fpgrowth import (
    create_basket_matrix,
    get_product_lookup,
    run_fpgrowth,
)
from src.data.generator import generate_transactions

# Import all modules
from src.data.loader import get_data_summary, load_transactions
from src.rules.generator import filter_rules, generate_rules
from src.ui.addon_tab import render_addon_tab
from src.ui.cdt_tab import render_cdt_tab
from src.ui.cohort_tab import render_cohort_tab
from src.ui.copurchase_tab import render_copurchase_tab
from src.ui.export import render_export_buttons
from src.ui.product_performance_tab import render_product_performance_tab
from src.ui.promotional_tab import render_promotional_tab
from src.ui.rules_tab import render_rules_tab
from src.ui.segmentation_tab import render_segmentation_tab
from src.ui.sidebar import render_sidebar
from src.ui.switching_tab import render_switching_tab
from src.ui.tree_tab import render_tree_tab

# Page config
st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .tab-content {
        padding: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    # Header
    st.markdown(
        '<div class="main-header">🛒 Market Basket Analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Association Rules • Co-purchase • Add-ons • Switching • Customer Choice Modelling • Customer Segmentation • Product Performance • Cohort Analysis • Promotional Analytics</div>',
        unsafe_allow_html=True,
    )

    # Render sidebar and get config
    config = render_sidebar()

    # Load data
    transactions_df = None
    product_lookup = {}

    try:
        if config["use_sample"]:
            with st.spinner("Generating sample data..."):
                transactions_df = generate_transactions(
                    n_transactions=1000, n_customers=200, n_products=100
                )
        elif config["uploaded_file"] is not None:
            with st.spinner("Loading transaction data..."):
                transactions_df = load_transactions(config["uploaded_file"], config["column_mapping"])

        if transactions_df is not None and not transactions_df.empty:
            # Create product lookup
            product_lookup = get_product_lookup(transactions_df)

            # Show data summary
            with st.expander("📊 Data Summary", expanded=False):
                summary = get_data_summary(transactions_df)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Transactions", f"{summary['n_transactions']:,}")
                with col2:
                    st.metric("Customers", f"{summary['n_customers']:,}")
                with col3:
                    st.metric("Products", f"{summary['n_products']:,}")
                with col4:
                    st.metric("Revenue", f"${summary['total_revenue']:,.2f}")

            # Run analysis when button clicked
            if config["run_analysis"]:
                run_analysis(transactions_df, product_lookup, config)
            else:
                st.info(
                    "👈 Configure parameters in sidebar and click **Run Analysis** to start"
                )

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.code(traceback.format_exc())


def run_analysis(transactions_df: pd.DataFrame, product_lookup: dict, config: dict):
    """Run the selected analysis."""

    analysis_mode = config["analysis_mode"]
    params = config["analysis_params"]

    # Common FP-Growth parameters
    fp_params = {
        "min_support": config["min_support"],
        "min_confidence": config["min_confidence"],
        "max_itemset_len": config["max_itemset_len"],
        "min_lift": config["min_lift"],
    }

    # Merge params
    all_params = {**fp_params, **params}

    try:
        if analysis_mode == "Association Rules":
            render_rules_analysis(transactions_df, product_lookup, all_params)
        elif analysis_mode == "Co-purchase":
            render_copurchase_tab(transactions_df, product_lookup, all_params)
        elif analysis_mode == "Add-on":
            render_addon_tab(transactions_df, product_lookup, all_params)
        elif analysis_mode == "Switching":
            render_switching_tab(transactions_df, product_lookup, all_params)
        elif analysis_mode == "Choice Prediction Model":
            render_tree_tab(transactions_df, product_lookup, all_params)
        elif analysis_mode == "Decision Tree & Patterns":
            render_cdt_tab(transactions_df, product_lookup, all_params)
        elif analysis_mode == "Customer Segmentation":
            render_segmentation_tab(transactions_df, product_lookup, all_params)
        elif analysis_mode == "Product Performance":
            render_product_performance_tab(transactions_df, product_lookup, all_params)
        elif analysis_mode == "Cohort Analysis":
            render_cohort_tab(transactions_df, product_lookup, all_params)
        elif analysis_mode == "Promotional Analytics":
            render_promotional_tab(transactions_df, product_lookup, all_params)
        else:
            st.warning(f"Unknown analysis mode: {analysis_mode}")

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.code(traceback.format_exc())


def render_rules_analysis(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict
):
    """Render association rules analysis (default mode with all visualizations)."""

    # Create basket matrix
    with st.spinner("Creating basket matrix..."):
        basket = create_basket_matrix(transactions_df)

    # Run FP-Growth
    with st.spinner(f"Running FP-Growth (min_support={params['min_support']:.3f})..."):
        freq_items = run_fpgrowth(
            basket, min_support=params["min_support"], max_len=params["max_itemset_len"]
        )

    if freq_items.empty:
        st.warning("No frequent itemsets found. Try lowering min_support.")
        return

    st.success(f"Found {len(freq_items)} frequent itemsets")

    # Generate rules
    with st.spinner("Generating association rules..."):
        rules = generate_rules(
            freq_items, metric="confidence", min_threshold=params["min_confidence"]
        )

    if rules.empty:
        st.warning("No association rules found. Try lowering min_confidence.")
        return

    st.success(f"Generated {len(rules)} association rules")

    # Filter rules
    filtered_rules = filter_rules(
        rules,
        min_support=params["min_support"],
        min_confidence=params["min_confidence"],
        min_lift=params["min_lift"],
        max_lift=params.get("max_lift", 100.0),
    )

    st.info(
        f"{len(filtered_rules)} rules after filtering (lift ≥ {params['min_lift']})"
    )

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Rules Table", "🕸️ Network Graph", "🔥 Heatmap", "📊 Scatter Plot"]
    )

    with tab1:
        render_rules_tab(filtered_rules, product_lookup, params)
        render_export_buttons(filtered_rules, product_lookup, prefix="rules")

    with tab2:
        st.subheader("Association Rules Network")
        from src.viz.network import create_network_graph

        net_fig = create_network_graph(
            filtered_rules,
            product_lookup=product_lookup,
            min_lift=params["min_lift"],
            max_nodes=50,
            max_edges=100,
        )
        st.plotly_chart(net_fig, use_container_width=True)

    with tab3:
        st.subheader("Rules Heatmap")
        from src.viz.heatmap import create_heatmap

        heat_fig = create_heatmap(
            filtered_rules,
            x_metric="support",
            y_metric="confidence",
            color_metric="lift",
        )
        st.plotly_chart(heat_fig, use_container_width=True)

    with tab4:
        st.subheader("Rules Scatter Plot")
        from src.viz.heatmap import create_scatter_heatmap

        scatter_fig = create_scatter_heatmap(
            filtered_rules,
            x_metric="support",
            y_metric="confidence",
            color_metric="lift",
            size_metric="lift",
        )
        st.plotly_chart(scatter_fig, use_container_width=True)


if __name__ == "__main__":
    main()
