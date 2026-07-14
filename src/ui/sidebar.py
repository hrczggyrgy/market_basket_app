"""Sidebar UI components."""

from typing import Any, Dict

import pandas as pd
import streamlit as st


def render_sidebar() -> Dict[str, Any]:
    """Render sidebar with file upload and analysis parameters."""
    st.sidebar.header("📁 Data Upload")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Transaction CSV",
        type=["csv"],
        help="CSV with columns: date, transaction_id, stockcode, product, customer_id, price, quantity",
    )

    use_sample = st.sidebar.checkbox("Use Sample Data", value=False)

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, nrows=5)
            cols = df.columns.tolist()
            uploaded_file.seek(0)

            # Auto-detect columns
            date_col = st.sidebar.selectbox(
                "Date Column", cols, index=cols.index("date") if "date" in cols else 0
            )
            trans_col = st.sidebar.selectbox(
                "Transaction ID Column",
                cols,
                index=cols.index("transaction_id") if "transaction_id" in cols else 0,
            )
            item_col = st.sidebar.selectbox(
                "Item Column (Stock Code)",
                cols,
                index=cols.index("stockcode") if "stockcode" in cols else 0,
            )
            product_col = st.sidebar.selectbox(
                "Product Name Column",
                cols,
                index=cols.index("product") if "product" in cols else 0,
            )
            customer_col = st.sidebar.selectbox(
                "Customer ID Column",
                cols,
                index=cols.index("customer_id") if "customer_id" in cols else 0,
            )
            price_col = st.sidebar.selectbox(
                "Price Column",
                cols,
                index=cols.index("price") if "price" in cols else 0,
            )
            qty_col = st.sidebar.selectbox(
                "Quantity Column",
                cols,
                index=cols.index("quantity") if "quantity" in cols else 0,
            )

            column_mapping = {
                "date": date_col,
                "transaction_id": trans_col,
                "stockcode": item_col,
                "product": product_col,
                "customer_id": customer_col,
                "price": price_col,
                "quantity": qty_col,
            }
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            column_mapping = {}
    else:
        column_mapping = {}

    st.sidebar.divider()
    st.sidebar.header("⚙️ FP-Growth Parameters")

    min_support = st.sidebar.slider(
        "Min Support",
        0.0005,
        0.05,
        0.002,
        0.0005,
        help="Minimum support threshold (fraction of transactions)",
    )

    min_confidence = st.sidebar.slider(
        "Min Confidence",
        0.01,
        0.5,
        0.1,
        0.01,
        help="Minimum confidence for association rules (typical range: 0.05-0.3)",
    )

    max_itemset_len = st.sidebar.slider(
        "Max Itemset Length",
        2,
        6,
        3,
        help="Maximum number of items in frequent itemsets",
    )

    min_lift = st.sidebar.slider(
        "Min Lift", 0.5, 5.0, 1.2, 0.1, help="Minimum lift threshold for rules"
    )

    st.sidebar.divider()
    st.sidebar.header("📊 Analysis Options")

    # Main analysis category
    analysis_category = st.sidebar.radio(
        "Analysis Category",
        [
            "Association Rules",
            "Decision Intelligence",
            "Customer Segmentation",
            "Product Performance",
            "Cohort Analysis",
            "Promotional Analytics",
        ],
        index=0,
    )

    # Sub-modes within each category
    if analysis_category == "Association Rules":
        analysis_mode = st.sidebar.radio(
            "Association Rules Mode",
            [
                "Association Rules",  # default with all visualizations
                "Co-purchase",  # affinity analysis
                "Add-on",  # complementary products
                "Switching",  # product switching
            ],
            index=0,
        )
    elif analysis_category == "Decision Intelligence":
        analysis_mode = st.sidebar.radio(
            "Decision Intelligence Mode",
            [
                "Choice Prediction Model",  # supervised (existing tree_tab)
                "Decision Tree & Patterns",  # unsupervised CDT (new)
            ],
            index=0,
        )
    elif analysis_category == "Customer Segmentation":
        analysis_mode = "Customer Segmentation"
    elif analysis_category == "Product Performance":
        analysis_mode = "Product Performance"
    elif analysis_category == "Cohort Analysis":
        analysis_mode = "Cohort Analysis"
    elif analysis_category == "Promotional Analytics":
        analysis_mode = "Promotional Analytics"
    else:
        analysis_mode = "Association Rules"

    # Analysis-specific options
    analysis_params = {}

    if analysis_mode == "Co-purchase":
        analysis_params["top_n_products"] = st.sidebar.slider(
            "Top N Products", 10, 200, 50
        )
        analysis_params["min_lift"] = st.sidebar.slider("Min Lift", 1.0, 3.0, 1.5, 0.1)

    elif analysis_mode == "Add-on":
        analysis_params["min_support"] = st.sidebar.slider(
            "Min Support", 0.0005, 0.01, 0.002, 0.0005
        )
        analysis_params["min_lift"] = st.sidebar.slider("Min Lift", 1.0, 3.0, 1.2, 0.1)
        analysis_params["top_n"] = st.sidebar.slider("Top N Recommendations", 5, 20, 10)

    elif analysis_mode == "Switching":
        analysis_params["window_days"] = st.sidebar.slider("Window (days)", 30, 365, 90)
        analysis_params["min_transactions"] = st.sidebar.slider(
            "Min Customer Transactions", 2, 10, 3
        )

    elif analysis_mode == "Choice Prediction Model":
        analysis_params["max_depth"] = st.sidebar.slider("Max Tree Depth", 2, 8, 4)
        analysis_params["min_samples_leaf"] = st.sidebar.slider(
            "Min Samples Leaf", 5, 50, 10
        )
        analysis_params["prediction_window"] = st.sidebar.slider(
            "Prediction Window (days)", 7, 90, 30
        )

    elif analysis_mode == "Decision Tree & Patterns":
        st.sidebar.markdown("**Similarity**")
        analysis_params["similarity_method"] = st.sidebar.selectbox(
            "Similarity Method", ["yules_q", "jaccard"], index=0
        )
        analysis_params["min_cooccurrence"] = st.sidebar.slider(
            "Min Co-occurrence", 2, 20, 5
        )

        st.sidebar.markdown("**Clustering**")
        analysis_params["linkage_method"] = st.sidebar.selectbox(
            "Linkage Method", ["average", "complete", "single"], index=0
        )
        analysis_params["min_k"] = st.sidebar.slider("Min Clusters (k)", 2, 10, 2)
        analysis_params["max_k"] = st.sidebar.slider("Max Clusters (k)", 3, 20, 15)

        st.sidebar.markdown("**Tree Building**")
        analysis_params["min_cluster_size"] = st.sidebar.slider(
            "Min Cluster Size", 2, 10, 3
        )
        analysis_params["quality_threshold"] = st.sidebar.slider(
            "Quality Threshold (%)", 40, 80, 60
        )

        st.sidebar.markdown("**Behavioral**")
        analysis_params["top_n_products"] = st.sidebar.slider(
            "Top N Products", 20, 200, 50
        )
        analysis_params["min_lift"] = st.sidebar.slider("Min Lift", 1.0, 3.0, 1.2, 0.1)
        analysis_params["max_sub"] = st.sidebar.slider(
            "Max Substitution", 0.0, 0.5, 0.3, 0.05
        )

    elif analysis_mode == "Customer Segmentation":
        analysis_params["rfm_method"] = st.sidebar.radio(
            "RFM Method", ["Quantile (Classic)", "K-Means"], key="sidebar_rfm_method"
        )
        analysis_params["n_segments"] = st.sidebar.slider(
            "K-Means Segments", 3, 12, 8, key="sidebar_n_segments"
        )
        analysis_params["behavioral_clusters"] = st.sidebar.slider(
            "Behavioral Clusters", 3, 10, 6, key="sidebar_behav_clusters"
        )
        analysis_params["value_horizon"] = st.sidebar.slider(
            "CLV Horizon (days)", 30, 365, 90, key="sidebar_value_horizon"
        )

    elif analysis_mode == "Product Performance":
        analysis_params["top_n_products"] = st.sidebar.slider(
            "Top N Products", 10, 100, 20, key="prod_top_n"
        )
        analysis_params["lifecycle_period"] = st.sidebar.selectbox(
            "Lifecycle Period", ["Monthly", "Weekly"], key="prod_lifecycle_period"
        )
        analysis_params["elasticity_min_periods"] = st.sidebar.slider(
            "Elasticity Min Periods", 10, 50, 20, key="prod_elasticity_min"
        )

    elif analysis_mode == "Cohort Analysis":
        analysis_params["cohort_period"] = st.sidebar.selectbox(
            "Cohort Period",
            ["Weekly", "Monthly", "Quarterly"],
            index=1,
            key="cohort_period",
        )
        analysis_params["cohort_metric"] = st.sidebar.selectbox(
            "Metric",
            [
                "Retention Rate",
                "Revenue per Customer",
                "Number of Customers",
                "Average Order Value",
            ],
            index=0,
            key="cohort_metric",
        )
        analysis_params["max_periods"] = st.sidebar.slider(
            "Max Periods to Show", 3, 24, 12, key="cohort_max_periods"
        )

    elif analysis_mode == "Promotional Analytics":
        analysis_params["price_change_threshold"] = st.sidebar.slider(
            "Price Drop Threshold (%)", 5, 50, 15, key="promo_price_drop"
        )
        analysis_params["min_duration_days"] = st.sidebar.slider(
            "Min Promo Duration (days)", 1, 14, 3, key="promo_min_dur"
        )
        analysis_params["max_duration_days"] = st.sidebar.slider(
            "Max Promo Duration (days)", 14, 60, 30, key="promo_max_dur"
        )
        analysis_params["baseline_window"] = st.sidebar.slider(
            "Baseline Window (days)", 14, 90, 30, key="promo_baseline"
        )
        analysis_params["promo_window"] = st.sidebar.slider(
            "Promo Window (days)", 7, 30, 14, key="promo_window"
        )

    run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary", width="stretch")

    return {
        "uploaded_file": uploaded_file,
        "use_sample": use_sample,
        "column_mapping": column_mapping,
        "min_support": min_support,
        "min_confidence": min_confidence,
        "max_itemset_len": max_itemset_len,
        "min_lift": min_lift,
        "analysis_mode": analysis_mode,
        "analysis_params": analysis_params,
        "run_analysis": run_analysis,
    }


def render_data_info(df: pd.DataFrame):
    """Display data summary in sidebar."""
    with st.sidebar.expander("📈 Data Summary", expanded=False):
        st.write(f"**Transactions:** {df['transaction_id'].nunique():,}")
        st.write(f"**Customers:** {df['customer_id'].nunique():,}")
        st.write(f"**Products:** {df['stockcode'].nunique():,}")
        st.write(
            f"**Date Range:** {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
        )
        st.write(f"**Total Revenue:** ${(df['price'] * df['quantity']).sum():,.2f}")
