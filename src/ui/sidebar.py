"""Sidebar UI components."""

from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.config import Config


def render_sidebar() -> Config:
    """Render sidebar with file upload and analysis parameters."""
    st.sidebar.header(" Data Upload")

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
    st.sidebar.header(" FP-Growth Parameters")

    min_support = st.sidebar.slider(
        "Min Support",
        0.0005,
        0.05,
        0.002,
        0.0005,
        help="Minimum support threshold (fraction of transactions)",
        key="sidebar_min_support",
    )

    min_confidence = st.sidebar.slider(
        "Min Confidence",
        0.01,
        0.5,
        0.1,
        0.01,
        help="Minimum confidence for association rules (typical range: 0.05-0.3)",
        key="sidebar_min_confidence",
    )

    max_itemset_len = st.sidebar.slider(
        "Max Itemset Length",
        2,
        6,
        3,
        help="Maximum number of items in frequent itemsets",
        key="sidebar_max_itemset_len",
    )

    min_lift = st.sidebar.slider(
        "Min Lift",
        0.5,
        5.0,
        1.2,
        0.1,
        help="Minimum lift threshold for rules",
        key="sidebar_min_lift",
    )

    st.sidebar.divider()
    st.sidebar.header(" Analysis Options")

    # Main analysis category - new structure with backward compat
    analysis_category = st.sidebar.radio(
        "Analysis Category",
        [
            "Association Rules",
            "CDT & Assortment",  # NEW primary
            "Pricing & Promotions",  # NEW primary
            "Customer Segmentation",
            "Product Performance",
            "Cohort Analysis",
            "Promotional Analytics",  # Legacy - kept for compat
        ],
        index=0,
        key="sidebar_analysis_category",
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
            key="sidebar_analysis_mode_assoc",
        )
    elif analysis_category == "CDT & Assortment":
        analysis_mode = st.sidebar.radio(
            "CDT & Assortment Mode",
            [
                "CDT Builder",  # enhanced CDT with community detection
                "Demand Transference",  # delist simulation & substitution
                "Assortment Optimizer",  # MILP/heuristic range optimization
                "CDT Benchmark",  # synthetic validation
            ],
            index=0,
            key="sidebar_analysis_mode_cdt",
        )
    elif analysis_category == "Pricing & Promotions":
        analysis_mode = st.sidebar.radio(
            "Pricing & Promotions Mode",
            [
                "Elasticity Analysis",  # price elasticity estimation
                "KVI Identification",  # key value item scoring
                "Price Curve Diagnostics",  # tier clustering & violations
                "Promo Uplift Modeling",  # causal uplift estimation
                "Elasticity Benchmark",  # synthetic-data validation
            ],
            index=0,
            key="sidebar_analysis_mode_pricing",
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
    analysis_params: Dict[str, Any] = {}

    # Global Bayesian sampling mode (Fast ADVI / Full NUTS)
    analysis_params["bayesian_mode"] = st.sidebar.radio(
        "Bayesian Sampling Mode",
        ["fast (ADVI)", "full (NUTS)"],
        index=0,
        key="bayesian_mode_global",
        help="Fast = ADVI (variational inference, approximate but quick). "
        "Full = NUTS (MCMC, exact but slower).",
    )

    if analysis_mode == "Co-purchase":
        analysis_params["top_n_products"] = st.sidebar.slider(
            "Top N Products", 10, 200, 50, key="copurchase_top_n"
        )
        analysis_params["min_lift"] = st.sidebar.slider(
            "Min Lift", 1.0, 3.0, 1.5, 0.1, key="copurchase_min_lift"
        )

    elif analysis_mode == "Add-on":
        analysis_params["min_support"] = st.sidebar.slider(
            "Min Support", 0.0005, 0.01, 0.002, 0.0005, key="addon_min_support"
        )
        analysis_params["min_lift"] = st.sidebar.slider(
            "Min Lift", 1.0, 3.0, 1.2, 0.1, key="addon_min_lift"
        )
        analysis_params["top_n"] = st.sidebar.slider(
            "Top N Recommendations", 5, 20, 10, key="addon_top_n"
        )

    elif analysis_mode == "Switching":
        analysis_params["window_days"] = st.sidebar.slider(
            "Window (days)", 30, 365, 90, key="switching_window"
        )
        analysis_params["min_transactions"] = st.sidebar.slider(
            "Min Customer Transactions", 2, 10, 3, key="switching_min_trans"
        )

    elif analysis_mode == "CDT Builder":
        st.sidebar.markdown("**Similarity**")
        analysis_params["similarity_methods"] = st.sidebar.multiselect(
            "Similarity Methods",
            ["phi", "jaccard", "pmi", "cosine_tfidf"],
            default=["phi"],
            key="cdt_similarity_methods",
        )
        analysis_params["min_cooccurrence"] = st.sidebar.slider(
            "Min Co-occurrence", 2, 20, 5, key="cdt_min_cooc"
        )

        st.sidebar.markdown("**Community Detection**")
        analysis_params["community_method"] = st.sidebar.selectbox(
            "Community Method",
            ["none", "label_propagation", "louvain", "leiden"],
            index=1,
            key="cdt_community_method",
        )
        analysis_params["community_resolution"] = st.sidebar.slider(
            "Resolution", 0.5, 2.0, 1.0, 0.1, key="cdt_community_resolution"
        )
        analysis_params["graph_min_weight"] = st.sidebar.slider(
            "Graph Min Weight", 0.0, 0.5, 0.1, 0.05, key="cdt_graph_min_weight"
        )
        analysis_params["graph_max_degree"] = st.sidebar.slider(
            "Graph Max Degree", 10, 100, 50, key="cdt_graph_max_degree"
        )

        st.sidebar.markdown("**Clustering**")
        analysis_params["linkage_method"] = st.sidebar.selectbox(
            "Linkage Method", ["average", "complete", "single"], index=0, key="cdt_linkage"
        )
        analysis_params["min_k"] = st.sidebar.slider("Min Clusters (k)", 2, 10, 2, key="cdt_min_k")
        analysis_params["max_k"] = st.sidebar.slider("Max Clusters (k)", 3, 20, 15, key="cdt_max_k")

        st.sidebar.markdown("**Tree Building**")
        analysis_params["min_cluster_size"] = st.sidebar.slider(
            "Min Cluster Size", 2, 10, 3, key="cdt_min_cluster"
        )
        analysis_params["quality_threshold"] = st.sidebar.slider(
            "Quality Threshold (%)", 40, 80, 60, key="cdt_quality"
        )
        analysis_params["split_criterion"] = st.sidebar.selectbox(
            "Split Criterion",
            ["mutual_info", "gini", "entropy", "mixed"],
            index=0,
            key="cdt_split_criterion",
        )
        analysis_params["split_alpha"] = st.sidebar.slider(
            "Split Alpha (entropy/Gini mix)", 0.0, 1.0, 0.5, 0.1, key="cdt_split_alpha"
        )
        analysis_params["extract_from_text"] = st.sidebar.checkbox(
            "Extract Attributes from Product Text", value=False, key="cdt_extract_text"
        )

        st.sidebar.markdown("**Behavioral**")
        analysis_params["top_n_products"] = st.sidebar.slider(
            "Top N Products", 20, 200, 50, key="cdt_top_n"
        )
        analysis_params["min_lift"] = st.sidebar.slider(
            "Min Lift", 1.0, 3.0, 1.2, 0.1, key="cdt_min_lift"
        )
        analysis_params["max_sub"] = st.sidebar.slider(
            "Max Substitution", 0.0, 0.5, 0.3, 0.05, key="cdt_max_sub"
        )

    elif analysis_mode == "Demand Transference":
        analysis_params["substitution_source"] = st.sidebar.selectbox(
            "Substitution Source", ["switching", "cdt"], index=0, key="dt_sub_source"
        )
        analysis_params["delist_products"] = st.sidebar.multiselect(
            "Products to Delist", [], key="dt_delist_products"
        )
        analysis_params["max_recovery"] = st.sidebar.slider(
            "Max Recovery Constraint", 0.5, 1.0, 1.0, 0.05, key="dt_max_recovery"
        )
        analysis_params["show_cannibalization"] = st.sidebar.checkbox(
            "Show Cannibalization", value=True, key="dt_show_cannibalization"
        )

    elif analysis_mode == "Assortment Optimizer":
        analysis_params["max_skus"] = st.sidebar.slider(
            "Max SKUs", 20, 500, 100, key="assort_max_skus"
        )
        analysis_params["min_coverage"] = st.sidebar.slider(
            "Min Coverage %", 50, 95, 80, key="assort_min_coverage"
        )
        analysis_params["objective"] = st.sidebar.selectbox(
            "Objective", ["revenue", "margin"], index=0, key="assort_objective"
        )
        analysis_params["solver"] = st.sidebar.selectbox(
            "Solver", ["heuristic", "milp"], index=0, key="assort_solver"
        )
        analysis_params["time_limit"] = st.sidebar.slider(
            "Time Limit (s)", 10, 300, 60, key="assort_time_limit"
        )
        analysis_params["generate_scenarios"] = st.sidebar.button(
            "Generate Scenarios", key="assort_gen_scenarios"
        )

    elif analysis_mode == "CDT Benchmark":
        analysis_params["bench_n_products"] = st.sidebar.slider(
            "Products", 10, 80, 30, key="cdt_bench_n_products"
        )
        analysis_params["bench_n_clusters"] = st.sidebar.slider(
            "True Clusters", 2, 6, 3, key="cdt_bench_n_clusters"
        )
        analysis_params["bench_n_customers"] = st.sidebar.slider(
            "Customers", 50, 500, 200, key="cdt_bench_n_customers"
        )
        analysis_params["bench_noise"] = st.sidebar.slider(
            "Noise Level", 0.0, 0.5, 0.2, 0.05, key="cdt_bench_noise"
        )

    elif analysis_mode == "Elasticity Analysis":
        analysis_params["elasticity_method"] = st.sidebar.selectbox(
            "Method",
            ["loglog_ols", "hierarchical_eb", "bayesian_hierarchical", "xgb"],
            index=0,
            key="price_elasticity_method",
        )
        analysis_params["min_periods"] = st.sidebar.slider(
            "Min Periods", 5, 50, 10, key="price_min_periods"
        )
        analysis_params["min_price_variation"] = st.sidebar.slider(
            "Min Price Variation", 0.01, 0.5, 0.05, 0.01, key="price_min_var"
        )
        analysis_params["show_shap"] = st.sidebar.checkbox(
            "Show SHAP Values", value=False, key="price_show_shap"
        )

    elif analysis_mode == "KVI Identification":
        analysis_params["kvi_method"] = st.sidebar.selectbox(
            "Method", ["xgb_importance", "rfm_elasticity"], index=0, key="kvi_method"
        )
        analysis_params["top_k_kvi"] = st.sidebar.slider("Top K KVI", 10, 100, 20, key="kvi_top_k")
        analysis_params["margin_weighted"] = st.sidebar.checkbox(
            "Margin-Weighted (if cost available)", value=False, key="kvi_margin_weighted"
        )

    elif analysis_mode == "Price Curve Diagnostics":
        analysis_params["price_curve_method"] = st.sidebar.selectbox(
            "Clustering Method", ["kmeans", "gmm"], index=0, key="price_curve_method"
        )
        analysis_params["n_tiers"] = st.sidebar.slider(
            "Number of Tiers", 2, 5, 3, key="price_curve_tiers"
        )

    elif analysis_mode == "Promo Uplift Modeling":
        analysis_params["promo_drop_threshold"] = st.sidebar.slider(
            "Promo Drop Threshold (%)", 5, 50, 15, key="promo_drop_thresh"
        )
        analysis_params["promo_baseline_window"] = st.sidebar.slider(
            "Baseline Window (days)", 14, 90, 28, key="promo_baseline_window"
        )
        analysis_params["uplift_method"] = st.sidebar.selectbox(
            "Uplift Method", ["t_learner", "s_learner"], index=0, key="uplift_method"
        )
        analysis_params["base_n_estimators"] = st.sidebar.slider(
            "Base Learner n_estimators", 50, 500, 200, key="uplift_n_est"
        )
        analysis_params["base_max_depth"] = st.sidebar.slider(
            "Base Learner max_depth", 3, 10, 5, key="uplift_max_depth"
        )
        analysis_params["propensity_stratification"] = st.sidebar.checkbox(
            "Propensity Stratification", value=True, key="uplift_propensity"
        )

    elif analysis_mode == "Elasticity Benchmark":
        analysis_params["benchmark_n_skus"] = st.sidebar.slider(
            "Number of SKUs", 6, 50, 20, key="bench_n_skus"
        )
        analysis_params["benchmark_n_weeks"] = st.sidebar.slider(
            "Weeks of Data", 20, 104, 52, key="bench_n_weeks"
        )
        analysis_params["benchmark_n_categories"] = st.sidebar.slider(
            "Categories", 1, 6, 3, key="bench_n_cats"
        )

    elif analysis_mode == "Choice Prediction Model":
        analysis_params["max_depth"] = st.sidebar.slider(
            "Max Tree Depth", 2, 8, 4, key="choice_max_depth"
        )
        analysis_params["min_samples_leaf"] = st.sidebar.slider(
            "Min Samples Leaf", 5, 50, 10, key="choice_min_leaf"
        )
        analysis_params["prediction_window"] = st.sidebar.slider(
            "Prediction Window (days)", 7, 90, 30, key="choice_pred_window"
        )

    elif analysis_mode == "Customer Segmentation":
        analysis_params["rfm_method"] = st.sidebar.radio(
            "Segmentation Method",
            [
                "Behavioral Clustering (Recommended)",
                "RFM Quantile (Simple, legacy)",
                "RFM K-Means (Simple, legacy)",
            ],
            index=0,
            key="sidebar_rfm_method",
        )
        analysis_params["n_segments"] = st.sidebar.slider(
            "K-Means / Behavioral Segments", 3, 12, 8, key="sidebar_n_segments"
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
            key="sidebar_cohort_period",
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
            key="sidebar_cohort_metric",
        )
        analysis_params["max_periods"] = st.sidebar.slider(
            "Max Periods to Show", 3, 24, 12, key="sidebar_cohort_max_periods"
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

    # BUG 1 FIX: Store run_analysis in session_state to persist across reruns
    if "run_analysis_triggered" not in st.session_state:
        st.session_state.run_analysis_triggered = False

    col_run, col_clear = st.sidebar.columns([2, 1])
    with col_run:
        run_analysis_clicked = st.button(
            " Run Analysis", type="primary", width="stretch", key="run_analysis_btn"
        )
    with col_clear:
        clear_clicked = st.button(" Clear", width="stretch", key="clear_analysis_btn")

    if run_analysis_clicked:
        st.session_state.run_analysis_triggered = True
    if clear_clicked:
        st.session_state.run_analysis_triggered = False

    run_analysis = st.session_state.run_analysis_triggered

    return Config(
        uploaded_file=uploaded_file,
        use_sample=use_sample,
        column_mapping=column_mapping,
        min_support=min_support,
        min_confidence=min_confidence,
        max_itemset_len=max_itemset_len,
        min_lift=min_lift,
        analysis_mode=analysis_mode,
        analysis_params=analysis_params,
        run_analysis=run_analysis,
    )


def render_data_info(df: pd.DataFrame):
    """Display data summary in sidebar."""
    with st.sidebar.expander(" Data Summary", expanded=False):
        st.write(f"**Transactions:** {df['transaction_id'].nunique():,}")
        st.write(f"**Customers:** {df['customer_id'].nunique():,}")
        st.write(f"**Products:** {df['stockcode'].nunique():,}")

        min_date = df["date"].min()
        max_date = df["date"].max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        else:
            date_range = "N/A"
        st.write(f"**Date Range:** {date_range}")

        st.write(f"**Total Revenue:** ${(df['price'] * df['quantity']).sum():,.2f}")
