"""Sidebar UI components."""

import hashlib
from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.config import Config


@st.cache_data
def _detect_columns_from_file(file_bytes: bytes) -> Dict[str, str]:
    """Detect columns from uploaded file (cached by file hash)."""
    import io

    df = pd.read_csv(io.BytesIO(file_bytes), nrows=5)
    cols = df.columns.tolist()

    # Auto-detect common column names
    def find_col(candidates):
        for c in candidates:
            if c in cols:
                return c
        return cols[0] if cols else ""

    return {
        "date": find_col(["date", "transaction_date", "dt"]),
        "transaction_id": find_col(["transaction_id", "txn_id", "order_id", "basket_id"]),
        "stockcode": find_col(["stockcode", "item_code", "sku", "product_code"]),
        "product": find_col(["product", "product_name", "item_name", "description"]),
        "customer_id": find_col(["customer_id", "cust_id", "client_id", "user_id"]),
        "price": find_col(["price", "unit_price", "amount", "sales_price"]),
        "quantity": find_col(["quantity", "qty", "units", "quantity_sold"]),
    }


def render_sidebar() -> Config:
    """Render sidebar with file upload and analysis parameters."""
    st.sidebar.header(" Data Upload")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Transaction CSV",
        type=["csv"],
        help="CSV with columns: date, transaction_id, stockcode, product, customer_id, price, quantity",
    )

    use_sample = st.sidebar.checkbox("Use Sample Data", value=False)

    # Initialize session state for column mapping
    if "column_mapping_confirmed" not in st.session_state:
        st.session_state.column_mapping_confirmed = False
    if "loaded_df" not in st.session_state:
        st.session_state.loaded_df = None

    if uploaded_file:
        try:
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0)

            # Cache column detection by file hash
            file_hash = hashlib.md5(file_bytes).hexdigest()[:16]

            if "cached_column_mapping" not in st.session_state:
                st.session_state.cached_column_mapping = {}

            if file_hash not in st.session_state.cached_column_mapping:
                st.session_state.cached_column_mapping[file_hash] = _detect_columns_from_file(
                    file_bytes
                )

            detected_mapping = st.session_state.cached_column_mapping[file_hash]
            cols = list(detected_mapping.values())

            # Show column mapping with detected values as defaults
            with st.sidebar.expander(
                " Column Mapping", expanded=not st.session_state.column_mapping_confirmed
            ):
                date_col = st.selectbox(
                    "Date Column",
                    cols,
                    index=cols.index(detected_mapping["date"])
                    if detected_mapping["date"] in cols
                    else 0,
                )
                trans_col = st.selectbox(
                    "Transaction ID Column",
                    cols,
                    index=cols.index(detected_mapping["transaction_id"])
                    if detected_mapping["transaction_id"] in cols
                    else 0,
                )
                item_col = st.selectbox(
                    "Item Column (Stock Code)",
                    cols,
                    index=cols.index(detected_mapping["stockcode"])
                    if detected_mapping["stockcode"] in cols
                    else 0,
                )
                product_col = st.selectbox(
                    "Product Name Column",
                    cols,
                    index=cols.index(detected_mapping["product"])
                    if detected_mapping["product"] in cols
                    else 0,
                )
                customer_col = st.selectbox(
                    "Customer ID Column",
                    cols,
                    index=cols.index(detected_mapping["customer_id"])
                    if detected_mapping["customer_id"] in cols
                    else 0,
                )
                price_col = st.selectbox(
                    "Price Column",
                    cols,
                    index=cols.index(detected_mapping["price"])
                    if detected_mapping["price"] in cols
                    else 0,
                )
                qty_col = st.selectbox(
                    "Quantity Column",
                    cols,
                    index=cols.index(detected_mapping["quantity"])
                    if detected_mapping["quantity"] in cols
                    else 0,
                )

                col_confirm, col_reset = st.columns(2)
                with col_confirm:
                    if st.button(" Confirm Mapping", type="primary", use_container_width=True):
                        st.session_state.column_mapping_confirmed = True
                        st.session_state.cached_column_mapping[file_hash] = {
                            "date": date_col,
                            "transaction_id": trans_col,
                            "stockcode": item_col,
                            "product": product_col,
                            "customer_id": customer_col,
                            "price": price_col,
                            "quantity": qty_col,
                        }
                        st.rerun()
                with col_reset:
                    if st.button(" Reset", use_container_width=True):
                        st.session_state.column_mapping_confirmed = False
                        st.rerun()

            if st.session_state.column_mapping_confirmed:
                column_mapping = st.session_state.cached_column_mapping[file_hash]
            else:
                column_mapping = detected_mapping

        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            column_mapping = {}
    else:
        column_mapping = {}
        st.session_state.column_mapping_confirmed = False

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
        with st.sidebar.expander("**Similarity**", expanded=True):
            analysis_params["similarity_methods"] = st.multiselect(
                "Similarity Methods",
                ["phi", "jaccard", "pmi", "cosine_tfidf"],
                default=["phi"],
                key="cdt_similarity_methods",
            )
            analysis_params["min_cooccurrence"] = st.slider(
                "Min Co-occurrence", 2, 20, 5, key="cdt_min_cooc"
            )

        with st.sidebar.expander("**Community Detection**", expanded=False):
            analysis_params["community_method"] = st.selectbox(
                "Community Method",
                ["none", "label_propagation", "louvain", "leiden"],
                index=1,
                key="cdt_community_method",
            )
            analysis_params["community_resolution"] = st.slider(
                "Resolution", 0.5, 2.0, 1.0, 0.1, key="cdt_community_resolution"
            )
            analysis_params["graph_min_weight"] = st.slider(
                "Graph Min Weight", 0.0, 0.5, 0.1, 0.05, key="cdt_graph_min_weight"
            )
            analysis_params["graph_max_degree"] = st.slider(
                "Graph Max Degree", 10, 100, 50, key="cdt_graph_max_degree"
            )

        with st.sidebar.expander("**Clustering**", expanded=True):
            analysis_params["linkage_method"] = st.selectbox(
                "Linkage Method", ["average", "complete", "single"], index=0, key="cdt_linkage"
            )
            analysis_params["min_k"] = st.slider("Min Clusters (k)", 2, 10, 2, key="cdt_min_k")
            analysis_params["max_k"] = st.slider("Max Clusters (k)", 3, 20, 15, key="cdt_max_k")

        with st.sidebar.expander("**Tree Building**", expanded=True):
            analysis_params["min_cluster_size"] = st.slider(
                "Min Cluster Size", 2, 10, 3, key="cdt_min_cluster"
            )
            analysis_params["quality_threshold"] = st.slider(
                "Quality Threshold (%)", 40, 80, 60, key="cdt_quality"
            )
            analysis_params["split_criterion"] = st.selectbox(
                "Split Criterion",
                ["mutual_info", "gini", "entropy", "mixed"],
                index=0,
                key="cdt_split_criterion",
            )
            analysis_params["split_alpha"] = st.slider(
                "Split Alpha (entropy/Gini mix)", 0.0, 1.0, 0.5, 0.1, key="cdt_split_alpha"
            )
            analysis_params["extract_from_text"] = st.checkbox(
                "Extract Attributes from Product Text", value=False, key="cdt_extract_text"
            )

        with st.sidebar.expander("**Behavioral**", expanded=False):
            analysis_params["top_n_products"] = st.slider(
                "Top N Products", 20, 200, 50, key="cdt_top_n"
            )
            analysis_params["min_lift"] = st.slider(
                "Min Lift", 1.0, 3.0, 1.2, 0.1, key="cdt_min_lift"
            )
            analysis_params["max_sub"] = st.slider(
                "Max Substitution", 0.0, 0.5, 0.3, 0.05, key="cdt_max_sub"
            )

    elif analysis_mode == "Demand Transference":
        # Dynamic SKU list for delist products
        loaded_df = st.session_state.get("loaded_df")
        sku_options = loaded_df["stockcode"].unique().tolist() if loaded_df is not None else []

        analysis_params["substitution_source"] = st.sidebar.selectbox(
            "Substitution Source", ["switching", "cdt"], index=0, key="dt_sub_source"
        )
        analysis_params["delist_products"] = st.sidebar.multiselect(
            "Products to Delist", sku_options, key="dt_delist_products"
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

        # Bayesian sampling mode - ONLY shown when bayesian_hierarchical is selected
        if analysis_params["elasticity_method"] == "bayesian_hierarchical":
            analysis_params["bayesian_mode"] = st.sidebar.radio(
                "Bayesian Sampling Mode",
                ["fast (ADVI)", "full (NUTS)"],
                index=0,
                key="bayesian_mode_elasticity",
                help="Fast = ADVI (variational inference, approximate but quick). Full = NUTS (MCMC, exact but slower).",
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
        # Legacy mode banner
        st.sidebar.info(
            "⚠️ **Legacy Mode** — Use **Pricing & Promotions → Promo Uplift Modeling** for causal uplift estimation"
        )
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

    # Data quality validation
    validation_warnings = validate_data_quality(
        st.session_state.get("loaded_df"), analysis_mode, analysis_params
    )

    # BUG 1 FIX: Store run_analysis in session_state to persist across reruns
    if "run_analysis_triggered" not in st.session_state:
        st.session_state.run_analysis_triggered = False

    # Data quality validation (only if data is loaded)
    loaded_df = st.session_state.get("loaded_df")
    if loaded_df is not None:
        validation_warnings = validate_data_quality(loaded_df, analysis_mode, analysis_params)
        if validation_warnings:
            with st.sidebar.container():
                for w in validation_warnings:
                    if w["level"] == "error":
                        st.sidebar.error(w["message"])
                    else:
                        st.sidebar.warning(w["message"])

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


def validate_data_quality(df: pd.DataFrame, analysis_mode: str, params: dict) -> list[dict]:
    """Validate data quality before running analysis.

    Returns list of warning dicts with 'level' (error/warning/info) and 'message'.
    """
    if df is None or df.empty:
        return []

    warnings = []

    # Basic counts
    n_transactions = df["transaction_id"].nunique() if "transaction_id" in df.columns else 0
    n_customers = df["customer_id"].nunique() if "customer_id" in df.columns else 0
    n_products = df["stockcode"].nunique() if "stockcode" in df.columns else 0

    # Null checks
    customer_null_pct = (
        df["customer_id"].isnull().mean() * 100 if "customer_id" in df.columns else 0
    )
    date_null_pct = df["date"].isnull().mean() * 100 if "date" in df.columns else 0
    price_null_pct = df["price"].isnull().mean() * 100 if "price" in df.columns else 0

    # General checks
    if n_transactions < 100:
        warnings.append(
            {
                "level": "error",
                "message": f"Only {n_transactions} transactions — too few for reliable analysis",
            }
        )
    elif n_transactions < 500:
        warnings.append(
            {
                "level": "warning",
                "message": f"Only {n_transactions} transactions — results may be directional",
            }
        )

    if customer_null_pct > 20:
        warnings.append(
            {
                "level": "error",
                "message": f"{customer_null_pct:.0f}% null customer_id — segmentation will drop rows",
            }
        )
    elif customer_null_pct > 5:
        warnings.append(
            {
                "level": "warning",
                "message": f"{customer_null_pct:.0f}% null customer_id — some customers unassigned",
            }
        )

    if date_null_pct > 10:
        warnings.append(
            {
                "level": "warning",
                "message": f"{date_null_pct:.0f}% null dates — cohort/time-series analysis affected",
            }
        )

    if price_null_pct > 5:
        warnings.append(
            {
                "level": "warning",
                "message": f"{price_null_pct:.0f}% null prices — elasticity/KVI unreliable",
            }
        )

    # Mode-specific checks
    if analysis_mode in ["Promo Uplift Modeling", "Elasticity Analysis"]:
        if n_transactions < 1000:
            warnings.append(
                {
                    "level": "warning",
                    "message": f"{n_transactions} transactions — uplift/elasticity needs more data",
                }
            )
        # Check price variation
        if "price" in df.columns and "stockcode" in df.columns:
            price_cv = df.groupby("stockcode")["price"].apply(
                lambda x: x.std() / x.mean() if x.mean() > 0 else 0
            )
            low_var_pct = (price_cv < 0.03).mean() * 100
            if low_var_pct > 50:
                warnings.append(
                    {
                        "level": "warning",
                        "message": f"{low_var_pct:.0f}% of SKUs have low price variation (CV<3%) — elasticity estimates unreliable",
                    }
                )

    if analysis_mode in ["Customer Segmentation", "Promo Uplift Modeling"]:
        if n_customers < 50:
            warnings.append(
                {
                    "level": "error",
                    "message": f"Only {n_customers} customers — segmentation/uplift modeling unreliable",
                }
            )
        elif n_customers < 200:
            warnings.append(
                {
                    "level": "warning",
                    "message": f"Only {n_customers} customers — consider fewer segments",
                }
            )

    if analysis_mode == "Promo Uplift Modeling":
        # Check for promo detection feasibility
        baseline_window = params.get("promo_baseline_window", 28)
        if n_transactions < baseline_window * 10:
            warnings.append(
                {
                    "level": "warning",
                    "message": f"Baseline window ({baseline_window}d) may be too long for {n_transactions} transactions",
                }
            )

    return warnings
