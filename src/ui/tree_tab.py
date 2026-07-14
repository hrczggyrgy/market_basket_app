"""Customer Choice Modelling analysis tab with persistent tab state."""

import pandas as pd
import streamlit as st

from src.models.decision_tree import (
    build_customer_features,
    extract_tree_rules,
    predict_for_customer,
    train_decision_tree,
)
from src.ui.export import render_analytics_export
from src.ui.tabs import persistent_tabs
from src.viz.decision_tree import (
    plot_decision_tree,
    plot_feature_importance,
    plot_tree_rules,
)


def render_tree_tab(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render customer choice modelling analysis tab with persistent sub-tabs."""
    st.header("🌳 Customer Choice Modelling - Product Purchase Prediction")

    if transactions_df.empty:
        st.warning("No transaction data available")
        return

    # Product selection
    st.subheader("Select Target Product")

    products = transactions_df["stockcode"].unique()
    target_product = st.selectbox(
        "Product to Predict",
        options=products,
        format_func=lambda x: product_lookup.get(x, x),
        key="tree_target_product",
    )

    if not target_product:
        st.info("Please select a product")
        return

    # Model parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        max_depth = st.slider(
            "Max Tree Depth", 2, 8, params.get("max_depth", 4), key="tree_max_depth"
        )
    with col2:
        min_samples_leaf = st.slider(
            "Min Samples Leaf",
            5,
            50,
            params.get("min_samples_leaf", 10),
            key="tree_min_leaf",
        )
    with col3:
        pred_window = st.slider(
            "Prediction Window (days)",
            7,
            90,
            params.get("prediction_window", 30),
            key="tree_pred_window",
        )

    # Build and train
    with st.spinner(
        f"Building features and training tree for {product_lookup.get(target_product, target_product)}..."
    ):
        # Build features
        X, y = build_customer_features(
            transactions_df,
            target_product,
            prediction_window_days=pred_window,
            min_history_days=60,
        )

        if X.empty or y.sum() == 0:
            st.error(
                "Insufficient data for this product. Try a different product or longer prediction window."
            )
            return

        # Train model
        model, metrics = train_decision_tree(
            X,
            y,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
        )

    if model is None:
        st.error(metrics.get("error", "Training failed"))
        return

    # Display metrics
    st.subheader("Model Performance")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test Accuracy", f"{metrics['test_accuracy']:.2%}")
    with col2:
        st.metric("Train Accuracy", f"{metrics['train_accuracy']:.2%}")
    with col3:
        st.metric("Positive Class Rate", f"{metrics['positive_class_rate']:.2%}")
    with col4:
        st.metric("Tree Depth", metrics["tree_depth"])

    # Feature importance
    st.subheader("Top Features")

    fig_imp = plot_feature_importance(
        model,
        X.columns.tolist(),
        top_n=15,
        title=f"Feature Importance for {product_lookup.get(target_product, target_product)}",
    )
    st.plotly_chart(fig_imp, width="stretch")

    # Persistent tabs for different views
    tab_labels = ["🌲 Decision Tree", "📝 Extracted Rules", "🎯 Customer Predictions"]
    selected = persistent_tabs(tab_labels, "tree_view_tabs", default_tab=0)

    if selected == 0:
        _render_tree_tab(model, X.columns.tolist(), product_lookup, target_product, max_depth)
    elif selected == 1:
        _render_rules_tab(model, X.columns.tolist(), target_product, product_lookup)
    elif selected == 2:
        _render_predictions_tab(model, X, product_lookup, target_product)


def _render_tree_tab(model, feature_names: list, product_lookup: dict, target_product: str, max_depth: int):
    """Render the decision tree visualization tab."""
    st.subheader("Interactive Decision Tree")

    fig_tree = plot_decision_tree(
        model,
        feature_names,
        class_names=["Not Buy", "Buy"],
        max_depth=min(max_depth, 4),
        title=f"Decision Tree: Will Customer Buy {product_lookup.get(target_product, target_product)}?",
    )
    st.plotly_chart(fig_tree, width="stretch")


def _render_rules_tab(model, feature_names: list, target_product: str, product_lookup: dict):
    """Render the extracted rules tab."""
    st.subheader("Extracted Decision Rules")

    rules = extract_tree_rules(model, feature_names)

    if rules:
        # Filter to leaf rules
        leaf_rules = [r for r in rules if r.get("is_leaf", False)]

        if leaf_rules:
            # Show as table
            rule_df = pd.DataFrame(leaf_rules)
            rule_df["Conditions"] = rule_df["conditions"].apply(
                lambda x: " AND ".join(x)
            )
            rule_df["Prediction"] = rule_df["prediction"]
            rule_df["P(Buy)"] = rule_df["probability"]
            rule_df["Samples"] = rule_df["samples"]

            display_cols = ["Conditions", "Prediction", "P(Buy)", "Samples"]
            st.dataframe(
                rule_df[display_cols].round(4),
                width="stretch",
                hide_index=True,
                height=400,
            )

            render_analytics_export(rule_df, f"Tree_Rules_{target_product}")

            # Visualization
            fig_rules = plot_tree_rules(rules, feature_names)
            st.plotly_chart(fig_rules, width="stretch")
        else:
            st.info("No leaf rules extracted")
    else:
        st.info("No rules extracted")


def _render_predictions_tab(model, X: pd.DataFrame, product_lookup: dict, target_product: str):
    """Render the customer predictions tab."""
    st.subheader("Individual Customer Predictions")

    # Select customer
    customers = X.index.tolist()
    if customers:
        selected_customer = st.selectbox(
            "Select Customer",
            options=customers[:100],  # Limit for performance
            key="tree_customer_select",
        )

        if selected_customer:
            prediction = predict_for_customer(model, X, selected_customer)

            if "error" not in prediction:
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Prediction", prediction["prediction"])
                    st.metric("P(Buy)", f"{prediction['probability_buy']:.2%}")
                    st.metric(
                        "P(Not Buy)", f"{prediction['probability_not_buy']:.2%}"
                    )

                with col2:
                    st.write("**Decision Path:**")
                    for condition in prediction["decision_path"]:
                        st.write(f"• {condition}")

                # Show customer features
                with st.expander("Customer Features"):
                    cust_features = X.loc[selected_customer]
                    non_zero = cust_features[cust_features != 0].sort_values(
                        ascending=False
                    )
                    st.dataframe(non_zero.round(4), width="stretch")
            else:
                st.error(prediction["error"])