"""Customer Choice Modelling analysis tab with persistent tab state."""

import pandas as pd
import streamlit as st

from src.models.decision_tree import (
    build_customer_features,
    compare_models,
    extract_tree_rules,
    predict_for_customer,
    train_decision_tree,
    train_xgboost,
)
from src.ui.export import render_analytics_export
from src.ui.tabs import persistent_tabs
from src.viz.decision_tree import (
    plot_decision_tree,
    plot_feature_importance,
    plot_tree_rules,
)


def render_tree_tab(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict, pipeline: dict = None
):
    """Render customer choice modelling analysis tab with persistent sub-tabs."""
    st.header(" Customer Choice Modelling - Product Purchase Prediction")

    if transactions_df.empty:
        st.warning("No transaction data available")
        return

    # Model selection
    st.sidebar.markdown("## Model Settings")
    model_type = st.sidebar.radio(
        "Choice Prediction Model",
        ["Simple Tree (Legacy)", "XGBoost (Recommended)"],
        key="tree_model_type",
    )
    use_shap = st.sidebar.checkbox(
        "Show SHAP Feature Importance",
        value=False,
        key="tree_shap_enabled",
        help="Compute SHAP values for XGBoost (slower but more informative)",
    )

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
        f"Building features and training {model_type} for {product_lookup.get(target_product, target_product)}..."
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

        if model_type.startswith("Simple Tree"):
            model, metrics = train_decision_tree(
                X,
                y,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight="balanced",
            )
            shap_values = None
        else:
            model, metrics = train_xgboost(X, y, return_shap=use_shap, max_depth=max_depth)
            shap_values = metrics.pop("shap_values", None)

    if model is None:
        st.error(metrics.get("error", "Training failed"))
        return

    # Model comparison
    st.subheader("Model Comparison")
    comp_df = compare_models(X, y)
    if not comp_df.empty:
        for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if col in comp_df.columns:
                comp_df[col] = comp_df[col].apply(lambda x: f"{x:.2%}")
        st.dataframe(comp_df, width="stretch", hide_index=True)

    # Display metrics
    st.subheader("Model Performance")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.2%}")
    with col2:
        st.metric("Train Accuracy", f"{metrics.get('train_accuracy', 0):.2%}")
    with col3:
        st.metric("Positive Class Rate", f"{metrics.get('positive_class_rate', 0):.2%}")
    with col4:
        val = metrics.get("tree_depth", metrics.get("n_features", ""))
        label = "Tree Depth" if "tree_depth" in metrics else "Features"
        st.metric(label, val)

    # Feature importance
    st.subheader("Top Features")

    if shap_values is not None and use_shap and "shap_feature_importance" in metrics:
        shap_imp = metrics["shap_feature_importance"]
        shap_df = pd.DataFrame(
            sorted(shap_imp.items(), key=lambda x: x[1], reverse=True)[:15],
            columns=["Feature", "Mean |SHAP|"],
        )
        st.dataframe(shap_df, width="stretch", hide_index=True)
    else:
        fig_imp = plot_feature_importance(
            model,
            X.columns.tolist(),
            top_n=15,
            title=f"Feature Importance for {product_lookup.get(target_product, target_product)}",
        )
        st.plotly_chart(fig_imp, width="stretch")

    # Persistent tabs for different views
    is_tree = model_type.startswith("Simple Tree")
    if is_tree:
        tab_labels = [" Decision Tree", " Extracted Rules", " Customer Predictions"]
        selected = persistent_tabs(tab_labels, "tree_view_tabs", default_tab=0)

        if selected == 0:
            _render_tree_tab(model, X.columns.tolist(), product_lookup, target_product, max_depth)
        elif selected == 1:
            _render_rules_tab(model, X.columns.tolist(), target_product, product_lookup)
        elif selected == 2:
            _render_predictions_tab(model, X, product_lookup, target_product)
    else:
        tab_labels = [" Predictions", " Feature Importance"]
        selected = persistent_tabs(tab_labels, "xgb_view_tabs", default_tab=0)

        if selected == 0:
            _render_predictions_tab(model, X, product_lookup, target_product)
        elif selected == 1:
            _render_xgb_features_tab(metrics, X, model)


def _render_tree_tab(
    model, feature_names: list, product_lookup: dict, target_product: str, max_depth: int
):
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
            rule_df["Conditions"] = rule_df["conditions"].apply(lambda x: " AND ".join(x))
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
            if hasattr(model, "decision_path"):
                prediction = predict_for_customer(model, X, selected_customer)
            else:
                # XGBoost prediction
                cust_features = X.loc[[selected_customer]]
                pred_class = model.predict(cust_features)[0]
                probs = model.predict_proba(cust_features)[0]
                prediction = {
                    "prediction": "Buy" if pred_class == 1 else "Not Buy",
                    "probability_buy": probs[1],
                    "probability_not_buy": probs[0],
                    "decision_path": [],
                }

            if "error" not in prediction:
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Prediction", prediction["prediction"])
                    st.metric("P(Buy)", f"{prediction['probability_buy']:.2%}")
                    st.metric("P(Not Buy)", f"{prediction['probability_not_buy']:.2%}")

                with col2:
                    decision_path = prediction.get("decision_path", [])
                    if decision_path:
                        st.write("**Decision Path:**")
                        for condition in decision_path:
                            st.write(f"• {condition}")
                    else:
                        st.write("**Feature Values:**")
                        cust_features = X.loc[[selected_customer]].iloc[0]
                        top_feats = cust_features.abs().sort_values(ascending=False).head(10)
                        for feat_name, feat_val in top_feats.items():
                            if feat_val != 0:
                                st.write(f"• {feat_name}: {feat_val:.4f}")

                # Show customer features
                with st.expander("Customer Features"):
                    cust_features = X.loc[selected_customer]
                    non_zero = cust_features[cust_features != 0].sort_values(ascending=False)
                    st.dataframe(non_zero.round(4), width="stretch")
            else:
                st.error(prediction["error"])


def _render_xgb_features_tab(metrics: dict, X: pd.DataFrame, model) -> None:
    """Render XGBoost feature importance tab."""
    st.subheader("XGBoost Feature Importance")

    import numpy as np

    fi = metrics.get("feature_importances", {})
    if fi:
        fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        fi_df = pd.DataFrame(fi_sorted[:20], columns=["Feature", "Importance"])
        st.dataframe(fi_df, width="stretch", hide_index=True)

    shap_fi = metrics.get("shap_feature_importance", {})
    if shap_fi:
        st.subheader("SHAP Feature Importance")
        sf_sorted = sorted(shap_fi.items(), key=lambda x: x[1], reverse=True)
        sf_df = pd.DataFrame(sf_sorted[:20], columns=["Feature", "Mean |SHAP|"])
        st.dataframe(sf_df, width="stretch", hide_index=True)

        if "shap_values" in metrics and "shap_test_data" in metrics:
            try:
                import matplotlib.pyplot as plt
                import shap

                sv = np.array(metrics["shap_values"])
                X_test = metrics["shap_test_data"]
                if sv.ndim == 2 and sv.shape[0] == X_test.shape[0]:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(sv, X_test, feature_names=X_test.columns.tolist(), show=False)
                    st.pyplot(fig)
                    plt.close()
            except Exception as exc:
                st.warning(f"SHAP summary plot unavailable: {exc}")
