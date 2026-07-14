"""Models module initialization."""

from .decision_tree import (
    build_customer_features,
    extract_tree_rules,
    get_top_features,
    predict_for_customer,
    train_decision_tree,
)
from .enhanced_tree import (
    calibrate_model,
    compare_models,
    cross_validate_model,
    get_model_feature_importance,
    get_shap_dependence_data,
    get_shap_explanations,
    predict_with_explanation,
    train_gradient_boosting,
    train_lightgbm,
    train_random_forest,
    train_xgboost,
    tune_hyperparameters,
)
from .enhanced_tree import (
    extract_tree_rules as extract_enhanced_tree_rules,
)

__all__ = [
    "build_customer_features",
    "train_decision_tree",
    "extract_tree_rules",
    "get_top_features",
    "predict_for_customer",
    "cross_validate_model",
    "compare_models",
    "tune_hyperparameters",
    "get_shap_explanations",
    "get_shap_dependence_data",
    "extract_enhanced_tree_rules",
    "predict_with_explanation",
    "get_model_feature_importance",
    "calibrate_model",
    "train_xgboost",
    "train_lightgbm",
    "train_random_forest",
    "train_gradient_boosting",
]
