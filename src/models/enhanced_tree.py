"""Enhanced tree models with XGBoost, LightGBM, SHAP explanations, and cross-validation."""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# Optional imports
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def get_available_models() -> Dict[str, bool]:
    """Check which modeling libraries are available."""
    return {
        "sklearn_tree": True,
        "sklearn_rf": True,
        "sklearn_gb": True,
        "xgboost": XGBOOST_AVAILABLE,
        "lightgbm": LIGHTGBM_AVAILABLE,
        "shap": SHAP_AVAILABLE,
    }


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgboost",
    max_depth: int = 5,
    min_samples_leaf: int = 10,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    class_weight: str = "balanced",
    random_state: int = 42,
    calibrate: bool = False,
    **kwargs,
) -> Tuple[Any, Dict]:
    """
    Train a classification model with comprehensive metrics.

    Args:
        X: Feature matrix
        y: Target vector
        model_type: One of 'decision_tree', 'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm'
        max_depth: Maximum tree depth
        min_samples_leaf: Minimum samples per leaf
        n_estimators: Number of trees (for ensemble methods)
        learning_rate: Learning rate (for boosting)
        class_weight: Class weighting strategy
        random_state: Random seed
        calibrate: Whether to calibrate probabilities
        **kwargs: Additional model-specific parameters

    Returns:
        (model, metrics_dict)
    """
    if len(y.unique()) < 2:
        return None, {"error": "Only one class present in target"}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = _create_model(
        model_type=model_type,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        class_weight=class_weight,
        random_state=random_state,
        **kwargs,
    )

    if model is None:
        return None, {"error": f"Model type '{model_type}' not available"}

    model.fit(X_train, y_train)

    if calibrate:
        model = CalibratedClassifierCV(model, method="isotonic", cv=3)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = _compute_metrics(y_test, y_pred, y_prob, X_train, X_test, model, y_train)

    return model, metrics


def _create_model(
    model_type: str,
    max_depth: int,
    min_samples_leaf: int,
    n_estimators: int,
    learning_rate: float,
    class_weight: str,
    random_state: int,
    **kwargs,
):
    """Create model instance based on type."""

    if model_type == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=kwargs.get("min_samples_split", 20),
            class_weight=class_weight,
            random_state=random_state,
        )

    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=kwargs.get("min_samples_split", 20),
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    elif model_type == "xgboost" and XGBOOST_AVAILABLE:
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_samples_leaf,
            learning_rate=learning_rate,
            scale_pos_weight=kwargs.get("scale_pos_weight", 1),
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        )

    elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_samples=min_samples_leaf,
            learning_rate=learning_rate,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    return None


def _compute_metrics(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model: Any,
    y_train: pd.Series,
) -> Dict:
    """Compute comprehensive model metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    # Feature importances
    if hasattr(model, "feature_importances_"):
        feature_importances = dict(zip(X_train.columns, model.feature_importances_))
    elif hasattr(model, "coef_"):
        feature_importances = dict(zip(X_train.columns, np.abs(model.coef_[0])))
    else:
        feature_importances = {}

    # Calibration metrics
    try:
        from sklearn.calibration import calibration_curve

        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        calibration_error = np.mean(np.abs(prob_true - prob_pred))
    except Exception:
        calibration_error = np.nan

    # Precision-Recall AUC
    try:
        pr_auc = np.trapz(*precision_recall_curve(y_test, y_prob)[:2][::-1])
    except Exception:
        pr_auc = np.nan

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "train_accuracy": model.score(X_train, y_train) if hasattr(model, "score") else np.nan,
        "test_accuracy": model.score(X_test, y_test) if hasattr(model, "score") else np.nan,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_test, y_prob) if len(y_test.unique()) > 1 else 0.5,
        "pr_auc": pr_auc,
        "calibration_error": calibration_error,
        "n_features": X_train.shape[1],
        "n_samples": X_train.shape[0],
        "positive_class_rate": y_test.mean(),
        "feature_importances": feature_importances,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # Model-specific attributes
    if hasattr(model, "get_depth"):
        metrics["tree_depth"] = model.get_depth()
    if hasattr(model, "get_n_leaves"):
        metrics["n_leaves"] = model.get_n_leaves()
    if hasattr(model, "n_estimators"):
        metrics["n_estimators"] = model.n_estimators

    return metrics


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgboost",
    cv_folds: int = 5,
    max_depth: int = 5,
    min_samples_leaf: int = 10,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    class_weight: str = "balanced",
    random_state: int = 42,
    scoring: str = "roc_auc",
    **kwargs,
) -> Dict:
    """Perform cross-validation with multiple metrics."""

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]

    base_model = _create_model(
        model_type=model_type,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        class_weight=class_weight,
        random_state=random_state,
        **kwargs,
    )

    if base_model is None:
        return {"error": f"Model type '{model_type}' not available"}

    results = {}
    for metric in scoring_metrics:
        scores = cross_val_score(base_model, X, y, cv=cv, scoring=metric, n_jobs=-1)
        results[f"cv_{metric}_mean"] = scores.mean()
        results[f"cv_{metric}_std"] = scores.std()
        results[f"cv_{metric}_scores"] = scores.tolist()

    results["cv_folds"] = cv_folds
    return results


def hyperparameter_tune(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgboost",
    param_grid: Optional[Dict] = None,
    cv_folds: int = 3,
    n_iter: int = 20,
    random_state: int = 42,
    scoring: str = "roc_auc",
    method: str = "random",
) -> Tuple[Any, Dict]:
    """
    Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

    Returns:
        (best_model, best_params_and_scores)
    """
    base_model = _create_model(
        model_type=model_type,
        max_depth=5,
        min_samples_leaf=10,
        n_estimators=100,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=random_state,
    )

    if base_model is None:
        return None, {"error": f"Model type '{model_type}' not available"}

    if param_grid is None:
        param_grid = _get_default_param_grid(model_type)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    if method == "grid":
        search = GridSearchCV(base_model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=0)
    else:
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=random_state,
            verbose=0,
        )

    search.fit(X, y)

    results = {
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "best_estimator": search.best_estimator_,
        "cv_results": search.cv_results_,
    }

    return search.best_estimator_, results


def _get_default_param_grid(model_type: str) -> Dict:
    """Get default parameter grid for each model type."""
    if model_type == "xgboost":
        return {
            "max_depth": [3, 4, 5, 6, 7],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "n_estimators": [50, 100, 200, 300],
            "min_child_weight": [1, 5, 10, 20],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "scale_pos_weight": [1, 2, 5, 10],
        }
    elif model_type == "lightgbm":
        return {
            "max_depth": [3, 4, 5, 6, 7, -1],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "n_estimators": [50, 100, 200, 300],
            "min_child_samples": [5, 10, 20, 50],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.1, 1, 10],
            "reg_lambda": [0, 0.1, 1, 10],
        }
    elif model_type == "random_forest":
        return {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_leaf": [1, 2, 5, 10, 20],
            "min_samples_split": [2, 5, 10, 20],
            "max_features": ["sqrt", "log2", None],
        }
    elif model_type == "gradient_boosting":
        return {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [3, 4, 5, 6],
            "min_samples_leaf": [5, 10, 20],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
        }
    else:  # decision_tree
        return {
            "max_depth": [3, 4, 5, 6, 7, 8, 10, None],
            "min_samples_leaf": [1, 2, 5, 10, 20, 50],
            "min_samples_split": [2, 5, 10, 20],
        }


def get_shap_explanations(
    model: Any,
    X: pd.DataFrame,
    sample_size: int = 100,
    random_state: int = 42,
) -> Dict:
    """
    Generate SHAP explanations for tree-based models.

    Returns:
        Dict with shap_values, feature_importance, and plots data
    """
    if not SHAP_AVAILABLE:
        return {"error": "SHAP not installed. Run: pip install shap"}

    # Sample data for speed
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=random_state)
    else:
        X_sample = X

    try:
        # Create SHAP explainer
        if hasattr(model, "predict_proba"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # For binary classification, shap_values is list of two arrays
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
        else:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample).values
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]  # Positive class

        # Feature importance (mean absolute SHAP)
        feature_importance = pd.DataFrame(
            {
                "feature": X_sample.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                "mean_shap": shap_values.mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)

        return {
            "shap_values": shap_values,
            "X_sample": X_sample,
            "feature_importance": feature_importance,
            "explainer": explainer,
        }
    except Exception as e:
        return {"error": f"SHAP computation failed: {str(e)}"}


def get_shap_dependence_data(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: str = None,
) -> pd.DataFrame:
    """Get data for SHAP dependence plot."""
    feat_idx = X.columns.get_loc(feature)
    shap_feat = shap_values[:, feat_idx]

    df = pd.DataFrame(
        {
            "feature_value": X[feature].values,
            "shap_value": shap_feat,
        }
    )

    if interaction_feature and interaction_feature in X.columns:
        df["interaction_value"] = X[interaction_feature].values

    return df


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    model_types: List[str] = None,
    cv_folds: int = 5,
    **model_kwargs,
) -> pd.DataFrame:
    """Compare multiple model types using cross-validation."""
    if model_types is None:
        available = get_available_models()
        model_types = [k for k, v in available.items() if v and k != "shap"]

    results = []
    for model_type in model_types:
        try:
            cv_results = cross_validate_model(
                X, y, model_type=model_type, cv_folds=cv_folds, **model_kwargs
            )
            row = {"model": model_type}
            for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]:
                row[f"cv_{metric}_mean"] = cv_results.get(f"cv_{metric}_mean", np.nan)
                row[f"cv_{metric}_std"] = cv_results.get(f"cv_{metric}_std", np.nan)
            results.append(row)
        except Exception as e:
            results.append({"model": model_type, "error": str(e)})

    return pd.DataFrame(results)


def extract_tree_rules(
    model: Any,
    feature_names: List[str],
    target_names: List[str] = ["Not Buy", "Buy"],
    max_depth: int = None,
) -> List[Dict]:
    """Extract rules from tree-based models (supports DecisionTree, RF, GB, XGB, LGBM)."""
    if model is None:
        return []

    rules = []

    # Handle ensemble methods - extract from individual trees
    if hasattr(model, "estimators_"):
        # Random Forest / Gradient Boosting
        estimators = model.estimators_
        if hasattr(estimators, "ravel"):
            estimators = estimators.ravel()

        for i, estimator in enumerate(estimators[:10]):  # Limit to first 10 trees
            tree_rules = _extract_single_tree_rules(
                estimator, feature_names, target_names, max_depth
            )
            for rule in tree_rules:
                rule["tree_index"] = i
            rules.extend(tree_rules)

    elif hasattr(model, "get_booster"):
        # XGBoost / LightGBM
        if hasattr(model, "get_booster"):  # XGBoost
            model.get_booster().trees_to_dataframe()
        else:  # LightGBM
            model.booster_.trees_to_dataframe()

        # Parse tree structure (simplified)
        pass

    else:
        # Single decision tree
        rules = _extract_single_tree_rules(model, feature_names, target_names, max_depth)

    return rules


def _extract_single_tree_rules(
    model: DecisionTreeClassifier,
    feature_names: List[str],
    target_names: List[str],
    max_depth: int = None,
) -> List[Dict]:
    """Extract rules from a single decision tree."""
    tree = model.tree_
    rules = []

    def recurse(node, depth, path_conditions):
        if max_depth is not None and depth > max_depth:
            return

        if tree.feature[node] != -2:  # Not a leaf
            feature_idx = tree.feature[node]
            threshold = tree.threshold[node]
            feature_name = feature_names[feature_idx]

            # Left child (<= threshold)
            left_path = path_conditions + [f"{feature_name} <= {threshold:.2f}"]
            recurse(tree.children_left[node], depth + 1, left_path)

            # Right child (> threshold)
            right_path = path_conditions + [f"{feature_name} > {threshold:.2f}"]
            recurse(tree.children_right[node], depth + 1, right_path)
        else:
            # Leaf node
            n_samples = tree.n_node_samples[node]
            value = tree.value[node][0]
            total = value.sum()
            probs = value / total if total > 0 else [0, 0]
            predicted_class = np.argmax(value)

            rules.append(
                {
                    "conditions": path_conditions,
                    "prediction": target_names[predicted_class],
                    "probability": probs[predicted_class],
                    "class_distribution": dict(zip(target_names, value.astype(int))),
                    "samples": n_samples,
                    "purity": max(probs),
                }
            )

    recurse(0, 0, [])
    return rules


def predict_with_explanation(
    model: Any,
    features: pd.DataFrame,
    customer_id: str,
    shap_explainer: Any = None,
) -> Dict:
    """Get prediction with full explanation for a customer."""
    if model is None or customer_id not in features.index:
        return {"error": "Model not available or customer not found"}

    cust_features = features.loc[[customer_id]]
    prediction = model.predict(cust_features)[0]
    probability = (
        model.predict_proba(cust_features)[0] if hasattr(model, "predict_proba") else [0, 0]
    )

    # Decision path for tree models
    path_conditions = []
    if hasattr(model, "decision_path"):
        node_indicator = model.decision_path(cust_features)
        leaf_id = model.apply(cust_features)[0]
        feature_names = features.columns.tolist()

        for node_id in node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]:
            if model.tree_.feature[node_id] != -2:
                feature_idx = model.tree_.feature[node_id]
                threshold = model.tree_.threshold[node_id]
                feature_name = feature_names[feature_idx]
                value = cust_features.iloc[0, feature_idx]
                if value <= threshold:
                    path_conditions.append(
                        f"{feature_name} <= {threshold:.2f} (value: {value:.2f})"
                    )
                else:
                    path_conditions.append(f"{feature_name} > {threshold:.2f} (value: {value:.2f})")

    # SHAP explanation if available
    shap_explanation = None
    if shap_explainer is not None and SHAP_AVAILABLE:
        try:
            shap_vals = shap_explainer.shap_values(cust_features)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # Positive class
            shap_explanation = pd.DataFrame(
                {
                    "feature": features.columns,
                    "shap_value": shap_vals[0],
                    "feature_value": cust_features.iloc[0].values,
                }
            ).sort_values("shap_value", key=abs, ascending=False)
        except Exception:
            pass

    return {
        "customer_id": customer_id,
        "prediction": "Buy" if prediction == 1 else "Not Buy",
        "probability_buy": probability[1] if len(probability) > 1 else probability[0],
        "probability_not_buy": probability[0] if len(probability) > 1 else 1 - probability[0],
        "decision_path": path_conditions,
        "leaf_id": leaf_id if "leaf_id" in locals() else None,
        "shap_explanation": shap_explanation,
    }


def get_model_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """Get feature importances from any supported model type."""
    if model is None:
        return pd.DataFrame()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)


def calibrate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "isotonic",
    cv: int = 3,
) -> Any:
    """Calibrate model probabilities."""
    calibrated = CalibratedClassifierCV(model, method=method, cv=cv)
    calibrated.fit(X_train, y_train)
    return calibrated


# Convenience functions for specific model types
def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    max_depth: int = 5,
    min_samples_leaf: int = 10,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    class_weight: str = "balanced",
    random_state: int = 42,
    calibrate: bool = False,
    **kwargs,
) -> Tuple[Any, Dict]:
    """Train XGBoost model with default parameters."""
    return train_model(
        X,
        y,
        model_type="xgboost",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        class_weight=class_weight,
        random_state=random_state,
        calibrate=calibrate,
        **kwargs,
    )


def train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    max_depth: int = 5,
    min_samples_leaf: int = 10,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    class_weight: str = "balanced",
    random_state: int = 42,
    calibrate: bool = False,
    **kwargs,
) -> Tuple[Any, Dict]:
    """Train LightGBM model with default parameters."""
    return train_model(
        X,
        y,
        model_type="lightgbm",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        class_weight=class_weight,
        random_state=random_state,
        calibrate=calibrate,
        **kwargs,
    )


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    max_depth: int = 5,
    min_samples_leaf: int = 10,
    n_estimators: int = 100,
    class_weight: str = "balanced",
    random_state: int = 42,
    calibrate: bool = False,
    **kwargs,
) -> Tuple[Any, Dict]:
    """Train Random Forest model with default parameters."""
    return train_model(
        X,
        y,
        model_type="random_forest",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_state,
        calibrate=calibrate,
        **kwargs,
    )


def train_gradient_boosting(
    X: pd.DataFrame,
    y: pd.Series,
    max_depth: int = 5,
    min_samples_leaf: int = 10,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    random_state: int = 42,
    calibrate: bool = False,
    **kwargs,
) -> Tuple[Any, Dict]:
    """Train Gradient Boosting model with default parameters."""
    return train_model(
        X,
        y,
        model_type="gradient_boosting",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
        calibrate=calibrate,
        **kwargs,
    )


# Alias for hyperparameter tuning
tune_hyperparameters = hyperparameter_tune
