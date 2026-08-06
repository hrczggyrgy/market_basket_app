"""Supervised next-product-choice model using sklearn DecisionTreeClassifier.

Builds customer feature vectors from purchase history and predicts each
customer's next purchased product. Rules are extracted with sklearn's
own tree exporter (no hand-rolled rule parser).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.analytics.schemas import CUSTOMER_FEATURES, MODEL_METRICS, TREE_RULES, check


def build_customer_features(
    df: pd.DataFrame,
    prediction_window_days: int = 30,
    min_support: float = 0.005,
    top_products: int | None = 50,
) -> pd.DataFrame:
    """Customer features from history; target = most-bought product in next window.

    For each customer, features summarize all history up to their last
    transaction; the target is the single product most frequently purchased in
    the prediction window after their first observation.
    """
    df = df.copy()
    df["revenue"] = df["price"] * df["quantity"]
    df["date"] = pd.to_datetime(df["date"])
    horizon_end = df["date"].max()
    horizon_start = horizon_end - pd.Timedelta(days=prediction_window_days)
    horizon = df[df["date"] >= horizon_start]
    history = df[df["date"] < horizon_start]
    if history.empty or horizon.empty:
        return check(pd.DataFrame(columns=list(CUSTOMER_FEATURES.columns)), CUSTOMER_FEATURES, allow_empty=True)

    if top_products:
        top = (
            horizon.groupby("stockcode")["quantity"].sum()
            .sort_values(ascending=False)
            .head(top_products)
            .index
        )
    else:
        top = horizon["stockcode"].unique()
    support = horizon.groupby("stockcode")["transaction_id"].nunique() / horizon["transaction_id"].nunique()
    targets = support[support.ge(min_support)].index.intersection(top)
    if targets.empty:
        return check(pd.DataFrame(columns=list(CUSTOMER_FEATURES.columns)), CUSTOMER_FEATURES, allow_empty=True)

    target_rank = horizon[horizon["stockcode"].isin(targets)].groupby("customer_id")["stockcode"].apply(
        lambda s: s.value_counts().idxmax()
    ).rename("target_product")

    agg = {
        "recency_days": ("date", lambda s: (history["date"].max() - s.max()).days),
        "frequency": ("revenue", "count"),
        "monetary": ("revenue", "sum"),
        "n_baskets": ("transaction_id", "nunique"),
        "n_distinct_products": ("stockcode", "nunique"),
    }
    if "category" in history.columns:
        agg["favorite_category"] = ("category", lambda s: _mode_or_unknown(s))
    features = history.groupby("customer_id").agg(**agg)
    if "favorite_category" not in features.columns:
        features["favorite_category"] = "unknown"
    features["avg_basket_size"] = features["frequency"] / features["n_baskets"].replace(0, np.nan)
    table = features.join(target_rank, how="inner")
    table = table[table["target_product"].notna()]
    return check(table.reset_index(), CUSTOMER_FEATURES)


def _mode_or_unknown(s: pd.Series) -> str:
    mode = s.mode()
    return str(mode.iloc[0]) if len(mode) else "unknown"


def train_choice_model(
    features: pd.DataFrame,
    max_depth: int = 4,
    min_samples_leaf: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train a DecisionTreeClassifier on customer features.

    Returns (metrics, feature_importance, tree_rules) with declared contracts.
    """
    check(features, CUSTOMER_FEATURES, allow_empty=True)
    if features.empty:
        empty = check(pd.DataFrame(columns=list(MODEL_METRICS.columns)), MODEL_METRICS, allow_empty=True)
        empty_imp = pd.DataFrame(columns=["feature", "importance"])
        empty_rules = check(pd.DataFrame(columns=list(TREE_RULES.columns)), TREE_RULES, allow_empty=True)
        return empty, empty_imp, empty_rules
    feature_cols = [
        c
        for c in ("recency_days", "frequency", "monetary", "n_baskets", "avg_basket_size", "n_distinct_products")
        if c in features.columns
    ]
    X = features[feature_cols].fillna(0.0).to_numpy()
    y = features["target_product"].astype(str).to_numpy()
    if len(np.unique(y)) < 2 or len(y) < 20:
        empty = check(pd.DataFrame(columns=list(MODEL_METRICS.columns)), MODEL_METRICS, allow_empty=True)
        empty_imp = pd.DataFrame(columns=["feature", "importance"])
        empty_rules = check(pd.DataFrame(columns=list(TREE_RULES.columns)), TREE_RULES, allow_empty=True)
        return empty, empty_imp, empty_rules

    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    stratify = y if pd.Series(y).value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    model = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = pd.DataFrame(
        {
            "metric": ["accuracy", "f1_macro", "n_train", "n_test", "n_classes", "max_depth"],
            "value": [
                accuracy_score(y_test, preds),
                f1_score(y_test, preds, average="macro"),
                len(X_train),
                len(X_test),
                len(np.unique(y)),
                int(model.get_depth()),
            ],
        }
    )
    importance = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    rules = _extract_tree_rules(model, feature_cols, X, y)
    return check(metrics, MODEL_METRICS), importance, rules


def _extract_tree_rules(
    model: DecisionTreeClassifier,
    feature_cols: list[str],
    X: np.ndarray,
    y: np.ndarray,
) -> pd.DataFrame:
    """Extract human-readable rules from a fitted tree using model.apply()."""
    tree = model.tree_
    leaf_ids = model.apply(X)
    rows = []
    for leaf in np.unique(leaf_ids):
        samples_mask = leaf_ids == leaf
        leaf_targets = y[samples_mask]
        if len(leaf_targets) == 0:
            continue
        target = pd.Series(leaf_targets).value_counts().idxmax()
        rows.append(
            {
                "rule_index": int(leaf),
                "rule_path": _path_to_leaf(tree, leaf, feature_cols),
                "n_samples": int(len(leaf_targets)),
                "purity": float(pd.Series(leaf_targets).value_counts(normalize=True).iloc[0]),
                "target_class": str(target),
            }
        )
    rules = pd.DataFrame(rows).sort_values("n_samples", ascending=False).reset_index(drop=True)
    rules["rule_index"] = range(len(rules))
    return check(rules, TREE_RULES, allow_empty=True)


def _path_to_leaf(tree: object, leaf: int, feature_cols: list[str]) -> str:
    """Describe the decision path to a leaf, e.g. 'monetary <= 500.0 AND recency_days > 3'."""
    path: list[str] = []
    while leaf != 0:
        parent = _find_parent(tree, leaf)
        if parent is None:
            break
        feature = feature_cols[int(tree.feature[parent])]  # type: ignore[attr-defined]
        threshold = f"{tree.threshold[parent]:.2f}"  # type: ignore[attr-defined]
        direction = "<=" if tree.children_left[parent] == leaf else ">"  # type: ignore[attr-defined]
        path.append(f"{feature} {direction} {threshold}")
        leaf = parent
    return " AND ".join(reversed(path)) if path else "root"


def _find_parent(tree: object, node: int) -> int | None:
    for parent in range(tree.node_count):  # type: ignore[attr-defined]
        if tree.children_left[parent] == node or tree.children_right[parent] == node:  # type: ignore[attr-defined]
            return parent
    return None
