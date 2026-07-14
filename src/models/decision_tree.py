"""Decision tree model for product purchase prediction with advanced feature engineering."""

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


def build_customer_features(
    transactions_df: pd.DataFrame,
    target_product: str,
    prediction_window_days: int = 30,
    min_history_days: int = 60,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build enhanced features for each customer to predict purchase of target_product.

    Customer Features:
    - RFM (Recency, Frequency, Monetary) + extensions
    - Product purchase history (top products one-hot)
    - Category/brand preferences
    - Temporal patterns (seasonality, day-of-week, intervals)
    - Basket diversity & composition
    - Price sensitivity
    - Customer lifetime value proxies
    - Churn risk indicators
    - Product affinity scores

    Product Features (for target product):
    - Popularity rank
    - Category/brand stats
    - Price tier
    - Co-purchase patterns

    Interaction Features:
    - Customer-product affinity
    - Historical purchase probability
    - Cross-sell potential

    Target:
    - Binary: Will customer buy target_product in prediction window?
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # Determine cutoff dates
    max_date = df["date"].max()
    prediction_start = max_date - pd.Timedelta(days=prediction_window_days)
    history_start = prediction_start - pd.Timedelta(days=min_history_days)

    # Split data
    history_df = df[(df["date"] >= history_start) & (df["date"] < prediction_start)].copy()
    future_df = df[df["date"] >= prediction_start].copy()

    # Get all customers from history
    all_customers = history_df["customer_id"].unique()

    # Target: did customer buy target_product in future period?
    future_target = future_df[future_df["stockcode"] == target_product]["customer_id"].unique()
    y = pd.Series(0, index=all_customers, name="will_buy")
    y.loc[y.index.intersection(future_target)] = 1

    # ============================================================
    # BUILD FEATURES
    # ============================================================
    features = pd.DataFrame(index=all_customers)

    # ------------------------------------------------------------
    # 1. ENHANCED RFM FEATURES
    # ------------------------------------------------------------
    rfm = history_df.groupby("customer_id").agg(
        # Recency
        recency_days=("date", lambda x: (prediction_start - x.max()).days),
        # Frequency
        frequency=("transaction_id", "nunique"),
        n_orders=("transaction_id", "nunique"),
        n_items_total=("quantity", "sum"),
        # Monetary
        monetary=("revenue", "sum"),
        avg_order_value=("revenue", "mean"),
        max_order_value=("revenue", "max"),
        min_order_value=("revenue", "min"),
        std_order_value=("revenue", "std"),
        # Product diversity
        n_unique_products=("stockcode", "nunique"),
        n_unique_categories=(
            ("category", "nunique")
            if "category" in history_df.columns
            else ("stockcode", "nunique")
        ),
        n_unique_brands=(
            ("brand", "nunique") if "brand" in history_df.columns else ("stockcode", "nunique")
        ),
        # Temporal
        first_purchase=("date", "min"),
        last_purchase=("date", "max"),
        # Price stats
        avg_price_paid=("price", "mean"),
        max_price_paid=("price", "max"),
        min_price_paid=("price", "min"),
        # Quantity stats
        avg_quantity=("quantity", "mean"),
        max_quantity=("quantity", "max"),
    )

    # Derived RFM features
    rfm["customer_lifetime_days"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days
    rfm["purchase_interval"] = np.where(
        rfm["frequency"] > 1,
        rfm["customer_lifetime_days"] / (rfm["frequency"] - 1),
        rfm["customer_lifetime_days"],
    )
    rfm["avg_items_per_order"] = rfm["n_items_total"] / rfm["frequency"]
    rfm["revenue_per_item"] = rfm["monetary"] / rfm["n_items_total"].replace(0, np.nan)
    rfm["order_value_cv"] = rfm["std_order_value"] / rfm["avg_order_value"].replace(
        0, np.nan
    )  # coefficient of variation

    # Recency segments
    rfm["is_recent_buyer"] = (rfm["recency_days"] <= 7).astype(int)
    rfm["is_lapsing"] = (rfm["recency_days"] > rfm["purchase_interval"]).astype(int)

    features = features.join(rfm)

    # ------------------------------------------------------------
    # 2. PRODUCT PURCHASE HISTORY (Top N products one-hot)
    # ------------------------------------------------------------
    top_products = history_df["stockcode"].value_counts().head(50).index.tolist()

    if target_product in top_products:
        top_products.remove(target_product)  # Don't leak target

    product_purchases = pd.crosstab(history_df["customer_id"], history_df["stockcode"])
    product_purchases = product_purchases.reindex(columns=top_products, fill_value=0)
    product_purchases = (product_purchases > 0).astype(int)
    product_purchases.columns = [f"bought_{col}" for col in product_purchases.columns]

    # Also add purchase counts for top products
    product_counts = pd.crosstab(history_df["customer_id"], history_df["stockcode"])
    product_counts = product_counts.reindex(columns=top_products, fill_value=0)
    product_counts.columns = [f"count_{col}" for col in product_counts.columns]

    features = features.join(product_purchases, how="left").fillna(0)
    features = features.join(product_counts, how="left").fillna(0)

    # ------------------------------------------------------------
    # 3. CATEGORY/BRAND PREFERENCES
    # ------------------------------------------------------------
    if "category" in history_df.columns:
        cat_purchases = pd.crosstab(history_df["customer_id"], history_df["category"])
        cat_purchases = (cat_purchases > 0).astype(int)
        cat_purchases.columns = [f"cat_{col}" for col in cat_purchases.columns]

        # Category spend share
        cat_revenue = (
            history_df.groupby(["customer_id", "category"])["revenue"].sum().unstack(fill_value=0)
        )
        cat_revenue.columns = [f"cat_rev_{col}" for col in cat_revenue.columns]
        cat_total = cat_revenue.sum(axis=1)
        cat_share = cat_revenue.div(cat_total, axis=0).fillna(0)
        cat_share.columns = [f"cat_share_{col}" for col in cat_share.columns]

        features = features.join(cat_purchases, how="left").fillna(0)
        features = features.join(cat_share, how="left").fillna(0)

    if "brand" in history_df.columns:
        brand_purchases = pd.crosstab(history_df["customer_id"], history_df["brand"])
        brand_purchases = (brand_purchases > 0).astype(int)
        brand_purchases.columns = [f"brand_{col}" for col in brand_purchases.columns]

        brand_revenue = (
            history_df.groupby(["customer_id", "brand"])["revenue"].sum().unstack(fill_value=0)
        )
        brand_revenue.columns = [f"brand_rev_{col}" for col in brand_revenue.columns]
        brand_total = brand_revenue.sum(axis=1)
        brand_share = brand_revenue.div(brand_total, axis=0).fillna(0)
        brand_share.columns = [f"brand_share_{col}" for col in brand_share.columns]

        features = features.join(brand_purchases, how="left").fillna(0)
        features = features.join(brand_share, how="left").fillna(0)

    # ------------------------------------------------------------
    # 4. TEMPORAL PATTERNS
    # ------------------------------------------------------------
    # Day of week patterns
    dow_dist = (
        history_df.groupby(["customer_id", history_df["date"].dt.dayofweek])
        .size()
        .unstack(fill_value=0)
    )
    dow_dist.columns = [f"dow_{col}" for col in dow_dist.columns]
    dow_total = dow_dist.sum(axis=1)
    dow_share = dow_dist.div(dow_total, axis=0).fillna(0)
    dow_share.columns = [f"dow_share_{col}" for col in dow_share.columns]
    features = features.join(dow_share, how="left").fillna(0)

    # Month patterns
    month_dist = (
        history_df.groupby(["customer_id", history_df["date"].dt.month])
        .size()
        .unstack(fill_value=0)
    )
    month_dist.columns = [f"month_{col}" for col in month_dist.columns]
    month_total = month_dist.sum(axis=1)
    month_share = month_dist.div(month_total, axis=0).fillna(0)
    month_share.columns = [f"month_share_{col}" for col in month_share.columns]
    features = features.join(month_share, how="left").fillna(0)

    # Hour patterns (if time available)
    if history_df["date"].dt.hour.nunique() > 1:
        hour_dist = (
            history_df.groupby(["customer_id", history_df["date"].dt.hour])
            .size()
            .unstack(fill_value=0)
        )
        hour_dist.columns = [f"hour_{col}" for col in hour_dist.columns]
        hour_total = hour_dist.sum(axis=1)
        hour_share = hour_dist.div(hour_total, axis=0).fillna(0)
        hour_share.columns = [f"hour_share_{col}" for col in hour_share.columns]
        features = features.join(hour_share, how="left").fillna(0)

    # Time since first/last purchase
    features["days_since_first_purchase"] = (prediction_start - features["first_purchase"]).dt.days
    features["days_since_last_purchase"] = (prediction_start - features["last_purchase"]).dt.days

    # Purchase regularity (CV of intervals)
    customer_intervals = (
        history_df.sort_values(["customer_id", "date"])
        .groupby("customer_id")["date"]
        .diff()
        .dt.days
    )
    interval_stats = customer_intervals.groupby(history_df["customer_id"]).agg(
        ["mean", "std", "min", "max"]
    )
    interval_stats.columns = [
        "interval_mean",
        "interval_std",
        "interval_min",
        "interval_max",
    ]
    interval_stats["interval_cv"] = interval_stats["interval_std"] / interval_stats[
        "interval_mean"
    ].replace(0, np.nan)
    interval_stats["is_regular"] = (interval_stats["interval_cv"] < 0.5).astype(int)
    features = features.join(interval_stats, how="left").fillna(0)

    # ------------------------------------------------------------
    # 5. BASKET DIVERSITY & COMPOSITION
    # ------------------------------------------------------------
    # Herfindahl-Hirschman Index for product concentration
    product_shares = (
        history_df.groupby(["customer_id", "stockcode"])["quantity"].sum().unstack(fill_value=0)
    )
    product_shares = product_shares.div(product_shares.sum(axis=1), axis=0).fillna(0)
    hhi = (product_shares**2).sum(axis=1)
    features["product_hhi"] = hhi
    features["effective_products"] = 1 / hhi.replace(0, np.nan)

    # Category HHI
    if "category" in history_df.columns:
        cat_shares = (
            history_df.groupby(["customer_id", "category"])["quantity"].sum().unstack(fill_value=0)
        )
        cat_shares = cat_shares.div(cat_shares.sum(axis=1), axis=0).fillna(0)
        cat_hhi = (cat_shares**2).sum(axis=1)
        features["category_hhi"] = cat_hhi
        features["effective_categories"] = 1 / cat_hhi.replace(0, np.nan)

    # Average basket size per transaction
    basket_sizes = history_df.groupby(["customer_id", "transaction_id"])["quantity"].sum()
    features["avg_basket_size"] = basket_sizes.groupby("customer_id").mean()
    features["max_basket_size"] = basket_sizes.groupby("customer_id").max()
    features["basket_size_std"] = basket_sizes.groupby("customer_id").std().fillna(0)

    # ------------------------------------------------------------
    # 6. PRICE SENSITIVITY
    # ------------------------------------------------------------
    # Price tier preferences
    price_tiers = pd.qcut(
        history_df["price"],
        q=4,
        labels=["budget", "value", "premium", "luxury"],
        duplicates="drop",
    )
    tier_dist = history_df.groupby(["customer_id", price_tiers]).size().unstack(fill_value=0)
    tier_dist.columns = [f"tier_{col}" for col in tier_dist.columns]
    tier_total = tier_dist.sum(axis=1)
    tier_share = tier_dist.div(tier_total, axis=0).fillna(0)
    tier_share.columns = [f"tier_share_{col}" for col in tier_share.columns]
    features = features.join(tier_share, how="left").fillna(0)

    # Price elasticity proxy: correlation between price and quantity
    price_qty_corr = history_df.groupby("customer_id").apply(
        lambda x: x["price"].corr(x["quantity"]) if len(x) > 2 and x["price"].std() > 0 else 0
    )
    features["price_quantity_corr"] = price_qty_corr.fillna(0)

    # Avg discount/promotion sensitivity (if multiple prices for same product)
    if len(history_df) > 0:
        price_variation = (
            history_df.groupby(["customer_id", "stockcode"])["price"].std().unstack(fill_value=0)
        )
        features["avg_price_variation"] = price_variation.mean(axis=1)
        features["max_price_variation"] = price_variation.max(axis=1)

    # ------------------------------------------------------------
    # 7. CUSTOMER LIFETIME VALUE PROXIES
    # ------------------------------------------------------------
    features["estimated_annual_revenue"] = features["monetary"] / (
        features["customer_lifetime_days"] / 365
    ).replace(0, np.nan)
    features["estimated_annual_orders"] = features["frequency"] / (
        features["customer_lifetime_days"] / 365
    ).replace(0, np.nan)
    features["predicted_clv"] = features["estimated_annual_revenue"] * 3  # 3-year horizon

    # ------------------------------------------------------------
    # 8. CHURN RISK INDICATORS
    # ------------------------------------------------------------
    features["recency_to_interval_ratio"] = features["recency_days"] / features[
        "purchase_interval"
    ].replace(0, np.nan)
    features["is_at_risk"] = (features["recency_to_interval_ratio"] > 2).astype(int)
    features["is_churned"] = (features["recency_to_interval_ratio"] > 4).astype(int)

    # Velocity change (recent vs historical frequency)
    recent_cutoff = prediction_start - pd.Timedelta(days=30)
    recent_freq = (
        history_df[history_df["date"] >= recent_cutoff]
        .groupby("customer_id")["transaction_id"]
        .nunique()
    )
    features["recent_frequency"] = recent_freq.reindex(features.index).fillna(0)
    monthly_freq = features["frequency"] / (features["customer_lifetime_days"] / 30).clip(lower=1)
    features["freq_velocity"] = features["recent_frequency"] / monthly_freq.replace(0, np.nan)

    # ------------------------------------------------------------
    # 9. PRODUCT-SPECIFIC FEATURES (Target Product Context)
    # ------------------------------------------------------------
    # Global product stats
    product_stats = df[df["stockcode"] == target_product]
    if len(product_stats) > 0:
        features["target_product_popularity"] = len(product_stats) / df["transaction_id"].nunique()
        features["target_product_avg_price"] = product_stats["price"].mean()
        features["target_product_avg_qty"] = product_stats["quantity"].mean()
        features["target_product_revenue_share"] = (
            product_stats["revenue"].sum() / df["revenue"].sum()
        )

        # Category/brand of target
        if "category" in product_stats.columns:
            target_cat = product_stats["category"].iloc[0]
            features["target_category"] = target_cat
        if "brand" in product_stats.columns:
            target_brand = product_stats["brand"].iloc[0]
            features["target_brand"] = target_brand

    # Customer's affinity to target product's category/brand
    if "category" in history_df.columns and "target_category" in features.columns:
        cat_affinity = (
            history_df[history_df["category"] == features["target_category"].iloc[0]]
            .groupby("customer_id")["revenue"]
            .sum()
        )
        features["target_category_affinity"] = cat_affinity.reindex(features.index).fillna(0)

    if "brand" in history_df.columns and "target_brand" in features.columns:
        brand_affinity = (
            history_df[history_df["brand"] == features["target_brand"].iloc[0]]
            .groupby("customer_id")["revenue"]
            .sum()
        )
        features["target_brand_affinity"] = brand_affinity.reindex(features.index).fillna(0)

    # ------------------------------------------------------------
    # 10. CO-PURCHASE / CROSS-SELL FEATURES
    # ------------------------------------------------------------
    # Products frequently bought with target
    target_transactions = history_df[history_df["stockcode"] == target_product][
        "transaction_id"
    ].unique()
    if len(target_transactions) > 0:
        co_purchase = history_df[
            (history_df["transaction_id"].isin(target_transactions))
            & (history_df["stockcode"] != target_product)
        ]["stockcode"].value_counts()
        top_copurchase = co_purchase.head(10).index.tolist()

        for coprod in top_copurchase:
            coprod_bought = history_df[
                (history_df["customer_id"].isin(features.index))
                & (history_df["stockcode"] == coprod)
            ]["customer_id"].unique()
            features[f"bought_with_target_{coprod}"] = features.index.isin(coprod_bought).astype(
                int
            )

    # ------------------------------------------------------------
    # 11. CUSTOMER-PRODUCT INTERACTION FEATURES
    # ------------------------------------------------------------
    # Historical purchase probability for target product
    target_history = (
        history_df[history_df["stockcode"] == target_product].groupby("customer_id").size()
    )
    features["target_purchase_count"] = target_history.reindex(features.index).fillna(0)
    features["target_purchase_freq"] = features["target_purchase_count"] / features[
        "frequency"
    ].replace(0, np.nan)
    features["has_bought_target_before"] = (features["target_purchase_count"] > 0).astype(int)

    # Days since last target purchase
    last_target_purchase = (
        history_df[history_df["stockcode"] == target_product].groupby("customer_id")["date"].max()
    )
    features["days_since_target_purchase"] = (
        prediction_start - last_target_purchase
    ).dt.days.reindex(features.index)

    # ------------------------------------------------------------
    # 12. SEQUENTIAL PATTERNS
    # ------------------------------------------------------------
    # First/last product purchased
    first_product = history_df.sort_values("date").groupby("customer_id")["stockcode"].first()
    last_product = history_df.sort_values("date").groupby("customer_id")["stockcode"].last()
    features["first_product"] = first_product
    features["last_product"] = last_product
    features["first_is_target"] = (features["first_product"] == target_product).astype(int)
    features["last_is_target"] = (features["last_product"] == target_product).astype(int)

    # Drop raw date columns and non-numeric
    drop_cols = ["first_purchase", "last_purchase", "first_product", "last_product"]
    if "target_category" in features.columns:
        drop_cols.append("target_category")
    if "target_brand" in features.columns:
        drop_cols.append("target_brand")
    features = features.drop(columns=drop_cols, errors="ignore")

    # Convert any remaining object columns to numeric
    for col in features.select_dtypes(include=["object"]).columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    # Fill NaN
    features = features.fillna(0)

    # Replace inf with large numbers
    features = features.replace([np.inf, -np.inf], 1e10)

    # Ensure target aligns
    y = y.reindex(features.index).fillna(0).astype(int)

    return features, y


def train_decision_tree(
    X: pd.DataFrame,
    y: pd.Series,
    max_depth: int = 5,
    min_samples_leaf: int = 10,
    min_samples_split: int = 20,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> Tuple[DecisionTreeClassifier, Dict]:
    """Train decision tree classifier with enhanced metrics."""
    if len(y.unique()) < 2:
        return None, {"error": "Only one class present in target"}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Comprehensive metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "train_accuracy": model.score(X_train, y_train),
        "test_accuracy": model.score(X_test, y_test),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_test, y_prob) if len(y_test.unique()) > 1 else 0.5,
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "positive_class_rate": y.mean(),
        "feature_importances": dict(zip(X.columns, model.feature_importances_)),
        "tree_depth": model.get_depth(),
        "n_leaves": model.get_n_leaves(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return model, metrics


def extract_tree_rules(
    model: DecisionTreeClassifier,
    feature_names: List[str],
    target_names: List[str] = ["Not Buy", "Buy"],
) -> List[Dict]:
    """Extract human-readable rules from decision tree paths."""
    if model is None:
        return []

    tree = model.tree_
    rules = []

    def recurse(node, depth, path_conditions):
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


def get_top_features(
    model: DecisionTreeClassifier, feature_names: List[str], n: int = 20
) -> pd.DataFrame:
    """Get top N most important features."""
    if model is None:
        return pd.DataFrame()

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:n]

    return pd.DataFrame(
        {
            "feature": [feature_names[i] for i in indices],
            "importance": [importances[i] for i in indices],
        }
    )


def predict_for_customer(
    model: DecisionTreeClassifier, features: pd.DataFrame, customer_id: str
) -> Dict:
    """Get prediction and explanation for a specific customer."""
    if model is None or customer_id not in features.index:
        return {"error": "Model not available or customer not found"}

    cust_features = features.loc[[customer_id]]
    prediction = model.predict(cust_features)[0]
    probability = model.predict_proba(cust_features)[0]

    # Get decision path
    node_indicator = model.decision_path(cust_features)
    leaf_id = model.apply(cust_features)[0]

    # Extract path conditions
    feature_names = features.columns.tolist()
    path_conditions = []

    for node_id in node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]:
        if model.tree_.feature[node_id] != -2:
            feature_idx = model.tree_.feature[node_id]
            threshold = model.tree_.threshold[node_id]
            feature_name = feature_names[feature_idx]

            value = cust_features.iloc[0, feature_idx]
            if value <= threshold:
                path_conditions.append(f"{feature_name} <= {threshold:.2f} (value: {value:.2f})")
            else:
                path_conditions.append(f"{feature_name} > {threshold:.2f} (value: {value:.2f})")

    return {
        "customer_id": customer_id,
        "prediction": "Buy" if prediction == 1 else "Not Buy",
        "probability_buy": probability[1],
        "probability_not_buy": probability[0],
        "decision_path": path_conditions,
        "leaf_id": leaf_id,
    }


def get_feature_groups(features: pd.DataFrame) -> Dict[str, List[str]]:
    """Group features by type for interpretability."""
    groups = {
        "RFM": [],
        "Product_History": [],
        "Category_Brand": [],
        "Temporal": [],
        "Basket_Diversity": [],
        "Price_Sensitivity": [],
        "CLV_Churn": [],
        "Product_Specific": [],
        "CoPurchase": [],
        "Interaction": [],
        "Sequential": [],
    }

    for col in features.columns:
        if any(
            prefix in col
            for prefix in [
                "recency",
                "frequency",
                "monetary",
                "avg_order",
                "max_order",
                "min_order",
                "std_order",
                "n_unique",
                "customer_lifetime",
                "purchase_interval",
                "avg_items",
                "revenue_per",
                "order_value_cv",
                "is_recent",
                "is_lapsing",
            ]
        ):
            groups["RFM"].append(col)
        elif col.startswith("bought_") or col.startswith("count_"):
            groups["Product_History"].append(col)
        elif col.startswith("cat_") or col.startswith("brand_"):
            groups["Category_Brand"].append(col)
        elif any(
            prefix in col
            for prefix in [
                "dow_",
                "month_",
                "hour_",
                "days_since",
                "interval_",
                "is_regular",
            ]
        ):
            groups["Temporal"].append(col)
        elif any(kw in col for kw in ["hhi", "effective_", "basket_size"]):
            groups["Basket_Diversity"].append(col)
        elif any(kw in col for kw in ["tier_", "price_quantity", "price_variation"]):
            groups["Price_Sensitivity"].append(col)
        elif any(
            kw in col
            for kw in [
                "annual_",
                "predicted_clv",
                "recency_to_interval",
                "is_at_risk",
                "is_churned",
                "freq_velocity",
            ]
        ):
            groups["CLV_Churn"].append(col)
        elif any(
            kw in col
            for kw in [
                "target_product",
                "target_category",
                "target_brand",
                "target_category_affinity",
                "target_brand_affinity",
            ]
        ):
            groups["Product_Specific"].append(col)
        elif "bought_with_target" in col:
            groups["CoPurchase"].append(col)
        elif any(kw in col for kw in ["target_purchase", "has_bought_target", "days_since_target"]):
            groups["Interaction"].append(col)
        elif any(kw in col for kw in ["first_is_target", "last_is_target"]):
            groups["Sequential"].append(col)
        else:
            # Default to RFM
            groups["RFM"].append(col)

    # Remove empty groups
    return {k: v for k, v in groups.items() if v}


def get_feature_importance_by_group(
    model: DecisionTreeClassifier, features: pd.DataFrame
) -> pd.DataFrame:
    """Get aggregated feature importance by feature group."""
    groups = get_feature_groups(features)
    importances = dict(zip(features.columns, model.feature_importances_))

    group_importance = {}
    for group, cols in groups.items():
        group_importance[group] = sum(importances.get(c, 0) for c in cols)

    return (
        pd.DataFrame(
            {
                "group": list(group_importance.keys()),
                "importance": list(group_importance.values()),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
