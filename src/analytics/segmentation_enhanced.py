"""Enhanced Customer Segmentation with CLV Prediction, Survival Analysis, and Ensemble Methods."""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")

# Optional imports
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

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


def compute_rfm_features(
    transactions_df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Compute comprehensive RFM features per customer."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    if snapshot_date is None:
        snapshot_date = df["date"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("customer_id")
        .agg(
            recency_days=("date", lambda x: (snapshot_date - x.max()).days),
            frequency=("transaction_id", "nunique"),
            monetary=("revenue", "sum"),
            avg_order_value=("revenue", "mean"),
            max_order_value=("revenue", "max"),
            n_items=("quantity", "sum"),
            n_unique_products=("stockcode", "nunique"),
            n_unique_categories=(
                ("category", "nunique")
                if "category" in df.columns
                else ("stockcode", "nunique")
            ),
            first_purchase=("date", "min"),
            last_purchase=("date", "max"),
            avg_price_paid=("price", "mean"),
            std_order_value=("revenue", "std"),
        )
        .reset_index()
    )

    # Derived features
    rfm["customer_lifetime_days"] = (
        rfm["last_purchase"] - rfm["first_purchase"]
    ).dt.days
    rfm["purchase_interval"] = np.where(
        rfm["frequency"] > 1,
        rfm["customer_lifetime_days"] / (rfm["frequency"] - 1),
        rfm["customer_lifetime_days"],
    )
    rfm["items_per_order"] = rfm["n_items"] / rfm["frequency"]
    rfm["revenue_per_item"] = rfm["monetary"] / rfm["n_items"].replace(0, np.nan)
    rfm["order_value_cv"] = rfm["std_order_value"] / rfm["avg_order_value"].replace(
        0, np.nan
    )

    # Recency segments
    rfm["recency_segment"] = pd.qcut(
        rfm["recency_days"],
        q=4,
        labels=["Recent", "Active", "Lapsing", "Churned"],
        duplicates="drop",
    )
    rfm["frequency_segment"] = pd.qcut(
        rfm["frequency"].rank(method="first"),
        q=4,
        labels=["Low", "Medium", "High", "Very High"],
        duplicates="drop",
    )
    rfm["monetary_segment"] = pd.qcut(
        rfm["monetary"].rank(method="first"),
        q=4,
        labels=["Low", "Medium", "High", "Very High"],
        duplicates="drop",
    )

    return rfm


def rfm_segmentation(
    rfm_df: pd.DataFrame, method: str = "quantile", n_segments: int = 8
) -> pd.DataFrame:
    """Perform RFM-based customer segmentation."""
    df = rfm_df.copy()

    if method == "quantile":
        # Classic RFM scoring (1-4 per dimension)
        for dim in ["recency_days", "frequency", "monetary"]:
            if dim == "recency_days":
                df[f"{dim}_score"] = pd.qcut(
                    df[dim], q=4, labels=[4, 3, 2, 1], duplicates="drop"
                )
            else:
                df[f"{dim}_score"] = pd.qcut(
                    df[dim].rank(method="first"),
                    q=4,
                    labels=[1, 2, 3, 4],
                    duplicates="drop",
                )

        df["rfm_score"] = (
            df["recency_days_score"].astype(str)
            + df["frequency_score"].astype(str)
            + df["monetary_score"].astype(str)
        )

        # Segment mapping
        segment_map = {
            "444": "Champions",
            "443": "Champions",
            "434": "Champions",
            "344": "Champions",
            "442": "Loyal",
            "433": "Loyal",
            "432": "Loyal",
            "343": "Loyal",
            "334": "Loyal",
            "424": "Potential Loyalists",
            "423": "Potential Loyalists",
            "333": "Potential Loyalists",
            "324": "Potential Loyalists",
            "441": "New Customers",
            "431": "New Customers",
            "422": "New Customers",
            "342": "New Customers",
            "421": "Promising",
            "332": "Promising",
            "323": "Promising",
            "322": "Need Attention",
            "233": "Need Attention",
            "232": "Need Attention",
            "223": "About to Sleep",
            "222": "About to Sleep",
            "133": "About to Sleep",
            "221": "At Risk",
            "212": "At Risk",
            "123": "At Risk",
            "122": "At Risk",
            "211": "Cannot Lose Them",
            "113": "Cannot Lose Them",
            "112": "Cannot Lose Them",
            "111": "Lost",
        }

        df["segment"] = df["rfm_score"].map(segment_map).fillna("Other")

    elif method == "kmeans":
        # K-means clustering on RFM
        features = ["recency_days", "frequency", "monetary"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])

        kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(X_scaled)

        # Label clusters by characteristics
        cluster_labels = {}
        for c in range(n_segments):
            cluster_data = df[df["cluster"] == c]
            avg_rec = cluster_data["recency_days"].mean()
            avg_freq = cluster_data["frequency"].mean()
            avg_mon = cluster_data["monetary"].mean()

            if avg_rec < df["recency_days"].quantile(0.25) and avg_mon > df[
                "monetary"
            ].quantile(0.75):
                label = "High Value"
            elif avg_rec < df["recency_days"].quantile(0.5) and avg_freq > df[
                "frequency"
            ].quantile(0.5):
                label = "Active"
            elif avg_rec > df["recency_days"].quantile(0.75):
                label = "Churned/At Risk"
            elif avg_freq == 1:
                label = "One-time Buyers"
            else:
                label = f"Cluster {c}"
            cluster_labels[c] = label

        df["segment"] = df["cluster"].map(cluster_labels)

    elif method == "gmm":
        # Gaussian Mixture Model for probabilistic segmentation
        features = ["recency_days", "frequency", "monetary"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])

        gmm = GaussianMixture(n_components=n_segments, random_state=42, n_init=10)
        df["cluster"] = gmm.fit_predict(X_scaled)

        # Label clusters
        cluster_labels = {}
        for c in range(n_segments):
            cluster_data = df[df["cluster"] == c]
            avg_rec = cluster_data["recency_days"].mean()
            avg_freq = cluster_data["frequency"].mean()
            avg_mon = cluster_data["monetary"].mean()

            if avg_rec < df["recency_days"].quantile(0.25) and avg_mon > df[
                "monetary"
            ].quantile(0.75):
                label = "High Value"
            elif avg_rec < df["recency_days"].quantile(0.5) and avg_freq > df[
                "frequency"
            ].quantile(0.5):
                label = "Active"
            elif avg_rec > df["recency_days"].quantile(0.75):
                label = "Churned/At Risk"
            elif avg_freq == 1:
                label = "One-time Buyers"
            else:
                label = f"Segment {c}"
            cluster_labels[c] = label

        df["segment"] = df["cluster"].map(cluster_labels)

    return df


def behavioral_segmentation(
    transactions_df: pd.DataFrame, n_clusters: int = 6, method: str = "kmeans"
) -> pd.DataFrame:
    """Behavioral segmentation based on purchase patterns."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # Build behavioral features
    behavioral = (
        df.groupby("customer_id")
        .agg(
            # Temporal
            days_active=("date", lambda x: (x.max() - x.min()).days + 1),
            purchase_frequency=("transaction_id", "nunique"),
            avg_days_between=(
                "date",
                lambda x: (x.max() - x.min()).days / max(x.nunique() - 1, 1),
            ),
            # Monetary
            total_revenue=("revenue", "sum"),
            avg_order_value=("revenue", "mean"),
            revenue_std=("revenue", "std"),
            # Product
            n_products=("stockcode", "nunique"),
            n_categories=(
                ("category", "nunique")
                if "category" in df.columns
                else ("stockcode", "nunique")
            ),
            # Basket
            avg_basket_size=("quantity", "mean"),
            max_basket_size=("quantity", "max"),
            # Price
            avg_price=("price", "mean"),
            price_cv=("price", lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
            # Temporal patterns
            weekend_ratio=("date", lambda x: (x.dt.dayofweek >= 5).mean()),
            night_ratio=(
                "date",
                lambda x: (x.dt.hour >= 18).mean() if x.dt.hour.nunique() > 1 else 0,
            ),
        )
        .reset_index()
    )

    # Fill NaN
    behavioral = behavioral.fillna(0)

    # Features for clustering
    feature_cols = [c for c in behavioral.columns if c != "customer_id"]
    
    # Use RobustScaler to handle outliers
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(behavioral[feature_cols])

    if method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        behavioral["cluster"] = kmeans.fit_predict(X_scaled)
    elif method == "agglomerative":
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        behavioral["cluster"] = clustering.fit_predict(X_scaled)
    elif method == "gmm":
        gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
        behavioral["cluster"] = gmm.fit_predict(X_scaled)
    elif method == "dbscan":
        # DBSCAN for density-based clustering
        clustering = DBSCAN(eps=1.5, min_samples=5)
        behavioral["cluster"] = clustering.fit_predict(X_scaled)
        n_clusters = len(set(behavioral["cluster"])) - (1 if -1 in behavioral["cluster"].values else 0)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Label clusters
    cluster_profiles = behavioral.groupby("cluster")[feature_cols].mean()
    labels = {}
    for c in behavioral["cluster"].unique():
        if c == -1:  # DBSCAN noise
            labels[c] = "Outliers"
            continue
        profile = cluster_profiles.loc[c]
        if profile["total_revenue"] > cluster_profiles["total_revenue"].quantile(0.75):
            labels[c] = "High Value"
        elif profile["purchase_frequency"] > cluster_profiles[
            "purchase_frequency"
        ].quantile(0.75):
            labels[c] = "Frequent Buyers"
        elif profile["avg_days_between"] < cluster_profiles[
            "avg_days_between"
        ].quantile(0.25):
            labels[c] = "Regular Shoppers"
        elif profile["n_products"] > cluster_profiles["n_products"].quantile(0.75):
            labels[c] = "Variety Seekers"
        elif profile["weekend_ratio"] > 0.5:
            labels[c] = "Weekend Shoppers"
        else:
            labels[c] = f"Segment {c}"

    behavioral["segment"] = behavioral["cluster"].map(labels)

    return behavioral


def predict_clv(
    transactions_df: pd.DataFrame,
    prediction_horizon_days: int = 90,
    model_type: str = "gradient_boosting",
    features_to_use: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Predict Customer Lifetime Value using ML models.
    
    Returns:
        DataFrame with predicted CLV and model metrics
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    snapshot_date = df["date"].max()
    cutoff_date = snapshot_date - pd.Timedelta(days=prediction_horizon_days)

    # Historical (before cutoff) and future (after cutoff)
    hist = df[df["date"] < cutoff_date]
    future = df[df["date"] >= cutoff_date]

    # Historical features
    features = (
        hist.groupby("customer_id")
        .agg(
            recency=("date", lambda x: int((cutoff_date - x.max()).days)),
            frequency=("transaction_id", "nunique"),
            monetary=("revenue", "sum"),
            avg_order=("revenue", "mean"),
            n_products=("stockcode", "nunique"),
            n_categories=(
                ("category", "nunique")
                if "category" in hist.columns
                else ("stockcode", "nunique")
            ),
            avg_items_per_order=("quantity", "mean"),
            avg_price_paid=("price", "mean"),
            price_std=("price", "std"),
            first_purchase=("date", "min"),
            last_purchase=("date", "max"),
        )
        .reset_index()
    )

    # Ensure numeric types
    features["lifetime_days"] = pd.to_numeric(
        (features["last_purchase"] - features["first_purchase"]).dt.days, errors="coerce"
    ).fillna(1)
    
    features["recency"] = pd.to_numeric(features["recency"], errors="coerce").fillna(0).astype(int)

    # Add derived features
    features["monetary_per_order"] = features["monetary"] / features["frequency"].replace(0, 1)
    features["purchase_rate"] = features["frequency"] / features["lifetime_days"].replace(0, 1) * 30
    features["recency_ratio"] = features["recency"] / features["lifetime_days"].replace(0, 1)

    # Future actuals (for validation)
    future_rev = future.groupby("customer_id")["revenue"].sum().reset_index()
    future_rev.columns = ["customer_id", "future_revenue"]
    features = features.merge(future_rev, on="customer_id", how="left").fillna(
        {"future_revenue": 0}
    )

    # Target: future revenue in prediction window
    y = features["future_revenue"]
    X = features.drop(
        columns=["customer_id", "first_purchase", "last_purchase", "future_revenue"]
    )

    # Select features
    if features_to_use:
        X = X[features_to_use]

    # Handle infinite/NaN
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create model
    if model_type == "xgboost" and XGBOOST_AVAILABLE:
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    elif model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=42,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=42,
        )

    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    metrics = {
        "train_mae": mean_absolute_error(y_train, y_pred_train),
        "test_mae": mean_absolute_error(y_test, y_pred_test),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test),
    }

    # Feature importance
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
    elif hasattr(model, "coef_"):
        feature_importance = pd.DataFrame({
            "feature": X.columns,
            "importance": np.abs(model.coef_),
        }).sort_values("importance", ascending=False)
    else:
        feature_importance = pd.DataFrame()

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    metrics["cv_mae_mean"] = -cv_scores.mean()
    metrics["cv_mae_std"] = cv_scores.std()

    # Full predictions
    features["predicted_clv"] = model.predict(X)
    features["predicted_clv"] = features["predicted_clv"].clip(lower=0)

    # CLV-based segments
    features["clv_segment"] = pd.qcut(
        features["predicted_clv"],
        q=4,
        labels=["Bronze", "Silver", "Gold", "Platinum"],
        duplicates="drop",
    )

    # Return customer_id, predicted_clv, clv_segment, and metrics
    result_df = features[["customer_id", "predicted_clv", "clv_segment"]].copy()
    result_df = result_df.merge(
        features[["customer_id", "recency", "frequency", "monetary"]], on="customer_id"
    )

    return result_df, {"metrics": metrics, "feature_importance": feature_importance, "model": model}


def survival_analysis(
    transactions_df: pd.DataFrame,
    time_col: str = "date",
    event_col: str = "transaction_id",
    customer_col: str = "customer_id",
) -> Dict:
    """
    Perform survival analysis on customer churn.
    
    Uses lifelines library for Kaplan-Meier and Cox Proportional Hazards.
    """
    if not LIFELINES_AVAILABLE:
        return {"error": "lifelines not installed. Run: pip install lifelines"}

    df = transactions_df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Calculate time to next purchase for each customer
    df = df.sort_values([customer_col, time_col])
    df["next_purchase"] = df.groupby(customer_col)[time_col].shift(-1)
    df["days_to_next"] = (df["next_purchase"] - df[time_col]).dt.days
    
    # For last purchase, censor at snapshot date
    snapshot = df[time_col].max() + pd.Timedelta(days=1)
    df["days_to_next"] = df["days_to_next"].fillna((snapshot - df[time_col]).dt.days)
    
    # Event: 1 if made another purchase, 0 if censored
    df["event"] = df["next_purchase"].notna().astype(int)
    
    # One row per customer (use last observation)
    surv_df = df.groupby(customer_col).last().reset_index()
    surv_df = surv_df[[customer_col, "days_to_next", "event"]].copy()
    surv_df.columns = [customer_col, "duration", "event"]
    
    # Kaplan-Meier estimate
    kmf = KaplanMeierFitter()
    kmf.fit(surv_df["duration"], surv_df["event"])
    
    # Median survival time
    median_survival = kmf.median_survival_time_
    
    # Survival curves by segment if available
    results = {
        "kaplan_meier": {
            "survival_function": kmf.survival_function_.to_dict(),
            "median_survival_days": median_survival,
            "confidence_interval": kmf.confidence_interval_.to_dict(),
        },
        "n_customers": len(surv_df),
        "n_events": surv_df["event"].sum(),
        "censoring_rate": (surv_df["event"] == 0).mean(),
    }
    
    # Cox Proportional Hazards if we have covariates
    if len(df.columns) > 3:
        # Build covariates (RFM features)
        rfm = compute_rfm_features(transactions_df)
        rfm = rfm.set_index("customer_id")
        
        # Merge with survival data
        cox_df = surv_df.set_index(customer_col).join(rfm[["recency_days", "frequency", "monetary"]])
        cox_df = cox_df.dropna()
        
        if len(cox_df) > 20:
            cph = CoxPHFitter()
            try:
                cph.fit(cox_df, duration_col="duration", event_col="event")
                results["cox_model"] = {
                    "summary": cph.summary.to_dict(),
                    "concordance_index": cph.concordance_index_,
                    "log_likelihood": cph.log_likelihood_,
                }
            except Exception as e:
                results["cox_model"] = {"error": str(e)}
    
    return results


def ensemble_segmentation(
    transactions_df: pd.DataFrame,
    n_segments: int = 8,
    methods: List[str] = None,
) -> pd.DataFrame:
    """
    Ensemble segmentation combining multiple methods.
    
    Uses consensus clustering to combine RFM, behavioral, and value-based segmentations.
    """
    if methods is None:
        methods = ["rfm_kmeans", "behavioral_kmeans", "value_based"]
    
    all_segmentations = []
    
    # RFM K-means
    if "rfm_kmeans" in methods:
        rfm = compute_rfm_features(transactions_df)
        rfm_seg = rfm_segmentation(rfm, method="kmeans", n_segments=n_segments)
        rfm_seg = rfm_seg[["customer_id", "segment"]].rename(columns={"segment": "rfm_segment"})
        all_segmentations.append(rfm_seg)
    
    # Behavioral K-means
    if "behavioral_kmeans" in methods:
        beh_seg = behavioral_segmentation(transactions_df, n_clusters=n_segments, method="kmeans")
        beh_seg = beh_seg[["customer_id", "segment"]].rename(columns={"segment": "behavioral_segment"})
        all_segmentations.append(beh_seg)
    
    # Value-based
    if "value_based" in methods:
        val_seg = value_based_segmentation(transactions_df, prediction_horizon_days=90)
        val_seg = val_seg[["customer_id", "value_segment"]].rename(columns={"value_segment": "value_segment"})
        all_segmentations.append(val_seg)
    
    # Merge all segmentations
    if not all_segmentations:
        return pd.DataFrame()
    
    ensemble = all_segmentations[0]
    for seg in all_segmentations[1:]:
        ensemble = ensemble.merge(seg, on="customer_id", how="outer")
    
    # Consensus clustering: assign final segment based on majority vote
    segment_cols = [c for c in ensemble.columns if c != "customer_id"]
    
    def consensus_segment(row):
        votes = row[segment_cols].value_counts()
        if len(votes) > 0:
            return votes.index[0]
        return "Unknown"
    
    ensemble["ensemble_segment"] = ensemble.apply(consensus_segment, axis=1)
    
    # Also compute agreement score
    ensemble["agreement_score"] = ensemble[segment_cols].apply(
        lambda x: x.value_counts().iloc[0] / len(x), axis=1
    )
    
    return ensemble


def value_based_segmentation(
    transactions_df: pd.DataFrame, prediction_horizon_days: int = 90
) -> pd.DataFrame:
    """Value-based segmentation with predicted CLV."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    snapshot_date = df["date"].max()
    cutoff_date = snapshot_date - pd.Timedelta(days=prediction_horizon_days)

    hist = df[df["date"] < cutoff_date]
    future = df[df["date"] >= cutoff_date]

    features = (
        hist.groupby("customer_id")
        .agg(
            recency=("date", lambda x: int((cutoff_date - x.max()).days)),
            frequency=("transaction_id", "nunique"),
            monetary=("revenue", "sum"),
            avg_order=("revenue", "mean"),
            n_products=("stockcode", "nunique"),
            lifetime_days=("date", lambda x: int((x.max() - x.min()).days + 1)),
        )
        .reset_index()
    )

    features["lifetime_days"] = pd.to_numeric(
        features["lifetime_days"], errors="coerce"
    ).fillna(1)
    
    features["recency"] = pd.to_numeric(
        features["recency"], errors="coerce"
    ).fillna(0).astype(int)

    features["predicted_clv"] = (
        features["monetary"]
        / features["frequency"].replace(0, 1)
        * features["frequency"]
        * (365 / features["lifetime_days"].replace(0, 1))
        * 2
    )

    future_rev = future.groupby("customer_id")["revenue"].sum().reset_index()
    future_rev.columns = ["customer_id", "future_revenue"]
    features = features.merge(future_rev, on="customer_id", how="left").fillna(
        {"future_revenue": 0}
    )

    conditions = [
        (features["predicted_clv"] > features["predicted_clv"].quantile(0.8))
        & (features["recency"] < 30),
        (features["predicted_clv"] > features["predicted_clv"].quantile(0.6))
        & (features["recency"] < 60),
        (features["frequency"] > 5) & (features["recency"] < 90),
        (features["frequency"] == 1) & (features["recency"] < 30),
        (features["recency"] > 180),
    ]
    choices = ["VIP", "High Potential", "Loyal", "New", "Churned"]
    features["value_segment"] = np.select(conditions, choices, default="Regular")

    return features


def get_segment_profiles(
    transactions_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    segment_col: str = "segment",
) -> pd.DataFrame:
    """Get detailed profiles for each segment."""
    df = transactions_df.merge(
        segments_df[["customer_id", segment_col]], on="customer_id", how="left"
    )
    df["revenue"] = df["price"] * df["quantity"]

    profiles = (
        df.groupby(segment_col)
        .agg(
            n_customers=("customer_id", "nunique"),
            n_transactions=("transaction_id", "nunique"),
            total_revenue=("revenue", "sum"),
            avg_order_value=("revenue", "mean"),
            avg_items_per_order=("quantity", "mean"),
            avg_products_per_customer=(
                "stockcode",
                lambda x: (
                    x.nunique() / df[df[segment_col] == x.name]["customer_id"].nunique()
                ),
            ),
            top_category=(
                "category",
                lambda x: (
                    x.mode().iloc[0]
                    if "category" in df.columns and not x.mode().empty
                    else "N/A"
                ),
            ),
            top_brand=(
                "brand",
                lambda x: (
                    x.mode().iloc[0]
                    if "brand" in df.columns and not x.mode().empty
                    else "N/A"
                ),
            ),
            repeat_rate=(
                "transaction_id",
                lambda x: (
                    (
                        x.nunique()
                        / df[df[segment_col] == x.name]["customer_id"].nunique()
                    )
                    if df[df[segment_col] == x.name]["customer_id"].nunique() > 0
                    else 0
                ),
            ),
        )
        .reset_index()
    )

    profiles["revenue_per_customer"] = (
        profiles["total_revenue"] / profiles["n_customers"]
    )
    profiles["revenue_share"] = (
        profiles["total_revenue"] / profiles["total_revenue"].sum()
    )
    profiles["customer_share"] = profiles["n_customers"] / profiles["n_customers"].sum()

    return profiles


def get_available_models() -> Dict[str, bool]:
    """Check which ML libraries are available."""
    return {
        "lifelines": LIFELINES_AVAILABLE,
        "xgboost": XGBOOST_AVAILABLE,
        "lightgbm": LIGHTGBM_AVAILABLE,
    }