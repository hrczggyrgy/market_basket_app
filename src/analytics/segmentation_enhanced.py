"""Enhanced Customer Segmentation with CLV Prediction, Survival Analysis, and Ensemble Methods."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import (
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

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


MIN_CLUSTER_SIZE = 5


def _label_rfm_clusters(profiles: pd.DataFrame) -> dict:
    n_clusters = len(profiles)
    ranked = pd.DataFrame(
        {
            "rec_rank": profiles["recency_days"].rank(),
            "freq_rank": profiles["frequency"].rank(ascending=False),
            "mon_rank": profiles["monetary"].rank(ascending=False),
        }
    )
    labels = {}
    for c in profiles.index:
        r = ranked.loc[c, "rec_rank"]
        mr = ranked.loc[c, "mon_rank"]
        fr = ranked.loc[c, "freq_rank"]

        if mr <= 2 and r <= 2:
            labels[c] = "Champions"
        elif mr <= 2:
            labels[c] = "Big Spenders"
        elif fr <= 2 and r <= 2:
            labels[c] = "Frequent Buyers"
        elif fr <= 2:
            labels[c] = "Loyal"
        elif r <= 2 and mr <= 3:
            labels[c] = "Promising"
        elif r >= n_clusters - 1:
            labels[c] = "Dormant"
        elif fr >= n_clusters - 1 and mr >= n_clusters - 1:
            labels[c] = "At Risk"
        else:
            p = profiles.loc[c]
            labels[c] = (
                f"Regular ({int(p['recency_days'])}d, {p['frequency']:.1f}x, ${p['monetary']:.0f})"
            )
    return labels


def _label_behavioral_clusters(profiles: pd.DataFrame) -> dict:
    key_dims = ["total_revenue", "purchase_frequency", "n_products"]
    present = [d for d in key_dims if d in profiles.columns]
    if not present:
        return {c: f"Segment {c}" for c in profiles.index}

    n_clusters = len(profiles)
    ranked = pd.DataFrame({d: profiles[d].rank(ascending=False) for d in present})
    has_weekend = "weekend_ratio" in profiles.columns
    labels = {}

    for c in profiles.index:
        rev = ranked.loc[c, "total_revenue"]
        freq = ranked.loc[c, "purchase_frequency"]
        prod = ranked.loc[c, "n_products"]

        if rev <= 2 and freq <= 2:
            labels[c] = "High Value"
        elif has_weekend and profiles.loc[c, "weekend_ratio"] > 0.5:
            labels[c] = "Weekend Shoppers"
        elif freq <= 2 and prod <= 2:
            labels[c] = "Frequent Buyers"
        elif prod <= 2:
            labels[c] = "Variety Seekers"
        elif freq <= 2:
            labels[c] = "Regular Shoppers"
        elif rev <= 2:
            labels[c] = "Big Spenders"
        elif freq >= n_clusters - 1 and rev >= n_clusters - 1:
            labels[c] = "Light Buyers"
        else:
            p = profiles.loc[c]
            details = ", ".join(f"{d}={p[d]:.1f}" for d in present[:2])
            labels[c] = f"Mid-Tier ({details})"
    return labels


def compute_cluster_quality_metrics(
    features: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute silhouette score and Davies-Bouldin index for a clustering.

    Returns empty dict when fewer than 2 clusters or evaluation fails.
    """
    unique = set(labels)
    n_clusters = len(unique - {-1}) if -1 in unique else len(unique)
    if n_clusters < 2:
        return {}

    mask = labels != -1
    valid_count = mask.sum()
    if valid_count < n_clusters or valid_count < MIN_CLUSTER_SIZE:
        return {}

    try:
        sil = silhouette_score(features[mask], labels[mask])
        db = davies_bouldin_score(features[mask], labels[mask])
        sizes = pd.Series(labels[mask]).value_counts()
        return {
            "silhouette_score": round(sil, 4),
            "davies_bouldin_score": round(db, 4),
            "n_clusters": n_clusters,
            "cluster_size_min": int(sizes.min()),
            "cluster_size_max": int(sizes.max()),
            "cluster_size_mean": round(sizes.mean(), 1),
            "cluster_size_std": round(sizes.std(), 1),
        }
    except Exception:
        return {}


def compute_cluster_stability(
    transactions_df: pd.DataFrame,
    n_clusters: int = 6,
    n_iterations: int = 10,
    method: str = "kmeans",
    sample_frac: float = 0.8,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate cluster stability across random seeds and subsamples.

    Returns mean/std/min/max adjusted Rand index vs reference clustering.
    """
    from sklearn.metrics import adjusted_rand_score
    from sklearn.utils import resample

    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    cat_col = "category" if "category" in df.columns else "stockcode"

    behavioral = (
        df.groupby("customer_id")
        .agg(
            days_active=("date", lambda x: (x.max() - x.min()).days + 1),
            purchase_frequency=("transaction_id", "nunique"),
            avg_days_between=(
                "date",
                lambda x: (x.max() - x.min()).days / max(x.nunique() - 1, 1),
            ),
            total_revenue=("revenue", "sum"),
            avg_order_value=("revenue", "mean"),
            n_products=("stockcode", "nunique"),
            n_categories=(cat_col, "nunique"),
            avg_basket_size=("quantity", "mean"),
            avg_price=("price", "mean"),
            weekend_ratio=("date", lambda x: (x.dt.dayofweek >= 5).mean()),
        )
        .fillna(0)
        .reset_index()
    )

    feature_cols = [c for c in behavioral.columns if c != "customer_id"]
    if len(behavioral) < max(n_clusters * 2, 10):
        return {}

    scaler = StandardScaler()
    X = scaler.fit_transform(behavioral[feature_cols])

    ref = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(X)

    scores = []
    for i in range(n_iterations):
        rs = seed + i + 1
        subsample_idx = resample(
            range(len(X)), replace=False, n_samples=int(len(X) * sample_frac), random_state=rs
        )
        X_sub = X[subsample_idx]
        pred = KMeans(n_clusters=n_clusters, random_state=rs, n_init=10).fit_predict(X_sub)
        ref_sub = ref[subsample_idx]
        scores.append(adjusted_rand_score(ref_sub, pred))

    scores = np.array(scores)
    return {
        "mean_ari": round(scores.mean(), 4),
        "std_ari": round(scores.std(), 4),
        "min_ari": round(scores.min(), 4),
        "max_ari": round(scores.max(), 4),
    }


class CLVPrediction:
    """Container for CLV prediction results with validation and baselines."""

    def __init__(
        self,
        transactions_df: pd.DataFrame,
        prediction_horizon_days: int = 90,
    ):
        self.prediction_horizon_days = prediction_horizon_days
        df = transactions_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["revenue"] = df["price"] * df["quantity"]

        snapshot_date = df["date"].max()
        self.cutoff_date = snapshot_date - pd.Timedelta(prediction_horizon_days, unit="D")

        self.hist = df[df["date"] < self.cutoff_date].copy()
        self.future = df[df["date"] >= self.cutoff_date].copy()
        self.transactions_df = df

    def build_features(self) -> pd.DataFrame:
        """Build historical features and merge future actuals."""
        cat_col = "category" if "category" in self.hist.columns else "stockcode"

        features = (
            self.hist.groupby("customer_id")
            .agg(
                recency=("date", lambda x: int((self.cutoff_date - x.max()).days)),
                frequency=("transaction_id", "nunique"),
                monetary=("revenue", "sum"),
                avg_order=("revenue", "mean"),
                n_products=("stockcode", "nunique"),
                n_categories=(cat_col, "nunique"),
                avg_items_per_order=("quantity", "mean"),
                avg_price_paid=("price", "mean"),
                price_std=("price", "std"),
                first_purchase=("date", "min"),
                last_purchase=("date", "max"),
            )
            .reset_index()
        )

        features["lifetime_days"] = pd.to_numeric(
            (features["last_purchase"] - features["first_purchase"]).dt.days,
            errors="coerce",
        ).fillna(1)

        features["recency"] = (
            pd.to_numeric(features["recency"], errors="coerce").fillna(0).astype(int)
        )

        features["monetary_per_order"] = features["monetary"] / features["frequency"].replace(0, 1)
        features["purchase_rate"] = (
            features["frequency"] / features["lifetime_days"].replace(0, 1) * 30
        )
        features["recency_ratio"] = features["recency"] / features["lifetime_days"].replace(0, 1)

        future_rev = self.future.groupby("customer_id")["revenue"].sum().reset_index()
        future_rev.columns = ["customer_id", "future_revenue"]
        features = features.merge(future_rev, on="customer_id", how="left").fillna(
            {"future_revenue": 0}
        )

        self.features = features
        return features

    def baseline_predictions(self) -> pd.DataFrame:
        """Compute simple benchmark baselines for CLV comparison."""
        if not hasattr(self, "features"):
            self.build_features()

        f = self.features
        baselines = pd.DataFrame({"customer_id": f["customer_id"]})

        # Baseline 1: trailing 90-day revenue (same window)
        baselines["baseline_trailing_revenue"] = f["monetary"] * (
            self.prediction_horizon_days / f["lifetime_days"].clip(lower=1)
        )

        # Baseline 2: average daily revenue extrapolated
        daily_rate = f["monetary"] / f["lifetime_days"].clip(lower=1)
        baselines["baseline_extrapolated"] = daily_rate * self.prediction_horizon_days

        # Baseline 3: simple heuristic (AOV * frequency_in_window)
        baseline_freq = f["frequency"] * (
            self.prediction_horizon_days / f["lifetime_days"].clip(lower=1)
        )
        baselines["baseline_aov_times_freq"] = f["avg_order"] * baseline_freq.fillna(0)

        # Errors vs actual
        actual = f.set_index("customer_id")["future_revenue"]
        for col in [c for c in baselines.columns if c != "customer_id"]:
            pred = baselines.set_index("customer_id")[col]
            common = actual.index.intersection(pred.index)
            baselines[f"{col}_mae"] = (actual[common] - pred[common]).abs().mean()

        self.baselines = baselines
        return baselines


def predict_clv_v2(
    transactions_df: pd.DataFrame,
    prediction_horizon_days: int = 90,
    model_type: str = "gradient_boosting",
    features_to_use: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Predict CLV with benchmark baselines and out-of-time validation.

    Uses the CLVPrediction container internally.
    """
    cv = CLVPrediction(transactions_df, prediction_horizon_days)
    features = cv.build_features()
    baselines = cv.baseline_predictions()

    # Merge baselines
    features = features.merge(baselines, on="customer_id", how="left", suffixes=("", "_bl"))

    y = features["future_revenue"]
    X = features.drop(columns=["customer_id", "first_purchase", "last_purchase", "future_revenue"])
    if features_to_use:
        X = X[features_to_use]

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    else:
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=42,
        )

    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "test_mae": mean_absolute_error(y_test, y_pred_test),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "test_r2": r2_score(y_test, y_pred_test),
    }

    # Add baseline metrics for comparison
    if hasattr(cv, "baselines"):
        bl_cols = [c for c in cv.baselines.columns if c.endswith("_mae")]
        for col in bl_cols:
            metrics[f"baseline_{col}"] = cv.baselines[col].iloc[0]

    # Feature importance
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        fi = fi.sort_values("importance", ascending=False)
    else:
        fi = pd.DataFrame()

    # Refit on all data
    model.fit(X, y)
    features["predicted_clv"] = np.clip(model.predict(X), a_min=0, a_max=None)

    features["clv_segment"] = pd.qcut(
        features["predicted_clv"],
        q=4,
        labels=["Bronze", "Silver", "Gold", "Platinum"],
        duplicates="drop",
    )

    result_df = features[["customer_id", "predicted_clv", "clv_segment"]].copy()
    result_df = result_df.merge(
        features[["customer_id", "recency", "frequency", "monetary"]], on="customer_id"
    )

    return result_df, {"metrics": metrics, "feature_importance": fi, "model": model}


def compute_rfm_features(
    transactions_df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Compute comprehensive RFM features per customer."""
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    if snapshot_date is None:
        snapshot_date = df["date"].max() + pd.Timedelta(1, unit="D")

    cat_col_rfm = "category" if "category" in df.columns else "stockcode"
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
            n_unique_categories=(cat_col_rfm, "nunique"),
            first_purchase=("date", "min"),
            last_purchase=("date", "max"),
            avg_price_paid=("price", "mean"),
            std_order_value=("revenue", "std"),
        )
        .reset_index()
    )

    # Derived features
    rfm["customer_lifetime_days"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days
    rfm["purchase_interval"] = np.where(
        rfm["frequency"] > 1,
        rfm["customer_lifetime_days"] / (rfm["frequency"] - 1),
        rfm["customer_lifetime_days"],
    )
    rfm["items_per_order"] = rfm["n_items"] / rfm["frequency"]
    rfm["revenue_per_item"] = rfm["monetary"] / rfm["n_items"].replace(0, np.nan)
    rfm["order_value_cv"] = rfm["std_order_value"] / rfm["avg_order_value"].replace(0, np.nan)

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
                df[f"{dim}_score"] = pd.qcut(df[dim], q=4, labels=[4, 3, 2, 1], duplicates="drop")
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

        # Segment mapping — full 4x4x4 = 64 combinations via weighted score
        r = df["recency_days_score"].astype(int)
        f = df["frequency_score"].astype(int)
        m = df["monetary_score"].astype(int)

        conditions = [
            (r >= 3) & (f >= 3) & (m >= 3),  # 333-444
            (r >= 3) & (f >= 3) & (m == 2),  # 33x
            (r >= 3) & (f == 2) & (m >= 3),  # 32x
            (r >= 3) & (f >= 2) & (m >= 2),  # 322-432 except above
            (r >= 3) & (f == 2) & (m == 1),  # 321
            (r == 2) & (f >= 3) & (m >= 3),  # 23x
            (r == 2) & (f == 3) & (m == 2),  # 232
            (r == 2) & (f == 2) & (m >= 3),  # 223
            (r == 2) & (f >= 2) & (m == 2),  # 222, 322(already)
            (r == 2) & (f == 2) & (m == 1),  # 221
            (r == 2) & (f == 1) & (m >= 3),  # 213
            (r == 2) & (f == 1) & (m == 2),  # 212
            (r == 2) & (f == 1) & (m == 1),  # 211
            (r == 1) & (f >= 3) & (m >= 3),  # 133
            (r == 1) & (f >= 2) & (m >= 2),  # 122-132
            (r == 1) & (f >= 2) & (m == 1),  # 121
            (r == 1) & (f == 1) & (m >= 2),  # 112
            (r == 1) & (f == 1) & (m == 1),  # 111
        ]
        choices = [
            "Champions",
            "Loyal",
            "Potential Loyalists",
            "Promising",
            "New Customers",
            "Need Attention",
            "Need Attention",
            "Potential Loyalists",
            "Regular",
            "At Risk",
            "Need Attention",
            "At Risk",
            "Cannot Lose Them",
            "About to Sleep",
            "About to Sleep",
            "About to Sleep",
            "Cannot Lose Them",
            "Lost",
        ]
        df["segment"] = np.select(conditions, choices, default="Other")

    elif method in ("kmeans", "gmm"):
        features = ["recency_days", "frequency", "monetary"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])

        if method == "kmeans":
            model = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
        else:
            model = GaussianMixture(n_components=n_segments, random_state=42, n_init=10)
        df["cluster"] = model.fit_predict(X_scaled)

        cluster_profiles = df.groupby("cluster")[features].mean()
        cluster_labels = _label_rfm_clusters(cluster_profiles)
        df["segment"] = df["cluster"].map(cluster_labels)

    return df


def behavioral_segmentation(
    transactions_df: pd.DataFrame,
    n_clusters: int = 6,
    method: str = "kmeans",
    return_metrics: bool = False,
) -> pd.DataFrame:
    """Behavioral segmentation based on purchase patterns.

    Args:
        transactions_df: Transaction data
        n_clusters: Number of clusters
        method: Clustering algorithm ('kmeans', 'agglomerative', 'gmm', 'dbscan')
        return_metrics: Also return cluster quality metrics dict

    Returns:
        DataFrame with cluster assignments; if return_metrics is True,
        returns (DataFrame, metrics_dict).
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    cat_col_behav = "category" if "category" in df.columns else "stockcode"

    behavioral = (
        df.groupby("customer_id")
        .agg(
            days_active=("date", lambda x: (x.max() - x.min()).days + 1),
            purchase_frequency=("transaction_id", "nunique"),
            avg_days_between=(
                "date",
                lambda x: (x.max() - x.min()).days / max(x.nunique() - 1, 1),
            ),
            total_revenue=("revenue", "sum"),
            avg_order_value=("revenue", "mean"),
            revenue_std=("revenue", "std"),
            n_products=("stockcode", "nunique"),
            n_categories=(cat_col_behav, "nunique"),
            avg_basket_size=("quantity", "mean"),
            max_basket_size=("quantity", "max"),
            avg_price=("price", "mean"),
            price_cv=("price", lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
            weekend_ratio=("date", lambda x: (x.dt.dayofweek >= 5).mean()),
        )
        .reset_index()
    )

    behavioral = behavioral.fillna(0)

    feature_cols = [c for c in behavioral.columns if c != "customer_id"]
    n_samples = len(behavioral)

    # Minimum sample-size guard
    min_required = max(n_clusters * MIN_CLUSTER_SIZE, 10)
    if n_samples < min_required:
        behavioral["cluster"] = 0
        behavioral["segment"] = "Other"
        behavioral["cluster_distance"] = 0.0
        if return_metrics:
            return behavioral, {}
        return behavioral

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(behavioral[feature_cols])

    if method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        behavioral["cluster"] = kmeans.fit_predict(X_scaled)
        # Distance-to-centroid confidence (KMeans only)
        distances = kmeans.transform(X_scaled)
        behavioral["cluster_distance"] = distances.min(axis=1)
        max_dist = distances.max(axis=1)
        behavioral["cluster_confidence"] = np.where(
            max_dist > 0, 1 - behavioral["cluster_distance"] / max_dist, 1.0
        )
    elif method == "agglomerative":
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        behavioral["cluster"] = clustering.fit_predict(X_scaled)
        behavioral["cluster_distance"] = 0.0
    elif method == "gmm":
        gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
        behavioral["cluster"] = gmm.fit_predict(X_scaled)
        # GMM confidence = max responsibility
        behavioral["cluster_distance"] = 1.0 - gmm.predict_proba(X_scaled).max(axis=1)
    elif method == "dbscan":
        clustering = DBSCAN(eps=1.5, min_samples=5)
        behavioral["cluster"] = clustering.fit_predict(X_scaled)
        behavioral["cluster_distance"] = np.where(behavioral["cluster"] == -1, 1.0, 0.0)
        actual_n = len(set(behavioral["cluster"])) - (
            1 if -1 in behavioral["cluster"].values else 0
        )
        if actual_n > 0:
            n_clusters = actual_n
    else:
        raise ValueError(f"Unknown method: {method}")

    # Cluster quality metrics
    quality_metrics = compute_cluster_quality_metrics(X_scaled, behavioral["cluster"].values)

    # Label clusters
    cluster_profiles = behavioral.groupby("cluster")[feature_cols].mean()
    labels = _label_behavioral_clusters(cluster_profiles)
    for c in behavioral["cluster"].unique():
        if c == -1:
            labels[c] = "Outliers"

    # Drop clusters below MIN_CLUSTER_SIZE
    cluster_sizes = behavioral["cluster"].value_counts()
    small_clusters = cluster_sizes[cluster_sizes < MIN_CLUSTER_SIZE].index
    valid_clusters = set(behavioral["cluster"].unique()) - set(small_clusters)
    if not small_clusters.empty and len(valid_clusters) > 0:
        for sc in small_clusters:
            if sc == -1:
                continue
            behavioral.loc[behavioral["cluster"] == sc, "cluster"] = -1
            labels.pop(sc, None)

    behavioral["segment"] = behavioral["cluster"].map(labels).fillna("Outliers")

    if return_metrics:
        return behavioral, quality_metrics
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
        DataFrame with predicted CLV and a dict of metrics + diagnostics
        including baseline comparisons, fold-level R², and low-confidence flags.
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    snapshot_date = df["date"].max()
    cutoff_date = snapshot_date - pd.Timedelta(prediction_horizon_days, unit="D")

    hist = df[df["date"] < cutoff_date]
    future = df[df["date"] >= cutoff_date]

    cat_col_hist = "category" if "category" in hist.columns else "stockcode"

    features = (
        hist.groupby("customer_id")
        .agg(
            recency=("date", lambda x: int((cutoff_date - x.max()).days)),
            frequency=("transaction_id", "nunique"),
            monetary=("revenue", "sum"),
            avg_order=("revenue", "mean"),
            n_products=("stockcode", "nunique"),
            n_categories=(cat_col_hist, "nunique"),
            avg_items_per_order=("quantity", "mean"),
            avg_price_paid=("price", "mean"),
            price_std=("price", "std"),
            first_purchase=("date", "min"),
            last_purchase=("date", "max"),
        )
        .reset_index()
    )

    features["lifetime_days"] = pd.to_numeric(
        (features["last_purchase"] - features["first_purchase"]).dt.days, errors="coerce"
    ).fillna(1)

    features["recency"] = pd.to_numeric(features["recency"], errors="coerce").fillna(0).astype(int)

    features["monetary_per_order"] = features["monetary"] / features["frequency"].replace(0, 1)
    features["purchase_rate"] = features["frequency"] / features["lifetime_days"].replace(0, 1) * 30
    features["recency_ratio"] = features["recency"] / features["lifetime_days"].replace(0, 1)

    future_rev = future.groupby("customer_id")["revenue"].sum().reset_index()
    future_rev.columns = ["customer_id", "future_revenue"]
    features = features.merge(future_rev, on="customer_id", how="left").fillna(
        {"future_revenue": 0}
    )

    y = features["future_revenue"]
    X = features.drop(columns=["customer_id", "first_purchase", "last_purchase", "future_revenue"])
    if features_to_use:
        X = X[features_to_use]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # --- Baselines ---
    naive_mean_pred = np.full_like(y, y.mean())
    baseline_mae = mean_absolute_error(y, naive_mean_pred)
    baseline_r2 = r2_score(y, naive_mean_pred)
    metrics = {
        "baseline_naive_mean_mae": baseline_mae,
        "baseline_naive_mean_r2": baseline_r2,
    }

    # Heuristic baseline: trailing daily revenue rate * horizon
    daily_rate = features["monetary"] / features["lifetime_days"].clip(lower=1)
    heuristic_pred = daily_rate * prediction_horizon_days
    metrics["baseline_heuristic_mae"] = mean_absolute_error(y, heuristic_pred)
    metrics["baseline_heuristic_r2"] = r2_score(y, heuristic_pred)

    # --- Train / test split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    else:
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=42,
        )

    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    metrics["train_mae"] = mean_absolute_error(y_train, y_pred_train)
    metrics["test_mae"] = mean_absolute_error(y_test, y_pred_test)
    metrics["train_r2"] = r2_score(y_train, y_pred_train)
    metrics["test_r2"] = r2_score(y_test, y_pred_test)

    # Track holdout customers for audit
    test_idx = X_test.index
    features["_in_validation_holdout"] = False
    features.loc[features.index.isin(test_idx), "_in_validation_holdout"] = True

    # Feature importance (using the train-only model)
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        fi = fi.sort_values("importance", ascending=False)
    elif hasattr(model, "coef_"):
        fi = pd.DataFrame({"feature": X.columns, "importance": np.abs(model.coef_)})
        fi = fi.sort_values("importance", ascending=False)
    else:
        fi = pd.DataFrame()

    # Cross-validation — report both MAE and R² per fold
    cv_mae = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)

    metrics["cv_mae_mean"] = -cv_mae.mean()
    metrics["cv_mae_std"] = cv_mae.std()
    metrics["cv_r2_mean"] = cv_r2.mean()
    metrics["cv_r2_std"] = cv_r2.std()
    metrics["cv_r2_folds"] = [round(s, 4) for s in cv_r2]
    metrics["cv_negative_r2_folds"] = int((cv_r2 < 0).sum())

    # Refit on ALL data for final deployed predictions
    model.fit(X, y)
    features["predicted_clv"] = np.clip(model.predict(X), a_min=0, a_max=None)

    features["clv_segment"] = pd.qcut(
        features["predicted_clv"],
        q=4,
        labels=["Bronze", "Silver", "Gold", "Platinum"],
        duplicates="drop",
    )

    # Low-confidence flags
    features["clv_low_confidence"] = (features["frequency"] < 3) | (features["lifetime_days"] < 30)

    result_df = features[
        [
            "customer_id",
            "predicted_clv",
            "clv_segment",
            "clv_low_confidence",
            "_in_validation_holdout",
        ]
    ].copy()
    result_df = result_df.merge(
        features[["customer_id", "recency", "frequency", "monetary"]], on="customer_id"
    )

    return result_df, {"metrics": metrics, "feature_importance": fi, "model": model}


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
    snapshot = df[time_col].max() + pd.Timedelta(1, unit="D")
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
        cox_df = surv_df.set_index(customer_col).join(
            rfm[["recency_days", "frequency", "monetary"]]
        )
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
        beh_seg = beh_seg[["customer_id", "segment"]].rename(
            columns={"segment": "behavioral_segment"}
        )
        all_segmentations.append(beh_seg)

    # Value-based
    if "value_based" in methods:
        val_seg = value_based_segmentation(transactions_df, prediction_horizon_days=90)
        val_seg = val_seg[["customer_id", "value_segment"]].rename(
            columns={"value_segment": "value_segment"}
        )
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
            max_count = votes.max()
            top = votes[votes == max_count]
            return top.index[0] if len(top) == 1 else top.sort_index().index[0]
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
    """Value-based segmentation using survival-adjusted revenue projection, not CLV.

    Computes annualized historical revenue adjusted by a survival probability that
    down-weights customers with long recency relative to their observed lifetime.
    The result is a forward-looking revenue rate, not a true ML-based CLV prediction.
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    snapshot_date = df["date"].max()
    cutoff_date = snapshot_date - pd.Timedelta(prediction_horizon_days, unit="D")

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

    features["lifetime_days"] = pd.to_numeric(features["lifetime_days"], errors="coerce").fillna(1)
    features["recency"] = pd.to_numeric(features["recency"], errors="coerce").fillna(0).astype(int)

    # Annualized historical revenue
    annual_value = features["monetary"] / (features["lifetime_days"].clip(lower=1) / 365)

    # Survival probability: customers with long recency relative to lifetime likely churned
    denom = features["lifetime_days"].clip(lower=1) + features["recency"]
    survival_prob = np.clip(1 - features["recency"] / denom, 0, 1)

    # 2-year projected value = annual rate * survival * 2 (heuristic, not ML)
    features["predicted_clv"] = annual_value * survival_prob * 2

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
            n_products=("stockcode", "nunique"),
            top_category=(
                "category",
                lambda x: (
                    x.mode().iloc[0] if "category" in df.columns and not x.mode().empty else "N/A"
                ),
            ),
            top_brand=(
                "brand",
                lambda x: (
                    x.mode().iloc[0] if "brand" in df.columns and not x.mode().empty else "N/A"
                ),
            ),
        )
        .reset_index()
    )

    # Compute ratios after groupby (avoids x.name scope bug with lambdas)
    profiles["avg_products_per_customer"] = profiles["n_products"] / profiles["n_customers"]
    profiles["repeat_rate"] = profiles["n_transactions"] / profiles["n_customers"]
    profiles["revenue_per_customer"] = profiles["total_revenue"] / profiles["n_customers"]
    profiles["revenue_share"] = profiles["total_revenue"] / profiles["total_revenue"].sum()
    profiles["customer_share"] = profiles["n_customers"] / profiles["n_customers"].sum()

    return profiles


# =====================================================================
# ENHANCED SEGMENTATION FEATURES & METHODS
# =====================================================================

def compute_enhanced_behavioral_features(
    transactions_df: pd.DataFrame,
    snapshot_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Compute enhanced behavioral features beyond classic RFM.
    
    Adds: basket complexity, switching entropy, category breadth, repeat rate,
    time-to-second-purchase, price sensitivity, promotion affinity, etc.
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    if snapshot_date is None:
        snapshot_date = df["date"].max() + pd.Timedelta(1, unit="D")

    cat_col = "category" if "category" in df.columns else "stockcode"

    # Helper: get basket IDs
    if "transaction_id" in transactions_df.columns:
        df["_basket"] = df["transaction_id"]
    else:
        df["_basket"] = (
            df["customer_id"].astype(str)
            + "_"
            + df["date"].dt.strftime("%Y%m%d")
        )

    # Pre-compute basket-level stats
    basket_stats = df.groupby("_basket").agg(
        basket_value=("revenue", "sum"),
        basket_units=("quantity", "sum"),
        basket_depth=("stockcode", "nunique"),
        basket_categories=(cat_col, "nunique"),
    ).reset_index()

    # Customer-level aggregations
    features = (
        df.groupby("customer_id")
        .agg(
            # Classic RFM
            recency_days=("date", lambda x: (snapshot_date - x.max()).days),
            frequency=("transaction_id", "nunique"),
            monetary=("revenue", "sum"),
            avg_order_value=("revenue", "mean"),
            max_order_value=("revenue", "max"),
            n_items=("quantity", "sum"),
            n_unique_products=("stockcode", "nunique"),
            n_unique_categories=(cat_col, "nunique"),
            first_purchase=("date", "min"),
            last_purchase=("date", "max"),
            avg_price_paid=("price", "mean"),
            std_order_value=("revenue", "std"),
            # Behavioral
            n_baskets=("_basket", "nunique"),
            n_weekend_purchases=("date", lambda x: (x.dt.dayofweek >= 5).sum()),
        )
        .reset_index()
    )

    # Merge basket stats
    cust_basket = basket_stats.groupby(
        df.groupby("_basket")["customer_id"].first()
    ).agg(
        avg_basket_value=("basket_value", "mean"),
        avg_basket_depth=("basket_depth", "mean"),
        avg_basket_categories=("basket_categories", "mean"),
        std_basket_value=("basket_value", "std"),
    ).reset_index()
    cust_basket.columns = ["customer_id", "avg_basket_value", "avg_basket_depth", "avg_basket_categories", "std_basket_value"]
    features = features.merge(cust_basket, on="customer_id", how="left")

    # Derived features
    features["customer_lifetime_days"] = (features["last_purchase"] - features["first_purchase"]).dt.days
    features["purchase_interval"] = np.where(
        features["frequency"] > 1,
        features["customer_lifetime_days"] / (features["frequency"] - 1),
        features["customer_lifetime_days"],
    )
    features["items_per_order"] = features["n_items"] / features["frequency"]
    features["revenue_per_item"] = features["monetary"] / features["n_items"].replace(0, np.nan)
    features["order_value_cv"] = features["std_order_value"] / features["avg_order_value"].replace(0, np.nan)
    features["weekend_ratio"] = features["n_weekend_purchases"] / features["frequency"]

    # ========== NEW ENHANCED FEATURES ==========
    
    # 1. Category breadth (diversity)
    features["category_breadth"] = features["n_unique_categories"] / features["n_unique_products"].replace(0, 1)
    
    # 2. Product concentration (HHI)
    cust_product = pd.crosstab(transactions_df["customer_id"], transactions_df["stockcode"])
    cust_product = (cust_product > 0).astype(int)
    product_shares = cust_product.div(cust_product.sum(axis=1), axis=0)
    product_hhi = (product_shares ** 2).sum(axis=1).rename("product_hhi")
    features = features.set_index("customer_id").join(product_hhi).reset_index()
    features["product_hhi"] = features["product_hhi"].fillna(1.0)
    
    # 3. Repeat rate & time-to-second-purchase
    purchase_seq = transactions_df.sort_values(["customer_id", "date"]).groupby("customer_id")["date"].apply(list).reset_index()
    purchase_seq.columns = ["customer_id", "purchase_dates"]
    
    def compute_repeat_and_tt2p(dates):
        dates = sorted(dates)
        n_purchases = len(dates)
        if n_purchases < 2:
            return {"repeat_rate": 0, "tt2p_days": np.nan, "median_interpurchase_days": np.nan}
        repeat = sum(1 for i in range(1, n_purchases) if dates[i] != dates[i-1])
        gaps = [(dates[i] - dates[i-1]).days for i in range(1, n_purchases) if dates[i] != dates[i-1]]
        return {
            "repeat_rate": repeat / (n_purchases - 1) if n_purchases > 1 else 0,
            "tt2p_days": gaps[0] if gaps else np.nan,
            "median_interpurchase_days": np.median(gaps) if gaps else np.nan,
        }
    
    seq_features = purchase_seq["purchase_dates"].apply(compute_repeat_and_tt2p).apply(pd.Series)
    purchase_seq = pd.concat([purchase_seq, seq_features], axis=1)
    features = features.merge(purchase_seq[["customer_id", "repeat_rate", "tt2p_days", "median_interpurchase_days"]], on="customer_id", how="left")
    
    # 4. Switching entropy (category level)
    if "category" in transactions_df.columns:
        cust_cat_seq = transactions_df.sort_values(["customer_id", "date"]).groupby("customer_id")["category"].apply(list).reset_index()
        
        def compute_switching_entropy(cats):
            if len(cats) < 2:
                return 0
            transitions = [(cats[i], cats[i+1]) for i in range(len(cats)-1) if cats[i] != cats[i+1]]
            if not transitions:
                return 0
            trans_counts = pd.Series(transitions).value_counts()
            probs = trans_counts / trans_counts.sum()
            return entropy(probs)
        
        cat_entropy = cust_cat_seq["category"].apply(compute_switching_entropy).rename("switching_entropy")
        cust_cat_seq = pd.concat([cust_cat_seq, cat_entropy], axis=1)
        features = features.merge(cust_cat_seq[["customer_id", "switching_entropy"]], on="customer_id", how="left")
    else:
        features["switching_entropy"] = 0
    
    # 5. Net switching tendency (using switching module)
    from src.analytics.switching import compute_customer_loyalty_metrics
    loyalty_metrics = compute_customer_loyalty_metrics(transactions_df)
    loyalty_metrics = loyalty_metrics[["customer_id", "switch_count", "switch_rate", "concentration_hhi", "loyalty_segment"]]
    features = features.merge(loyalty_metrics, on="customer_id", how="left")
    
    # 5. Price sensitivity proxy (CV of price paid)
    features["price_sensitivity"] = features["std_order_value"] / features["avg_order_value"].replace(0, np.nan)
    
    # 6. Promotion affinity (if promo flags exist)
    # Already captured by price_cv / price_sensitivity
    
    # 7. Basket penetration per customer
    total_baskets = transactions_df["_basket"].nunique() if "_basket" in transactions_df.columns else transactions_df["transaction_id"].nunique()
    features["basket_penetration"] = features["n_baskets"] / total_baskets if total_baskets > 0 else 0
    
    # 8. Average days between purchases (interpurchase time)
    features["avg_days_between"] = features["median_interpurchase_days"]
    
    # 9. Seasonality of purchase behavior
    features["month_of_year"] = features["first_purchase"].dt.month
    
    # Fill NaNs
    features = features.fillna({
        "tt2p_days": features["median_interpurchase_days"],
        "switching_entropy": 0,
        "switch_count": 0,
        "switch_rate": 0,
        "concentration_hhi": 1.0,
        "loyalty_segment": "Unknown",
        "price_sensitivity": 0,
        "tt2p_days": features["purchase_interval"],
    })
    
    return features


def compute_rfm_features(
    transactions_df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Compute comprehensive RFM features per customer (backward compatible)."""
    # Use enhanced features but keep only RFM for backward compatibility
    enhanced = compute_enhanced_behavioral_features(transactions_df, snapshot_date)
    # Keep only classic RFM columns for compatibility
    rfm_cols = [
        "customer_id", "recency_days", "frequency", "monetary",
        "avg_order_value", "max_order_value", "n_items", "n_unique_products",
        "n_unique_categories", "first_purchase", "last_purchase",
        "avg_price_paid", "std_order_value", "customer_lifetime_days",
        "purchase_interval", "items_per_order", "revenue_per_item", "order_value_cv",
        "recency_segment", "frequency_segment", "monetary_segment",
    ]
    available = [c for c in rfm_cols if c in enhanced.columns]
    return enhanced[available]


def compute_segment_migration(
    transactions_df: pd.DataFrame,
    segment_col: str = "segment",
    period_freq: str = "M",
    snapshot_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Compute segment migration matrix over time.
    
    Returns transition matrix showing segment-to-segment movement per period.
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    
    if snapshot_date is None:
        snapshot_date = df["date"].max()
    
    periods = df["date"].dt.to_period("M").unique()
    periods = sorted(periods)
    
    migrations = []
    
    for i in range(1, len(periods)):
        prev_period = periods[i-1]
        curr_period = periods[i]
        
        prev_end = prev_period.to_timestamp(how="end")
        prev_start = df["date"].min()
        
        prev_data = df[(df["date"] >= prev_start) & (df["date"] <= prev_end)]
        curr_data = df[(df["date"] >= prev_period.to_timestamp()) & (df["date"] <= curr_period.to_timestamp(how="end"))]
        
        if len(prev_data) < 100 or len(curr_data) < 100:
            continue
            
        from src.analytics.segmentation import behavioral_segmentation
        
        prev_seg = behavioral_segmentation(prev_data, n_clusters=6, method="kmeans")
        curr_seg = behavioral_segmentation(curr_data, n_clusters=6, method="kmeans")
        
        common = set(prev_seg["customer_id"]) & set(curr_seg["customer_id"])
        if len(common) < 10:
            continue
            
        prev_seg = prev_seg[prev_seg["customer_id"].isin(common)]
        curr_seg = curr_seg[curr_seg["customer_id"].isin(common)]
        
        merged = prev_seg[["customer_id", "segment"]].merge(
            curr_seg[["customer_id", "segment"]], on="customer_id", suffixes=("_from", "_to")
        )
        
        trans = merged.groupby(["segment_from", "segment_to"]).size().reset_index(name="count")
        trans["period"] = str(curr_period)
        migrations.append(trans)
    
    if not migrations:
        return pd.DataFrame()
    
    return pd.concat(migrations, ignore_index=True)


def compute_segment_migration_matrix(
    transitions: pd.DataFrame,
    segment_order: List[str] = None,
) -> pd.DataFrame:
    """Convert segment transitions to a square matrix."""
    if transitions.empty:
        return pd.DataFrame()
    
    agg = transitions.groupby(["segment_from", "segment_to"])["count"].sum().reset_index()
    
    all_segments = set(transitions["segment_from"].unique()) | set(transitions["segment_to"].unique())
    if segment_order is None:
        segment_order = sorted(all_segments)
    
    matrix = pd.DataFrame(0, index=segment_order, columns=segment_order)
    for _, row in agg.iterrows():
        if row["segment_from"] in matrix.index and row["segment_to"] in matrix.columns:
            matrix.loc[row["segment_from"], row["segment_to"]] = row["count"]
    
    return matrix


def compute_segment_retention(
    transactions_df: pd.DataFrame,
    segment_assignments: pd.DataFrame,
    period_freq: str = "M",
    max_periods: int = 12,
) -> pd.DataFrame:
    """
    Compute retention curves by segment.
    
    Returns DataFrame with columns: segment, period, retention_rate, n_customers
    """
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["period"] = df["date"].dt.to_period("M")
    
    # Merge segment assignments
    df = df.merge(segment_assignments[["customer_id", "segment"]], on="customer_id", how="left")
    
    # Get cohort (first period per customer per segment)
    cohort = df.groupby(["customer_id", "segment"])["period"].min().reset_index()
    cohort.columns = ["customer_id", "segment", "cohort_period"]
    
    # Merge back
    df = df.merge(cohort, on=["customer_id", "segment"])
    df["period_number"] = (df["period"] - df["cohort_period"]).apply(lambda x: x.n)
    df = df[df["period_number"] >= 0]
    
    # Retention by segment and period
    retention = (
        df.groupby(["segment", "cohort_period", "period_number"])["customer_id"]
        .nunique()
        .reset_index()
    )
    
    cohort_sizes = retention[retention["period_number"] == 0][["segment", "cohort_period", "customer_id"]].rename(
        columns={"customer_id": "cohort_size"}
    )
    
    retention = retention.merge(cohort_sizes, on=["segment", "cohort_period"])
    retention["retention_rate"] = retention["customer_id"] / retention["cohort_size"]
    
    # Average across cohorts
    avg_retention = (
        retention.groupby(["segment", "period_number"])["retention_rate"]
        .mean()
        .reset_index()
    )
    
    return avg_retention


def compute_umap_projection(
    features: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Compute UMAP projection for visualization."""
    if not UMAP_AVAILABLE:
        # Fallback to PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        return pca.fit_transform(features)
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def compute_pca_projection(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Compute PCA projection for visualization."""
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(features)


def label_segment_business(profile: pd.Series, segment_id: int) -> Tuple[str, str]:
    """
    Generate business label and action recommendation for a segment.
    
    Returns (label, action)
    """
    # Extract key metrics
    recency = profile.get("recency_days", profile.get("avg_recency", 999))
    freq = profile.get("frequency", profile.get("avg_frequency", profile.get("purchase_frequency", 0)))
    monetary = profile.get("monetary", profile.get("avg_monetary", profile.get("total_revenue", 0)))
    depth = profile.get("avg_basket_depth", profile.get("n_products", 0))
    switch = profile.get("switch_rate", profile.get("switching_entropy", 0))
    repeat = profile.get("repeat_rate", 0)
    
    # High value
    if monetary > profile.get("monetary_75", monetary * 2) and freq > profile.get("freq_75", freq * 2):
        return "Champions", "Protect & reward: VIP access, early product previews, exclusive offers"
    
    if monetary > profile.get("monetary_75", monetary * 2):
        return "Big Spenders", "Cross-sell premium bundles, increase basket depth"
    
    if freq > profile.get("freq_75", freq * 2) and recency < 30:
        return "Frequent Buyers", "Loyalty program, subscription offers, referral incentives"
    
    if recency < 30 and freq < profile.get("freq_25", 1):
        return "New Customers", "Welcome series, second-purchase incentive, onboarding"
    
    if recency > 180 and monetary > profile.get("monetary_50", monetary):
        return "Dormant High Value", "Win-back campaign with personalized offer, reactivation discount"
    
    if recency > 180:
        return "Churned", "Low-cost reactivation, survey for feedback"
    
    if switch > 0.5:
        return "Switchers", "Lock-in with bundles, personalized recommendations, reduce friction"
    
    if repeat < 0.2:
        return "One-time Buyers", "Second-purchase trigger, product education, sample offers"
    
    if depth > profile.get("depth_75", depth * 2):
        return "Variety Seekers", "Cross-sell new arrivals, recommendation engine, subscription box"
    
    return "Regular", "Standard engagement, periodic promotions, feedback surveys"


def get_segment_recommendations(segment_name: str) -> Dict[str, List[str]]:
    """Get detailed action playbook for each segment."""
    playbooks = {
        "Champions": {
            "primary": ["VIP tier access", "Early product access", "Exclusive events"],
            "secondary": ["Referral program", "Co-creation feedback", "Birthday/anniversary gifts"],
            "avoid": ["Aggressive discounting", "Generic mass emails"],
        },
        "Big Spenders": {
            "primary": ["Premium bundles", "Cross-sell premium SKUs", "Concierge service"],
            "secondary": ["Volume discounts", "Subscription auto-replenish"],
            "avoid": ["Low-margin promotions", "Irrelevant cross-sell"],
        },
        "Frequent Buyers": {
            "primary": ["Loyalty points accelerator", "Subscription auto-ship", "Referral bonus"],
            "secondary": ["Predictive replenishment", "Exclusive content"],
            "avoid": ["Win-back campaigns", "Generic newsletters"],
        },
        "New Customers": {
            "primary": ["Welcome sequence", "Second-purchase discount (15-20%)", "Product education"],
            "secondary": ["First-review incentive", "Social proof showcase"],
            "avoid": ["Premium upsell immediately", "Complex loyalty terms"],
        },
        "Dormant High Value": {
            "primary": ["Personalized win-back offer (20-30%)", "New arrival preview", "Dedicated support"],
            "secondary": ["Survey for churn reason", "Re-engagement email series"],
            "avoid": ["Generic blast emails", "Irrelevant cross-sell"],
        },
        "Churned": {
            "primary": ["Low-cost reactivation", "Exit survey", "Competitor intel"],
            "secondary": ["Seasonal re-engagement", "New category intro"],
            "avoid": ["High-cost retention", "Complex offers"],
        },
        "Switchers": {
            "primary": ["Bundle lock-in", "Personalized recommendations", "Reduce switching friction"],
            "secondary": ["Category loyalty bonus", "Subscription for staples"],
            "avoid": ["Price wars", "Generic promotions"],
        },
        "One-time Buyers": {
            "primary": ["Second-purchase trigger (24-48h)", "Product how-to content", "Sample with next order"],
            "secondary": ["Review request", "Social proof emails"],
            "avoid": ["High-value upsell", "Complex loyalty enrollment"],
        },
        "Variety Seekers": {
            "primary": ["New arrival alerts", "Discovery box subscription", "AI recommendations"],
            "secondary": ["Cross-category bundles", "Tasting/sample packs"],
            "avoid": ["Same-product repeat push", "Static assortment"],
        },
        "Regular": {
            "primary": ["Seasonal promotions", "Feedback surveys", "Standard loyalty accrual"],
            "secondary": ["Category expansion", "Occasion-based offers"],
            "avoid": ["Over-communication", "Premium positioning"],
        },
    }
    return playbooks.get(segment_name, {
        "primary": ["Standard engagement", "Periodic promotions"],
        "secondary": ["Feedback collection", "Category expansion"],
        "avoid": ["Over-personalization", "Aggressive tactics"],
    })


def compute_umap_projection(
    features: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Compute UMAP projection for visualization."""
    if not UMAP_AVAILABLE:
        # Fallback to PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        return pca.fit_transform(features)
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def compute_pca_projection(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Compute PCA projection for visualization."""
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(features)


def label_segment_business(profile: pd.Series, segment_id: int) -> Tuple[str, str]:
    """
    Generate business label and action recommendation for a segment.
    
    Returns (label, action)
    """
    # Extract key metrics
    recency = profile.get("recency_days", profile.get("avg_recency", 999))
    freq = profile.get("frequency", profile.get("avg_frequency", profile.get("purchase_frequency", 0)))
    monetary = profile.get("monetary", profile.get("avg_monetary", profile.get("total_revenue", 0)))
    depth = profile.get("avg_basket_depth", profile.get("n_products", 0))
    switch = profile.get("switch_rate", profile.get("switching_entropy", 0))
    repeat = profile.get("repeat_rate", 0)
    
    # High value
    if monetary > profile.get("monetary_75", monetary * 2) and freq > profile.get("freq_75", freq * 2):
        return "Champions", "Protect & reward: VIP access, early product previews, exclusive offers"
    
    if monetary > profile.get("monetary_75", monetary * 2):
        return "Big Spenders", "Cross-sell premium bundles, increase basket depth"
    
    if freq > profile.get("freq_75", freq * 2) and recency < 30:
        return "Frequent Buyers", "Loyalty program, subscription offers, referral incentives"
    
    if recency < 30 and freq < profile.get("freq_25", 1):
        return "New Customers", "Welcome series, second-purchase incentive, onboarding"
    
    if recency > 180 and monetary > profile.get("monetary_50", monetary):
        return "Dormant High Value", "Win-back campaign with personalized offer, reactivation discount"
    
    if recency > 180:
        return "Churned", "Low-cost reactivation, survey for feedback"
    
    if switch > 0.5:
        return "Switchers", "Lock-in with bundles, personalized recommendations, reduce friction"
    
    if repeat < 0.2:
        return "One-time Buyers", "Second-purchase trigger, product education, sample offers"
    
    if depth > profile.get("depth_75", depth * 2):
        return "Variety Seekers", "Cross-sell new arrivals, recommendation engine, subscription box"
    
    return "Regular", "Standard engagement, periodic promotions, feedback surveys"


def get_segment_recommendations(segment_name: str) -> Dict[str, List[str]]:
    """Get detailed action playbook for each segment."""
    playbooks = {
        "Champions": {
            "primary": ["VIP tier access", "Early product access", "Exclusive events"],
            "secondary": ["Referral program", "Co-creation feedback", "Birthday/anniversary gifts"],
            "avoid": ["Aggressive discounting", "Generic mass emails"],
        },
        "Big Spenders": {
            "primary": ["Premium bundles", "Cross-sell premium SKUs", "Concierge service"],
            "secondary": ["Volume discounts", "Subscription auto-replenish"],
            "avoid": ["Low-margin promotions", "Irrelevant cross-sell"],
        },
        "Frequent Buyers": {
            "primary": ["Loyalty points accelerator", "Subscription auto-ship", "Referral bonus"],
            "secondary": ["Predictive replenishment", "Exclusive content"],
            "avoid": ["Win-back campaigns", "Generic newsletters"],
        },
        "New Customers": {
            "primary": ["Welcome sequence", "Second-purchase discount (15-20%)", "Product education"],
            "secondary": ["First-review incentive", "Social proof showcase"],
            "avoid": ["Premium upsell immediately", "Complex loyalty terms"],
        },
        "Dormant High Value": {
            "primary": ["Personalized win-back offer (20-30%)", "New arrival preview", "Dedicated support"],
            "secondary": ["Survey for churn reason", "Re-engagement email series"],
            "avoid": ["Generic blast emails", "Irrelevant cross-sell"],
        },
        "Churned": {
            "primary": ["Low-cost reactivation", "Exit survey", "Competitor intel"],
            "secondary": ["Seasonal re-engagement", "New category intro"],
            "avoid": ["High-cost retention", "Complex offers"],
        },
        "Switchers": {
            "primary": ["Bundle lock-in", "Personalized recommendations", "Reduce switching friction"],
            "secondary": ["Category loyalty bonus", "Subscription for staples"],
            "avoid": ["Price wars", "Generic promotions"],
        },
        "One-time Buyers": {
            "primary": ["Second-purchase trigger (24-48h)", "Product how-to content", "Sample with next order"],
            "secondary": ["Review request", "Social proof emails"],
            "avoid": ["High-value upsell", "Complex loyalty enrollment"],
        },
        "Variety Seekers": {
            "primary": ["New arrival alerts", "Discovery box subscription", "AI recommendations"],
            "secondary": ["Cross-category bundles", "Tasting/sample packs"],
            "avoid": ["Same-product repeat push", "Static assortment"],
        },
        "Regular": {
            "primary": ["Seasonal promotions", "Feedback surveys", "Standard loyalty accrual"],
            "secondary": ["Category expansion", "Occasion-based offers"],
            "avoid": ["Over-communication", "Premium positioning"],
        },
    }
    return playbooks.get(segment_name, {
        "primary": ["Standard engagement", "Periodic promotions"],
        "secondary": ["Feedback collection", "Category expansion"],
        "avoid": ["Over-personalization", "Aggressive tactics"],
    })


def compute_umap_projection(
    features: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Compute UMAP projection for visualization."""
    if not UMAP_AVAILABLE:
        # Fallback to PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        return pca.fit_transform(features)
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def compute_pca_projection(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Compute PCA projection for visualization."""
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(features)


def label_segment_business(profile: pd.Series, segment_id: int) -> Tuple[str, str]:
    """
    Generate business label and action recommendation for a segment.
    
    Returns (label, action)
    """
    # Extract key metrics
    recency = profile.get("recency_days", profile.get("avg_recency", 999))
    freq = profile.get("frequency", profile.get("avg_frequency", profile.get("purchase_frequency", 0)))
    monetary = profile.get("monetary", profile.get("avg_monetary", profile.get("total_revenue", 0)))
    depth = profile.get("avg_basket_depth", profile.get("n_products", 0))
    switch = profile.get("switch_rate", profile.get("switching_entropy", 0))
    repeat = profile.get("repeat_rate", 0)
    
    # High value
    if monetary > profile.get("monetary_75", monetary * 2) and freq > profile.get("freq_75", freq * 2):
        return "Champions", "Protect & reward: VIP access, early product previews, exclusive offers"
    
    if monetary > profile.get("monetary_75", monetary * 2):
        return "Big Spenders", "Cross-sell premium bundles, increase basket depth"
    
    if freq > profile.get("freq_75", freq * 2) and recency < 30:
        return "Frequent Buyers", "Loyalty program, subscription offers, referral incentives"
    
    if recency < 30 and freq < profile.get("freq_25", 1):
        return "New Customers", "Welcome series, second-purchase incentive, onboarding"
    
    if recency > 180 and monetary > profile.get("monetary_50", monetary):
        return "Dormant High Value", "Win-back campaign with personalized offer, reactivation discount"
    
    if recency > 180:
        return "Churned", "Low-cost reactivation, survey for feedback"
    
    if switch > 0.5:
        return "Switchers", "Lock-in with bundles, personalized recommendations, reduce friction"
    
    if repeat < 0.2:
        return "One-time Buyers", "Second-purchase trigger, product education, sample offers"
    
    if depth > profile.get("depth_75", depth * 2):
        return "Variety Seekers", "Cross-sell new arrivals, recommendation engine, subscription box"
    
    return "Regular", "Standard engagement, periodic promotions, feedback surveys"


def get_segment_recommendations(segment_name: str) -> Dict[str, List[str]]:
    """Get detailed action playbook for each segment."""
    playbooks = {
        "Champions": {
            "primary": ["VIP tier access", "Early product access", "Exclusive events"],
            "secondary": ["Referral program", "Co-creation feedback", "Birthday/anniversary gifts"],
            "avoid": ["Aggressive discounting", "Generic mass emails"],
        },
        "Big Spenders": {
            "primary": ["Premium bundles", "Cross-sell premium SKUs", "Concierge service"],
            "secondary": ["Volume discounts", "Subscription auto-replenish"],
            "avoid": ["Low-margin promotions", "Irrelevant cross-sell"],
        },
        "Frequent Buyers": {
            "primary": ["Loyalty points accelerator", "Subscription auto-ship", "Referral bonus"],
            "secondary": ["Predictive replenishment", "Exclusive content"],
            "avoid": ["Win-back campaigns", "Generic newsletters"],
        },
        "New Customers": {
            "primary": ["Welcome sequence", "Second-purchase discount (15-20%)", "Product education"],
            "secondary": ["First-review incentive", "Social proof showcase"],
            "avoid": ["Premium upsell immediately", "Complex loyalty terms"],
        },
        "Dormant High Value": {
            "primary": ["Personalized win-back offer (20-30%)", "New arrival preview", "Dedicated support"],
            "secondary": ["Survey for churn reason", "Re-engagement email series"],
            "avoid": ["Generic blast emails", "Irrelevant cross-sell"],
        },
        "Churned": {
            "primary": ["Low-cost reactivation", "Exit survey", "Competitor intel"],
            "secondary": ["Seasonal re-engagement", "New category intro"],
            "avoid": ["High-cost retention", "Complex offers"],
        },
        "Switchers": {
            "primary": ["Bundle lock-in", "Personalized recommendations", "Reduce switching friction"],
            "secondary": ["Category loyalty bonus", "Subscription for staples"],
            "avoid": ["Price wars", "Generic promotions"],
        },
        "One-time Buyers": {
            "primary": ["Second-purchase trigger (24-48h)", "Product how-to content", "Sample with next order"],
            "secondary": ["Review request", "Social proof emails"],
            "avoid": ["High-value upsell", "Complex loyalty enrollment"],
        },
        "Variety Seekers": {
            "primary": ["New arrival alerts", "Discovery box subscription", "AI recommendations"],
            "secondary": ["Cross-category bundles", "Tasting/sample packs"],
            "avoid": ["Same-product repeat push", "Static assortment"],
        },
        "Regular": {
            "primary": ["Seasonal promotions", "Feedback surveys", "Standard loyalty accrual"],
            "secondary": ["Category expansion", "Occasion-based offers"],
            "avoid": ["Over-communication", "Premium positioning"],
        },
    }
    return playbooks.get(segment_name, {
        "primary": ["Standard engagement", "Periodic promotions"],
        "secondary": ["Feedback collection", "Category expansion"],
        "avoid": ["Over-personalization", "Aggressive tactics"],
    })


def compute_umap_projection(
    features: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Compute UMAP projection for visualization."""
    if not UMAP_AVAILABLE:
        # Fallback to PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        return pca.fit_transform(features)
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def compute_pca_projection(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Compute PCA projection for visualization."""
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(features)


def label_segment_business(profile: pd.Series, segment_id: int) -> Tuple[str, str]:
    """
    Generate business label and action recommendation for a segment.
    
    Returns (label, action)
    """
    # Extract key metrics
    recency = profile.get("recency_days", profile.get("avg_recency", 999))
    freq = profile.get("frequency", profile.get("avg_frequency", profile.get("purchase_frequency", 0)))
    monetary = profile.get("monetary", profile.get("avg_monetary", profile.get("total_revenue", 0)))
    depth = profile.get("avg_basket_depth", profile.get("n_products", 0))
    switch = profile.get("switch_rate", profile.get("switching_entropy", 0))
    repeat = profile.get("repeat_rate", 0)
    
    # High value
    if monetary > profile.get("monetary_75", monetary * 2) and freq > profile.get("freq_75", freq * 2):
        return "Champions", "Protect & reward: VIP access, early product previews, exclusive offers"
    
    if monetary > profile.get("monetary_75", monetary * 2):
        return "Big Spenders", "Cross-sell premium bundles, increase basket depth"
    
    if freq > profile.get("freq_75", freq * 2) and recency < 30:
        return "Frequent Buyers", "Loyalty program, subscription offers, referral incentives"
    
    if recency < 30 and freq < profile.get("freq_25", 1):
        return "New Customers", "Welcome series, second-purchase incentive, onboarding"
    
    if recency > 180 and monetary > profile.get("monetary_50", monetary):
        return "Dormant High Value", "Win-back campaign with personalized offer, reactivation discount"
    
    if recency > 180:
        return "Churned", "Low-cost reactivation, survey for feedback"
    
    if switch > 0.5:
        return "Switchers", "Lock-in with bundles, personalized recommendations, reduce friction"
    
    if repeat < 0.2:
        return "One-time Buyers", "Second-purchase trigger, product education, sample offers"
    
    if depth > profile.get("depth_75", depth * 2):
        return "Variety Seekers", "Cross-sell new arrivals, recommendation engine, subscription box"
    
    return "Regular", "Standard engagement, periodic promotions, feedback surveys"


def get_segment_recommendations(segment_name: str) -> Dict[str, List[str]]:
    """Get detailed action playbook for each segment."""
    playbooks = {
        "Champions": {
            "primary": ["VIP tier access", "Early product access", "Exclusive events"],
            "secondary": ["Referral program", "Co-creation feedback", "Birthday/anniversary gifts"],
            "avoid": ["Aggressive discounting", "Generic mass emails"],
        },
        "Big Spenders": {
            "primary": ["Premium bundles", "Cross-sell premium SKUs", "Concierge service"],
            "secondary": ["Volume discounts", "Subscription auto-replenish"],
            "avoid": ["Low-margin promotions", "Irrelevant cross-sell"],
        },
        "Frequent Buyers": {
            "primary": ["Loyalty points accelerator", "Subscription auto-ship", "Referral bonus"],
            "secondary": ["Predictive replenishment", "Exclusive content"],
            "avoid": ["Win-back campaigns", "Generic newsletters"],
        },
        "New Customers": {
            "primary": ["Welcome sequence", "Second-purchase discount (15-20%)", "Product education"],
            "secondary": ["First-review incentive", "Social proof showcase"],
            "avoid": ["Premium upsell immediately", "Complex loyalty terms"],
        },
        "Dormant High Value": {
            "primary": ["Personalized win-back offer (20-30%)", "New arrival preview", "Dedicated support"],
            "secondary": ["Survey for churn reason", "Re-engagement email series"],
            "avoid": ["Generic blast emails", "Irrelevant cross-sell"],
        },
        "Churned": {
            "primary": ["Low-cost reactivation", "Exit survey", "Competitor intel"],
            "secondary": ["Seasonal re-engagement", "New category intro"],
            "avoid": ["High-cost retention", "Complex offers"],
        },
        "Switchers": {
            "primary": ["Bundle lock-in", "Personalized recommendations", "Reduce switching friction"],
            "secondary": ["Category loyalty bonus", "Subscription for staples"],
            "avoid": ["Price wars", "Generic promotions"],
        },
        "One-time Buyers": {
            "primary": ["Second-purchase trigger (24-48h)", "Product how-to content", "Sample with next order"],
            "secondary": ["Review request", "Social proof emails"],
            "avoid": ["High-value upsell", "Complex loyalty enrollment"],
        },
        "Variety Seekers": {
            "primary": ["New arrival alerts", "Discovery box subscription", "AI recommendations"],
            "secondary": ["Cross-category bundles", "Tasting/sample packs"],
            "avoid": ["Same-product repeat push", "Static assortment"],
        },
        "Regular": {
            "primary": ["Seasonal promotions", "Feedback surveys", "Standard loyalty accrual"],
            "secondary": ["Category expansion", "Occasion-based offers"],
            "avoid": ["Over-communication", "Premium positioning"],
        },
    }
    return playbooks.get(segment_name, {
        "primary": ["Standard engagement", "Periodic promotions"],
        "secondary": ["Feedback collection", "Category expansion"],
        "avoid": ["Over-personalization", "Aggressive tactics"],
    })


def compute_umap_projection(
    features: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Compute UMAP projection for visualization."""
    if not UMAP_AVAILABLE:
        # Fallback to PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        return pca.fit_transform(features)
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def compute_pca_projection(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Compute PCA projection for visualization."""
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(features)


def label_segment_business(profile: pd.Series, segment_id: int) -> Tuple[str, str]:
    """
    Generate business label and action recommendation for a segment.
    
    Returns (label, action)
    """
    # Extract key metrics
    recency = profile.get("recency_days", profile.get("avg_recency", 999))
    freq = profile.get("frequency", profile.get("avg_frequency", profile.get("purchase_frequency", 0)))
    monetary = profile.get("monetary", profile.get("avg_monetary", profile.get("total_revenue", 0)))
    depth = profile.get("avg_basket_depth", profile.get("n_products", 0))
    switch = profile.get("switch_rate", profile.get("switching_entropy", 0))
    repeat = profile.get("repeat_rate", 0)
    
    # High value
    if monetary > profile.get("monetary_75", monetary * 2) and freq > profile.get("freq_75", freq * 2):
        return "Champions", "Protect & reward: VIP access, early product previews, exclusive offers"
    
    if monetary > profile.get("monetary_75", monetary * 2):
        return "Big Spenders", "Cross-sell premium bundles, increase basket depth"
    
    if freq > profile.get("freq_75", freq * 2) and recency < 30:
        return "Frequent Buyers", "Loyalty program, subscription offers, referral incentives"
    
    if recency < 30 and freq < profile.get("freq_25", 1):
        return "New Customers", "Welcome series, second-purchase incentive, onboarding"
    
    if recency > 180 and monetary > profile.get("monetary_50", monetary):
        return "Dormant High Value", "Win-back campaign with personalized offer, reactivation discount"
    
    if recency > 180:
        return "Churned", "Low-cost reactivation, survey for feedback"
    
    if switch > 0.5:
        return "Switchers", "Lock-in with bundles, personalized recommendations, reduce friction"
    
    if repeat < 0.2:
        return "One-time Buyers", "Second-purchase trigger, product education, sample offers"
    
    if depth > profile.get("depth_75", depth * 2):
        return "Variety Seekers", "Cross-sell new arrivals, recommendation engine, subscription box"
    
    return "Regular", "Standard engagement, periodic promotions, feedback surveys"


def get_segment_recommendations(segment_name: str) -> Dict[str, List[str]]:
    """Get detailed action playbook for each segment."""
    playbooks = {
        "Champions": {
            "primary": ["VIP tier access", "Early product access", "Exclusive events"],
            "secondary": ["Referral program", "Co-creation feedback", "Birthday/anniversary gifts"],
            "avoid": ["Aggressive discounting", "Generic mass emails"],
        },
        "Big Spenders": {
            "primary": ["Premium bundles", "Cross-sell premium SKUs", "Concierge service"],
            "secondary": ["Volume discounts", "Subscription auto-replenish"],
            "avoid": ["Low-margin promotions", "Irrelevant cross-sell"],
        },
        "Frequent Buyers": {
            "primary": ["Loyalty points accelerator", "Subscription auto-ship", "Referral bonus"],
            "secondary": ["Predictive replenishment", "Exclusive content"],
            "avoid": ["Win-back campaigns", "Generic newsletters"],
        },
        "New Customers": {
            "primary": ["Welcome sequence", "Second-purchase discount (15-20%)", "Product education"],
            "secondary": ["First-review incentive", "Social proof showcase"],
            "avoid": ["Premium upsell immediately", "Complex loyalty terms"],
        },
        "Dormant High Value": {
            "primary": ["Personalized win-back offer (20-30%)", "New arrival preview", "Dedicated support"],
            "secondary": ["Survey for churn reason", "Re-engagement email series"],
            "avoid": ["Generic blast emails", "Irrelevant cross-sell"],
        },
        "Churned": {
            "primary": ["Low-cost reactivation", "Exit survey", "Competitor intel"],
            "secondary": ["Seasonal re-engagement", "New category intro"],
            "avoid": ["High-cost retention", "Complex offers"],
        },
        "Switchers": {
            "primary": ["Bundle lock-in", "Personalized recommendations", "Reduce switching friction"],
            "secondary": ["Category loyalty bonus", "Subscription for staples"],
            "avoid": ["Price wars", "Generic promotions"],
        },
        "One-time Buyers": {
            "primary": ["Second-purchase trigger (24-48h)", "Product how-to content", "Sample with next order"],
            "secondary": ["Review request", "Social proof emails"],
            "avoid": ["High-value upsell", "Complex loyalty enrollment"],
        },
        "Variety Seekers": {
            "primary": ["New arrival alerts", "Discovery box subscription", "AI recommendations"],
            "secondary": ["Cross-category bundles", "Tasting/sample packs"],
            "avoid": ["Same-product repeat push", "Static assortment"],
        },
        "Regular": {
            "primary": ["Seasonal promotions", "Feedback surveys", "Standard loyalty accrual"],
            "secondary": ["Category expansion", "Occasion-based offers"],
            "avoid": ["Over-communication", "Premium positioning"],
        },
    }
    return playbooks.get(segment_name, {
        "primary": ["Standard engagement", "Periodic promotions"],
        "secondary": ["Feedback collection", "Category expansion"],
        "avoid": ["Over-personalization", "Aggressive tactics"],
    })


def compute_umap_projection(
    features: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Compute UMAP projection for visualization."""
    if not UMAP_AVAILABLE:
        # Fallback to PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        return pca.fit_transform(features)
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def compute_pca_projection(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Compute PCA projection for visualization."""
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(features)


def label_segment_business(profile: pd.Series, segment_id: int) -> Tuple[str, str]:
    """
    Generate business label and action recommendation for a segment.
    
    Returns (label, action)
    """
    # Extract key metrics
    recency = profile.get("recency_days", profile.get("avg_recency", 999))
    freq = profile.get("frequency", profile.get("avg_frequency", profile.get("purchase_frequency", 0)))
    monetary = profile.get("monetary", profile.get("avg_monetary", profile.get("total_revenue", 0)))
    depth = profile.get("avg_basket_depth", profile.get("n_products", 0))
    switch = profile.get("switch_rate", profile.get("switching_entropy", 0))
    repeat = profile.get("repeat_rate", 0)
    
    # High value
    if monetary > profile.get("monetary_75", monetary * 2) and freq > profile.get("freq_75", freq * 2):
        return "Champions", "Protect & reward: VIP access, early product previews, exclusive offers"
    
    if monetary > profile.get("monetary_75", monetary * 2):
        return "Big Spenders", "Cross-sell premium bundles, increase basket depth"
    
    if freq > profile.get("freq_75", freq * 2) and recency < 30:
        return "Frequent Buyers", "Loyalty program, subscription offers, referral incentives"
    
    if recency < 30 and freq < profile.get("freq_25", 1):
        return "New Customers", "Welcome series, second-purchase incentive, onboarding"
    
    if recency > 180 and monetary > profile.get("monetary_50", monetary):
        return "Dormant High Value", "Win-back campaign with personalized offer, reactivation discount"
    
    if recency > 180:
        return "Churned", "Low-cost reactivation, survey for feedback"
    
    if switch > 0.5:
        return "Switchers", "Lock-in with bundles, personalized recommendations, reduce friction"
    
    if repeat < 0.2:
        return "One-time Buyers", "Second-purchase trigger, product education, sample offers"
    
    if depth > profile.get("depth_75", depth * 2):
        return "Variety Seekers", "Cross-sell new arrivals, recommendation engine, subscription box"
    
    return "Regular", "Standard engagement, periodic promotions, feedback surveys"


def get_segment_recommendations(segment_name: str) -> Dict[str, List[str]]:
    """Get detailed action playbook for each segment."""
    playbooks = {
        "Champions": {
            "primary": ["VIP tier access", "Early product access", "Exclusive events"],
            "secondary": ["Referral program", "Co-creation feedback", "Birthday/anniversary gifts"],
            "avoid": ["Aggressive discounting", "Generic mass emails"],
        },
        "Big Spenders": {
            "primary": ["Premium bundles", "Cross-sell premium SKUs", "Concierge service"],
            "secondary": ["Volume discounts", "Subscription auto-replenish"],
            "avoid": ["Low-margin promotions", "Irrelevant cross-sell"],
        },
        "Frequent Buyers": {
            "primary": ["Loyalty points accelerator", "Subscription auto-ship", "Referral bonus"],
            "secondary": ["Predictive replenishment", "Exclusive content"],
            "avoid": ["Win-back campaigns", "Generic newsletters"],
        },
        "New Customers": {
            "primary": ["Welcome sequence", "Second-purchase discount (15-20%)", "Product education"],
            "secondary": ["First-review incentive", "Social proof showcase"],
            "avoid": ["Premium upsell immediately", "Complex loyalty terms"],
        },
        "Dormant High Value": {
            "primary": ["Personalized win-back offer (20-30%)", "New arrival preview", "Dedicated support"],
            "secondary": ["Survey for churn reason", "Re-engagement email series"],
            "avoid": ["Generic blast emails", "Irrelevant cross-sell"],
        },
        "Churned": {
            "primary": ["Low-cost reactivation", "Exit survey", "Competitor intel"],
            "secondary": ["Seasonal re-engagement", "New category intro"],
            "avoid": ["High-cost retention", "Complex offers"],
        },
        "Switchers": {
            "primary": ["Bundle lock-in", "Personalized recommendations", "Reduce switching friction"],
            "secondary": ["Category loyalty bonus", "Subscription for staples"],
            "avoid": ["Price wars", "Generic promotions"],
        },
        "One-time Buyers": {
            "primary": ["Second-purchase trigger (24-48h)", "Product how-to content", "Sample with next order"],
            "secondary": ["Review request", "Social proof emails"],
            "avoid": ["High-value upsell", "Complex loyalty enrollment"],
        },
        "Variety Seekers": {
            "primary": ["New arrival alerts", "Discovery box subscription", "AI recommendations"],
            "secondary": ["Cross-category bundles", "Tasting/sample packs"],
            "avoid": ["Same-product repeat push", "Static assortment"],
        },
        "Regular": {
            "primary": ["Seasonal promotions", "Feedback surveys", "Standard loyalty accrual"],
            "secondary": ["Category expansion", "Occasion-based offers"],
            "avoid": ["Over-communication", "Premium positioning"],
        },
    }
    return playbooks.get(segment_name, {
        "primary": ["Standard engagement", "Periodic promotions"],
        "secondary": ["Feedback collection", "Category expansion"],
        "avoid": ["Over-personalization", "Aggressive tactics"],
    })


def compute_umap_projection(
    features: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Compute UMAP projection for visualization."""
    if not UMAP_AVAILABLE:
        # Fallback to PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        return pca.fit_transform(features)
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def compute_pca_projection(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Compute PCA projection for visualization."""
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(features)


def label_segment_business(profile: pd.Series, segment_id: int) -> Tuple[str, str]:
    """
    Generate business label and action recommendation for a segment.
    
    Returns (label, action)
    """
    # Extract key metrics
    recency = profile.get("recency_days", profile.get("avg_recency", 999))
    freq = profile.get("frequency", profile.get("avg_frequency", profile.get("purchase_frequency", 0)))
    monetary = profile.get("monetary", profile.get("avg_monetary", profile.get("total_revenue", 0)))
    depth = profile.get("avg_basket_depth", profile.get("n_products", 0))
    switch = profile.get("switch_rate", profile.get("switching_entropy", 0))
    repeat = profile.get("repeat_rate", 0)
    
    # High value
    if monetary > profile.get("monetary_75", monetary * 2) and freq > profile.get("freq_75", freq * 2):
        return "Champions", "Protect & reward: VIP access, early product previews, exclusive offers"
    
    if monetary > profile.get("monetary_75", monetary * 2):
        return "Big Spenders", "Cross-sell premium bundles, increase basket depth"
    
    if freq > profile.get("freq_75", freq * 2) and recency < 30:
        return "Frequent Buyers", "Loyalty program, subscription offers, referral incentives"
    
    if recency < 30 and freq < profile.get("freq_25", 1):
        return "New Customers", "Welcome series, second-purchase incentive, onboarding"
    
    if recency > 180 and monetary > profile.get("monetary_50", monetary):
        return "Dormant High Value", "Win-back campaign with personalized offer, reactivation discount"
    
    if recency > 180:
        return "Churned", "Low-cost reactivation, survey for feedback"
    
    if switch > 0.5:
        return "Switchers", "Lock-in with bundles, personalized recommendations, reduce friction"
    
    if repeat < 0.2:
        return "One-time Buyers", "Second-purchase trigger, product education, sample offers"
    
    if depth > profile.get("depth_75", depth * 2):
        return "Variety Seekers", "Cross-sell new arrivals, recommendation engine, subscription box"
    
    return "Regular", "Standard engagement, periodic promotions, feedback surveys"


def get_segment_recommendations(segment_name: str) -> Dict[str, List[str]]:
    """Get detailed action playbook for each segment."""
    playbooks = {
        "Champions": {
            "primary": ["VIP tier access", "Early product access", "Exclusive events"],
            "secondary": ["Referral program", "Co-creation feedback", "Birthday/anniversary gifts"],
            "avoid": ["Aggressive discounting", "Generic mass emails"],
        },
        "Big Spenders": {
            "primary": ["Premium bundles", "Cross-sell premium SKUs", "Concierge service"],
            "secondary": ["Volume discounts", "Subscription auto-replenish"],
            "avoid": ["Low-margin promotions", "Irrelevant cross-sell"],
        },
        "Frequent Buyers": {
            "primary": ["Loyalty points accelerator", "Subscription auto-ship", "Referral bonus"],
            "secondary": ["Predictive replenishment", "Exclusive content"],
            "avoid": ["Win-back campaigns", "Generic newsletters"],
        },
        "New Customers": {
            "primary": ["Welcome sequence", "Second-purchase discount (15-20%)", "Product education"],
            "secondary": ["First-review incentive", "Social proof showcase"],
            "avoid": ["Premium upsell immediately", "Complex loyalty terms"],
        },
        "Dormant High Value": {
            "primary": ["Personalized win-back offer (20-30%)", "New arrival preview", "Dedicated support"],
            "secondary": ["Survey for churn reason", "Re-engagement email series"],
            "avoid": ["Generic blast emails", "Irrelevant cross-sell"],
        },
        "Churned": {
            "primary": ["Low-cost reactivation", "Exit survey", "Competitor intel"],
            "secondary": ["Seasonal re-engagement", "New category intro"],
            "avoid": ["High-cost retention", "Complex offers"],
        },
        "Switchers": {
            "primary": ["Bundle lock-in", "Personalized recommendations", "Reduce switching friction"],
            "secondary": ["Category loyalty bonus", "Subscription for staples"],
            "avoid": ["Price wars", "Generic promotions"],
        },
        "One-time Buyers": {
            "primary": ["Second-purchase trigger (24-48h)", "Product how-to content", "Sample with next order"],
            "secondary": ["Review request", "Social proof emails"],
            "avoid": ["High-value upsell", "Complex loyalty enrollment"],
        },
        "Variety Seekers": {
            "primary": ["New arrival alerts", "Discovery box subscription", "AI recommendations"],
            "secondary": ["Cross-category bundles", "Tasting/sample packs"],
            "avoid": ["Same-product repeat push", "Static assortment"],
        },
        "Regular": {
            "primary": ["Seasonal promotions", "Feedback surveys", "Standard loyalty accrual"],
            "secondary": ["Category expansion", "Occasion-based offers"],
            "avoid": ["Over-communication", "Premium positioning"],
        },
    }
    return playbooks.get(segment_name, {
        "primary": ["Standard engagement", "Periodic promotions"],
        "secondary": ["Feedback collection", "Category expansion"],
        "avoid": ["Over-personalization", "Aggressive tactics"],
    })


def compute_umap_projection(
    features: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Compute UMAP projection for visualization."""
    if not UMAP_AVAILABLE:
        # Fallback to PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        return pca.fit_transform(features)
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def compute_pca_projection(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Compute PCA projection for visualization."""
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(features)
