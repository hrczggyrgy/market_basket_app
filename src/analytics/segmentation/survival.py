"""Survival Analysis for Customer Retention using lifelines."""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter

    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

from src.analytics.schemas import SURVIVAL_DIAGNOSTICS, SURVIVAL_PREDICTIONS, check


def survival_analysis(
    transactions_df: pd.DataFrame,
    prediction_horizon_days: int = 90,
    freq: str = "D",
    as_of_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Survival analysis for customer churn/retention.

    Uses Kaplan-Meier estimator and Cox Proportional Hazards model
    to estimate customer survival probability and identify risk factors.

    Args:
        transactions_df: Transaction data
        prediction_horizon_days: Horizon for survival prediction
        freq: 'D' for daily, 'W' for weekly
        as_of_date: If provided, compute features ONLY on data up to this date
            (temporal holdout to prevent leakage). Use for training on historical data.

    Returns:
        (survival_predictions_df, diagnostics_df)
    """
    if not LIFELINES_AVAILABLE:
        raise ImportError("lifelines required: pip install lifelines")

    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # Apply temporal holdout if as_of_date provided
    if as_of_date is not None:
        df = df[df["date"] <= pd.Timestamp(as_of_date)]

    # Prepare duration data
    snapshot_date = df["date"].max()
    cutoff_date = snapshot_date - pd.Timedelta(90, unit="D")  # 90-day observation window

    # Time to first purchase after observation window
    future = df[df["date"] >= cutoff_date]
    time_to_next = future.groupby("customer_id")["date"].min().reset_index()
    time_to_next.columns = ["customer_id", "next_purchase_date"]

    # All customers in observation period
    obs_customers = df[df["date"] < cutoff_date]["customer_id"].unique()

    # Create duration data
    customer_data = df[df["customer_id"].isin(obs_customers)].copy()
    first_purchase = customer_data.groupby("customer_id")["date"].min().reset_index()
    first_purchase.columns = ["customer_id", "first_purchase"]

    last_purchase = customer_data.groupby("customer_id")["date"].max().reset_index()
    last_purchase.columns = ["customer_id", "last_purchase"]

    # Merge
    df_surv = first_purchase.merge(last_purchase, on="customer_id")
    df_surv = df_surv.merge(time_to_next, on="customer_id", how="left")

    # Duration: time from first to last (or censored)
    df_surv["duration"] = (df_surv["last_purchase"] - df_surv["first_purchase"]).dt.days
    df_surv["event"] = df_surv["next_purchase_date"].notna().astype(int)

    # Fit Kaplan-Meier
    kmf = KaplanMeierFitter()
    kmf.fit(df_surv["duration"], event_observed=df_surv["event"])

    # Fit Cox PH model if enough features
    # Create features for Cox model
    features = (
        customer_data.groupby("customer_id")
        .agg(
            frequency=("transaction_id", "nunique"),
            monetary=("price", "sum"),
            avg_basket=("quantity", "mean"),
            n_products=("stockcode", "nunique"),
        )
        .reset_index()
    )

    # Merge first_purchase and last_purchase from df_surv
    features = features.merge(
        df_surv[["customer_id", "first_purchase", "last_purchase"]], on="customer_id", how="left"
    )

    features["lifetime_days"] = (features["last_purchase"] - features["first_purchase"]).dt.days
    features["recency"] = (snapshot_date - features["last_purchase"]).dt.days

    # Merge with survival data
    surv_features = features.merge(
        df_surv[["customer_id", "duration", "event"]], on="customer_id", how="left"
    )
    surv_features["duration"] = surv_features["duration"].fillna(0)
    surv_features["event"] = surv_features["event"].fillna(0)

    # Fit Cox model
    cox_features = ["frequency", "monetary", "avg_basket", "n_products", "lifetime_days"]
    available_features = [c for c in cox_features if c in surv_features.columns]

    cox = CoxPHFitter()
    cox.fit(
        surv_features[available_features + ["duration", "event"]],
        duration_col="duration",
        event_col="event",
    )

    # Predict survival probability at horizon
    horizon = 90  # days
    surv_func = cox.predict_survival_function(surv_features[available_features])
    survival_prob = surv_func.loc[horizon] if horizon in surv_func.index else surv_func.iloc[-1]

    result = pd.DataFrame(
        {
            "customer_id": surv_features["customer_id"],
            "survival_prob": survival_prob.values,
            "churn_risk": 1 - survival_prob.values,
        }
    )

    diagnostics = pd.DataFrame(
        [
            {"metric": "concordance_index", "value": cox.concordance_index_},
            {"metric": "n_events", "value": float(surv_features["event"].sum())},
            {"metric": "n_censored", "value": float((surv_features["event"] == 0).sum())},
            {"metric": "model_params", "value": str(cox.params_.to_dict())},
        ],
        columns=["metric", "value"],
    )

    return check(result, SURVIVAL_PREDICTIONS), check(diagnostics, SURVIVAL_DIAGNOSTICS)


def kaplan_meier_estimates(
    transactions_df: pd.DataFrame,
    observation_window_days: int = 365,
    as_of_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Compute Kaplan-Meier survival estimates for all customers.

    Args:
        transactions_df: Transaction data
        observation_window_days: Days in observation window
        as_of_date: If provided, compute features ONLY on data up to this date
            (temporal holdout to prevent leakage).

    Returns:
        DataFrame with timeline and survival_prob columns.
    """
    if not LIFELINES_AVAILABLE:
        raise ImportError("lifelines required: pip install lifelines")

    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]

    # Apply temporal holdout if as_of_date provided
    if as_of_date is not None:
        df = df[df["date"] <= pd.Timestamp(as_of_date)]

    snapshot = df["date"].max()
    cutoff = snapshot - pd.Timedelta(observation_window_days, unit="D")

    df_obs = df[df["date"] < cutoff]

    # Time to next purchase after observation
    future = df[df["date"] >= cutoff]
    next_purchase = future.groupby("customer_id")["date"].min().reset_index()
    next_purchase.columns = ["customer_id", "next_date"]

    first_purchase = df_obs.groupby("customer_id")["date"].min().reset_index()
    first_purchase.columns = ["customer_id", "first_date"]
    last_purchase = df_obs.groupby("customer_id")["date"].max().reset_index()
    last_purchase.columns = ["customer_id", "last_date"]

    df_surv = first_purchase.merge(last_purchase, on="customer_id", how="left")
    df_surv = df_surv.merge(next_purchase, on="customer_id", how="left")

    snapshot = df["date"].max()
    df_surv["duration"] = (df_surv["last_date"] - df_surv["first_date"]).dt.days
    df_surv["event"] = df_surv["next_date"].notna().astype(int)

    kmf = KaplanMeierFitter()
    kmf.fit(df_surv["duration"], event_observed=df_surv["event"])

    return pd.DataFrame(
        {
            "timeline": kmf.survival_function_.index,
            "survival_prob": kmf.survival_function_.values.flatten(),
        }
    )
