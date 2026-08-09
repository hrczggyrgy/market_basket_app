"""Promotional analytics: detection, baseline/incrementality, ROI, timing, halo, uplift.

Uplift estimation uses hand-rolled T/S learners over sklearn base estimators
(HistGradientBoostingRegressor / RandomForestRegressor) plus propensity-score
stratification and Qini/AUUC evaluation. All causal estimates are indicative
correlational results, not decision-grade incrementality.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, mannwhitneyu
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

from src.analytics.schemas import (
    CATEGORY_CANNIBALIZATION,
    CATEGORY_PROMO_TIMELINE,
    PROMO_BASELINE,
    PROMO_CANNIBALIZATION,
    PROMO_HALO,
    PROMO_LIFT,
    PROMO_PERIODS,
    PROMO_ROI,
    PROMO_TIMING_DOW,
    PROMO_TIMING_MONTH,
    PROMO_WATERFALL,
    QINI_CURVE,
    UPLIFT_METRICS,
    UPLIFT_SCORES,
    check,
)

_DAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
_MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


def mark_promo_transactions(
    df: pd.DataFrame,
    promo_periods: pd.DataFrame,
    start_col: str = "start_date",
    end_col: str = "end_date",
) -> pd.DataFrame:
    """Vectorized interval join marking transactions inside any promo period."""
    df = df.copy()
    df["is_promo"] = False
    if len(promo_periods) == 0:
        return df
    promo = promo_periods.copy()
    promo[start_col] = pd.to_datetime(promo[start_col])
    promo[end_col] = pd.to_datetime(promo[end_col])
    merged = df.merge(promo[["stockcode", start_col, end_col]], on="stockcode", how="left")
    conds = (merged["date"] >= merged[start_col]) & (merged["date"] <= merged[end_col])
    in_promo = conds.groupby(level=0).any()
    df.loc[in_promo, "is_promo"] = True
    return df


def detect_promotions(
    df: pd.DataFrame,
    price_change_threshold: float = 0.15,
    min_duration_days: int = 3,
    max_duration_days: int = 60,
    gap_threshold_days: int = 1,
) -> pd.DataFrame:
    """Detect promo periods from price drops vs per-SKU 90th-percentile baseline."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    baseline_prices = df.groupby("stockcode")["price"].quantile(0.9)
    df["baseline_price"] = df["stockcode"].map(baseline_prices)
    df["price_drop_pct"] = (df["baseline_price"] - df["price"]) / df["baseline_price"]
    df["is_promo"] = df["price_drop_pct"] >= price_change_threshold
    if not df["is_promo"].any():
        return check(pd.DataFrame(columns=list(PROMO_PERIODS.columns)), PROMO_PERIODS, allow_empty=True)

    non_promo_days = df.loc[~df["is_promo"], "date"].nunique()
    promotions = []
    for stockcode, prod_df in df.groupby("stockcode", sort=False):
        flagged = prod_df[prod_df["is_promo"]].sort_values("date")
        if len(flagged) == 0:
            continue
        flagged = flagged.copy()
        date_diff = flagged["date"].diff().dt.days
        flagged["_group"] = ((date_diff.isna()) | (date_diff > gap_threshold_days)).cumsum()
        for _, group in flagged.groupby("_group"):
            start_date = group["date"].min()
            end_date = group["date"].max()
            duration = (end_date - start_date).days + 1
            if not (min_duration_days <= duration <= max_duration_days):
                continue
            baseline_sales = df[(df["stockcode"] == stockcode) & (~df["is_promo"])]
            promo_qty = group["quantity"].sum()
            baseline_qty = baseline_sales["quantity"].sum()
            promo_revenue = group["revenue"].sum()
            baseline_revenue = baseline_sales["revenue"].sum()
            qty_lift = (promo_qty / duration) / (baseline_qty / non_promo_days) - 1 if baseline_qty > 0 and non_promo_days > 0 else 0.0
            revenue_lift = (promo_revenue / duration) / (baseline_revenue / non_promo_days) - 1 if baseline_revenue > 0 and non_promo_days > 0 else 0.0
            promotions.append(
                {
                    "stockcode": stockcode,
                    "product_name": group["product"].iloc[0] if "product" in group.columns else stockcode,
                    "start_date": start_date,
                    "end_date": end_date,
                    "duration_days": duration,
                    "avg_discount_pct": group["price_drop_pct"].mean() * 100,
                    "promo_revenue": promo_revenue,
                    "baseline_revenue": baseline_revenue,
                    "promo_qty": promo_qty,
                    "baseline_qty": baseline_qty,
                    "promo_orders": group["transaction_id"].nunique(),
                    "baseline_orders": baseline_sales["transaction_id"].nunique(),
                    "promo_customers": group["customer_id"].nunique(),
                    "baseline_customers": baseline_sales["customer_id"].nunique(),
                    "qty_lift": qty_lift,
                    "revenue_lift": revenue_lift,
                    "avg_promo_price": group["price"].mean(),
                    "avg_baseline_price": baseline_sales["price"].mean() if len(baseline_sales) else 0.0,
                }
            )
    table = pd.DataFrame(promotions)
    return check(table, PROMO_PERIODS, allow_empty=True)


def compute_category_promo_timeline(
    df: pd.DataFrame,
    promo_periods: pd.DataFrame,
    freq: str = "W",
) -> pd.DataFrame:
    """Promo vs non-promo revenue per (category, period).

    Aggregates marked promo transactions (mark_promo_transactions) by category
    and period. Each (category, period) row carries:
    - promo_revenue: revenue from transactions flagged in-promo.
    - non_promo_revenue: revenue from the same category/period NOT in promo.
    - n_promos: distinct promoted SKUs active that period.
    - avg_discount_pct: mean per-transaction discount depth among promo rows.

    Requires a ``category`` column and a PROMO_PERIODS table.
    """
    required = {"category", "date", "transaction_id", "price", "quantity", "stockcode"}
    if not required.issubset(df.columns) or df.empty:
        return check(
            pd.DataFrame(columns=list(CATEGORY_PROMO_TIMELINE.columns)),
            CATEGORY_PROMO_TIMELINE,
            allow_empty=True,
        )
    if "is_promo" not in df.columns and len(promo_periods) == 0:
        return check(
            pd.DataFrame(columns=list(CATEGORY_PROMO_TIMELINE.columns)),
            CATEGORY_PROMO_TIMELINE,
            allow_empty=True,
        )

    t = df.copy()
    t["date"] = pd.to_datetime(t["date"])
    t["_revenue"] = t["price"] * t["quantity"]
    if "is_promo" not in t.columns:
        t = mark_promo_transactions(t, promo_periods)
    t["_period"] = t["date"].dt.to_period(freq).astype(str)

    if not t["is_promo"].any():
        return check(
            pd.DataFrame(columns=list(CATEGORY_PROMO_TIMELINE.columns)),
            CATEGORY_PROMO_TIMELINE,
            allow_empty=True,
        )

    # per-transaction discount depth vs 90th-percentile stock baseline
    baseline_price = t.groupby("stockcode")["price"].transform(lambda s: s.quantile(0.9))
    t["_discount_pct"] = (
        (baseline_price - t["price"]) / baseline_price.replace(0, np.nan) * 100
    ).fillna(0.0)
    # A price above the baseline is not a discount: clamp to 0%
    t["_discount_pct"] = t["_discount_pct"].clip(lower=0.0)

    promo = t[t["is_promo"]]
    base = t[~t["is_promo"]]

    promo_agg = (
        promo.groupby(["category", "_period"])
        .agg(
            promo_revenue=("_revenue", "sum"),
            n_promos=("stockcode", "nunique"),
            avg_discount_pct=("_discount_pct", "mean"),
        )
        .reset_index()
    )
    base_agg = (
        base.groupby(["category", "_period"])
        .agg(non_promo_revenue=("_revenue", "sum"))
        .reset_index()
    )

    table = promo_agg.merge(base_agg, on=["category", "_period"], how="outer").fillna(0.0)
    table["period"] = table["_period"]
    table = table[
        ["category", "period", "promo_revenue", "non_promo_revenue", "n_promos", "avg_discount_pct"]
    ].sort_values(["category", "period"])
    table["n_promos"] = table["n_promos"].astype(int)
    return check(table.copy(), CATEGORY_PROMO_TIMELINE, allow_empty=True)


def _expand_promo_weeks(promo_periods: pd.DataFrame) -> pd.DataFrame:
    """Expand promo periods to (stockcode, week) rows."""
    rows = []
    for _, promo in promo_periods.iterrows():
        start = pd.Period(promo["start_date"], "W")
        end = pd.Period(promo["end_date"], "W")
        week = start
        while week <= end:
            rows.append({"stockcode": promo["stockcode"], "week": week, "is_promo": True})
            week = week + 1
    if not rows:
        return pd.DataFrame(columns=["stockcode", "week", "is_promo"])
    return pd.DataFrame(rows).drop_duplicates(subset=["stockcode", "week"])


def compute_promo_baseline(
    df: pd.DataFrame,
    promo_periods: pd.DataFrame,
    seasonal_period: int = 52,
) -> pd.DataFrame:
    """Product-week baseline via STL decomposition (4-week rolling fallback if short).
    
    Promo weeks are masked out before STL fitting to avoid leakage of promotional
    spikes into the baseline trend/seasonal components.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    df["week"] = df["date"].dt.to_period("W")
    weekly = (
        df.groupby(["stockcode", "week"])
        .agg(actual_units=("quantity", "sum"), actual_revenue=("revenue", "sum"), avg_price=("price", "mean"))
        .reset_index()
    )
    promo_weekly = _expand_promo_weeks(promo_periods)
    weekly = weekly.merge(promo_weekly, on=["stockcode", "week"], how="left")
    weekly["is_promo"] = weekly["is_promo"].eq(True)

    results = []
    for stockcode, sku in weekly.groupby("stockcode", sort=False):
        sku = sku.sort_values("week").reset_index(drop=True)
        # Mask promo weeks before fitting baseline
        non_promo_mask = ~sku["is_promo"]
        units_non_promo = sku.loc[non_promo_mask, "actual_units"].to_numpy(dtype=float)
        revenue_non_promo = sku.loc[non_promo_mask, "actual_revenue"].to_numpy(dtype=float)
        
        if len(sku) >= seasonal_period * 2 and non_promo_mask.sum() >= seasonal_period:
            try:
                # Fit STL on non-promo data only
                units_interp = np.interp(
                    np.arange(len(sku)),
                    np.where(non_promo_mask)[0],
                    units_non_promo
                )
                revenue_interp = np.interp(
                    np.arange(len(sku)),
                    np.where(non_promo_mask)[0],
                    revenue_non_promo
                )
                baseline_units = np.maximum(STL(units_interp, seasonal=seasonal_period, robust=True).fit().trend, 0)
                baseline_units += np.maximum(STL(units_interp, seasonal=seasonal_period, robust=True).fit().seasonal, 0)
                baseline_revenue = np.maximum(STL(revenue_interp, seasonal=seasonal_period, robust=True).fit().trend, 0)
                baseline_revenue += np.maximum(STL(revenue_interp, seasonal=seasonal_period, robust=True).fit().seasonal, 0)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                baseline_units = sku["actual_units"].rolling(4, min_periods=1).mean().shift(1).fillna(sku["actual_units"])
                baseline_revenue = sku["actual_revenue"].rolling(4, min_periods=1).mean().shift(1).fillna(sku["actual_revenue"])
        else:
            baseline_units = sku["actual_units"].rolling(4, min_periods=1).mean().shift(1).fillna(sku["actual_units"])
            baseline_revenue = sku["actual_revenue"].rolling(4, min_periods=1).mean().shift(1).fillna(sku["actual_revenue"])
        sku["baseline_units"] = baseline_units
        sku["baseline_revenue"] = baseline_revenue
        sku["incremental_units"] = sku["actual_units"] - sku["baseline_units"]
        sku["incremental_revenue"] = sku["actual_revenue"] - sku["baseline_revenue"]
        sku["incrementality_pct"] = np.where(sku["actual_revenue"] > 0, sku["incremental_revenue"] / sku["actual_revenue"] * 100, 0.0)
        results.append(sku)
    table = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=list(PROMO_BASELINE.columns))
    return check(table, PROMO_BASELINE, allow_empty=True)


def pre_post_promo_comparison(
    df: pd.DataFrame,
    promo_periods: pd.DataFrame,
    baseline_days: int = 30,
) -> pd.DataFrame:
    """Per-promo pre/post comparison vs the same-duration window before the promo.

    THIS IS NOT A DIFFERENCE-IN-DIFFERENCES (DiD) ESTIMATOR. DiD requires a control group
    that is unaffected by the treatment. Here we only compare the same SKU's revenue
    before vs during the promo, which confounds promo effects with seasonality and trends.
    
    USE FOR DESCRIPTIVE PRE/POST COMPARISON ONLY. NOT FOR CAUSAL INFERENCE.
    """
    import warnings
    warnings.warn(
        "pre_post_promo_comparison is a simple pre/post comparison, NOT a DiD estimator. "
        "Results are confounded by seasonality and trends. NOT for causal inference.",
        UserWarning,
        stacklevel=2
    )
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    rows = []
    for promo_id, promo in promo_periods.iterrows():
        stockcode = promo["stockcode"]
        start = promo["start_date"]
        end = promo["end_date"]
        duration = (end - start).days + 1
        baseline_start = start - pd.Timedelta(days=duration)
        baseline_end = start - pd.Timedelta(days=1)
        treated = df[(df["stockcode"] == stockcode) & (df["date"] >= start) & (df["date"] <= end)]
        control = df[(df["stockcode"] == stockcode) & (df["date"] >= baseline_start) & (df["date"] <= baseline_end)]
        t_rev, c_rev = treated["revenue"].sum(), control["revenue"].sum()
        t_qty, c_qty = treated["quantity"].sum(), control["quantity"].sum()
        t_ord, c_ord = treated["transaction_id"].nunique(), control["transaction_id"].nunique()
        lift_revenue = (t_rev / c_rev - 1) * 100 if c_rev > 0 else 0.0
        lift_qty = (t_qty / c_qty - 1) * 100 if c_qty > 0 else 0.0
        lift_orders = (t_ord / c_ord - 1) * 100 if c_ord > 0 else 0.0
        p_value = _basket_revenue_pvalue(treated, control)
        rows.append(
            {
                "promo_id": promo_id,
                "stockcode": stockcode,
                "start_date": start,
                "end_date": end,
                "treated_revenue": t_rev,
                "control_revenue": c_rev,
                "treated_qty": t_qty,
                "control_qty": c_qty,
                "treated_orders": t_ord,
                "control_orders": c_ord,
                "lift_revenue_pct": lift_revenue,
                "lift_qty_pct": lift_qty,
                "lift_orders_pct": lift_orders,
                "p_value": p_value,
                "significant": bool(p_value is not None and p_value < 0.05),
            }
        )
    table = pd.DataFrame(rows, columns=list(PROMO_LIFT.columns))
    return check(table, PROMO_LIFT, allow_empty=True)


def _basket_revenue_pvalue(treated: pd.DataFrame, control: pd.DataFrame) -> float | None:
    t_baskets = treated.groupby("transaction_id")["revenue"].sum()
    c_baskets = control.groupby("transaction_id")["revenue"].sum()
    if len(t_baskets) < 3 or len(c_baskets) < 3:
        return None
    try:
        return float(mannwhitneyu(t_baskets, c_baskets, alternative="two-sided").pvalue)
    except ValueError:
        return None


def compute_incrementality_waterfall(
    baseline_df: pd.DataFrame,
    halo_revenue: pd.DataFrame | None = None,
    cannibalization_revenue: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Waterfall: baseline + incremental + halo - cannibalization = net incremental."""
    agg = (
        baseline_df.groupby("stockcode")
        .agg(
            baseline_revenue=("baseline_revenue", "sum"),
            actual_revenue=("actual_revenue", "sum"),
            incremental_revenue=("incremental_revenue", "sum"),
        )
        .reset_index()
    )
    if halo_revenue is not None and not halo_revenue.empty:
        halo = halo_revenue.groupby("promo_product")["halo_revenue"].sum().rename("halo_revenue")
        agg = agg.merge(halo, left_on="stockcode", right_index=True, how="left")
    else:
        agg["halo_revenue"] = 0
    if cannibalization_revenue is not None and not cannibalization_revenue.empty:
        cann = cannibalization_revenue.groupby("stockcode")["cannibalization_revenue"].sum()
        agg = agg.merge(cann, on="stockcode", how="left")
    else:
        agg["cannibalization_revenue"] = 0
    agg["halo_revenue"] = agg["halo_revenue"].fillna(0)
    agg["cannibalization_revenue"] = agg["cannibalization_revenue"].fillna(0)
    for col in ("acceleration_revenue", "switching_revenue", "stockpiling_revenue"):
        if col not in agg.columns:
            agg[col] = 0.0
    agg["net_incremental_revenue"] = agg["incremental_revenue"] + agg["halo_revenue"] - agg["cannibalization_revenue"]
    agg["roi"] = np.where(agg["baseline_revenue"] > 0, agg["net_incremental_revenue"] / agg["baseline_revenue"], 0.0)
    return check(agg, PROMO_WATERFALL, allow_empty=True)


def _diff_means(treated: np.ndarray, control: np.ndarray) -> float:
    return float(np.mean(treated) - np.mean(control))


def promo_roi_analysis(
    df: pd.DataFrame,
    promo_periods: pd.DataFrame,
    margin_pct: float = 0.3,
    promo_cost_pct: float = 0.15,
    baseline_days: int = 30,
    ci_level: float = 0.9,
    n_resamples: int = 1000,
) -> pd.DataFrame:
    """Promo ROI with a percentile bootstrap CI on total incremental revenue."""
    lift = pre_post_promo_comparison(df, promo_periods, baseline_days=baseline_days)
    if lift.empty:
        return check(pd.DataFrame(columns=list(PROMO_ROI.columns)), PROMO_ROI, allow_empty=True)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    rows = []
    for _, row in lift.iterrows():
        start, end = row["start_date"], row["end_date"]
        duration = (end - start).days + 1
        treated = df[(df["stockcode"] == row["stockcode"]) & (df["date"] >= start) & (df["date"] <= end)]
        control = df[
            (df["stockcode"] == row["stockcode"])
            & (df["date"] >= start - pd.Timedelta(days=duration))
            & (df["date"] < start)
        ]
        t_baskets = treated.groupby("transaction_id")["revenue"].sum().to_numpy()
        c_baskets = control.groupby("transaction_id")["revenue"].sum().to_numpy()
        inc_rev = row["treated_revenue"] - row["control_revenue"]
        ci_low = ci_high = float(inc_rev)
        if len(t_baskets) >= 3 and len(c_baskets) >= 3:
            try:
                # Bootstrap on total incremental revenue (not basket mean)
                def _total_diff(t, c):
                    return float(np.sum(t) - np.sum(c))
                res = bootstrap(
                    (t_baskets, c_baskets),
                    _total_diff,
                    n_resamples=n_resamples,
                    confidence_level=ci_level,
                    method="percentile",
                )
                ci_low, ci_high = res.confidence_interval.low, res.confidence_interval.high
            except ValueError:
                pass
        promo_cost = row["treated_revenue"] * promo_cost_pct
        incremental_profit = inc_rev * margin_pct - promo_cost
        roi_pct = (incremental_profit / promo_cost * 100) if promo_cost > 0 else 0.0
        rows.append(
            {
                "stockcode": row["stockcode"],
                "incremental_revenue": inc_rev,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "incremental_profit": incremental_profit,
                "promo_cost": promo_cost,
                "roi_pct": roi_pct,
            }
        )
    table = pd.DataFrame(rows).sort_values("roi_pct", ascending=False).reset_index(drop=True)
    return check(table, PROMO_ROI)


def promotion_timing_analysis(df: pd.DataFrame, promo_periods: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Promo vs baseline revenue lift by day of week and by month."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df = mark_promo_transactions(df, promo_periods)
    promo = df[df["is_promo"]]
    base = df[~df["is_promo"]]
    dow = _timing_table(promo, base, "dow")
    dow["day_name"] = dow["dow"].map(_DAY_NAMES)
    month = _timing_table(promo, base, "month")
    month["month_name"] = month["month"].map(_MONTH_NAMES)
    return {
        "by_day_of_week": check(dow, PROMO_TIMING_DOW, allow_empty=True),
        "by_month": check(month, PROMO_TIMING_MONTH, allow_empty=True),
    }


def _timing_table(promo: pd.DataFrame, base: pd.DataFrame, key: str) -> pd.DataFrame:
    p = promo.groupby(key).agg(promo_revenue=("revenue", "sum"), promo_orders=("transaction_id", "nunique"))
    b = base.groupby(key).agg(base_revenue=("revenue", "sum"), base_orders=("transaction_id", "nunique"))
    table = p.join(b, how="outer").fillna(0.0).reset_index()
    table["revenue_lift"] = (table["promo_revenue"] / table["base_revenue"].replace(0, np.nan) - 1) * 100
    return table


def halo_effect_analysis(df: pd.DataFrame, promo_periods: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """Basket-level halo: other products lifted in promo baskets vs pre-promo baskets."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    rows = []
    for _, promo in promo_periods.iterrows():
        stockcode = promo["stockcode"]
        start, end = promo["start_date"], promo["end_date"]
        promo_txns = df[(df["stockcode"] == stockcode) & (df["date"] >= start) & (df["date"] <= end)]["transaction_id"].unique()
        if len(promo_txns) == 0:
            continue
        halo = df[(df["transaction_id"].isin(promo_txns)) & (df["stockcode"] != stockcode)]
        baseline_start = start - pd.Timedelta(days=window_days * 4)
        baseline_end = start - pd.Timedelta(days=1)
        pre_txns = df[(df["stockcode"] == stockcode) & (df["date"] >= baseline_start) & (df["date"] <= baseline_end)]["transaction_id"].unique()
        baseline = df[(df["transaction_id"].isin(pre_txns)) & (df["stockcode"] != stockcode)] if len(pre_txns) else df.iloc[0:0]
        halo_agg = halo.groupby("stockcode").agg(halo_revenue=("revenue", "sum"), halo_orders=("transaction_id", "nunique"))
        base_agg = baseline.groupby("stockcode").agg(base_revenue=("revenue", "sum"), base_orders=("transaction_id", "nunique"))
        merged = halo_agg.join(base_agg, how="outer").fillna(0.0).reset_index().rename(columns={"stockcode": "halo_product"})
        merged["promo_product"] = stockcode
        merged["revenue_lift"] = (merged["halo_revenue"] / merged["base_revenue"].replace(0, np.nan) - 1) * 100
        rows.append(merged)
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=list(PROMO_HALO.columns))
    return check(table, PROMO_HALO, allow_empty=True)


def compute_cannibalization_analysis(
    df: pd.DataFrame,
    promo_periods: pd.DataFrame,
    window_days: int = 30,
    peer_revenue_floor: float = 1.0,
) -> pd.DataFrame:
    """Cross-effect / cannibalization: revenue lost by peer SKUs during a promo.

    For every promoted SKU we compare its category peers' revenue during the
    promo window against the same-length window immediately before it. When a
    peer's revenue drops during the promo, the shortfall is treated as
    cannibalized (substituted) revenue. The cannibalization index expresses
    that shortfall relative to the peer's pre-promo revenue:

        cannibalization_index = cannibalized_revenue / base_revenue

    Peers are limited to same-category SKUs with meaningful pre-promo revenue.
    Returns a per-(promo-product, peer) table, contract-validated.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    has_category = "category" in df.columns

    if has_category:
        categories = df.groupby("stockcode")["category"].first()
    else:
        categories = None

    rows: list[dict[str, float | int | str]] = []
    for _, promo in promo_periods.iterrows():
        sku = promo["stockcode"]
        start, end = pd.Timestamp(promo["start_date"]), pd.Timestamp(promo["end_date"])
        duration = (end - start).days + 1
        pre_start = start - pd.Timedelta(days=duration)
        pre_end = start - pd.Timedelta(days=1)

        promo_sku = df[df["stockcode"] == sku]
        if promo_sku.empty:
            continue
        sku_category = categories[sku] if has_category and sku in categories.index else "UNKNOWN"
        if has_category:
            peers = df[(df["stockcode"] != sku) & (df["category"] == sku_category)]["stockcode"].unique()
        else:
            peers = df[df["stockcode"] != sku]["stockcode"].unique()

        in_promo = df[(df["stockcode"].isin(peers)) & (df["date"] >= start) & (df["date"] <= end)]
        in_pre = df[(df["stockcode"].isin(peers)) & (df["date"] >= pre_start) & (df["date"] <= pre_end)]

        promo_agg = in_promo.groupby("stockcode").agg(
            promo_revenue=("revenue", "sum"), promo_orders=("transaction_id", "nunique")
        )
        pre_agg = in_pre.groupby("stockcode").agg(
            base_revenue=("revenue", "sum"), base_orders=("transaction_id", "nunique")
        )
        merged = promo_agg.join(pre_agg, how="outer").fillna(0.0).reset_index().rename(columns={"stockcode": "peer_product"})

        for _, peer in merged.iterrows():
            base_rev = float(peer["base_revenue"])
            promo_rev = float(peer["promo_revenue"])
            cannibalized = max(0.0, base_rev - promo_rev)
            if base_rev < peer_revenue_floor:
                continue
            index = cannibalized / base_rev if base_rev > 0 else 0.0
            rows.append(
                {
                    "promo_product": sku,
                    "peer_product": peer["peer_product"],
                    "category": str(sku_category),
                    "promo_revenue": promo_rev,
                    "base_revenue": base_rev,
                    "promo_orders": int(peer["promo_orders"]),
                    "base_orders": int(peer["base_orders"]),
                    "cannibalized_revenue": cannibalized,
                    "cannibalization_index": float(min(index, 1.0)),
                }
            )

    if not rows:
        return check(pd.DataFrame(columns=list(PROMO_CANNIBALIZATION.columns)), PROMO_CANNIBALIZATION, allow_empty=True)
    table = pd.DataFrame(rows, columns=list(PROMO_CANNIBALIZATION.columns))
    return check(table, PROMO_CANNIBALIZATION)


def compute_category_cannibalization(
    df: pd.DataFrame,
    promo_periods: pd.DataFrame,
    window_days: int = 30,
    peer_revenue_floor: float = 1.0,
) -> pd.DataFrame:
    """Cross-category cannibalization matrix from promo windows.

    For every promo period (a promoted SKU in ``promo_category``), compare each
    peer category's revenue during the promo window against the same-length
    window immediately before it. A drop in the peer's revenue is treated as
    revenue cannibalized by the promo:

        cannibalized_revenue  = max(0, base_revenue - promo_revenue)
        cannibalization_index = cannibalized_revenue / base_revenue   (clamped to [0, 1])

    The promoted SKU's own category is excluded so rows represent
    cross-category substitution. Results are aggregated per (promo_category,
    peer_category) pair across all promos.

    Args:
        df: Transaction DataFrame (needs a ``category`` column).
        promo_periods: PROMO_PERIODS-validated table.
        window_days: Maximum promo window length considered.
        peer_revenue_floor: Peer categories with less pre-promo revenue in a
            window are ignored (noise guard).

    Returns:
        DataFrame validated against CATEGORY_CANNIBALIZATION (empty when no
        category data or no cannibalization detected).
    """
    empty = pd.DataFrame(columns=list(CATEGORY_CANNIBALIZATION.columns))
    if "category" not in df.columns or df.empty:
        return check(empty, CATEGORY_CANNIBALIZATION, allow_empty=True)
    if promo_periods is None or promo_periods.empty:
        return check(empty, CATEGORY_CANNIBALIZATION, allow_empty=True)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    sku_category = df.groupby("stockcode")["category"].first()

    rows: list[dict[str, float | int | str]] = []
    for _, promo in promo_periods.iterrows():
        sku = promo["stockcode"]
        if sku not in sku_category.index:
            continue
        promo_category = sku_category[sku]
        start, end = pd.Timestamp(promo["start_date"]), pd.Timestamp(promo["end_date"])
        duration = (end - start).days + 1
        if duration > window_days:
            duration = window_days
            start = end - pd.Timedelta(days=duration - 1)
        pre_start = start - pd.Timedelta(days=duration)
        pre_end = start - pd.Timedelta(days=1)

        peer_categories = [c for c in df["category"].unique() if c != promo_category]
        in_promo = df[(df["date"] >= start) & (df["date"] <= end) & (df["category"].isin(peer_categories))]
        in_pre = df[(df["date"] >= pre_start) & (df["date"] <= pre_end) & (df["category"].isin(peer_categories))]

        promo_agg = in_promo.groupby("category")["revenue"].sum().rename("promo_revenue")
        pre_agg = in_pre.groupby("category")["revenue"].sum().rename("base_revenue")
        merged = promo_agg.to_frame().join(pre_agg, how="outer").fillna(0.0).reset_index().rename(
            columns={"category": "peer_category"}
        )

        for _, peer in merged.iterrows():
            base_rev = float(peer["base_revenue"])
            promo_rev = float(peer["promo_revenue"])
            cannibalized = max(0.0, base_rev - promo_rev)
            if base_rev < peer_revenue_floor:
                continue
            rows.append(
                {
                    "promo_category": str(promo_category),
                    "peer_category": str(peer["peer_category"]),
                    "n_promos": 1,
                    "promo_revenue": promo_rev,
                    "base_revenue": base_rev,
                    "cannibalized_revenue": cannibalized,
                    "cannibalization_index": float(min(cannibalized / base_rev, 1.0)),
                }
            )

    if not rows:
        return check(empty, CATEGORY_CANNIBALIZATION, allow_empty=True)

    table = pd.DataFrame(rows)
    grouped = (
        table.groupby(["promo_category", "peer_category"], as_index=False)
        .agg(
            n_promos=("n_promos", "sum"),
            promo_revenue=("promo_revenue", "sum"),
            base_revenue=("base_revenue", "sum"),
            cannibalized_revenue=("cannibalized_revenue", "sum"),
        )
    )
    grouped["cannibalization_index"] = (
        grouped["cannibalized_revenue"] / grouped["base_revenue"].replace(0, np.nan)
    ).clip(0.0, 1.0)
    grouped = grouped[list(CATEGORY_CANNIBALIZATION.columns)].sort_values(
        ["promo_category", "peer_category"]
    ).reset_index(drop=True)
    return check(grouped, CATEGORY_CANNIBALIZATION)


# ============================================================================
# Uplift modeling (T/S learners over sklearn base estimators)
# ============================================================================


def build_uplift_dataset(
    df: pd.DataFrame,
    promo_periods: pd.DataFrame,
    prediction_window_days: int = 7,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Customer-product-week uplift dataset: features, treatment, next-week qty."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    df["week"] = df["date"].dt.to_period("W")
    df = mark_promo_transactions(df, promo_periods)
    weekly = (
        df.groupby(["customer_id", "stockcode", "week"])
        .agg(
            total_qty=("quantity", "sum"),
            total_rev=("revenue", "sum"),
            avg_price=("price", "mean"),
            n_txns=("transaction_id", "nunique"),
            is_promo=("is_promo", "max"),
        )
        .reset_index()
    )
    weekly = weekly.sort_values(["customer_id", "stockcode", "week"])
    weekly["next_week_qty"] = weekly.groupby(["customer_id", "stockcode"])["total_qty"].shift(-1)
    weekly = weekly.dropna(subset=["next_week_qty"])
    weekly["treatment"] = weekly["is_promo"].astype(int)
    weekly["week_of_year"] = weekly["week"].dt.weekofyear
    weekly["month"] = weekly["week"].dt.month
    weekly["qty_lag1"] = weekly.groupby(["customer_id", "stockcode"])["total_qty"].shift(1).fillna(0.0)
    weekly["price_lag1"] = weekly.groupby(["customer_id", "stockcode"])["avg_price"].shift(1).fillna(0.0)
    cust = (
        df.groupby("customer_id")
        .agg(cust_total_qty=("quantity", "sum"), cust_total_rev=("revenue", "sum"), cust_n_products=("stockcode", "nunique"), cust_n_txns=("transaction_id", "nunique"))
        .reset_index()
    )
    weekly = weekly.merge(cust, on="customer_id", how="left")
    prod = (
        df.groupby("stockcode")
        .agg(prod_total_qty=("quantity", "sum"), prod_total_rev=("revenue", "sum"), prod_price_cv=("price", lambda s: s.std() / s.mean() if s.mean() > 0 else 0))
        .reset_index()
    )
    weekly = weekly.merge(prod, on="stockcode", how="left")
    feature_cols = [
        "total_qty", "total_rev", "avg_price", "n_txns", "week_of_year", "month",
        "qty_lag1", "price_lag1", "cust_total_qty", "cust_total_rev", "cust_n_products", "cust_n_txns",
        "prod_total_qty", "prod_total_rev", "prod_price_cv",
    ]
    X = weekly[feature_cols].fillna(0.0)
    return X, weekly["treatment"].astype(int), weekly["next_week_qty"].astype(float)


def estimate_propensity_score(X: pd.DataFrame, treatment: pd.Series) -> pd.Series:
    """P(T=1|X) via logistic regression on standardized features."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X, treatment)
    return pd.Series(model.predict_proba(X)[:, 1], index=X.index)


def check_propensity_overlap(
    propensity: pd.Series,
    treatment: pd.Series,
    min_overlap: float = 0.1,
) -> dict:
    """Overlap diagnostics: ranges, overlap proportion, warnings."""
    treated_ps = propensity[treatment == 1]
    control_ps = propensity[treatment == 0]
    if len(treated_ps) == 0 or len(control_ps) == 0:
        return {"overlap": False, "overlap_proportion": 0.0, "warnings": ["No treated or control units"]}
    treated_range = (treated_ps.min(), treated_ps.max())
    control_range = (control_ps.min(), control_ps.max())
    lower, upper = max(treated_range[0], control_range[0]), min(treated_range[1], control_range[1])
    if lower >= upper:
        overlap = 0.0
    else:
        treated_in = ((treated_ps >= lower) & (treated_ps <= upper)).mean()
        control_in = ((control_ps >= lower) & (control_ps <= upper)).mean()
        overlap = min(treated_in, control_in)
    warnings_list = []
    if overlap < min_overlap:
        warnings_list.append(f"Insufficient propensity overlap: {overlap:.1%} < {min_overlap:.1%}")
    return {
        "overlap": overlap >= min_overlap,
        "overlap_proportion": float(overlap),
        "treated_range": treated_range,
        "control_range": control_range,
        "overlap_range": (lower, upper),
        "warnings": warnings_list,
    }


def train_uplift_learner(
    X: pd.DataFrame,
    treatment: pd.Series,
    y: pd.Series,
    learner: str = "s",
    base_estimator: str = "rf",
    n_estimators: int = 200,
    max_depth: int = 5,
    random_state: int = 42,
) -> tuple[object, pd.Series]:
    """T- or S-learner uplift. Returns (model, uplift scores over X)."""
    if base_estimator == "hgb":
        def _make() -> HistGradientBoostingRegressor:
            return HistGradientBoostingRegressor(
                max_iter=n_estimators, max_depth=max_depth, random_state=random_state
            )
    elif base_estimator == "rf":
        def _make() -> RandomForestRegressor:
            return RandomForestRegressor(n_estimators=min(n_estimators, 200), max_depth=max_depth, random_state=random_state)
    else:
        raise ValueError(f"Unknown base_estimator: {base_estimator}")
    X = X.reset_index(drop=True)
    treatment = treatment.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if learner == "t":
        X_t, y_t = X[treatment == 1], y[treatment == 1]
        X_c, y_c = X[treatment == 0], y[treatment == 0]
        if len(X_t) < 10 or len(X_c) < 10:
            raise ValueError("Insufficient treated/control samples")
        model_t, model_c = _make(), _make()
        model_t.fit(X_t, y_t)
        model_c.fit(X_c, y_c)
        uplift = model_t.predict(X) - model_c.predict(X)
        return (model_t, model_c), pd.Series(uplift, index=X.index)
    if learner == "s":
        X_with_t = X.copy()
        X_with_t["treatment"] = treatment
        model = _make()
        model.fit(X_with_t, y)
        X_t1 = X.copy()
        X_t1["treatment"] = 1
        X_t0 = X.copy()
        X_t0["treatment"] = 0
        uplift = model.predict(X_t1) - model.predict(X_t0)
        return model, pd.Series(uplift, index=X.index)
    raise ValueError(f"Unknown learner: {learner}")


def evaluate_uplift_model(
    y: pd.Series,
    treatment: pd.Series,
    uplift_pred: pd.Series,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Qini coefficient + AUUC and the curves, using sklearn's auc."""
    y_sorted = y.iloc[np.argsort(-uplift_pred.to_numpy())].to_numpy()
    t_sorted = treatment.iloc[np.argsort(-uplift_pred.to_numpy())].to_numpy()
    n = len(y_sorted)
    bin_size = max(1, n // n_bins)
    qini_x_vals, qini_y_vals = [0.0], [0.0]
    cum = 0.0
    for i in range(n_bins):
        start, end = i * bin_size, min((i + 1) * bin_size, n)
        t_bin, y_bin = t_sorted[start:end], y_sorted[start:end]
        n_t, n_c = int(t_bin.sum()), len(t_bin) - int(t_bin.sum())
        if n_t > 0 and n_c > 0:
            cum += (y_bin[t_bin == 1].mean() - y_bin[t_bin == 0].mean()) * n_t
        qini_x_vals.append(end / n)
        qini_y_vals.append(cum / n)
    qini_x = np.array(qini_x_vals)
    qini_y = np.array(qini_y_vals)
    random_y = qini_x * qini_y[-1]
    qini = auc(qini_x, qini_y)
    auuc = auc(qini_x, qini_y) - auc(qini_x, random_y)
    auuc_norm = max(0.0, auuc / (qini_y[-1] - random_y[-1])) if qini_y[-1] != random_y[-1] else 0.0
    top_end = min(3 * bin_size, n)
    t_top, y_top = t_sorted[:top_end], y_sorted[:top_end]
    uplift_at_k = (y_top[t_top == 1].mean() - y_top[t_top == 0].mean()) if t_top.sum() > 0 and (len(t_top) - t_top.sum()) > 0 else 0.0
    metrics = pd.DataFrame(
        {
            "metric": ["qini_coefficient", "auuc", "auuc_normalized", "uplift_at_top_k"],
            "value": [float(qini), float(auuc), float(auuc_norm), float(uplift_at_k)],
        }
    )

    ci_lower, ci_upper = _bootstrap_qini_ci(t_sorted, y_sorted, uplift_pred.to_numpy(), qini_x, n_bins)
    curve = pd.DataFrame(
        {
            "x": qini_x,
            "qini_y": qini_y,
            "random_y": random_y,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "qini_coefficient": float(qini),
            "auuc": float(auuc),
        }
    )
    return check(metrics, UPLIFT_METRICS), check(curve, QINI_CURVE)


def _bootstrap_qini_ci(
    treatment: pd.Series,
    y: pd.Series,
    uplift_pred: pd.Series,
    qini_x: np.ndarray,
    n_bins: int = 10,
    n_resamples: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile-bootstrap CI band for the Qini curve (per-fraction permutation)."""
    order = np.argsort(-np.asarray(uplift_pred).reshape(-1))
    y_sorted = np.asarray(y)[order]
    t_sorted = np.asarray(treatment)[order]
    n = len(y_sorted)
    bin_size = max(1, n // n_bins)
    rng = np.random.default_rng(seed)
    runs = np.zeros((n_resamples, len(qini_x)))
    edges = np.arange(1, n_bins + 1) * bin_size

    def _qini_curve(perm_or_idxs: np.ndarray) -> np.ndarray:
        ts, ys = t_sorted[perm_or_idxs], y_sorted[perm_or_idxs]
        out = np.zeros(len(qini_x))
        cum = 0.0
        prev = 0
        for j, end in enumerate(edges):
            start, stop = prev, min(int(end), len(ts))
            t_bin, y_bin = ts[start:stop], ys[start:stop]
            n_t, n_c = int(t_bin.sum()), len(t_bin) - int(t_bin.sum())
            if n_t > 0 and n_c > 0:
                cum += (y_bin[t_bin == 1].mean() - y_bin[t_bin == 0].mean()) * n_t
            out[j] = cum / n
            prev = stop
        return out

    if n >= 10:
        for i in range(n_resamples):
            perm = rng.integers(0, n, size=n)
            runs[i] = _qini_curve(perm)
        lower = np.percentile(runs, 2.5, axis=0)
        upper = np.percentile(runs, 97.5, axis=0)
    else:
        lower = np.zeros(len(qini_x))
        upper = np.zeros(len(qini_x))
    return lower, upper


def score_uplift_by_customer(
    X: pd.DataFrame,
    uplift_pred: pd.Series,
    customer_ids: pd.Series,
    treatment: Optional[pd.Series] = None,
    propensity: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Aggregate uplift predictions to customer level with CIs, propensity and
    treatment flags. When treatment/propensity series are supplied they are
    used; otherwise the columns are left NaN (structural placeholders)."""
    n = len(X)
    if treatment is None:
        treatment = pd.Series([np.nan] * n, index=X.index)
    if propensity is None:
        propensity = pd.Series([np.nan] * n, index=X.index)
    table = pd.DataFrame(
        {
            "customer_id": customer_ids.reset_index(drop=True).to_numpy(),
            "uplift": uplift_pred.reset_index(drop=True).to_numpy(),
            "propensity_score": propensity.reset_index(drop=True).to_numpy(),
            "treatment_flag": treatment.reset_index(drop=True).to_numpy(),
        }
    )
    rows = []
    for customer_id, group in table.groupby("customer_id", sort=False):
        ci_lower, ci_upper = _bootstrap_mean_ci(group["uplift"].to_numpy(dtype=float))
        rows.append(
            {
                "customer_id": customer_id,
                "uplift": float(group["uplift"].mean()),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "propensity_score": float(group["propensity_score"].mean()),
                "treatment_flag": float(group["treatment_flag"].mean()),
                "acceleration_uplift": np.nan,
                "switching_uplift": np.nan,
                "stockpiling_uplift": np.nan,
            }
        )
    return check(pd.DataFrame(rows, columns=list(UPLIFT_SCORES.columns)), UPLIFT_SCORES)


def _bootstrap_mean_ci(values: np.ndarray, n_resamples: int = 50, seed: int = 42) -> tuple[float, float]:
    """Percentile-bootstrap CI on the mean of a small array (NaN if too few)."""
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    resampled = rng.choice(values, size=(n_resamples, len(values)), replace=True)
    means = resampled.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
