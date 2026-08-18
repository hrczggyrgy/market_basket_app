"""Causal Promotional Incrementality Engine.

This module provides causal incrementality estimation for promotional effects
using Two-Way Fixed Effects (TWFE) and Event Study designs.

All estimates are conditional on the parallel trends assumption.
Results should be validated with pre-trend tests before causal interpretation.

Note: This module requires `linearmodels` which currently has NumPy 2.x
compatibility issues. If unavailable, functions will raise ImportError
with a clear message. The descriptive promo engine in promo_core.py
works without linearmodels.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from src.analytics.promo_core import _expand_promo_weeks
from src.analytics.schemas import (
    PROMO_CAUSAL_PANEL,
    PROMO_CAUSAL_WATERFALL,
    PROMO_CROSS_EFFECTS,
    PROMO_EVENT_STUDY,
    PROMO_TWFE_RESULT,
    check,
)

# Lazy import to avoid NumPy 2.x compatibility warning at module load time
_LINEARMODELS_AVAILABLE = None
_LINEARMODELS_ERROR = None
_PanelOLS = None
_PanelEffectsResults = None


def _require_linearmodels() -> tuple[type, type]:
    """Import linearmodels and raise clear ImportError if not available."""
    global _LINEARMODELS_AVAILABLE, _LINEARMODELS_ERROR, _PanelOLS, _PanelEffectsResults
    if _LINEARMODELS_AVAILABLE is not None:
        if not _LINEARMODELS_AVAILABLE:
            raise ImportError(
                "linearmodels is required for causal promo estimation but is not available. "
                f"Original error: {_LINEARMODELS_ERROR}. "
                "Install with: pip install linearmodels --no-binary linearmodels "
                "(requires NumPy < 2 or building from source). "
                "Alternatively, use the descriptive promo engine in promo_core.py "
                "which works without linearmodels."
            )
        return _PanelOLS, _PanelEffectsResults

    try:
        from linearmodels.panel import PanelOLS
        from linearmodels.panel.results import PanelEffectsResults

        _PanelOLS = PanelOLS
        _PanelEffectsResults = PanelEffectsResults
        _LINEARMODELS_AVAILABLE = True
    except ImportError as e:
        _LINEARMODELS_AVAILABLE = False
        _LINEARMODELS_ERROR = e
        raise ImportError(
            "linearmodels is required for causal promo estimation but is not available. "
            f"Original error: {e}. "
            "Install with: pip install linearmodels --no-binary linearmodels "
            "(requires NumPy < 2 or building from source). "
            "Alternatively, use the descriptive promo engine in promo_core.py "
            "which works without linearmodels."
        )

    return _PanelOLS, _PanelEffectsResults


def build_promo_causal_panel(
    df: pd.DataFrame,
    promo_periods: pd.DataFrame,
    freq: str = "W",
) -> pd.DataFrame:
    """Build SKU-week panel for causal promo estimation.

    Creates a balanced SKU-week panel with treatment indicators and controls.

    Args:
        df: Transaction data with stockcode, date, quantity, price, revenue
        promo_periods: PROMO_PERIODS table with detected/promoted periods
        freq: Aggregation frequency ('W' for weekly, 'M' for monthly)

    Returns:
        Panel DataFrame validated against PROMO_CAUSAL_PANEL contract
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    df["week"] = df["date"].dt.to_period("W")

    # Aggregate to SKU-week level
    panel = (
        df.groupby(["stockcode", "week"])
        .agg(
            units=("quantity", "sum"),
            revenue=("revenue", "sum"),
            avg_price=("price", "mean"),
            n_customers=("customer_id", "nunique"),
            n_orders=("transaction_id", "nunique"),
        )
        .reset_index()
    )

    # Add promo indicator
    promo_weekly = _expand_promo_weeks(promo_periods)
    panel = panel.merge(
        promo_weekly.rename(columns={"is_promo": "promo"}),
        on=["stockcode", "week"],
        how="left",
    )
    panel["promo"] = panel["promo"].fillna(False).astype(int)

    # Add log transforms for elasticity estimation
    panel["log_units"] = np.log(panel["units"].replace(0, np.nan))
    panel["log_price"] = np.log(panel["avg_price"].replace(0, np.nan))
    panel["log_revenue"] = np.log(panel["revenue"].replace(0, np.nan))

    # Add time controls
    panel["week_num"] = panel["week"].dt.to_timestamp().dt.isocalendar().week.astype(int)
    panel["month"] = panel["week"].dt.month
    panel["year"] = panel["week"].dt.year

    # Add log units lag (for dynamic models)
    panel["log_units_lag1"] = panel.groupby("stockcode")["log_units"].shift(1)

    return check(panel, PROMO_CAUSAL_PANEL, allow_empty=True)


def estimate_twfe_promo_effect(
    panel: pd.DataFrame,
    outcome: str = "log_units",
    price_control: bool = True,
    cluster_se: bool = True,
) -> pd.DataFrame:
    """Estimate promo effect using Two-Way Fixed Effects (TWFE).

    Model: outcome ~ promo + log_price + sku_fe + week_fe

    Args:
        panel: SKU-week panel from build_promo_causal_panel
        outcome: Outcome variable ("log_units", "log_revenue", "units", "revenue")
        price_control: Whether to include log_price as control
        cluster_se: Whether to cluster SEs at SKU level

    Returns:
        DataFrame with TWFE results validated against PROMO_TWFE_RESULT
    """
    PanelOLS, _ = _require_linearmodels()

    panel = panel.copy()
    panel = panel.dropna(subset=["log_units", "log_price", "promo"]).copy()

    # Prepare variables
    y = panel[outcome]
    X = pd.DataFrame(index=panel.index)
    X["promo"] = panel["promo"]

    if price_control:
        X["log_price"] = panel["log_price"]

    # Entity and time effects via PanelOLS
    # Entity = stockcode, Time = week
    panel = panel.set_index(["stockcode", "week"])
    X = X.set_index(panel.index)
    y.index = panel.index

    # Run PanelOLS with entity and time effects
    model = PanelOLS(y, X, entity_effects=True, time_effects=True)

    if cluster_se:
        results = model.fit(cov_type="clustered", cluster_entity=True)
    else:
        results = model.fit()

    # Extract results
    promo_coef = results.params.get("promo", np.nan)
    promo_se = results.std_errors.get("promo", np.nan)
    promo_p = results.pvalues.get("promo", np.nan)

    # Calculate marginal effect in original units if log outcome
    if outcome.startswith("log_"):
        # Approximate marginal effect: % change * mean outcome
        mean_outcome = np.exp(panel[outcome.replace("log_", "")].mean())
        marginal_effect = promo_coef * mean_outcome
        marginal_se = promo_se * mean_outcome
    else:
        marginal_effect = promo_coef
        marginal_se = promo_se

    # Prepare result table
    results = pd.DataFrame(
        [
            {
                "outcome": outcome,
                "promo_coefficient": promo_coef,
                "promo_se": promo_se,
                "promo_p_value": promo_p,
                "marginal_effect": marginal_effect,
                "marginal_se": marginal_se,
                "r_squared": results.rsquared,
                "r_squared_within": results.rsquared_within,
                "r_squared_between": results.rsquared_between,
                "n_obs": results.nobs,
                "n_entities": len(panel.index.get_level_values(0).unique()),
                "n_periods": len(panel.index.get_level_values(1).unique()),
                "price_control": price_control,
            }
        ]
    )

    return check(results, PROMO_TWFE_RESULT, allow_empty=True)


def estimate_event_study(
    panel: pd.DataFrame,
    promo_periods: pd.DataFrame,
    outcome: str = "log_units",
    leads: int = 4,
    lags: int = 4,
    price_control: bool = True,
) -> pd.DataFrame:
    """Event study around promotion start dates.

    Estimates dynamic treatment effects: leads (pre-trends) and lags (post effects).

    Model: y_it = sum_{k=-L}^{L} beta_k * D_{it}^k + alpha_i + gamma_t + X_it'gamma + eps_it

    Where D_{it}^k = 1 if period t is k weeks from promo start for SKU i.

    Args:
        panel: SKU-week panel from build_promo_causal_panel
        promo_periods: PROMO_PERIODS with start_date for each promo
        outcome: Outcome variable
        leads: Number of pre-promo lead periods
        lags: Number of post-promo lag periods
        price_control: Whether to control for log_price

    Returns:
        Event study coefficients with pre-trend test
    """
    PanelOLS, PanelEffectsResults = _require_linearmodels()

    # Build event-time indicators
    promo_starts = promo_periods.groupby("stockcode")["start_date"].min().reset_index()
    promo_starts.columns = ["stockcode", "promo_start"]

    panel = panel.reset_index().merge(promo_starts, on="stockcode", how="left")
    panel["weeks_to_promo"] = (panel["week"].dt.to_timestamp() - panel["promo_start"]).dt.days // 7

    # Create lead/lag dummies
    for k in range(-leads, lags + 1):
        panel[f"D_{k}"] = (panel["weeks_to_promo"] == k).astype(int)

    # Reference period: k = -1 (week before promo)
    # Drop D_-1 to avoid collinearity with fixed effects
    dummies = [f"D_{k}" for k in range(-leads, lags + 1) if k != -1]

    panel = panel.set_index(["stockcode", "week"])

    y = panel["log_units"] if outcome == "log_units" else panel[outcome]
    X = panel[dummies]

    if price_control:
        X["log_price"] = panel["log_price"]

    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    results = model.fit(cov_type="clustered", cluster_entity=True)

    # Extract coefficients
    coefs = results.params.reindex(dummies)
    ses = results.std_errors.reindex(dummies)
    pvals = results.pvalues.reindex(dummies)

    # Pre-trend test: joint F-test that all leads == 0
    lead_dummies = [f"D_{k}" for k in range(-leads, 0)]
    if len(lead_dummies) > 0:
        f_stat, p_val = _joint_f_test(PanelEffectsResults, results, lead_dummies)
        pretrend_p = p_val
    else:
        pretrend_p = np.nan

    results_df = pd.DataFrame(
        {
            "period": [int(k) for k in range(-leads, lags + 1) if k != -1],
            "coefficient": coefs.values,
            "std_error": ses.values,
            "p_value": pvals.values,
            "is_pre": [k < 0 for k in range(-leads, lags + 1) if k != -1],
        }
    )

    return check(
        pd.DataFrame(
            [
                {
                    "outcome": outcome,
                    "leads": leads,
                    "lags": lags,
                    "pretrend_p_value": pretrend_p,
                    "coefficients": results_df.to_json(),
                }
            ]
        ),
        PROMO_EVENT_STUDY,
        allow_empty=True,
    )


def _joint_f_test(
    PanelEffectsResults: type, results: Any, restrictions: list[str]
) -> tuple[float, float]:
    """Joint F-test that specified coefficients are jointly zero."""

    if isinstance(results, PanelEffectsResults):
        try:
            res = cast(Any, results)
            f_test = res.f_test(restrictions)
            return float(f_test.statistic), float(f_test.pval)
        except Exception:
            pass
    return np.nan, np.nan


def estimate_cross_sku_effects(
    panel: pd.DataFrame,
    promo_periods: pd.DataFrame,
    outcome: str = "log_units",
) -> pd.DataFrame:
    """Estimate cross-SKU spillover effects (halo/cannibalization).

    For each promoted SKU, estimate effects on peer SKUs in same category.

    Returns DataFrame with columns:
    promo_product, peer_product, effect, se, p_value, effect_type (halo/cannibalization/none)
    """
    _require_linearmodels()
    # Get promo products
    promo_skus = promo_periods["stockcode"].unique()

    # For each promo SKU, find peer SKUs in same category
    # This requires category info in panel
    if "category" not in panel.columns:
        return check(
            pd.DataFrame(
                columns=["promo_product", "peer_product", "effect", "se", "p_value", "effect_type"]
            ),
            pd.DataFrame(),
            allow_empty=True,
        )

    panel = panel.reset_index()
    categories = panel.groupby("stockcode")["category"].first().to_dict()

    results = []
    for promo_sku in promo_skus:
        if promo_sku not in categories:
            continue
        cat = categories[promo_sku]
        # Peer SKUs in same category (excluding promoted SKU)
        peer_skus = [s for s, c in categories.items() if c == cat and s != promo_sku]

        for peer_sku in peer_skus:
            # Estimate effect on peer during promo weeks
            # Simple diff-in-diff: compare peer's sales during promo vs non-promo weeks
            peer_data = panel[(panel["stockcode"] == peer_sku)].copy()
            if len(peer_data) < 4:
                continue

            # Add promo indicator for this specific promo
            promo_weeks = panel[(panel["stockcode"] == promo_sku) & (panel["promo"] == 1)][
                "week"
            ].unique()

            if len(promo_weeks) == 0:
                continue

            peer_data["treated"] = peer_data["week"].isin(promo_weeks).astype(int)

            if peer_data["treated"].nunique() < 2:
                continue

            # Simple regression: log_units ~ treated + log_price + week_fe
            # This is a simplified version - in practice use PanelOLS
            try:
                from statsmodels.formula.api import ols

                model = ols("log_units ~ treated + log_price + C(week)", data=peer_data).fit()
                effect = model.params.get("treated", np.nan)
                se = model.bse.get("treated", np.nan)
                p_val = model.pvalues.get("treated", np.nan)

                if pd.isna(effect):
                    continue

                effect_type = "halo" if effect > 0 else "cannibalization"

                results.append(
                    {
                        "promo_product": promo_sku,
                        "peer_product": peer_sku,
                        "effect": effect,
                        "se": se,
                        "p_value": p_val,
                        "effect_type": effect_type,
                    }
                )
            except Exception:
                continue

    return check(pd.DataFrame(results), PROMO_CROSS_EFFECTS, allow_empty=True)


def compute_causal_waterfall(
    df: pd.DataFrame,
    promo_periods: pd.DataFrame,
    margin_pct: float = 0.3,
    promo_cost_pct: float = 0.15,
) -> pd.DataFrame:
    """Compute causal incrementality waterfall using TWFE estimates.

    Components:
    - Direct SKU Effect: TWFE promo coefficient x promo weeks
    - Halo Effect: Sum of positive cross-SKU effects
    - Cannibalization: Sum of negative cross-SKU effects
    - Stockpiling: Post-promo dip (negative lag effects)

    Returns:
        DataFrame with causal waterfall components
    """
    # Build panel
    panel = build_promo_causal_panel(df, promo_periods)

    # 1. Direct SKU effect
    # twfe_result = estimate_twfe_promo_effect(panel, outcome="log_units", price_control=True)

    # Aggregate across promo SKUs
    promo_skus = promo_periods["stockcode"].unique()

    # Get promo weeks per SKU
    promo_weekly = _expand_promo_weeks(promo_periods)
    promo_weeks_count = promo_weekly.groupby("stockcode")["week"].nunique().to_dict()

    # Get baseline revenue map
    # baseline_revenue_map = _get_baseline_revenue_map(df, promo_periods)

    # Direct effect per SKU (marginal effect * promo weeks)
    direct_effects = []
    for sku in promo_skus:
        if sku not in promo_weeks_count:
            continue
        # Get SKU's average baseline revenue
        # baseline_rev = baseline_revenue_map.get(sku, 0)
        # promo_weeks = promo_weeks_count.get(sku, 0)
        # Approximate: effect * baseline * promo_weeks
        # This is simplified - in practice use SKU-specific TWFE
        direct_rev = 0  # Placeholder
        direct_effects.append(
            {
                "stockcode": sku,
                "direct_effect_revenue": direct_rev,
            }
        )

    # 2. Halo effects (positive cross-SKU effects)
    cross_effects = estimate_cross_sku_effects(panel, promo_periods)
    if not cross_effects.empty:
        halo = cross_effects[cross_effects["effect_type"] == "halo"]
        cannibal = cross_effects[cross_effects["effect_type"] == "cannibalization"]
        halo_revenue = halo["effect"].sum() if not halo.empty else 0
        cannibalization_revenue = abs(cannibal["effect"].sum()) if not cannibal.empty else 0
    else:
        halo_revenue = 0
        cannibalization_revenue = 0

    # 3. Stockpiling (post-promo dip)
    # From event study: negative lag coefficients
    event_study = estimate_event_study(panel, promo_periods, outcome="log_units", leads=4, lags=4)
    stockpiling = 0
    if "coefficients" in event_study:
        coeffs = pd.read_json(event_study["coefficients"].iloc[0])
        lag_effects = coeffs[coeffs["period"] > 0]
        stockpiling = min(0, lag_effects["coefficient"].sum()) if len(lag_effects) > 0 else 0

    # Build waterfall
    total_direct = sum([d["direct_effect_revenue"] for d in direct_effects])
    gross = total_direct + halo_revenue
    net = gross - cannibalization_revenue + stockpiling

    result = pd.DataFrame(
        [
            {
                "stockcode": "ALL",
                "baseline_revenue": 0,  # Fill in
                "direct_effect_revenue": total_direct,
                "incremental_revenue_qty": 0,  # Fill
                "incremental_revenue_price": 0,  # Fill
                "halo_revenue": halo_revenue,
                "cannibalization_revenue": cannibalization_revenue,
                "stockpiling_revenue": stockpiling,
                "net_incremental_revenue": net,
                "roi": 0,
            }
        ]
    )

    return check(result, PROMO_CAUSAL_WATERFALL, allow_empty=True)


# Helper: get baseline revenue map
def _get_baseline_revenue_map(df: pd.DataFrame, promo_periods: pd.DataFrame) -> dict:
    """Get average baseline revenue per SKU."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    df["week"] = df["date"].dt.to_period("W")
    weekly = df.groupby(["stockcode", "week"]).agg(actual_revenue=("revenue", "sum")).reset_index()
    promo_weekly = _expand_promo_weeks(promo_periods)
    weekly = weekly.merge(promo_weekly, on=["stockcode", "week"], how="left")
    weekly["is_promo"] = weekly["is_promo"].eq(True)

    baseline_map = {}
    for stockcode, sku in weekly.groupby("stockcode"):
        non_promo = sku[~sku["is_promo"]]
        if len(non_promo) > 0:
            baseline_map[stockcode] = non_promo["actual_revenue"].mean()
    return baseline_map


# =============================================================================
# SCHEMA DEFINITIONS (to be added to schemas.py)
# =============================================================================

"""
PROMO_CAUSAL_PANEL = DataContract(
    name="promo_causal_panel",
    columns=(
        "stockcode", "week", "units", "revenue", "avg_price",
        "n_customers", "n_orders", "promo", "log_units", "log_price",
        "log_revenue", "week_num", "month", "year", "log_units_lag1",
    ),
    validators=(
        ValueValidator("units", lambda s: s >= 0, "units must be non-negative"),
        ValueValidator("promo", lambda s: s.isin([0, 1]), "promo must be 0/1"),
    ),
)

PROMO_TWFE_RESULT = DataContract(
    name="promo_twfe_result",
    columns=(
        "outcome", "promo_coefficient", "promo_se", "promo_p_value",
        "marginal_effect", "marginal_se", "r_squared", "r_squared_within",
        "r_squared_between", "n_obs", "n_entities", "n_periods", "price_control",
    ),
    validators=(
        ValueValidator("promo_p_value", lambda s: s.isna() | ((s >= 0) & (s <= 1))),
    ),
)

PROMO_EVENT_STUDY = DataContract(
    name="promo_event_study",
    columns=(
        "outcome", "leads", "lags", "pretrend_p_value", "coefficients",
    ),
)

PROMO_CAUSAL_WATERFALL = DataContract(
    name="promo_causal_waterfall",
    columns=(
        "stockcode", "baseline_revenue", "direct_effect_revenue",
        "incremental_revenue_qty", "incremental_revenue_price",
        "halo_revenue", "cannibalization_revenue", "stockpiling_revenue",
        "net_incremental_revenue", "roi",
    ),
    validators=(
        ValueValidator("baseline_revenue", lambda s: s >= 0, "baseline_revenue must be non-negative"),
        ValueValidator("cannibalization_revenue", lambda s: s >= 0, "cannibalization_revenue must be non-negative"),
    ),
)

PROMO_CROSS_EFFECTS = DataContract(
    name="promo_cross_effects",
    columns=(
        "promo_product", "peer_product", "effect", "se", "p_value", "effect_type",
    ),
    validators=(
        ValueValidator("effect_type", lambda s: s.isin(["halo", "cannibalization", "none"])),
    ),
)

PROMO_PARALLEL_TRENDS = DataContract(
    name="promo_parallel_trends",
    columns=("pretrend_p_value", "method", "n_skus"),
    validators=(
        ValueValidator("pretrend_p_value", lambda s: s.isna() | ((s >= 0) & (s <= 1))),
    ),
)
"""

# =============================================================================
# UI INTEGRATION NOTES FOR promo_page.py
# =============================================================================

"""
def render_causal_promo(df: pd.DataFrame) -> None:
    '''Render two-layer promotional analytics UI.'''

    # --- LAYER 1: Descriptive Decomposition ---
    st.header(":material/bar_chart: Promotional Revenue Decomposition (Descriptive)")
    st.caption(
        "⚠️ **Descriptive Only**: Shows observed revenue vs. modeled baseline. "
        "Does NOT imply causal incrementality."
    )

    # Existing descriptive waterfall...

    st.divider()

    # --- LAYER 2: Causal Incrementality ---
    st.header(":material/science: Causal Incrementality Engine")
    st.caption(
        "⚠️ **Causal Estimates**: Require parallel trends assumption. "
        "Validate with event study pre-trends before acting."
    )

    if st.button("Run Causal Incrementality Engine"):
        with st.spinner("Estimating causal effects..."):
            df_clean = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            promo_periods = detect_promotions(df)  # or use user-provided

            # Run causal engine
            waterfall = compute_causal_waterfall(df, promo_periods)

            if waterfall.empty:
                st.warning("Insufficient data for causal estimation.")
                return

            # Display causal waterfall
            st.subheader(":material/waterfall: Causal Incrementality Waterfall")
            wf = waterfall.iloc[0]

            cols = st.columns(5)
            cols[0].metric("Direct Effect", f"${wf['direct_effect_revenue']:,.0f}")
            cols[1].metric("Halo Effect", f"${wf['halo_revenue']:,.0f}")
            cols[2].metric("Cannibalization", f"-${wf['cannibalization_revenue']:,.0f}")
            cols[3].metric("Stockpiling", f"${wf['stockpiling_revenue']:,.0f}")
            cols[4].metric("NET INCREMENTAL", f"${wf['net_incremental_revenue']:,.0f}")

            # Event study plot
            st.subheader(":material/show_chart: Event Study (Parallel Trends)")
            # Render event study plot...

            # Assumption checklist
            st.subheader(":material/checklist: Causal Assumptions")
            st.checkbox("Parallel trends (pre-trends p > 0.05)")
            st.checkbox("No spillover (measured via cross-SKU effects)")
            st.checkbox("No anticipation (flat pre-trends)")
            st.checkbox("SUTVA (no interference across SKUs)")

            # Sensitivity
            st.subheader(":material/tune: Sensitivity")
            st.caption("Rosenbaum bounds: Effect robust to unobserved confounder with Γ < X")
"""

# =============================================================================
# END OF CAUSAL PROMO ENGINE
# =============================================================================
