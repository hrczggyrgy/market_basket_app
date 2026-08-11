"""Decision Center — cross-domain signal aggregation.

Runs the fast engines (overview, pricing, product, switching, promotion) and
optionally the heavy ones (CLV retention, assortment scenarios), then merges
every domain's insights and opportunities into two ranked tables for the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.analytics.insights import (
    generate_assortment_insights,
    generate_customer_insights,
    generate_overview_insights,
    generate_product_insights,
    generate_promotion_insights,
    generate_switching_insights,
)
from src.analytics.opportunities import (
    generate_assortment_opportunities,
    generate_cross_sell_opportunities,
    generate_promotion_opportunities,
    generate_retention_opportunities,
    generate_switching_opportunities,
)
from src.analytics.performance import (
    compute_sku_rationalization_df,
    product_lifecycle_stage,
    xyz_analysis,
)
from src.analytics.pricing import run_pricing_analysis
from src.analytics.promo_core import (
    compute_cannibalization_analysis,
    compute_incrementality_waterfall,
    compute_promo_baseline,
    detect_promotions,
    pre_post_promo_comparison,
    promo_roi_analysis,
)
from src.analytics.schemas import OPPORTUNITY_LIST, PRICING_INSIGHTS, check
from src.analytics.transference import (
    compute_demand_transference_matrix,
    compute_substitutable_demand_percentage,
    delist_impact_analysis,
)

_EMPTY = pd.DataFrame()


def _revenue_by_product(df: pd.DataFrame) -> pd.Series:
    return (df["price"] * df["quantity"]).groupby(df["stockcode"]).sum()


def _top_delist_candidates(sdp_df: pd.DataFrame, n: int = 10) -> list[str]:
    if sdp_df is None or sdp_df.empty:
        return []
    return sdp_df.sort_values("sdp", ascending=False).head(n)["stockcode"].astype(str).tolist()


@dataclass
class DecisionCenterAnalysis:
    """Aggregated cross-domain signal set."""

    insights: pd.DataFrame = field(default_factory=lambda: _EMPTY)
    opportunities: pd.DataFrame = field(default_factory=lambda: _EMPTY)
    n_signals: int = 0
    n_opportunities: int = 0
    n_risks: int = 0
    total_opportunity_value: float = 0.0
    domains_covered: list[str] = field(default_factory=list)


def run_decision_center(
    df: pd.DataFrame,
    *,
    include_clv: bool = False,
    include_assortment: bool = False,
    top_n: int = 8,
) -> DecisionCenterAnalysis:
    """Run the cross-domain engines and merge signals.

    Args:
        df: transaction frame.
        include_clv: run the (slow) BG/NBD CLV engine.
        include_assortment: run the (slow) scenario simulator.
        top_n: per-domain opportunity cap.

    Returns:
        Aggregated DecisionCenterAnalysis with contract-validated tables.
    """
    insight_parts: list[pd.DataFrame] = []
    opp_parts: list[pd.DataFrame] = []
    domains: list[str] = []

    # --- Overview (fast, from raw data) ------------------------------------
    overview_insights = generate_overview_insights(df)
    if not overview_insights.empty:
        insight_parts.append(overview_insights)
        domains.append("overview")

    # --- Pricing ------------------------------------------------------------
    try:
        pricing = run_pricing_analysis(df, min_periods=5)
        if not pricing.insights.empty:
            insight_parts.append(pricing.insights)
            domains.append("pricing")
        if not pricing.opportunities.empty:
            opp_parts.append(pricing.opportunities.head(top_n))
    except Exception:
        pass

    # --- Product / performance ---------------------------------------------
    try:
        xyz = xyz_analysis(df)
        lifecycle = product_lifecycle_stage(df)
        rationalization = compute_sku_rationalization_df(df)
        product_insights = generate_product_insights(rationalization, xyz, lifecycle)
        if not product_insights.empty:
            insight_parts.append(product_insights)
            domains.append("product")
    except Exception:
        pass

    # --- Switching / transference ------------------------------------------
    try:
        dt = compute_demand_transference_matrix(df)
        if not dt.empty:
            sdp = compute_substitutable_demand_percentage(dt, df)
            delist = delist_impact_analysis(df, dt, _top_delist_candidates(sdp))
            switching_insights = generate_switching_insights(dt, sdp, delist)
            if not switching_insights.empty:
                insight_parts.append(switching_insights)
                domains.append("switching")
            rev_by_product = _revenue_by_product(df)
            switching_opps = generate_switching_opportunities(sdp, delist, rev_by_product, top_n=top_n)
            if not switching_opps.empty:
                opp_parts.append(switching_opps)
    except Exception:
        pass

    # --- Promotion ----------------------------------------------------------
    try:
        promos = detect_promotions(df)
        if not promos.empty:
            baseline = compute_promo_baseline(df, promos)
            waterfall = compute_incrementality_waterfall(baseline)
            roi = promo_roi_analysis(df, promos, n_resamples=200)
            lift = pre_post_promo_comparison(df, promos)
            cannibalization = compute_cannibalization_analysis(df, promos)
            promo_insights = generate_promotion_insights(waterfall, roi, lift, cannibalization)
            if not promo_insights.empty:
                insight_parts.append(promo_insights)
                domains.append("promotion")
            promo_opps = generate_promotion_opportunities(waterfall, roi, top_n=top_n)
            if not promo_opps.empty:
                opp_parts.append(promo_opps)
    except Exception:
        pass

    # --- Cross-sell ---------------------------------------------------------
    try:
        from src.analytics.copurchase import get_top_affinity_pairs

        affinity = get_top_affinity_pairs(df, top_n=50, min_cooccurrence=3)
        addon = _addon_recs_from_affinity(affinity)
        cross_opps = generate_cross_sell_opportunities(addon, affinity, _revenue_by_product(df), top_n=top_n)
        if not cross_opps.empty:
            opp_parts.append(cross_opps)
    except Exception:
        pass

    # --- CLV / retention (heavy, opt-in) ------------------------------------
    if include_clv:
        try:
            from src.analytics.clv import compute_clv_customer_df, predict_clv_bg_nbd

            predictions, _ = predict_clv_bg_nbd(df)
            clv_customers = compute_clv_customer_df(df, predictions=predictions)
            customer_insights = generate_customer_insights(clv_customers, predictions)
            if not customer_insights.empty:
                insight_parts.append(customer_insights)
                domains.append("customer")
            retention_opps = generate_retention_opportunities(clv_customers, top_n=top_n)
            if not retention_opps.empty:
                opp_parts.append(retention_opps)
        except Exception:
            pass

    # --- Assortment scenarios (heavy, opt-in) -------------------------------
    if include_assortment:
        try:
            from src.analytics.assortment import compare_assortment_scenarios

            scenarios = compare_assortment_scenarios(df, [], n_scenarios=3)
            if not scenarios.empty:
                assortment_insights = generate_assortment_insights(scenarios)
                if not assortment_insights.empty:
                    insight_parts.append(assortment_insights)
                    domains.append("assortment")
                assortment_opps = generate_assortment_opportunities(scenarios, None, top_n=top_n)
                if not assortment_opps.empty:
                    opp_parts.append(assortment_opps)
        except Exception:
            pass

    # Filter out empty DataFrames to avoid FutureWarning
    insight_parts = [part for part in insight_parts if not part.empty]
    insights = pd.concat(insight_parts, ignore_index=True) if insight_parts else _EMPTY
    opp_parts = [part for part in opp_parts if not part.empty]
    opportunities = pd.concat(opp_parts, ignore_index=True) if opp_parts else _EMPTY

    if not insights.empty:
        insights = insights.sort_values(
            by=["impact_value", "confidence"], ascending=[False, True], na_position="last"
        ).reset_index(drop=True)
        insights = check(insights, PRICING_INSIGHTS, allow_empty=True)
    if not opportunities.empty:
        opportunities = opportunities.sort_values(
            by="value", ascending=False, na_position="last"
        ).reset_index(drop=True)
        opportunities = check(opportunities, OPPORTUNITY_LIST, allow_empty=True)

    return DecisionCenterAnalysis(
        insights=insights,
        opportunities=opportunities,
        n_signals=int(len(insights)),
        n_opportunities=int(len(opportunities)),
        n_risks=int((insights["kind"] == "risk").sum()) if not insights.empty else 0,
        total_opportunity_value=float(opportunities["value"].fillna(0.0).sum()) if not opportunities.empty else 0.0,
        domains_covered=domains,
    )


def _addon_recs_from_affinity(affinity_df: pd.DataFrame) -> pd.DataFrame:
    """Project AFFINITY_PAIRS onto the ADDON_RECS shape (anchor/addon views).

    affinity = P(A and B) / (P(A) * P(B)), so:
        support   = affinity * support_a * support_b   (= P(A and B))
        confidence = affinity * support_b               (= P(B | A))
    """
    if affinity_df is None or affinity_df.empty:
        return _EMPTY
    rows = []
    for _, row in affinity_df.iterrows():
        affinity = float(row["affinity"])
        support_a = float(row["support_a"])
        support_b = float(row["support_b"])
        rows.append(
            {
                "anchor": row["product_a"],
                "addon": row["product_b"],
                "support": affinity * support_a * support_b,
                "confidence": affinity * support_b,
                "lift": affinity,
                "cooccurrence": row["cooccurrence"],
            }
        )
    return pd.DataFrame(rows)
