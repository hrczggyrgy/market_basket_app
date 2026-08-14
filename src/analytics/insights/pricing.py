"""Pricing-domain insight generation.

Turns elasticity / KVI / decision-matrix outputs into structured, evidence-backed
``Insight`` objects that answer "so what?" and "what should we do?" rather than
just presenting another chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.intelligence import Insight, insights_to_dataframe
from src.analytics.schemas import PRICING_INSIGHTS, check

_USABLE_STATUSES = ("estimated", "weak")

_DECISION_META: dict[str, dict[str, str]] = {
    "invest": {
        "kind": "opportunity",
        "evidence": (
            "High-KVI, price-elastic traffic drivers: price increases would "
            "sacrifice volume disproportionately."
        ),
        "action": (
            "Maintain price competitiveness; prioritize availability; never "
            "trade down to private label."
        ),
    },
    "protect": {
        "kind": "growth",
        "evidence": "High-KVI, price-inelastic: they can carry margin without volume risk.",
        "action": "Hold price; protect availability; avoid needless discounting.",
    },
    "price_lever": {
        "kind": "opportunity",
        "evidence": "Low-KVI, price-elastic: candidate promotional price levers.",
        "action": "Consider targeted promotional investment or a controlled -5% price test.",
    },
    "review": {
        "kind": "efficiency",
        "evidence": "Low-KVI, price-inelastic: low strategic importance, review last.",
        "action": "Review assortment depth last; do not invest in price.",
    },
}


def _aggregate_confidence(confidences: pd.Series, default: str = "medium") -> str:
    """Aggregate per-SKU confidence tiers into a single insight confidence."""
    if confidences.empty:
        return default
    high = float((confidences == "high").mean())
    low = float((confidences == "low").mean())
    if high >= 0.5:
        return "high"
    if low > 0.5:
        return "low"
    return "medium"


def _revenue_by_sku(kvi_df: pd.DataFrame) -> pd.Series:
    if kvi_df.empty or "total_revenue" not in kvi_df.columns:
        return pd.Series(dtype=float)
    return kvi_df.set_index("stockcode")["total_revenue"].astype(float)


def generate_pricing_insights(
    elasticity_df: pd.DataFrame,
    elasticity_status_df: pd.DataFrame,
    kvi_df: pd.DataFrame,
    decision_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Build Pricing Insights from the full pricing pipeline.

    Args:
        elasticity_df: ELASTICITY output (estimable SKUs with diagnostics).
        elasticity_status_df: ELASTICITY_STATUS output (coverage view, all SKUs).
        kvi_df: KVI_SCORES output.
        decision_matrix: PRICING_DECISION_MATRIX output.

    Returns:
        DataFrame validated against PRICING_INSIGHTS.
    """
    insights: list[Insight] = []
    rev_by_sku = _revenue_by_sku(kvi_df)

    status = (
        elasticity_status_df
        if elasticity_status_df is not None and not elasticity_status_df.empty
        else pd.DataFrame()
    )
    if not status.empty:
        n_total = int(len(status))
        n_est = int((status["elasticity_status"] == "estimated").sum())
        n_weak = int((status["elasticity_status"] == "weak").sum())
        n_insuf_var = int((status["elasticity_status"] == "insufficient_variation").sum())
        n_insuf_obs = int((status["elasticity_status"] == "insufficient_observations").sum())
        n_insuf_pts = int((status["elasticity_status"] == "insufficient_price_points").sum())
        n_unavail = int((status["elasticity_status"] == "unavailable").sum())
        usable = n_est + n_weak
        coverage = usable / n_total if n_total else 0.0

        usable_skus = status.loc[status["elasticity_status"].isin(_USABLE_STATUSES), "stockcode"]
        covered_rev = float(usable_skus.map(rev_by_sku).fillna(0.0).sum())
        total_rev = float(rev_by_sku.sum())
        uncovered_rev = max(total_rev - covered_rev, 0.0)

        if coverage < 0.5:
            insights.append(
                Insight(
                    domain="pricing",
                    entity="all SKUs",
                    kind="risk",
                    title=f"Only {coverage:.0%} of SKUs have a usable elasticity estimate",
                    evidence=(
                        f"{usable} of {n_total} SKUs are estimable "
                        f"({n_insuf_var} lack price variation, {n_insuf_obs} have too few "
                        f"observations, {n_insuf_pts} lack distinct price points, "
                        f"{n_unavail} are unavailable). "
                        f"€{uncovered_rev:,.0f} of revenue is not price-manageable on evidence."
                    ),
                    action=(
                        "Add price variation (or use hierarchical/causal estimates) "
                        "before re-pricing SKUs without estimates."
                    ),
                    confidence="high" if n_total >= 30 else "medium",
                    impact_value=uncovered_rev if uncovered_rev > 0 else None,
                    sample_size=n_total,
                    stability=round(coverage, 3),
                    evidence_level=2,  # descriptive
                    n_transition_pairs=0,
                    n_unique_products=0,
                    confidence_gate=False,  # evidence_level < 3
                )
            )
        else:
            insights.append(
                Insight(
                    domain="pricing",
                    entity="all SKUs",
                    kind="efficiency",
                    title=f"Elasticity coverage is solid ({coverage:.0%} of SKUs)",
                    evidence=(
                        f"{usable} of {n_total} SKUs have usable estimates; "
                        f"€{covered_rev:,.0f} of €{total_rev:,.0f} revenue is covered."
                    ),
                    action="Reprice with confidence on high-confidence estimates only.",
                    confidence="high" if n_total >= 30 else "medium",
                    impact_value=None,
                    sample_size=n_total,
                    stability=round(coverage, 3),
                    evidence_level=2,  # descriptive
                    n_transition_pairs=0,
                    n_unique_products=0,
                    confidence_gate=False,  # evidence_level < 3
                )
            )

        if n_weak > 0:
            weak_skus = status.loc[status["elasticity_status"] == "weak", "stockcode"]
            weak_rev = float(weak_skus.map(rev_by_sku).fillna(0.0).sum())
            insights.append(
                Insight(
                    domain="pricing",
                    entity="low-confidence SKUs",
                    kind="watch",
                    title=f"{n_weak} SKUs have unreliable elasticity estimates",
                    evidence=(
                        f"Low-confidence estimates (wide CI or not significant) cover "
                        f"€{weak_rev:,.0f} of revenue."
                    ),
                    action="Do not reprice on these estimates; collect more price variation first.",
                    confidence="medium",
                    impact_value=weak_rev if weak_rev > 0 else None,
                    sample_size=n_weak,
                    evidence_level=2,  # descriptive
                    n_transition_pairs=0,
                    n_unique_products=0,
                    confidence_gate=False,  # evidence_level < 3
                )
            )

        # Revenue-at-risk for non-estimable high-revenue SKUs
        non_estimable = status[~status["elasticity_status"].isin(_USABLE_STATUSES)]
        if not non_estimable.empty:
            non_est_rev = float(non_estimable["stockcode"].map(rev_by_sku).fillna(0.0).sum())
            if non_est_rev > 0:
                # Find high-revenue SKUs without estimates
                high_rev_threshold = rev_by_sku.quantile(0.75) if len(rev_by_sku) > 0 else 0
                high_rev_non_est = non_estimable[
                    non_estimable["stockcode"].map(rev_by_sku).fillna(0) >= high_rev_threshold
                ]
                if not high_rev_non_est.empty:
                    high_rev_count = len(high_rev_non_est)
                    high_rev_value = float(
                        high_rev_non_est["stockcode"].map(rev_by_sku).fillna(0.0).sum()
                    )
                    insights.append(
                        Insight(
                            domain="pricing",
                            entity=f"{high_rev_count} high-revenue SKUs",
                            kind="risk",
                            title=f"€{high_rev_value:,.0f} revenue lacks price elasticity evidence",
                            evidence=(
                                f"{high_rev_count} high-revenue SKUs (top quartile) cannot be reliably "
                                f"priced due to insufficient price variation or data. "
                                f"Total affected revenue: €{high_rev_value:,.0f}."
                            ),
                            action=(
                                "Introduce price variation on these SKUs through controlled tests "
                                "or use hierarchical/causal estimation methods."
                            ),
                            confidence="high",
                            impact_value=high_rev_value,
                            sample_size=high_rev_count,
                            evidence_level=3,  # predictive
                            n_transition_pairs=0,
                            n_unique_products=0,
                            confidence_gate=True,  # evidence_level >= 3
                        )
                    )

        # Price-variation opportunity insights
        if n_insuf_var > 0 or n_insuf_pts > 0:
            low_var_skus = status[
                status["elasticity_status"].isin(
                    ["insufficient_variation", "insufficient_price_points"]
                )
            ]
            if not low_var_skus.empty:
                low_var_rev = float(low_var_skus["stockcode"].map(rev_by_sku).fillna(0.0).sum())
                insights.append(
                    Insight(
                        domain="pricing",
                        entity=f"{len(low_var_skus)} low-variation SKUs",
                        kind="opportunity",
                        title=f"Price variation needed on {len(low_var_skus)} SKUs (€{low_var_rev:,.0f} revenue)",
                        evidence=(
                            "These SKUs lack sufficient price variation for elasticity estimation. "
                            "Introducing controlled price variation would enable evidence-based pricing."
                        ),
                        action=(
                            "Run controlled price tests or introduce tiered pricing to generate "
                            "the variation needed for reliable elasticity estimation."
                        ),
                        confidence="high",
                        impact_value=low_var_rev,
                        sample_size=len(low_var_skus),
                        evidence_level=2,  # descriptive
                        n_transition_pairs=0,
                        n_unique_products=0,
                        confidence_gate=False,  # evidence_level < 3
                    )
                )

    dm = (
        decision_matrix
        if decision_matrix is not None and not decision_matrix.empty
        else pd.DataFrame()
    )
    if not dm.empty:
        for decision in ("invest", "protect", "price_lever", "review"):
            grp = dm[dm["decision"] == decision]
            if grp.empty:
                continue
            n = int(len(grp))
            rev = float(grp["total_revenue"].sum())
            meta = _DECISION_META[decision]
            entities = ", ".join(grp["stockcode"].astype(str).head(5).tolist())
            if n > 5:
                entities += f" +{n - 5} more"
            insights.append(
                Insight(
                    domain="pricing",
                    entity=entities,
                    kind=meta["kind"],
                    title=f"{n} SKUs: {decision}",
                    evidence=f"{meta['evidence']} Aggregate revenue: €{rev:,.0f}.",
                    action=meta["action"],
                    confidence=_aggregate_confidence(grp["elasticity_confidence"]),
                    impact_value=rev,
                    sample_size=n,
                    evidence_level=3,  # predictive
                    n_transition_pairs=0,
                    n_unique_products=0,
                    confidence_gate=True,  # evidence_level >= 3
                )
            )

    if elasticity_df is not None and not elasticity_df.empty:
        extreme = elasticity_df[np.abs(elasticity_df["elasticity"]) > 5]
        if not extreme.empty:
            extreme_rev = float(extreme["stockcode"].map(rev_by_sku).fillna(0.0).sum())
            insights.append(
                Insight(
                    domain="pricing",
                    entity=", ".join(extreme["stockcode"].astype(str).head(5).tolist()),
                    kind="watch",
                    title=f"{len(extreme)} SKUs have extreme elasticity estimates (|e| > 5)",
                    evidence=(
                        "Economically implausible elasticity often signals data quality "
                        "issues or model misspecification."
                    ),
                    action="Validate price data and re-estimate before acting.",
                    confidence="low",
                    impact_value=extreme_rev if extreme_rev > 0 else None,
                    sample_size=int(len(extreme)),
                    evidence_level=1,  # exploratory
                    n_transition_pairs=0,
                    n_unique_products=0,
                    confidence_gate=False,  # evidence_level < 3
                )
            )

    table = insights_to_dataframe(insights)
    return check(table, PRICING_INSIGHTS, allow_empty=True)
