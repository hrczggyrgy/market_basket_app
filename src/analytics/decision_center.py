"""Decision Center — cross-domain signal aggregation.

Reads aggregated cross-domain signals from the ResultStore.
No direct engine calls are made from this module;
all engine results should be written to the ResultStore
via the AnalysisExecutor before calling run_decision_center.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.orchestration.analysis_registry import get
from src.orchestration.result_store import (
    ResultStore,
    get_schema_version_func,
    make_key,
    param_hash,
)

_EMPTY_INSIGHTS = pd.DataFrame(
    columns=[
        "domain",
        "entity",
        "kind",
        "title",
        "evidence",
        "impact_value",
        "confidence",
        "sample_size",
        "stability",
        "action",
        "evidence_level",
        "n_transition_pairs",
        "n_unique_products",
        "confidence_gate",
    ]
)

_EMPTY_OPPORTUNITIES = pd.DataFrame(
    columns=[
        "domain",
        "entity",
        "title",
        "value",
        "confidence",
        "action",
        "source",
        "rationale",
    ]
)


@dataclass
class DecisionCenterAnalysis:
    """Aggregated cross-domain signal set read from ResultStore."""

    insights: pd.DataFrame = field(default_factory=lambda: _EMPTY_INSIGHTS)
    opportunities: pd.DataFrame = field(default_factory=lambda: _EMPTY_OPPORTUNITIES)
    n_signals: int = 0
    n_opportunities: int = 0
    n_risks: int = 0
    total_opportunity_value: float = 0.0
    domains_covered: list[str] = field(default_factory=list)


def run_decision_center(
    df: pd.DataFrame,
    *,
    dataset_id: str = "default",
) -> DecisionCenterAnalysis:
    """Read aggregated cross-domain signals from ResultStore.

    Reads the following tier-A domains from the ResultStore:
    - overview, pricing, product, switching, promotion, cross_sell

    These analyses are tier A (instant recomputation) and their results
    should be cached in the ResultStore by the AnalysisExecutor
    before this function is called.

    Args:
        df: transaction frame (used only for type-checking; dataset_id
            determines which cached results to read).
        dataset_id: Identifier for the dataset whose cached results
            should be read from the ResultStore.

    Returns:
        DecisionCenterAnalysis populated with results read from ResultStore.
    """
    store = ResultStore()
    hp = param_hash({}, schema_version=get_schema_version_func())

    # Tier A analysis keys that decision_center depends on
    tier_a_keys = [
        "overview",
        "pricing",
        "product",
        "switching",
        "promotion",
        "cross_sell",
    ]

    insight_parts: list[pd.DataFrame] = []
    opp_parts: list[pd.DataFrame] = []
    domains: list[str] = []

    for key in tier_a_keys:
        try:
            spec = get(key)
        except KeyError:
            continue

        # Read from ResultStore
        cache_key = make_key(dataset_id, key, spec.version, hp)
        if not store.has(dataset_id, key, spec.version, hp):
            # Result not cached; for tier A this should be rare
            # since they recompute instantly, but we gracefully skip
            continue

        result = store.get(dataset_id, key, spec.version, hp)

        if key == "overview":
            if not result.empty:
                insight_parts.append(result)
                domains.append("overview")
        elif key == "pricing":
            # Expected structure: {"insights": df, "opportunities": df}
            if not result.get("insights", pd.DataFrame()).empty:
                insight_parts.append(result["insights"])
                domains.append("pricing")
            if not result.get("opportunities", pd.DataFrame()).empty:
                opp_parts.append(result["opportunities"].head(8))
        elif key == "product":
            if not result.empty:
                insight_parts.append(result)
                domains.append("product")
        elif key == "switching":
            if not result.empty:
                insight_parts.append(result)
                domains.append("switching")
            # Switching opportunities may also be stored
            if not result.get("opportunities", pd.DataFrame()).empty:
                opp_parts.append(result["opportunities"].head(8))
        elif key == "promotion":
            # Expected structure: waterfall, roi, lift, cannibalization, etc.
            if not result.get("insights", pd.DataFrame()).empty:
                insight_parts.append(result["insights"])
                domains.append("promotion")
            if not result.get("opportunities", pd.DataFrame()).empty:
                opp_parts.append(result["opportunities"].head(8))
        elif key == "cross_sell":
            if not result.empty:
                insight_parts.append(result)
                domains.append("cross_sell")
                # Cross-sell opportunities
                if not result.get("opportunities", pd.DataFrame()).empty:
                    opp_parts.append(result["opportunities"].head(8))

    # Filter out empty DataFrames to avoid FutureWarning
    insight_parts = [part for part in insight_parts if not part.empty]
    insights = pd.concat(insight_parts, ignore_index=True) if insight_parts else _EMPTY_INSIGHTS
    opp_parts = [part for part in opp_parts if not part.empty]
    opportunities = pd.concat(opp_parts, ignore_index=True) if opp_parts else _EMPTY_OPPORTUNITIES

    if not insights.empty:
        insights = insights.sort_values(
            by=["impact_value", "confidence"], ascending=[False, True], na_position="last"
        ).reset_index(drop=True)
        from src.analytics.schemas import PRICING_INSIGHTS, check
        insights = check(insights, PRICING_INSIGHTS, allow_empty=True)
    if not opportunities.empty:
        opportunities = opportunities.sort_values(
            by="value", ascending=False, na_position="last"
        ).reset_index(drop=True)
        from src.analytics.schemas import OPPORTUNITY_LIST, check
        opportunities = check(opportunities, OPPORTUNITY_LIST, allow_empty=True)

    n_signals = int(len(insights))
    n_opportunities = int(len(opportunities))
    n_risks = int((insights["kind"] == "risk").sum()) if not insights.empty else 0
    total_opportunity_value = (
        float(opportunities["value"].fillna(0.0).sum())
        if not opportunities.empty
        else 0.0
    )

    return DecisionCenterAnalysis(
        insights=insights,
        opportunities=opportunities,
        n_signals=n_signals,
        n_opportunities=n_opportunities,
        n_risks=n_risks,
        total_opportunity_value=total_opportunity_value,
        domains_covered=domains,
    )
