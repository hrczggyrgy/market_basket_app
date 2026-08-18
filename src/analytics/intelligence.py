"""Core insight and opportunity data structures.

Provides the Insight and Opportunity dataclasses that represent structured
analytical outputs, along with conversion functions to DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Insight:
    """A structured insight from analytics.

    Represents a single analytical finding with standardized fields for
    rendering in the UI decision center.
    """

    domain: str  # e.g., "assortment", "pricing", "customer"
    entity: str  # what the insight is about (SKU, customer, etc.)
    kind: str  # "opportunity", "risk", "growth", "leakage", "anomaly", "efficiency", "watch"
    title: str  # human-readable headline
    evidence: str  # supporting data/explanation
    action: str  # recommended action
    confidence: str  # "high", "medium", "low", "insufficient"
    impact_value: float | None = None  # quantitative impact (revenue, etc.)
    sample_size: int | None = None  # n observations
    stability: float | None = None  # [0,1] stability score
    evidence_level: int | None = (
        None  # 1-5: exploratory, descriptive, predictive, quasi-causal, causal
    )
    n_transition_pairs: int | None = None  # number of switching transition pairs
    n_unique_products: int | None = None  # number of unique products involved in switching
    confidence_gate: bool | None = None  # whether insight meets minimum evidence threshold


@dataclass(frozen=True)
class Opportunity:
    """A structured opportunity from analytics.

    Represents a specific actionable opportunity with quantified value.
    """

    domain: str  # e.g., "assortment", "pricing", "customer"
    entity: str  # what the opportunity is about
    title: str  # human-readable headline
    value: float  # quantified value (revenue, savings, etc.)
    confidence: str  # "high", "medium", "low", "insufficient"
    action: str  # recommended action
    source: str  # analytical method that generated this
    rationale: str  # explanation of why this is an opportunity


def insights_to_dataframe(insights: list[Insight]) -> pd.DataFrame:
    """Convert a list of Insight objects to a DataFrame.

    Args:
        insights: List of Insight dataclass instances

    Returns:
        DataFrame with columns matching PRICING_INSIGHTS contract
    """
    if not insights:
        return pd.DataFrame(
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
            ],
            dtype={
                "evidence_level": "Int64",
                "n_transition_pairs": "Int64",
                "n_unique_products": "Int64",
                "confidence_gate": "boolean",
            }
        )

    data = [
        {
            "domain": insight.domain,
            "entity": insight.entity,
            "kind": insight.kind,
            "title": insight.title,
            "evidence": insight.evidence,
            "impact_value": insight.impact_value,
            "confidence": insight.confidence,
            "sample_size": insight.sample_size,
            "stability": insight.stability,
            "action": insight.action,
            "evidence_level": insight.evidence_level if insight.evidence_level is not None else 2,
            "n_transition_pairs": insight.n_transition_pairs if insight.n_transition_pairs is not None else 0,
            "n_unique_products": insight.n_unique_products if insight.n_unique_products is not None else 0,
            "confidence_gate": insight.confidence_gate if insight.confidence_gate is not None else False,
        }
        for insight in insights
    ]

    df = pd.DataFrame(data)
    # Ensure proper types for validation
    df["evidence_level"] = df["evidence_level"].fillna(2).astype("Int64")
    df["n_transition_pairs"] = df["n_transition_pairs"].fillna(0).astype("Int64")
    df["n_unique_products"] = df["n_unique_products"].fillna(0).astype("Int64")
    df["confidence_gate"] = df["confidence_gate"].fillna(False).astype(bool)

    return df


def opportunities_to_dataframe(opportunities: list[Opportunity]) -> pd.DataFrame:
    """Convert a list of Opportunity objects to a DataFrame.

    Args:
        opportunities: List of Opportunity dataclass instances

    Returns:
        DataFrame with columns matching OPPORTUNITY_LIST contract
    """
    if not opportunities:
        return pd.DataFrame(
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

    data = [
        {
            "domain": opp.domain,
            "entity": opp.entity,
            "title": opp.title,
            "value": opp.value,
            "confidence": opp.confidence,
            "action": opp.action,
            "source": opp.source,
            "rationale": opp.rationale,
        }
        for opp in opportunities
    ]

    return pd.DataFrame(data)
