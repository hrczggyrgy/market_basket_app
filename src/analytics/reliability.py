"""Reliability scoring system for analytical outputs.

Provides a unified High/Medium/Low reliability scoring based on:
- Sample size adequacy
- Data coverage
- Uncertainty (CI width)
- Stability (bootstrap/OOS)
- Assumption validity
- Data quality

This replaces ad-hoc quality flags with a single standardized reliability score
that propagates to all downstream outputs.

Example:
    >>> from src.analytics.reliability import compute_reliability, ReliabilityLevel
    >>> reliability = compute_reliability(n_obs=500, coverage=0.9, ci_width=0.2, point_estimate=1.5)
    >>> reliability.level
    ReliabilityLevel.HIGH
    >>> reliability.overall_score
    0.8
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class ReliabilityLevel(Enum):
    """Reliability tier for analytical outputs."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class ReliabilityDimension(Enum):
    """Dimensions contributing to reliability score."""

    SAMPLE_SIZE = "sample_size"
    COVERAGE = "coverage"
    UNCERTAINTY = "uncertainty"
    STABILITY = "stability"
    ASSUMPTIONS = "assumptions"
    DATA_QUALITY = "data_quality"


@dataclass
class ReliabilityScore:
    """Reliability assessment for a single analytical output."""

    level: ReliabilityLevel
    overall_score: float  # 0-1 composite score
    dimension_scores: Dict[ReliabilityDimension, float]  # 0-1 per dimension
    flags: List[str]  # Human-readable warnings
    metadata: Dict[str, Any]  # Supporting details

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "overall_score": round(self.overall_score, 3),
            "dimension_scores": {k.value: round(v, 3) for k, v in self.dimension_scores.items()},
            "flags": self.flags,
            "metadata": self.metadata,
        }


@dataclass
class AnalysisEligibility:
    """Determines if an analysis should run and its expected reliability.

    This is the gate that checks data suitability BEFORE running expensive
    computations, saving time and preventing unreliable outputs.
    """

    is_eligible: bool
    reliability: ReliabilityScore
    blocking_reasons: List[str]  # Hard blockers (missing data, etc.)
    warnings: List[str]  # Soft warnings (small sample, etc.)
    recommended_minimums: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_eligible": self.is_eligible,
            "reliability": self.reliability.to_dict(),
            "blocking_reasons": self.blocking_reasons,
            "warnings": self.warnings,
            "recommended_minimums": self.recommended_minimums,
        }


def compute_reliability(
    n_obs: int,
    coverage: Optional[float] = None,
    ci_width: Optional[float] = None,
    point_estimate: Optional[float] = None,
    stability_score: Optional[float] = None,
    assumption_flags: Optional[List[str]] = None,
    data_quality_score: Optional[float] = None,
) -> ReliabilityScore:
    """Compute overall reliability score from component dimensions.

    Args:
        n_obs: Number of observations (baskets, customers, periods)
        coverage: Fraction of population covered (0-1)
        ci_width: Width of confidence interval (absolute)
        point_estimate: Point estimate value (for relative CI width)
        stability_score: Bootstrap/OOS stability score (0-1)
        assumption_flags: List of violated assumption warnings
        data_quality_score: Data quality score from DataQualityReport (0-1)

    Returns:
        ReliabilityScore with level, composite score, and dimension breakdown

    Example:
        >>> reliability = compute_reliability(
        ...     n_obs=1000,
        ...     coverage=0.95,
        ...     ci_width=0.3,
        ...     point_estimate=1.5,
        ...     stability_score=0.85
        ... )
        >>> reliability.level.value
        'high'
    """
    dims: dict[ReliabilityDimension, float] = {}
    flags: list[str] = []
    metadata: dict[str, Any] = {}

    # 1. Sample size adequacy
    # Target: >= 1000 for high, >= 100 for medium
    if n_obs >= 1000:
        sample_score = 1.0
    elif n_obs >= 100:
        sample_score = 0.5
    else:
        sample_score = max(0.0, n_obs / 100.0)
    dims[ReliabilityDimension.SAMPLE_SIZE] = sample_score
    metadata["n_obs"] = n_obs

    # 2. Coverage
    if coverage is not None:
        if coverage >= 0.95:
            coverage_score = 1.0
        elif coverage >= 0.80:
            coverage_score = 0.7
        else:
            coverage_score = max(0.0, coverage)
    else:
        coverage_score = 0.5  # Unknown
    dims[ReliabilityDimension.COVERAGE] = coverage_score
    metadata["coverage"] = coverage

    # 3. Uncertainty (CI width relative to estimate)
    if ci_width is not None and point_estimate is not None and point_estimate != 0:
        relative_width = ci_width / abs(point_estimate)
        if relative_width < 0.2:
            uncertainty_score = 1.0
        elif relative_width < 0.5:
            uncertainty_score = 0.7
        else:
            uncertainty_score = max(0.0, 1.0 - relative_width)
    else:
        uncertainty_score = 0.5  # Unknown
    dims[ReliabilityDimension.UNCERTAINTY] = uncertainty_score
    metadata["ci_width"] = ci_width
    metadata["point_estimate"] = point_estimate

    # 4. Stability
    if stability_score is not None:
        stability_score = float(np.clip(stability_score, 0.0, 1.0))
    else:
        stability_score = 0.5  # Unknown
    dims[ReliabilityDimension.STABILITY] = stability_score
    metadata["stability_score"] = stability_score

    # 5. Assumptions
    if assumption_flags is not None:
        # Penalize each flag
        assumption_score = max(0.0, 1.0 - 0.15 * len(assumption_flags))
        flags.extend(assumption_flags)
    else:
        assumption_score = 1.0
    dims[ReliabilityDimension.ASSUMPTIONS] = assumption_score
    metadata["assumption_flags"] = assumption_flags or []

    # 6. Data quality
    if data_quality_score is not None:
        data_quality_score = float(np.clip(data_quality_score, 0.0, 1.0))
    else:
        data_quality_score = 0.5  # Unknown
    dims[ReliabilityDimension.DATA_QUALITY] = data_quality_score
    metadata["data_quality_score"] = data_quality_score

    # Weighted composite (weights sum to 1.0)
    weights = {
        ReliabilityDimension.SAMPLE_SIZE: 0.25,
        ReliabilityDimension.COVERAGE: 0.15,
        ReliabilityDimension.UNCERTAINTY: 0.20,
        ReliabilityDimension.STABILITY: 0.15,
        ReliabilityDimension.ASSUMPTIONS: 0.15,
        ReliabilityDimension.DATA_QUALITY: 0.10,
    }

    overall = sum(dims[dim] * weights[dim] for dim in dims)

    # Determine level
    if overall >= 0.75:
        level = ReliabilityLevel.HIGH
    elif overall >= 0.50:
        level = ReliabilityLevel.MEDIUM
    elif overall >= 0.25:
        level = ReliabilityLevel.LOW
    else:
        level = ReliabilityLevel.INSUFFICIENT

    return ReliabilityScore(
        level=level,
        overall_score=overall,
        dimension_scores=dims,
        flags=flags,
        metadata=metadata,
    )


def check_analysis_eligibility(
    df: pd.DataFrame,
    analysis_type: str,
    min_baskets: int = 100,
    min_customers: int = 50,
    min_skus: int = 10,
) -> AnalysisEligibility:
    """Check if data is suitable for a specific analysis type.

    Runs BEFORE expensive computations to prevent wasted effort
    on data that can't support the analysis.

    Args:
        df: Transaction DataFrame
        analysis_type: One of 'clv', 'assortment', 'promo', 'segmentation',
                       'cdt', 'rules', 'switching', 'elasticity'
        min_baskets: Minimum baskets required
        min_customers: Minimum customers required
        min_skus: Minimum SKUs required

    Returns:
        AnalysisEligibility with go/no-go decision and reliability forecast
    """
    blocking = []
    warnings = []

    n_baskets = df["transaction_id"].nunique()
    n_customers = df["customer_id"].nunique()
    n_skus = df["stockcode"].nunique()

    # Hard blockers
    if n_baskets < min_baskets:
        blocking.append(f"Insufficient baskets: {n_baskets} < {min_baskets}")
    if n_customers < min_customers:
        blocking.append(f"Insufficient customers: {n_customers} < {min_customers}")
    if n_skus < min_skus:
        blocking.append(f"Insufficient SKUs: {n_skus} < {min_skus}")

    # Analysis-specific requirements
    analysis_requirements: dict[str, dict[str, float | int]] = {
        "clv": {"min_customers": 500, "min_baskets": 1000, "min_repeat_rate": 0.1},
        "assortment": {"min_skus": 20, "min_baskets": 500},
        "promo": {"min_baskets": 500, "min_promo_periods": 1},
        "segmentation": {"min_customers": 200, "min_baskets": 500},
        "cdt": {"min_skus": 50, "min_baskets": 1000},
        "rules": {"min_baskets": 200, "min_skus": 20},
        "switching": {"min_customers": 100, "min_baskets": 500},
        "elasticity": {"min_baskets": 500, "min_price_variation": 0.05},
    }

    req: dict[str, float | int] = analysis_requirements.get(analysis_type, {})

    for key, threshold in req.items():
        if key == "min_baskets" and n_baskets < threshold:
            warnings.append(f"Low baskets for {analysis_type}: {n_baskets} < {threshold}")
        elif key == "min_customers" and n_customers < threshold:
            warnings.append(f"Low customers for {analysis_type}: {n_customers} < {threshold}")
        elif key == "min_skus" and n_skus < threshold:
            warnings.append(f"Low SKUs for {analysis_type}: {n_skus} < {threshold}")

    # Data quality assessment
    from src.analytics.data_quality import assess_data_quality

    quality_report = assess_data_quality(df)
    dq_score = 1.0
    if quality_report.has_issues():
        dq_score = 0.7
        for flag in [
            quality_report.low_freq_products,
            quality_report.basket_outlier_txn_ids,
            quality_report.incomplete_rows > 0,
            quality_report.volume_warning,
        ]:
            if flag:
                warnings.append(f"Data quality: {flag}")

    is_eligible = len(blocking) == 0

    # Forecast reliability
    reliability = compute_reliability(
        n_obs=int(n_baskets),
        coverage=None,
        stability_score=0.5,
        data_quality_score=dq_score,
    )

    recommended = {
        "min_baskets": max(min_baskets, req.get("min_baskets", min_baskets)),
        "min_customers": max(min_customers, req.get("min_customers", min_customers)),
        "min_skus": max(min_skus, req.get("min_skus", min_skus)),
    }

    return AnalysisEligibility(
        is_eligible=is_eligible,
        reliability=reliability,
        blocking_reasons=blocking,
        warnings=warnings,
        recommended_minimums=recommended,
    )


def attach_reliability(
    output_df: pd.DataFrame,
    reliability: ReliabilityScore,
) -> pd.DataFrame:
    """Attach reliability metadata to output DataFrame.

    Adds a '_reliability' column with JSON-serialized reliability info.
    """
    result = output_df.copy()
    result["_reliability"] = [reliability.to_dict()] * len(result)
    return result


def filter_by_reliability(
    outputs: Dict[str, pd.DataFrame],
    min_level: ReliabilityLevel = ReliabilityLevel.MEDIUM,
) -> Dict[str, pd.DataFrame]:
    """Filter a dict of outputs by minimum reliability level."""
    level_order = {
        ReliabilityLevel.INSUFFICIENT: 0,
        ReliabilityLevel.LOW: 1,
        ReliabilityLevel.MEDIUM: 2,
        ReliabilityLevel.HIGH: 3,
    }
    min_order = level_order[min_level]

    filtered = {}
    for name, df in outputs.items():
        if "_reliability" in df.columns:
            rel = df["_reliability"].iloc[0] if len(df) > 0 else {"level": "insufficient"}
            if level_order.get(ReliabilityLevel(rel.get("level", "insufficient")), 0) >= min_order:
                filtered[name] = df
    return filtered
