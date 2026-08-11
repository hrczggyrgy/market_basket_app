"""Unit tests for the Pricing opportunity generator."""

from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.opportunities import generate_pricing_opportunities
from src.analytics.pricing import compute_pricing_decision_matrix, run_pricing_analysis
from src.analytics.schemas import OPPORTUNITY_LIST
from tests.unit.pricing_fixtures import build_kvi_fixture, build_pricing_df


def _decision_matrix() -> pd.DataFrame:
    return compute_pricing_decision_matrix(build_kvi_fixture())


def test_opportunities_from_decision_matrix() -> None:
    dm = _decision_matrix()
    opps = generate_pricing_opportunities(pd.DataFrame(), pd.DataFrame(), build_kvi_fixture(), dm)
    OPPORTUNITY_LIST.validate(opps)
    assert not opps.empty

    by_entity = opps.set_index("entity")
    assert "ELASTIC_HI" in by_entity.index  # invest -> protected traffic driver
    assert "INELASTIC" in by_entity.index  # protect -> margin carrier
    assert "PRICE_LEVER" in by_entity.index  # price_lever -> -5% test

    # review and insufficient-evidence SKUs never become opportunities
    assert "REVIEW" not in by_entity.index
    assert "CONST_SKU" not in by_entity.index
    assert "SHORT_SKU" not in by_entity.index


def test_opportunities_illustrative_value_semantics() -> None:
    dm = _decision_matrix()
    opps = generate_pricing_opportunities(pd.DataFrame(), pd.DataFrame(), build_kvi_fixture(), dm)
    OPPORTUNITY_LIST.validate(opps)
    price_lever = opps[opps["entity"] == "PRICE_LEVER"].iloc[0]
    invest = opps[opps["entity"] == "ELASTIC_HI"].iloc[0]
    # price_lever value = illustrative incremental revenue of a -5% cut (> 0)
    assert price_lever["value"] > 0
    # invest/protect value = revenue exposure
    assert invest["value"] == pytest.approx(632000.0)


def test_opportunities_exclude_low_confidence() -> None:
    """A weak SKU must not produce an opportunity (decision gating)."""
    dm = _decision_matrix()
    opps = generate_pricing_opportunities(pd.DataFrame(), pd.DataFrame(), build_kvi_fixture(), dm)
    assert "WEAK" not in set(opps["entity"])


def test_opportunities_empty_input() -> None:
    opps = generate_pricing_opportunities(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert opps.empty
    OPPORTUNITY_LIST.validate(opps, allow_empty=True)


def test_opportunities_all_insufficient_evidence() -> None:
    """All-insufficient decision matrix yields no opportunities."""
    kvi = build_kvi_fixture()
    kvi = kvi.assign(elasticity_status="insufficient_variation", abs_elasticity=pd.NA)
    dm = compute_pricing_decision_matrix(kvi)
    assert (dm["decision"] == "insufficient_evidence").all()
    opps = generate_pricing_opportunities(pd.DataFrame(), pd.DataFrame(), kvi, dm)
    assert opps.empty
    OPPORTUNITY_LIST.validate(opps, allow_empty=True)


def test_opportunities_synthetic_pipeline() -> None:
    analysis = run_pricing_analysis(build_pricing_df(), min_periods=5)
    opps = generate_pricing_opportunities(
        analysis.elasticity,
        analysis.elasticity_status,
        analysis.kvi,
        analysis.decision_matrix,
    )
    OPPORTUNITY_LIST.validate(opps, allow_empty=True)
    if not opps.empty:
        # opportunities must be ranked by value
        assert (opps["value"].fillna(-1).sort_values(ascending=False).values == opps["value"].fillna(-1).values).all()
        assert opps["entity"].str.len().notna().all()
