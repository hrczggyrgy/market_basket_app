"""Unit tests for the Pricing page UI functions."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ui.tabs.pricing_page import _render_price_simulation, _render_scorecard


def test_render_scorecard_guard_invalid_type():
    """Test that _render_scorecard returns early with error for non-PricingAnalysis input."""
    # Pass a dict that looks like it might have the right structure but isn't a PricingAnalysis
    invalid_analysis = {
        "elasticity_status": pd.DataFrame({"stockcode": ["A"], "elasticity_status": ["estimated"]}),
        "elasticity": pd.DataFrame(),
        "confidence": pd.DataFrame(),
        "kvi": pd.DataFrame(),
        "decision_matrix": pd.DataFrame(),
        "insights": pd.DataFrame(),
        "opportunities": pd.DataFrame(),
    }

    # This should not crash and should return early
    _render_scorecard(invalid_analysis)  # type: ignore[arg-type]


def test_render_price_simulation_guard_invalid_type():
    """Test that _render_price_simulation returns early with error for non-PricingAnalysis input."""
    # Pass a dict that looks like it might have the right structure but isn't a PricingAnalysis
    invalid_analysis = {
        "elasticity": pd.DataFrame({"stockcode": ["A"], "elasticity": [-1.5], "avg_price": [10.0], "avg_weekly_qty": [100.0], "n_obs": [30]}),
        "elasticity_status": pd.DataFrame({"stockcode": ["A"], "elasticity_status": ["estimated"]}),
        "confidence": pd.DataFrame({"stockcode": ["A"], "confidence": ["high"]}),
        "kvi": pd.DataFrame(),
        "decision_matrix": pd.DataFrame(),
        "insights": pd.DataFrame(),
        "opportunities": pd.DataFrame(),
    }

    # This should not crash and should return early
    _render_price_simulation(invalid_analysis)  # type: ignore[arg-type]


def test_render_scorecard_guard_none_input():
    """Test that _render_scorecard handles None input gracefully."""
    _render_scorecard(None)  # type: ignore[arg-type]


def test_render_price_simulation_guard_none_input():
    """Test that _render_price_simulation handles None input gracefully."""
    _render_price_simulation(None)  # type: ignore[arg-type]


def test_render_scorecard_guard_string_input():
    """Test that _render_scorecard handles string input gracefully."""
    _render_scorecard("not an analysis object")  # type: ignore[arg-type]


def test_render_price_simulation_guard_string_input():
    """Test that _render_price_simulation handles string input gracefully."""
    _render_price_simulation("not an analysis object")  # type: ignore[arg-type]


def test_render_scorecard_guard_empty_dict():
    """Test that _render_scorecard handles empty dict input gracefully."""
    _render_scorecard({})  # type: ignore[arg-type]


def test_render_price_simulation_guard_empty_dict():
    """Test that _render_price_simulation handles empty dict input gracefully."""
    _render_price_simulation({})  # type: ignore[arg-type]