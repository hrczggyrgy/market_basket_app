"""Consolidated UI component tests using single CSV data source.

This replaces multiple duplicate test files:
- test_ui_features.py
- test_pricing_page.py
- test_insights_engines.py
- test_opportunities_pricing.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.data import load_transactions
from src.analytics.schemas import check


class TestUIFeatures:
    """Test UI feature utilities."""

    def test_get_product_lookup(self, sample_df: pd.DataFrame):
        """Product lookup creation."""
        from src.ui.features import get_product_lookup
        lookup = get_product_lookup(sample_df)
        assert len(lookup) == sample_df["stockcode"].nunique()
        assert "stockcode" in lookup.columns

    def test_get_basket_matrix(self, sample_df: pd.DataFrame):
        """Basket matrix creation."""
        from src.ui.features import get_basket_matrix
        matrix = get_basket_matrix(sample_df)
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.shape[0] == sample_df["transaction_id"].nunique()

    def test_get_detected_promotions(self, sample_df: pd.DataFrame):
        """Detected promotions retrieval."""
        from src.ui.features import get_detected_promotions
        promos = get_detected_promotions(sample_df)
        assert isinstance(promos, pd.DataFrame)


class TestInsightsEngines:
    """Test insight generation engines."""

    def test_overview_insights(self, sample_df: pd.DataFrame):
        """Overview insights generation."""
        from src.analytics.insights.overview import generate_overview_insights
        result = generate_overview_insights(sample_df)
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert "domain" in result.columns
            assert "kind" in result.columns

    def test_pricing_insights(self, sample_df: pd.DataFrame):
        """Pricing insights generation."""
        from src.analytics.insights.pricing import generate_pricing_insights
        from src.analytics.pricing.elasticity import estimate_loglog_elasticity
        elast = estimate_loglog_elasticity(sample_df, min_periods=5)
        if len(elast) > 0:
            result = generate_pricing_insights(elast, pd.DataFrame())
            assert isinstance(result, pd.DataFrame)

    def test_switching_insights(self, sample_df: pd.DataFrame):
        """Switching insights generation."""
        from src.analytics.insights.switching import generate_switching_insights
        from src.analytics.transference import compute_demand_transference_matrix
        from src.analytics.transference import compute_substitutable_demand_percentage
        dt = compute_demand_transference_matrix(sample_df)
        sdp = compute_substitutable_demand_percentage(dt, sample_df)
        result = generate_switching_insights(dt, sdp)
        assert isinstance(result, pd.DataFrame)

    def test_promotion_insights(self, sample_df: pd.DataFrame):
        """Promotion insights generation."""
        from src.analytics.insights.promotion import generate_promotion_insights
        from src.analytics.promo_core import detect_promotions, compute_promo_baseline
        promos = detect_promotions(sample_df)
        if len(promos) > 0:
            baseline = compute_promo_baseline(sample_df, promos)
            result = generate_promotion_insights(promos, baseline)
            assert isinstance(result, pd.DataFrame)


class TestOpportunities:
    """Test opportunity generation."""

    def test_cross_sell_opportunities(self, sample_df: pd.DataFrame):
        """Cross-sell opportunity generation."""
        from src.analytics.opportunities.cross_sell import generate_cross_sell_opportunities
        from src.analytics.rules import generate_rules
        rules = generate_rules(sample_df, min_support=0.01, min_confidence=0.1)
        if len(rules) > 0:
            result = generate_cross_sell_opportunities(rules, sample_df)
            assert isinstance(result, pd.DataFrame)

    def test_switching_opportunities(self, sample_df: pd.DataFrame):
        """Switching opportunities."""
        from src.analytics.opportunities.switching import generate_switching_opportunities
        from src.analytics.transference import compute_demand_transference_matrix
        from src.analytics.transference import compute_substitutable_demand_percentage
        dt = compute_demand_transference_matrix(sample_df)
        sdp = compute_substitutable_demand_percentage(dt, sample_df)
        result = generate_switching_opportunities(dt)
        assert isinstance(result, pd.DataFrame)

    def test_promotion_opportunities(self, sample_df: pd.DataFrame):
        """Promotion opportunities."""
        from src.analytics.opportunities.promotion import generate_promotion_opportunities
        from src.analytics.promo_core import detect_promotions, compute_promo_baseline
        promos = detect_promotions(sample_df)
        if len(promos) > 0:
            baseline = compute_promo_baseline(sample_df, promos)
            result = generate_promotion_opportunities(promos, baseline)
            assert isinstance(result, pd.DataFrame)


class TestAssortmentOpportunities:
    """Test assortment optimization opportunities."""

    def test_assortment_optimization(self, sample_df: pd.Data.DataFrame):
        """Assortment optimization heuristic."""
        from src.analytics.assortment import optimize_assortment_heuristic
        result = optimize_assortment_heuristic(sample_df, max_skus=50, min_coverage=0.8)
        assert isinstance(result, list)
        assert len(result) <= 50


class TestDelistSimulator:
    """Test delist simulation."""

    def test_delist_impact(self, sample_df: pd.DataFrame):
        """Delist impact analysis."""
        from src.analytics.transference import delist_impact_analysis
        from src.analytics.transference import compute_demand_transference_matrix
        dt = compute_demand_transference_matrix(sample_df)
        if len(dt) > 0:
            result = delist_impact_analysis(dt, sample_df, top_n=5)
            assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])