"""Tests for pricing reliability improvements (Wave 1).

Tests verify that:
1. Missing elasticity is never treated as zero
2. Estimability statuses are granular and correct
3. Confidence framework gates decisions appropriately
4. Price simulation respects confidence levels
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics.pricing.decision import compute_pricing_decision_matrix
from src.analytics.pricing.elasticity import (
    _check_estimable,
    classify_elasticity_confidence,
    compute_elasticity_status,
    estimate_loglog_elasticity,
)
from src.analytics.pricing.kvi import compute_kvi_score


def test_check_estimable_constant_price():
    """Test that constant prices are correctly identified."""
    prices = pd.Series([10.0, 10.0, 10.0])
    qtys = pd.Series([100, 95, 90])
    
    result = _check_estimable(np.log(prices), np.log(qtys))
    assert result == "near_constant_price"


def test_check_estimable_insufficient_price_points():
    """Test that insufficient distinct price points are identified."""
    prices = pd.Series([10.0, 10.5])  # Only 2 distinct prices (less than _MIN_DISTINCT_PRICES=3)
    qtys = pd.Series([100, 95])
    
    result = _check_estimable(np.log(prices), np.log(qtys))
    # With only 2 distinct prices, it should be detected
    assert result == "insufficient_price_points"


def test_check_estimable_extreme_values():
    """Test that extreme log values are identified as data quality issues."""
    prices = pd.Series([1e10, 1e11, 1e12])  # Extreme values with variation
    qtys = pd.Series([100, 95, 90])
    
    result = _check_estimable(np.log(prices), np.log(qtys))
    assert result == "extreme_values"


def test_check_estimable_valid_data():
    """Test that valid data passes estimability checks."""
    prices = pd.Series([10.0, 10.5, 11.0, 10.8, 10.2])
    qtys = pd.Series([100, 95, 90, 92, 98])
    
    result = _check_estimable(np.log(prices), np.log(qtys))
    assert result is None


def test_elasticity_status_insufficient_price_points():
    """Test that insufficient_price_points status is correctly assigned."""
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10, freq='W'),
        'stockcode': ['A'] * 10,
        'price': [10.0, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5],  # Only 2 distinct prices
        'quantity': [100, 95, 90, 92, 98, 105, 102, 97, 93, 96],
    })
    
    status = compute_elasticity_status(df, min_periods=5)
    assert not status.empty
    # With only 2 distinct price points, should be insufficient_price_points
    assert status['elasticity_status'].iloc[0] == 'insufficient_price_points'


def test_elasticity_status_insufficient_variation():
    """Test that insufficient_variation status is correctly assigned."""
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10, freq='W'),
        'stockcode': ['A'] * 10,
        'price': [10.0, 10.01, 10.02, 9.99, 10.01, 10.0, 10.01, 9.98, 10.02, 10.0],
        'quantity': [100, 95, 90, 92, 98, 105, 102, 97, 93, 96],
    })
    
    status = compute_elasticity_status(df, min_periods=5, min_price_variation=0.05)
    assert not status.empty
    assert status['elasticity_status'].iloc[0] == 'insufficient_variation'


def test_elasticity_never_imputed_to_zero():
    """Test that missing elasticity is never imputed to zero in KVI scoring."""
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10, freq='W'),
        'stockcode': ['A'] * 10,
        'product': ['Product A'] * 10,
        'price': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        'quantity': [100, 95, 90, 92, 98, 105, 102, 97, 93, 96],
        'customer_id': ['C1'] * 5 + ['C2'] * 5,
        'transaction_id': [f'T{i}' for i in range(10)],
    })
    
    # Compute status (should be non-estimable)
    status = compute_elasticity_status(df, min_periods=5)
    
    # Compute KVI with status
    kvi = compute_kvi_score(df, elasticity_df=None, elasticity_status_df=status)
    
    # Verify abs_elasticity is NaN for non-estimable SKUs
    assert kvi['abs_elasticity'].isna().iloc[0]
    # Verify elasticity_status is not "estimated"
    assert kvi['elasticity_status'].iloc[0] != 'estimated'


def test_decision_matrix_gates_on_evidence():
    """Test that decision matrix gates on elasticity evidence."""
    # Test the decision logic directly with mock data
    kvi_df = pd.DataFrame({
        'stockcode': ['A', 'B', 'C', 'D'],
        'category': ['Cat1', 'Cat1', 'Cat2', 'Cat2'],
        'kvi_score': [0.8, 0.6, 0.4, 0.2],
        'total_revenue': [1000, 800, 600, 400],
        'abs_elasticity': [1.5, 0.8, 1.2, np.nan],
        'elasticity_status': ['estimated', 'estimated', 'estimated', 'insufficient_variation'],
    })
    elasticity_df = pd.DataFrame({
        'stockcode': ['A', 'B', 'C'],
        'elasticity': [-1.5, -0.8, -1.2],
    })
    
    decision = compute_pricing_decision_matrix(kvi_df, elasticity_df)
    
    # SKUs with valid elasticity should get a decision
    estimable_skus = decision[decision['decision'] != 'insufficient_evidence']
    non_estimable_skus = decision[decision['decision'] == 'insufficient_evidence']
    
    # Verify that non-estimable SKU (D) is in insufficient_evidence
    assert len(non_estimable_skus) == 1
    assert non_estimable_skus['stockcode'].iloc[0] == 'D'
    
    # Verify estimable SKUs have real decisions
    assert len(estimable_skus) == 3
    assert estimable_skus['decision'].isin(['invest', 'protect', 'price_lever', 'review']).all()


def test_confidence_classification():
    """Test that confidence classification works correctly."""
    df = pd.DataFrame({
        'stockcode': ['A', 'B', 'C'],
        'elasticity': [-1.5, -0.8, -2.0],
        'ci_lower': [-1.8, -1.5, -3.0],
        'ci_upper': [-1.2, -0.1, -1.0],
        'p_value': [0.01, 0.10, 0.001],
        'n_obs': [50, 20, 100],
    })
    
    conf = classify_elasticity_confidence(df)
    
    # Check that confidence column exists
    assert 'confidence' in conf.columns
    
    # Check that all confidences are valid
    assert conf['confidence'].isin(['high', 'medium', 'low']).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
