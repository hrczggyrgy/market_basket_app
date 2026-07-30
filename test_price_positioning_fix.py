#!/usr/bin/env python3
"""
Test script to verify the fix for compute_price_positioning_index function.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.analytics.product_performance import compute_price_positioning_index

def test_with_category_column():
    """Test the function when category column is present."""
    print("Testing with category column...")
    
    # Create test data with category column
    data = {
        'stockcode': ['A', 'A', 'B', 'B', 'C', 'C'],
        'price': [10, 12, 20, 22, 15, 18],
        'category': ['Cat1', 'Cat1', 'Cat1', 'Cat1', 'Cat2', 'Cat2']
    }
    df = pd.DataFrame(data)
    
    result = compute_price_positioning_index(df)
    
    # Check that we get the expected columns
    expected_columns = ['stockcode', 'category', 'product_price', 'category_median_price', 'price_index']
    assert list(result.columns) == expected_columns, f"Expected columns {expected_columns}, got {list(result.columns)}"
    
    # Check that we have data (not empty)
    assert len(result) > 0, "Result should not be empty when category column is present"
    
    # Check specific values
    # Product A: prices 10, 12 -> median 11
    # Product B: prices 20, 22 -> median 21
    # Product C: prices 15, 18 -> median 16.5
    # Category Cat1: prices 10,12,20,22 -> median 16
    # Category Cat2: prices 15,18 -> median 16.5
    # Price index for A: 11/16 = 0.6875
    # Price index for B: 21/16 = 1.3125
    # Price index for C: 16.5/16.5 = 1.0
    
    result_a = result[result['stockcode'] == 'A'].iloc[0]
    result_b = result[result['stockcode'] == 'B'].iloc[0]
    result_c = result[result['stockcode'] == 'C'].iloc[0]
    
    assert abs(result_a['product_price'] - 11.0) < 0.001
    assert abs(result_b['product_price'] - 21.0) < 0.001
    assert abs(result_c['product_price'] - 16.5) < 0.001
    
    assert abs(result_a['category_median_price'] - 16.0) < 0.001
    assert abs(result_b['category_median_price'] - 16.0) < 0.001
    assert abs(result_c['category_median_price'] - 16.5) < 0.001
    
    assert abs(result_a['price_index'] - 0.6875) < 0.001
    assert abs(result_b['price_index'] - 1.3125) < 0.001
    assert abs(result_c['price_index'] - 1.0) < 0.001
    
    print("✓ Test with category column passed")
    return True

def test_without_category_column():
    """Test the function when category column is not present."""
    print("Testing without category column...")
    
    # Create test data without category column
    data = {
        'stockcode': ['A', 'A', 'B', 'B'],
        'price': [10, 12, 20, 22]
        # No category column
    }
    df = pd.DataFrame(data)
    
    result = compute_price_positioning_index(df)
    
    # Check that we get the expected columns even when category is missing
    expected_columns = ['stockcode', 'category', 'product_price', 'category_median_price', 'price_index']
    assert list(result.columns) == expected_columns, f"Expected columns {expected_columns}, got {list(result.columns)}"
    
    # Check that the result is empty (no rows)
    assert len(result) == 0, f"Expected empty result when category column is missing, got {len(result)} rows"
    
    print("✓ Test without category column passed")
    return True

def test_empty_dataframe():
    """Test the function with an empty DataFrame."""
    print("Testing with empty DataFrame...")
    
    # Create empty DataFrame with required columns except category
    df = pd.DataFrame(columns=['stockcode', 'price'])
    
    result = compute_price_positioning_index(df)
    
    # Check that we get the expected columns
    expected_columns = ['stockcode', 'category', 'product_price', 'category_median_price', 'price_index']
    assert list(result.columns) == expected_columns, f"Expected columns {expected_columns}, got {list(result.columns)}"
    
    # Check that the result is empty
    assert len(result) == 0, f"Expected empty result, got {len(result)} rows"
    
    print("✓ Test with empty DataFrame passed")
    return True

if __name__ == "__main__":
    print("Testing compute_price_positioning_index function...")
    
    try:
        test_with_category_column()
        test_without_category_column()
        test_empty_dataframe()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)