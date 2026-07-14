"""Tests for association rule generation and filtering."""

import pandas as pd
import pytest

from src.rules.generator import (
    add_extended_metrics,
    filter_rules,
    format_rules_for_display,
    generate_rules,
    _empty_rules_df,
)


def create_sample_rules():
    """Create sample rules DataFrame for testing."""
    data = {
        "antecedents": [
            frozenset(["A"]),
            frozenset(["B"]),
            frozenset(["A", "B"]),
            frozenset(["C"]),
        ],
        "consequents": [
            frozenset(["B"]),
            frozenset(["C"]),
            frozenset(["C"]),
            frozenset(["A"]),
        ],
        "antecedent support": [0.4, 0.3, 0.2, 0.25],
        "consequent support": [0.3, 0.2, 0.2, 0.4],
        "support": [0.2, 0.15, 0.1, 0.1],
        "confidence": [0.5, 0.5, 0.5, 0.4],
        "lift": [1.67, 2.5, 2.5, 1.0],
    }
    return pd.DataFrame(data)


def test_empty_rules_df():
    """Test empty rules DataFrame creation."""
    empty_df = _empty_rules_df()
    assert empty_df.empty
    assert "antecedents" in empty_df.columns
    assert "consequents" in empty_df.columns
    assert "lift" in empty_df.columns
    assert "leverage" in empty_df.columns
    assert "conviction" in empty_df.columns


def test_add_extended_metrics():
    """Test extended metrics calculation."""
    rules = create_sample_rules()
    rules_with_metrics = add_extended_metrics(rules)
    
    # Check that new columns are added
    assert "leverage" in rules_with_metrics.columns
    assert "conviction" in rules_with_metrics.columns
    assert "zhangs_metric" in rules_with_metrics.columns
    assert "collective_strength" in rules_with_metrics.columns
    assert "chi_squared" in rules_with_metrics.columns
    assert "phi_coefficient" in rules_with_metrics.columns
    assert "cosine" in rules_with_metrics.columns
    assert "kulczynski" in rules_with_metrics.columns
    assert "imbalance_ratio" in rules_with_metrics.columns
    assert "odds_ratio" in rules_with_metrics.columns
    assert "certainty_factor" in rules_with_metrics.columns
    assert "added_value" in rules_with_metrics.columns
    assert "sebag_schoenauer" in rules_with_metrics.columns
    assert "jaccard" in rules_with_metrics.columns
    assert "gini_index" in rules_with_metrics.columns
    assert "laplace" in rules_with_metrics.columns
    
    # Check some calculations
    # Leverage = P(A,B) - P(A)P(B)
    # Rule 0: support=0.2, ant_support=0.4, cons_support=0.3
    # leverage = 0.2 - (0.4 * 0.3) = 0.2 - 0.12 = 0.08
    assert abs(rules_with_metrics.iloc[0]["leverage"] - 0.08) < 0.01


def test_filter_rules_min_support():
    """Test filtering by minimum support."""
    rules = create_sample_rules()
    rules = add_extended_metrics(rules)
    # Add length columns that filter_rules expects
    rules["antecedent_len"] = rules["antecedents"].apply(len)
    rules["consequent_len"] = rules["consequents"].apply(len)
    
    filtered = filter_rules(rules, min_support=0.15)
    assert len(filtered) == 2  # Rules with support >= 0.15


def test_filter_rules_min_confidence():
    """Test filtering by minimum confidence."""
    rules = create_sample_rules()
    rules = add_extended_metrics(rules)
    # Add length columns that filter_rules expects
    rules["antecedent_len"] = rules["antecedents"].apply(len)
    rules["consequent_len"] = rules["consequents"].apply(len)
    
    filtered = filter_rules(rules, min_confidence=0.45)
    assert len(filtered) == 3  # Rules with confidence >= 0.45


def test_filter_rules_min_lift():
    """Test filtering by minimum lift."""
    rules = create_sample_rules()
    rules = add_extended_metrics(rules)
    # Add length columns that filter_rules expects
    rules["antecedent_len"] = rules["antecedents"].apply(len)
    rules["consequent_len"] = rules["consequents"].apply(len)
    
    filtered = filter_rules(rules, min_lift=2.0)
    assert len(filtered) == 2  # Rules with lift >= 2.0


def test_filter_rules_max_lift():
    """Test filtering by maximum lift."""
    rules = create_sample_rules()
    rules = add_extended_metrics(rules)
    # Add length columns that filter_rules expects
    rules["antecedent_len"] = rules["antecedents"].apply(len)
    rules["consequent_len"] = rules["consequents"].apply(len)
    
    filtered = filter_rules(rules, max_lift=2.0)
    assert len(filtered) == 2  # Rules with lift <= 2.0


def test_filter_rules_min_leverage():
    """Test filtering by minimum leverage."""
    rules = create_sample_rules()
    rules = add_extended_metrics(rules)
    # Add length columns that filter_rules expects
    rules["antecedent_len"] = rules["antecedents"].apply(len)
    rules["consequent_len"] = rules["consequents"].apply(len)
    
    filtered = filter_rules(rules, min_leverage=0.05)
    assert len(filtered) >= 1  # At least one rule with leverage >= 0.05


def test_filter_rules_min_conviction():
    """Test filtering by minimum conviction."""
    rules = create_sample_rules()
    rules = add_extended_metrics(rules)
    # Add length columns that filter_rules expects
    rules["antecedent_len"] = rules["antecedents"].apply(len)
    rules["consequent_len"] = rules["consequents"].apply(len)
    
    filtered = filter_rules(rules, min_conviction=1.5)
    assert len(filtered) >= 1


def test_filter_rules_antecedent_len():
    """Test filtering by antecedent length."""
    rules = create_sample_rules()
    rules = add_extended_metrics(rules)
    
    # Add antecedent_len column manually since our sample doesn't have it
    rules["antecedent_len"] = rules["antecedents"].apply(len)
    rules["consequent_len"] = rules["consequents"].apply(len)
    
    filtered = filter_rules(rules, max_antecedent_len=1)
    assert len(filtered) == 3  # Only rules with antecedent length <= 1


def test_filter_rules_consequent_len():
    """Test filtering by consequent length."""
    rules = create_sample_rules()
    rules = add_extended_metrics(rules)
    
    rules["antecedent_len"] = rules["antecedents"].apply(len)
    rules["consequent_len"] = rules["consequents"].apply(len)
    
    filtered = filter_rules(rules, max_consequent_len=1)
    assert len(filtered) == 4  # All rules have consequent length 1


def test_filter_rules_combined():
    """Test combined filtering."""
    rules = create_sample_rules()
    rules = add_extended_metrics(rules)
    rules["antecedent_len"] = rules["antecedents"].apply(len)
    rules["consequent_len"] = rules["consequents"].apply(len)
    
    filtered = filter_rules(
        rules,
        min_support=0.1,
        min_confidence=0.45,
        min_lift=2.0,  # Changed from 1.5 to 2.0 to filter out Rule 0
        max_lift=3.0,
    )
    assert len(filtered) == 2  # Should filter out Rule 0 (lift=1.67 < 2.0)


def test_filter_rules_empty():
    """Test filtering empty DataFrame."""
    empty_rules = _empty_rules_df()
    filtered = filter_rules(empty_rules, min_support=0.1)
    assert filtered.empty


def test_format_rules_for_display():
    """Test formatting rules for display."""
    rules = create_sample_rules()
    rules = add_extended_metrics(rules)
    rules["antecedent_len"] = rules["antecedents"].apply(len)
    rules["consequent_len"] = rules["consequents"].apply(len)
    
    product_lookup = {"A": "Product A", "B": "Product B", "C": "Product C"}
    
    display_rules = format_rules_for_display(rules, product_lookup)
    
    assert "antecedents_str" in display_rules.columns
    assert "consequents_str" in display_rules.columns
    assert "rule" in display_rules.columns
    
    # Check that product names are used
    assert "Product A" in display_rules.iloc[0]["antecedents_str"]
    assert "Product B" in display_rules.iloc[0]["consequents_str"]


def test_format_rules_for_display_no_lookup():
    """Test formatting rules without product lookup."""
    rules = create_sample_rules()
    rules = add_extended_metrics(rules)
    rules["antecedent_len"] = rules["antecedents"].apply(len)
    rules["consequent_len"] = rules["consequents"].apply(len)
    
    display_rules = format_rules_for_display(rules, None)
    
    assert "antecedents_str" in display_rules.columns
    assert "consequents_str" in display_rules.columns
    assert "rule" in display_rules.columns
    
    # Check that raw item names are used
    assert "A" in display_rules.iloc[0]["antecedents_str"]


def test_leverage_calculation():
    """Test leverage calculation specifically."""
    # P(A,B) = 0.2, P(A) = 0.4, P(B) = 0.3
    # leverage = 0.2 - 0.4*0.3 = 0.2 - 0.12 = 0.08
    rules = pd.DataFrame({
        "antecedents": [frozenset(["A"])],
        "consequents": [frozenset(["B"])],
        "antecedent support": [0.4],
        "consequent support": [0.3],
        "support": [0.2],
        "confidence": [0.5],
        "lift": [1.67],
    })
    
    result = add_extended_metrics(rules)
    assert abs(result.iloc[0]["leverage"] - 0.08) < 0.01


def test_conviction_calculation():
    """Test conviction calculation."""
    # P(B) = 0.3, confidence = 0.5
    # conviction = (1 - 0.3) / (1 - 0.5) = 0.7 / 0.5 = 1.4
    rules = pd.DataFrame({
        "antecedents": [frozenset(["A"])],
        "consequents": [frozenset(["B"])],
        "antecedent support": [0.4],
        "consequent support": [0.3],
        "support": [0.2],
        "confidence": [0.5],
        "lift": [1.67],
    })
    
    result = add_extended_metrics(rules)
    assert abs(result.iloc[0]["conviction"] - 1.4) < 0.01


def test_conviction_infinite_when_confidence_is_1():
    """Test conviction is infinite when confidence is 1."""
    rules = pd.DataFrame({
        "antecedents": [frozenset(["A"])],
        "consequents": [frozenset(["B"])],
        "antecedent support": [0.5],
        "consequent support": [0.5],
        "support": [0.5],
        "confidence": [1.0],
        "lift": [2.0],
    })
    
    result = add_extended_metrics(rules)
    assert result.iloc[0]["conviction"] == float("inf")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])