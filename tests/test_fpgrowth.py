"""Tests for FP-Growth algorithm."""

import pandas as pd
import pytest

from src.algorithms.frequent_itemsets import (
    create_basket_matrix,
    get_product_lookup,
    run_fpgrowth,
)


@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing."""
    return pd.DataFrame(
        {
            "transaction_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5],
            "stockcode": [
                "A",
                "B",
                "C",
                "A",
                "B",
                "A",
                "C",
                "D",
                "B",
                "C",
                "A",
                "D",
            ],
            "quantity": [1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1],
            "customer_id": [100, 100, 100, 101, 101, 102, 102, 102, 103, 103, 104, 104],
            "price": [10.0, 20.0, 15.0, 10.0, 20.0, 10.0, 15.0, 25.0, 20.0, 15.0, 10.0, 25.0],
        }
    )


@pytest.fixture
def sample_transactions_with_product():
    """Create sample transaction data with product column."""
    return pd.DataFrame(
        {
            "transaction_id": [1, 1, 2, 2],
            "stockcode": ["A", "B", "A", "C"],
            "product": ["Apple", "Banana", "Apple", "Cherry"],
            "quantity": [1, 1, 1, 1],
            "customer_id": [100, 100, 101, 101],
            "price": [10.0, 20.0, 10.0, 15.0],
        }
    )


class TestCreateBasketMatrix:
    """Tests for create_basket_matrix function."""

    def test_basic_basket_matrix(self, sample_transactions):
        """Test basic basket matrix creation."""
        basket = create_basket_matrix(sample_transactions)
        
        assert isinstance(basket, pd.DataFrame)
        assert basket.shape[0] == 5  # 5 transactions
        assert "A" in basket.columns
        assert "B" in basket.columns
        assert "C" in basket.columns
        assert "D" in basket.columns
        
        # Check values are boolean (True/False)
        assert basket["A"].dtype == bool
        
        # Transaction 1 has A, B, C (index is transaction_id = 1)
        assert basket.loc[1, "A"] == True
        assert basket.loc[1, "B"] == True
        assert basket.loc[1, "C"] == True
        assert basket.loc[1, "D"] == False

    def test_basket_matrix_with_min_quantity(self, sample_transactions):
        """Test basket matrix with min_quantity parameter."""
        basket = create_basket_matrix(sample_transactions, min_quantity=2)
        
        # The basket index uses transaction_id values
        # With min_quantity=2, transaction 1 (all quantities=1) gets filtered out
        # Transaction 2 has A with quantity 2
        assert 2 in basket.index
        
        # Transaction 2 has A with quantity 2, so A column should exist
        assert "A" in basket.columns
        assert basket.loc[2, "A"] == True
        
        # B column might not exist since no transaction has B with quantity >= 2
        # This is expected behavior - items that don't meet min_quantity are dropped


class TestGetProductLookup:
    """Tests for get_product_lookup function."""

    def test_product_lookup(self, sample_transactions_with_product):
        """Test product lookup creation."""
        lookup = get_product_lookup(sample_transactions_with_product)
        
        assert isinstance(lookup, dict)
        assert lookup["A"] == "Apple"
        assert lookup["B"] == "Banana"
        assert lookup["C"] == "Cherry"

    def test_product_lookup_with_product_column(self, sample_transactions_with_product):
        """Test product lookup with separate product column."""
        lookup = get_product_lookup(sample_transactions_with_product)
        
        assert lookup["A"] == "Apple"
        assert lookup["B"] == "Banana"
        assert lookup["C"] == "Cherry"


class TestRunFpGrowth:
    """Tests for run_fpgrowth function."""

    def test_fpgrowth_basic(self, sample_transactions):
        """Test basic FP-Growth execution."""
        basket = create_basket_matrix(sample_transactions)
        freq_items = run_fpgrowth(basket, min_support=0.2, max_len=3)
        
        assert isinstance(freq_items, pd.DataFrame)
        assert "support" in freq_items.columns
        assert "itemsets" in freq_items.columns
        assert len(freq_items) > 0
        
        # Check that itemsets are frozensets
        for itemset in freq_items["itemsets"]:
            assert isinstance(itemset, frozenset)

    def test_fpgrowth_min_support(self, sample_transactions):
        """Test FP-Growth with different min_support values."""
        basket = create_basket_matrix(sample_transactions)
        
        # High support - should find fewer itemsets
        freq_high = run_fpgrowth(basket, min_support=0.8, max_len=3)
        
        # Low support - should find more itemsets
        freq_low = run_fpgrowth(basket, min_support=0.1, max_len=3)
        
        assert len(freq_low) >= len(freq_high)

    def test_fpgrowth_max_len(self, sample_transactions):
        """Test FP-Growth with max_len parameter."""
        basket = create_basket_matrix(sample_transactions)
        
        freq_len2 = run_fpgrowth(basket, min_support=0.1, max_len=2)
        freq_len3 = run_fpgrowth(basket, min_support=0.1, max_len=3)
        
        # Max len 3 should find itemsets of size up to 3
        max_len_found = max(len(i) for i in freq_len3["itemsets"])
        assert max_len_found <= 3
        
        # Max len 2 should not find itemsets larger than 2
        max_len_found = max(len(i) for i in freq_len2["itemsets"])
        assert max_len_found <= 2

    def test_fpgrowth_empty_basket(self):
        """Test FP-Growth with empty basket."""
        empty_basket = pd.DataFrame()
        freq_items = run_fpgrowth(empty_basket, min_support=0.1, max_len=3)
        
        assert freq_items.empty
        assert list(freq_items.columns) == ["support", "itemsets"]

    def test_fpgrowth_single_item(self):
        """Test FP-Growth with single item transactions."""
        transactions = pd.DataFrame({
            "transaction_id": [1, 2, 3],
            "stockcode": ["A", "A", "B"],
            "quantity": [1, 1, 1],
        })
        basket = create_basket_matrix(transactions)
        freq_items = run_fpgrowth(basket, min_support=0.3, max_len=3)
        
        assert len(freq_items) >= 1
        # Item A appears in 2/3 transactions (support=0.67)
        # Item B appears in 1/3 transactions (support=0.33)
        assert len(freq_items) == 2  # Both should be found at min_support=0.3