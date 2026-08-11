import pandas as pd
import numpy as np
from src.analytics.transference import compute_substitutable_demand_percentage, compute_demand_transference_matrix
from src.analytics.switching import compute_switching_matrix, compute_transition_matrix
from src.analytics.insights.switching import generate_switching_insights
from src.analytics.schemas import PRICING_INSIGHTS


def test_sdp_denominator_fix():
    """Test that SDP is computed as switching transfer revenue / product revenue (not total revenue)."""
    # Create simple transaction data
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'transaction_id': range(10),
        'stockcode': ['A'] * 5 + ['B'] * 5,
        'product': ['prod'] * 10,
        'customer_id': ['C'] * 10,
        'price': [10.0] * 10,
        'quantity': [1] * 10,
    })
    # Product A: revenue = 5*10*1 = 50
    # Product B: revenue = 5*10*1 = 50
    # Total revenue = 100

    # Create switching matrix where all switches from A go to B (5 switches)
    # and all switches from B go to A (5 switches)
    switching_df = pd.DataFrame({
        'from_product': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
        'to_product':   ['B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A'],
        'count': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    })
    # Compute demand transference
    dt_df = compute_demand_transference_matrix(df, switching_df=switching_df)
    # For product A: observed_switching_transfer_revenue sum = 5 * (switch_rate * revenue_A)
    # switch_rate for A->B = 5/5 = 1.0
    # revenue_A = 50
    # observed_switching_transfer_revenue for each row = 1.0 * 50 = 50
    # sum for A = 5 * 50 = 250
    # Similarly for B: sum = 5 * 50 = 250
    # Product revenue A = 50, B = 50
    # SDP_A = 250 / 50 = 5.0 -> clipped to 1.0
    # SDP_B = 250 / 50 = 5.0 -> clipped to 1.0
    sdp_df = compute_substitutable_demand_percentage(dt_df, df)
    assert not sdp_df.empty
    # SDP should be clipped to 1.0
    assert np.allclose(sdp_df['sdp'], 1.0), f"SDP values: {sdp_df['sdp'].tolist()}"


def test_sdp_zero_when_no_switching():
    """Test that SDP is zero when there is no switching away from a product."""
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=6),
        'transaction_id': range(6),
        'stockcode': ['A'] * 3 + ['B'] * 3,
        'product': ['prod'] * 6,
        'customer_id': ['C'] * 6,
        'price': [10.0] * 6,
        'quantity': [1] * 6,
    })
    # No switching: each customer buys only one product
    switching_df = pd.DataFrame(columns=['from_product', 'to_product', 'count'])
    dt_df = compute_demand_transference_matrix(df, switching_df=switching_df)
    assert dt_df.empty
    sdp_df = compute_substitutable_demand_percentage(dt_df, df)
    assert not sdp_df.empty
    # SDP should be 0 for both products
    assert np.allclose(sdp_df['sdp'], 0.0), f"SDP values: {sdp_df['sdp'].tolist()}"


def test_switching_matrix_row_normalization():
    """Test that switching matrix rows sum to 1 (or 0 for no switches)."""
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=12),
        'transaction_id': range(12),
        'stockcode': ['A'] * 4 + ['B'] * 4 + ['C'] * 4,
        'product': ['prod'] * 12,
        'customer_id': ['C1'] * 4 + ['C2'] * 4 + ['C3'] * 4,
        'price': [10.0] * 12,
        'quantity': [1] * 12,
    })
    # Create switching matrix:
    # From A: 3 switches to B, 1 switch to C
    # From B: 2 switches to A, 2 switches to C
    # From C: 0 switches
    switching_data = []
    # A -> B three times
    for _ in range(3):
        switching_data.append({'from_product': 'A', 'to_product': 'B', 'count': 1})
    # A -> C one time
    switching_data.append({'from_product': 'A', 'to_product': 'C', 'count': 1})
    # B -> A two times
    for _ in range(2):
        switching_data.append({'from_product': 'B', 'to_product': 'A', 'count': 1})
    # B -> C two times
    for _ in range(2):
        switching_data.append({'from_product': 'B', 'to_product': 'C', 'count': 1})
    switching_df = pd.DataFrame(switching_data)
    # Compute switching matrix (should be row-normalized)
    matrix = compute_switching_matrix(df, switching_df=switching_df)
    assert not matrix.empty
    # Check that row sums are 1 (or 0)
    for from_prod in ['A', 'B', 'C']:
        if from_prod in matrix['from_product'].values:
            row_sum = matrix[matrix['from_product'] == from_prod]['count'].sum()
            assert np.isclose(row_sum, 1.0), f"Row sum for {from_prod} is {row_sum}, expected 1.0"
        else:
            # If from_prod not in matrix, it means no switches from that product
            # That's fine for C in our case? Actually, we have no switches from C, so C should not appear in from_product
            pass
    # Specifically, C should not appear as from_product because there are no switches from C
    assert 'C' not in matrix['from_product'].values, "C should not have any switches out"


def test_transition_matrix_normalization():
    """Test that transition matrix rows sum to 1 (including absorbing state)."""
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=12),
        'transaction_id': range(12),
        'stockcode': ['A'] * 4 + ['B'] * 4 + ['C'] * 4,
        'product': ['prod'] * 12,
        'customer_id': ['C1'] * 4 + ['C2'] * 4 + ['C3'] * 4,
        'price': [10.0] * 12,
        'quantity': [1] * 12,
    })
    # Same switching data as above
    switching_data = []
    for _ in range(3):
        switching_data.append({'from_product': 'A', 'to_product': 'B', 'count': 1})
    switching_data.append({'from_product': 'A', 'to_product': 'C', 'count': 1})
    for _ in range(2):
        switching_data.append({'from_product': 'B', 'to_product': 'A', 'count': 1})
    for _ in range(2):
        switching_data.append({'from_product': 'B', 'to_product': 'C', 'count': 1})
    switching_df = pd.DataFrame(switching_data)
    # Compute transition matrix (should be row-normalized with absorbing state)
    trans_matrix = compute_transition_matrix(df, switching_df=switching_df)
    assert not trans_matrix.empty
    # Check that each row sums to 1.0
    for from_prod in trans_matrix.index:
        row_sum = trans_matrix.loc[from_prod].sum()
        assert np.isclose(row_sum, 1.0), f"Transition row sum for {from_prod} is {row_sum}, expected 1.0"


def test_column_renaming():
    """Test that old column names are replaced with new ones."""
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5),
        'transaction_id': range(5),
        'stockcode': ['A', 'B', 'A', 'B', 'A'],
        'product': ['prod'] * 5,
        'customer_id': ['C'] * 5,
        'price': [10.0] * 5,
        'quantity': [1] * 5,
    })
    switching_df = compute_switching_matrix(df)
    # Ensure observed_switching_recovery_proxy is not in switching_df (should not be present)
    assert 'observed_switching_recovery_proxy' not in switching_df.columns
    # Check transference functions
    dt_df = compute_demand_transference_matrix(df, switching_df=switching_df)
    # Ensure observed_switching_recovery_proxy is not in dt_df
    assert 'observed_switching_recovery_proxy' not in dt_df.columns
    # Ensure observed_switching_transfer_revenue is in dt_df
    assert 'observed_switching_transfer_revenue' in dt_df.columns
    sdp_df = compute_substitutable_demand_percentage(dt_df, df)
    # SDP column name unchanged
    assert 'sdp' in sdp_df.columns


def test_insights_evidence_fields():
    """Test that switching insights include evidence level and related fields."""
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=20),
        'transaction_id': range(20),
        'stockcode': (['A'] * 10) + (['B'] * 10),
        'product': ['prod'] * 20,
        'customer_id': (['C1'] * 10) + (['C2'] * 10),
        'price': [10.0] * 20,
        'quantity': [1] * 20,
    })
    # Create switching data with enough transitions to get evidence level >= 3
    switching_data = []
    # Create 25 transition pairs (5 from A to B repeated 5 times, etc.)
    for i in range(5):
        for _ in range(5):
            switching_data.append({'from_product': 'A', 'to_product': 'B', 'count': 1})
            switching_data.append({'from_product': 'B', 'to_product': 'A', 'count': 1})
    switching_df = pd.DataFrame(switching_data)
    # Compute required dataframes
    dt_df = compute_demand_transference_matrix(df, switching_df=switching_df)
    sdp_df = compute_substitutable_demand_percentage(dt_df, df)
    # Generate insights
    insights_df = generate_switching_insights(dt_df, sdp_df, delist_impact_df=None)
    assert not insights_df.empty
    # Check that the new columns exist
    assert 'evidence_level' in insights_df.columns
    assert 'n_transition_pairs' in insights_df.columns
    assert 'n_unique_products' in insights_df.columns
    assert 'confidence_gate' in insights_df.columns
    # Check that evidence_level is between 1 and 5
    assert insights_df['evidence_level'].between(1, 5).all()
    # Check that n_transition_pairs and n_unique_products are non-negative
    assert (insights_df['n_transition_pairs'] >= 0).all()
    assert (insights_df['n_unique_products'] >= 0).all()
    # Check that confidence_gate is boolean
    assert insights_df['confidence_gate'].isin([True, False]).all()
    # Check that the evidence string contains the evidence level
    assert insights_df['evidence'].str.contains('Evidence Level').all()


if __name__ == "__main__":
    test_sdp_denominator_fix()
    test_sdp_zero_when_no_switching()
    test_switching_matrix_row_normalization()
    test_transition_matrix_normalization()
    test_column_renaming()
    test_insights_evidence_fields()
    print("All tests passed!")