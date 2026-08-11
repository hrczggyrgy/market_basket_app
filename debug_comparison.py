import pandas as pd
import numpy as np
from src.analytics.switching import compute_switching_matrix

# Recreate the test data
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

print('Input df:')
print(df[['date', 'transaction_id', 'stockcode', 'customer_id']].head(10))
print('...')
print()

print('Input switching_df:')
print(switching_df)
print()

# Call compute_switching_matrix with just df (computes from raw data)
matrix1 = compute_switching_matrix(df)
print('Output from compute_switching_matrix(df):')
print(matrix1)
print()

# Call compute_switching_matrix with switching_df (uses provided matrix)
matrix2 = compute_switching_matrix(df, switching_df=switching_df)
print('Output from compute_switching_matrix(df, switching_df=switching_df):')
print(matrix2)
print()

# Check what the test is checking for matrix2
print('What the test checks for matrix2 (sum of count per from_product):')
if not matrix2.empty:
    for from_prod in ['A', 'B', 'C']:
        if from_prod in matrix2['from_product'].values:
            row_sum = matrix2[matrix2['from_product'] == from_prod]['count'].sum()
            print(f'  {from_prod}: {row_sum}')
        else:
            print(f'  {from_prod}: not in matrix')
