import pandas as pd
import numpy as np
from src.analytics.switching import compute_switching_matrix

# Recreate the test data
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

print("Input switching_df:")
print(switching_df)
print()

# Call the function
matrix = compute_switching_matrix(pd.DataFrame(), switching_df=switching_df)

print("Output matrix:")
print(matrix)
print()

if not matrix.empty:
    print("Grouped by from_product:")
    for from_prod in matrix['from_product'].unique():
        subset = matrix[matrix['from_product'] == from_prod]
        print(f"  {from_prod}:")
        print(subset[['to_product', 'count', 'pct']])
        print(f"  Sum of counts: {subset['count'].sum()}")
        print(f"  Sum of pct: {subset['pct'].sum()}")
        print()
