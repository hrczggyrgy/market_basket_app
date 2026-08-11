import pandas as pd
import numpy as np
from src.analytics.switching import compute_switching_matrix, compute_transition_matrix

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

print('Input switching_df:')
print(switching_df)
print()

# Call compute_switching_matrix
switching_matrix = compute_switching_matrix(pd.DataFrame(), switching_df=switching_df)

print('Output from compute_switching_matrix:')
print(switching_matrix)
print()

# Call compute_transition_matrix
try:
    trans_matrix = compute_transition_matrix(pd.DataFrame(), switching_df=switching_df)
    print('Output from compute_transition_matrix:')
    print(trans_matrix)
    print()
    
    if not trans_matrix.empty:
        print('Row sums:')
        for from_prod in trans_matrix.index:
            row_sum = trans_matrix.loc[from_prod].sum()
            print(f'  {from_prod}: {row_sum}')
except Exception as e:
    print(f'Error in compute_transition_matrix: {e}')
    import traceback
    traceback.print_exc()
