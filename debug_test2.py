import pandas as pd
from tests.unit.test_switching import _create_sample_switching_data
from src.analytics.switching import compute_switching_matrix

df = _create_sample_switching_data()
matrix = compute_switching_matrix(df, window_days=90, min_transactions=2)

print("Input df shape:", df.shape)
print("Output matrix:")
print(matrix)
print()

if not matrix.empty:
    print("Sum of pct column:", matrix["pct"].sum())
    print("Expected by test: ~1.0")
    print()
    
    # Let's also check what the original implementation would have done
    # by manually computing what we expect
    
    # First, let's see what switching sequences we can derive from the data
    print("For reference, let's compute what we expect:")
    
    # Convert to sequences per customer
    df_sorted = df.sort_values(["customer_id", "date"])
    switches = []
    
    for customer in df_sorted["customer_id"].unique():
        customer_df = df_sorted[df_sorted["customer_id"] == customer]
        customer_df = customer_df.sort_values("date")
        
        # Get sequences of products
        products = customer_df["stockcode"].tolist()
        dates = customer_df["date"].tolist()
        
        # Look for switches (consecutive different products)
        for i in range(len(products) - 1):
            if products[i] != products[i+1]:
                switches.append({
                    'from_product': products[i],
                    'to_product': products[i+1],
                    'customer_id': customer
                })
    
    print(f"Found {len(switches)} switches:")
    for s in switches:
        print(f"  {s['from_product']} -> {s['to_product']} (customer {s['customer_id']})")
    
    # Count switches by from->to
    from collections import defaultdict
    switch_counts = defaultdict(int)
    for s in switches:
        switch_counts[(s['from_product'], s['to_product'])] += 1
    
    print("\nSwitch counts:")
    for (f, t), c in switch_counts.items():
        print(f"  {f} -> {t}: {c}")
    
    # Compute proportions per from_product
    from_prod_totals = defaultdict(int)
    for (f, t), c in switch_counts.items():
        from_prod_totals[f] += c
    
    print("\nExpected proportions:")
    expected_rows = []
    for (f, t), c in switch_counts.items():
        proportion = c / from_prod_totals[f]
        expected_rows.append({
            'from_product': f,
            'to_product': t,
            'count': c,  # raw count
            'pct': proportion  # proportion
        })
        print(f"  {f} -> {t}: count={c}, pct={proportion:.2f}")
    
    expected_df = pd.DataFrame(expected_rows)
    print("\nExpected matrix:")
    print(expected_df)
    print(f"Sum of expected pct column: {expected_df['pct'].sum()}")
