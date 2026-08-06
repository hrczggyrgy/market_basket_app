import pandas as pd
import numpy as np
from src.analytics.pricing.price_curves import diagnose_price_curves_1d

# Create a dataset where categories have fewer products than n_tiers
df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=10, freq="D"),
    "transaction_id": ["T" + str(i) for i in range(10)],
    "stockcode": ["SKU001", "SKU002", "SKU003"],
    "product": ["Product A", "Product B", "Product C"],
    "customer_id": ["CUST001"] * 10,
    "price": [10.0, 20.0, 15.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
    "quantity": [1, 2, 1, 3, 2, 1, 4, 1, 2, 1],
    "category": ["Cat1", "Cat1", "Cat2"],  # Cat1 has 2 products, Cat2 has 1 product
    "brand": ["Brand1"] * 10,
    "size": ["1L"] * 10
)

print("Testing with categories having fewer than n_tiers products...")
result = diagnose_price_curves_1d(df, n_tiers=3)
print("Result columns:", result.columns.tolist())
print("Result shape:", result.shape)
print("Columns present:", "tier" in result.columns, "tier_label" in result.columns)
print(result)