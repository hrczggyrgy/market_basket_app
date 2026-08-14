"""Fixtures for pricing tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_kvi_fixture() -> pd.DataFrame:
    """Build a KVI fixture DataFrame for testing with SKUs matching decision matrix expectations."""
    return pd.DataFrame(
        {
            "stockcode": ["ELASTIC_HI", "INELASTIC", "PRICE_LEVER", "REVIEW", "CONST_SKU", "SHORT_SKU", "WEAK", "HIGH_KVI_LOW_ELAST"],
            "category": ["cat"] * 8,
            "total_revenue": [1000.0, 2000.0, 1500.0, 1200.0, 800.0, 500.0, 600.0, 900.0],
            "basket_penetration": [0.8, 0.6, 0.7, 0.5, 0.3, 0.2, 0.4, 0.6],
            "trip_incidence": [0.3, 0.4, 0.35, 0.25, 0.1, 0.05, 0.15, 0.2],
            "abs_elasticity": [1.8, 0.4, 1.5, 0.7, np.nan, np.nan, np.nan, 0.3],
            "elasticity_status": ["estimated", "estimated", "estimated", "estimated", "insufficient_price_points", "insufficient_observations", "weak", "estimated"],
            "kvi_score": [0.9, 0.8, 0.1, 0.15, 0.05, 0.02, 0.4, 0.7],
        }
    )


def build_kvi_decision_fixture() -> pd.DataFrame:
    """Build a KVI fixture with SKUs matching the decision matrix test expectations."""
    return pd.DataFrame(
        {
            "stockcode": ["ELASTIC_HI", "INELASTIC", "PRICE_LEVER", "REVIEW", "CONST_SKU", "SHORT_SKU"],
            "category": ["cat"] * 6,
            "total_revenue": [1000.0, 2000.0, 1500.0, 1200.0, 800.0, 500.0],
            "basket_penetration": [0.8, 0.6, 0.7, 0.5, 0.3, 0.2],
            "trip_incidence": [0.3, 0.4, 0.35, 0.25, 0.1, 0.05],
            "abs_elasticity": [1.8, 0.4, 1.5, 0.7, np.nan, np.nan],
            "elasticity_status": ["estimated", "estimated", "estimated", "estimated", "insufficient_price_points", "insufficient_observations"],
            "kvi_score": [0.9, 0.8, 0.7, 0.5, 0.3, 0.2],
        }
    )


def build_pricing_df() -> pd.DataFrame:
    """Build a pricing analysis fixture DataFrame for testing."""
    # Create data with price variation within each SKU to enable elasticity estimation
    np.random.seed(42)
    n_days = 50
    n_skus = 10
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    
    rows = []
    for i, date in enumerate(dates):
        for sku_idx in range(n_skus):
            # Each SKU has a base price with some variation
            base_price = 5.0 + sku_idx  # SKU001: ~5, SKU002: ~6, etc.
            # Add more daily variation (±30%) to ensure weekly CV > 0.05
            price = base_price * np.random.uniform(0.7, 1.3)
            rows.append({
                "date": date,
                "transaction_id": f"T{i:06d}_{sku_idx:02d}",
                "stockcode": f"SKU{sku_idx + 1:03d}",
                "product": f"Product {sku_idx + 1}",
                "customer_id": f"CUST{(i * n_skus + sku_idx) % 100:03d}",
                "price": round(price, 2),
                "quantity": np.random.randint(1, 5),
            })
    
    return pd.DataFrame(rows)


def build_true_elasticity_df() -> pd.DataFrame:
    """Build a fixture with known true elasticities for validation tests.
    
    Generates data from log(qty) = alpha + beta * log(price) + noise
    where beta is the known elasticity for each SKU.
    """
    np.random.seed(123)
    n_weeks = 20
    dates = pd.date_range("2024-01-01", periods=n_weeks, freq="W")
    
    # SKUs with known elasticities
    sku_configs = [
        {"stockcode": "ELASTIC_HI", "elasticity": -1.8, "base_price": 10.0, "base_qty": 100},
        {"stockcode": "INELASTIC", "elasticity": -0.4, "base_price": 10.0, "base_qty": 100},
        {"stockcode": "PRICE_LEVER", "elasticity": -1.5, "base_price": 10.0, "base_qty": 100},
        {"stockcode": "REVIEW", "elasticity": -0.7, "base_price": 10.0, "base_qty": 100},
        {"stockcode": "CONST_SKU", "elasticity": 0.0, "base_price": 10.0, "base_qty": 100},  # constant price
        {"stockcode": "SHORT_SKU", "elasticity": -1.0, "base_price": 10.0, "base_qty": 100},  # fewer obs
    ]
    
    rows = []
    for config in sku_configs:
        sku = config["stockcode"]
        true_elast = config["elasticity"]
        base_price = config["base_price"]
        base_qty = config["base_qty"]
        
        # Determine number of weeks for this SKU
        n_sku_weeks = n_weeks if sku != "SHORT_SKU" else 3  # SHORT_SKU has only 3 weeks
        
        for w in range(n_sku_weeks):
            date = dates[w]
            # Generate price with variation
            if sku == "CONST_SKU":
                price = base_price  # Constant price
            else:
                # Log-normal price variation
                price = base_price * np.exp(np.random.normal(0, 0.2))
            
            # Generate quantity from the true model: log(qty) = alpha + beta*log(price) + noise
            alpha = np.log(base_qty) - true_elast * np.log(base_price)
            log_qty = alpha + true_elast * np.log(price) + np.random.normal(0, 0.1)
            qty = max(1, int(np.exp(log_qty)))
            
            rows.append({
                "date": date,
                "transaction_id": f"T{w:06d}_{sku}",
                "stockcode": sku,
                "product": f"Product {sku}",
                "customer_id": f"CUST{w:03d}",
                "price": round(price, 2),
                "quantity": qty,
            })
    
    return pd.DataFrame(rows)
