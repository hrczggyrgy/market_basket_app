"""FeatureStore: canonical fact tables and feature registry for market_basket_app."""

from __future__ import annotations

from . import basket, customer, product, registry, switching

# Key exports for convenience
FeatureRegistry = registry.FeatureRegistry
build_dim_product = product.build_dim_product
build_fact_product_day = product.build_fact_product_day
build_fact_product_week = product.build_fact_product_week
build_fact_basket = basket.build_fact_basket
build_fact_customer = customer.build_fact_customer
build_fact_customer_product = customer.build_fact_customer_product
build_customer_sequence = switching.build_customer_sequence

__all__ = [
    "FeatureRegistry",
    "build_dim_product",
    "build_fact_product_day",
    "build_fact_product_week",
    "build_fact_basket",
    "build_fact_customer",
    "build_fact_customer_product",
    "build_customer_sequence",
]
