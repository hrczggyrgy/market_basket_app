"""FeatureRegistry: central registry for feature tables and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FeatureMetadata:
    """Metadata describing a feature table."""
    name: str
    description: str
    schema: tuple[str, ...]
    version: str = "1.0.0"
    tags: tuple[str, ...] = field(default_factory=tuple)

class FeatureRegistry:
    """Registry managing feature table metadata and lifecycle."""
    _features: dict[str, FeatureMetadata] = field(default_factory=dict)

    def register(self, name: str, metadata: FeatureMetadata) -> None:
        """Register a feature table in the registry."""
        self._features[name] = metadata

    def unregister(self, name: str) -> None:
        """Unregister a feature table from the registry."""
        self._features.pop(name, None)

    def get(self, name: str) -> FeatureMetadata | None:
        """Retrieve metadata for a registered feature table."""
        return self._features.get(name)

    def list_features(self) -> list[str]:
        """List all registered feature table names."""
        return sorted(self._features.keys())

    def validate_table(self, name: str, df: pd.DataFrame) -> bool:
        """Validate that a DataFrame matches the registered schema."""
        metadata = self._features.get(name)
        if metadata is None:
            return False
        if len(df.columns) != len(metadata.schema):
            return False
        columns_set = set(df.columns)
        schema_set = set(metadata.schema)
        if not schema_set.issubset(columns_set):
            return False
        return True

    def __contains__(self, name: str) -> bool:
        """Check if a feature table is registered."""
        return name in self._features

    def __len__(self) -> int:
        """Return the number of registered feature tables."""
        return len(self._features)

    def __iter__(self):
        """Iterate over registered feature names."""
        return iter(self._features)

# Pre-register canonical feature tables
_default_registry: FeatureRegistry | None = None

def get_default_registry() -> FeatureRegistry:
    """Get or create the default feature registry instance."""
    global _default_registry
    if _default_registry is None:
        _default_registry = FeatureRegistry()
        _register_canonical_features(_default_registry)
    return _default_registry

def _register_canonical_features(registry: FeatureRegistry) -> None:
    """Register canonical feature tables for market_basket_app."""
    registry.register(
        "dim_product",
        FeatureMetadata(
            name="dim_product",
            description="Product dimension table with key attributes",
            schema=("stockcode", "product", "category", "brand", "total_revenue", "n_orders", "n_customers"),
            tags=("dimension", "product", "static"),
        ),
    )
    registry.register(
        "fact_product_day",
        FeatureMetadata(
            name="fact_product_day",
            description="Daily product fact table with daily aggregates",
            schema=("stockcode", "date", "units", "revenue", "n_transactions", "n_customers"),
            tags=("fact", "product", "daily"),
        ),
    )
    registry.register(
        "fact_product_week",
        FeatureMetadata(
            name="fact_product_week",
            description="Weekly product fact table with weekly aggregates",
            schema=("stockcode", "iso_week", "units", "revenue", "n_transactions", "n_customers"),
            tags=("fact", "product", "weekly"),
        ),
    )
    registry.register(
        "fact_basket",
        FeatureMetadata(
            name="fact_basket",
            description="Basket-level fact table",
            schema=("transaction_id", "customer_id", "date", "basket_size", "n_products", "basket_revenue", "iso_week"),
            tags=("fact", "basket", "transaction"),
        ),
    )
    registry.register(
        "fact_customer",
        FeatureMetadata(
            name="fact_customer",
            description="Customer fact table with aggregated metrics",
            schema=("customer_id", "n_orders", "n_line_items", "n_products", "total_revenue", "first_date", "last_date", "active_days", "avg_order_value"),
            tags=("fact", "customer", "aggregated"),
        ),
    )
    registry.register(
        "fact_customer_product",
        FeatureMetadata(
            name="fact_customer_product",
            description="Customer-product interaction matrix",
            schema=("customer_id", "stockcode", "units", "revenue", "n_transactions"),
            tags=("fact", "customer", "product", "interaction"),
        ),
    )
    registry.register(
        "customer_sequence",
        FeatureMetadata(
            name="customer_sequence",
            description="Customer switching/sequence analysis",
            schema=("customer_id", "sequence_id", "start_date", "end_date", "n_transactions", "n_products", "revenue"),
            tags=("switching", "sequence", "customer"),
        ),
    )
