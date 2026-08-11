"""Cached Feature Store.

Single immutable precomputation of the expensive representations shared by
most tabs: product/customer/basket aggregates, a sparse customer x product
matrix, a weekly product panel and the product lookup. Purists: this module
is Streamlit-free; the UI layer owns caching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from src.analytics.data import derive_product_lookup


@dataclass
class FeatureStore:
    """Everything the analytics tabs consume.

    Tables are derived once; consumers must not mutate them.
    """

    df: pd.DataFrame
    product_lookup: pd.DataFrame
    customer_features: pd.DataFrame
    product_features: pd.DataFrame
    basket_features: pd.DataFrame
    weekly_product_panel: pd.DataFrame
    customer_product_binary: sparse.csr_matrix
    customer_product_counts: sparse.csr_matrix
    customers: np.ndarray
    products: np.ndarray
    has_category: bool = False
    _category_map: pd.Series | None = field(default=None, repr=False)

    def category_map(self) -> pd.Series:
        """stockcode -> category (first observed value per product)."""
        if self._category_map is None:
            if self.has_category:
                self._category_map = (
                    self.df.drop_duplicates("stockcode")
                    .set_index("stockcode")["category"]
                )
            else:
                self._category_map = pd.Series(dtype=object)
        return self._category_map


def _iso_week(date_series: pd.Series) -> pd.Series:
    """ISO (year, week) as an integer label, e.g. 2026 week 3 -> 202603."""
    iso = date_series.dt.isocalendar()
    return iso["year"] * 100 + iso["week"]


def build_feature_store(df: pd.DataFrame) -> FeatureStore:
    """Build all shared representations with a limited number of grouped passes.

    Args:
        df: TRANSACTIONS-validated line-item DataFrame.

    Returns:
        FeatureStore containing every table described in the docstring.
    """
    d = df.copy()
    if d.empty:
        return FeatureStore(
            df=d,
            product_lookup=derive_product_lookup(df),
            customer_features=pd.DataFrame(
                columns=["customer_id", "n_orders", "n_line_items", "n_products",
                         "total_revenue", "first_date", "last_date", "active_days",
                         "avg_order_value"]
            ),
            product_features=pd.DataFrame(
                columns=["stockcode", "n_orders", "n_line_items", "units",
                         "total_revenue", "n_customers", "avg_price", "first_date",
                         "last_date"]
            ),
            basket_features=pd.DataFrame(
                columns=["transaction_id", "customer_id", "date", "basket_size",
                         "n_products", "basket_revenue", "iso_week"]
            ),
            weekly_product_panel=pd.DataFrame(
                columns=["stockcode", "iso_week", "units", "revenue", "avg_price",
                         "n_transactions", "n_customers", "active_days"]
            ),
            customer_product_binary=sparse.csr_matrix((0, 0)),
            customer_product_counts=sparse.csr_matrix((0, 0)),
            customers=np.array([], dtype=object),
            products=np.array([], dtype=object),
            has_category="category" in d.columns,
        )

    d["price_times_qty"] = d["price"] * d["quantity"]
    d["iso_week"] = _iso_week(d["date"])
    d["revenue"] = d["price_times_qty"]

    # ---- product-level aggregates (one grouped pass) ----
    product_stats = d.groupby("stockcode").agg(
        n_orders=("transaction_id", "nunique"),
        n_line_items=("transaction_id", "size"),
        units=("quantity", "sum"),
        total_revenue=("price_times_qty", "sum"),
        n_customers=("customer_id", "nunique"),
        avg_price=("price", "mean"),
        first_date=("date", "min"),
        last_date=("date", "max"),
    )
    if "category" in d.columns:
        product_stats["category"] = (
            d.drop_duplicates("stockcode").set_index("stockcode")["category"]
        )
    product_features = product_stats.reset_index()

    # customer-level aggregates (one grouped pass)
    cust = d.groupby("customer_id").agg(
        n_orders=("transaction_id", "nunique"),
        n_line_items=("transaction_id", "size"),
        n_products=("stockcode", "nunique"),
        total_revenue=("price_times_qty", "sum"),
        first_date=("date", "min"),
        last_date=("date", "max"),
    )
    days = cust["last_date"] - cust["first_date"]
    cust["active_days"] = days.dt.days + 1
    avg_order_value = cust["total_revenue"] / cust["n_orders"]
    cust["avg_order_value"] = avg_order_value
    cust.insert(0, "customer_id", cust.index)
    customer_features = cust.reset_index(drop=True)

    # basket-level aggregates (one grouped pass)
    basket = d.groupby("transaction_id").agg(
        customer_id=("customer_id", "first"),
        date=("date", "first"),
        basket_size=("quantity", "sum"),
        n_products=("stockcode", "nunique"),
        basket_revenue=("price_times_qty", "sum"),
        iso_week=("iso_week", "first"),
    )
    basket_features = basket.reset_index()

    # weekly product panel (product x week)
    weekly = (
        d.groupby(["stockcode", "iso_week"], as_index=False)
        .agg(
            units=("quantity", "sum"),
            revenue=("price_times_qty", "sum"),
            avg_price=("price", "mean"),
            n_transactions=("transaction_id", "nunique"),
            n_customers=("customer_id", "nunique"),
            active_days=("date", "nunique"),
        )
    )
    weekly_product_panel = weekly

    # sparse customer x product matrices (counts + binary)
    customers_array = d["customer_id"].unique()
    products_array = d["stockcode"].unique()
    customer_index = {c: i for i, c in enumerate(customers_array)}
    product_index = {p: j for j, p in enumerate(products_array)}
    rows = d["customer_id"].map(customer_index).to_numpy(dtype=np.int64)
    cols = d["stockcode"].map(product_index).to_numpy(dtype=np.int64)
    vals = d["quantity"].to_numpy(dtype=np.float32)
    shape = (len(customers_array), len(products_array))
    counts = sparse.csr_matrix(
        (vals, (rows, cols)), shape=shape, dtype=np.float32
    )
    counts.sum_duplicates()
    counts.sort_indices()
    binary = (counts > 0).astype(np.float32)
    binary.sum_duplicates()

    return FeatureStore(
        df=d,
        product_lookup=derive_product_lookup(df),
        customer_features=customer_features,
        product_features=product_features,
        basket_features=basket_features,
        weekly_product_panel=weekly_product_panel,
        customer_product_binary=binary,
        customer_product_counts=counts,
        customers=customers_array,
        products=products_array,
        has_category="category" in d.columns,
    )
