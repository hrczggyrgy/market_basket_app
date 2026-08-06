"""Seeded synthetic transaction generator.

Used to produce the checked-in sample fixture and to build ground-truth
datasets for the validation benchmarks. The generator is deterministic and
produces realistic structure: customer segments with different purchase
rates, category affinity themes, and promo price windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = (
    "Coffee",
    "Snacks",
    "Beverages",
    "Bakery",
    "Dairy",
    "Household",
    "Personal Care",
    "Pet",
)

THEMES = (
    ("Coffee 01", "Dairy 01", "Bakery 01"),
    ("Coffee 02", "Dairy 02", "Snacks 01"),
    ("Beverages 01", "Snacks 02", "Snacks 03"),
    ("Beverages 02", "Snacks 04"),
    ("Bakery 02", "Dairy 03", "Coffee 03"),
    ("Household 01", "Household 02"),
    ("Personal Care 01", "Personal Care 02"),
    ("Pet 01", "Pet 02", "Household 03"),
    ("Coffee 04", "Bakery 03", "Dairy 04"),
    ("Beverages 03", "Snacks 05", "Snacks 06"),
)

SEGMENT_LAMBDA = {"champion": (18, 0.10), "regular": (8, 0.20), "occasional": (3, 0.30)}


def _product_catalog(n_products: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    cats = [CATEGORIES[i % len(CATEGORIES)] for i in range(n_products)]
    brands = [f"B{(i % 3) + 1}" for i in range(n_products)]
    sizes = [rng.choice(["S", "M", "L"], p=[0.3, 0.5, 0.2]) for _ in range(n_products)]
    flavors = [f"F{i % 4 + 1}" for i in range(n_products)]
    price = np.exp(rng.normal(2.0, 0.55, n_products)).round(2)
    popularity = 1.0 / (np.arange(n_products) + 1)
    catalog = pd.DataFrame(
        {
            "stockcode": [f"SKU{i:03d}" for i in range(n_products)],
            "product": [f"{cats[i]} {(i // len(CATEGORIES)) + 1:02d}" for i in range(n_products)],
            "category": cats,
            "brand": brands,
            "size": sizes,
            "flavor": flavors,
            "price": price,
            "cost": (price * 0.6).round(2),
            "popularity": popularity / popularity.sum(),
        }
    )
    promo_mask = rng.random(n_products) < 0.2
    catalog["promo_capable"] = promo_mask
    return catalog


def _promo_windows(catalog: pd.DataFrame, n_days: int, seed: int) -> dict[str, list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    windows: dict[str, list[tuple[int, int]]] = {}
    for sku in catalog.loc[catalog["promo_capable"], "stockcode"]:
        n_win = int(rng.integers(1, 4))
        starts = rng.integers(5, max(6, n_days - 25), size=n_win)
        windows[sku] = [(int(s), int(min(s + rng.integers(7, 12), n_days))) for s in starts]
    return windows


def generate_transactions(
    n_customers: int = 300,
    n_products: int = 60,
    n_days: int = 180,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic transaction DataFrame with canonical schema.

    Returns a DataFrame with columns: date, transaction_id, stockcode, product,
    customer_id, price, quantity, category, brand, size, flavor, promo_flag, cost.
    """
    rng = np.random.default_rng(seed)
    catalog = _product_catalog(n_products)
    promos = _promo_windows(catalog, n_days, seed)
    theme_products = {p for theme in THEMES for p in theme if p in set(catalog["product"])}

    active_days: dict[str, set[int]] = {
        sku: {d for s, e in wins for d in range(s, e + 1)} for sku, wins in promos.items()
    }
    base_weights = catalog["popularity"].to_numpy().copy()
    sku_by_name = catalog.set_index("product")["stockcode"].to_dict()
    product_by_name = catalog.set_index("product")
    days = pd.date_range(end=pd.Timestamp("2024-12-31"), periods=n_days, freq="D")
    segments = list(SEGMENT_LAMBDA.keys())
    rows: list[tuple] = []
    txn_counter = 0

    for c in range(n_customers):
        customer_id = f"CUST{c:04d}"
        segment = segments[c % len(segments)]
        n_purchases = int(rng.poisson(SEGMENT_LAMBDA[segment][0]))
        purchase_days = rng.integers(0, n_days, size=n_purchases)
        for p_day in purchase_days:
            txn_counter += 1
            txn_id = f"TXN{txn_counter:06d}"
            date = days[int(p_day)]
            basket_size = int(rng.integers(2, 7))
            picked: list[str] = []
            if rng.random() < 0.25 and theme_products:
                theme = THEMES[int(rng.integers(0, len(THEMES)))]
                available = [p for p in theme if p in theme_products]
                if available:
                    picked.extend(available)
            weights = base_weights.copy()
            for sku, act in active_days.items():
                if int(p_day) in act:
                    weights[catalog.index[catalog["stockcode"] == sku][0]] *= 3.0
            while len(picked) < basket_size:
                product_name = rng.choice(catalog["product"], p=weights / weights.sum())
                if product_name not in picked:
                    picked.append(product_name)
            for product_name in picked:
                prod = product_by_name.loc[product_name]
                on_promo = False
                price = float(prod["price"])
                for start, end in promos.get(prod["stockcode"], []):
                    if start <= int(p_day) <= end:
                        on_promo = True
                        price = round(price * 0.7, 2)
                        break
                rows.append(
                    (
                        date,
                        txn_id,
                        str(prod["stockcode"]),
                        product_name,
                        customer_id,
                        price,
                        int(rng.integers(1, 4)),
                        prod["category"],
                        prod["brand"],
                        prod["size"],
                        prod["flavor"],
                        bool(on_promo),
                        round(price * 0.6, 2),
                    )
                )

    df = pd.DataFrame(
        rows,
        columns=[
            "date",
            "transaction_id",
            "stockcode",
            "product",
            "customer_id",
            "price",
            "quantity",
            "category",
            "brand",
            "size",
            "flavor",
            "promo_flag",
            "cost",
        ],
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "transaction_id"]).reset_index(drop=True)


def write_sample_fixture(
    path: str, *, n_customers: int = 300, n_products: int = 60, n_days: int = 180, seed: int = 42
) -> pd.DataFrame:
    """Generate the checked-in sample fixture and write it to `path`."""
    df = generate_transactions(
        n_customers=n_customers, n_products=n_products, n_days=n_days, seed=seed
    )
    df.to_csv(path, index=False)
    return df
