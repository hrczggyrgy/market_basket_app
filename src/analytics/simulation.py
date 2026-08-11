"""Config-driven transaction simulator.

Refactors the fixed kernel of ``sample_data.generate_transactions`` into a
single ``SimulationConfig`` object that describes the world to simulate.
The generator is split into module stages (catalog -> promotions -> customers
-> calendar -> baskets -> transactions) with one entry point,
``generate_sample_transactions(config)``. Output and seeding are deterministic
given the config, and the 13-column schema from ``sample_data`` is preserved.

Predefined ``SCENARIOS`` (standard, promo_heavy, seasonal, high_switching)
enable side-by-side regime comparisons from the UI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analytics.sample_data import (
    AFFINITY_THEMES,
    CATEGORIES,
    CATEGORY_PRICE_PARAMS,
    SEGMENTS,
    SUBSTITUTION_GROUPS,
    _customer_lifecycle,
)

SCHEMA_COLUMNS = (
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
)


@dataclass(frozen=True)
class SimulationConfig:
    """Everything that drives a single simulation run.

    All fields are optional; the frozen dataclass gives a lightweight,
    hashable configuration that can be built from scenario presets.
    """

    name: str = "standard"
    label: str = "Standard"
    description: str = "Balanced baseline retail scenario."
    n_customers: int = 1000
    n_products: int = 200
    n_days: int = 365
    seed: int = 42
    end_date: str = "2024-12-31"

    # Calendar analogues
    seasonal_strength: float = 1.0   # scales seasonal amplitude around 1.0
    weekend_strength: float = 1.0    # scales Sat/Sun weekend boost around 1.0
    holiday_strength: float = 1.0    # scales holiday demand bump around 1.0

    # Segments
    purchase_scale: float = 1.0      # multiplies every segment purchase_rate
    basket_scale: float = 1.0        # multiplies every segment avg_basket_size
    churn_scale: float = 1.0         # multiplies every segment churn_prob

    # Basket building
    theme_probability: float = 0.15  # chance a basket starts from an affinity theme
    affinity_boost: float = 3.0      # popularity^boost weighting within a category
    substitution_strength: float = 0.0  # chance to pull in a substitute-category SKU

    # Promotions
    promo_penetration: float = 0.25  # share of SKUs that are promo-capable
    promo_window_scale: float = 1.0  # multiplies # promo windows per SKU
    promo_discount_scale: float = 1.0  # multiplies discount depth (capped)

    # Returns
    return_rate: float = 0.0         # 0-1 fraction of line items refunded (+7d)


# --------------------------------------------------------------------------- #
# Scenario presets
# ------------------------------------------------------------------------- #

SCENARIO_DEFS: dict[str, dict] = {
    "standard": {
        "name": "standard",
        "label": "Standard",
        "description": "Balanced baseline retail scenario.",
        "n_customers": 1000,
        "n_products": 500,
        "n_days": 730,
    },
    "promo_heavy": {
        "name": "promo_heavy",
        "label": "Promo-heavy",
        "description": "Frequent, deep discounts on many SKUs; price-sensitive customers.",
        "n_customers": 1000,
        "n_products": 500,
        "n_days": 730,
        "promo_penetration": 0.4,
        "promo_window_scale": 1.5,
        "promo_discount_scale": 1.4,
    },
    "seasonal": {
        "name": "seasonal",
        "label": "Seasonal",
        "description": "Strong seasonal and weekend swings with holiday spikes.",
        "n_customers": 1000,
        "n_products": 500,
        "n_days": 730,
        "seasonal_strength": 2.0,
        "weekend_strength": 2.0,
        "holiday_strength": 2.0,
    },
    "high_switching": {
        "name": "high_switching",
        "label": "High switching",
        "description": "Low loyalty, high churn, substitution-heavy baskets.",
        "n_customers": 1000,
        "n_products": 500,
        "n_days": 730,
        "purchase_scale": 1.2,
        "basket_scale": 0.9,
        "churn_scale": 1.8,
        "substitution_strength": 0.6,
        "promo_discount_scale": 1.2,
    },
}

SCENARIOS: dict[str, SimulationConfig] = {
    key: SimulationConfig(**params) for key, params in SCENARIO_DEFS.items()
}


def config_for(name: str) -> SimulationConfig:
    """Return a scenario config by key (falls back to standard)."""
    return SCENARIOS.get(name, SCENARIOS["standard"])


# --------------------------------------------------------------------------- #
# Catalog
# ------------------------------------------------------------------------- #

def _assign_categories(rng: np.random.Generator, n_products: int) -> np.ndarray:
    cat_probs = np.array([0.15, 0.20, 0.15, 0.12, 0.10, 0.12, 0.08, 0.08])
    return rng.choice(CATEGORIES, size=n_products, p=cat_probs)


def generate_catalog(config: SimulationConfig, seed_offset: int = 1000) -> pd.DataFrame:
    """Build the product catalog (prices, Pareto popularity, promo capability)."""
    rng = np.random.default_rng(config.seed + seed_offset)
    n = config.n_products
    cats = _assign_categories(rng, n)

    price = np.zeros(n)
    for i, cat in enumerate(cats):
        mu, sigma = CATEGORY_PRICE_PARAMS[cat]
        price[i] = round(rng.lognormal(mu, sigma), 2)
    price = np.clip(price, 0.99, 99.99)

    ranks = np.arange(1, n + 1)
    popularity = 1.0 / (ranks ** 1.15)
    popularity = popularity / popularity.sum()

    cost = (price * rng.uniform(0.55, 0.70, n)).round(2)
    strikes = rng.random(n) < config.promo_penetration

    cat_to_sub: dict[str, int] = {}
    for gi, group in enumerate(SUBSTITUTION_GROUPS):
        for cat in group:
            cat_to_sub[cat] = gi

    catalog = pd.DataFrame(
        {
            "stockcode": [f"SKU{i:04d}" for i in range(n)],
            "product": [f"{cats[i]} {i+1:03d}" for i in range(n)],
            "category": cats,
            "brand": [f"B{(i % 5) + 1}" for i in range(n)],
            "size": rng.choice(["S", "M", "L", "XL"], size=n, p=[0.2, 0.45, 0.25, 0.1]),
            "flavor": [f"F{i % 5 + 1}" for i in range(n)],
            "price": price,
            "cost": cost,
            "popularity": popularity,
            "promo_capable": strikes,
        }
    )
    catalog["sub_group"] = catalog["category"].map(cat_to_sub).fillna(-1).astype(int)
    return catalog


# --------------------------------------------------------------------------- #
# Promotions
# ------------------------------------------------------------------------- #

_PROMO_TYPES = ("discount", "bogo", "multibuy", "clearance")
_PROMO_PROBS = (0.55, 0.20, 0.15, 0.10)


def _promo_windows(catalog: pd.DataFrame, config: SimulationConfig, seed: int) -> dict[str, list[dict]]:
    rng = np.random.default_rng(seed)
    windows: dict[str, list[dict]] = {}
    for sku in catalog.loc[catalog["promo_capable"], "stockcode"]:
        n_win = max(1, int(rng.integers(1, 5) * config.promo_window_scale))
        windows[sku] = []
        for _ in range(n_win):
            ptype = rng.choice(_PROMO_TYPES, p=_PROMO_PROBS)
            duration = int(rng.integers(5, 21))
            max_start = max(5, config.n_days - duration - 5)
            if max_start <= 5:
                continue
            start = int(rng.integers(5, max_start))
            end = min(start + duration, config.n_days - 1)

            base_discount = rng.uniform(0.15, 0.25)
            discount = min(0.60, base_discount * config.promo_discount_scale)
            if ptype == "discount":
                params = {"discount": round(discount, 2)}
            elif ptype == "bogo":
                params = {"bogo_qty": 1}
            elif ptype == "multibuy":
                params = {"min_qty": int(rng.integers(2, 4)), "discount": round(min(0.55, discount + 0.05), 2)}
            else:  # clearance
                params = {"discount": round(min(0.75, rng.uniform(0.40, 0.55) * config.promo_discount_scale), 2)}
            windows[sku].append({"start": start, "end": end, "type": ptype, "params": params})
    return windows


# --------------------------------------------------------------------------- #
# Calendar
# ------------------------------------------------------------------------- #

def _seasonal_demand(day_of_year: int, category: str, strength: float = 1.0, year: int = 2024) -> float:
    """Return seasonal multiplier for category on given day (1-366 for leap years).

    Enhanced with leap year support and holiday effects.
    """
    # Handle leap years
    days_in_year = 366 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 365

    t = 2 * np.pi * day_of_year / days_in_year
    seasonal = {
        "Coffee": 1.0 + 0.15 * np.cos(t - np.pi / 2),
        "Beverages": 1.0 + 0.20 * np.sin(t),
        "Bakery": 1.0 + 0.10 * np.cos(t - np.pi),
        "Snacks": 1.0 + 0.10 * np.sin(t + np.pi / 4),
        "Dairy": 1.0 + 0.05 * np.cos(t),
        "Household": 1.0 + 0.08 * np.cos(t - np.pi),
        "Personal Care": 1.0 + 0.07 * np.sin(t),
        "Pet": 1.0,
    }
    base = seasonal.get(category, 1.0)

    # Add holiday effects
    holiday_boost = 1.0
    if (day_of_year >= 359 and day_of_year <= days_in_year) or (day_of_year <= 2):
        holiday_boost = 1.2  # Christmas/New Year
    elif 332 <= day_of_year <= 333:
        holiday_boost = 1.3  # Black Friday

    return 1.0 + strength * (base * holiday_boost - 1.0)


def _weekly_demand(day_of_week: int, category: str, strength: float = 1.0) -> float:
    weekend_boost = {
        "Coffee": 1.25,
        "Bakery": 1.35,
        "Snacks": 1.30,
        "Beverages": 1.20,
        "Dairy": 1.10,
    }
    if day_of_week >= 5:
        base = weekend_boost.get(category, 1.05)
        return 1.0 + strength * (base - 1.0)
    return 1.0


def _holiday_multiplier(month: int, day: int, strength: float = 1.0) -> float:
    """US retail holiday bumps: Thanksgiving week and pre-Christmas."""
    if (month == 11 and 22 <= day <= 30) or (month == 12 and day >= 20):
        return 1.0 + strength * 0.20
    return 1.0


# --------------------------------------------------------------------------- #
# Customers
# ------------------------------------------------------------------------- #

def generate_customers(config: SimulationConfig, seed_offset: int = 3000) -> pd.DataFrame:
    """Assign customer_id -> (segment, acquisition_day) across the horizon."""
    rng = np.random.default_rng(config.seed + seed_offset)
    seg_names = list(SEGMENTS.keys())
    seg_weights = np.array([SEGMENTS[s]["weight"] for s in seg_names])
    seg_weights = seg_weights / seg_weights.sum()

    acquisition = np.linspace(0, config.n_customers, config.n_days)
    acquisition = np.diff(np.concatenate([[0], acquisition])).astype(int)
    acquisition = np.maximum(acquisition, 1)
    diff = config.n_customers - acquisition.sum()
    if diff > 0:
        acquisition[:diff] += 1

    rows = []
    cust_idx = 0
    for day_idx, n_new in enumerate(acquisition):
        for _ in range(n_new):
            if cust_idx >= config.n_customers:
                break
            seg = rng.choice(seg_names, p=seg_weights)
            rows.append(
                {
                    "customer_id": f"CUST{cust_idx:06d}",
                    "segment": seg,
                    "acquisition_day": day_idx,
                }
            )
            cust_idx += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Basket building
# ------------------------------------------------------------------------- #

def _pick_basket_products(
    catalog: pd.DataFrame,
    picked: list[str],
    basket_size: int,
    rng: np.random.Generator,
    affinity_boost: float = 3.0,
    substitution_strength: float = 0.0,
) -> list[str]:
    """Fill a basket using popularity + category affinity + substitution.

    Pure-numpy implementation over precomputed index arrays; avoids pandas
    boolean indexing in the hot loop.
    """
    products = catalog["product"].to_numpy()
    categories = catalog["category"].to_numpy()
    sub_groups = catalog["sub_group"].to_numpy()
    weights = catalog["popularity"].to_numpy()

    product_index = {p: i for i, p in enumerate(products)}
    picked_set = {product_index[p] for p in picked}
    available = np.ones(len(products), dtype=bool)
    for i in picked_set:
        available[i] = False

    cat_cache: dict[str, np.ndarray] = {}
    sub_cache: dict[int, np.ndarray] = {}

    def _candidates(idx: int) -> int | None:
        w = weights[idx] ** affinity_boost
        total = w.sum()
        if total <= 0:
            return None
        return int(rng.choice(idx, p=w / total))

    while len(picked_set) < basket_size and available.any():
        picked_cats: set[str] = set()
        picked_sub: set[int] = set()
        for i in picked_set:
            picked_cats.add(categories[i])
            picked_sub.add(sub_groups[i])

        affinity_mask = np.zeros(len(products), dtype=bool)
        for c in picked_cats:
            if c not in cat_cache:
                cat_cache[c] = categories == c
            affinity_mask |= cat_cache[c]
        affinity_mask &= available

        if rng.random() < 0.4:
            if substitution_strength > 0 and rng.random() < substitution_strength:
                sub_mask = np.zeros(len(products), dtype=bool)
                for s in picked_sub:
                    if s not in sub_cache:
                        sub_cache[s] = sub_groups == s
                    sub_mask |= sub_cache[s]
                sub_mask &= available
                sub_mask &= ~affinity_mask
                if sub_mask.any():
                    chosen = _candidates(np.nonzero(sub_mask)[0])
                    if chosen is not None:
                        picked_set.add(chosen)
                        available[chosen] = False
                        continue
            else:
                if affinity_mask.any():
                    chosen = _candidates(np.nonzero(affinity_mask)[0])
                    if chosen is not None:
                        picked_set.add(chosen)
                        available[chosen] = False
                        continue

        w = weights.copy()
        w[~available] = 0
        if w.sum() <= 0:
            break
        chosen = int(rng.choice(len(products), p=w / w.sum()))
        picked_set.add(chosen)
        available[chosen] = False

    return [products[i] for i in picked_set]


# --------------------------------------------------------------------------- #
# Main generator
# ------------------------------------------------------------------------- #

def generate_sample_transactions(config: SimulationConfig) -> pd.DataFrame:
    """Generate the full transaction set from a config. Schema fixed to 13 cols."""
    rng = np.random.default_rng(config.seed)
    catalog = generate_catalog(config)
    promos = _promo_windows(catalog, config, config.seed + 2000)
    customers = generate_customers(config)

    active_promo_days: dict[str, dict[int, dict]] = {}
    for sku, wins in promos.items():
        active_promo_days[sku] = {}
        for w in wins:
            for d in range(w["start"], w["end"] + 1):
                active_promo_days[sku][d] = w

    product_by_name = catalog.set_index("product")
    prod_rows: dict[str, tuple] = {}
    for prod_name, prod in product_by_name.iterrows():
        prod_rows[prod_name] = (
            prod["stockcode"],
            prod["category"],
            prod["brand"],
            prod["size"],
            prod["flavor"],
            float(prod["price"]),
        )
    days = pd.date_range(end=pd.Timestamp(config.end_date), periods=config.n_days, freq="D")

    rows: list[tuple] = []
    txn_counter = 0
    customer_states: dict[str, dict] = {
        r["customer_id"]: {
            "segment": r["segment"],
            "acquisition_day": int(r["acquisition_day"]),
            "last_purchase_day": -1,
            "purchase_count": 0,
            "total_spent": 0.0,
        }
        for _, r in customers.iterrows()
    }
    # tracks whether a customer is still active
    active: set[str] = set(customer_states)

    for day_idx in range(config.n_days):
        date = days[day_idx]
        dow = date.dayofweek
        doy = date.dayofyear
        holiday = _holiday_multiplier(date.month, date.day, config.holiday_strength)

        for customer_id in list(active):
            state = customer_states[customer_id]
            segment = state["segment"]
            seg_params = SEGMENTS[segment]

            days_since_last = day_idx - state["last_purchase_day"] if state["last_purchase_day"] >= 0 else 999
            if rng.random() < seg_params["churn_prob"] * max(1, days_since_last / 30) * config.churn_scale:
                active.discard(customer_id)
                continue

            base_rate = seg_params["purchase_rate"] * config.purchase_scale
            lifecycle = _customer_lifecycle(day_idx - state["acquisition_day"], segment)
            purchase_prob = base_rate * lifecycle["purchase_mult"] * holiday
            if rng.random() > purchase_prob:
                continue

            txn_counter += 1
            txn_id = f"TXN{txn_counter:08d}"

            base_basket = seg_params["avg_basket_size"]
            basket_size = int(rng.poisson(max(1.5, base_basket * config.basket_scale)))
            basket_size = max(1, min(basket_size, 12))

            picked: list[str] = []
            if rng.random() < config.theme_probability and AFFINITY_THEMES:
                theme_name = rng.choice(list(AFFINITY_THEMES.keys()))
                for cat in AFFINITY_THEMES[theme_name]:
                    cat_products = catalog[catalog["category"] == cat]["product"].tolist()
                    if cat_products and len(picked) < basket_size:
                        prod = rng.choice(cat_products)
                        if prod not in picked:
                            picked.append(prod)

            picked = _pick_basket_products(
                catalog,
                picked,
                basket_size,
                rng,
                affinity_boost=config.affinity_boost,
                substitution_strength=config.substitution_strength,
            )
            state["last_purchase_day"] = day_idx
            state["purchase_count"] += 1

            for product_name in picked:
                sku, cat, brand, size, flavor, base_price = prod_rows[product_name]
                on_promo = False
                promo_type = "none"
                price = base_price

                if sku in active_promo_days and day_idx in active_promo_days[sku]:
                    promo = active_promo_days[sku][day_idx]
                    on_promo = True
                    promo_type = promo["type"]
                    params = promo["params"]
                    if promo_type in ("discount", "clearance"):
                        price = round(price * (1 - params["discount"]), 2)

                base_qty = int(rng.choice([1, 2, 3, 4, 5], p=[0.55, 0.25, 0.12, 0.05, 0.03]))
                qty = base_qty
                if promo_type == "bogo" and base_qty >= 1:
                    qty = base_qty + params["bogo_qty"]
                elif promo_type == "multibuy" and base_qty >= params["min_qty"]:
                    price = round(price * (1 - params["discount"]), 2)

                if _weekly_demand(dow, cat, config.weekend_strength) > 1.0 and rng.random() < 0.3:
                    qty += 1
                if _seasonal_demand(doy, cat, config.seasonal_strength, year=2024) > 1.0 and rng.random() < 0.2:
                    qty += 1

                cost = round(price * 0.6, 2)
                rows.append(
                    (
                        date,
                        txn_id,
                        str(sku),
                        product_name,
                        customer_id,
                        price,
                        qty,
                        cat,
                        brand,
                        size,
                        flavor,
                        on_promo,
                        cost,
                    )
                )

    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "transaction_id"]).reset_index(drop=True)
    return _apply_returns(df, config)


def _apply_returns(df: pd.DataFrame, config: SimulationConfig) -> pd.DataFrame:
    """Emit negative line items for a share of purchases, lagged ~7 days later.

    Refund rows reuse the original transaction but with a dated record 7 days
    after purchase and negative price/quantity so the data-quality layer
    detects them as returns (report-only, excluded from analysis).
    """
    if config.return_rate <= 0 or df.empty:
        return df
    rng = np.random.default_rng(config.seed + 9000)
    mask = rng.random(len(df)) < config.return_rate
    ret = df.loc[mask].copy()
    if ret.empty:
        return df
    ret["date"] = ret["date"] + pd.Timedelta(days=7)
    ret["price"] = -ret["price"].abs()
    ret["quantity"] = -ret["quantity"].abs()
    # returns land in their own transaction id to avoid distorting basket metrics
    ret["txn_id_suffix"] = range(len(ret))
    ret["transaction_id"] = ret["transaction_id"] + "-R" + ret["txn_id_suffix"].astype(str)
    ret = ret.drop(columns=["txn_id_suffix"])
    return pd.concat([df, ret], ignore_index=True).sort_values(["date", "transaction_id"]).reset_index(drop=True)


def calibration_report(df: pd.DataFrame, config: SimulationConfig) -> pd.DataFrame:
    """Compare simulated output against config targets and sanity bounds.

    Returns a metric/actual/target-style DataFrame used to verify a generated
    dataset looks like the world described by ``config``.
    """
    metrics: list[dict[str, float | str]] = []

    def add(name: str, value: float, target: str | float) -> None:
        metrics.append({"metric": name, "value": round(float(value), 4), "target": str(target)})

    if not df.empty:
        add("rows", len(df), "> 0")
        add("customers", df["customer_id"].nunique(), int(config.n_customers * 0.5))
        add("products", df["stockcode"].nunique(), int(config.n_products * 0.5))
        add("promo_share", df["promo_flag"].mean(), f"~{config.promo_penetration}")
        add("max_qty", df["quantity"].max(), 1.0)
        add("min_price", df["price"].min(), "> 0")
        add(
            "avg_basket_size",
            df.groupby("transaction_id")["quantity"].sum().mean(),
            "> 1.0",
        )
        add("avg_units_per_customer", df.groupby("customer_id")["quantity"].sum().mean(), "> 1.0")
        add("revenue_pareto_top20",
            df.assign(revenue=df["price"] * df["quantity"])
            .groupby("stockcode")["revenue"].sum()
            .nlargest(max(1, int(df["stockcode"].nunique() * 0.2)))
            .sum()
            / (df["price"] * df["quantity"]).sum(),
            "~0.8")

    return pd.DataFrame(metrics, columns=["metric", "value", "target"])


def generate_sample_fixture(config: SimulationConfig, path: str) -> pd.DataFrame:
    """Generate, write to `path` as CSV, and return the DataFrame."""
    df = generate_sample_transactions(config)
    df.to_csv(path, index=False)
    return df
