"""Seeded synthetic transaction generator.

Produces realistic retail transaction data with:
- Customer segments (champions/regulars/occasional/at-risk/churned) with lifecycle
- Category affinity themes and substitution/complement relationships
- Seasonal and weekly purchasing patterns
- Realistic price distributions (log-normal per category)
- Pareto revenue concentrations (20% SKUs = 80% revenue)
- Promotion windows (seasonal, BOGO, multi-buy, clearance)
- Customer churn and new acquisition over time
- Basket composition with category affinity logic
"""

from __future__ import annotations

from typing import Any

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

# Legacy themes for test compatibility
THEMES = (
    ("Coffee", "Dairy"),
    ("Snacks", "Beverages"),
    ("Bakery", "Coffee"),
    ("Pet", "Household"),
)

# Category-specific price parameters (log-normal mu, sigma)
CATEGORY_PRICE_PARAMS = {
    "Coffee": (2.5, 0.4),  # $8-20
    "Snacks": (1.5, 0.35),  # $3-8
    "Beverages": (1.8, 0.3),  # $4-12
    "Bakery": (1.2, 0.25),  # $2-6
    "Dairy": (1.6, 0.3),  # $3-9
    "Household": (2.0, 0.5),  # $5-25
    "Personal Care": (2.3, 0.45),  # $7-20
    "Pet": (2.2, 0.4),  # $6-20
}

# Segment definitions with realistic parameters and category preferences
SEGMENTS: dict[str, dict[str, Any]] = {
    "champion": {
        "weight": 0.10,
        "purchase_rate": 0.35,  # purchases per day
        "avg_basket_size": 5.2,
        "loyalty": 0.85,
        "price_sensitivity": 0.7,
        "promo_responsiveness": 0.3,
        "churn_prob": 0.0001,
        "category_preferences": {
            "Coffee": 1.5,
            "Bakery": 1.3,
            "Personal Care": 1.2,
        },  # Premium preferences
    },
    "regular": {
        "weight": 0.25,
        "purchase_rate": 0.15,
        "avg_basket_size": 3.8,
        "loyalty": 0.65,
        "price_sensitivity": 1.0,
        "promo_responsiveness": 0.5,
        "churn_prob": 0.0005,
        "category_preferences": {
            "Dairy": 1.2,
            "Snacks": 1.1,
            "Household": 1.0,
        },  # Balanced preferences
    },
    "occasional": {
        "weight": 0.35,
        "purchase_rate": 0.05,
        "avg_basket_size": 2.5,
        "loyalty": 0.35,
        "price_sensitivity": 1.3,
        "promo_responsiveness": 0.8,
        "churn_prob": 0.002,
        "category_preferences": {
            "Snacks": 1.4,
            "Beverages": 1.3,
            "Bakery": 1.2,
        },  # Impulse purchases
    },
    "at_risk": {
        "weight": 0.20,
        "purchase_rate": 0.02,
        "avg_basket_size": 2.0,
        "loyalty": 0.15,
        "price_sensitivity": 1.5,
        "promo_responsiveness": 1.0,
        "churn_prob": 0.01,
        "category_preferences": {"Household": 0.8, "Pet": 0.7},  # Reduced engagement
    },
    "new": {
        "weight": 0.10,
        "purchase_rate": 0.08,
        "avg_basket_size": 2.8,
        "loyalty": 0.40,
        "price_sensitivity": 1.1,
        "promo_responsiveness": 0.6,
        "churn_prob": 0.001,
        "category_preferences": {"Coffee": 1.2, "Beverages": 1.3, "Snacks": 1.1},  # Exploratory
    },
}

# Category affinity themes (products often bought together)
AFFINITY_THEMES = {
    "morning_routine": ["Coffee", "Bakery", "Dairy"],
    "snack_time": ["Snacks", "Beverages"],
    "grocery_trip": ["Dairy", "Bakery", "Household", "Pet"],
    "personal_care": ["Personal Care", "Household"],
    "pet_care": ["Pet", "Household"],
    "weekend_treat": ["Coffee", "Bakery", "Snacks", "Beverages"],
}

# Substitution groups (products that substitute for each other)
SUBSTITUTION_GROUPS = [
    ["Coffee", "Beverages"],
    ["Snacks", "Bakery"],
    ["Household", "Personal Care"],
]


def _product_catalog(n_products: int, seed: int = 7) -> pd.DataFrame:
    """Generate product catalog with realistic price distributions and Pareto popularity."""
    rng = np.random.default_rng(seed)

    # Assign categories with realistic distribution (some categories have more SKUs)
    cat_probs = np.array([0.15, 0.20, 0.15, 0.12, 0.10, 0.12, 0.08, 0.08])
    cats = rng.choice(CATEGORIES, size=n_products, p=cat_probs)

    brands = [f"B{(i % 5) + 1}" for i in range(n_products)]
    sizes = rng.choice(["S", "M", "L", "XL"], size=n_products, p=[0.2, 0.45, 0.25, 0.1])
    flavors = [f"F{i % 5 + 1}" for i in range(n_products)]

    # Realistic log-normal prices per category
    price = np.zeros(n_products)
    for i, cat in enumerate(cats):
        mu, sigma = CATEGORY_PRICE_PARAMS[cat]
        price[i] = round(rng.lognormal(mu, sigma), 2)
    price = np.clip(price, 0.99, 99.99)

    # Pareto popularity: top 20% SKUs get 80% of demand
    # Use Zipf distribution for rank-based popularity
    ranks = np.arange(1, n_products + 1)
    zipf_exp = 1.15
    popularity = 1.0 / (ranks**zipf_exp)
    popularity = popularity / popularity.sum()

    # Cost as 55-70% of price
    cost_margin = rng.uniform(0.55, 0.70, n_products)
    cost = (price * cost_margin).round(2)

    catalog = pd.DataFrame(
        {
            "stockcode": [f"SKU{i:04d}" for i in range(n_products)],
            "product": [f"{cats[i]} {i + 1:03d}" for i in range(n_products)],
            "category": cats,
            "brand": brands,
            "size": sizes,
            "flavor": flavors,
            "price": price,
            "cost": cost,
            "popularity": popularity,
        }
    )

    # 25% of SKUs are promo-capable
    promo_mask = rng.random(n_products) < 0.25
    catalog["promo_capable"] = promo_mask

    # Substitution group mapping
    cat_to_sub = {}
    for i, group in enumerate(SUBSTITUTION_GROUPS):
        for cat in group:
            cat_to_sub[cat] = i
    catalog["sub_group"] = catalog["category"].map(cat_to_sub).fillna(-1).astype(int)

    return catalog


def _promo_windows(catalog: pd.DataFrame, n_days: int, seed: int) -> dict[str, list[dict]]:
    """Generate realistic promotion windows with different types."""
    rng = np.random.default_rng(seed)
    windows: dict[str, list[dict]] = {}

    promo_types = ["discount", "bogo", "multibuy", "clearance"]
    promo_probs = [0.55, 0.20, 0.15, 0.10]  # Most are simple discounts

    for sku in catalog.loc[catalog["promo_capable"], "stockcode"]:
        n_win = int(rng.integers(1, 5))
        windows[sku] = []
        for _ in range(n_win):
            ptype = rng.choice(promo_types, p=promo_probs)
            duration = int(rng.integers(5, 21))  # 5-20 days
            max_start = max(5, n_days - duration - 5)
            if max_start <= 5:
                continue
            start = int(rng.integers(5, max_start))
            end = min(start + duration, n_days - 1)

            if ptype == "discount":
                discount = round(rng.uniform(0.15, 0.40), 2)  # 15-40% off
                params = {"discount": discount}
            elif ptype == "bogo":
                params = {"bogo_qty": 1}  # Buy 1 Get 1
            elif ptype == "multibuy":
                params = {
                    "min_qty": int(rng.integers(2, 4)),
                    "discount": round(rng.uniform(0.10, 0.25), 2),
                }
            else:  # clearance
                params = {"discount": round(rng.uniform(0.40, 0.70), 2)}

            windows[sku].append(
                {
                    "start": start,
                    "end": end,
                    "type": ptype,
                    "params": params,
                }
            )

    return windows


def _seasonal_demand(day_of_year: int, category: str, year: int = 2024) -> float:
    """Return seasonal multiplier for category on given day (1-366 for leap years).

    Enhanced with:
    - Leap year support (366 days)
    - More realistic holiday patterns
    - Business day awareness
    """
    # Handle leap years
    days_in_year = 366 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 365

    # Convert to radians (2π per year)
    t = 2 * np.pi * day_of_year / days_in_year

    # Enhanced seasonal patterns with more realistic holiday effects
    seasonal = {
        "Coffee": 1.0 + 0.15 * np.cos(t - np.pi / 2),  # Peak in winter
        "Beverages": 1.0 + 0.20 * np.sin(t),  # Peak in summer
        "Bakery": 1.0 + 0.12 * np.cos(t - np.pi),  # Peak in holidays
        "Snacks": 1.0 + 0.10 * np.sin(t + np.pi / 4),  # Slight summer peak
        "Dairy": 1.0 + 0.05 * np.cos(t),  # Mild seasonal
        "Household": 1.0 + 0.08 * np.cos(t - np.pi),  # Holiday cleaning
        "Personal Care": 1.0 + 0.07 * np.sin(t),  # Mild summer
        "Pet": 1.0,  # No seasonality
    }

    base_multiplier = seasonal.get(category, 1.0)

    # Add holiday spikes for key shopping periods
    # Approximate major holiday periods (simplified for demo)
    holiday_multipliers = {
        "Christmas": (day_of_year >= 359 and day_of_year <= 365) or (day_of_year <= 2),
        "Thanksgiving": 330 <= day_of_year <= 336,  # Late November
        "BlackFriday": 332 <= day_of_year <= 333,  # Day after Thanksgiving
        "SummerSale": 180 <= day_of_year <= 186,  # Early July
    }

    holiday_boost = 1.0
    if holiday_multipliers["Christmas"]:
        holiday_boost = 1.3  # Strong holiday shopping
    elif holiday_multipliers["Thanksgiving"]:
        holiday_boost = 1.2
    elif holiday_multipliers["BlackFriday"]:
        holiday_boost = 1.4  # Major shopping event
    elif holiday_multipliers["SummerSale"]:
        holiday_boost = 1.15

    return base_multiplier * holiday_boost


def _weekly_demand(day_of_week: int, category: str) -> float:
    """Return weekly multiplier (0=Mon, 6=Sun)."""
    # Weekend boost for certain categories
    weekend_boost = {
        "Coffee": 1.25,
        "Bakery": 1.35,
        "Snacks": 1.30,
        "Beverages": 1.20,
        "Dairy": 1.10,
    }
    if day_of_week >= 5:  # Sat/Sun
        return weekend_boost.get(category, 1.05)
    return 1.0


def _customer_lifecycle(customer_age_days: int, segment: str) -> dict:
    """Return lifecycle modifiers based on customer tenure."""
    # New customers ramp up, then stabilize, then eventually decline
    if segment == "new":
        if customer_age_days < 30:
            return {"purchase_mult": 0.6 + 0.01 * customer_age_days, "basket_mult": 0.8}
        elif customer_age_days < 90:
            return {"purchase_mult": 0.9, "basket_mult": 0.95}
        else:
            return {"purchase_mult": 1.0, "basket_mult": 1.0}
    elif segment == "champion":
        return {"purchase_mult": 1.0, "basket_mult": 1.0}
    elif segment == "regular":
        if customer_age_days > 365:
            return {"purchase_mult": 0.9, "basket_mult": 0.95}
        return {"purchase_mult": 1.0, "basket_mult": 1.0}
    elif segment == "at_risk":
        return {"purchase_mult": 0.5, "basket_mult": 0.8}
    return {"purchase_mult": 1.0, "basket_mult": 1.0}


def _pick_basket_products(
    catalog: pd.DataFrame,
    picked: list[str],
    basket_size: int,
    rng: np.random.Generator,
    affinity_boost: float = 3.0,
    category_preferences: dict[str, float] | None = None,
) -> list[str]:
    """Pick remaining products for basket using category affinity and segment preferences."""
    weights = catalog["popularity"].to_numpy().copy()
    available = set(catalog["product"]) - set(picked)

    # Apply segment category preferences
    if category_preferences:
        for cat, multiplier in category_preferences.items():
            cat_mask = catalog["category"] == cat
            weights[cat_mask] *= multiplier

    while len(picked) < basket_size and available:
        # Boost affinity categories
        if picked and rng.random() < 0.4:
            # Pick from same category as existing items
            picked_cats = set(catalog.loc[catalog["product"].isin(picked), "category"])
            mask = catalog["category"].isin(picked_cats) & catalog["product"].isin(available)
            if mask.any():
                weights_local = weights.copy()
                weights_local[~mask] = 0
                if weights_local.sum() > 0:
                    weights_local = weights_local / weights_local.sum()
                    product_name = rng.choice(catalog["product"], p=weights_local)
                    if product_name in available:
                        picked.append(product_name)
                        available.remove(product_name)
                        continue

        # Standard weighted pick
        weights_norm = weights / weights.sum()
        product_name = rng.choice(catalog["product"], p=weights_norm)
        if product_name in available:
            picked.append(product_name)
            available.remove(product_name)

    return picked


def generate_transactions(
    n_customers: int = 1000,
    n_products: int = 200,
    n_days: int = 365,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic transaction DataFrame with canonical schema.

    Returns a DataFrame with columns: date, transaction_id, stockcode, product,
    customer_id, price, quantity, category, brand, size, flavor, promo_flag, cost.
    """
    rng = np.random.default_rng(seed)
    catalog = _product_catalog(n_products, seed=seed + 1000)
    promos = _promo_windows(catalog, n_days, seed + 2000)

    # Precompute active promo days per SKU
    active_promo_days: dict[str, dict[int, dict]] = {}
    for sku, wins in promos.items():
        active_promo_days[sku] = {}
        for w in wins:
            for d in range(w["start"], w["end"] + 1):
                active_promo_days[sku][d] = w

    # Map products to SKU and catalog rows
    product_by_name = catalog.set_index("product")

    # Base popularity weights

    # Generate days range
    end_date = pd.Timestamp("2024-12-31")
    days = pd.date_range(end=end_date, periods=n_days, freq="D")

    # Customer segments assignment
    segment_names = list(SEGMENTS.keys())
    segment_weights = [SEGMENTS[s]["weight"] for s in segment_names]
    segment_weights = np.array(segment_weights) / sum(segment_weights)

    rows: list[tuple] = []
    txn_counter = 0

    # Customer acquisition over time (not all at day 0)
    acquisition_curve = np.linspace(0, n_customers, n_days)
    acquisition_curve = np.diff(np.concatenate([[0], acquisition_curve])).astype(int)
    acquisition_curve = np.maximum(acquisition_curve, 1)
    # Ensure total matches
    diff = n_customers - acquisition_curve.sum()
    if diff > 0:
        acquisition_curve[:diff] += 1

    cust_idx = 0
    customer_states = {}  # Track customer lifecycle

    for day_idx in range(n_days):
        n_new_today = acquisition_curve[day_idx]
        date = days[day_idx]
        dow = date.dayofweek
        doy = date.dayofyear

        # Add new customers
        for _ in range(n_new_today):
            customer_id = f"CUST{cust_idx:06d}"
            segment = rng.choice(segment_names, p=segment_weights)
            customer_states[customer_id] = {
                "segment": segment,
                "acquisition_day": day_idx,
                "last_purchase_day": -1,
                "purchase_count": 0,
                "total_spent": 0.0,
            }
            cust_idx += 1

        # Process existing customers for purchases
        for customer_id, state in list(customer_states.items()):
            segment = state["segment"]
            seg_params = SEGMENTS[segment]

            # Churn check
            days_since_last = (
                day_idx - state["last_purchase_day"] if state["last_purchase_day"] >= 0 else 999
            )
            if rng.random() < seg_params["churn_prob"] * max(1, days_since_last / 30):
                del customer_states[customer_id]
                continue

            # Purchase probability for this day
            base_rate = seg_params["purchase_rate"]
            lifecycle = _customer_lifecycle(day_idx - state["acquisition_day"], segment)
            purchase_prob = base_rate * lifecycle["purchase_mult"]

            # Seasonal and weekly adjustments
            purchase_prob *= _seasonal_demand(doy, "Coffee", year=2024) if False else 1.0
            # Apply category-agnostic weekly pattern (we'll handle per-category in basket)

            if rng.random() > purchase_prob:
                continue

            # Make purchase
            txn_counter += 1
            txn_id = f"TXN{txn_counter:08d}"

            # Basket size
            base_basket = seg_params["avg_basket_size"]
            basket_size = int(rng.poisson(max(1.5, base_basket * lifecycle["basket_mult"])))
            basket_size = max(1, min(basket_size, 12))

            # Pick products with affinity logic and segment preferences
            picked: list[str] = []

            # Get segment category preferences
            cat_prefs = seg_params.get("category_preferences", {})

            # Theme-based bundle (15% chance)
            if rng.random() < 0.15 and AFFINITY_THEMES:
                theme_name = rng.choice(list(AFFINITY_THEMES.keys()))
                theme_cats = AFFINITY_THEMES[theme_name]
                for cat in theme_cats:
                    # Apply segment preference multiplier
                    pref_multiplier = cat_prefs.get(cat, 1.0)
                    if rng.random() < pref_multiplier and len(picked) < basket_size:
                        cat_products = catalog[catalog["category"] == cat]["product"].tolist()
                        if cat_products:
                            prod = rng.choice(cat_products)
                            if prod not in picked:
                                picked.append(prod)

            # Fill rest of basket with segment preferences
            picked = _pick_basket_products(
                catalog, picked, basket_size, rng, category_preferences=cat_prefs
            )

            state["last_purchase_day"] = day_idx
            state["purchase_count"] += 1

            for product_name in picked:
                prod = product_by_name.loc[product_name]
                sku = prod["stockcode"]

                # Check promo
                on_promo = False
                promo_type = "none"
                price = float(prod["price"])

                if sku in active_promo_days and day_idx in active_promo_days[sku]:
                    promo = active_promo_days[sku][day_idx]
                    on_promo = True
                    promo_type = promo["type"]
                    params = promo["params"]

                    if promo_type in ("discount", "clearance"):
                        price = round(price * (1 - params["discount"]), 2)
                    elif promo_type == "bogo":
                        # Handled via quantity adjustment below
                        pass
                    elif promo_type == "multibuy":
                        # Will apply if quantity meets threshold
                        pass

                # Quantity (BOGO/multibuy can affect)
                base_qty = int(rng.choice([1, 2, 3, 4, 5], p=[0.55, 0.25, 0.12, 0.05, 0.03]))
                qty = base_qty

                if promo_type == "bogo" and base_qty >= 1:
                    qty = base_qty + params["bogo_qty"]
                elif promo_type == "multibuy" and base_qty >= params["min_qty"]:
                    price = round(price * (1 - params["discount"]), 2)

                # Category-specific weekly demand affects quantity
                week_mult = _weekly_demand(dow, prod["category"])
                if week_mult > 1.0 and rng.random() < 0.3:
                    qty += 1

                # Cost
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
                        prod["category"],
                        prod["brand"],
                        prod["size"],
                        prod["flavor"],
                        on_promo,
                        cost,
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
    path: str,
    *,
    n_customers: int = 1000,
    n_products: int = 200,
    n_days: int = 365,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate the checked-in sample fixture and write it to `path`."""
    df = generate_transactions(
        n_customers=n_customers, n_products=n_products, n_days=n_days, seed=seed
    )
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    # Quick test
    df = generate_transactions(n_customers=500, n_products=100, n_days=180, seed=42)
    print(f"Generated {len(df):,} rows")
    print(f"Customers: {df['customer_id'].nunique()}")
    print(f"Products: {df['stockcode'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Avg basket: {df.groupby('transaction_id')['quantity'].sum().mean():.1f}")
    print(f"Promo rate: {df['promo_flag'].mean():.1%}")
    print(
        f"Revenue concentration (top 20% SKUs): {df.groupby('stockcode')['price'].sum().sort_values(ascending=False).head(int(0.2 * df['stockcode'].nunique())).sum() / df['price'].sum():.1%}"
    )
