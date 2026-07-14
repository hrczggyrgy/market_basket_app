"""Synthetic FMCG transaction data generator with realistic purchasing behaviour.

Generates customer archetypes, promotional calendars, seasonality,
brand loyalty dynamics, and customer lifecycle for a complete FMCG dataset.
"""

import random
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# FMCG catalog: (stockcode, product_name, category, subcategory, brand, brand_tier, flavour, pack_size, base_price)
FMCG_CATALOG = [
    # BEVERAGES - Cola
    (
        "CL001",
        "Cola Regular 330ml",
        "BEVERAGES",
        "COLA",
        "ColaCo",
        "MAINSTREAM",
        "REGULAR",
        "330ML",
        0.89,
    ),
    (
        "CL002",
        "Cola Zero 330ml",
        "BEVERAGES",
        "COLA",
        "ColaCo",
        "MAINSTREAM",
        "ZERO",
        "330ML",
        0.89,
    ),
    (
        "CL003",
        "Cola Cherry 330ml",
        "BEVERAGES",
        "COLA",
        "ColaCo",
        "MAINSTREAM",
        "CHERRY",
        "330ML",
        0.89,
    ),
    (
        "CL004",
        "Cola Regular 2L",
        "BEVERAGES",
        "COLA",
        "ColaCo",
        "MAINSTREAM",
        "REGULAR",
        "2L",
        1.89,
    ),
    ("CL005", "Cola Zero 2L", "BEVERAGES", "COLA", "ColaCo", "MAINSTREAM", "ZERO", "2L", 1.89),
    ("CL006", "Cola PL 2L", "BEVERAGES", "COLA", "SmartSave", "PRIVATE", "REGULAR", "2L", 0.65),
    (
        "CL007",
        "Cola PL 330ml 6pk",
        "BEVERAGES",
        "COLA",
        "SmartSave",
        "PRIVATE",
        "REGULAR",
        "330ML_6PK",
        2.50,
    ),
    # BEVERAGES - Water
    (
        "W001",
        "Spring Water Still 500ml",
        "BEVERAGES",
        "WATER",
        "AquaPure",
        "MAINSTREAM",
        "STILL",
        "500ML",
        0.55,
    ),
    (
        "W002",
        "Spring Water Sparkling 500ml",
        "BEVERAGES",
        "WATER",
        "AquaPure",
        "MAINSTREAM",
        "SPARKLING",
        "500ML",
        0.65,
    ),
    (
        "W003",
        "Water PL Still 6x1.5L",
        "BEVERAGES",
        "WATER",
        "SmartSave",
        "PRIVATE",
        "STILL",
        "6X1.5L",
        2.20,
    ),
    # BEVERAGES - Juice
    (
        "J001",
        "Juice Orange 1L",
        "BEVERAGES",
        "JUICE",
        "FruitCo",
        "MAINSTREAM",
        "ORANGE",
        "1L",
        1.49,
    ),
    ("J002", "Juice Apple 1L", "BEVERAGES", "JUICE", "FruitCo", "MAINSTREAM", "APPLE", "1L", 1.49),
    ("J003", "Juice Mango 1L", "BEVERAGES", "JUICE", "FruitCo", "MAINSTREAM", "MANGO", "1L", 1.69),
    (
        "J004",
        "Juice Orange PL 1L",
        "BEVERAGES",
        "JUICE",
        "SmartSave",
        "PRIVATE",
        "ORANGE",
        "1L",
        0.89,
    ),
    # BEVERAGES - Beer
    ("B001", "Lager Premium 6pk", "BEVERAGES", "BEER", "BrewCo", "PREMIUM", "LAGER", "6PK", 5.99),
    (
        "B002",
        "Lager Mainstream 6pk",
        "BEVERAGES",
        "BEER",
        "BrewCo",
        "MAINSTREAM",
        "LAGER",
        "6PK",
        4.49,
    ),
    ("B003", "Ale Premium 4pk", "BEVERAGES", "BEER", "CraftBrew", "PREMIUM", "ALE", "4PK", 5.49),
    ("B004", "Lager PL 6pk", "BEVERAGES", "BEER", "SmartSave", "PRIVATE", "LAGER", "6PK", 3.29),
    # DAIRY - Yoghurt
    (
        "Y001",
        "Yoghurt Strawberry 150g",
        "DAIRY",
        "YOGHURT",
        "DairyCo",
        "MAINSTREAM",
        "STRAWBERRY",
        "150G",
        0.75,
    ),
    (
        "Y002",
        "Yoghurt Vanilla 150g",
        "DAIRY",
        "YOGHURT",
        "DairyCo",
        "MAINSTREAM",
        "VANILLA",
        "150G",
        0.75,
    ),
    (
        "Y003",
        "Yoghurt Blueberry 150g",
        "DAIRY",
        "YOGHURT",
        "DairyCo",
        "MAINSTREAM",
        "BLUEBERRY",
        "150G",
        0.75,
    ),
    (
        "Y004",
        "Yoghurt Peach 150g",
        "DAIRY",
        "YOGHURT",
        "DairyCo",
        "MAINSTREAM",
        "PEACH",
        "150G",
        0.75,
    ),
    ("Y005", "Yoghurt Greek 500g", "DAIRY", "YOGHURT", "GreekCo", "PREMIUM", "PLAIN", "500G", 1.99),
    (
        "Y006",
        "Yoghurt PL Strawberry 150g",
        "DAIRY",
        "YOGHURT",
        "SmartSave",
        "PRIVATE",
        "STRAWBERRY",
        "150G",
        0.45,
    ),
    # DAIRY - Milk
    ("M001", "Milk Whole 2L", "DAIRY", "MILK", "DairyCo", "MAINSTREAM", "WHOLE", "2L", 1.29),
    ("M002", "Milk Semi-Skimmed 2L", "DAIRY", "MILK", "DairyCo", "MAINSTREAM", "SEMI", "2L", 1.29),
    ("M003", "Milk Skimmed 2L", "DAIRY", "MILK", "DairyCo", "MAINSTREAM", "SKIMMED", "2L", 1.29),
    ("M004", "Milk Oat 1L", "DAIRY", "MILK", "PlantCo", "PREMIUM", "OAT", "1L", 1.89),
    # DAIRY - Cheese
    (
        "C001",
        "Cheddar Mature 200g",
        "DAIRY",
        "CHEESE",
        "DairyCo",
        "MAINSTREAM",
        "MATURE",
        "200G",
        2.49,
    ),
    ("C002", "Cheddar Mild 200g", "DAIRY", "CHEESE", "DairyCo", "MAINSTREAM", "MILD", "200G", 2.49),
    ("C003", "Cheddar PL 200g", "DAIRY", "CHEESE", "SmartSave", "PRIVATE", "MATURE", "200G", 1.69),
    # SNACKS - Crisps
    (
        "S001",
        "Crisps Ready Salted 150g",
        "SNACKS",
        "CRISPS",
        "SnackCo",
        "MAINSTREAM",
        "READY_SALTED",
        "150G",
        1.79,
    ),
    (
        "S002",
        "Crisps Cheese Onion 150g",
        "SNACKS",
        "CRISPS",
        "SnackCo",
        "MAINSTREAM",
        "CHEESE_ONION",
        "150G",
        1.79,
    ),
    ("S003", "Crisps BBQ 150g", "SNACKS", "CRISPS", "SnackCo", "MAINSTREAM", "BBQ", "150G", 1.79),
    (
        "S004",
        "Crisps Salt Vinegar 150g",
        "SNACKS",
        "CRISPS",
        "SnackCo",
        "MAINSTREAM",
        "SALT_VINEGAR",
        "150G",
        1.79,
    ),
    (
        "S005",
        "Crisps PL 150g",
        "SNACKS",
        "CRISPS",
        "SmartSave",
        "PRIVATE",
        "READY_SALTED",
        "150G",
        0.99,
    ),
    (
        "S006",
        "Crisps Loaded 200g",
        "SNACKS",
        "CRISPS",
        "SnackCo",
        "PREMIUM",
        "SOUR_CREAM",
        "200G",
        2.49,
    ),
    # SNACKS - Biscuits
    (
        "BIS01",
        "Biscuits Chocolate 200g",
        "SNACKS",
        "BISCUITS",
        "BakeCo",
        "MAINSTREAM",
        "CHOCOLATE",
        "200G",
        1.49,
    ),
    (
        "BIS02",
        "Biscuits Digestive 400g",
        "SNACKS",
        "BISCUITS",
        "BakeCo",
        "MAINSTREAM",
        "DIGESTIVE",
        "400G",
        1.89,
    ),
    (
        "BIS03",
        "Biscuits Digestive PL 400g",
        "SNACKS",
        "BISCUITS",
        "SmartSave",
        "PRIVATE",
        "DIGESTIVE",
        "400G",
        0.99,
    ),
    (
        "BIS04",
        "Biscuits Cookies 150g",
        "SNACKS",
        "BISCUITS",
        "BakeCo",
        "PREMIUM",
        "CHOC_CHIP",
        "150G",
        2.29,
    ),
    # SNACKS - Nuts
    (
        "N001",
        "Nuts Roasted Salted 200g",
        "SNACKS",
        "NUTS",
        "NutCo",
        "MAINSTREAM",
        "SALTED",
        "200G",
        2.99,
    ),
    ("N002", "Nuts Mixed 200g", "SNACKS", "NUTS", "NutCo", "MAINSTREAM", "MIXED", "200G", 3.49),
    # HOUSEHOLD - Detergent
    (
        "D001",
        "Detergent Regular 1.5L",
        "HOUSEHOLD",
        "DETERGENT",
        "CleanCo",
        "MAINSTREAM",
        "REGULAR",
        "1.5L",
        4.99,
    ),
    (
        "D002",
        "Detergent Bio 1.5L",
        "HOUSEHOLD",
        "DETERGENT",
        "CleanCo",
        "MAINSTREAM",
        "BIO",
        "1.5L",
        4.99,
    ),
    (
        "D003",
        "Detergent Capsules 30pk",
        "HOUSEHOLD",
        "DETERGENT",
        "CleanCo",
        "MAINSTREAM",
        "CAPSULES",
        "30PK",
        7.99,
    ),
    (
        "D004",
        "Detergent PL 1.5L",
        "HOUSEHOLD",
        "DETERGENT",
        "SmartSave",
        "PRIVATE",
        "REGULAR",
        "1.5L",
        2.99,
    ),
    # HOUSEHOLD - Fabric Conditioner
    (
        "F001",
        "Fabric Softener Lavender 1L",
        "HOUSEHOLD",
        "FABRIC",
        "CleanCo",
        "MAINSTREAM",
        "LAVENDER",
        "1L",
        3.49,
    ),
    (
        "F002",
        "Fabric Softener Fresh 1L",
        "HOUSEHOLD",
        "FABRIC",
        "CleanCo",
        "MAINSTREAM",
        "FRESH",
        "1L",
        3.49,
    ),
    (
        "F003",
        "Fabric Softener PL 1L",
        "HOUSEHOLD",
        "FABRIC",
        "SmartSave",
        "PRIVATE",
        "FRESH",
        "1L",
        1.99,
    ),
    # HOUSEHOLD - Cleaning
    (
        "CLN01",
        "Kitchen Cleaner Spray",
        "HOUSEHOLD",
        "CLEANING",
        "CleanCo",
        "MAINSTREAM",
        "CITRUS",
        "500ML",
        2.49,
    ),
    (
        "CLN02",
        "Kitchen Cleaner PL",
        "HOUSEHOLD",
        "CLEANING",
        "SmartSave",
        "PRIVATE",
        "CITRUS",
        "500ML",
        1.29,
    ),
    (
        "CLN03",
        "Bathroom Cleaner Spray",
        "HOUSEHOLD",
        "CLEANING",
        "CleanCo",
        "MAINSTREAM",
        "BLEACH",
        "500ML",
        2.49,
    ),
    # PERSONAL CARE - Shampoo
    (
        "P001",
        "Shampoo Normal 400ml",
        "PERSONAL_CARE",
        "SHAMPOO",
        "HairCo",
        "MAINSTREAM",
        "NORMAL",
        "400ML",
        3.49,
    ),
    (
        "P002",
        "Shampoo Dry 400ml",
        "PERSONAL_CARE",
        "SHAMPOO",
        "HairCo",
        "MAINSTREAM",
        "DRY",
        "400ML",
        3.49,
    ),
    (
        "P003",
        "Shampoo Colour 400ml",
        "PERSONAL_CARE",
        "SHAMPOO",
        "HairCo",
        "MAINSTREAM",
        "COLOUR",
        "400ML",
        3.49,
    ),
    (
        "P004",
        "Shampoo PL 400ml",
        "PERSONAL_CARE",
        "SHAMPOO",
        "SmartSave",
        "PRIVATE",
        "NORMAL",
        "400ML",
        1.99,
    ),
    # PERSONAL CARE - Toothpaste
    (
        "T001",
        "Toothpaste Whitening 75ml",
        "PERSONAL_CARE",
        "TOOTHPASTE",
        "DentCo",
        "MAINSTREAM",
        "WHITENING",
        "75ML",
        2.99,
    ),
    (
        "T002",
        "Toothpaste Sensitive 75ml",
        "PERSONAL_CARE",
        "TOOTHPASTE",
        "DentCo",
        "MAINSTREAM",
        "SENSITIVE",
        "75ML",
        3.49,
    ),
    (
        "T003",
        "Toothpaste PL 75ml",
        "PERSONAL_CARE",
        "TOOTHPASTE",
        "SmartSave",
        "PRIVATE",
        "WHITENING",
        "75ML",
        1.49,
    ),
    # PERSONAL CARE - Soap
    (
        "SOAP1",
        "Soap Bar 3pk",
        "PERSONAL_CARE",
        "SOAP",
        "BodyCo",
        "MAINSTREAM",
        "MOISTURISING",
        "3PK",
        2.49,
    ),
    (
        "SOAP2",
        "Soap Liquid 250ml",
        "PERSONAL_CARE",
        "SOAP",
        "BodyCo",
        "MAINSTREAM",
        "MOISTURISING",
        "250ML",
        2.99,
    ),
    (
        "SOAP3",
        "Soap PL Liquid 250ml",
        "PERSONAL_CARE",
        "SOAP",
        "SmartSave",
        "PRIVATE",
        "MOISTURISING",
        "250ML",
        1.29,
    ),
]

# Catalog indices
FMCG_SKU = 0
FMCG_NAME = 1
FMCG_CATEGORY = 2
FMCG_SUBCATEGORY = 3
FMCG_BRAND = 4
FMCG_BRAND_TIER = 5
FMCG_FLAVOUR = 6
FMCG_PACK_SIZE = 7
FMCG_PRICE = 8

# Category-level seasonality multipliers (month 1-12)
CATEGORY_SEASONALITY = {
    "BEVERAGES": [0.85, 0.85, 0.90, 0.95, 1.05, 1.25, 1.35, 1.30, 1.00, 0.90, 0.90, 1.05],
    "DAIRY": [1.00] * 12,
    "SNACKS": [0.90, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.10, 1.00, 1.00, 1.05, 1.20],
    "HOUSEHOLD": [1.10, 1.00, 1.05, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.15],
    "PERSONAL_CARE": [1.00] * 12,
}

# Day-of-week purchase probability weights (Mon-Sun)
DOW_WEIGHTS = [0.09, 0.10, 0.11, 0.15, 0.21, 0.20, 0.14]

# Customer archetypes
ARCHETYPES = {
    "loyal": {
        "share": 0.15,
        "weibull_lambda": 5,
        "weibull_k": 3.0,
        "basket_lambda": 6,
        "loyalty_alpha": 12,
        "churn_days": None,
        "promo_sensitivity": 0.3,
    },
    "switcher": {
        "share": 0.20,
        "weibull_lambda": 8,
        "weibull_k": 1.5,
        "basket_lambda": 5,
        "loyalty_alpha": 1.5,
        "churn_days": None,
        "promo_sensitivity": 0.9,
    },
    "variety_seeker": {
        "share": 0.10,
        "weibull_lambda": 6,
        "weibull_k": 2.0,
        "basket_lambda": 8,
        "loyalty_alpha": 0.5,
        "churn_days": None,
        "promo_sensitivity": 0.6,
    },
    "stock_up": {
        "share": 0.25,
        "weibull_lambda": 14,
        "weibull_k": 2.5,
        "basket_lambda": 15,
        "loyalty_alpha": 6,
        "churn_days": None,
        "promo_sensitivity": 0.4,
    },
    "occasional": {
        "share": 0.20,
        "weibull_lambda": 21,
        "weibull_k": 1.2,
        "basket_lambda": 4,
        "loyalty_alpha": 4,
        "churn_days": None,
        "promo_sensitivity": 0.3,
    },
    "lapsing": {
        "share": 0.10,
        "weibull_lambda": 8,
        "weibull_k": 1.0,
        "basket_lambda": 4,
        "loyalty_alpha": 3,
        "churn_days": 180,
        "promo_sensitivity": 0.5,
    },
}

# Pack size → quantity mapping
PACK_SIZE_QTY = {
    "330ML": (1, 6),
    "2L": (1, 2),
    "330ML_6PK": 1,
    "500ML": (1, 6),
    "6X1.5L": 1,
    "1L": (1, 3),
    "6PK": 1,
    "4PK": 1,
    "150G": (1, 4),
    "500G": 1,
    "200G": (1, 3),
    "400G": 1,
    "75ML": (1, 2),
    "400ML": 1,
    "250ML": 1,
    "1.5L": 1,
    "30PK": 1,
    "3PK": 1,
}


def _sample_qty(pack_size: str) -> int:
    if pack_size not in PACK_SIZE_QTY:
        return 1
    val = PACK_SIZE_QTY[pack_size]
    if isinstance(val, int):
        return val
    return random.randint(val[0], val[1])


def _build_product_index(catalog: List[tuple]) -> dict:
    idx = {}
    for p in catalog:
        idx[p[FMCG_SKU]] = {
            "name": p[FMCG_NAME],
            "category": p[FMCG_CATEGORY],
            "subcategory": p[FMCG_SUBCATEGORY],
            "brand": p[FMCG_BRAND],
            "brand_tier": p[FMCG_BRAND_TIER],
            "flavour": p[FMCG_FLAVOUR],
            "pack_size": p[FMCG_PACK_SIZE],
            "base_price": p[FMCG_PRICE],
        }
    return idx


def _build_category_map(catalog: List[tuple]) -> Dict[str, List[tuple]]:
    cat_map = defaultdict(list)
    for p in catalog:
        cat_map[p[FMCG_CATEGORY]].append(p)
    return dict(cat_map)


def _build_brand_map(catalog: List[tuple]) -> Dict[str, List[str]]:
    brand_map = defaultdict(set)
    for p in catalog:
        brand_map[p[FMCG_CATEGORY]].add(p[FMCG_BRAND])
    return {k: sorted(v) for k, v in brand_map.items()}


def _generate_promo_calendar(
    catalog: List[tuple],
    start: date,
    end: date,
    promos_per_sku: int = 4,
    depth_range: Tuple[float, float] = (0.20, 0.40),
    seed: int = 42,
) -> Dict[str, List[Tuple[date, date, float]]]:
    rng = random.Random(seed)
    cal = {}
    total_days = (end - start).days
    for p in catalog:
        sku = p[FMCG_SKU]
        base_price = p[FMCG_PRICE]
        n_promos = max(1, promos_per_sku)
        if total_days < 60:
            n_promos = 1
        starts = sorted(
            rng.sample(range(30, total_days - 30), min(n_promos, max(1, total_days - 60)))
        )
        cal[sku] = []
        for d in starts:
            ps = start + timedelta(days=d)
            pe = ps + timedelta(days=rng.randint(10, 18))
            if pe > end:
                pe = end
            discount = rng.uniform(depth_range[0], depth_range[1])
            promo_price = round(base_price * (1 - discount), 2)
            cal[sku].append((ps, pe, promo_price))
    return cal


def _apply_dow_rejection(date: date, archetype: dict) -> bool:
    prob = DOW_WEIGHTS[date.weekday()]
    threshold = 0.25 if archetype["weibull_lambda"] > 30 else 0.15
    return np.random.random() < prob * (1 + threshold)


def _monthly_acquisition_rate(month: int) -> float:
    base = 0.03
    if month in (1, 9):
        return base * 2.5
    return base


def _churn_hazard(days_since_last_purchase: int) -> float:
    if days_since_last_purchase > 180:
        return 0.40
    elif days_since_last_purchase > 90:
        return 0.20
    return 0.02


def _reactivation_probability(days_since_churn: int, is_on_promo: bool) -> float:
    if days_since_churn < 60:
        return 0.02
    if is_on_promo:
        return 0.12
    return 0.03


def generate_transactions(
    n_customers: int = 500,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    monthly_acquisition_rate: float = 0.03,
    promos_per_sku: int = 4,
    promo_depth_range: Tuple[float, float] = (0.20, 0.40),
    archetype_mix: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic FMCG transaction data with realistic purchasing patterns.

    Produces ~30k-60k rows for 500 customers over 3 years (adjustable via
    archetype parameters and n_customers).

    Args:
        n_customers: Number of customer IDs to generate
        start_date: Start of data period
        end_date: End of data period
        monthly_acquisition_rate: Base monthly new customer growth rate
        promos_per_sku: Promotions per SKU per year
        promo_depth_range: (min, max) discount fraction
        archetype_mix: Override default archetype shares
        seed: Random seed

    Returns:
        DataFrame with columns: date, transaction_id, stockcode, product,
        customer_id, price, quantity, category, brand, size, flavour
    """
    np.random.seed(seed)
    random.seed(seed)

    catalog = FMCG_CATALOG
    cat_map = _build_category_map(catalog)
    categories = sorted(cat_map.keys())
    brand_map = _build_brand_map(catalog)

    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    total_days = (end - start).days

    # Promotional calendar
    promo_calendar = _generate_promo_calendar(
        catalog, start, end, promos_per_sku, promo_depth_range, seed
    )

    # Archetype allocation
    mix = archetype_mix or {k: v["share"] for k, v in ARCHETYPES.items()}
    archetype_names = list(mix.keys())
    archetype_shares = list(mix.values())
    archetype_shares = [s / sum(archetype_shares) for s in archetype_shares]

    # Customer assignment
    customer_archetypes = np.random.choice(archetype_names, n_customers, p=archetype_shares)
    customer_ids = [f"CUST{i:04d}" for i in range(1, n_customers + 1)]

    # Dirichlet brand-preference alpha per customer per category
    cust_brand_alpha: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cid, arch_name in zip(customer_ids, customer_archetypes):
        arch = ARCHETYPES[arch_name]
        cust_brand_alpha[cid] = {}
        for cat in categories:
            brands = brand_map[cat]
            n_brands = len(brands)
            alpha_base = arch["loyalty_alpha"]
            alphas = [alpha_base] * n_brands
            cust_brand_alpha[cid][cat] = dict(zip(brands, alphas))

    # Customer acquisition timeline
    customer_first_active: Dict[str, date] = {}
    customer_last_active: Dict[str, Optional[date]] = {}
    customer_status: Dict[str, str] = {}  # active / churned / reactivated

    for i, cid in enumerate(customer_ids):
        cohort_month = int(
            np.random.exponential(12 / (monthly_acquisition_rate * n_customers / 12))
        )
        cohort_month = min(cohort_month, total_days - 30)
        cohort_date = start + timedelta(days=cohort_month)
        customer_first_active[cid] = cohort_date
        customer_last_active[cid] = None
        customer_status[cid] = "active"

    transactions: List[dict] = []
    txn_id = 1
    max_txn_digits = 9

    for i, cid in enumerate(customer_ids):
        arch_name = customer_archetypes[i]
        arch = ARCHETYPES[arch_name]
        lam = arch["weibull_lambda"]
        k = arch["weibull_k"]
        basket_lambda = arch["basket_lambda"]
        promo_sensitivity = arch["promo_sensitivity"]
        churn_days = arch["churn_days"]

        first_date = customer_first_active[cid]
        current = first_date
        days_since_last = 0
        churn_date: Optional[date] = None

        while current <= end:
            # Weibull inter-purchase time
            gap = max(1, int(np.random.weibull(k) * lam))
            current += timedelta(days=gap)
            if current > end:
                break

            # Churn check
            if (
                customer_status[cid] == "active"
                and churn_days is not None
                and days_since_last > churn_days
                and np.random.random() < _churn_hazard(days_since_last)
            ):
                customer_status[cid] = "churned"
                churn_date = current
                break

            # Reactivation for churned customers (if we re-encounter them)
            if customer_status[cid] == "churned" and churn_date is not None:
                reactivate = _reactivation_probability((current - churn_date).days, False)
                if np.random.random() < reactivate:
                    customer_status[cid] = "reactivated"

            # Day-of-week rejection sampling
            if not _apply_dow_rejection(current, arch):
                continue

            days_since_last = 0
            customer_last_active[cid] = current

            # Basket size
            basket_size = max(1, int(np.random.poisson(basket_lambda * 0.6) + 1))
            basket_size = min(basket_size, 25)

            # Decide how many categories to buy from
            n_cats = min(max(1, int(np.random.poisson(2.0))), len(categories))
            cat_weights = np.array(
                [CATEGORY_SEASONALITY.get(cat, [1] * 12)[current.month - 1] for cat in categories]
            )
            cat_weights = cat_weights / cat_weights.sum()
            trip_categories = np.random.choice(categories, n_cats, replace=False, p=cat_weights)

            for cat in trip_categories:
                if basket_size <= 0:
                    break
                cat_products = cat_map[cat]
                cat_brands = brand_map[cat]

                # Sample brand from Dirichlet preference
                alphas = [cust_brand_alpha[cid][cat][b] for b in cat_brands]
                alpha_sum = sum(alphas)
                probs = [a / alpha_sum for a in alphas]
                chosen_brand = np.random.choice(cat_brands, p=probs)

                # Filter products by chosen brand
                brand_products = [p for p in cat_products if p[FMCG_BRAND] == chosen_brand]
                if not brand_products:
                    continue

                # Pick a specific product (by flavour/variant)
                chosen_product = random.choice(brand_products)
                sku = chosen_product[FMCG_SKU]

                # Quantity based on pack size
                qty = _sample_qty(chosen_product[FMCG_PACK_SIZE])

                # Price with small daily variation + promo check
                base = chosen_product[FMCG_PRICE]
                promo_price = None
                for ps, pe, pp in promo_calendar.get(sku, []):
                    if ps <= current <= pe:
                        promo_price = pp
                        break
                if promo_price is not None:
                    price = promo_price
                    # Promo-sensitive customers may buy extra
                    if np.random.random() < promo_sensitivity:
                        qty += random.randint(1, min(3, qty + 1))
                else:
                    price = round(base * np.random.uniform(0.95, 1.05), 2)

                row = {
                    "date": current.strftime("%Y-%m-%d"),
                    "transaction_id": f"INV{txn_id:0{max_txn_digits}d}",
                    "stockcode": sku,
                    "product": chosen_product[FMCG_NAME],
                    "customer_id": cid,
                    "price": price,
                    "quantity": min(qty, 99),
                    "category": chosen_product[FMCG_CATEGORY],
                    "brand": chosen_product[FMCG_BRAND],
                    "size": chosen_product[FMCG_PACK_SIZE],
                    "flavour": chosen_product[FMCG_FLAVOUR],
                }
                transactions.append(row)

                # Update Dirichlet preference for chosen brand (habit reinforcement)
                cust_brand_alpha[cid][cat][chosen_brand] += 1

                # Occasionally pick a second product from another brand (variety)
                if (
                    arch_name in ("variety_seeker", "switcher")
                    and np.random.random() < 0.25
                    and basket_size > 1
                ):
                    other_brands = [b for b in cat_brands if b != chosen_brand]
                    if other_brands:
                        other_brand = np.random.choice(other_brands)
                        other_products = [p for p in cat_products if p[FMCG_BRAND] == other_brand]
                        if other_products:
                            other = random.choice(other_products)
                            osku = other[FMCG_SKU]
                            oprice = round(other[FMCG_PRICE] * np.random.uniform(0.95, 1.05), 2)
                            oqty = _sample_qty(other[FMCG_PACK_SIZE])
                            row2 = {
                                "date": current.strftime("%Y-%m-%d"),
                                "transaction_id": f"INV{txn_id:0{max_txn_digits}d}",
                                "stockcode": osku,
                                "product": other[FMCG_NAME],
                                "customer_id": cid,
                                "price": oprice,
                                "quantity": min(oqty, 99),
                                "category": other[FMCG_CATEGORY],
                                "brand": other[FMCG_BRAND],
                                "size": other[FMCG_PACK_SIZE],
                                "flavour": other[FMCG_FLAVOUR],
                            }
                            transactions.append(row2)

                basket_size -= 1

            txn_id += 1

    df = pd.DataFrame(transactions)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "transaction_id"]).reset_index(drop=True)
    return df


def save_sample_data(output_path: str = "data/sample_transactions.csv", **kwargs):
    """Generate and save sample data."""
    df = generate_transactions(**kwargs)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} transactions, {df['transaction_id'].nunique()} unique orders")
    print(f"Saved to {output_path}")
    return df


if __name__ == "__main__":
    save_sample_data()
