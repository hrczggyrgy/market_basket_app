"""Test synthetic data realism - verify archetype behaviors match labels."""

import pandas as pd

# Generate synthetic data with archetype tracking
print("Regenerating with archetype tracking...")

# We'll manually track archetypes by re-running the generator logic
import numpy as np
import random
from datetime import date, timedelta
from collections import defaultdict

# Re-import generator constants
from src.data.generator import (
    ARCHETYPES,
    FMCG_CATALOG,
    FMCG_SKU,
    FMCG_NAME,
    FMCG_CATEGORY,
    FMCG_BRAND,
    FMCG_PACK_SIZE,
    FMCG_PRICE,
    CATEGORY_SEASONALITY,
    DOW_WEIGHTS,
    PACK_SIZE_QTY,
    _build_category_map,
    _build_brand_map,
    _generate_promo_calendar,
    _apply_dow_rejection,
    _monthly_acquisition_rate,
    _churn_hazard,
    _reactivation_probability,
    _sample_qty,
)

# Generate with archetype tracking
np.random.seed(42)
random.seed(42)

catalog = FMCG_CATALOG
cat_map = _build_category_map(catalog)
categories = sorted(cat_map.keys())
brand_map = _build_brand_map(catalog)

start = date(2022, 1, 1)
end = date(2024, 12, 31)
total_days = (end - start).days

promo_calendar = _generate_promo_calendar(catalog, start, end, 4, (0.20, 0.40), 42)

# Archetype allocation
archetype_names = list(ARCHETYPES.keys())
archetype_shares = [ARCHETYPES[k]["share"] for k in archetype_names]
archetype_shares = [s / sum(archetype_shares) for s in archetype_shares]

n_customers = 200
customer_archetypes = np.random.choice(archetype_names, n_customers, p=archetype_shares)
customer_ids = [f"CUST{i:04d}" for i in range(1, n_customers + 1)]

# Track archetype assignments
customer_archetype_map = dict(zip(customer_ids, customer_archetypes))

# Dirichlet brand-preference alpha per customer per category
cust_brand_alpha = {}
for cid, arch_name in zip(customer_ids, customer_archetypes):
    arch = ARCHETYPES[arch_name]
    cust_brand_alpha[cid] = {}
    for cat in categories:
        brands = brand_map[cat]
        n_brands = len(brands)
        alpha_base = arch["loyalty_alpha"]
        alphas = [alpha_base] * n_brands
        cust_brand_alpha[cid][cat] = dict(zip(brands, alphas))

# Generate transactions with archetype tracking
transactions = []
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

    cohort_month = int(np.random.exponential(12 / (0.03 * n_customers / 12)))
    cohort_month = min(cohort_month, total_days - 30)
    cohort_date = start + timedelta(days=cohort_month)
    current = cohort_date
    days_since_last = 0

    while current <= end:
        gap = max(1, int(np.random.weibull(k) * lam))
        current += timedelta(days=gap)
        if current > end:
            break

        if not _apply_dow_rejection(current, arch):
            continue

        days_since_last = 0

        basket_size = max(1, int(np.random.poisson(basket_lambda * 0.6) + 1))
        basket_size = min(basket_size, 25)

        n_cats = min(max(1, int(np.random.poisson(2.0))), len(categories))
        cat_weights = np.array([CATEGORY_SEASONALITY.get(cat, [1] * 12)[current.month - 1] for cat in categories])
        cat_weights = cat_weights / cat_weights.sum()
        trip_categories = np.random.choice(categories, n_cats, replace=False, p=cat_weights)

        for cat in trip_categories:
            if basket_size <= 0:
                break
            cat_products = cat_map[cat]
            cat_brands = brand_map[cat]

            alphas = [cust_brand_alpha[cid][cat][b] for b in cat_brands]
            alpha_sum = sum(alphas)
            probs = [a / alpha_sum for a in alphas]
            chosen_brand = np.random.choice(cat_brands, p=probs)

            brand_products = [p for p in cat_products if p[FMCG_BRAND] == chosen_brand]
            if not brand_products:
                continue

            chosen_product = random.choice(brand_products)
            sku = chosen_product[FMCG_SKU]

            qty = _sample_qty(chosen_product[FMCG_PACK_SIZE])

            base = chosen_product[FMCG_PRICE]
            promo_price = None
            for ps, pe, pp in promo_calendar.get(sku, []):
                if ps <= current <= pe:
                    promo_price = pp
                    break

            if promo_price is not None:
                price = promo_price
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
            }
            transactions.append(row)

            cust_brand_alpha[cid][cat][chosen_brand] += 1

            if arch_name in ("variety_seeker", "switcher") and np.random.random() < 0.25 and basket_size > 1:
                other_brands = [b for b in cat_brands if b != chosen_brand]
                if other_brands:
                    other_brand = np.random.choice(other_brands)
                    other_products = [p for p in cat_products if p[FMCG_BRAND] == other_brand]
                    if other_products:
                        other = random.choice(other_products)
                        osku = other[0]
                        oprice = round(other[8] * np.random.uniform(0.95, 1.05), 2)
                        oqty = _sample_qty(other[7])
                        row2 = {
                            "date": current.strftime("%Y-%m-%d"),
                            "transaction_id": f"INV{txn_id:0{max_txn_digits}d}",
                            "stockcode": osku,
                            "product": other[1],
                            "customer_id": cid,
                            "price": oprice,
                            "quantity": min(oqty, 99),
                            "category": other[2],
                            "brand": other[4],
                        }
                        transactions.append(row2)

            basket_size -= 1

        txn_id += 1

df = pd.DataFrame(transactions)
df["date"] = pd.to_datetime(df["date"])

print(f"Generated {len(df)} transactions for {n_customers} customers")

# Analyze brand switching by archetype
print("\n" + "="*60)
print("Analyzing brand switching by archetype")
print("="*60)

archetype_switching = {}

for archetype in ARCHETYPES.keys():
    archetype_customers = [cid for cid, arch in customer_archetype_map.items() if arch == archetype]
    archetype_df = df[df["customer_id"].isin(archetype_customers)]
    
    if len(archetype_df) < 10:
        print(f"\n{archetype}: Not enough data ({len(archetype_df)} transactions)")
        continue
    
    # Compute brand switching rate
    # Count how many times customers switch brands within the same category
    brand_switch_count = 0
    total_brand_transitions = 0
    
    for cid in archetype_customers:
        cust_df = archetype_df[archetype_df["customer_id"] == cid].sort_values("date")
        if len(cust_df) < 2:
            continue
        
        for i in range(1, len(cust_df)):
            prev_row = cust_df.iloc[i-1]
            curr_row = cust_df.iloc[i]
            
            # Same category, different brand = switch
            if prev_row["category"] == curr_row["category"] and prev_row["brand"] != curr_row["brand"]:
                brand_switch_count += 1
            total_brand_transitions += 1
    
    if total_brand_transitions > 0:
        switch_rate = brand_switch_count / total_brand_transitions
        archetype_switching[archetype] = {
            "switch_rate": switch_rate,
            "n_customers": len(archetype_customers),
            "n_transactions": len(archetype_df),
        }
        print(f"\n{archetype}:")
        print(f"  Customers: {len(archetype_customers)}")
        print(f"  Transactions: {len(archetype_df)}")
        print(f"  Brand switch rate: {switch_rate:.3f}")

# Verify expected ordering
print("\n" + "="*60)
print("Verification: Do archetypes behave as labeled?")
print("="*60)

if "loyal" in archetype_switching and "switcher" in archetype_switching:
    loyal_rate = archetype_switching["loyal"]["switch_rate"]
    switcher_rate = archetype_switching["switcher"]["switch_rate"]
    
    print(f"\nLoyal archetype switch rate: {loyal_rate:.3f}")
    print(f"Switcher archetype switch rate: {switcher_rate:.3f}")
    
    if switcher_rate > loyal_rate:
        print("✓ Switchers have higher brand switching rate than loyal customers (as expected)")
    else:
        print("✗ BUG: Switchers do NOT have higher brand switching rate than loyal customers")

# Check variety_seeker
if "variety_seeker" in archetype_switching:
    variety_rate = archetype_switching["variety_seeker"]["switch_rate"]
    print(f"\nVariety seeker switch rate: {variety_rate:.3f}")
    
    if variety_rate > loyal_rate:
        print("✓ Variety seekers have higher brand switching rate than loyal customers (as expected)")
    else:
        print("⚠ Variety seekers do not have higher brand switching rate than loyal customers")

# Check stock_up (should have lower switching due to bulk buying)
if "stock_up" in archetype_switching:
    stock_rate = archetype_switching["stock_up"]["switch_rate"]
    print(f"\nStock-up switch rate: {stock_rate:.3f}")
    
    if stock_rate < switcher_rate:
        print("✓ Stock-up customers have lower brand switching rate than switchers (as expected)")
    else:
        print("⚠ Stock-up customers do not have lower brand switching rate than switchers")

print("\n" + "="*60)
print("Synthetic data realism check complete")
print("="*60)
