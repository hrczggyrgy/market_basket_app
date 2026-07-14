"""Synthetic transaction data generator matching the exact schema."""

import random
from datetime import timedelta

import numpy as np
import pandas as pd

# Product catalog with realistic names
PRODUCT_CATALOG = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", "HOME", "WHITE", "SMALL", "HEART"),
    ("71053", "WHITE METAL LANTERN", "HOME", "WHITE", "MEDIUM", "LANTERN"),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER", "HOME", "CREAM", "SMALL", "HEART"),
    (
        "84029G",
        "KNITTED UNION FLAG HOT WATER BOTTLE",
        "HOME",
        "MULTI",
        "MEDIUM",
        "UNION FLAG",
    ),
    ("84029E", "RED WOOLLY HOT WATER BOTTLE", "HOME", "RED", "MEDIUM", "WOOLLY"),
    ("22752", "SET 7 BABUSHKA NESTING BOXES", "HOME", "MULTI", "SMALL", "NESTING"),
    ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER", "HOME", "CLEAR", "SMALL", "STAR"),
    ("23235", "WOODEN STAR FROSTED T-LIGHT HOLDER", "HOME", "WOOD", "SMALL", "STAR"),
    ("22745", "POPPYS PLAYHOUSE BEDROOM SET", "TOYS", "POPPY", "MEDIUM", "BEDROOM"),
    (
        "22746",
        "POPPYS PLAYHOUSE LIVINGROOM SET",
        "TOYS",
        "POPPY",
        "MEDIUM",
        "LIVINGROOM",
    ),
    ("22747", "POPPYS PLAYHOUSE KITCHEN SET", "TOYS", "POPPY", "MEDIUM", "KITCHEN"),
    ("22748", "POPPYS PLAYHOUSE BATHROOM SET", "TOYS", "POPPY", "MEDIUM", "BATHROOM"),
    ("22749", "POPPYS PLAYHOUSE HALL SET", "TOYS", "POPPY", "MEDIUM", "HALL"),
    (
        "21212",
        "CERAMIC HEART FAIRY CAKE MONEY BOX",
        "HOME",
        "CERAMIC",
        "SMALL",
        "HEART",
    ),
    ("22423", "REGENCY CAKESTAND 3 TIER", "HOME", "REGENCY", "LARGE", "CAKESTAND"),
    ("84879", "ASSORTED COLOUR BIRD ORNAMENTS", "HOME", "MULTI", "SMALL", "BIRD"),
    (
        "22632",
        "HAND WARMER UNION JACK",
        "ACCESSORIES",
        "UNION JACK",
        "SMALL",
        "HAND WARMER",
    ),
    ("22633", "HAND WARMER RED POLKA DOT", "ACCESSORIES", "RED", "SMALL", "POLKA DOT"),
    ("22868", "LUNCH BAG RED RETROSPOT", "ACCESSORIES", "RED", "MEDIUM", "RETROSPOT"),
    ("22867", "LUNCH BAG BLUE RETROSPOT", "ACCESSORIES", "BLUE", "MEDIUM", "RETROSPOT"),
    ("22382", "VINTAGE CHRISTMAS BUNTING", "SEASONAL", "VINTAGE", "LARGE", "CHRISTMAS"),
    ("22381", "CHRISTMAS BUNTING", "SEASONAL", "MULTI", "LARGE", "CHRISTMAS"),
    ("85162A", "VINTAGE JOKE PASTIES", "SEASONAL", "VINTAGE", "SMALL", "JOKE"),
    ("85162B", "VINTAGE JOKE PASTIES", "SEASONAL", "VINTAGE", "SMALL", "JOKE"),
    ("85162C", "VINTAGE JOKE PASTIES", "SEASONAL", "VINTAGE", "SMALL", "JOKE"),
    ("22720", "BIRTHDAY CANDLES PINK", "SEASONAL", "PINK", "SMALL", "BIRTHDAY"),
    ("22719", "BIRTHDAY CANDLES BLUE", "SEASONAL", "BLUE", "SMALL", "BIRTHDAY"),
    ("22348", "CHILDS GARDEN RAKE BLUE", "TOYS", "BLUE", "SMALL", "GARDEN"),
    ("22349", "CHILDS GARDEN RAKE PINK", "TOYS", "PINK", "SMALL", "GARDEN"),
    ("22345", "CHILDS GARDEN SPADE BLUE", "TOYS", "BLUE", "SMALL", "GARDEN"),
    ("22346", "CHILDS GARDEN SPADE PINK", "TOYS", "PINK", "SMALL", "GARDEN"),
    ("22350", "CHILDS GARDEN FORK BLUE", "TOYS", "BLUE", "SMALL", "GARDEN"),
    ("22351", "CHILDS GARDEN FORK PINK", "TOYS", "PINK", "SMALL", "GARDEN"),
    ("21978", "CHOCOLATE HOT WATER BOTTLE", "HOME", "BROWN", "MEDIUM", "CHOCOLATE"),
    ("21979", "PINK HOT WATER BOTTLE", "HOME", "PINK", "MEDIUM", "HOT WATER BOTTLE"),
    ("22628", "SPACEBOY LUNCH BOX", "ACCESSORIES", "SPACEBOY", "MEDIUM", "LUNCH BOX"),
    ("22629", "SPACEGIRL LUNCH BOX", "ACCESSORIES", "SPACEGIRL", "MEDIUM", "LUNCH BOX"),
    (
        "22622",
        "LUNCH BAG SPACEBOY DESIGN",
        "ACCESSORIES",
        "SPACEBOY",
        "MEDIUM",
        "LUNCH BAG",
    ),
    (
        "22623",
        "LUNCH BAG SPACEGIRL DESIGN",
        "ACCESSORIES",
        "SPACEGIRL",
        "MEDIUM",
        "LUNCH BAG",
    ),
    (
        "22625",
        "LUNCH BAG DOLLY GIRL DESIGN",
        "ACCESSORIES",
        "DOLLY",
        "MEDIUM",
        "LUNCH BAG",
    ),
    (
        "22626",
        "LUNCH BAG WOODY BOY DESIGN",
        "ACCESSORIES",
        "WOODY",
        "MEDIUM",
        "LUNCH BAG",
    ),
    ("22627", "LUNCH BAG APPLE DESIGN", "ACCESSORIES", "APPLE", "MEDIUM", "LUNCH BAG"),
    (
        "22624",
        "LUNCH BAG FLOWER DESIGN",
        "ACCESSORIES",
        "FLOWER",
        "MEDIUM",
        "LUNCH BAG",
    ),
    (
        "84969",
        "PLASTERS IN TIN SPACEBOY",
        "ACCESSORIES",
        "SPACEBOY",
        "SMALL",
        "PLASTERS",
    ),
    (
        "84970",
        "PLASTERS IN TIN SPACEGIRL",
        "ACCESSORIES",
        "SPACEGIRL",
        "SMALL",
        "PLASTERS",
    ),
    ("84971", "PLASTERS IN TIN CIRCUS", "ACCESSORIES", "CIRCUS", "SMALL", "PLASTERS"),
    (
        "84972",
        "PLASTERS IN TIN STRONG MAN",
        "ACCESSORIES",
        "STRONG MAN",
        "SMALL",
        "PLASTERS",
    ),
    ("22620", "WOODEN TRAIN SET", "TOYS", "WOOD", "LARGE", "TRAIN"),
    ("22619", "WOODEN DOLLS HOUSE FURNITURE", "TOYS", "WOOD", "MEDIUM", "FURNITURE"),
    ("22618", "WOODEN DOLLS HOUSE", "TOYS", "WOOD", "LARGE", "HOUSE"),
    ("22617", "WOODEN PLAY FOOD SET", "TOYS", "WOOD", "MEDIUM", "FOOD"),
    ("22616", "WOODEN PLAY KITCHEN SET", "TOYS", "WOOD", "LARGE", "KITCHEN"),
    ("22615", "WOODEN PLAY SHOP SET", "TOYS", "WOOD", "LARGE", "SHOP"),
    ("22614", "WOODEN PLAY TOOL SET", "TOYS", "WOOD", "MEDIUM", "TOOLS"),
    ("22613", "WOODEN PLAY GARDEN SET", "TOYS", "WOOD", "MEDIUM", "GARDEN"),
    ("22612", "WOODEN PLAY DOCTOR SET", "TOYS", "WOOD", "MEDIUM", "DOCTOR"),
    ("22611", "WOODEN PLAY VET SET", "TOYS", "WOOD", "MEDIUM", "VET"),
    ("85099B", "JUMBO BAG RED RETROSPOT", "ACCESSORIES", "RED", "LARGE", "RETROSPOT"),
    ("85099C", "JUMBO BAG PINK POLKADOT", "ACCESSORIES", "PINK", "LARGE", "POLKADOT"),
    ("85099D", "JUMBO BAG BLUE POLKADOT", "ACCESSORIES", "BLUE", "LARGE", "POLKADOT"),
    ("22469", "WOODEN TOY TRAIN SET", "TOYS", "WOOD", "MEDIUM", "TRAIN"),
    ("22470", "WOODEN TOY CAR SET", "TOYS", "WOOD", "MEDIUM", "CAR"),
    ("22471", "WOODEN TOY PLANE SET", "TOYS", "WOOD", "MEDIUM", "PLANE"),
    ("22472", "WOODEN TOY BOAT SET", "TOYS", "WOOD", "MEDIUM", "BOAT"),
    (
        "22099",
        "RETROSPOT CHARACTER LUNCH BAG",
        "ACCESSORIES",
        "RETROSPOT",
        "MEDIUM",
        "LUNCH BAG",
    ),
    (
        "22100",
        "POLKADOT CHARACTER LUNCH BAG",
        "ACCESSORIES",
        "POLKADOT",
        "MEDIUM",
        "LUNCH BAG",
    ),
    (
        "22101",
        "STRIPE CHARACTER LUNCH BAG",
        "ACCESSORIES",
        "STRIPE",
        "MEDIUM",
        "LUNCH BAG",
    ),
    (
        "22102",
        "FLORAL CHARACTER LUNCH BAG",
        "ACCESSORIES",
        "FLORAL",
        "MEDIUM",
        "LUNCH BAG",
    ),
    (
        "22093",
        "RETROSPOT CHARACTER PENCIL CASE",
        "ACCESSORIES",
        "RETROSPOT",
        "SMALL",
        "PENCIL CASE",
    ),
    (
        "22094",
        "POLKADOT CHARACTER PENCIL CASE",
        "ACCESSORIES",
        "POLKADOT",
        "SMALL",
        "PENCIL CASE",
    ),
    (
        "22095",
        "STRIPE CHARACTER PENCIL CASE",
        "ACCESSORIES",
        "STRIPE",
        "SMALL",
        "PENCIL CASE",
    ),
    (
        "22096",
        "FLORAL CHARACTER PENCIL CASE",
        "ACCESSORIES",
        "FLORAL",
        "SMALL",
        "PENCIL CASE",
    ),
    ("85048", "CHOCOLATE BAR", "FOOD", "CHOCOLATE", "SMALL", "BAR"),
    ("85049", "CHOCOLATE BAR", "FOOD", "CHOCOLATE", "SMALL", "BAR"),
    ("85050", "CHOCOLATE BAR", "FOOD", "CHOCOLATE", "SMALL", "BAR"),
    ("85051", "CHOCOLATE BAR", "FOOD", "CHOCOLATE", "SMALL", "BAR"),
    ("85052", "CHOCOLATE BAR", "FOOD", "CHOCOLATE", "SMALL", "BAR"),
    (
        "23084",
        "PACK OF 72 RETROSPOT CAKE CASES",
        "FOOD",
        "RETROSPOT",
        "LARGE",
        "CAKE CASES",
    ),
    (
        "23085",
        "PACK OF 72 POLKADOT CAKE CASES",
        "FOOD",
        "POLKADOT",
        "LARGE",
        "CAKE CASES",
    ),
    ("23086", "PACK OF 72 STRIPE CAKE CASES", "FOOD", "STRIPE", "LARGE", "CAKE CASES"),
    ("23087", "PACK OF 72 FLORAL CAKE CASES", "FOOD", "FLORAL", "LARGE", "CAKE CASES"),
    ("22112", "RETROSPOT CHARACTER APRON", "HOME", "RETROSPOT", "MEDIUM", "APRON"),
    ("22113", "POLKADOT CHARACTER APRON", "HOME", "POLKADOT", "MEDIUM", "APRON"),
    ("22114", "STRIPE CHARACTER APRON", "HOME", "STRIPE", "MEDIUM", "APRON"),
    ("22115", "FLORAL CHARACTER APRON", "HOME", "FLORAL", "MEDIUM", "APRON"),
    ("84826", "MAGNETIC CALENDAR", "HOME", "MULTI", "MEDIUM", "CALENDAR"),
    ("84827", "MAGNETIC CALENDAR", "HOME", "MULTI", "MEDIUM", "CALENDAR"),
    ("84828", "MAGNETIC CALENDAR", "HOME", "MULTI", "MEDIUM", "CALENDAR"),
    ("84829", "MAGNETIC CALENDAR", "HOME", "MULTI", "MEDIUM", "CALENDAR"),
    ("22636", "CHILDRENS CUTLERY SPACEBOY", "HOME", "SPACEBOY", "SMALL", "CUTLERY"),
    ("22637", "CHILDRENS CUTLERY SPACEGIRL", "HOME", "SPACEGIRL", "SMALL", "CUTLERY"),
    ("22638", "CHILDRENS CUTLERY DOLLY GIRL", "HOME", "DOLLY", "SMALL", "CUTLERY"),
    ("22639", "CHILDRENS CUTLERY WOODY BOY", "HOME", "WOODY", "SMALL", "CUTLERY"),
    ("84187", "RED RETROSPOT MUG", "HOME", "RED", "MEDIUM", "MUG"),
    ("84188", "BLUE RETROSPOT MUG", "HOME", "BLUE", "MEDIUM", "MUG"),
    ("84189", "PINK POLKADOT MUG", "HOME", "PINK", "MEDIUM", "MUG"),
    ("84190", "BLUE POLKADOT MUG", "HOME", "BLUE", "MEDIUM", "MUG"),
    ("22463", "GREEN RETROSPOT MUG", "HOME", "GREEN", "MEDIUM", "MUG"),
    ("22464", "YELLOW RETROSPOT MUG", "HOME", "YELLOW", "MEDIUM", "MUG"),
    ("22465", "ORANGE RETROSPOT MUG", "HOME", "ORANGE", "MEDIUM", "MUG"),
    ("22466", "PURPLE RETROSPOT MUG", "HOME", "PURPLE", "MEDIUM", "MUG"),
    ("84077", "SET/20 RED RETROSPOT PAPER NAPKINS", "HOME", "RED", "SMALL", "NAPKINS"),
    (
        "84078",
        "SET/20 BLUE RETROSPOT PAPER NAPKINS",
        "HOME",
        "BLUE",
        "SMALL",
        "NAPKINS",
    ),
    ("84079", "SET/20 PINK POLKADOT PAPER NAPKINS", "HOME", "PINK", "SMALL", "NAPKINS"),
    ("84080", "SET/20 BLUE POLKADOT PAPER NAPKINS", "HOME", "BLUE", "SMALL", "NAPKINS"),
    (
        "22455",
        "SET/20 GREEN RETROSPOT PAPER NAPKINS",
        "HOME",
        "GREEN",
        "SMALL",
        "NAPKINS",
    ),
    (
        "22456",
        "SET/20 YELLOW RETROSPOT PAPER NAPKINS",
        "HOME",
        "YELLOW",
        "SMALL",
        "NAPKINS",
    ),
    (
        "22457",
        "SET/20 ORANGE RETROSPOT PAPER NAPKINS",
        "HOME",
        "ORANGE",
        "SMALL",
        "NAPKINS",
    ),
    (
        "22458",
        "SET/20 PURPLE RETROSPOT PAPER NAPKINS",
        "HOME",
        "PURPLE",
        "SMALL",
        "NAPKINS",
    ),
    (
        "22459",
        "SET/20 WHITE RETROSPOT PAPER NAPKINS",
        "HOME",
        "WHITE",
        "SMALL",
        "NAPKINS",
    ),
    (
        "22460",
        "SET/20 BLACK RETROSPOT PAPER NAPKINS",
        "HOME",
        "BLACK",
        "SMALL",
        "NAPKINS",
    ),
    ("84081", "SET/20 RED RETROSPOT PAPER PLATES", "HOME", "RED", "SMALL", "PLATES"),
    ("84082", "SET/20 BLUE RETROSPOT PAPER PLATES", "HOME", "BLUE", "SMALL", "PLATES"),
    ("84083", "SET/20 PINK POLKADOT PAPER PLATES", "HOME", "PINK", "SMALL", "PLATES"),
    ("84084", "SET/20 BLUE POLKADOT PAPER PLATES", "HOME", "BLUE", "SMALL", "PLATES"),
    (
        "22461",
        "SET/20 GREEN RETROSPOT PAPER PLATES",
        "HOME",
        "GREEN",
        "SMALL",
        "PLATES",
    ),
    (
        "22462",
        "SET/20 YELLOW RETROSPOT PAPER PLATES",
        "HOME",
        "YELLOW",
        "SMALL",
        "PLATES",
    ),
    ("85230", "GIFT WRAP RETROSPOT", "HOME", "RETROSPOT", "LARGE", "GIFT WRAP"),
    ("85231", "GIFT WRAP POLKADOT", "HOME", "POLKADOT", "LARGE", "GIFT WRAP"),
    ("85232", "GIFT WRAP STRIPE", "HOME", "STRIPE", "LARGE", "GIFT WRAP"),
    ("85233", "GIFT WRAP FLORAL", "HOME", "FLORAL", "LARGE", "GIFT WRAP"),
    ("22467", "GIFT WRAP GREEN RETROSPOT", "HOME", "GREEN", "LARGE", "GIFT WRAP"),
    ("22468", "GIFT WRAP YELLOW RETROSPOT", "HOME", "YELLOW", "LARGE", "GIFT WRAP"),
]


def _infer_brand(name: str) -> str:
    """Infer brand from product name."""
    brands = [
        "POPPYS",
        "SPACEBOY",
        "SPACEGIRL",
        "DOLLY",
        "WOODY",
        "RETROSPOT",
        "POLKADOT",
        "FLORAL",
        "STRIPE",
        "VINTAGE",
        "REGENCY",
        "UNION JACK",
    ]
    for brand in brands:
        if brand in name.upper():
            return brand.title() if brand != "UNION JACK" else "Union Jack"
    return "Generic"


def _infer_size(name: str) -> str:
    """Infer size from product name."""
    name_upper = name.upper()
    if any(
        w in name_upper
        for w in [
            "JUMBO",
            "LARGE",
            "SET/72",
            "PACK OF 72",
            "3 TIER",
            "HOUSE",
            "KITCHEN SET",
            "SHOP SET",
            "TRAIN SET",
            "DOLLS HOUSE",
        ]
    ):
        return "LARGE"
    elif any(
        w in name_upper
        for w in [
            "MEDIUM",
            "LUNCH BOX",
            "LUNCH BAG",
            "CALENDAR",
            "APRON",
            "CAKESTAND",
            "MUG",
            "HOT WATER BOTTLE",
            "CUTLERY",
            "BEDROOM",
            "LIVINGROOM",
            "BATHROOM",
            "HALL",
            "FURNITURE",
            "GARDEN SET",
            "TOOL SET",
            "DOCTOR SET",
            "VET SET",
        ]
    ):
        return "MEDIUM"
    else:
        return "SMALL"


def _infer_flavor(name: str) -> str:
    """Infer flavor/design variant from product name."""
    name_upper = name.upper()
    variants = {
        "HEART": "HEART",
        "STAR": "STAR",
        "BIRD": "BIRD",
        "UNION FLAG": "UNION FLAG",
        "UNION JACK": "UNION JACK",
        "POLKA DOT": "POLKA DOT",
        "POLKADOT": "POLKADOT",
        "RETROSPOT": "RETROSPOT",
        "FLORAL": "FLORAL",
        "STRIPE": "STRIPE",
        "VINTAGE": "VINTAGE",
        "CHRISTMAS": "CHRISTMAS",
        "BIRTHDAY": "BIRTHDAY",
        "GARDEN": "GARDEN",
        "WOOLLY": "WOOLLY",
        "CHOCOLATE": "CHOCOLATE",
        "SPACEBOY": "SPACEBOY",
        "SPACEGIRL": "SPACEGIRL",
        "DOLLY": "DOLLY",
        "WOODY": "WOODY",
        "APPLE": "APPLE",
        "FLOWER": "FLOWER",
        "CIRCUS": "CIRCUS",
        "STRONG MAN": "STRONG MAN",
        "TRAIN": "TRAIN",
        "CAR": "CAR",
        "PLANE": "PLANE",
        "BOAT": "BOAT",
        "BEDROOM": "BEDROOM",
        "LIVINGROOM": "LIVINGROOM",
        "KITCHEN": "KITCHEN",
        "BATHROOM": "BATHROOM",
        "HALL": "HALL",
        "FOOD": "FOOD",
        "CALENDAR": "CALENDAR",
        "NAPKINS": "NAPKINS",
        "PLATES": "PLATES",
        "GIFT WRAP": "GIFT WRAP",
        "CUTLERY": "CUTLERY",
        "MUG": "MUG",
        "CAKE CASES": "CAKE CASES",
        "CANDLES": "CANDLES",
        "RAKE": "RAKE",
        "SPADE": "SPADE",
        "FORK": "FORK",
        "PLASTERS": "PLASTERS",
        "LANTERN": "LANTERN",
        "T-LIGHT": "T-LIGHT",
        "COAT HANGER": "COAT HANGER",
        "MONEY BOX": "MONEY BOX",
        "ORNAMENTS": "ORNAMENTS",
        "HAND WARMER": "HAND WARMER",
        "BUNTING": "BUNTING",
        "JOKE": "JOKE",
        "APRON": "APRON",
        "BAR": "BAR",
        "PENCIL CASE": "PENCIL CASE",
        "LUNCH BOX": "LUNCH BOX",
        "LUNCH BAG": "LUNCH BAG",
    }
    for key, val in variants.items():
        if key in name_upper:
            return val
    return "STANDARD"


def _infer_category(name: str) -> str:
    """Infer category from product name."""
    name_upper = name.upper()
    if any(
        w in name_upper
        for w in [
            "T-LIGHT",
            "LANTERN",
            "HANGER",
            "MONEY BOX",
            "CAKESTAND",
            "ORNAMENTS",
            "BUNTING",
            "NAPKINS",
            "PLATES",
            "GIFT WRAP",
            "CALENDAR",
            "MUG",
            "HOT WATER BOTTLE",
            "CUTLERY",
            "APRON",
            "COAT HANGER",
        ]
    ):
        return "HOME"
    elif any(
        w in name_upper
        for w in [
            "TOY",
            "PLAY",
            "DOLLS",
            "TRAIN",
            "CAR",
            "PLANE",
            "BOAT",
            "GARDEN",
            "TOOL",
            "DOCTOR",
            "VET",
            "HOUSE",
            "KITCHEN",
            "SHOP",
            "FURNITURE",
            "RAKE",
            "SPADE",
            "FORK",
            "NESTING",
        ]
    ):
        return "TOYS"
    elif any(
        w in name_upper
        for w in [
            "LUNCH",
            "BAG",
            "CASE",
            "WARMER",
            "CUTLERY",
            "PENCIL",
            "BIRTHDAY",
            "CANDLES",
            "CHRISTMAS",
            "VINTAGE",
            "UNION JACK",
            "JOKE",
        ]
    ):
        return "ACCESSORIES"
    elif any(w in name_upper for w in ["CHOCOLATE", "CAKE", "BAR", "PACK", "FOOD"]):
        return "FOOD"
    else:
        return "HOME"


def generate_transactions(
    n_transactions: int = 1000,
    n_customers: int = 200,
    n_products: int = 100,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
    include_attributes: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic transaction data matching the exact schema:
    date, transaction_id, stockcode, product, customer_id, price, quantity
    (+ optional: category, brand, size, flavor)
    """
    np.random.seed(seed)
    random.seed(seed)

    # Select subset of products
    products = PRODUCT_CATALOG[:n_products]
    stockcodes = [p[0] for p in products]
    product_names = [p[1] for p in products]
    product_categories = [p[2] for p in products]
    product_brands = [p[3] for p in products]
    product_sizes = [p[4] for p in products]
    product_flavors = [p[5] for p in products]

    # Customer IDs
    customer_ids = [f"CUST{i:04d}" for i in range(1, n_customers + 1)]

    # Date range
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    date_range = (end - start).days

    # Product prices (realistic range)
    base_prices = {code: round(np.random.uniform(1.5, 25.0), 2) for code in stockcodes}

    # Customer preferences (some customers prefer certain product types)
    customer_prefs = {}
    for cust in customer_ids:
        # Each customer has affinity for 5-15 products
        n_pref = np.random.randint(5, 15)
        pref_products = np.random.choice(stockcodes, n_pref, replace=False)
        customer_prefs[cust] = set(pref_products)

    transactions = []
    transaction_id = 1

    for _ in range(n_transactions):
        # Random date
        days_offset = np.random.randint(0, date_range + 1)
        trans_date = start + timedelta(days=int(days_offset))

        # Random customer
        customer = np.random.choice(customer_ids)

        # Number of items in basket (1-8, weighted toward smaller)
        n_items = np.random.choice(range(1, 9), p=[0.3, 0.25, 0.2, 0.1, 0.07, 0.04, 0.02, 0.02])

        # Select products for this basket
        # Bias toward customer preferences
        pref = customer_prefs[customer]
        if len(pref) > 0 and np.random.random() < 0.7:
            # 70% chance to include preferred products
            n_pref = min(len(pref), np.random.randint(1, min(4, len(pref) + 1)))
            selected_pref = np.random.choice(list(pref), n_pref, replace=False)
            remaining = n_items - n_pref
            other_products = [p for p in stockcodes if p not in pref]
            if remaining > 0 and len(other_products) > 0:
                selected_other = np.random.choice(
                    other_products, min(remaining, len(other_products)), replace=False
                )
                basket_items = list(selected_pref) + list(selected_other)
            else:
                basket_items = list(selected_pref)
        else:
            basket_items = np.random.choice(
                stockcodes, min(n_items, len(stockcodes)), replace=False
            )

        # Create transaction rows
        for stockcode in basket_items:
            quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.6, 0.2, 0.1, 0.07, 0.03])
            price = base_prices[stockcode] * np.random.uniform(0.9, 1.1)  # Small price variation
            price = round(price, 2)

            idx = stockcodes.index(stockcode)
            product_name = product_names[idx]

            row = {
                "date": trans_date.strftime("%Y-%m-%d"),
                "transaction_id": f"INV{transaction_id:06d}",
                "stockcode": stockcode,
                "product": product_name,
                "customer_id": customer,
                "price": price,
                "quantity": quantity,
            }

            if include_attributes:
                row["category"] = product_categories[idx]
                row["brand"] = product_brands[idx]
                row["size"] = product_sizes[idx]
                row["flavor"] = product_flavors[idx]

            transactions.append(row)

        transaction_id += 1

    df = pd.DataFrame(transactions)
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
