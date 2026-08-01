"""Lightweight attribute extraction from product text using regex/dictionary heuristics.

This module provides simple rule-based parsing from product descriptions.
No heavy ML dependencies - uses regex and dictionary lookups only.
"""

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


# Common brand patterns - can be extended
BRAND_PATTERNS = [
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Title case words
]

# Size/volume patterns
SIZE_PATTERNS = [
    (r'(\d+(?:\.\d+)?)\s*(ML|ml|L|l|G|g|KG|kg|OZ|oz|LB|lb)', 'volume'),
    (r'(\d+(?:\.\d+)?)\s*(PCS|pcs|PK|pk|CT|ct)', 'count'),
    (r'(\d+)\s*[xX]\s*(\d+(?:\.\d+)?)\s*(ML|ml|L|l|G|g)', 'multipack'),
]

# Flavor/variant patterns
FLAVOR_KEYWORDS = [
    'vanilla', 'chocolate', 'strawberry', 'mint', 'lemon', 'orange',
    'apple', 'blueberry', 'raspberry', 'cherry', 'grape', 'peach',
    'caramel', 'hazelnut', 'almond', 'coconut', 'coffee', 'mocha',
    'original', 'classic', 'plain', 'natural', 'unsweetened',
    'sweetened', 'light', 'dark', 'white', 'milk', 'cream',
    'cheese', 'onion', 'garlic', 'herb', 'spicy', 'mild',
    'bbq', 'sour', 'salt', 'vinegar', 'pepper', 'chili',
]

# Unit patterns
UNIT_PATTERNS = {
    'volume': ['ML', 'ml', 'L', 'l', 'OZ', 'oz', 'GAL', 'gal'],
    'weight': ['G', 'g', 'KG', 'kg', 'LB', 'lb', 'OZ', 'oz'],
    'count': ['PCS', 'pcs', 'PK', 'pk', 'CT', 'ct', 'EA', 'ea'],
}


def extract_brand(product_text: str, enable: bool = True) -> Optional[str]:
    """Extract brand from product text using simple heuristics.
    
    Args:
        product_text: Product description text
        enable: Whether to attempt brand extraction (default True)
    """
    if not enable or not product_text or not isinstance(product_text, str):
        return None
    
    # Try to find brand-like patterns (Title Case words at start)
    text = product_text.strip()
    
    # Common pattern: Brand Name + Product Description
    # e.g., "Coca Cola Zero Sugar" -> "Coca Cola"
    words = text.split()
    if len(words) >= 2:
        # Check if first 1-3 words look like a brand (Title Case)
        for i in range(min(3, len(words))):
            candidate = ' '.join(words[:i+1])
            if candidate[0].isupper() and all(w[0].isupper() or w.islower() for w in candidate.split()):
                # Simple heuristic: brand is usually first 1-3 words
                if i == 0 or (i > 0 and words[i][0].isupper()):
                    return candidate
    
    return words[0] if words else None


def extract_pack_size(product_text: str, enable: bool = True) -> Tuple[Optional[float], Optional[str]]:
    """Extract pack size and unit from product text.
    
    Args:
        product_text: Product description text
        enable: Whether to attempt pack size extraction (default True)
    
    Returns:
        Tuple of (size_value, unit) where unit is one of 'volume', 'weight', 'count'
    """
    if not enable or not product_text or not isinstance(product_text, str):
        return None, None
    
    text = product_text.upper()
    
    # Check for multipack pattern first (e.g., "6 x 330ML")
    multipack_match = re.search(r'(\d+)\s*[xX]\s*(\d+(?:\.\d+)?)\s*(ML|ML|L|G|G|KG|KG)', text)
    if multipack_match:
        count = float(multipack_match.group(1))
        size = float(multipack_match.group(2))
        unit = multipack_match.group(3).upper()
        total = count * size
        if unit in ['ML', 'L']:
            return total, 'volume'
        elif unit in ['G', 'KG']:
            return total, 'weight'
    
    # Standard size patterns
    for pattern, unit_type in SIZE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            try:
                value = float(match.group(1))
                unit = match.group(2).upper()
                if unit in ['ML', 'L']:
                    return value, 'volume'
                elif unit in ['G', 'KG']:
                    return value, 'weight'
                elif unit in ['PCS', 'PC', 'PK', 'CT', 'EA']:
                    return value, 'count'
            except (ValueError, IndexError):
                continue
    
    return None, None


def extract_unit(product_text: str, enable: bool = True) -> Optional[str]:
    """Extract unit of measure from product text.
    
    Args:
        product_text: Product description text
        enable: Whether to attempt unit extraction (default True)
    """
    if not enable or not product_text or not isinstance(product_text, str):
        return None
    
    text = product_text.upper()
    
    for unit_type, units in UNIT_PATTERNS.items():
        for unit in units:
            if re.search(rf'\b{re.escape(unit)}\b', text):
                return unit_type
    
    return None


def extract_flavor_variant(product_text: str, enable: bool = True) -> Optional[str]:
    """Extract flavor/variant from product text.
    
    Args:
        product_text: Product description text
        enable: Whether to attempt flavor extraction (default True)
    """
    if not enable or not product_text or not isinstance(product_text, str):
        return None
    
    text = product_text.lower()
    
    for flavor in FLAVOR_KEYWORDS:
        # Word boundary search
        if re.search(rf'\b{re.escape(flavor)}\b', text):
            return flavor.title()
    
    return None


def extract_attributes_from_product_text(
    product_text: str,
    enable_brand: bool = True,
    enable_flavor: bool = True,
    enable_size: bool = True,
    enable_unit: bool = True,
) -> Dict[str, Optional[str]]:
    """Extract all attributes from product text.
    
    Args:
        product_text: Product description text
        enable_brand: Whether to extract brand (default True)
        enable_flavor: Whether to extract flavor/variant (default True)
        enable_size: Whether to extract pack size (default True)
        enable_unit: Whether to extract unit type (default True)
    
    Returns:
        Dict with keys: brand, pack_size, pack_unit, flavor_variant, unit_type
    """
    return {
        'brand': extract_brand(product_text, enable=enable_brand),
        'pack_size': extract_pack_size(product_text, enable=enable_size)[0],
        'pack_unit': extract_pack_size(product_text, enable=enable_size)[1],
        'flavor_variant': extract_flavor_variant(product_text, enable=enable_flavor),
        'unit_type': extract_unit(product_text, enable=enable_unit),
    }


def enrich_products_df(
    products_df: pd.DataFrame,
    product_col: str = 'product',
    brand_col: str = 'brand',
    size_col: str = 'size',
    flavor_col: str = 'flavor',
    category_col: str = 'category'
) -> pd.DataFrame:
    """Enrich products DataFrame with derived attributes from product text.
    
    Only fills missing values - never overwrites existing columns.
    Adds columns with 'derived_' prefix to distinguish from provided data.
    """
    df = products_df.copy()
    
    # Extract attributes for each product
    derived_attrs = df[product_col].apply(extract_attributes_from_product_text)
    
    # Add derived columns (only where missing)
    if 'derived_brand' not in df.columns:
        df['derived_brand'] = derived_attrs.apply(lambda x: x['brand'])
    
    if 'derived_pack_size' not in df.columns:
        df['derived_pack_size'] = derived_attrs.apply(lambda x: x['pack_size'])
    
    if 'derived_pack_unit' not in df.columns:
        df['derived_pack_unit'] = derived_attrs.apply(lambda x: x['pack_unit'])
    
    if 'derived_flavor' not in df.columns:
        df['derived_flavor'] = derived_attrs.apply(lambda x: x['flavor_variant'])
    
    if 'derived_unit_type' not in df.columns:
        df['derived_unit_type'] = derived_attrs.apply(lambda x: x['unit_type'])
    
    # Attribute source tracking
    df['attribute_source'] = df.apply(
        lambda row: _determine_attribute_source(
            row, brand_col, size_col, flavor_col
        ), axis=1
    )
    
    return df


def _determine_attribute_source(
    row: pd.Series,
    brand_col: str,
    size_col: str,
    flavor_col: str
) -> str:
    """Determine whether attributes came from provided columns or were derived."""
    sources = []
    
    if brand_col in row.index and pd.notna(row[brand_col]):
        sources.append('brand_provided')
    elif 'derived_brand' in row.index and pd.notna(row.get('derived_brand')):
        sources.append('brand_derived')
    else:
        sources.append('brand_missing')
    
    if size_col in row.index and pd.notna(row[size_col]):
        sources.append('size_provided')
    elif 'derived_pack_size' in row.index and pd.notna(row.get('derived_pack_size')):
        sources.append('size_derived')
    else:
        sources.append('size_missing')
    
    if flavor_col in row.index and pd.notna(row[flavor_col]):
        sources.append('flavor_provided')
    elif 'derived_flavor' in row.index and pd.notna(row.get('derived_flavor')):
        sources.append('flavor_derived')
    else:
        sources.append('flavor_missing')
    
    return ' | '.join(sources)


def get_attribute_coverage(df: pd.DataFrame) -> Dict[str, Dict]:
    """Get coverage statistics for product attributes."""
    coverage = {}
    
    attr_cols = [
        ('brand', 'derived_brand'),
        ('size', 'derived_pack_size'),
        ('flavor', 'derived_flavor'),
        ('unit', 'derived_unit_type'),
    ]
    
    for provided, derived in attr_cols:
        provided_count = df[provided].notna().sum() if provided in df.columns else 0
        derived_count = df[derived].notna().sum() if derived in df.columns else 0
        total = len(df)
        
        coverage[provided] = {
            'provided_count': int(provided_count),
            'derived_count': int(derived_count),
            'total': int(total),
            'coverage_pct': round((provided_count + derived_count) / total * 100, 1) if total > 0 else 0,
            'provided_pct': round(provided_count / total * 100, 1) if total > 0 else 0,
            'derived_pct': round(derived_count / total * 100, 1) if total > 0 else 0,
        }
    
    return coverage


# Re-export for convenience
__all__ = [
    'extract_brand',
    'extract_pack_size',
    'extract_unit',
    'extract_flavor_variant',
    'extract_attributes_from_product_text',
    'enrich_products_df',
    'get_attribute_coverage',
]