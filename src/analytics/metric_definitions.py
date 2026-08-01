"""Canonical metric definitions, denominators, and user-facing help text.

Single source of truth for metric semantics across all tabs.
Calculations remain in their respective analytics modules.
"""

from typing import Dict, List


METRIC_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "revenue": {
        "label": "Revenue",
        "definition": "Sum of (price × quantity) across all transaction lines.",
        "denominator": "N/A (absolute sum)",
        "caveat": "Includes negative quantities (returns) as revenue reductions. No separate return identification.",
        "help": "Total sales value. Negative quantities reduce revenue. Returns not separately flagged.",
    },
    "basket": {
        "label": "Basket (Trip)",
        "definition": "Unique transaction_id if present; otherwise (customer_id, date) pair as proxy.",
        "denominator": "N/A (count of unique baskets)",
        "caveat": "Same-day multi-trip customers conflated if transaction_id missing.",
        "help": "A shopping trip. Falls back to customer+date when transaction_id unavailable.",
    },
    "basket_penetration": {
        "label": "Basket Penetration",
        "definition": "Baskets containing the product ÷ Total baskets.",
        "denominator": "Total unique baskets in period",
        "caveat": "Trip-level reach. Not a customer-level metric.",
        "help": "Fraction of shopping trips that include this product. Trip incidence alias.",
    },
    "shopper_penetration": {
        "label": "Shopper Penetration",
        "definition": "Unique customers buying the product ÷ Total unique customers.",
        "denominator": "Total unique customers in period",
        "caveat": "Customer-level reach. Differs from basket penetration for multi-trip buyers.",
        "help": "Fraction of shoppers who purchased this product at least once.",
    },
    "repeat_rate": {
        "label": "Repeat Rate",
        "definition": "Customers with 2+ purchase dates for the product ÷ Total customers buying the product.",
        "denominator": "Unique buyers of the product",
        "caveat": "Based on purchase occasions (dates), not transaction lines. Returns not excluded.",
        "help": "Share of buyers who came back for the same product. Higher = more habitual.",
    },
    "asp": {
        "label": "Average Selling Price (ASP)",
        "definition": "Mean of price across all transaction lines for the product.",
        "denominator": "Transaction lines for the product",
        "caveat": "Promotion-contaminated. Use median_price for baseline pricing decisions.",
        "help": "Mean unit price paid. Includes promotional prices. Median is more robust baseline.",
    },
    "price_cv": {
        "label": "Price Coefficient of Variation",
        "definition": "Standard deviation of price ÷ Mean price (per product, across time).",
        "denominator": "Mean price of the product",
        "caveat": "Low CV (<3–5%) makes elasticity estimation unreliable.",
        "help": "Price variability measure. Higher = more price changes observed. Needed for elasticity.",
    },
    "revenue_trend_index": {
        "label": "Revenue Trend Index",
        "definition": "Revenue in most recent 4 weeks ÷ Revenue in prior 4 weeks.",
        "denominator": "Revenue in prior 4-week window",
        "caveat": "Hardcoded 4-week windows. Seasonal effects not adjusted.",
        "help": ">1 = growing, <1 = declining. Short-term momentum indicator.",
    },
    "inferred_promotion": {
        "label": "Inferred Promotion Flag",
        "definition": "Price drop > threshold (default 15%) sustained for min/max duration, with volume lift.",
        "denominator": "N/A (binary flag per product-week)",
        "caveat": "No true promo calendar. Confounds clearance, markdowns, data errors. Not validated.",
        "help": "Heuristic promo detection from price/volume patterns only. Not ground truth.",
    },
    "elasticity": {
        "label": "Price Elasticity of Demand",
        "definition": "d log(quantity) / d log(price) estimated via log-log regression (weekly aggregates).",
        "denominator": "Weekly price variation (coefficient of variation)",
        "caveat": "Observational. Confounded by promotions, seasonality, stockouts. Not causal.",
        "help": "Elastic (< -1): demand sensitive. Inelastic (-1 to 0): demand stable. Positive: likely bias.",
    },
    "transfer_rate": {
        "label": "Demand Transfer Rate",
        "definition": "Switching probability from product A → B × Revenue share of A.",
        "denominator": "Revenue of delisted product",
        "caveat": "Assumes observed switching = substitution. No causal evidence. No control for availability.",
        "help": "Estimated fraction of delisted demand transferring to alternatives. Scenario only.",
    },
    "basket_value_uplift": {
        "label": "Basket Value Uplift",
        "definition": "Mean basket value WHEN product present − Mean basket value WHEN absent.",
        "denominator": "Mean basket value without product",
        "caveat": "Associative only. Confounded by basket size, trip type, shopper segments.",
        "help": "Halo metric: products associated with higher-value trips. Not causal incrementality.",
    },
    "copurchase_index": {
        "label": "Co-purchase Index",
        "definition": "(P(A∩B) / (P(A)×P(B))) × 100. Index > 100 = above-average pairing.",
        "denominator": "Expected co-occurrence under independence",
        "caveat": "Symmetric affinity. Not directional. High index ≠ bundle causality.",
        "help": "Indexed lift. >180 strong complement (Circana). <50 potential substitutes or unrelated.",
    },
    "kvi_score": {
        "label": "KVI Score (Internal Proxy)",
        "definition": "XGBoost feature importance of price in demand model, or RFM+Elasticity composite.",
        "denominator": "N/A (model-derived score)",
        "caveat": "Internal transaction-data proxy only. No competitor prices, recall surveys, or market share.",
        "help": "Identifies items where price changes correlate with demand shifts in YOUR data. Not market KVI.",
    },
}


METRIC_GROUPS: Dict[str, List[str]] = {
    "core": ["revenue", "basket", "basket_penetration", "shopper_penetration"],
    "loyalty": ["repeat_rate", "reorder_interval_days", "time_to_second_purchase_days"],
    "pricing": ["asp", "price_cv", "elasticity", "kvi_score"],
    "assortment": ["revenue_trend_index", "transfer_rate", "basket_value_uplift", "copurchase_index"],
    "promotional": ["inferred_promotion"],
}


def get_metric_definition(metric_key: str) -> Dict[str, str]:
    """Get full definition for a metric key."""
    return METRIC_DEFINITIONS.get(metric_key, {
        "label": metric_key.replace("_", " ").title(),
        "definition": "No definition available.",
        "denominator": "N/A",
        "caveat": "Metric not in canonical registry.",
        "help": "No help text available.",
    })


def get_metric_help(metric_key: str) -> str:
    """Get concise help text for tooltip."""
    return get_metric_definition(metric_key).get("help", "")


def get_metric_caveat(metric_key: str) -> str:
    """Get caveat/limitation text."""
    return get_metric_definition(metric_key).get("caveat", "")


def get_metric_label(metric_key: str) -> str:
    """Get display label."""
    return get_metric_definition(metric_key).get("label", metric_key)


def list_metrics_by_group(group: str) -> List[str]:
    """List metric keys in a group."""
    return METRIC_GROUPS.get(group, [])


def all_metric_keys() -> List[str]:
    """All registered metric keys."""
    return list(METRIC_DEFINITIONS.keys())