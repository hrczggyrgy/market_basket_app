"""Market Basket Analytics — Customer Decision Intelligence.

Core analytics modules for market basket analysis, customer segmentation,
demand transference, assortment optimization, pricing elasticity, and more.
"""

# Module imports for public API
from . import (
    assortment,
    basket_metrics,
    category,
    cdt,
    choice_model,
    clv,
    cohort,
    copurchase,
    data,
    performance,
    pricing,
    promo,
    rules,
    segmentation,
    switching,
    transference,
)
from .config import *
from .data_quality import *
from .reliability import *
from .schemas import *
from .validation import *

__version__ = "2.0.0"

__all__ = [
    # Config
    "AnalyticsConfig", "DEFAULT_CONFIG", "get_config", "update_config", "set_global_config",
    # Data Quality
    "DataQualityError", "DataQualityReport", "assess_data_quality", "generate_quality_summary",
    # Reliability
    "ReliabilityLevel", "compute_reliability",
    # Schemas
    "ABC_CLASSES", "ADDON_RECS", "AFFINITY_PAIRS", "ASSORTMENT_EFFICIENCY", "ASSORTMENT_EVALUATION",
    "ASSORTMENT_SCENARIO", "ASSORTMENT_SOLUTION", "BASKET_COMPOSITION", "BASKET_OVER_TIME",
    "BASKET_PENETRATION", "BEHAVIORAL_FEATURES", "BEHAVIORAL_SEGMENTS", "CATEGORY_KPIS",
    "CATEGORY_MANAGER_SCORECARD", "CATEGORY_PROMO_TIMELINE", "CATEGORY_SCORECARD",
    "CATEGORY_SWITCHING", "CATEGORY_TREND", "CAUSAL_UPLIFT", "CDT_ASSIGNMENTS", "CDT_ATTRIBUTES",
    "CDT_COMMUNITY", "CDT_OPTIMAL_K", "CDT_QUALITY", "CDT_TREE_NODES", "CDT_TREE_PRODUCTS",
    "CDT_TREE_SCORE", "CDT_VALIDATION", "CLUSTER_QUALITY", "CLUSTER_STABILITY", "CLV_CUSTOMER",
    "CLV_DIAGNOSTICS", "CLV_PREDICTIONS", "COHORT_DECAY", "COHORT_LTV", "COHORT_RETENTION",
    "COHORT_SIZES", "CROSS_ELASTICITY", "CUSTOMER_ENTROPY", "CUSTOMER_FEATURES", "DELIST_IMPACT",
    "DEMAND_TRANSFERENCE", "ELASTICITY", "FREQUENT_ITEMSETS", "HIERARCHICAL_ELASTICITY",
    "HIGH_VALUE_SWITCHING", "INFERRED_CATEGORIES", "IPT_CV", "IV_ELASTICITY", "KVI_SCORES",
    "LIFECYCLE", "LOYALTY_METRICS", "MODEL_METRICS", "NODE_DELIST_IMPACT", "POP_COMPARISON",
    "PRICE_CURVE_1D", "PRICE_CURVE_MULTI", "PRODUCT_METRICS", "PRODUCT_VELOCITY", "PROMO_BASELINE",
    "PROMO_HALO", "PROMO_LIFT", "PROMO_PERIODS", "PROMO_ROI", "PROMO_TIMING_DOW",
    "PROMO_TIMING_MONTH", "PROMO_WATERFALL", "QINI_CURVE", "RDD_ELASTICITY", "RECOVERY_HHI",
    "REPEAT_RATE", "RFM_FEATURES", "RFM_SEGMENTS", "RULES", "RULES_TABLE", "SDP_SCORES",
    "SECOND_PURCHASE", "SKU_RATIONALIZATION", "SURVIVAL_DIAGNOSTICS", "SURVIVAL_PREDICTIONS",
    "SWITCHING_MATRIX", "SWITCHING_MATRIX_CI", "SWITCHING_OPPORTUNITY", "SWITCHING_SUBSTITUTION",
    "SYNTHETIC_CONTROL", "CAUSAL_UPLIFT", "TRANSACTIONS", "TRANSFERENCE_CI", "TREE_RULES",
    "UPLIFT_METRICS", "UPLIFT_PROPENSITY", "UPLIFT_SCORES", "VALUE_BASED_SEGMENTS", "XYZ_CLASSES",
    "YOY_COMPARISON", "DataContract", "EmptyResult", "SchemaError", "ValueValidator",
    "check", "contract", "is_empty_result", "make_empty_result", "validate_referential_integrity",
    # Modules
    "assortment", "basket_metrics", "category", "cdt", "choice_model", "clv", "cohort",
    "copurchase", "data", "performance", "pricing", "promo", "rules", "segmentation",
    "switching", "transference",
]
