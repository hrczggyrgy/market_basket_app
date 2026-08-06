"""UI tabs package for Market Basket Intelligence."""

from .overview import MODE_SPEC as OVERVIEW_MODE_SPEC
from .rules import MODE_SPEC as RULES_MODE_SPEC
from .copurchase import MODE_SPEC as COPURCHASE_MODE_SPEC
from .switching import MODE_SPEC as SWITCHING_MODE_SPEC
from .cohorts import MODE_SPEC as COHORTS_MODE_SPEC
from .performance import MODE_SPEC as PERFORMANCE_MODE_SPEC
from .cdt_page import MODE_SPEC as CDT_MODE_SPEC
from .segmentation import MODE_SPEC as SEGMENTATION_MODE_SPEC
from .pricing_page import MODE_SPEC as PRICING_MODE_SPEC
from .promo_page import MODE_SPEC as PROMO_MODE_SPEC
from .assortment_page import MODE_SPEC as ASSORTMENT_MODE_SPEC
from .clv_page import MODE_SPEC as CLV_MODE_SPEC

__all__ = [
    "OVERVIEW_MODE_SPEC",
    "RULES_MODE_SPEC",
    "COPURCHASE_MODE_SPEC",
    "SWITCHING_MODE_SPEC",
    "COHORTS_MODE_SPEC",
    "PERFORMANCE_MODE_SPEC",
    "CDT_MODE_SPEC",
    "SEGMENTATION_MODE_SPEC",
    "PRICING_MODE_SPEC",
    "PROMO_MODE_SPEC",
    "ASSORTMENT_MODE_SPEC",
    "CLV_MODE_SPEC",
]