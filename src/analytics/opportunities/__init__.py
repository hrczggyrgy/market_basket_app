"""Opportunity Engine: ranked commercial opportunities across domains.

Every module feeds opportunities into the shared OPPORTUNITY_LIST shape so the
Decision Center can rank them by EUR impact.
"""

from src.analytics.opportunities.assortment import generate_assortment_opportunities
from src.analytics.opportunities.basket import generate_basket_opportunities
from src.analytics.opportunities.cdt import generate_cdt_opportunities
from src.analytics.opportunities.cohort import generate_cohort_opportunities
from src.analytics.opportunities.cross_sell import generate_cross_sell_opportunities
from src.analytics.opportunities.pricing import generate_pricing_opportunities
from src.analytics.opportunities.product import generate_product_opportunities
from src.analytics.opportunities.promotion import generate_promotion_opportunities
from src.analytics.opportunities.retention import generate_retention_opportunities
from src.analytics.opportunities.switching import generate_switching_opportunities

__all__ = [
    "generate_assortment_opportunities",
    "generate_basket_opportunities",
    "generate_cohort_opportunities",
    "generate_cross_sell_opportunities",
    "generate_cdt_opportunities",
    "generate_product_opportunities",
    "generate_pricing_opportunities",
    "generate_promotion_opportunities",
    "generate_retention_opportunities",
    "generate_switching_opportunities",
]
