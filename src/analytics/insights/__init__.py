"""Insight Engine: per-domain generators producing structured Insights.

Each module emits evidence-backed ``Insight`` objects (serialized to the shared
insight contract) that the UI ranks and renders as decision cards.
"""

from src.analytics.insights.assortment import generate_assortment_insights
from src.analytics.insights.basket import generate_basket_insights
from src.analytics.insights.cohort import generate_cohort_insights
from src.analytics.insights.customer import generate_customer_insights
from src.analytics.insights.cdt import generate_cdt_insights
from src.analytics.insights.overview import generate_overview_insights
from src.analytics.insights.pricing import generate_pricing_insights
from src.analytics.insights.product import generate_product_insights
from src.analytics.insights.promotion import generate_promotion_insights
from src.analytics.insights.switching import generate_switching_insights

__all__ = [
    "generate_assortment_insights",
    "generate_basket_insights",
    "generate_cohort_insights",
    "generate_customer_insights",
    "generate_cdt_insights",
    "generate_overview_insights",
    "generate_pricing_insights",
    "generate_product_insights",
    "generate_promotion_insights",
    "generate_switching_insights",
]
