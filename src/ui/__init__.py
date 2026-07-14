"""UI module initialization."""

from .addon_tab import render_addon_tab
from .cohort_tab import render_cohort_tab
from .copurchase_tab import render_copurchase_tab
from .export import render_analytics_export, render_export_buttons
from .product_performance_tab import render_product_performance_tab
from .promotional_tab import render_promotional_tab
from .rules_tab import render_rules_tab
from .segmentation_tab import render_segmentation_tab
from .sidebar import render_data_info, render_sidebar
from .switching_tab import render_switching_tab
from .tabs import persistent_tabs, persistent_tabs_container, tabbed_view
from .tree_tab import render_tree_tab

__all__ = [
    "render_sidebar",
    "render_data_info",
    "render_rules_tab",
    "render_copurchase_tab",
    "render_addon_tab",
    "render_switching_tab",
    "render_tree_tab",
    "render_segmentation_tab",
    "render_product_performance_tab",
    "render_cohort_tab",
    "render_promotional_tab",
    "render_export_buttons",
    "render_analytics_export",
    "persistent_tabs",
    "persistent_tabs_container",
    "tabbed_view",
]
