"""Shared UI Components for Strategic Decision-Making.

This package provides reusable components for building
category-manager decision interfaces.
"""

from src.ui.components.decision_matrix import (
    MatrixConfig,
    render_bubble_matrix,
    render_decision_matrix,
)
from src.ui.components.download import (
    create_metadata,
    render_download_buttons,
)
from src.ui.components.filter_bar import (
    FilterBarConfig,
    FilterConfig,
    create_filter_configs_from_df,
    render_filter_bar,
)
from src.ui.components.methodology_badge import (
    EVIDENCE_COLORS,
    EVIDENCE_ICONS,
    render_evidence_inline,
    render_evidence_legend,
    render_methodology_badge,
)
from src.ui.components.methodology_badge import (
    EvidenceClass as MethodologyEvidenceClass,
)
from src.ui.components.reliability_badge import (
    RELIABILITY_COLORS,
    RELIABILITY_ICONS,
    render_reliability_badge,
    render_reliability_inline,
    render_reliability_legend,
)
from src.ui.components.strategic_table import (
    ColumnConfig,
    EvidenceClass,
    ReliabilityLevel,
    TableConfig,
    create_strategic_table_config,
    render_strategic_table,
)

__all__ = [
    "MethodologyEvidenceClass",
    # strategic_table
    "ColumnConfig",
    "TableConfig",
    "EvidenceClass",
    "ReliabilityLevel",
    "render_strategic_table",
    "create_strategic_table_config",
    # decision_matrix
    "MatrixConfig",
    "render_decision_matrix",
    "render_bubble_matrix",
    # reliability_badge
    "render_reliability_badge",
    "render_reliability_inline",
    "render_reliability_legend",
    "RELIABILITY_COLORS",
    "RELIABILITY_ICONS",
    # filter_bar
    "FilterConfig",
    "FilterBarConfig",
    "render_filter_bar",
    "create_filter_configs_from_df",
    # methodology_badge
    "EvidenceClass",
    "render_methodology_badge",
    "render_evidence_legend",
    "render_evidence_inline",
    "EVIDENCE_COLORS",
    "EVIDENCE_ICONS",
    # download
    "render_download_buttons",
    "create_metadata",
]
