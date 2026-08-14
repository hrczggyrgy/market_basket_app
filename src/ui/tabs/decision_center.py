"""Decision Center — cross-module "Today's Signals" hub.

Aggregates the ranked decisions, insights and opportunities produced by every
module that adopts the Retail Decision Intelligence pattern: Overview, Pricing,
Product, Switching, Promotion, Cross-sell and (opt-in) CLV + Assortment.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.decision_center import run_decision_center
from src.ui.components import render_insight_cards, render_metric_row, render_opportunity_table
from src.ui.registry import ModeSpec


def _render_scorecard(analysis: object) -> None:
    render_metric_row(
        [
            {
                "label": "Today's Signals",
                "value": str(analysis.n_signals),
                "help": "Structured insights across modules.",
            },
            {
                "label": "Ranked decisions",
                "value": str(analysis.n_opportunities),
                "help": "Actionable opportunities, value-ranked.",
            },
            {
                "label": "Illustrative opportunity value",
                "value": f"€{analysis.total_opportunity_value:,.0f}",
                "help": "Sum of opportunity values (see each card for semantics).",
            },
            {
                "label": "Open risks",
                "value": str(analysis.n_risks),
                "help": "Insights flagged as risks.",
            },
        ]
    )


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/dashboard: Decision Center")
    st.caption(
        "Today's signals across the business — ranked opportunities and risks. "
        "Each signal carries evidence, confidence and a recommended action."
    )

    with st.expander("Engines", expanded=False):
        include_clv = st.checkbox(
            "Include customer CLV engine (slow)",
            value=False,
            help="Runs the BG/NBD model fit; adds retention opportunities.",
        )
        include_assortment = st.checkbox(
            "Include assortment scenario engine (slow)",
            value=False,
            help="Runs the assortment scenario simulator.",
        )

    with st.spinner("Running decision engines..."):
        analysis = run_decision_center(
            df, include_clv=include_clv, include_assortment=include_assortment
        )

    _render_scorecard(analysis)
    st.caption(
        f"Domains contributing: {', '.join(analysis.domains_covered) if analysis.domains_covered else 'none'}"
    )

    st.divider()
    st.subheader(":material/radar: Today's Signals")
    render_insight_cards(analysis.insights)

    st.divider()
    st.subheader(":material/task_alt: Top Decisions / Opportunities")
    render_opportunity_table(analysis.opportunities)


MODE_SPEC: ModeSpec = ModeSpec(
    key="decision_center",
    label="Decision Center",
    icon=":material/dashboard:",
    handler=render,
    requires=(),
)
