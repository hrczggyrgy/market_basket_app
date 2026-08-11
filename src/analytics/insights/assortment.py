"""Assortment-domain insight generation.

Coverage / recovery realism and scenario trade-offs turned into structured
insights: how much revenue a slimmer assortment can expect to keep.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.intelligence import Insight, insights_to_dataframe
from src.analytics.schemas import PRICING_INSIGHTS, check


def generate_assortment_insights(
    scenario_df: pd.DataFrame,
    solution_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build Assortment insights.

    Args:
        scenario_df: ASSORTMENT_SCENARIO output (method comparison).
        solution_df: optional ASSORTMENT_SOLUTION output (chosen solution).

    Returns:
        DataFrame validated against PRICING_INSIGHTS with ``domain`` = "assortment".
    """
    insights: list[Insight] = []

    if scenario_df is None or scenario_df.empty:
        return check(insights_to_dataframe(insights), PRICING_INSIGHTS, allow_empty=True)

    best = scenario_df.sort_values("expected_revenue", ascending=False).iloc[0]
    n_skus = int(best["n_skus"])
    kept_rev = float(best["kept_revenue"])
    recovered = float(best["recovered_revenue"])
    unmet = float(best["unmet_demand"])
    coverage = float(best["coverage"])
    recovery = float(best["recovery_rate"])

    total_rev = kept_rev + recovered + unmet
    cut_pct = 1.0 - kept_rev / total_rev if total_rev > 0 else 0.0

    insights.append(
        Insight(
            domain="assortment",
            entity="best scenario",
            kind="opportunity",
            title=f"Slimming to {n_skus} SKUs keeps {coverage:.0%} of revenue",
            evidence=(
                f"Scenario '{best['method']}': €{kept_rev:,.0f} kept + €{recovered:,.0f} "
                f"recovered from switching = €{best['expected_revenue']:,.0f} expected "
                f"revenue; €{unmet:,.0f} unmet demand ({recovery:.0%} of lost revenue "
                f"recovered)."
            ),
            action="Treat recovery as optimistic until validated causally; run a market test.",
            confidence="low",
            impact_value=float(best["expected_revenue"]),
            sample_size=n_skus,
            stability=round(recovery, 3),
        )
    )

    if cut_pct > 0.3:
        insights.append(
            Insight(
                domain="assortment",
                entity="assortment cut",
                kind="risk",
                title=f"{cut_pct:.0%} of SKUs cut for {coverage:.0%} revenue coverage",
                evidence=(
                    "A large SKU cut still leaves double-digit unmet demand, "
                    "which risks sending customers to competitors entirely."
                ),
                action="Prioritize keeping unique-demand SKUs and deepen substitutes first.",
                confidence="medium",
                impact_value=unmet if unmet > 0 else None,
            )
        )
    else:
        insights.append(
            Insight(
                domain="assortment",
                entity="assortment cut",
                kind="efficiency",
                title=f"A {cut_pct:.0%} SKU cut keeps {coverage:.0%} of revenue",
                evidence="Most revenue is concentrated in a small core of SKUs.",
                action="Proceed cautiously; validate the long tail before cutting.",
                confidence="medium",
                impact_value=unmet if unmet > 0 else None,
            )
        )

    if solution_df is not None and not solution_df.empty:
        n_kept = int((solution_df["selected"] == 1).sum()) if "selected" in solution_df.columns else int(len(solution_df))
        kept_sol_rev = float(solution_df["revenue"].sum())
        insights.append(
            Insight(
                domain="assortment",
                entity="optimized solution",
                kind="efficiency",
                title=f"Optimized solution: {n_kept} SKUs / €{kept_sol_rev:,.0f}",
                evidence="Solution from the MILP/heuristic optimizer, ranked by revenue with switching recovery.",
                action="Compare against current assortment before any change.",
                confidence="medium",
                sample_size=n_kept,
            )
        )

    table = insights_to_dataframe(insights)
    return check(table, PRICING_INSIGHTS, allow_empty=True)
