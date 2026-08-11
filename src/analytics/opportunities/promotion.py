"""Promotion opportunity generation.

Winning promos become scale-up opportunities; value-destroying promos become
stop/re-shape actions with the avoided loss as the value at stake.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.insights.promotion import classify_promo_score
from src.analytics.intelligence import Opportunity, opportunities_to_dataframe
from src.analytics.schemas import OPPORTUNITY_LIST, check

_WIN_MIN_ROI_PCT = 50.0


def generate_promotion_opportunities(
    waterfall_df: pd.DataFrame,
    roi_df: pd.DataFrame | None = None,
    top_n: int = 10,
    min_roi_pct: float = _WIN_MIN_ROI_PCT,
) -> pd.DataFrame:
    """Build rankable promotion opportunities.

    Args:
        waterfall_df: PROMO_WATERFALL output (per-SKU scorecard inputs).
        roi_df: optional PROMO_ROI output (ROI per SKU).
        top_n: maximum number of opportunities.
        min_roi_pct: ROI (%) floor for a scale-up recommendation.

    Returns:
        DataFrame validated against OPPORTUNITY_LIST with ``domain`` = "promotion".
    """
    opportunities: list[Opportunity] = []

    if waterfall_df is None or waterfall_df.empty:
        return check(opportunities_to_dataframe(opportunities), OPPORTUNITY_LIST, allow_empty=True)

    work = waterfall_df.copy()
    if roi_df is not None and not roi_df.empty:
        work = work.merge(roi_df[["stockcode", "roi_pct"]], on="stockcode", how="left")
    else:
        work["roi_pct"] = np.nan
    work["score"] = work.apply(classify_promo_score, axis=1)

    wins = work[work["score"] == "WIN"].sort_values("roi_pct", ascending=False)
    for _, row in wins.head(top_n).iterrows():
        roi = float(row["roi_pct"]) if pd.notna(row["roi_pct"]) else 0.0
        if roi < min_roi_pct:
            continue
        opportunities.append(
            Opportunity(
                domain="promotion",
                entity=str(row["stockcode"]),
                title=f"Scale the winning promo on {row['stockcode']}",
                action="Repeat with the same depth/timing; test one broader variant.",
                source="promo_waterfall",
                rationale=(
                    f"Net incremental €{float(row['net_incremental_revenue']):,.0f} "
                    f"(ROI {roi:.0%})."
                ),
                value=round(float(row["net_incremental_revenue"]), 0),
                confidence="high" if roi >= 2 * min_roi_pct else "medium",
            )
        )

    destroys = work[work["score"] == "DESTROYS_VALUE"].sort_values("net_incremental_revenue")
    for _, row in destroys.head(max(0, top_n - len(opportunities))).iterrows():
        avoided = -float(row["net_incremental_revenue"])
        opportunities.append(
            Opportunity(
                domain="promotion",
                entity=str(row["stockcode"]),
                title=f"Stop the value-destroying promo on {row['stockcode']}",
                action="Suspend; verify with a causal estimate before any re-run.",
                source="promo_waterfall",
                rationale=f"Net incremental revenue of €{avoided:,.0f} lost during the promo window.",
                value=round(avoided, 0),
                confidence="medium",
            )
        )

    table = opportunities_to_dataframe(opportunities)
    if not table.empty:
        table = table.sort_values("value", ascending=False, na_position="last").reset_index(drop=True)
    return check(table, OPPORTUNITY_LIST, allow_empty=True)
