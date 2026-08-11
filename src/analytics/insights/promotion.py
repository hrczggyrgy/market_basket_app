"""Promotion-domain insight generation.

Promo scorecard (WIN / MIXED / INEFFECTIVE / DESTROYS VALUE), ROI and
cannibalization signals turned into structured insights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.intelligence import Insight, insights_to_dataframe
from src.analytics.schemas import PRICING_INSIGHTS, check

_ROI_BREAKEVEN = 0.0  # incremental profit >= 0 after promo cost
_WIN_MIN_ROI_PCT = 50.0


def classify_promo_score(row: pd.Series) -> str:
    """Scorecard bucket for one promo-waterfall row (by SKU).

    - WIN: positive net incremental revenue with ROI at/above break-even.
    - MIXED: quantity lifts but margin erodes (net incremental slightly
      negative, or ROI below break-even despite positive volume).
    - INEFFECTIVE: no meaningful incremental demand (qty component ~ 0).
    - DESTROYS VALUE: net incremental clearly negative relative to the
      volume moved (or volume itself negative).
    """
    net = float(row.get("net_incremental_revenue", 0.0) or 0.0)
    qty = float(row.get("incremental_revenue_qty", 0.0) or 0.0)
    price = float(row.get("incremental_revenue_price", 0.0) or 0.0)
    roi = float(row.get("roi", 0.0) or 0.0)

    volume = abs(qty) if qty != 0 else max(abs(price), 1.0)
    # No incremental volume -> ineffective (use absolute qty)
    if abs(qty) <= 0.01 * max(abs(qty), abs(price), 1.0):
        return "INEFFECTIVE"
    # Loss dominates the volume moved -> the promo destroyed more than it added.
    if qty < 0 or net < -0.5 * volume:
        return "DESTROYS_VALUE"
    if roi < _ROI_BREAKEVEN:
        return "MIXED"
    return "WIN"


def generate_promotion_insights(
    waterfall_df: pd.DataFrame,
    roi_df: pd.DataFrame | None = None,
    lift_df: pd.DataFrame | None = None,
    cannibalization_df: pd.DataFrame | None = None,
    min_roi_pct: float = _WIN_MIN_ROI_PCT,
) -> pd.DataFrame:
    """Build Promotion insights from the promo scorecard inputs.

    Args:
        waterfall_df: PROMO_WATERFALL output (per-SKU net incremental revenue).
        roi_df: optional PROMO_ROI output (per-SKU ROI with CI).
        lift_df: optional PROMO_LIFT output (per-promo significance).
        cannibalization_df: optional PROMO_CANNIBALIZATION output.
        min_roi_pct: ROI (%) threshold for a "strong WIN".

    Returns:
        DataFrame validated against PRICING_INSIGHTS with ``domain`` = "promotion".
    """
    insights: list[Insight] = []

    if waterfall_df is None or waterfall_df.empty:
        return check(insights_to_dataframe(insights), PRICING_INSIGHTS, allow_empty=True)

    work = waterfall_df.copy()
    if roi_df is not None and not roi_df.empty:
        work = work.merge(roi_df[["stockcode", "roi_pct"]], on="stockcode", how="left")
    else:
        work["roi_pct"] = np.nan

    work["score"] = work.apply(classify_promo_score, axis=1)

    for score, meta in {
        "WIN": {
            "kind": "opportunity",
            "title_fmt": "{n} SKUs' promos are clear wins",
            "action": "Repeat the playbook: similar depth, timing and duration.",
        },
        "MIXED": {
            "kind": "efficiency",
            "title_fmt": "{n} SKUs' promos are mixed: volume up, margin down",
            "action": "Re-shape these promos — shallower discount or shorter window to protect margin.",
        },
        "INEFFECTIVE": {
            "kind": "leakage",
            "title_fmt": "{n} SKUs' promos generated no incremental volume",
            "action": "Reallocate the promo budget; the discount is pure margin give-away.",
        },
        "DESTROYS_VALUE": {
            "kind": "risk",
            "title_fmt": "{n} SKUs' promos destroyed value",
            "action": "Stop these promotions; validate with a causal test before re-running.",
        },
    }.items():
        grp = work[work["score"] == score]
        if grp.empty:
            continue
        n = int(len(grp))
        net_sum = float(grp["net_incremental_revenue"].sum())
        entities = ", ".join(grp["stockcode"].astype(str).head(5).tolist())
        if n > 5:
            entities += f" +{n - 5} more"
        insights.append(
            Insight(
                domain="promotion",
                entity=entities,
                kind=meta["kind"],
                title=meta["title_fmt"].format(n=n),
                evidence=(
                    f"Aggregate net incremental revenue: €{net_sum:,.0f} across {n} SKUs. "
                    f"Waterfall components are descriptive, not causal."
                ),
                action=meta["action"],
                confidence="medium",
                impact_value=abs(net_sum) if net_sum != 0 else None,
                sample_size=n,
            )
        )

    strong = work[work["roi_pct"].ge(min_roi_pct)]
    if not strong.empty:
        best = strong.sort_values("roi_pct", ascending=False).iloc[0]
        insights.append(
            Insight(
                domain="promotion",
                entity=str(best["stockcode"]),
                kind="opportunity",
                title=f"{best['stockcode']} returned {best['roi_pct']:.0%} ROI",
                evidence=(
                    f"ROI above the {min_roi_pct:.0%} threshold; net incremental "
                    f"€{best['net_incremental_revenue']:,.0f}."
                ),
                action="Use as the benchmark promo; scale carefully with the halo/cannibalization check.",
                confidence="high" if float(best["roi_pct"]) >= 2 * min_roi_pct else "medium",
                impact_value=float(best["net_incremental_revenue"]),
            )
        )

    if lift_df is not None and not lift_df.empty:
        n_ns = int((~lift_df["significant"]).sum())
        if n_ns > 0:
            insights.append(
                Insight(
                    domain="promotion",
                    entity="non-significant promos",
                    kind="watch",
                    title=f"{n_ns} promos show no statistically significant lift",
                    evidence="These promos cannot be judged incrementally from the lift test alone.",
                    action="Treat as inconclusive; use causal estimates or more data before judging.",
                    confidence="medium",
                    sample_size=int(len(lift_df)),
                )
            )

    if cannibalization_df is not None and not cannibalization_df.empty:
        high = cannibalization_df[cannibalization_df["cannibalization_index"] >= 0.5]
        if not high.empty:
            top = high.sort_values("cannibalized_revenue", ascending=False).iloc[0]
            insights.append(
                Insight(
                    domain="promotion",
                    entity=str(top["promo_product"]),
                    kind="leakage",
                    title=f"{top['promo_product']} cannibalizes {top['peer_product']}",
                    evidence=(
                        f"€{top['cannibalized_revenue']:,.0f} of peer revenue displaced "
                        f"(index {top['cannibalization_index']:.0%})."
                    ),
                    action="Check whether the promo is category-net-positive before judging.",
                    confidence="medium",
                    impact_value=float(top["cannibalized_revenue"]),
                )
            )

    table = insights_to_dataframe(insights)
    return check(table, PRICING_INSIGHTS, allow_empty=True)
