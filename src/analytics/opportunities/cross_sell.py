"""Cross-sell opportunity generation.

Rankable add-on / co-purchase opportunities from affinity pairs and add-on
recommendations, with an order-of-magnitude revenue value where support data
permits an estimate.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.intelligence import Opportunity, opportunities_to_dataframe
from src.analytics.schemas import OPPORTUNITY_LIST, check

_MIN_LIFT = 1.5  # add-on must clear this lift to be worth promoting
_MIN_COOCCURRENCE = 10


def generate_cross_sell_opportunities(
    addon_df: pd.DataFrame,
    affinity_df: pd.DataFrame | None = None,
    revenue_by_product: pd.Series | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Build rankable cross-sell opportunities.

    Args:
        addon_df: ADDON_RECS output (anchor -> addon with lift/support).
        affinity_df: optional AFFINITY_PAIRS output (pair-level affinity).
        revenue_by_product: optional per-product revenue Series (used to size
            the illustrative opportunity value).
        top_n: maximum number of opportunities.

    Returns:
        DataFrame validated against OPPORTUNITY_LIST with ``domain`` = "cross_sell".
    """
    opportunities: list[Opportunity] = []

    if addon_df is not None and not addon_df.empty:
        work = addon_df.copy()
        work = work[work["lift"].ge(_MIN_LIFT)]
        if "cooccurrence" in work.columns:
            work = work[work["cooccurrence"].ge(_MIN_COOCCURRENCE)]
        work = work.sort_values(["lift", "support"], ascending=False).head(top_n)

        for _, row in work.iterrows():
            anchor = str(row["anchor"])
            addon = str(row["addon"])
            lift = float(row["lift"])
            value: float = 0.0
            if revenue_by_product is not None and anchor in revenue_by_product.index:
                anchor_rev = float(revenue_by_product.loc[anchor])
                # Illustrative: every 1% of anchor basket that attaches the addon.
                value = round(anchor_rev * 0.01 * min(lift - 1.0, 3.0), 0)
            opportunities.append(
                Opportunity(
                    domain="cross_sell",
                    entity=addon,
                    title=f"Promote {addon} as an add-on to {anchor}",
                    action="Add a 'frequently bought together' placement and a bundle test.",
                    source="cooccurrence_lift",
                    rationale=(
                        f"Add-on lift {lift:.1f}x with support {float(row['support']):.0%} "
                        f"and {int(row.get('cooccurrence', 0))} co-occurrences."
                    ),
                    value=value,
                    confidence="high"
                    if lift >= 2.0 and float(row["support"]) >= 0.05
                    else "medium",
                )
            )

    if affinity_df is not None and not affinity_df.empty and len(opportunities) < top_n:
        work = affinity_df.copy()
        work = work[work["affinity"].ge(_MIN_LIFT)]
        work = work.sort_values(["affinity", "cooccurrence"], ascending=False).head(
            top_n - len(opportunities)
        )
        for _, row in work.iterrows():
            a, b = str(row["product_a"]), str(row["product_b"])
            value = 0.0
            if revenue_by_product is not None and b in revenue_by_product.index:
                value = round(
                    float(revenue_by_product.loc[b])
                    * 0.01
                    * min(float(row["affinity"]) - 1.0, 3.0),
                    0,
                )
            opportunities.append(
                Opportunity(
                    domain="cross_sell",
                    entity=f"{b} (paired with {a})",
                    title=f"Co-purchase pair {a} + {b} deserves a joint placement",
                    action="Test joint display / bundle; track basket-size lift.",
                    source="affinity",
                    rationale=(
                        f"Affinity {float(row['affinity']):.1f}x, "
                        f"{int(row['cooccurrence'])} co-occurrences, "
                        f"support {float(row['support_b']):.0%}."
                    ),
                    value=value,
                    confidence="medium",
                )
            )

    table = opportunities_to_dataframe(opportunities)
    if not table.empty:
        table = table.sort_values("value", ascending=False, na_position="last").reset_index(
            drop=True
        )
    return check(table, OPPORTUNITY_LIST, allow_empty=True)
