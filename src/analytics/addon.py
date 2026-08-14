"""Add-on / impulse recommendation analysis.

Ranks candidate add-on products for anchor products by lift, so true
impulse items surface above staples.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.rules import create_basket_matrix
from src.analytics.schemas import ADDON_RECS, check


def _lift_table(df: pd.DataFrame) -> pd.DataFrame:
    basket = create_basket_matrix(df)
    n = len(basket)
    cooccur = (basket.to_numpy(dtype=bool).T @ basket.to_numpy(dtype=bool).astype(int)).astype(int)
    support = basket.sum() / n
    products = basket.columns.tolist()
    rows = []
    for i, a in enumerate(products):
        for j, b in enumerate(products):
            if i == j or cooccur[i, j] == 0:
                continue
            p_ab = cooccur[i, j] / n
            p_a, p_b = support.iloc[i], support.iloc[j]
            rows.append(
                {
                    "anchor": a,
                    "addon": b,
                    "support": float(p_ab),
                    "confidence": float(p_ab / p_a),
                    "lift": float(p_ab / (p_a * p_b)),
                    "cooccurrence": int(cooccur[i, j]),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return check(pd.DataFrame(columns=list(ADDON_RECS.columns)), ADDON_RECS, allow_empty=True)
    return check(table, ADDON_RECS)


def get_addon_recommendations(
    df: pd.DataFrame,
    anchor: str,
    top_n: int = 10,
    min_lift: float = 1.2,
) -> pd.DataFrame:
    """Top add-on products for a single anchor, ranked by lift."""
    table = _lift_table(df)
    recs = table[(table["anchor"] == anchor) & table["lift"].ge(min_lift)]
    recs = recs.sort_values(["lift", "support"], ascending=False).head(top_n).reset_index(drop=True)
    return check(recs, ADDON_RECS, allow_empty=True)


def get_anchor_addon_matrix(df: pd.DataFrame, top_n_anchors: int = 20) -> pd.DataFrame:
    """Anchor x addon lift matrix for heatmap rendering."""
    table = _lift_table(df)
    anchors = (
        table.groupby("anchor")["addon"]
        .count()
        .sort_values(ascending=False)
        .head(top_n_anchors)
        .index
    )
    matrix = table[table["anchor"].isin(anchors)].pivot(
        index="anchor", columns="addon", values="lift"
    )
    return matrix.fillna(0.0)


def get_addon_by_category(
    df: pd.DataFrame,
    category: str,
    top_n: int = 10,
    min_lift: float = 1.1,
) -> pd.DataFrame:
    """Best cross-category add-ons for anchors within one category."""
    table = _lift_table(df)
    products = df[df["category"] == category]["stockcode"].unique().tolist()
    if not products:
        return check(pd.DataFrame(columns=list(ADDON_RECS.columns)), ADDON_RECS, allow_empty=True)
    anchors_in_cat = df.loc[df["stockcode"].isin(products), "stockcode"].unique().tolist()
    candidates = df.loc[~df["stockcode"].isin(anchors_in_cat), "stockcode"].unique().tolist()
    recs = table[
        table["anchor"].isin(anchors_in_cat)
        & table["addon"].isin(candidates)
        & table["lift"].ge(min_lift)
    ]
    recs = recs.sort_values(["lift", "support"], ascending=False).head(top_n).reset_index(drop=True)
    return check(recs, ADDON_RECS, allow_empty=True)
