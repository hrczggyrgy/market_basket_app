"""CDT opportunity generation.

Products that are close in the Customer Decision Tree become bundling
opportunities; products in different branches become cross-selling opportunities.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.cdt.tree import build_cdt, tree_to_dataframe
from src.analytics.intelligence import Opportunity, opportunities_to_dataframe
from src.analytics.schemas import OPPORTUNITY_LIST, check


def generate_cdt_opportunities(
    df: pd.DataFrame,
    max_depth: int = 3,
    top_n: int = 10,
) -> pd.DataFrame:
    """Build CDT-based opportunities.

    Args:
        df: Raw transaction DataFrame with at least columns:
            customer_id, date, stockcode, price, quantity.
        max_depth: Maximum depth to grow the CDT.
        top_n: Maximum number of opportunities.

    Returns:
        DataFrame validated against OPPORTUNITY_LIST with ``domain`` = "cdt".
    """
    opportunities: list[Opportunity] = []

    # Build the CDT
    try:
        root = build_cdt(df, max_depth=max_depth)
    except Exception:
        return check(opportunities_to_dataframe(opportunities), OPPORTUNITY_LIST, allow_empty=True)

    # Convert tree to dataframe
    tree_df = tree_to_dataframe(root)
    if tree_df.empty:
        return check(opportunities_to_dataframe(opportunities), OPPORTUNITY_LIST, allow_empty=True)

    # We look for sibling nodes (same parent) that are leaves - these are similar products
    # Group by parent and collect leaves
    if "parent" in tree_df.columns and "is_leaf" in tree_df.columns:
        # We'll consider each parent that has at least two leaf children
        parent_groups = tree_df[tree_df["is_leaf"]].groupby("parent")
        for parent_id, group in parent_groups:
            if len(group) >= 2:
                # Take up to 2 pairs from this group (or just one opportunity per parent)
                # For simplicity, we create one opportunity per parent group
                products = ", ".join(group["product"].astype(str).head(3).tolist())
                opportunities.append(
                    Opportunity(
                        domain="cdt",
                        entity=products,
                        title=f"Product bundling opportunity: {products}",
                        action="Bundle or cross-promote these frequently co-purchased products.",
                        source="cdt_siblings",
                        rationale=(
                            f"These products share a parent node in the CDT at depth "
                            f"{group['depth'].iloc[0] if 'depth' in group.columns else 'unknown'}."
                        ),
                        value=len(group),  # Number of products in the group
                        confidence="medium",
                    )
                )
                # We break after one group to avoid too many opportunities, but we can continue if needed
                # We'll limit to top_n groups
                if len(opportunities) >= top_n:
                    break

    # If we need more opportunities, we can look at the root's children (if any)
    if len(opportunities) < top_n:
        # Get the direct children of the root
        if "depth" in tree_df.columns:
            children = tree_df[tree_df["depth"] == 1]
            if not children.empty:
                # We can consider each child as a branch for cross-selling
                # For simplicity, we take the two largest branches (by count) and suggest cross-selling between them
                if "count" in children.columns:
                    children = children.sort_values("count", ascending=False)
                    if len(children) >= 2:
                        child1 = children.iloc[0]
                        child2 = children.iloc[1]
                        opportunities.append(
                            Opportunity(
                                domain="cdt",
                                entity=f"{child1['product']} vs {child2['product']}",
                                title=f"Cross-sell opportunity: {child1['product']} -> {child2['product']}",
                                action="Promote products from different CDT branches to increase basket size.",
                                source="cdt_branches",
                                rationale=(
                                    f"Products in branch '{child1['product']}' (count {child1.get('count', 0)}) "
                                    f"and branch '{child2['product']}' (count {child2.get('count', 0)}) "
                                    f"are in different branches of the CDT."
                                ),
                                value=abs(float(child1.get("count", 0)) - float(child2.get("count", 0))),
                                confidence="low",
                            )
                        )

    # Sort by value descending and trim to top_n
    opportunities = sorted(opportunities, key=lambda x: x.value, reverse=True)[:top_n]

    table = opportunities_to_dataframe(opportunities)
    if not table.empty:
        table = table.sort_values("value", ascending=False, na_position="last").reset_index(drop=True)
    return check(table, OPPORTUNITY_LIST, allow_empty=True)
