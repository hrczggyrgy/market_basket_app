"""CDT opportunity generation.

Products that are close in the Customer Decision Tree become bundling
opportunities; products in different branches become cross-selling opportunities.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.cdt.attributes import build_transaction_derived_attributes
from src.analytics.cdt.similarity import build_similarity_matrix
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
        attributes_df = build_transaction_derived_attributes(df)
        similarity_matrix = build_similarity_matrix(df)
        root = build_cdt(attributes_df, similarity_matrix, max_depth=max_depth)
    except Exception:
        return check(opportunities_to_dataframe(opportunities), OPPORTUNITY_LIST, allow_empty=True)

    # Convert tree to dataframe
    nodes_df, products_df = tree_to_dataframe(root)
    if nodes_df.empty:
        return check(opportunities_to_dataframe(opportunities), OPPORTUNITY_LIST, allow_empty=True)

    # We look for sibling nodes (same parent) that are leaves - these are similar products
    # Group by parent and collect leaves
    if "parent_id" in nodes_df.columns and "is_leaf" in nodes_df.columns:
        # We'll consider each parent that has at least two leaf children
        parent_groups = nodes_df[nodes_df["is_leaf"] == 1].groupby("parent_id")
        for _parent_id, group in parent_groups:
            if len(group) >= 2:
                # Take up to 2 pairs from this group (or just one opportunity per parent)
                # For simplicity, we create one opportunity per parent group
                # Need to join with products_df to get product names
                group_products = group.merge(products_df, on="node_id")
                products = ", ".join(group_products["stockcode"].astype(str).head(3).tolist())
                opportunities.append(
                    Opportunity(
                        domain="cdt",
                        entity=products,
                        title=f"Product bundling opportunity: {products}",
                        action="Bundle or cross-promote these frequently co-purchased products.",
                        source="cdt_siblings",
                        rationale=(
                            "These products share a parent node in the CDT."
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
        # Root is node_id='node_1'
        children = nodes_df[nodes_df["parent_id"] == "node_1"]
        if not children.empty:
            # Sort by size descending
            children = children.sort_values("size", ascending=False)
            if len(children) >= 2:
                child1 = children.iloc[0]
                child2 = children.iloc[1]

                # Get products for these children
                child1_products = products_df[products_df["node_id"] == child1["node_id"]]["stockcode"].tolist()
                child2_products = products_df[products_df["node_id"] == child2["node_id"]]["stockcode"].tolist()

                p1 = ", ".join(child1_products[:2])
                p2 = ", ".join(child2_products[:2])

                opportunities.append(
                    Opportunity(
                        domain="cdt",
                        entity=f"{p1} vs {p2}",
                        title=f"Cross-sell opportunity: {p1} -> {p2}",
                        action="Promote products from different CDT branches to increase basket size.",
                        source="cdt_branches",
                        rationale=(
                            f"Products in branch '{child1['name']}' (size {child1.get('size', 0)}) "
                            f"and branch '{child2['name']}' (size {child2.get('size', 0)}) "
                            f"are in different branches of the CDT."
                        ),
                        value=abs(float(child1.get("size", 0)) - float(child2.get("size", 0))),
                        confidence="low",
                    )
                )

    # Sort by value descending and trim to top_n
    opportunities = sorted(opportunities, key=lambda x: x.value, reverse=True)[:top_n]

    table = opportunities_to_dataframe(opportunities)
    if not table.empty:
        table = table.sort_values("value", ascending=False, na_position="last").reset_index(
            drop=True
        )
    return check(table, OPPORTUNITY_LIST, allow_empty=True)
