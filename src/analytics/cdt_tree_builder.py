"""Oracle-style CDT: Bottom-Up Tree Builder.

Constructs the Customer Decision Tree by:
1. Starting from hierarchical clusters (bottom level = products)
2. Testing attributes (brand, size, flavor, category) against cluster structure
3. Selecting best attribute split at each level using mutual information
4. Recursing until no meaningful splits remain
5. Scoring tree quality against unconstrained clustering baseline
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TreeNode:
    """Node in the Customer Decision Tree."""

    node_id: str
    name: str
    products: List[str] = field(default_factory=list)
    attribute: Optional[str] = None  # splitting attribute (e.g., "brand")
    attribute_value: Optional[str] = None  # value for this branch (e.g., "Brand A")
    children: List["TreeNode"] = field(default_factory=list)
    similarity_within: float = 0.0
    size: int = 0
    is_leaf: bool = True
    cluster_id: Optional[int] = None  # original cluster ID from hierarchical clustering

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "products": self.products,
            "attribute": self.attribute,
            "attribute_value": self.attribute_value,
            "similarity_within": self.similarity_within,
            "size": self.size,
            "is_leaf": self.is_leaf,
            "cluster_id": self.cluster_id,
            "children": [c.to_dict() for c in self.children],
        }


def compute_mutual_information(
    cluster_assignments: Dict[str, int],
    attribute_values: Dict[str, str],
) -> float:
    """
    Compute mutual information between cluster assignments and attribute values.

    MI(C; A) = Σ P(c,a) log(P(c,a) / (P(c)P(a)))

    Higher MI means the attribute better explains the cluster structure.

    Args:
        cluster_assignments: Dict[product_id -> cluster_id]
        attribute_values: Dict[product_id -> attribute_value]

    Returns:
        Mutual information score (>= 0)
    """
    products = list(cluster_assignments.keys())
    if not products:
        return 0.0

    n = len(products)

    # Joint distribution P(c, a)
    joint_counts = defaultdict(int)
    cluster_counts = defaultdict(int)
    attr_counts = defaultdict(int)

    for prod in products:
        c = cluster_assignments[prod]
        a = attribute_values.get(prod, "UNKNOWN")
        joint_counts[(c, a)] += 1
        cluster_counts[c] += 1
        attr_counts[a] += 1

    mi = 0.0
    for (c, a), count in joint_counts.items():
        p_ca = count / n
        p_c = cluster_counts[c] / n
        p_a = attr_counts[a] / n
        if p_ca > 0 and p_c > 0 and p_a > 0:
            mi += p_ca * np.log(p_ca / (p_c * p_a))

    return mi


def compute_attribute_split_quality(
    products: List[str],
    cluster_assignments: Dict[str, int],
    attribute_values: Dict[str, str],
    similarity_matrix: pd.DataFrame,
    min_cluster_size: int = 3,
) -> Tuple[float, Dict[str, List[str]]]:
    """
    Evaluate how well an attribute splits a set of products.

    Returns:
        (mutual_information, dict of attribute_value -> product_list)
    """
    # Filter to products in this set with valid attribute values
    relevant_products = [
        p for p in products if p in attribute_values and p in cluster_assignments
    ]
    if len(relevant_products) < min_cluster_size:
        return 0.0, {}

    # Get cluster assignments for relevant products
    sub_assignments = {p: cluster_assignments[p] for p in relevant_products}
    sub_attrs = {p: attribute_values[p] for p in relevant_products}

    mi = compute_mutual_information(sub_assignments, sub_attrs)

    # Group products by attribute value
    attr_groups = defaultdict(list)
    for p in relevant_products:
        attr_groups[attribute_values[p]].append(p)

    # Filter groups below min size
    attr_groups = {k: v for k, v in attr_groups.items() if len(v) >= min_cluster_size}

    return mi, dict(attr_groups)


def find_best_attribute_split(
    products: List[str],
    cluster_assignments: Dict[str, int],
    attributes_df: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    min_cluster_size: int = 3,
    candidate_attributes: Optional[List[str]] = None,
) -> Tuple[Optional[str], Dict[str, List[str]], float]:
    """
    Find the attribute that best explains the cluster structure.

    Args:
        products: List of product IDs in this subtree
        cluster_assignments: Global cluster assignments
        attributes_df: DataFrame with product attributes (index=product_id, cols=attribute names)
        similarity_matrix: For computing within-group similarity
        min_cluster_size: Minimum products per attribute value group
        candidate_attributes: List of attribute columns to test (None = all)

    Returns:
        (best_attribute_name, groups_dict, mutual_information_score)
    """
    if candidate_attributes is None:
        candidate_attributes = attributes_df.columns.tolist()

    best_attr = None
    best_groups = {}
    best_mi = 0.0

    for attr in candidate_attributes:
        if attr not in attributes_df.columns:
            continue

        attr_values = attributes_df[attr].dropna().to_dict()
        mi, groups = compute_attribute_split_quality(
            products,
            cluster_assignments,
            attr_values,
            similarity_matrix,
            min_cluster_size,
        )

        if mi > best_mi and len(groups) >= 2:  # Need at least 2 groups to split
            best_mi = mi
            best_attr = attr
            best_groups = groups

    return best_attr, best_groups, best_mi


def compute_within_group_similarity(
    products: List[str],
    similarity_matrix: pd.DataFrame,
) -> float:
    """Compute average pairwise similarity within a group of products."""
    if len(products) < 2:
        return 1.0

    sims = []
    for i, a in enumerate(products):
        for b in products[i + 1 :]:
            if a in similarity_matrix.index and b in similarity_matrix.columns:
                sims.append(similarity_matrix.loc[a, b])

    return float(np.mean(sims)) if sims else 0.0


def build_cdt_recursive(
    products: List[str],
    cluster_assignments: Dict[str, int],
    attributes_df: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    node_id_counter: List[int],
    min_cluster_size: int = 3,
    candidate_attributes: Optional[List[str]] = None,
    parent_attr: Optional[str] = None,
) -> TreeNode:
    """
    Recursively build CDT from bottom up.

    Args:
        products: Products in this subtree
        cluster_assignments: Global cluster assignments from hierarchical clustering
        attributes_df: Product attributes DataFrame
        similarity_matrix: Product similarity matrix
        node_id_counter: Mutable counter for unique node IDs
        min_cluster_size: Minimum products per node
        candidate_attributes: Attributes available for splitting
        parent_attr: Attribute used by parent (avoid reusing)

    Returns:
        TreeNode root of this subtree
    """
    node_id_counter[0] += 1
    node_id = f"node_{node_id_counter[0]}"

    # Compute within-group similarity
    sim_within = compute_within_group_similarity(products, similarity_matrix)

    # Base case: too few products or no valid attributes left
    available_attrs = [
        a
        for a in (candidate_attributes or attributes_df.columns)
        if a != parent_attr and a in attributes_df.columns
    ]

    if len(products) < min_cluster_size or not available_attrs:
        return TreeNode(
            node_id=node_id,
            name=f"Leaf ({len(products)} products)",
            products=products,
            similarity_within=sim_within,
            size=len(products),
            is_leaf=True,
        )

    # Find best attribute split
    best_attr, groups, mi = find_best_attribute_split(
        products,
        cluster_assignments,
        attributes_df,
        similarity_matrix,
        min_cluster_size,
        available_attrs,
    )

    # No good split found - make leaf
    if best_attr is None or len(groups) < 2 or mi <= 0:
        return TreeNode(
            node_id=node_id,
            name=f"Leaf ({len(products)} products)",
            products=products,
            similarity_within=sim_within,
            size=len(products),
            is_leaf=True,
        )

    # Create internal node with children for each attribute value
    children = []
    for attr_value, group_products in groups.items():
        child = build_cdt_recursive(
            group_products,
            cluster_assignments,
            attributes_df,
            similarity_matrix,
            node_id_counter,
            min_cluster_size,
            candidate_attributes,
            parent_attr=best_attr,
        )
        child.attribute = best_attr
        child.attribute_value = attr_value
        child.name = f"{best_attr}={attr_value} ({len(group_products)})"
        children.append(child)

    # If only one child (shouldn't happen with len(groups) >= 2), make leaf
    if len(children) <= 1:
        return TreeNode(
            node_id=node_id,
            name=f"Leaf ({len(products)} products)",
            products=products,
            similarity_within=sim_within,
            size=len(products),
            is_leaf=True,
        )

    return TreeNode(
        node_id=node_id,
        name=f"Split: {best_attr} (MI={mi:.3f})",
        products=products,
        attribute=best_attr,
        children=children,
        similarity_within=sim_within,
        size=len(products),
        is_leaf=False,
    )


def score_tree(root: TreeNode, similarity_matrix: pd.DataFrame) -> float:
    """
    Score the CDT tree by computing weighted average within-cluster similarity.

    Higher score = better tree (more homogeneous clusters).
    """
    leaf_nodes = []

    def collect_leaves(node):
        if node.is_leaf:
            leaf_nodes.append(node)
        else:
            for child in node.children:
                collect_leaves(child)

    collect_leaves(root)

    if not leaf_nodes:
        return 0.0

    # Weighted average by cluster size
    total_size = sum(n.size for n in leaf_nodes)
    if total_size == 0:
        return 0.0

    weighted_sim = sum(n.similarity_within * n.size for n in leaf_nodes) / total_size
    return weighted_sim


def count_nodes(root: TreeNode) -> int:
    """Count total nodes in tree."""
    count = 1
    for child in root.children:
        count += count_nodes(child)
    return count


def count_leaves(root: TreeNode) -> int:
    """Count leaf nodes in tree."""
    if root.is_leaf:
        return 1
    return sum(count_leaves(child) for child in root.children)


def max_depth(root: TreeNode) -> int:
    """Get maximum depth of tree."""
    if root.is_leaf:
        return 1
    return 1 + max(max_depth(child) for child in root.children)


def tree_to_dataframe(root: TreeNode) -> pd.DataFrame:
    """Flatten tree to DataFrame for export."""
    rows = []

    def traverse(node, depth=0, parent_id=None):
        rows.append(
            {
                "node_id": node.node_id,
                "parent_id": parent_id,
                "depth": depth,
                "name": node.name,
                "attribute": node.attribute,
                "attribute_value": node.attribute_value,
                "products": ";".join(node.products) if node.products else "",
                "n_products": node.size,
                "similarity_within": node.similarity_within,
                "is_leaf": node.is_leaf,
                "cluster_id": node.cluster_id,
            }
        )
        for child in node.children:
            traverse(child, depth + 1, node.node_id)

    traverse(root)
    return pd.DataFrame(rows)


def tree_to_json(root: TreeNode) -> str:
    """Serialize tree to JSON string."""
    return json.dumps(root.to_dict(), indent=2)


def build_cdt(
    similarity_matrix: pd.DataFrame,
    cluster_assignments: Dict[str, int],
    attributes_df: pd.DataFrame,
    min_cluster_size: int = 3,
    quality_threshold: float = 0.60,
    candidate_attributes: Optional[List[str]] = None,
) -> Tuple[TreeNode, Dict]:
    """
    Build complete Customer Decision Tree.

    Args:
        similarity_matrix: Product similarity matrix
        cluster_assignments: Initial cluster assignments from hierarchical clustering
        attributes_df: Product attributes (index=product_id, cols=[brand, size, flavor, etc.])
        min_cluster_size: Minimum products per node
        quality_threshold: Minimum tree quality vs unconstrained baseline (0.60 = 60%)
        candidate_attributes: Attributes to consider for splitting

    Returns:
        (root_node, metadata_dict)
    """
    # Ensure attributes_df has all products
    all_products = list(similarity_matrix.index)
    attributes_df = attributes_df.reindex(all_products)

    # Compute unconstrained baseline (best possible clustering quality)
    from .cdt_clustering import compute_unconstrained_baseline

    unconstrained_baseline = compute_unconstrained_baseline(similarity_matrix)

    # Build tree
    node_id_counter = [0]
    root = build_cdt_recursive(
        all_products,
        cluster_assignments,
        attributes_df,
        similarity_matrix,
        node_id_counter,
        min_cluster_size,
        candidate_attributes,
    )

    # Compute tree quality
    tree_quality = score_tree(root, similarity_matrix)

    # Compare to baseline
    quality_ratio = (
        tree_quality / unconstrained_baseline if unconstrained_baseline > 0 else 0
    )
    passed_threshold = quality_ratio >= quality_threshold

    metadata = {
        "unconstrained_baseline": unconstrained_baseline,
        "tree_quality": tree_quality,
        "quality_ratio": quality_ratio,
        "passed_threshold": passed_threshold,
        "quality_threshold": quality_threshold,
        "n_nodes": count_nodes(root),
        "n_leaves": count_leaves(root),
        "max_depth": max_depth(root),
    }

    return root, metadata


def prune_tree(root: TreeNode, threshold: float = 0.60) -> TreeNode:
    """
    Prune tree branches that don't meet quality threshold.

    Note: This is a simple post-hoc pruning. The build process already
    stops splitting when MI <= 0, but this can further simplify.
    """
    if root.is_leaf:
        return root

    # Recursively prune children
    pruned_children = [prune_tree(child, threshold) for child in root.children]

    # If pruning reduced to <= 1 child, collapse to leaf
    if len(pruned_children) <= 1:
        return TreeNode(
            node_id=root.node_id,
            name=f"Leaf ({root.size} products)",
            products=root.products,
            similarity_within=root.similarity_within,
            size=root.size,
            is_leaf=True,
        )

    # Rebuild node with pruned children
    return TreeNode(
        node_id=root.node_id,
        name=root.name,
        products=root.products,
        attribute=root.attribute,
        children=pruned_children,
        similarity_within=root.similarity_within,
        size=root.size,
        is_leaf=False,
    )


def extract_product_attributes(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    attribute_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract product-level attributes from transaction data.

    Looks for columns like: category, brand, size, flavor, color, etc.
    """
    if attribute_cols is None:
        # Auto-detect common attribute columns
        candidates = [
            "category",
            "brand",
            "size",
            "flavor",
            "color",
            "variant",
            "type",
            "style",
        ]
        attribute_cols = [c for c in candidates if c in transactions_df.columns]

    if not attribute_cols:
        return pd.DataFrame(index=transactions_df[product_col].unique())

    # Get first non-null value per product for each attribute
    attr_data = {}
    for attr in attribute_cols:
        attr_series = transactions_df.groupby(product_col)[attr].first()
        attr_data[attr] = attr_series

    return pd.DataFrame(attr_data)
