"""Customer Decision Tree construction.

The CDT recursively partitions the catalogue by product attributes
(price/velocity/seasonality tier, etc.) into leaves that group behaviourally
similar products. Splits are chosen to maximize how well an attribute
explains either the reference cluster structure (mutual information /
entropy / gini gain) or the product similarity matrix (weighted
within-group similarity gain).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.analytics.schemas import CDT_TREE_NODES, CDT_TREE_PRODUCTS, CDT_TREE_SCORE, check


@dataclass
class TreeNode:
    """A node in the Customer Decision Tree."""

    node_id: str
    name: str
    products: list[str] = field(default_factory=list)
    attribute: str | None = None
    attribute_value: str | None = None
    children: list["TreeNode"] = field(default_factory=list)
    similarity_within: float = 0.0
    size: int = 0
    is_leaf: bool = True
    parent_id: str | None = None
    split_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "products": self.products,
            "attribute": self.attribute,
            "attribute_value": self.attribute_value,
            "similarity_within": self.similarity_within,
            "size": self.size,
            "is_leaf": self.is_leaf,
            "parent_id": self.parent_id,
            "split_score": self.split_score,
            "children": [c.to_dict() for c in self.children],
        }


def compute_mutual_information(
    cluster_assignments: dict[str, int],
    attribute_values: dict[str, str],
) -> float:
    """MI(C; A) over the products present in both mappings."""
    products = [p for p in cluster_assignments if p in attribute_values]
    if not products:
        return 0.0
    n = len(products)
    joint: dict[tuple[int, str], int] = {}
    cluster_counts: dict[int, int] = {}
    attr_counts: dict[str, int] = {}
    for p in products:
        c, a = cluster_assignments[p], attribute_values[p]
        joint[(c, a)] = joint.get((c, a), 0) + 1
        cluster_counts[c] = cluster_counts.get(c, 0) + 1
        attr_counts[a] = attr_counts.get(a, 0) + 1

    mi = 0.0
    for (c, a), count in joint.items():
        p_ca = count / n
        p_c = cluster_counts[c] / n
        p_a = attr_counts[a] / n
        if p_ca > 0:
            mi += p_ca * np.log2(p_ca / (p_c * p_a))
    return float(mi)


def _cluster_cluster_counts(
    cluster_assignments: dict[str, int], attribute_values: dict[str, str]
) -> tuple[dict[str, dict[int, int]], int]:
    """Attribute-value -> {cluster -> count} contingency for the shared products."""
    groups: dict[str, dict[int, int]] = {}
    total = 0
    for p, a in attribute_values.items():
        if p not in cluster_assignments:
            continue
        c = cluster_assignments[p]
        groups.setdefault(a, {})
        groups[a][c] = groups[a].get(c, 0) + 1
        total += 1
    return groups, total


def _entropy_of(counts: dict[int, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            h -= p * np.log2(p)
    return float(h)


def compute_entropy_gain(
    cluster_assignments: dict[str, int],
    products: list[str],
    attribute_values: dict[str, str],
) -> float:
    """Information-gain style entropy reduction of cluster entropy after split."""
    groups, total = _cluster_cluster_counts(
        {p: cluster_assignments[p] for p in products if p in cluster_assignments},
        attribute_values,
    )
    if total == 0 or len(groups) < 2:
        return 0.0
    all_counts: dict[int, int] = {}
    for group in groups.values():
        for c, count in group.items():
            all_counts[c] = all_counts.get(c, 0) + count
    parent_h = _entropy_of(all_counts)
    weighted = sum(
        (sum(g.values()) / total) * _entropy_of(g) for g in groups.values()
    )
    return float(parent_h - weighted)


def compute_gini_gain(
    cluster_assignments: dict[str, int],
    products: list[str],
    attribute_values: dict[str, str],
) -> float:
    """Gini impurity reduction of cluster labels after splitting by attribute."""
    groups, total = _cluster_cluster_counts(
        {p: cluster_assignments[p] for p in products if p in cluster_assignments},
        attribute_values,
    )
    if total == 0 or len(groups) < 2:
        return 0.0
    all_counts: dict[int, int] = {}
    for group in groups.values():
        for c, count in group.items():
            all_counts[c] = all_counts.get(c, 0) + count

    def gini(counts: dict[int, int]) -> float:
        s = sum(counts.values())
        if s == 0:
            return 0.0
        return 1.0 - sum((v / s) ** 2 for v in counts.values())

    parent = gini(all_counts)
    weighted = sum((sum(g.values()) / total) * gini(g) for g in groups.values())
    return float(parent - weighted)


def compute_within_group_similarity(
    products: list[str],
    similarity_matrix: pd.DataFrame,
) -> float:
    """Mean pairwise similarity within a group (1.0 for singletons)."""
    valid = [p for p in products if p in similarity_matrix.index]
    if len(valid) < 2:
        return 1.0
    sub = similarity_matrix.loc[valid, valid].to_numpy(dtype=float)
    triu = sub[np.triu_indices(len(valid), k=1)]
    triu = triu[np.isfinite(triu)]
    return float(np.mean(triu)) if len(triu) else 0.0


def compute_attribute_split_quality(
    products: list[str],
    attribute_values: dict[str, str],
    similarity_matrix: pd.DataFrame,
    min_cluster_size: int = 3,
    criterion: str = "entropy",
    cluster_assignments: dict[str, int] | None = None,
    alpha: float = 0.5,
) -> tuple[float, dict[str, list[str]]]:
    """Score an attribute's product groups by purity/similarity.

    ``criterion`` in {"entropy", "gini", "mutual_info", "similarity", "mixed"}.
    Purity criteria need ``cluster_assignments``; ``similarity`` uses the mean
    within-group similarity gain vs. the parent group. ``mixed`` blends the two
    with weight ``alpha`` on the purity term.
    """
    groups: dict[str, list[str]] = {}
    for p in products:
        if p not in attribute_values:
            continue
        groups.setdefault(attribute_values[p], []).append(p)
    groups = {v: ps for v, ps in groups.items() if len(ps) >= min_cluster_size}
    if len(groups) < 2:
        return 0.0, groups

    if criterion in {"entropy", "gini", "mutual_info", "mixed"} and cluster_assignments is None:
        return 0.0, groups

    assert cluster_assignments is not None

    parent_sim = compute_within_group_similarity(products, similarity_matrix)
    n = len(products)
    sim_gain = float(np.mean([parent_sim - compute_within_group_similarity(v, similarity_matrix) for v in groups.values()]))

    if criterion == "similarity":
        score = sim_gain
    elif criterion == "entropy":
        score = compute_entropy_gain(cluster_assignments, products, {p: attribute_values[p] for p in products if p in attribute_values})
    elif criterion == "gini":
        score = compute_gini_gain(cluster_assignments, products, {p: attribute_values[p] for p in products if p in attribute_values})
    elif criterion == "mutual_info":
        score = compute_mutual_information(cluster_assignments, {p: attribute_values[p] for p in products if p in attribute_values})
    else:  # mixed
        purity = compute_entropy_gain(cluster_assignments, products, {p: attribute_values[p] for p in products if p in attribute_values})
        score = alpha * purity + (1.0 - alpha) * sim_gain
    return float(max(score, 0.0)), groups


def find_best_attribute_split(
    products: list[str],
    attributes_df: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    min_cluster_size: int = 3,
    candidate_attributes: list[str] | None = None,
    criterion: str = "entropy",
    cluster_assignments: dict[str, int] | None = None,
    alpha: float = 0.5,
) -> tuple[str | None, dict[str, list[str]], float]:
    """Find the attribute whose value groups best explain the cluster structure."""
    if candidate_attributes is None:
        candidate_attributes = [c for c in attributes_df.columns if c != "stockcode"]

    best_attr: str | None = None
    best_groups: dict[str, list[str]] = {}
    best_score = 0.0
    for attr in candidate_attributes:
        if attr not in attributes_df.columns:
            continue
        attr_values = attributes_df.set_index("stockcode")[attr].dropna().to_dict()
        values = {p: v for p, v in attr_values.items() if p in products}
        score, groups = compute_attribute_split_quality(
            products,
            values,
            similarity_matrix,
            min_cluster_size=min_cluster_size,
            criterion=criterion,
            cluster_assignments=cluster_assignments,
            alpha=alpha,
        )
        if score > best_score and len(groups) >= 2:
            best_score = score
            best_attr = attr
            best_groups = groups
    return best_attr, best_groups, best_score


def build_cdt_recursive(
    products: list[str],
    attributes_df: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    node_counter: list[int],
    parent_id: str | None = None,
    min_cluster_size: int = 3,
    max_depth: int = 4,
    depth: int = 0,
    exclusion: set[str] | None = None,
    criterion: str = "entropy",
    cluster_assignments: dict[str, int] | None = None,
    alpha: float = 0.5,
) -> TreeNode:
    """Recursively build the CDT; root carries a fresh shared counter."""
    if exclusion is None:
        exclusion = set()
    node_counter[0] += 1
    node = TreeNode(
        node_id=f"node_{node_counter[0]}",
        name=f"Leaf ({len(products)} products)",
        products=list(products),
        similarity_within=compute_within_group_similarity(products, similarity_matrix),
        size=len(products),
        is_leaf=True,
        parent_id=parent_id,
    )

    if len(products) < min_cluster_size or depth >= max_depth:
        return node

    candidate_attrs = [c for c in attributes_df.columns if c != "stockcode" and c not in exclusion]
    attr, groups, score = find_best_attribute_split(
        products,
        attributes_df,
        similarity_matrix,
        min_cluster_size=min_cluster_size,
        candidate_attributes=candidate_attrs,
        criterion=criterion,
        cluster_assignments=cluster_assignments,
        alpha=alpha,
    )
    if attr is None or score <= 0.0:
        return node

    node.attribute = attr
    node.is_leaf = False
    node.split_score = score
    node.name = f"Split on {attr}"
    next_exclusion = exclusion | {attr}
    for value in sorted(groups):
        child = build_cdt_recursive(
            groups[value],
            attributes_df,
            similarity_matrix,
            node_counter,
            parent_id=node.node_id,
            min_cluster_size=min_cluster_size,
            max_depth=max_depth,
            depth=depth + 1,
            exclusion=next_exclusion,
            criterion=criterion,
            cluster_assignments=cluster_assignments,
            alpha=alpha,
        )
        child.attribute_value = value
        node.children.append(child)
    node.products = []
    return node


def build_cdt(
    attributes_df: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    cluster_assignments: dict[str, int] | None = None,
    *,
    min_cluster_size: int = 3,
    max_depth: int = 4,
    criterion: str = "entropy",
    alpha: float = 0.5,
) -> TreeNode:
    """Build a CDT over all catalog products."""
    products = attributes_df["stockcode"].tolist()
    return build_cdt_recursive(
        products,
        attributes_df,
        similarity_matrix,
        [0],
        min_cluster_size=min_cluster_size,
        max_depth=max_depth,
        criterion=criterion,
        cluster_assignments=cluster_assignments,
        alpha=alpha,
    )


def count_nodes(root: TreeNode) -> int:
    return 1 + sum(count_nodes(c) for c in root.children)


def count_leaves(root: TreeNode) -> int:
    if root.is_leaf:
        return 1
    return sum(count_leaves(c) for c in root.children)


def max_depth(root: TreeNode) -> int:
    if root.is_leaf:
        return 0
    return 1 + max(max_depth(c) for c in root.children)


def score_tree(root: TreeNode, similarity_matrix: pd.DataFrame) -> pd.DataFrame:
    """Score rows: coherence of leaves and tree coverage."""
    nodes = count_nodes(root)
    leaves = count_leaves(root)
    leaf_scores = []
    total_products = 0
    for leaf in _iter_nodes(root):
        if leaf.is_leaf:
            leaf_scores.append(leaf.similarity_within)
            total_products += len(leaf.products)
    mean_leaf_sim = float(np.mean(leaf_scores)) if leaf_scores else 0.0
    rows = [
        {"metric": "n_nodes", "value": float(nodes)},
        {"metric": "n_leaves", "value": float(leaves)},
        {"metric": "depth", "value": float(max_depth(root))},
        {"metric": "mean_leaf_similarity", "value": mean_leaf_sim},
        {"metric": "products_covered", "value": float(total_products)},
    ]
    return check(pd.DataFrame(rows, columns=list(CDT_TREE_SCORE.columns)), CDT_TREE_SCORE)


def _iter_nodes(root: TreeNode) -> Iterable[TreeNode]:
    yield root
    for child in root.children:
        yield from _iter_nodes(child)


def tree_to_dataframe(root: TreeNode) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten the tree into node and product-membership tables."""
    node_rows: list[dict[str, object]] = []
    product_rows: list[dict[str, str]] = []
    for node in _iter_nodes(root):
        node_rows.append(
            {
                "node_id": node.node_id,
                "name": node.name,
                "attribute": node.attribute if node.attribute is not None else "",
                "attribute_value": node.attribute_value if node.attribute_value is not None else "",
                "size": node.size if node.size else len(node.products),
                "is_leaf": 1 if node.is_leaf else 0,
                "similarity_within": float(node.similarity_within),
                "parent_id": node.parent_id if node.parent_id is not None else "",
            }
        )
        for product in node.products:
            product_rows.append({"node_id": node.node_id, "stockcode": product})
    nodes = pd.DataFrame(node_rows, columns=list(CDT_TREE_NODES.columns))
    products = pd.DataFrame(product_rows, columns=list(CDT_TREE_PRODUCTS.columns))
    return check(nodes, CDT_TREE_NODES), check(products, CDT_TREE_PRODUCTS, allow_empty=True)


def tree_to_json(root: TreeNode) -> str:
    """Serialize the tree (including children) to JSON."""
    return json.dumps(root.to_dict())


def prune_tree(
    root: TreeNode, threshold: float = 0.60, similarity_matrix: pd.DataFrame | None = None
) -> TreeNode:
    """Collapse leaves whose within-group similarity falls below a threshold.

    Children of a collapsed internal node take its place recursively; the
    node itself becomes a leaf.
    """
    if root.is_leaf:
        return root
    for child in root.children[:]:
        prune_tree(child, threshold, similarity_matrix)
    if root.children and all(c.is_leaf for c in root.children):
        combined_score = compute_within_group_similarity(
            [p for c in root.children for p in c.products],
            similarity_matrix if similarity_matrix is not None else pd.DataFrame(),
        ) if similarity_matrix is not None else 0.0
        if combined_score < threshold:
            combined: list[str] = []
            attrs: list[str | None] = []
            vals: list[str | None] = []
            for c in root.children:
                combined.extend(c.products)
                attrs.extend([c.attribute] * max(len(c.products), 1))
                vals.extend([c.attribute_value] * max(len(c.products), 1))
            root.children = []
            root.is_leaf = True
            root.products = combined
            root.similarity_within = (
                compute_within_group_similarity(combined, similarity_matrix)
                if similarity_matrix is not None
                else 0.0
            )
            root.size = len(combined)
    return root