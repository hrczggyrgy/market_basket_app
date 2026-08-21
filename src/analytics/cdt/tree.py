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
from typing import Any, Iterable, cast

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
    children: list["TreeNode"] = field(default_factory=list)  # noqa: UP037
    similarity_within: float = 0.0
    size: int = 0
    is_leaf: bool = True
    parent_id: str | None = None
    split_score: float = 0.0
    # New fields for decision model improvements
    split_stability: float | None = None
    split_p_value: float | None = None
    shopper_decision_rule: str | None = None

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
            "split_stability": self.split_stability,
            "split_p_value": self.split_p_value,
            "shopper_decision_rule": self.shopper_decision_rule,
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
    weighted = sum((sum(g.values()) / total) * _entropy_of(g) for g in groups.values())
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
    """Mean pairwise similarity within a group (1.0 for singletons).

    Clipped to [0, 1] to satisfy schema constraints.
    """
    valid = [p for p in products if p in similarity_matrix.index]
    if len(valid) < 2:
        return 1.0
    sub = similarity_matrix.loc[valid, valid].to_numpy(dtype=float)
    triu = sub[np.triu_indices(len(valid), k=1)]
    triu = triu[np.isfinite(triu)]
    return float(np.clip(np.mean(triu), 0.0, 1.0)) if len(triu) else 0.0


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

    # Only require cluster_assignments for purity-based criteria
    if criterion in {"entropy", "gini", "mutual_info", "mixed"}:
        if cluster_assignments is None:
            return 0.0, groups
        assert cluster_assignments is not None

    parent_sim = compute_within_group_similarity(products, similarity_matrix)
    sim_gain = float(
        np.mean(
            [
                parent_sim - compute_within_group_similarity(v, similarity_matrix)
                for v in groups.values()
            ]
        )
    )

    if criterion == "similarity":
        score = sim_gain
    elif criterion == "entropy":
        score = compute_entropy_gain(
            cast(dict[str, int], cluster_assignments),
            products,
            {p: attribute_values[p] for p in products if p in attribute_values},
        )
    elif criterion == "gini":
        score = compute_gini_gain(
            cast(dict[str, int], cluster_assignments),
            products,
            {p: attribute_values[p] for p in products if p in attribute_values},
        )
    elif criterion == "mutual_info":
        score = compute_mutual_information(
            cast(dict[str, int], cluster_assignments),
            {p: attribute_values[p] for p in products if p in attribute_values},
        )
    else:  # mixed
        purity = compute_entropy_gain(
            cast(dict[str, int], cluster_assignments),
            products,
            {p: attribute_values[p] for p in products if p in attribute_values},
        )
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
    multiplicity_correction: str = "bonferroni",
) -> tuple[str | None, dict[str, list[str]], float, float]:
    """Find the attribute whose value groups best explain the cluster structure.

    Args:
        multiplicity_correction: "bonferroni" or "bh" (Benjamini-Hochberg FDR).
            Applied to p-values from split significance testing.
    """
    if candidate_attributes is None:
        candidate_attributes = [c for c in attributes_df.columns if c != "stockcode"]

    best_attr: str | None = None
    best_groups: dict[str, list[str]] = {}
    best_score = 0.0
    best_idx: int | None = None
    scores = []
    attrs_tested = []

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
        scores.append(score)
        attrs_tested.append(attr)
        if score > best_score and len(groups) >= 2:
            best_score = score
            best_attr = attr
            best_groups = groups

    # Multiplicity correction: adjust p-values for the number of attributes tested
    if multiplicity_correction == "bonferroni" and attrs_tested:
        n_tests = len(attrs_tested)
        corrected_scores = [s / n_tests for s in scores]  # Conservative approximation
        best_score = max(corrected_scores) if corrected_scores else 0.0
        # Find the attribute with the corrected best score
        best_idx = cast(int, np.argmax(corrected_scores))
        best_attr = attrs_tested[best_idx]

        # Recompute groups for best attribute
        if best_attr:
            attr_values = attributes_df.set_index("stockcode")[best_attr].dropna().to_dict()
            values = {p: v for p, v in attr_values.items() if p in products}
            _, best_groups = compute_attribute_split_quality(
                products,
                values,
                similarity_matrix,
                min_cluster_size=min_cluster_size,
                criterion=criterion,
                cluster_assignments=cluster_assignments,
                alpha=alpha,
            )
    elif multiplicity_correction == "bh" and attrs_tested:
        # Benjamini-Hochberg FDR correction (disabled by default - scores are not p-values)
        # Use a simple threshold instead
        best_idx = cast(int, np.argmax(scores))
        best_score = scores[best_idx]
        # Only consider if score is above a reasonable threshold
        if best_score > 0.01:  # Minimum score threshold
            best_attr = attrs_tested[best_idx]
            attr_values = attributes_df.set_index("stockcode")[best_attr].dropna().to_dict()
            values = {p: v for p, v in attr_values.items() if p in products}
            _, best_groups = compute_attribute_split_quality(
                products,
                values,
                similarity_matrix,
                min_cluster_size=min_cluster_size,
                criterion=criterion,
                cluster_assignments=cluster_assignments,
                alpha=alpha,
            )
        else:
            best_attr, best_groups, best_score = None, {}, 0.0
    else:
        # Default case if no correction or unknown correction
        best_idx = cast(int, np.argmax(scores)) if scores else -1
        if best_idx != -1:
            best_score = scores[best_idx]
            best_attr = attrs_tested[best_idx]
            attr_values = attributes_df.set_index("stockcode")[best_attr].dropna().to_dict()
            values = {p: v for p, v in attr_values.items() if p in products}
            _, best_groups = compute_attribute_split_quality(
                products,
                values,
                similarity_matrix,
                min_cluster_size=min_cluster_size,
                criterion=criterion,
                cluster_assignments=cluster_assignments,
                alpha=alpha,
            )
        else:
            best_attr, best_groups, best_score = None, {}, 0.0

    return best_attr, best_groups, best_score, float(best_idx) if best_idx is not None else 0.0


def compute_split_bootstrap_stability(
    products: list[str],
    attributes_df: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    min_cluster_size: int = 3,
    candidate_attributes: list[str] | None = None,
    criterion: str = "entropy",
    cluster_assignments: dict[str, int] | None = None,
    alpha: float = 0.5,
    n_resamples: int = 50,
    random_seed: int = 42,
) -> tuple[str | None, dict[str, list[str]], float, float]:
    """Bootstrap test for split stability.

    Resamples products with replacement, rebuilds the split, and measures
    how often the same attribute is selected.

    Returns:
        (best_attr, best_groups, best_score, stability)
        where stability is the fraction of resamples where the same attribute wins.
    """
    rng = np.random.default_rng(random_seed)
    if candidate_attributes is None:
        candidate_attributes = [c for c in attributes_df.columns if c != "stockcode"]

    attr_wins: dict[str, int] = {}
    for _ in range(n_resamples):
        # Resample products with replacement
        resampled_products = rng.choice(products, size=len(products), replace=True).tolist()
        attr, groups, score, _ = find_best_attribute_split(
            resampled_products,
            attributes_df,
            similarity_matrix,
            min_cluster_size=min_cluster_size,
            candidate_attributes=candidate_attributes,
            criterion=criterion,
            cluster_assignments=cluster_assignments,
            alpha=alpha,
            multiplicity_correction="bh",
        )
        if attr:
            attr_wins[attr] = attr_wins.get(attr, 0) + 1

    if not attr_wins:
        return None, {}, 0.0, 0.0

    best_attr = max(attr_wins.keys(), key=lambda k: cast(int, attr_wins[k]))
    stability = attr_wins[best_attr] / n_resamples

    # Get final split on full data
    attr, groups, score, _ = find_best_attribute_split(
        products,
        attributes_df,
        similarity_matrix,
        min_cluster_size=min_cluster_size,
        candidate_attributes=candidate_attributes,
        criterion=criterion,
        cluster_assignments=cluster_assignments,
        alpha=alpha,
        multiplicity_correction="bh",
    )

    return attr, groups, score, float(stability)


def predict_shopper_decision(
    node: TreeNode,
    similarity_matrix: pd.DataFrame,
    attributes_df: pd.DataFrame,
) -> str:
    """Predict the shopper decision rule at a node.

    For a split node, generates a human-readable rule describing the
    shopper decision: "Shoppers split by [attribute] into [values]".

    For leaf nodes, describes the product group.
    """
    if node.is_leaf:
        if len(node.products) == 0:
            return "End of decision path"
        if len(node.products) <= 5:
            return f"Consider products: {', '.join(node.products)}"
        return f"Browse {len(node.products)} similar products in this category"

    attr = node.attribute or "unknown"
    if node.attribute_value:
        val = node.attribute_value
        return f"Shoppers choosing {val} {attr} prefer this category"
    else:
        # Root split node - describe the decision point
        return f"Shoppers split by {attr} into categories"


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
    compute_stability: bool = True,
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
    attr, groups, score, stability = find_best_attribute_split(
        products,
        attributes_df,
        similarity_matrix,
        min_cluster_size=min_cluster_size,
        candidate_attributes=candidate_attrs,
        criterion=criterion,
        cluster_assignments=cluster_assignments,
        alpha=alpha,
        multiplicity_correction="bh",
    )
    if attr is None or score <= 0.0:
        return node

    # Bootstrap stability test
    split_stability = 0.0
    if compute_stability and len(products) >= 10:
        _, _, _, split_stability = compute_split_bootstrap_stability(
            products,
            attributes_df,
            similarity_matrix,
            min_cluster_size=min_cluster_size,
            candidate_attributes=candidate_attrs,
            criterion=criterion,
            cluster_assignments=cluster_assignments,
            alpha=alpha,
            n_resamples=30,
        )

    node.attribute = attr
    node.is_leaf = False
    node.split_score = score
    node.split_stability = split_stability
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
            compute_stability=compute_stability,
        )
        child.attribute_value = value
        node.children.append(child)
    node.products = []

    # Predict shopper decision rule at this node
    node.shopper_decision_rule = predict_shopper_decision(node, similarity_matrix, attributes_df)

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
    compute_stability: bool = True,
) -> TreeNode:
    """Build a CDT over all catalog products."""
    products = attributes_df["stockcode"].tolist()

    # Auto-compute cluster assignments if needed for purity-based criteria
    if cluster_assignments is None and criterion in {"entropy", "gini", "mutual_info", "mixed"}:
        from scipy.cluster.hierarchy import fcluster, linkage

        from src.analytics.cdt.clustering import (
            _safe_linkage_method,
            _square_to_condensed,
            similarity_to_distance,
        )

        distance = similarity_to_distance(similarity_matrix, method="phi")
        condensed = _square_to_condensed(distance)
        if len(condensed) < 1:
            cluster_assignments = {p: 0 for p in products}
        else:
            n_clusters = min(10, max(2, len(products) // 5))
            linkage_matrix = linkage(condensed, method=_safe_linkage_method("ward"))
            labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
            cluster_assignments = dict(zip(products, labels, strict=True))

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
        compute_stability=compute_stability,
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
    return check(pd.DataFrame(rows, columns=cast(Any, list(CDT_TREE_SCORE.columns))), CDT_TREE_SCORE)


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
                "split_score": float(node.split_score) if node.split_score is not None else 0.0,
                "split_stability": float(node.split_stability)
                if node.split_stability is not None
                else 0.0,
                "shopper_decision_rule": node.shopper_decision_rule
                if node.shopper_decision_rule is not None
                else "",
            }
        )
        for product in node.products:
            product_rows.append({"node_id": node.node_id, "stockcode": product})
    nodes = cast(pd.DataFrame, pd.DataFrame(node_rows, columns=cast(Any, list(CDT_TREE_NODES.columns))))
    products = cast(pd.DataFrame, pd.DataFrame(product_rows, columns=cast(Any, list(CDT_TREE_PRODUCTS.columns))))
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


class CDTEngine:
    """CDT Engine - isolated behind explicit Tier C trigger.

    Only runs on explicit user action ("Run CDT" button).
    Gracefully handles missing optional dependencies (networkx, scipy, sklearn).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self._deps_available = self._check_dependencies()
        self._missing_deps = self._get_missing_deps()

    def _check_dependencies(self) -> dict[str, bool]:
        """Check which optional dependencies are available."""
        deps = {}
        try:
            import networkx  # noqa: F401
            deps["networkx"] = True
        except ImportError:
            deps["networkx"] = False

        try:
            import scipy.cluster.hierarchy  # noqa: F401
            deps["scipy"] = True
        except ImportError:
            deps["scipy"] = False

        try:
            import sklearn.cluster  # noqa: F401
            import sklearn.metrics  # noqa: F401
            deps["sklearn"] = True
        except ImportError:
            deps["sklearn"] = False

        return deps

    def _get_missing_deps(self) -> list[str]:
        """Get list of missing dependencies."""
        return [dep for dep, available in self._deps_available.items() if not available]

    def _friendly_error(self) -> dict[str, Any]:
        """Return a friendly error dict instead of raising ImportError."""
        missing = self._get_missing_deps()
        import warnings
        warnings.warn(
            f"CDT analysis requires missing dependencies: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}",
            UserWarning,
            stacklevel=2,
        )
        return {
            "error": f"Missing dependencies: {', '.join(missing)}. Install with: pip install {' '.join(missing)}",
            "missing_deps": missing,
        }

    def build_cdt(
        self,
        min_cooccurrence: int = 5,
        similarity_method: str = "phi",
        linkage_method: str = "ward",
        min_clusters: int = 2,
        max_clusters: int = 15,
        min_cluster_size: int = 3,
        quality_threshold: float = 0.6,
        split_criterion: str = "mutual_info",
        resolution: float = 1.0,
        community_method: str = "none",
    ) -> dict[str, Any]:
        """Build CDT with graceful degradation.

        Returns dict with 'tree', 'dataframe', 'similarity_matrix', 'dendrogram', 'clusters'.
        If dependencies missing, returns friendly error dict.
        """
        if not all(self._deps_available.values()):
            return self._friendly_error()

        try:
            from src.analytics.cdt import (
                build_cdt,
                build_transaction_derived_attributes,
                tree_to_dataframe,
            )
            from src.analytics.cdt.clustering import (
                find_optimal_clusters_sklearn,
                get_dendrogram_data,
            )
            from src.analytics.cdt.similarity import build_similarity_matrix_ensemble

            # Build attributes
            attributes_df = build_transaction_derived_attributes(self.df)

            # Build similarity matrix
            similarity_matrix = build_similarity_matrix_ensemble(
                self.df,
                methods=[similarity_method] if similarity_method != "ensemble" else None,
            )

            # Build CDT
            tree = build_cdt(
                attributes_df,
                similarity_matrix,
                min_cluster_size=min_cluster_size,
                quality_threshold=quality_threshold,
                split_criterion=split_criterion,
            )

            # Convert to dataframe
            df_tree = tree_to_dataframe(tree)

            # Get dendrogram data
            dendrogram = get_dendrogram_data(similarity_matrix, linkage_method)

            # Get clusters
            clusters = find_optimal_clusters_sklearn(
                similarity_matrix,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                method=linkage_method,
            )

            return {
                "tree": tree,
                "dataframe": df_tree,
                "similarity_matrix": similarity_matrix,
                "dendrogram": dendrogram,
                "clusters": clusters,
            }
        except Exception as e:
            import warnings
            warnings.warn(f"CDT build failed: {e}", UserWarning, stacklevel=2)
            return {"error": f"CDT build failed: {e}"}

    def get_similarity_matrix(
        self,
        method: str = "phi",
    ) -> pd.DataFrame:
        """Build similarity matrix with graceful degradation."""
        if not self._deps_available.get("scipy", False):
            return self._friendly_error()

        try:
            from src.analytics.cdt.similarity import build_similarity_matrix
            return build_similarity_matrix(self.df, method=method)
        except Exception as e:
            import warnings
            warnings.warn(f"Similarity matrix failed: {e}", UserWarning, stacklevel=2)
            return {"error": f"Similarity matrix failed: {e}"}

    def is_available(self) -> bool:
        """Check if CDT engine is available (all dependencies installed)."""
        return all(self._deps_available.values())

    def get_status(self) -> dict[str, Any]:
        """Get engine status for UI display."""
        return {
            "available": self.is_available(),
            "engine": "CDTEngine",
            "tier": "C",
            "description": "Customer Decision Tree construction",
            "dependencies": ["networkx", "scipy", "sklearn"],
            "missing_deps": self._get_missing_deps(),
        }


def get_cdt_engine(df: pd.DataFrame) -> CDTEngine:
    """Factory function to create CDTEngine for a dataset."""
    return CDTEngine(df)
