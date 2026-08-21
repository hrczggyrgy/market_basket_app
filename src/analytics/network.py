"""Network analysis engine for market basket app.

Provides network-based analysis of product relationships, customer flows,
and assortment optimization using graph algorithms.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class NetworkEngine:
    """Network Engine - isolated behind explicit Tier C trigger.

    Only runs on explicit user action ("Run Network Analysis" button).
    Gracefully handles missing optional dependencies (networkx).
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
            import networkx as nx  # noqa: F401
            deps["networkx"] = True
        except ImportError:
            deps["networkx"] = False
        return deps

    def _get_missing_deps(self) -> list[str]:
        """Get list of missing dependencies."""
        return [dep for dep, available in self._deps_available.items() if not available]

    def _friendly_error(self) -> dict[str, Any]:
        """Return a friendly error dict instead of raising ImportError."""
        missing = self._get_missing_deps()
        import warnings
        warnings.warn(
            f"Network analysis requires missing dependencies: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}",
            UserWarning,
            stacklevel=2,
        )
        return {
            "error": f"Missing dependencies: {', '.join(missing)}. Install with: pip install {' '.join(missing)}",
            "missing_deps": missing,
        }

    def build_co_purchase_network(
        self,
        top_n_products: int = 100,
        min_cooccurrence: int = 5,
        min_affinity: float = 0.0,
    ) -> dict[str, Any]:
        """Build co-purchase network from transaction data.

        Returns network metrics, centrality scores, and community structure.
        """
        if not self._deps_available.get("networkx", False):
            return self._friendly_error()

        try:
            import networkx as nx

            from src.analytics.copurchase import get_top_affinity_pairs

            # Get affinity pairs
            pairs = get_top_affinity_pairs(
                self.df,
                top_n=10000,
                min_cooccurrence=min_cooccurrence,
                top_n_products=top_n_products,
            )

            if pairs.empty:
                return {"error": "No co-purchase pairs found", "nodes": [], "edges": []}

            pairs = pairs[pairs["affinity"] >= min_affinity].head(1000)

            # Build network
            G = nx.Graph()

            for _, row in pairs.iterrows():
                G.add_edge(row["product_a"], row["product_b"],
                          weight=row["affinity"],
                          cooccurrence=row["cooccurrence"])

            # Compute centrality metrics
            pagerank = nx.pagerank(G, weight="weight")
            betweenness = nx.betweenness_centrality(G, weight="weight")
            degree = dict(G.degree())

            # Community detection
            try:
                communities = nx.community.greedy_modularity_communities(G, weight="weight")
                community_map = {}
                for i, comm in enumerate(communities):
                    for node in comm:
                        community_map[node] = i
            except Exception:
                community_map = {}

            # Node attributes
            nodes = []
            for node in G.nodes():
                nodes.append({
                    "stockcode": node,
                    "pagerank": pagerank.get(node, 0),
                    "betweenness": betweenness.get(node, 0),
                    "degree": degree.get(node, 0),
                    "community": community_map.get(node, -1),
                })

            # Edge attributes
            edges = []
            for u, v, data in G.edges(data=True):
                edges.append({
                    "source": u,
                    "target": v,
                    "weight": data.get("weight", 1),
                    "cooccurrence": data.get("cooccurrence", 0),
                })

            return {
                "nodes": nodes,
                "edges": edges,
                "n_nodes": G.number_of_nodes(),
                "n_edges": G.number_of_edges(),
                "density": nx.density(G),
            }
        except Exception as e:
            import warnings
            warnings.warn(f"Co-purchase network failed: {e}", UserWarning, stacklevel=2)
            return {"error": f"Co-purchase network failed: {e}"}

    def build_switching_network(
        self,
        window_days: int = 90,
        min_transactions: int = 3,
    ) -> dict[str, Any]:
        """Build switching network from customer sequences.

        Nodes = products, edges = observed switches with probabilities.
        """
        if not self._deps_available.get("networkx", False):
            return self._friendly_error()

        try:
            import networkx as nx

            from src.analytics.switching import compute_switching_matrix

            matrix = compute_switching_matrix(self.df, window_days, min_transactions)

            if matrix.empty:
                return {"error": "No switching data found", "nodes": [], "edges": []}

            G = nx.DiGraph()

            for _, row in matrix.iterrows():
                G.add_edge(row["from_product"], row["to_product"],
                          weight=row["pct"], count=row["count"])

            # Compute centrality
            pagerank = nx.pagerank(G, weight="weight")
            betweenness = nx.betweenness_centrality(G, weight="weight")
            in_degree = dict(G.in_degree())
            out_degree = dict(G.out_degree())

            nodes = []
            for node in G.nodes():
                nodes.append({
                    "stockcode": node,
                    "pagerank": pagerank.get(node, 0),
                    "betweenness": betweenness.get(node, 0),
                    "in_degree": in_degree.get(node, 0),
                    "out_degree": out_degree.get(node, 0),
                })

            edges = []
            for u, v, data in G.edges(data=True):
                edges.append({
                    "source": u,
                    "target": v,
                    "probability": data.get("weight", 0),
                    "count": data.get("count", 0),
                })

            return {
                "nodes": nodes,
                "edges": edges,
                "n_nodes": G.number_of_nodes(),
                "n_edges": G.number_of_edges(),
            }
        except Exception as e:
            import warnings
            warnings.warn(f"Switching network failed: {e}", UserWarning, stacklevel=2)
            return {"error": f"Switching network failed: {e}"}

    def get_network_summary(self) -> dict[str, Any]:
        """Get summary of available network analyses."""
        return {
            "available": self._deps_available.get("networkx", False),
            "engine": "NetworkEngine",
            "tier": "C",
            "description": "Network analysis of product relationships",
            "dependencies": ["networkx"],
            "missing_deps": self._get_missing_deps(),
            "analyses": [
                "co_purchase_network",
                "switching_network",
            ],
        }

    def is_available(self) -> bool:
        """Check if Network engine is available."""
        return self._deps_available.get("networkx", False)


def get_network_engine(df: pd.DataFrame) -> NetworkEngine:
    """Factory function to create NetworkEngine for a dataset."""
    return NetworkEngine(df)
