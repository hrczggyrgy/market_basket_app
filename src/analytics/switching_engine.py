"""SwitchingEngine - single entry point for switching/transference consolidation.

Owns the switching computation and provides cached access to switching edges,
transition matrices, status, and category-level switching.  Transference consumers
should request switching_edges from this engine to avoid duplicate computation.
"""

from __future__ import annotations

import pandas as pd

from .switching import (
    compute_category_switching_matrix,
    compute_switching_matrix,
    compute_switching_status,
    compute_transition_matrix,
    get_top_switching_paths,
)


class SwitchingEngine:
    """Engine that owns switching computation for a dataset.

    Guarantees switching is computed once per dataset and provides cached
    access to all derived switching tables.  Transference consumers should
    request ``switching_edges`` from this engine instead of computing the
    switching matrix themselves.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_days: int = 90,
        min_transactions: int = 3,
        seasonal_adjustment: bool = False,
    ) -> None:
        self.df = df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.window_days = window_days
        self.min_transactions = min_transactions
        self.seasonal_adjustment = seasonal_adjustment
        self._matrix: pd.DataFrame | None = None
        self._status: pd.DataFrame | None = None
        self._transition: pd.DataFrame | None = None
        self._top_paths: pd.DataFrame | None = None
        self._category_matrix: pd.DataFrame | None = None
        self._edges: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Core switching matrix (cached)
    # ------------------------------------------------------------------

    def compute_switching_matrix(self) -> pd.DataFrame:
        """Compute and cache the switching matrix.

        Returns:
            DataFrame with columns ``from_product``, ``to_product``, ``count``, ``pct``.
        """
        if self._matrix is None:
            self._matrix = compute_switching_matrix(
                self.df,
                window_days=self.window_days,
                min_transactions=self.min_transactions,
                seasonal_adjustment=self.seasonal_adjustment,
            )
        return self._matrix

    # ------------------------------------------------------------------
    # Switching status (cached)
    # ------------------------------------------------------------------

    def compute_switching_status(self, min_customers: int = 10, min_transitions: int = 5) -> pd.DataFrame:
        """Compute and cache per-product switching estimability status.

        Returns:
            DataFrame validated against ``SWITCHING_STATUS`` contract.
        """
        if self._status is None:
            self._status = compute_switching_status(
                self.df,
                window_days=self.window_days,
                min_transactions=self.min_transactions,
                min_customers=min_customers,
                min_transitions=min_transitions,
            )
        return self._status

    # ------------------------------------------------------------------
    # Transition matrix (cached)
    # ------------------------------------------------------------------

    def compute_transition_matrix(self, normalize: bool = True) -> pd.DataFrame:
        """Compute and cache the row-normalized transition matrix.

        Args:
            normalize: If True (default), returns P(to | from) with an absorbing
                     ``no_switch`` state. If False, returns raw counts.

        Returns:
            DataFrame (pivot format) with rows=from_product, columns=to_product.
        """
        if self._transition is None:
            self._transition = compute_transition_matrix(
                self.df,
                window_days=self.window_days,
                min_transactions=self.min_transactions,
                normalize=normalize,
            )
        return self._transition

    # ------------------------------------------------------------------
    # Top switching paths (cached)
    # ------------------------------------------------------------------

    def get_top_switching_paths(self, top_n: int = 20) -> pd.DataFrame:
        """Compute and cache the top N switching paths by count.

        Returns:
            DataFrame validated against ``SWITCHING_MATRIX`` contract.
        """
        if self._top_paths is None:
            self._top_paths = get_top_switching_paths(
                self.df,
                top_n=top_n,
                window_days=self.window_days,
                min_transactions=self.min_transactions,
            )
        return self._top_paths

    # ------------------------------------------------------------------
    # Category-level switching matrix (cached)
    # ------------------------------------------------------------------

    def compute_category_switching_matrix(
        self,
        product_lookup: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute and cache category-level switching matrix.

        Returns:
            DataFrame validated against ``CATEGORY_SWITCHING`` contract.
        """
        if self._category_matrix is None:
            self._category_matrix = compute_category_switching_matrix(
                self.df,
                window_days=self.window_days,
                min_transactions=self.min_transactions,
                product_lookup=product_lookup,
            )
        return self._category_matrix

    # ------------------------------------------------------------------
    # Sparse switching edges (cached, single computation)
    # ------------------------------------------------------------------

    def get_switching_edges(self) -> pd.DataFrame:
        """Return the sparse edge table ``(from_product, to_product, count, pct)``.

        This is the single source of switching edges for the dataset.  Transference
        consumers must use this edge table instead of recomputing the switching matrix.

        Returns:
            DataFrame with columns ``from_product``, ``to_product``, ``count``, ``pct``.
            Empty DataFrame if no switching was observed.
        """
        if self._edges is None:
            matrix = self.compute_switching_matrix()
            if matrix.empty:
                self._edges = matrix
            else:
                # Ensure canonical column order and types
                self._edges = matrix[
                    ["from_product", "to_product", "count", "pct"]
                ].copy()
                # Guarantee pct is float
                self._edges["pct"] = self._edges["pct"].astype(float)
        return self._edges

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def get_switching_summary(self) -> dict[str, int | float]:
        """Return summary statistics about switching behavior.

        Returns:
            Dict with keys: ``n_switching_pairs``, ``n_unique_products``,
            ``total_switches``, ``n_products_with_switching``.
        """
        edges = self.get_switching_edges()
        if edges.empty:
            return {
                "n_switching_pairs": 0,
                "n_unique_products": 0,
                "total_switches": 0,
                "n_products_with_switching": 0,
            }
        n_pairs = len(edges)
        n_products = len(
            set(edges["from_product"].unique()) | set(edges["to_product"].unique())
        )
        total_switches = int(edges["count"].sum())
        n_prods = len(
            set(edges["from_product"].unique()) | set(edges["to_product"].unique())
        )
        return {
            "n_switching_pairs": n_pairs,
            "n_unique_products": n_products,
            "total_switches": total_switches,
            "n_products_with_switching": n_prods,
        }

    # ------------------------------------------------------------------
    # Transition probabilities (sparse)
    # ------------------------------------------------------------------

    def get_transition_probabilities(self) -> pd.DataFrame:
        """Return sparse transition probabilities P(to | from).

        Returns:
            DataFrame with columns ``from_product``, ``to_product``, ``probability``.
            Includes the absorbing ``no_switch`` row if applicable.
        """
        matrix = self.compute_transition_matrix(normalize=True)
        if matrix.empty:
            return pd.DataFrame(columns=["from_product", "to_product", "probability"])
        # Convert pivot to long format
        probs = matrix.stack().reset_index()
        probs = probs.rename(columns={"level_0": "from_product", "level_1": "to_product", 0: "probability"})
        return probs


def get_switching_edges(df: pd.DataFrame, *, window_days: int = 90, min_transactions: int = 3) -> pd.DataFrame:
    """Standalone function: return sparse edge table (from_product, to_product, count, pct).

    Args:
        df: Transaction data
        window_days: Maximum gap between consecutive purchases to consider a switch
        min_transactions: Minimum transactions per customer to include

    Returns:
        DataFrame with columns ``from_product``, ``to_product``, ``count``, ``pct``.
    """
    from .switching import compute_switching_matrix

    matrix = compute_switching_matrix(df, window_days=window_days, min_transactions=min_transactions)
    if matrix.empty:
        return matrix
    return matrix[["from_product", "to_product", "count", "pct"]].copy()


def get_switching_summary(df: pd.DataFrame, *, window_days: int = 90, min_transactions: int = 3) -> dict[str, int | float]:
    """Standalone function: return summary stats about switching behavior.

    Args:
        df: Transaction data
        window_days: Maximum gap between consecutive purchases to consider a switch
        min_transactions: Minimum transactions per customer to include

    Returns:
        Dict with keys: ``n_switching_pairs``, ``n_unique_products``,
        ``total_switches``, ``n_products_with_switching``.
    """
    edges = get_switching_edges(df, window_days=window_days, min_transactions=min_transactions)
    if edges.empty:
        return {
            "n_switching_pairs": 0,
            "n_unique_products": 0,
            "total_switches": 0,
            "n_products_with_switching": 0,
        }
    n_pairs = len(edges)
    n_products = len(
        set(edges["from_product"].unique()) | set(edges["to_product"].unique())
    )
    total_switches = int(edges["count"].sum())
    return {
        "n_switching_pairs": n_pairs,
        "n_unique_products": n_products,
        "total_switches": total_switches,
        "n_products_with_switching": n_products,
    }


def get_transition_probabilities(df: pd.DataFrame, *, window_days: int = 90, min_transactions: int = 3) -> pd.DataFrame:
    """Standalone function: return sparse transition probabilities P(to | from).

    Args:
        df: Transaction data
        window_days: Maximum gap between consecutive purchases to consider a switch
        min_transactions: Minimum transactions per customer to include

    Returns:
        DataFrame with columns ``from_product``, ``to_product``, ``probability``.
    """
    from .switching import compute_transition_matrix

    matrix = compute_transition_matrix(df, window_days=window_days, min_transactions=min_transactions, normalize=True)
    if matrix.empty:
        return pd.DataFrame(columns=["from_product", "to_product", "probability"])
    probs = matrix.stack().reset_index()
    probs = probs.rename(columns={"level_0": "from_product", "level_1": "to_product", 0: "probability"})
    return probs
