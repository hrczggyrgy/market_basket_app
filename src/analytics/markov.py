"""Markov chain analysis engine for market basket app.

Provides Markov chain modeling of customer purchase sequences,
transition probabilities, and steady-state analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class MarkovEngine:
    """Markov Engine - isolated behind explicit Tier C trigger.

    Only runs on explicit user action ("Run Markov Analysis" button).
    Gracefully handles missing optional dependencies.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self._deps_available = True  # No external deps needed for basic Markov

    def _check_dependencies(self) -> dict[str, bool]:
        """Check which optional dependencies are available."""
        return {"numpy": True, "pandas": True}

    def build_transition_matrix(
        self,
        window_days: int = 90,
        min_transactions: int = 3,
        add_absorbing_state: bool = True,
    ) -> pd.DataFrame:
        """Build Markov transition matrix from customer sequences.

        Returns row-normalized transition probability matrix P(to | from).
        Includes optional absorbing "no_switch" state.
        """
        from src.analytics.switching import compute_transition_matrix

        matrix = compute_transition_matrix(
            self.df,
            window_days=window_days,
            min_transactions=min_transactions,
            normalize=True,
        )

        if matrix.empty:
            return pd.DataFrame()

        return matrix

    def compute_steady_state(
        self,
        transition_matrix: pd.DataFrame,
        max_iterations: int = 1000,
        tolerance: float = 1e-10,
    ) -> pd.Series:
        """Compute steady-state distribution of Markov chain.

        Solves π = πP for the stationary distribution.
        """
        if transition_matrix.empty:
            return pd.Series(dtype=float)

        P = transition_matrix.values
        n = P.shape[0]

        # Power iteration method
        pi = np.ones(n) / n  # Uniform initial distribution

        for _ in range(max_iterations):
            pi_new = pi @ P
            if np.max(np.abs(pi_new - pi)) < tolerance:
                pi = pi_new
                break
            pi = pi_new

        return pd.Series(pi, index=transition_matrix.index, name="steady_state")

    def compute_absorption_probabilities(
        self,
        transition_matrix: pd.DataFrame,
        transient_states: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute absorption probabilities for Markov chain with absorbing states.

        Returns probability of being absorbed in each absorbing state
        starting from each transient state.
        """
        if transition_matrix.empty:
            return pd.DataFrame()

        # Find absorbing states (states that transition to themselves with prob 1)
        absorbing = []
        transient = []

        for state in transition_matrix.index:
            row = transition_matrix.loc[state]
            if row.get(state, 0) >= 1.0 - 1e-10:
                absorbing.append(state)
            else:
                transient.append(state)

        if transient_states is not None:
            transient = [s for s in transient if s in transient_states]

        if not transient or not absorbing:
            return pd.DataFrame()

        P = transition_matrix.values
        len(transition_matrix.index)

        # Reorder: transient first, then absorbing
        state_order = transient + absorbing
        {state: i for i, state in enumerate(state_order)}
        P_reordered = P[np.ix_(state_order, state_order)]

        t = len(transient)
        len(absorbing)

        Q = P_reordered[:t, :t]  # Transient to transient
        R = P_reordered[:t, t:]  # Transient to absorbing

        # Fundamental matrix N = (I - Q)^(-1)
        try:
            N = np.linalg.inv(np.eye(t) - Q)
            B = N @ R  # Absorption probabilities
        except np.linalg.LinAlgError:
            return pd.DataFrame()

        return pd.DataFrame(B, index=transient, columns=absorbing)

    def compute_expected_steps_to_absorption(
        self,
        transition_matrix: pd.DataFrame,
    ) -> pd.Series:
        """Compute expected steps to absorption from each transient state."""
        if transition_matrix.empty:
            return pd.Series(dtype=float)

        # Find absorbing states
        transient = []
        for state in transition_matrix.index:
            row = transition_matrix.loc[state]
            if row.get(state, 0) < 1.0 - 1e-10:
                transient.append(state)

        if not transient:
            return pd.Series(dtype=float)

        P = transition_matrix.values
        len(transition_matrix.index)
        state_order = transient + [s for s in transition_matrix.index if s not in transient]
        {state: i for i, state in enumerate(state_order)}
        P_reordered = P[np.ix_(state_order, state_order)]

        t = len(transient)
        Q = P_reordered[:t, :t]

        try:
            N = np.linalg.inv(np.eye(t) - Q)
            expected_steps = N.sum(axis=1)
        except np.linalg.LinAlgError:
            return pd.Series(dtype=float)

        return pd.Series(expected_steps, index=transient, name="expected_steps")

    def simulate_chain(
        self,
        transition_matrix: pd.DataFrame,
        start_state: str,
        n_steps: int = 100,
        n_simulations: int = 1,
        random_seed: int = 42,
    ) -> list[list[str]]:
        """Simulate Markov chain paths."""
        if transition_matrix.empty or start_state not in transition_matrix.index:
            return []

        rng = np.random.default_rng(random_seed)
        states = transition_matrix.index.tolist()
        state_to_idx = {s: i for i, s in enumerate(states)}

        paths = []
        for _ in range(n_simulations):
            current = start_state
            path = [current]

            for _ in range(n_steps):
                idx = state_to_idx[current]
                probs = transition_matrix.iloc[idx].values
                # Normalize to handle numerical errors
                probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(probs)) / len(probs)
                next_idx = rng.choice(len(states), p=probs)
                current = states[next_idx]
                path.append(current)

                # Check for absorbing state
                if transition_matrix.loc[current, current] >= 1.0 - 1e-10:
                    break

            paths.append(path)

        return paths

    def get_markov_summary(self) -> dict[str, Any]:
        """Get summary of available Markov analyses."""
        return {
            "available": True,
            "engine": "MarkovEngine",
            "tier": "C",
            "description": "Markov chain analysis of customer purchase sequences",
            "dependencies": ["numpy", "pandas"],
            "analyses": [
                "transition_matrix",
                "steady_state",
                "absorption_probabilities",
                "expected_steps",
                "chain_simulation",
            ],
        }

    def is_available(self) -> bool:
        """Check if Markov engine is available."""
        return True


def get_markov_engine(df: pd.DataFrame) -> MarkovEngine:
    """Factory function to create MarkovEngine for a dataset."""
    return MarkovEngine(df)
