"""Bootstrap resampling utilities for uncertainty quantification.

Provides reusable helpers to compute percentile bootstrap confidence
intervals for any statistic derived from transaction data.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def bootstrap_ci(
    data: pd.DataFrame,
    statistic_fn: Callable[[pd.DataFrame], float],
    *,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> Dict[str, float]:
    """Compute percentile bootstrap confidence interval for a scalar statistic.

    Parameters
    ----------
    data : pd.DataFrame
        Original dataset.
    statistic_fn : callable
        Function that takes a DataFrame (resample) and returns a float.
    n_resamples : int
        Number of bootstrap replicates.
    ci_level : float
        Confidence level (e.g. 0.95 for 95 % CI).
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``'estimate'`` (point estimate on original data),
        ``'lower'``, ``'upper'`` (percentile CI bounds),
        ``'std_error'`` (bootstrap standard error),
        ``'n_resamples'`` (actual number of successful replicates).
    """
    rng = np.random.default_rng(random_seed)
    point = statistic_fn(data)

    n = len(data)
    replicates: List[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resample = data.iloc[idx]
        try:
            replicates.append(statistic_fn(resample))
        except Exception:
            continue

    if not replicates:
        return {
            "estimate": point,
            "lower": point,
            "upper": point,
            "std_error": 0.0,
            "n_resamples": 0,
        }

    arr = np.array(replicates)
    alpha = 1.0 - ci_level
    lower = float(np.percentile(arr, 100 * alpha / 2))
    upper = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    std_err = float(arr.std(ddof=1))

    return {
        "estimate": point,
        "lower": lower,
        "upper": upper,
        "std_error": std_err,
        "n_resamples": len(replicates),
    }


def bootstrap_transaction_samples(
    transactions_df: pd.DataFrame,
    n_resamples: int = 1000,
    random_seed: Optional[int] = None,
):
    """Yield bootstrap resamples of transaction data (generator).

    Each resample is a DataFrame of the same length, drawn with replacement
    at the **customer level** (resample customers, then take all their
    transactions) to preserve within-customer correlation structure.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Must contain a ``customer_id`` column.
    n_resamples : int
        Number of resamples to yield.
    random_seed : int, optional
        Random seed.

    Yields
    ------
    pd.DataFrame
        Resampled transaction DataFrame.
    """
    rng = np.random.default_rng(random_seed)
    customers = transactions_df["customer_id"].unique()
    n_cust = len(customers)

    for _ in range(n_resamples):
        cust_idx = rng.integers(0, n_cust, size=n_cust)
        sampled_customers = customers[cust_idx]
        resample = transactions_df[transactions_df["customer_id"].isin(sampled_customers)]
        yield resample


def bootstrap_ci_customer(
    transactions_df: pd.DataFrame,
    statistic_fn: Callable[[pd.DataFrame], float],
    *,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> Dict[str, float]:
    """Bootstrap CI with customer-level resampling (preserves basket structure).

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Transaction data with ``customer_id`` column.
    statistic_fn : callable
        Function taking a resampled DataFrame, returning a float.
    n_resamples : int
        Number of resamples.
    ci_level : float
        Confidence level.
    random_seed : int, optional

    Returns
    -------
    dict
        ``'estimate'``, ``'lower'``, ``'upper'``, ``'std_error'``,
        ``'n_resamples'``.
    """
    point = statistic_fn(transactions_df)
    replicates: List[float] = []

    for resample in bootstrap_transaction_samples(
        transactions_df, n_resamples=n_resamples, random_seed=random_seed
    ):
        try:
            replicates.append(statistic_fn(resample))
        except Exception:
            continue

    if not replicates:
        return {
            "estimate": point,
            "lower": point,
            "upper": point,
            "std_error": 0.0,
            "n_resamples": 0,
        }

    arr = np.array(replicates)
    alpha = 1.0 - ci_level
    lower = float(np.percentile(arr, 100 * alpha / 2))
    upper = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    std_err = float(arr.std(ddof=1))

    return {
        "estimate": point,
        "lower": lower,
        "upper": upper,
        "std_error": std_err,
        "n_resamples": len(replicates),
    }
