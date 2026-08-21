"""Consolidated DataFrame hashing utilities.

Provides deterministic, fast DataFrame hashing for cache keys.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import pandas as pd


def df_hash(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
    sample_size: int = 1000,
) -> str:
    """Fast hash of DataFrame content for cache keys.

    Uses a subset of columns for hashing to avoid hashing huge DataFrames.
    Hashes shape + column dtypes + sample of data.

    Args:
        df: DataFrame to hash
        cols: Specific columns to hash (None = all columns)
        sample_size: Max rows to sample for hashing (default 1000)

    Returns:
        16-character hex string
    """
    if cols is None:
        cols = df.columns.tolist()
    # Use a subset of columns for hashing to avoid hashing huge DataFrames
    # Hash shape + first/last rows + column dtypes
    sample = df[cols].head(sample_size) if len(df) > sample_size else df[cols]
    dtypes_str = str(sample.dtypes)
    content = f"{df.shape}{dtypes_str}{sample.to_numpy().tobytes()}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def df_hash_xxh64(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
    sample_size: int = 1000,
) -> str:
    """Fast xxhash64-based DataFrame hash (32-char hex).

    Requires xxhash package. Falls back to md5 if unavailable.

    Args:
        df: DataFrame to hash
        cols: Specific columns to hash (None = all columns)
        sample_size: Max rows to sample for hashing (default 1000)

    Returns:
        32-character hex string
    """
    try:
        import xxhash
    except ImportError:
        # Fallback to md5-based hash, padded to 32 chars
        return df_hash(df, cols, sample_size) * 2

    if cols is None:
        cols = df.columns.tolist()
    sample = df[cols].head(sample_size) if len(df) > sample_size else df[cols]
    dtypes_str = str(sample.dtypes)
    content = f"{df.shape}{dtypes_str}{sample.to_numpy().tobytes()}"
    h = xxhash.xxh64(content.encode("utf-8"))
    return h.hexdigest()
