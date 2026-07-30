"""Trace-cache utilities for PyMC Bayesian models."""

import hashlib
import json
from pathlib import Path

import streamlit as st

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _data_hash(transactions_df) -> str:
    """Deterministic hash of transaction data for cache key."""
    key_df = transactions_df[["date", "transaction_id", "stockcode", "price", "quantity"]].copy()
    key_df = key_df.sort_values(["transaction_id", "stockcode"])
    raw = key_df.to_csv(index=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _model_config_hash(model_config: dict) -> str:
    """Hash of the model configuration dict."""
    canonical = json.dumps(model_config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def trace_cache_key(transactions_df, model_config: dict | None = None) -> str:
    """Return a deterministic cache key for a PyMC trace.

    Combines the data fingerprint with an optional model-config hash so that
    changing either invalidates the cache automatically.
    """
    parts = [_data_hash(transactions_df)]
    if model_config:
        parts.append(_model_config_hash(model_config))
    return "_".join(parts)


@st.cache_resource
def get_trace_cache() -> dict:
    """Return the in-memory trace cache (persisted across reruns via st.cache_resource)."""
    return {}
