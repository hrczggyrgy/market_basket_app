"""Pipeline state management for sharing computed results between analysis modules."""

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
from pydantic import BaseModel

PIPELINE_KEY = "pipeline_store"


class PipelineState(BaseModel):
    """Type-safe pipeline state model."""

    transactions_df: Optional[pd.DataFrame] = None
    product_lookup: Optional[Dict[str, Any]] = None
    basket_matrix: Optional[pd.DataFrame] = None
    frequent_itemsets: Optional[pd.DataFrame] = None
    rules: Optional[pd.DataFrame] = None
    filtered_rules: Optional[pd.DataFrame] = None
    similarity_matrix: Optional[pd.DataFrame] = None
    linkage_matrix: Optional[Any] = None
    cluster_assignments: Optional[pd.DataFrame] = None
    cdt_tree: Optional[Any] = None
    cdt_metadata: Optional[Dict[str, Any]] = None
    sequences: Optional[Any] = None
    switching_matrix: Optional[pd.DataFrame] = None
    substitution_matrix: Optional[pd.DataFrame] = None
    bundling_matrix: Optional[pd.DataFrame] = None
    customer_features: Optional[pd.DataFrame] = None
    rfm_features: Optional[pd.DataFrame] = None
    segment_assignments: Optional[pd.DataFrame] = None
    segment_profiles: Optional[pd.DataFrame] = None
    clv_predictions: Optional[pd.DataFrame] = None
    elasticity_results: Optional[pd.DataFrame] = None
    kvi_scores: Optional[pd.DataFrame] = None
    price_curves: Optional[Any] = None
    promo_flags: Optional[pd.DataFrame] = None
    uplift_model: Optional[Any] = None
    uplift_results: Optional[pd.DataFrame] = None
    demand_transference_matrix: Optional[pd.DataFrame] = None
    assortment_scenarios: Optional[Any] = None
    assortment_evaluation: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True


def _get_state() -> PipelineState:
    """Get or initialize typed pipeline state."""
    if PIPELINE_KEY not in st.session_state:
        st.session_state[PIPELINE_KEY] = PipelineState()
    return st.session_state[PIPELINE_KEY]


def init_pipeline():
    """Initialize the pipeline store in session state."""
    _get_state()


def get_pipeline() -> PipelineState:
    """Get the typed pipeline state."""
    return _get_state()


def set_pipeline(key: str, value: Any):
    """Set a value in the pipeline store with type validation."""
    state = _get_state()
    if hasattr(state, key):
        setattr(state, key, value)
    else:
        raise KeyError(
            f"Invalid pipeline key: {key}. Valid keys: {list(state.model_fields.keys())}"
        )


def get_from_pipeline(key: str, default: Any = None) -> Any:
    """Get a value from the pipeline store."""
    state = _get_state()
    return getattr(state, key, default)


def has_pipeline_data(key: str) -> bool:
    """Check if pipeline has data for a key."""
    state = _get_state()
    val = getattr(state, key, None)
    return val is not None and (not isinstance(val, pd.DataFrame) or not val.empty)


def clear_pipeline(keys: Optional[list] = None):
    """Clear pipeline data. If keys provided, only clear those keys."""
    state = _get_state()
    if keys is None:
        for k in state.model_fields:
            setattr(state, k, None)
    else:
        for k in keys:
            if hasattr(state, k):
                setattr(state, k, None)


def invalidate_downstream(from_stage: str):
    """Invalidate pipeline stages that depend on a given stage.

    Stage dependency order:
    data -> basket -> (rules, cdt, segmentation, elasticity, promo)
    cdt -> demand_transference -> assortment
    segmentation -> promo_uplift
    """
    stage_order = [
        "transactions_df",
        "product_lookup",
        "basket_matrix",
        "frequent_itemsets",
        "rules",
        "filtered_rules",
        "similarity_matrix",
        "linkage_matrix",
        "cluster_assignments",
        "cdt_tree",
        "cdt_metadata",
        "sequences",
        "switching_matrix",
        "substitution_matrix",
        "bundling_matrix",
        "customer_features",
        "rfm_features",
        "segment_assignments",
        "segment_profiles",
        "clv_predictions",
        "elasticity_results",
        "kvi_scores",
        "price_curves",
        "promo_flags",
        "uplift_model",
        "uplift_results",
        "demand_transference_matrix",
        "assortment_scenarios",
        "assortment_evaluation",
    ]

    try:
        idx = stage_order.index(from_stage)
        # Clear all stages after this one
        state = _get_state()
        for stage in stage_order[idx + 1 :]:
            if hasattr(state, stage):
                setattr(state, stage, None)
    except ValueError:
        pass


def get_pipeline_summary() -> Dict[str, str]:
    """Get a summary of what's in the pipeline for debugging."""
    state = _get_state()
    summary = {}
    for k, v in state:
        if v is None:
            summary[k] = "empty"
        elif isinstance(v, pd.DataFrame):
            summary[k] = f"DataFrame({len(v)} rows x {len(v.columns)} cols)"
        elif isinstance(v, dict):
            summary[k] = f"dict({len(v)} keys)"
        elif isinstance(v, (list, tuple)):
            summary[k] = f"{type(v).__name__}({len(v)} items)"
        else:
            summary[k] = f"{type(v).__name__}"
    return summary
