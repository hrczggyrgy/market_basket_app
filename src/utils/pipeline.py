"""Pipeline state management for sharing computed results between analysis modules."""

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st


PIPELINE_KEY = "pipeline_store"


def init_pipeline():
    """Initialize the pipeline store in session state."""
    if PIPELINE_KEY not in st.session_state:
        st.session_state[PIPELINE_KEY] = {
            # Data layer
            "transactions_df": None,
            "product_lookup": None,
            "basket_matrix": None,
            
            # Association rules layer
            "frequent_itemsets": None,
            "rules": None,
            "filtered_rules": None,
            
            # CDT layer
            "similarity_matrix": None,
            "linkage_matrix": None,
            "cluster_assignments": None,
            "cdt_tree": None,
            "cdt_metadata": None,
            "sequences": None,
            "switching_matrix": None,
            "substitution_matrix": None,
            "bundling_matrix": None,
            
            # Segmentation layer
            "customer_features": None,
            "rfm_features": None,
            "segment_assignments": None,
            "segment_profiles": None,
            "clv_predictions": None,
            
            # Elasticity/Pricing layer
            "elasticity_results": None,
            "kvi_scores": None,
            "price_curves": None,
            
            # Promo Uplift layer
            "promo_flags": None,
            "uplift_model": None,
            "uplift_results": None,
            
            # Demand Transference layer
            "demand_transference_matrix": None,
            
            # Assortment layer
            "assortment_scenarios": None,
            "assortment_evaluation": None,
        }


def get_pipeline() -> Dict[str, Any]:
    """Get the pipeline store, initializing if needed."""
    init_pipeline()
    return st.session_state[PIPELINE_KEY]


def set_pipeline(key: str, value: Any):
    """Set a value in the pipeline store."""
    init_pipeline()
    st.session_state[PIPELINE_KEY][key] = value


def get_from_pipeline(key: str, default: Any = None) -> Any:
    """Get a value from the pipeline store."""
    init_pipeline()
    return st.session_state[PIPELINE_KEY].get(key, default)


def has_pipeline_data(key: str) -> bool:
    """Check if pipeline has data for a key."""
    init_pipeline()
    val = st.session_state[PIPELINE_KEY].get(key)
    return val is not None and (not isinstance(val, pd.DataFrame) or not val.empty)


def clear_pipeline(keys: Optional[list] = None):
    """Clear pipeline data. If keys provided, only clear those keys."""
    init_pipeline()
    if keys is None:
        for k in st.session_state[PIPELINE_KEY]:
            st.session_state[PIPELINE_KEY][k] = None
    else:
        for k in keys:
            if k in st.session_state[PIPELINE_KEY]:
                st.session_state[PIPELINE_KEY][k] = None


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
        for stage in stage_order[idx + 1:]:
            st.session_state[PIPELINE_KEY][stage] = None
    except ValueError:
        pass


def get_pipeline_summary() -> Dict[str, str]:
    """Get a summary of what's in the pipeline for debugging."""
    init_pipeline()
    summary = {}
    for k, v in st.session_state[PIPELINE_KEY].items():
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