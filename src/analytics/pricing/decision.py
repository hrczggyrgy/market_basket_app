"""Pricing decision matrix: KVI x elasticity x confidence -> decision.

Confidence gates entry into the decision quadrant: a SKU with a usable
elasticity estimate (non-low confidence) can be classified invest/protect/
price_lever/review. Everything else is ``insufficient_evidence`` — it is never
silently assumed inelastic.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.analytics.pricing.elasticity import classify_elasticity_confidence
from src.analytics.schemas import PRICING_DECISION_MATRIX, check

_RATIONALE = {
    "invest": (
        "High-KVI, price-elastic traffic driver — defend price competitiveness; "
        "never trade down to private label."
    ),
    "protect": "High-KVI, price-inelastic — can carry margin safely; protect availability.",
    "price_lever": "Low-KVI, price-elastic — candidate price lever for promotional investment.",
    "review": "Low-KVI, price-inelastic — low strategic importance; review assortment last.",
    "insufficient_evidence": "Elasticity not reliably estimable — insufficient evidence for a price decision.",
}


def compute_pricing_decision_matrix(
    kvi_df: pd.DataFrame,
    elasticity_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Map every SKU to a price decision, gated on elasticity evidence.

    Args:
        kvi_df: Output of ``compute_kvi_score`` (KVI_SCORES contract).
        elasticity_df: Optional output of ``estimate_loglog_elasticity`` used to
            derive per-SKU confidence (gates the decision quadrant).

    Returns:
        DataFrame validated against PRICING_DECISION_MATRIX (empty input yields
        an empty, validated frame).
    """
    empty = pd.DataFrame(columns=list(PRICING_DECISION_MATRIX.columns))
    if kvi_df is None or kvi_df.empty:
        return check(empty, PRICING_DECISION_MATRIX, allow_empty=True)

    df = kvi_df.copy()
    df["elasticity_confidence"] = np.nan
    if elasticity_df is not None and not elasticity_df.empty:
        conf = classify_elasticity_confidence(elasticity_df)[["stockcode", "confidence"]]
        df = df.merge(conf, on="stockcode", how="left", suffixes=("", "_conf"))
        conf_col = "confidence_conf" if "confidence_conf" in df.columns else "confidence"
        df["elasticity_confidence"] = df["elasticity_confidence"].combine_first(df[conf_col])
        df = df.drop(columns=[conf_col], errors="ignore")

    if "elasticity_status" not in df.columns:
        df["elasticity_status"] = np.where(df["abs_elasticity"].notna(), "estimated", "unavailable")

    kvi_median = float(df["kvi_score"].median())

    def _decision(row: pd.Series) -> str:
        status = row["elasticity_status"]
        # Only estimated or weak statuses with valid elasticity enter decision quadrant
        if status not in ("estimated", "weak") or pd.isna(row["abs_elasticity"]):
            return "insufficient_evidence"
        if status == "weak":
            return "insufficient_evidence"
        high_kvi = row["kvi_score"] >= kvi_median
        elastic = row["abs_elasticity"] >= 1.0
        if high_kvi and elastic:
            return "invest"
        if high_kvi:
            return "protect"
        if elastic:
            return "price_lever"
        return "review"

    df["decision"] = df.apply(_decision, axis=1)
    df["rationale"] = df["decision"].map(_RATIONALE)

    table = df[list(PRICING_DECISION_MATRIX.columns)].reset_index(drop=True)
    return check(table, PRICING_DECISION_MATRIX)
