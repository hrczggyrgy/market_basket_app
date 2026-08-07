"""Pricing & Elasticity tab."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.category import enrich_with_categories
from src.analytics.pricing import (
    estimate_loglog_elasticity,
    estimate_hierarchical_elasticity,
    compute_kvi_score,
    diagnose_price_curves_1d,
)
from src.ui.plots import PALETTE, empty_state, render_bar_with_ci, show
from src.ui.registry import ModeSpec


def _render_kvi_quadrant(kvi: pd.DataFrame) -> go.Figure:
    """KVI score vs revenue-share quadrant (Invest/Protect/Grow/Reconsider)."""
    fig = go.Figure()
    if kvi.empty:
        return empty_state("No KVI data for quadrant matrix")

    total = kvi["total_revenue"].sum()
    work = kvi.copy()
    work["revenue_share"] = work["total_revenue"] / total if total > 0 else 0.0

    x_med = float(work["revenue_share"].median())
    y_med = float(work["kvi_score"].median())

    # Quadrant thresholds via KPIs' own midpoints so the split is always visible
    x_thr = x_med if x_med > 0 else float(work["revenue_share"].mean())
    y_thr = y_med if y_med > 0 else 0.5

    colors = []
    for _, row in work.iterrows():
        if row["kvi_score"] >= y_thr and row["revenue_share"] >= x_thr:
            colors.append("Invest / Keep")
        elif row["kvi_score"] >= y_thr:
            colors.append("Grow Potential")
        elif row["revenue_share"] >= x_thr:
            colors.append("Cash Cow / Protect")
        else:
            colors.append("Reconsider")
    work["quadrant"] = colors

    palette = {
        "Invest / Keep": PALETTE[0],
        "Grow Potential": PALETTE[2],
        "Cash Cow / Protect": PALETTE[3],
        "Reconsider": PALETTE[4],
    }
    fig.add_trace(
        go.Scatter(
            x=work["revenue_share"],
            y=work["kvi_score"],
            mode="markers",
            text=work["stockcode"].astype(str),
            hoverinfo="x+y+text",
            marker={"size": work["total_revenue"].rank(pct=True) * 22 + 4, "color": work["quadrant"].map(palette)},
        )
    )
    for x, label in ((x_thr, "median revenue-share"),):
        fig.add_vline(x=x, line_dash="dash", line_color="#888888", annotation_text=label)
    for y, label in ((y_thr, "median KVI score"),):
        fig.add_hline(y=y, line_dash="dash", line_color="#888888", annotation_text=label)

    for qname, qx, qy in (
        ("Grow Potential (high KVI, low share)", 0.1, 0.99),
        ("Invest / Keep (high KVI, high share)", 0.76, 0.02),
        ("Reconsider (low KVI, low share)", 0.1, 0.02),
        ("Cash Cow / Protect (low KVI, high share)", 0.52, 0.99),
    ):
        fig.add_annotation(
            text=qname, x=qx, y=qy, xref="paper", yref="paper", showarrow=False,
            font={"size": 11, "color": "#888888"},
        )

    fig.update_layout(
        xaxis={"title": "Revenue Share of Category"},
        yaxis={"title": "KVI Score (0-1)"},
        hovermode="closest",
    )
    return fig


def render(df: pd.DataFrame) -> None:
    st.subheader(":material/price_check: Pricing & Elasticity")

    df, category_inferred = enrich_with_categories(df)
    if category_inferred:
        st.info("A `category` column was not supplied; categories were inferred from product descriptions (TF-IDF + KMeans).")

    tab1, tab2, tab3 = st.tabs(["Elasticity", "KVI Scores", "Price Curves"])

    with tab1:
        elast = estimate_loglog_elasticity(df, min_periods=5)
        if not elast.empty:
            if {"ci_lower", "ci_upper"}.issubset(elast.columns):
                elast_plot = elast.copy()
                elast_plot["label"] = elast_plot["stockcode"].astype(str)
                show(
                    render_bar_with_ci(
                        df=elast_plot,
                        x_col="label",
                        y_col="elasticity",
                        ci_lower_col="ci_lower",
                        ci_upper_col="ci_upper",
                        x_title="SKU",
                        y_title="Own-price Elasticity",
                        title="Own-price Elasticity with 95% CI",
                        height=420,
                    )
                )
                st.caption("Elasticity < -1 means elastic demand: a 1% price cut raises quantity by more than 1%.")
            else:
                st.dataframe(elast.sort_values("elasticity"), use_container_width=True, hide_index=True)

            hier = estimate_hierarchical_elasticity(df, min_periods=5)
            if not hier.empty:
                st.caption("Hierarchical (Shrunk) Elasticity")
                st.dataframe(hier[["stockcode", "category", "elasticity_ols", "elasticity_shrunk", "shrink_weight"]],
                             use_container_width=True, hide_index=True)
        else:
            n_skus = df["stockcode"].nunique()
            st.warning(
                f"Insufficient data for elasticity estimation across {n_skus:,} SKUs. "
                "SKUs with constant or near-constant weekly prices (price CV < 5%), too few distinct "
                "price points (< 3), or too few weekly observations (< 5) are excluded rather than "
                "reported as inelastic (0.0)."
            )

    with tab2:
        kvi = compute_kvi_score(df, elasticity_df=elast, method="heuristic")
        if not kvi.empty:
            show(_render_kvi_quadrant(kvi))
            st.caption(
                "Quadrant split at median KVI score (y) and median revenue share (x). "
                "KVI score combines basket penetration, revenue, halo, elasticity and customer reach."
            )
            st.dataframe(kvi.sort_values("kvi_score", ascending=False).head(20),
                         use_container_width=True, hide_index=True)
            if "abs_elasticity" in kvi.columns and kvi["abs_elasticity"].eq(0).all():
                st.caption("abs_elasticity is 0 because elasticity is not estimable (no per-SKU price variation) — not a claim of perfect inelasticity.")

    with tab3:
        curves = diagnose_price_curves_1d(df, n_tiers=3)
        if not curves.empty:
            st.dataframe(curves[["stockcode", "category", "median_price", "pack_size_numeric",
                                 "price_per_unit", "tier_label", "has_violation"]],
                         use_container_width=True, hide_index=True)
            violations = curves[curves["has_violation"]]
            if not violations.empty:
                st.warning(f"Price curve violations: {len(violations)}")
                st.dataframe(violations, use_container_width=True, hide_index=True)


MODE_SPEC: ModeSpec = ModeSpec(
    key="pricing",
    label="Pricing",
    icon=":material/price_check:",
    handler=render,
)