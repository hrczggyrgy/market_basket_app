"""Pricing & Elasticity tab."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.category import enrich_with_categories
from src.analytics.pricing import (
    classify_elasticity_confidence,
    estimate_loglog_elasticity,
    estimate_hierarchical_elasticity,
    compute_kvi_score,
    diagnose_price_curves_1d,
)
from src.ui.plots import PALETTE, empty_state, render_bar_with_ci, show
from src.ui.registry import ModeSpec

CONFIDENCE_COLORS = {"high": PALETTE[2], "medium": PALETTE[3], "low": PALETTE[6]}


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


def _render_elasticity_confidence(elast: pd.DataFrame) -> None:
    st.subheader(":material/rule: Elasticity Confidence")
    conf = classify_elasticity_confidence(elast)
    if conf.empty:
        show(empty_state("No elasticity estimates to classify"))
        return

    tiers = st.multiselect(
        "Show confidence tiers",
        ["high", "medium", "low"],
        default=["high", "medium"],
        format_func=lambda t: {"high": "High (significant, tight CI)", "medium": "Medium", "low": "Low (wide CI / not significant)"}[t],
        key="elast_conf_tiers",
    )
    filtered = conf[conf["confidence"].isin(tiers)] if tiers else conf

    # Scatter: relative CI width vs point estimate, colored by tier
    work = filtered.copy()
    work["relative_ci_width"] = work["ci_width"] / work["elasticity"].abs().clip(lower=1e-6)
    fig = go.Figure(
        data=go.Scatter(
            x=work["elasticity"],
            y=work["relative_ci_width"],
            mode="markers",
            text=work["stockcode"].astype(str),
            customdata=work[["confidence", "direction", "ci_lower", "ci_upper"]].to_numpy(),
            hovertemplate="%{text}<br>elasticity %{x:.2f}<br>CI width %{y:.2f}x estimate<br>confidence %{customdata[0]}<br>direction %{customdata[1]}<br>CI [%{customdata[2]:.2f}, %{customdata[3]:.2f}]<extra></extra>",
            marker={
                "size": work["n_obs"].rank(pct=True) * 16 + 6,
                "color": work["confidence"].map(CONFIDENCE_COLORS),
                "line": {"width": 1, "color": "#333333"},
            },
        )
    )
    fig.add_vline(x=-1, line_dash="dot", line_color="#888888", annotation_text="elastic: |e|>1")
    fig.add_vline(x=1, line_dash="dot", line_color="#888888", annotation_text="elastic: |e|>1")
    fig.update_layout(
        xaxis={"title": "Own-price Elasticity (log-log OLS)"},
        yaxis={"title": "CI width as multiple of |elasticity|"},
        height=380,
    )
    show(fig)
    st.caption(
        "x = point estimate, y = CI width relative to the estimate (lower = tighter/more precise). "
        "Green = high confidence (p<0.05, tight CI), amber = medium, red = low (wide CI or not significant). "
        "SKUs with a wide CI should not be re-priced on the estimate alone."
    )

    st.dataframe(
        filtered.sort_values("confidence", key=lambda c: c.map({"high": 0, "medium": 1, "low": 2})),
        use_container_width=True,
        hide_index=True,
    )


def _render_kvi_elasticity_quadrant(kvi: pd.DataFrame) -> go.Figure:
    """KVI score (importance) vs |elasticity| (price sensitivity) quadrant."""
    fig = go.Figure()
    if kvi.empty:
        return empty_state("No KVI data for quadrant matrix")

    from src.analytics.pricing import compute_kvi_elasticity_quadrant

    quad = compute_kvi_elasticity_quadrant(kvi)
    if quad.empty:
        return empty_state("No KVI x elasticity data")

    palette = {
        "advocate": PALETTE[0],
        "protect": PALETTE[2],
        "promote": PALETTE[3],
        "defer": PALETTE[4],
    }
    fig.add_trace(
        go.Scatter(
            x=quad["kvi_score"],
            y=quad["abs_elasticity"],
            mode="markers",
            text=quad["stockcode"].astype(str),
            customdata=quad[["quadrant", "category"]].to_numpy(),
            hovertemplate="%{text}<br>KVI %{x:.2f} | |e| %{y:.2f}<br>%{customdata[0]} (cat: %{customdata[1]})<extra></extra>",
            marker={"size": quad["total_revenue"].rank(pct=True) * 22 + 4, "color": quad["quadrant"].map(palette)},
        )
    )
    kvi_med = float(quad["kvi_score"].median())
    fig.add_vline(x=kvi_med, line_dash="dash", line_color="#888888", annotation_text="median KVI")
    fig.add_hline(y=1.0, line_dash="dash", line_color="#888888", annotation_text="|e| = 1 (elastic)")

    for qname, qx, qy in (
        ("Advocate: protect price", 0.76, 0.98),
        ("Promote: price lever", 0.1, 0.98),
        ("Protect: keep & carry margin", 0.76, 0.05),
        ("Defer: review last", 0.1, 0.05),
    ):
        fig.add_annotation(
            text=qname, x=qx, y=qy, xref="paper", yref="paper", showarrow=False,
            font={"size": 11, "color": "#888888"},
        )
    fig.update_layout(
        xaxis={"title": "KVI Score (0-1, importance)", "range": [-0.05, 1.05]},
        yaxis={"title": "Abs Own-price Elasticity"},
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

                _render_elasticity_confidence(elast)
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
            show(_render_kvi_elasticity_quadrant(kvi))
            st.caption(
                "KVI importance (x) vs |own-price elasticity| (y). Advocates are high-KVI, price-sensitive "
                "traffic drivers: defend their price. Promotes are low-KVI, elastic SKUs -- use as price "
                "levers. Protects carry margin safely; Defer reviews them last."
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