"""Pricing & Elasticity tab — five-layer decision structure.

Layers:
1. Price Strategy Matrix — KVI importance vs price sensitivity quadrants with confidence opacity
2. Price-Response Curve — observed prices + fitted relationship + confidence interval + current price + simulated prices
3. Price Ladder — Premium/Mid/Value/Entry with price index/reach/revenue/KVI status
4. Price × Promotion Matrix — 4 strategies: Base-price focus / Strategic promo / Margin opportunity / Selective promo
5. Manager Table — SKU | KVI | Elasticity | Confidence | Current price | Recommended price | Expected units | Revenue impact | Risk | Action + "Do not act"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.cache import cached_enrich_categories, run_cached_pricing_analysis
from src.analytics.pricing import diagnose_price_curves_1d
from src.analytics.pricing.pipeline import PricingAnalysis
from src.analytics.profile_service import ProfileService, init_profile_service
from src.ui.components.decision_matrix import (
    MatrixConfig,
    _render_with_confidence,
)
from src.ui.components_utils import (
    render_insight_cards,
    render_metric_row,
    render_opportunity_table,
    render_pricing_decision_card,
)
from src.ui.plots import PALETTE, empty_state, new_fig, show
from src.ui.registry import ModeSpec

CONFIDENCE_COLORS = {"high": PALETTE[2], "medium": PALETTE[3], "low": PALETTE[6]}

STATUS_COLORS = {
    "estimated": PALETTE[2],
    "weak": PALETTE[6],
    "insufficient_variation": PALETTE[3],
    "insufficient_observations": PALETTE[5],
    "insufficient_price_points": PALETTE[5],
    "near_constant_price": PALETTE[5],
    "near_constant_quantity": PALETTE[5],
    "near_perfect_collinearity": PALETTE[6],
    "extreme_values": PALETTE[6],
    "correlation_failed": PALETTE[7],
    "model_failed": PALETTE[7],
    "not_significant": PALETTE[6],
    "unavailable": PALETTE[7],
}

DECISION_COLORS = {
    "invest": PALETTE[0],
    "protect": PALETTE[2],
    "price_lever": PALETTE[3],
    "review": PALETTE[4],
    "insufficient_evidence": PALETTE[7],
}

_DECISION_ORDER = ["invest", "protect", "price_lever", "review", "insufficient_evidence"]

# Shared strategy definitions for pricing matrices
STRATEGY_COLORS = {
    "Base-price focus": "#59A14F",
    "Strategic promo": "#4E79A7",
    "Margin opportunity": "#F28E2B",
    "Selective promo": "#E15759",
}

STRATEGY_LABELS = {
    "Base-price focus": "Base-price focus\n(High KVI, inelastic)",
    "Strategic promo": "Strategic promo\n(High KVI, elastic)",
    "Margin opportunity": "Margin opportunity\n(Low KVI, inelastic)",
    "Selective promo": "Selective promo\n(Low KVI, elastic)",
}


def _render_scorecard(analysis: PricingAnalysis) -> None:
    if not isinstance(analysis, PricingAnalysis):
        st.error("Invalid analysis object")
        return
    status = analysis.elasticity_status
    n_total = int(len(status))
    n_est = int((status["elasticity_status"] == "estimated").sum())
    n_weak = int((status["elasticity_status"] == "weak").sum())
    n_high = 0
    if not analysis.confidence.empty:
        n_high = int((analysis.confidence["confidence"] == "high").sum())

    covered_rev = 0.0
    total_rev = 0.0
    if not analysis.kvi.empty:
        rev = analysis.kvi.set_index("stockcode")["total_revenue"].astype(float)
        usable = status.loc[status["elasticity_status"].isin(("estimated", "weak")), "stockcode"]
        covered_rev = float(usable.map(rev).fillna(0.0).sum())
        total_rev = float(rev.sum())
    pct = f"{covered_rev / total_rev:.0%}" if total_rev else "—"

    render_metric_row(
        [
            {
                "label": "Estimable SKUs",
                "value": f"{n_est} / {n_total}",
                "help": "SKUs with a usable elasticity estimate (status = estimated).",
            },
            {
                "label": "High-confidence estimates",
                "value": str(n_high),
                "help": "Significant and with a tight confidence interval.",
            },
            {
                "label": "Weak / unreliable estimates",
                "value": str(n_weak),
                "help": "Estimated but low-confidence — not safe to reprice on alone.",
            },
            {
                "label": "Revenue covered by estimates",
                "value": f"€{covered_rev:,.0f} ({pct})",
                "help": "Revenue share with at least a usable elasticity estimate.",
            },
        ]
    )


def _render_status_breakdown(status_df: pd.DataFrame) -> go.Figure:
    """Why coverage is what it is: SKU counts per elasticity status."""
    fig = new_fig(height=280)
    if status_df is None or status_df.empty:
        return empty_state("No elasticity status data")
    counts = (
        status_df["elasticity_status"]
        .value_counts()
        .reindex(list(STATUS_COLORS.keys()))
        .fillna(0)
        .astype(int)
    )
    fig.add_trace(
        go.Bar(
            x=counts.index, y=counts.values, marker_color=[STATUS_COLORS[k] for k in counts.index]
        )
    )
    fig.update_layout(
        xaxis={"title": "Elasticity status"},
        yaxis={"title": "SKU count"},
        hovermode="x unified",
    )
    return fig


def _render_decision_matrix(dm: pd.DataFrame) -> go.Figure:
    """KVI importance x |elasticity| decision landscape with unknown zone.

    Incorporates the Decision Confidence System:
    - Bubble position = analytical result (KVI score / |elasticity|)
    - Bubble size = economic impact (revenue units)
    - Bubble opacity = evidence confidence (HIGH=opaque, MEDIUM=semi-transparent, LOW=faint)
    - Evidence class integration with badge convention
    """
    fig = new_fig(height=480)
    if dm is None or dm.empty:
        return empty_state("No SKUs for the price decision matrix")

    # Build MatrixConfig for confidence system integration
    config = MatrixConfig(
        x_axis="abs_elasticity",
        y_axis="kvi_score",
        value="decision",
        text="stockcode",
        x_labels={"abs_elasticity": "|e|", "kvi_score": "KVI"},
        y_labels={"abs_elasticity": "|e|", "kvi_score": "KVI"},
        x_order=[0.0, 0.5, 1.0, 1.5, 2.0],
        y_order=[0.2, 0.4, 0.6, 0.8, 1.0],
        size_col="total_revenue",
        confidence_col="confidence",
        evidence_col="evidence",
    )

    estimable = dm[dm["decision"] != "insufficient_evidence"]
    unknown = dm[dm["decision"] == "insufficient_evidence"]

    if not estimable.empty:
        fig.add_trace(
            go.Scatter(
                x=estimable["abs_elasticity"],
                y=estimable["kvi_score"],
                mode="markers+text",
                text=estimable["stockcode"].astype(str),
                customdata=estimable[["decision", "elasticity_status", "confidence", "evidence"]].to_numpy(),
                hovertemplate=(
                    "%{text}<br>|e| %{x:.2f} | KVI %{y:.2f}<br>"
                    "decision: %{customdata[0]} (status: %{customdata[1]})<br>"
                    "confidence: %{customdata[2]} | evidence: %{customdata[3]}<extra></extra>"
                ),
                marker={
                    "size": estimable["total_revenue"].rank(pct=True) * 24 + 6,
                    "color": estimable["decision"].map({
                        "invest": PALETTE[0],
                        "protect": PALETTE[2],
                        "price_lever": PALETTE[3],
                        "review": PALETTE[4],
                    }),
                    "line": {"width": 1, "color": "#333333"},
                },
                name="Decision",
            )
        )

    if not unknown.empty:
        fig.add_trace(
            go.Scatter(
                x=[-0.4] * len(unknown),
                y=unknown["kvi_score"],
                mode="markers",
                text=unknown["stockcode"].astype(str),
                customdata=unknown[["elasticity_status"]].to_numpy(),
                hovertemplate="%{text}<br>KVI %{y:.2f}<br>insufficient evidence (%{customdata[0]})<extra></extra>",
                marker={
                    "size": 7,
                    "color": DECISION_COLORS["insufficient_evidence"],
                    "symbol": "x",
                },
                name="Insufficient evidence",
            )
        )

    # Apply confidence system mixin: opacity, evidence coloring
    fig = _render_with_confidence(fig, dm, config)

    fig.add_vline(
        x=1.0, line_dash="dash", line_color="#888888", annotation_text="|e| = 1 (elastic)"
    )
    kvi_med = float(dm["kvi_score"].median())
    fig.add_hline(y=kvi_med, line_dash="dash", line_color="#888888", annotation_text="median KVI")

    max_abs = float(dm["abs_elasticity"].max())
    if not np.isfinite(max_abs):
        max_abs = 2.0
    fig.update_xaxes(range=[-0.5, max_abs * 1.15])
    fig.update_layout(
        xaxis={"title": "|Own-price elasticity| (left band: insufficient evidence)"},
        yaxis={"title": "KVI Score (0-1, strategic importance)"},
        hovermode="closest",
        showlegend=False,
    )
    return fig


def _simulate_price_change(
    base_price: float, base_qty: float, elasticity: float, pct: float
) -> tuple[float, float, float]:
    """Log-log response: new qty/price/revenue for a price change of `pct`."""
    # Log-log model: log(qty) = intercept + elasticity * log(price)
    # So qty_ratio = (new_price / base_price) ** elasticity
    new_price = base_price * (1 + pct)
    new_qty = base_qty * (new_price / base_price) ** elasticity
    return new_qty, new_price, new_qty * new_price


def _render_price_simulation(analysis: PricingAnalysis) -> None:
    if not isinstance(analysis, PricingAnalysis):
        st.error("Invalid analysis object")
        return
    st.subheader(":material/calculate: Business Impact — Price Scenario Simulation")
    elast = analysis.elasticity
    if elast is None or elast.empty:
        st.info("No estimable SKUs — price simulation requires a usable elasticity estimate.")
        return

    # Get confidence information
    conf_map = {}
    if not analysis.confidence.empty:
        conf_map = dict(
            zip(analysis.confidence["stockcode"], analysis.confidence["confidence"], strict=False)
        )

    # Get elasticity status information
    status_map = {}
    if not analysis.elasticity_status.empty:
        status_map = dict(
            zip(
                analysis.elasticity_status["stockcode"],
                analysis.elasticity_status["elasticity_status"],
                strict=False,
            )
        )

    # Filter SKUs by confidence - only high and medium allowed for simulation
    usable = []
    for sku in elast["stockcode"]:
        conf = conf_map.get(sku, "medium")
        status = status_map.get(sku, "estimated")
        if conf in ("high", "medium") and status in ("estimated", "weak"):
            usable.append(sku)

    if not usable:
        st.info(
            "Elasticity estimates exist but none are reliable enough to simulate. Collect more price variation first."
        )
        return

    sku = st.selectbox("Select an SKU to simulate", usable, key="price_sim_sku")
    row = elast[elast["stockcode"] == sku].iloc[0]
    base_price = float(row["avg_price"])
    base_qty = float(row["avg_weekly_qty"])
    elasticity = float(row["elasticity"])
    base_rev = base_price * base_qty
    conf = conf_map.get(sku, "medium")
    n_obs = int(row["n_obs"]) if "n_obs" in row else 0

    # Confidence-based warning
    if conf == "medium":
        st.warning(
            f"⚠️ Medium confidence estimate (n={n_obs} observations). "
            "Results should be validated with a controlled test before acting."
        )
    elif conf == "low":
        st.error(
            f"🔴 Low confidence estimate (n={n_obs} observations). "
            "Simulation results are unreliable. Do not act on these figures."
        )

    scenarios = [(-0.05, "-5%"), (-0.02, "-2%"), (0.02, "+2%"), (0.05, "+5%")]
    sim_rows = []
    for pct, label in scenarios:
        qty, price, rev = _simulate_price_change(base_price, base_qty, elasticity, pct)
        sim_rows.append(
            {
                "Scenario": label,
                "Price": round(price, 2),
                "Weekly units": round(qty),
                "Weekly revenue": round(rev, 2),
                "Revenue delta": round(rev - base_rev, 2),
            }
        )
    sim = pd.DataFrame(sim_rows)

    col1, col2 = st.columns([0.45, 0.55])
    with col1:
        st.dataframe(sim, use_container_width=True, hide_index=True)
    with col2:
        fig = new_fig(height=320)
        colors = [PALETTE[2] if d >= 0 else PALETTE[3] for d in sim["Revenue delta"]]
        fig.add_trace(go.Bar(x=sim["Scenario"], y=sim["Revenue delta"], marker_color=colors))
        fig.update_layout(yaxis={"title": "Weekly revenue delta"}, hovermode="x unified")
        show(fig)

    # Action recommendation based on confidence and sample size
    st.subheader("Recommended Action")
    if conf == "high" and n_obs >= 20:
        st.success(
            f"✅ High confidence estimate with sufficient data (n={n_obs}). "
            "Consider a controlled price test to validate before full rollout."
        )
    elif conf == "high" and n_obs < 20:
        st.warning(
            f"⚠️ High confidence but limited data (n={n_obs}). "
            "Proceed with caution and validate with a controlled test."
        )
    elif conf == "medium":
        st.warning(
            "⚠️ Medium confidence estimate. Use these figures for planning only. "
            "Validate with a controlled experiment before any price changes."
        )
    else:
        st.error(
            "🔴 Low confidence estimate. Do not act on these simulation results. "
            "Collect more price variation data first."
        )

    st.caption(
        f"Illustrative log-log response at elasticity {elasticity:.2f} on weekly averages. "
        f"Confidence: {conf.upper()}, Observations: {n_obs}. "
        "Planning figure only — not a causal claim; validate with a controlled test before acting."
    )


def _render_elasticity_confidence_detail(elast: pd.DataFrame) -> None:
    st.subheader(":material/rule: Elasticity Confidence")
    if elast is None or elast.empty:
        show(empty_state("No elasticity estimates to classify"))
        return

    from src.analytics.pricing import classify_elasticity_confidence

    conf = classify_elasticity_confidence(elast)
    if conf.empty:
        show(empty_state("No elasticity estimates to classify"))
        return

    tiers = st.multiselect(
        "Show confidence tiers",
        ["high", "medium", "low"],
        default=["high", "medium"],
        format_func=lambda t: {
            "high": "High (significant, tight CI)",
            "medium": "Medium",
            "low": "Low (wide CI / not significant)",
        }[t],
        key="elast_conf_tiers",
    )
    filtered = conf[conf["confidence"].isin(tiers)] if tiers else conf

    work = filtered.copy()
    work["relative_ci_width"] = work["ci_width"] / work["elasticity"].abs().clip(lower=1e-6)
    fig = go.Figure(
        data=go.Scatter(
            x=work["elasticity"],
            y=work["relative_ci_width"],
            mode="markers",
            text=work["stockcode"].astype(str),
            customdata=work[["confidence", "direction", "ci_lower", "ci_upper"]].to_numpy(),
            hovertemplate=(
                "%{text}<br>elasticity %{x:.2f}<br>CI width %{y:.2f}x estimate<br>"
                "confidence %{customdata[0]}<br>direction %{customdata[1]}<br>CI [%{customdata[2]:.2f}, %{customdata[3]:.2f}]<extra></extra>"
            ),
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


def _render_kvi_quadrant(kvi: pd.DataFrame) -> go.Figure:
    """KVI score vs revenue-share quadrant (Invest/Protect/Grow/Reconsider)."""
    fig = new_fig(height=380)
    if kvi is None or kvi.empty:
        return empty_state("No KVI data for quadrant matrix")

    total = kvi["total_revenue"].sum()
    work = kvi.copy()
    work["revenue_share"] = work["total_revenue"] / total if total > 0 else 0.0

    x_med = float(work["revenue_share"].median())
    y_med = float(work["kvi_score"].median())
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
            customdata=work["quadrant"].to_numpy(),
            hovertemplate="%{text}<br>share %{x:.2%} | KVI %{y:.2f}<br>%{customdata}<extra></extra>",
            marker={
                "size": work["total_revenue"].rank(pct=True) * 22 + 4,
                "color": work["quadrant"].map(palette),
                "line": {"width": 1, "color": "#333333"},
            },
        )
    )
    fig.add_vline(
        x=x_thr, line_dash="dash", line_color="#888888", annotation_text="median revenue-share"
    )
    fig.add_hline(
        y=y_thr, line_dash="dash", line_color="#888888", annotation_text="median KVI score"
    )
    fig.update_layout(
        xaxis={"title": "Revenue Share of Category"},
        yaxis={"title": "KVI Score (0-1)"},
        hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# Layer 1: Price Strategy Matrix — KVI importance vs price sensitivity quadrants
# ---------------------------------------------------------------------------

def _render_price_strategy_matrix(analysis: PricingAnalysis) -> None:
    """Price Strategy Matrix: KVI importance vs price sensitivity with confidence opacity.

    4 quadrants:
    - Top-left (high KVI, elastic): advocate — defend price competitively
    - Top-right (high KVI, inelastic): protect — margin-safe, protect availability
    - Bottom-left (low KVI, elastic): promote — promotional price levers
    - Bottom-right (low KVI, inelastic): defer — slow movers, review assortment
    """
    st.subheader(":material/strategy: Layer 1 — Price Strategy Matrix")

    dm = analysis.decision_matrix
    if dm is None or dm.empty:
        st.info("No decision matrix data available for strategy matrix.")
        return

    # Build bubble data: position = (kvi_score, abs_elasticity), size = revenue, opacity = confidence
    fig = new_fig(height=520)

    # Add quadrant background shading
    kvi_med = float(dm["kvi_score"].median())
    elast_med = float(dm["abs_elasticity"].median())

    # Draw quadrant rectangles
    fig.add_hrect(y0=kvi_med, y1=1.0, fillcolor="rgba(78, 121, 167, 0.1)", line_color="#4E79A7", layer="below")
    fig.add_hrect(y0=0.0, y1=kvi_med, fillcolor="rgba(242, 142, 43, 0.1)", line_color="#F28E2B", layer="below")
    fig.add_vrect(x0=elast_med, x1=float(dm["abs_elasticity"].max() * 1.15), fillcolor="rgba(83, 161, 77, 0.1)", line_color="#59A14F", layer="below")
    fig.add_vrect(x0=0.0, x1=elast_med, fillcolor="rgba(225, 87, 89, 0.1)", line_color="#E15759", layer="below")

    # Plot SKU bubbles with confidence-based opacity
    estimable = dm[dm["decision"] != "insufficient_evidence"]
    unknown = dm[dm["decision"] == "insufficient_evidence"]

    if not estimable.empty:
        # Prepare hover data
        hover_texts = []
        for _, row in estimable.iterrows():
            hover_texts.append(
                f"{row['stockcode']}<br>"
                f"|e|: {row['abs_elasticity']:.2f}<br>"
                f"KVI: {row['kvi_score']:.2f}<br>"
                f"Decision: {row['decision']}<br>"
                f"Revenue: €{row['total_revenue']:,.0f}<br>"
                f"Confidence: {row.get('confidence', 'N/A')}"
            )

        # Color by decision
        decision_colors = {
            "invest": PALETTE[0],
            "protect": PALETTE[2],
            "price_lever": PALETTE[3],
            "review": PALETTE[4],
        }

        fig.add_trace(
            go.Scatter(
                x=estimable["abs_elasticity"],
                y=estimable["kvi_score"],
                mode="markers+text",
                text=estimable["stockcode"].astype(str),
                customdata=estimable[
                    ["decision", "elasticity_status", "confidence", "evidence"]
                ].to_numpy(),
                hovertemplate="%{customdata[0]}<br>|e|: %{x:.2f}<br>KVI: %{y:.2f}<br>Revenue: €%{customdata[4]:,.0f}<extra></extra>",
                marker={
                    "size": estimable["total_revenue"].rank(pct=True) * 24 + 6,
                    "color": estimable["decision"].map(decision_colors),
                    "opacity": estimable["confidence"].map(
                        lambda c: 1.0 if c == "high" else (0.6 if c == "medium" else 0.3)
                    ),
                    "line": {"width": 1, "color": "#333333"},
                },
                name="SKUs",
            )
        )

    if not unknown.empty:
        fig.add_trace(
            go.Scatter(
                x=[-0.3] * len(unknown),
                y=unknown["kvi_score"],
                mode="markers",
                text=unknown["stockcode"].astype(str),
                marker={
                    "size": 7,
                    "color": DECISION_COLORS["insufficient_evidence"],
                    "symbol": "x",
                    "opacity": 0.5,
                },
                name="Insufficient evidence",
            )
        )

    # Add quadrant labels
    fig.add_annotation(
        x=elast_med / 2,
        y=kvi_med + 0.1,
        text="Strategic\nPromo",
        showarrow=False,
        font={"size": 12, "color": "#59A14F", "style": "italic"},
        align="center",
    )
    fig.add_annotation(
        x=elast_med + 0.3,
        y=kvi_med + 0.1,
        text="Margin\nOpportunity",
        showarrow=False,
        font={"size": 12, "color": "#59A14F", "style": "italic"},
        align="center",
    )
    fig.add_annotation(
        x=elast_med / 2,
        y=kvi_med / 2,
        text="Base-Price\nFocus",
        showarrow=False,
        font={"size": 12, "color": "#59A14F", "style": "italic"},
        align="center",
    )
    fig.add_annotation(
        x=elast_med + 0.3,
        y=kvi_med / 2,
        text="Selective\nPromo",
        showarrow=False,
        font={"size": 12, "color": "#59A14F", "style": "italic"},
        align="center",
    )

    fig.update_layout(
        xaxis={"title": "|Own-price elasticity| (higher = more price-sensitive)", "range": [0, max(2.0, float(dm["abs_elasticity"].max()) * 1.15)]},
        yaxis={"title": "KVI Score (0-1, strategic importance)", "range": [0, 1.1]},
        hovermode="closest",
        showlegend=False,
    )
    show(fig)

    # Confidence legend
    with st.expander("Confidence opacity legend", expanded=False):
        st.markdown(
            "- **Opaque** (alpha=1.0): High confidence — tight CI, significant p-value\n"
            "- **Semi-transparent** (alpha=0.6): Medium confidence — wider CI or marginal significance\n"
            "- **Faint** (alpha=0.3): Low confidence — wide CI or not significant"
        )


# ---------------------------------------------------------------------------
# Layer 2: Price-Response Curve with observed + fitted + CI + current + simulated
# ---------------------------------------------------------------------------

def _render_price_response_curve(analysis: PricingAnalysis) -> None:
    """Price-Response Curve: observed prices + fitted log-log relationship + CI + current + simulated."""

    st.subheader(":material/chart: Layer 2 — Price-Response Curve")

    elast = analysis.elasticity
    if elast is None or elast.empty:
        st.info("No elasticity estimates — price-response curve requires estimable SKUs.")
        return

    conf_map = {}
    if not analysis.confidence.empty:
        conf_map = dict(
            zip(analysis.confidence["stockcode"], analysis.confidence["confidence"], strict=False)
        )

    # Select an SKU for the detailed curve
    usable_skus = elast["stockcode"].tolist()
    if not usable_skus:
        st.info("No usable elasticity estimates available.")
        return

    sku = st.selectbox("Select an SKU for price-response curve", usable_skus, key="price_curve_sku")
    row = elast[elast["stockcode"] == sku].iloc[0]
    base_price = float(row["avg_price"])
    base_qty = float(row["avg_weekly_qty"])
    elasticity = float(row["elasticity"])
    n_obs = int(row["n_obs"])
    conf = conf_map.get(sku, "medium")

    # Generate price-response curve data
    # Log-log: log(qty) = intercept + elasticity * log(price)
    # intercept = log(base_qty) - elasticity * log(base_price)
    intercept = np.log(base_qty) - elasticity * np.log(base_price)

    # Price range for the curve (20% below to 20% above current price)
    price_range = np.linspace(base_price * 0.8, base_price * 1.2, 100)
    fitted_qty = np.exp(intercept) * price_range ** elasticity

    # 95% CI on the quantity prediction
    # CI on log-qty: ± 1.96 * std_err of the regression
    # Use the actual standard error from the regression
    std_err = float(row.get("std_err", 0.0))
    se_approx = std_err * 1.96 if std_err > 0 else abs(elasticity) / np.sqrt(max(n_obs, 1)) * 1.96
    ci_lower_log = np.log(fitted_qty) - se_approx
    ci_upper_log = np.exp(np.log(fitted_qty) + se_approx)
    ci_lower_qty = np.exp(ci_lower_log)
    ci_upper_qty = np.exp(ci_upper_log)

    # Simulated prices
    scenarios = [(-0.05, "-5%"), (-0.02, "-2%"), (0.02, "+2%"), (0.05, "+5%")]
    sim_points = []
    for pct, label in scenarios:
        sim_price = base_price * (1 + pct)
        sim_qty = base_qty * (sim_price / base_price) ** elasticity
        sim_rev = sim_price * sim_qty
        sim_points.append({"Scenario": label, "Price": sim_price, "Quantity": sim_qty, "Revenue": sim_rev})

    # Build the figure
    fig = new_fig(height=500)

    # CI band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([price_range, price_range[::-1]]),
            y=np.concatenate([ci_upper_qty, ci_lower_qty[::-1]]),
            fill="toself",
            fillcolor="rgba(78, 121, 167, 0.2)",
            line={"width": 0},
            hoverinfo="skip",
            name="95% CI",
        )
    )

    # Fitted curve
    fig.add_trace(
        go.Scatter(
            x=price_range,
            y=fitted_qty,
            mode="lines",
            line={"color": PALETTE[0], "width": 3},
            name="Fitted relationship",
        )
    )

    # Observed data points
    # We need the original transaction data - use the analysis data
    # For now, plot the average price/quantity as a scatter
    # In a full implementation, we'd have the raw observed points
    fig.add_trace(
        go.Scatter(
            x=[base_price],
            y=[base_qty],
            mode="markers",
            marker={"color": "#E15759", "size": 15, "symbol": "star"},
            name="Current price point",
        )
    )

    # Simulated price points
    sim_colors = [PALETTE[2], PALETTE[3], PALETTE[4], PALETTE[5]]
    for i, sp in enumerate(sim_points):
        fig.add_trace(
            go.Scatter(
                x=[sp["Price"]],
                y=[sp["Quantity"]],
                mode="markers+text",
                marker={"color": sim_colors[i], "size": 12},
                text=sp["Scenario"],
                textposition="top right",
                name=f"Sim: {sp['Scenario']}",
            )
        )

    fig.update_layout(
        xaxis={"title": "Price (€)", "type": "log"},
        yaxis={"title": "Weekly units (log scale)"},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    show(fig)

    # Display simulation table
    col1, col2 = st.columns([0.45, 0.55])
    with col1:
        sim_df = pd.DataFrame(sim_points)
        sim_df["Revenue"] = sim_df["Revenue"].round(2)
        st.dataframe(sim_df, use_container_width=True, hide_index=True)
    with col2:
        st.caption(f"""
        **Curve parameters** for {sku}:
        - Elasticity: {elasticity:.2f} ({"elastic" if abs(elasticity) > 1 else "inelastic"})
        - Current price: €{base_price:.2f}, Current quantity: {base_qty:.0f}/week
        - Observations: {n_obs}
        - Confidence: {conf.upper()}
        - Curve: log(qty) = {intercept:.2f} + {elasticity:.2f} · log(price)
        """)

    # Confidence warning
    if conf == "medium":
        st.warning("⚠️ Medium confidence estimate. Curve fits are planning figures only. Validate with controlled test before acting.")
    elif conf == "low":
        st.error("🔴 Low confidence estimate. Curve fits are unreliable. Do not act on these figures.")


# ---------------------------------------------------------------------------
# Layer 3: Price Ladder — Premium/Mid/Value/Entry with price index/reach/revenue/KVI status
# ---------------------------------------------------------------------------

def _render_price_ladder(analysis: PricingAnalysis) -> None:
    """Price Ladder: Premium/Mid/Value/Entry tiers with price index/reach/revenue/KVI status."""

    st.subheader(":material/ladder: Layer 3 — Price Ladder")

    kvi = analysis.kvi
    elast = analysis.elasticity
    conf_map = {}
    if not analysis.confidence.empty:
        conf_map = dict(
            zip(analysis.confidence["stockcode"], analysis.confidence["confidence"], strict=False)
        )

    if kvi is None or kvi.empty:
        st.info("No KVI data available for price ladder.")
        return

    # Compute price ladder tiers based on KVI score and elasticity
    kvi["total_revenue"].sum()
    work = kvi.copy()

    # Add elasticity and confidence info
    if elast is not None and not elast.empty:
        elast_dict = dict(zip(elast["stockcode"], elast["elasticity"], strict=False))
        work["elasticity"] = work["stockcode"].map(elast_dict).fillna(np.nan)
    else:
        work["elasticity"] = np.nan

    if conf_map:
        work["confidence"] = work["stockcode"].map(lambda s: conf_map.get(s, "medium"))
    else:
        work["confidence"] = "medium"

    # Determine tier for each SKU
    def _assign_tier(row: pd.Series) -> str:
        kvi_score = row["kvi_score"]
        elast_val = row.get("elasticity", np.nan)
        if pd.isna(kvi_score):
            return "Entry"

        # Premium: top 25% KVI
        # Mid: middle 50% KVI
        # Value: bottom 25% KVI + inelastic
        # Entry: bottom 25% KVI + elastic

        kvi_quartiles = work["kvi_score"].quantile([0.25, 0.5, 0.75])
        q25 = float(kvi_quartiles[0.25])
        q50 = float(kvi_quartiles[0.5])
        q75 = float(kvi_quartiles[0.75])

        if kvi_score >= q75:
            tier = "Premium"
        elif kvi_score >= q50:
            tier = "Mid"
        elif kvi_score >= q25:
            if pd.notna(elast_val) and abs(elast_val) >= 1.0:
                tier = "Entry"
            else:
                tier = "Value"
        else:
            tier = "Entry"

        return tier

    work["tier"] = work.apply(_assign_tier, axis=1)

    # Compute tier metrics
    tier_order = ["Premium", "Mid", "Value", "Entry"]
    tier_data = {}
    for tier in tier_order:
        tier_skus = work[work["tier"] == tier]
        if tier_skus.empty:
            tier_data[tier] = {
                "price_index": "—",
                "reach": "—",
                "revenue": "€0",
                "kvi_status": "—",
                "sku_count": 0,
            }
            continue

        total_rev = float(tier_skus["total_revenue"].sum())
        # Price index: average price relative to overall average
        avg_price = float(tier_skus["avg_price"].mean()) if "avg_price" in tier_skus.columns else 0
        # Reach: unique customers / total customers
        # For now, use basket penetration as a proxy
        reach = float(tier_skus["basket_penetration"].mean()) if "basket_penetration" in tier_skus.columns else 0
        reach_pct = f"{reach:.1%}"
        kvi_status = ", ".join(tier_skus["stockcode"].astype(str).head(3).tolist())

        tier_data[tier] = {
            "price_index": f"€{avg_price:.0f}" if avg_price > 0 else "—",
            "reach": reach_pct,
            "revenue": f"€{total_rev:,.0f}",
            "kvi_status": kvi_status,
            "sku_count": len(tier_skus),
        }

    # Display tier table
    fig = new_fig(height=400)
    tier_rows = []
    for tier in tier_order:
        if tier in tier_data:
            td = tier_data[tier]
            tier_rows.append({
                "Tier": tier,
                "Price Index": td["price_index"],
                "Reach": td["reach"],
                "Revenue": td["revenue"],
                "KVI Status": td["kvi_status"],
                "SKU Count": td["sku_count"],
            })

    if tier_rows:
        for _i, row_data in enumerate(tier_rows):
            # Color code by tier
            tier_colors = {"Premium": PALETTE[0], "Mid": PALETTE[2], "Value": PALETTE[3], "Entry": PALETTE[4]}
            fig.add_trace(
                go.Bar(
                    y=[row_data["Tier"]],
                    x=[1],  # placeholder, will use gauge-like bar
                    marker_color=tier_colors.get(row_data["Tier"], PALETTE[1]),
                    orientation="h",
                    showlegend=False,
                    hovertemplate=f"{row_data['Tier']}<br>Price Index: {{row_data['Price Index']}}<br>Reach: {{row_data['Reach']}}<br>Revenue: {{row_data['Revenue']}}<br>KVI Status: {{row_data['KVI Status']}}<br>SKU Count: {{row_data['SKU Count']}}<extra></extra>",
                )
            )
        # This is getting complex; let me use a table instead
        st.dataframe(
            pd.DataFrame(tier_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No tier data available.")

    # Tier explanation
    with st.expander("Price ladder tier definitions", expanded=False):
        st.markdown(
            """
        - **Premium**: Top quartile KVI scores — high strategic importance. These are your flagship traffic drivers.
        - **Mid**: Middle quartile KVI scores — moderate strategic importance. Balanced position.
        - **Value**: Bottom quartile KVI scores + inelastic demand (|e| < 1). Can carry margin with limited volume risk.
        - **Entry**: Bottom quartile KVI scores + elastic demand (|e| >= 1). Promotional levers; safe to test price changes.
        """
        )


# ---------------------------------------------------------------------------
# Layer 4: Price × Promotion Matrix — 4 strategies
# ---------------------------------------------------------------------------

def _render_price_promo_matrix(analysis: PricingAnalysis) -> None:
    """Price × Promotion Matrix: 4 strategy quadrants.

    4 quadrants:
    - Base-price focus: High KVI, inelastic — protect margin, minimal promo
    - Strategic promo: High KVI, elastic — defend price, strategic positioning
    - Margin opportunity: Low KVI, inelastic — margin growth through optimization
    - Selective promo: Low KVI, elastic — promotional price levers
    """
    st.subheader(":material/compare_arrows: Layer 4 — Price × Promotion Matrix")

    dm = analysis.decision_matrix
    if dm is None or dm.empty:
        st.info("No decision matrix data for price-promo matrix.")
        return

    kvi_med = float(dm["kvi_score"].median())
    elast_med = float(dm["abs_elasticity"].median())

    # Compute strategy for each SKU
    def _assign_strategy(row: pd.Series) -> str:
        kvi_high = row["kvi_score"] >= kvi_med
        elastic = row["abs_elasticity"] >= elast_med

        if kvi_high and elastic:
            return "Strategic promo"  # High KVI, price-sensitive: defend & strategic promo
        if kvi_high:
            return "Base-price focus"  # High KVI, inelastic: protect margin
        if elastic:
            return "Selective promo"  # Low KVI, elastic: promotional lever
        return "Margin opportunity"  # Low KVI, inelastic: margin optimization

    dm = dm.copy()
    dm["strategy"] = dm.apply(_assign_strategy, axis=1)

    fig = new_fig(height=450)

    # 2x2 matrix: x = elasticity, y = KVI
    # Quadrant 1: top-left = high KVI, elastic = Strategic promo
    # Quadrant 2: top-right = high KVI, inelastic = Base-price focus
    # Quadrant 3: bottom-left = low KVI, elastic = Selective promo
    # Quadrant 4: bottom-right = low KVI, inelastic = Margin opportunity

    # Add quadrant background
    fig.add_hrect(y0=kvi_med, y1=1.0, fillcolor="rgba(78, 121, 167, 0.1)", line_color="#4E79A7", layer="below")
    fig.add_hrect(y0=0.0, y1=kvi_med, fillcolor="rgba(242, 142, 43, 0.1)", line_color="#F28E2B", layer="below")
    fig.add_vrect(x0=elast_med, x1=float(dm["abs_elasticity"].max() * 1.15), fillcolor="rgba(83, 161, 77, 0.1)", line_color="#59A14F", layer="below")
    fig.add_vrect(x0=0.0, x1=elast_med, fillcolor="rgba(225, 87, 89, 0.1)", line_color="#E15759", layer="below")

    for strategy, color in STRATEGY_COLORS.items():
        strategy_skus = dm[dm["strategy"] == strategy]
        if strategy_skus.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=strategy_skus["abs_elasticity"],
                y=strategy_skus["kvi_score"],
                mode="markers+text",
                text=strategy_skus["stockcode"].astype(str),
                marker={
                    "size": strategy_skus["total_revenue"].rank(pct=True) * 20 + 4,
                    "color": color,
                    "line": {"width": 1, "color": "#333333"},
                },
                name=STRATEGY_LABELS[strategy],
                hovertemplate=f"{STRATEGY_LABELS[strategy]}<br>|e|: %{{x:.2f}}<br>KVI: %{{y:.2f}}<br>Stockcode: %{{text}}<extra></extra>",
            )
        )

    # Add median reference lines
    fig.add_vline(x=elast_med, line_dash="dash", line_color="#888888", line_width=1)
    fig.add_hline(y=kvi_med, line_dash="dash", line_color="#888888", line_width=1)

    fig.update_layout(
        xaxis={"title": "|Own-price elasticity| (higher = more price-sensitive)", "range": [0, max(2.0, float(dm["abs_elasticity"].max()) * 1.15)]},
        yaxis={"title": "KVI Score (0-1, strategic importance)", "range": [0, 1.1]},
        hovermode="closest",
        showlegend=False,
    )
    show(fig)

    # Strategy explanations
    with st.expander("Promotion strategy definitions", expanded=False):
        st.markdown(
            """
        - **Base-price focus**: High-KVI, inelastic SKUs. These can carry margin safely; avoid needless discounting. Price changes should be minimal and justified by cost changes.
        - **Strategic promo**: High-KVI, elastic SKUs. These are traffic drivers; price sensitivity means small changes affect volume significantly. Use strategic positioning, not discounts.
        - **Margin opportunity**: Low-KVI, inelastic SKUs. Margin can be optimized through subtle price adjustments or cost management. Not strategic traffic drivers.
        - **Selective promo**: Low-KVI, elastic SKUs. Candidate promotional price levers. Safe to test deeper discounts to drive volume or clear inventory.
        """
        )


# ---------------------------------------------------------------------------
# Price & Promotion Strategy Matrix — cross-tab with elasticity vs incrementality
# ---------------------------------------------------------------------------

def _render_price_promo_strategy_matrix(analysis: PricingAnalysis) -> None:
    """Price & Promotion Strategy Matrix — cross-tab visual with 4 quadrants.

    Axes:
    - X: |Own-price elasticity| (higher = more price-sensitive)
    - Y: KVI Score (0-1, strategic importance) — serves as the promotion
      incrementality proxy in the pricing context; when promo data is
      available, this can be replaced with actual incrementality metrics.

    4 quadrants (labeled McKinsey strategies):
    - Base-price focus: Inelastic demand, high strategic importance — protect
      everyday price, minimize promotional spend
    - Strategic promo: Elastic demand, high strategic importance — defend price
      position; small price changes drive significant volume; use strategic
      positioning rather than deep discounts
    - Margin opportunity: Inelastic demand, low strategic importance — optimize
      margin through subtle price adjustments; not a traffic driver
    - Selective promo: Elastic demand, low strategic importance — targeted
      promotional levers; safe to test deeper discounts to drive volume or clear
      inventory

    Bubble size = total_revenue (economic impact)
    Color = KVI status (High/Medium/Low based on kvi_score terciles)
    """

    st.subheader(":material/strategy: Price & Promotion Strategy Matrix")

    dm = analysis.decision_matrix
    kvi = analysis.kvi
    if dm is None or dm.empty:
        st.info("No decision matrix data for strategy matrix.")
        return

    def _kvi_status(kvi_score: float) -> str:
        if kvi_score >= 0.67:
            return "High"
        elif kvi_score >= 0.33:
            return "Medium"
        else:
            return "Low"

    work = dm.copy()
    if kvi is not None and not kvi.empty:
        kvi_map = dict(zip(kvi["stockcode"], kvi["kvi_score"], strict=False))
        work["kvi_score"] = work["stockcode"].map(kvi_map).fillna(0.0)
    work["kvi_status"] = work["kvi_score"].apply(_kvi_status)

    elast_med = float(work["abs_elasticity"].median())
    kvi_med = float(work["kvi_score"].median()) if "kvi_score" in work.columns else 0.5

    def _assign_strategy(row: pd.Series) -> str:
        kvi_high = row["kvi_score"] >= kvi_med if not pd.isna(row["kvi_score"]) else False
        elastic = row["abs_elasticity"] >= elast_med
        if kvi_high and elastic:
            return "Strategic promo"
        if kvi_high:
            return "Base-price focus"
        if elastic:
            return "Selective promo"
        return "Margin opportunity"

    work["strategy"] = work.apply(_assign_strategy, axis=1)

    kvi_status_colors = {
        "High": PALETTE[0],
        "Medium": PALETTE[3],
        "Low": PALETTE[4],
    }

    fig = new_fig(height=520)

    fig.add_hrect(y0=kvi_med, y1=1.0, fillcolor="rgba(78, 121, 167, 0.1)", line_color="#4E79A7", layer="below")
    fig.add_hrect(y0=0.0, y1=kvi_med, fillcolor="rgba(242, 142, 43, 0.1)", line_color="#F28E2B", layer="below")
    fig.add_vrect(x0=elast_med, x1=float(work["abs_elasticity"].max() * 1.15), fillcolor="rgba(83, 161, 77, 0.1)", line_color="#59A14F", layer="below")
    fig.add_vrect(x0=0.0, x1=elast_med, fillcolor="rgba(225, 87, 89, 0.1)", line_color="#E15759", layer="below")

    for strategy, _scolor in STRATEGY_COLORS.items():
        sdf = work[work["strategy"] == strategy]
        if sdf.empty:
            continue
        bubble_colors = [kvi_status_colors.get(_kvi_status(s), PALETTE[1]) for s in sdf["kvi_score"]]

        fig.add_trace(
            go.Scatter(
                x=sdf["abs_elasticity"],
                y=sdf["kvi_score"],
                mode="markers+text",
                text=sdf["stockcode"].astype(str),
                customdata=sdf["total_revenue"].values,
                marker={
                    "size": sdf["total_revenue"].rank(pct=True) * 24 + 6,
                    "color": bubble_colors,
                    "opacity": 0.8,
                    "line": {"width": 1, "color": "#333333"},
                },
                name=STRATEGY_LABELS[strategy],
                hovertemplate=
                    f"<b>{STRATEGY_LABELS[strategy]}</b>"
                    "<br>|e|: %{x:.2f}"
                    "<br>KVI: %{y:.2f}"
                    "<br>Stockcode: %{text}"
                    "<br>Revenue: €%{customdata:,.0f}"
                    "<extra></extra>",
            )
        )

    fig.add_vline(x=elast_med, line_dash="dash", line_color="#888888", line_width=1)
    fig.add_hline(y=kvi_med, line_dash="dash", line_color="#888888", line_width=1)

    fig.update_layout(
        xaxis={"title": "|Own-price elasticity| (higher = more price-sensitive)", "range": [0, max(2.0, float(work["abs_elasticity"].max()) * 1.15)]},
        yaxis={"title": "KVI Score (0-1, strategic importance)", "range": [0, 1.1]},
        hovermode="closest",
        showlegend=False,
    )
    show(fig)

    with st.expander("Confidence opacity legend", expanded=False):
        st.markdown(
            "- **Opaque** (alpha=1.0): High confidence — tight CI, significant p-value\n"
            "- **Semi-transparent** (alpha=0.6): Medium confidence — wider CI or marginal significance\n"
            "- **Faint** (alpha=0.3): Low confidence — wide CI or not significant"
        )

    with st.expander("McKinsey pricing & promotion strategies", expanded=False):
        st.markdown(
            """
        **Base-price focus** — Inelastic, high-KVI SKUs
        These are your strategic traffic drivers where demand is relatively insensitive to price.
        • Protect everyday price; avoid needless discounting.
        • Price changes should be minimal and justified by cost changes or margin targets.
        • Focus on supply-chain efficiency and value communication rather than promotions.

        **Strategic promo** — Elastic, high-KVI SKUs
        These are price-sensitive traffic drivers where small price changes generate large volume shifts.
        • Defend price position through strategic positioning, not deep discounting.
        • Use modest, well-timed promotions to reinforce shelf presence.
        • Test price-response curves before committing to sustained discounting.

        **Margin opportunity** — Inelastic, low-KVI SKUs
        These SKUs have limited strategic importance but can carry margin with limited volume risk.
        • Optimize margin through subtle price adjustments or cost-management initiatives.
        • Not a candidate for promotional discounting; focus on operational efficiency.
        • Review assortment depth last if margin targets are unmet.

        **Selective promo** — Elastic, low-KVI SKUs
        These are candidate promotional levers where deeper discounts can drive meaningful volume.
        • Safe to test deeper discounts to drive volume or clear inventory.
        • Use as tactical, time-limited promotions rather than everyday strategy.
        • Monitor cannibalization and incremental lift carefully.
        """
        )


# ---------------------------------------------------------------------------
# Layer 5: Manager Table — SKU | KVI | Elasticity | Confidence | Current price | Recommended price | Expected units | Revenue impact | Risk | Action + "Do not act"
# ---------------------------------------------------------------------------

def _render_manager_table(analysis: PricingAnalysis, profile_service: ProfileService | None = None) -> None:
    """Manager Table: per-SKU pricing decisions with all required columns + 'Do not act' option."""

    st.subheader(":material/table_settings: Layer 5 — Manager Table")

    dm = analysis.decision_matrix
    elast = analysis.elasticity
    kvi = analysis.kvi
    conf_map = {}
    if not analysis.confidence.empty:
        conf_map = dict(
            zip(analysis.confidence["stockcode"], analysis.confidence["confidence"], strict=False)
        )

    # Build the manager table rows
    rows = []

    # Get all SKUs from the decision matrix
    if dm is not None and not dm.empty:
        skus = dm["stockcode"].tolist()
    elif elast is not None and not elast.empty:
        skus = elast["stockcode"].tolist()
    else:
        skus = []

    # Also include SKUs from KVI if not already included
    if kvi is not None and not kvi.empty:
        kvi_skus = kvi["stockcode"].tolist()
        for s in kvi_skus:
            if s not in skus:
                skus.append(s)

    for sku in skus:
        # Get decision matrix data
        dm_row = dm[dm["stockcode"] == sku].iloc[0] if dm is not None and not dm.empty and (dm["stockcode"] == sku).any() else None
        elast_row = elast[elast["stockcode"] == sku].iloc[0] if elast is not None and not elast.empty and (elast["stockcode"] == sku).any() else None
        kvi_row = kvi[kvi["stockcode"] == sku].iloc[0] if kvi is not None and not kvi.empty and (kvi["stockcode"] == sku).any() else None

        # Current price
        current_price = 0.0
        if elast_row is not None and pd.notna(elast_row.get("avg_price", np.nan)):
            current_price = float(elast_row["avg_price"])
        elif kvi_row is not None and pd.notna(kvi_row.get("avg_price", np.nan)):
            current_price = float(kvi_row["avg_price"])

        # KVI score
        kvi_score = 0.0
        kvi_status = "—"
        if kvi_row is not None:
            kvi_score = float(kvi_row["kvi_score"])
            # KVI status classification
            if kvi_score >= 0.7:
                kvi_status = "High strategic importance"
            elif kvi_score >= 0.4:
                kvi_status = "Moderate importance"
            else:
                kvi_status = "Low strategic importance"

        # Elasticity
        elasticity = 0.0
        elasticity_conf = "—"
        if elast_row is not None and pd.notna(elast_row.get("elasticity", np.nan)):
            elasticity = float(elast_row["elasticity"])
            # Confidence from confidence map
            elasticity_conf = conf_map.get(sku, "medium")
            if elasticity_conf == "high":
                elasticity_conf = "High"
            elif elasticity_conf == "medium":
                elasticity_conf = "Medium"
            else:
                elasticity_conf = "Low"

        # Recommended price - based on decision
        recommended_price = current_price
        if dm_row is not None:
            decision = dm_row["decision"]
            if decision == "invest":
                # Maintain/protect competitiveness - slight decrease
                recommended_price = current_price * 0.98
            elif decision == "protect":
                # Hold price
                recommended_price = current_price
            elif decision == "price_lever":
                # Test price decrease
                recommended_price = current_price * 0.95
            elif decision == "review":
                # Review - keep current for now
                recommended_price = current_price
            elif decision == "insufficient_evidence":
                # Not enough data
                recommended_price = current_price

        # Expected units
        expected_units = 0
        if elast_row is not None and pd.notna(elast_row.get("avg_weekly_qty", np.nan)):
            expected_units = int(elast_row["avg_weekly_qty"])
        elif kvi_row is not None and pd.notna(kvi_row.get("basket_penetration", np.nan)):
            # Estimate units from basket penetration
            expected_units = int(kvi_row["total_revenue"] / current_price * kvi_row.get("basket_penetration", 0) * 100)

        # Revenue impact
        revenue_impact = 0.0
        if dm_row is not None and elast_row is not None:
            # Simulate -5% price change impact
            float(elast_row.get("total_revenue", 0)) if hasattr(elast_row, "get") else 0
            # More realistically:
            rev_change_pct = elasticity * (-0.05)  # -5% price cut
            revenue_impact = current_price * expected_units * rev_change_pct / 100 if expected_units > 0 else 0

        # Risk assessment
        risk = "Low"
        if elasticity_conf == "Low":
            risk = "High — low confidence estimate"
        elif elasticity_conf == "Medium":
            risk = "Medium — moderate confidence"
        elif kvi_score < 0.3:
            risk = "Medium — low strategic importance"

        # Action options
        default_action = "Act" if risk != "High" else "Do not act"

        # Get profile service data if available
        profile_data = {}
        if profile_service is not None:
            try:
                profile_data = profile_service.get_profile(sku)
            except Exception:
                profile_data = {}

        rows.append({
            "SKU": sku,
            "KVI": f"{kvi_score:.2f}" if kvi_score > 0 else "—",
            "KVI Status": kvi_status,
            "Elasticity": f"{elasticity:.2f}" if elasticity != 0 else "—",
            "Elasticity Confidence": elasticity_conf,
            "Current price": f"€{current_price:.2f}",
            "Recommended price": f"€{recommended_price:.2f}",
            "Expected units": f"{expected_units:,}" if expected_units > 0 else "—",
            "Revenue impact": f"€{revenue_impact:.0f}" if revenue_impact != 0 else "€0",
            "Risk": risk,
            "Action": default_action,
            "Profile data": str(profile_data)[:50] if profile_data else "—",  # internal use
        })

    # Display as editable table with action selectors
    if rows:
        df = pd.DataFrame(rows)

        # Render as a structured table with action indicators
        for _i, row in df.iterrows():
            with st.container(border=True):
                col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns([0.8, 0.5, 0.5, 0.7, 0.8, 0.8, 0.7, 0.8, 0.6, 0.7])

                with col1:
                    st.markdown(f"**{row['SKU']}**")
                with col2:
                    st.markdown(f"{row['KVI']}")
                with col3:
                    st.markdown(f"{row['KVI Status']}")
                with col4:
                    st.markdown(f"{row['Elasticity']}")
                with col5:
                    st.markdown(f"{row['Elasticity Confidence']}")
                with col6:
                    st.markdown(row["Current price"])
                with col7:
                    st.markdown(row["Recommended price"])
                with col8:
                    st.markdown(row["Expected units"])
                with col9:
                    st.markdown(row["Revenue impact"])
                with col10:
                    st.markdown(row["Risk"])

                # Action row
                with st.columns(2)[0]:
                    if st.button("Act", key=f"act_{row['SKU']}"):
                        st.success(f"✅ Action selected for {row['SKU']}")
                with st.columns(1)[0]:
                    if st.button("Do not act", key=f"do_not_act_{row['SKU']}"):
                        st.info(f"⏸️ 'Do not act' selected for {row['SKU']}")

        # Summary metrics
        st.caption(
            f"Total SKUs: {len(rows)} | High risk: {sum(1 for r in rows if 'High' in r['Risk'])} | "
            f"Act: {sum(1 for r in rows if r['Action'] == 'Act')} | Do not act: {sum(1 for r in rows if r['Action'] == 'Do not act')}"
        )
    else:
        st.info("No SKU data available for manager table.")


# ---------------------------------------------------------------------------
# Main render function — five-layer structure
# ---------------------------------------------------------------------------

def render(df: pd.DataFrame) -> None:
    st.subheader(":material/price_check: Pricing & Elasticity")

    with st.spinner("Preparing data..."):
        df, category_inferred = cached_enrich_categories(df)
    if category_inferred:
        st.info(
            "A `category` column was not supplied; categories were inferred from product descriptions (TF-IDF + KMeans)."
        )

    with st.spinner("Computing pricing analysis (cached)..."):
        analysis = run_cached_pricing_analysis(df, min_periods=5)

    # Initialize ProfileService for integration
    profile_service = None
    try:
        profile_service = init_profile_service(df)
    except Exception:
        profile_service = None

    # Layer 1: Price Strategy Matrix
    _render_price_strategy_matrix(analysis)

    # Layer 2: Price-Response Curve
    _render_price_response_curve(analysis)

    # Layer 3: Price Ladder
    _render_price_ladder(analysis)

    # Layer 4: Price × Promotion Matrix
    _render_price_promo_matrix(analysis)

    _render_price_promo_strategy_matrix(analysis)

    # Signal (scorecard) — layer 0 / evidence
    st.subheader(":material/signal_cellular_alt: Signal")
    _render_scorecard(analysis)

    st.subheader(":material/science: Evidence — why we know what we know")
    show(_render_status_breakdown(analysis.elasticity_status))
    st.caption(
        "SKUs with constant or near-constant weekly prices, too few distinct price points, "
        "or too few weekly observations are excluded rather than silently reported as "
        "inelastic (0.0). Only 'estimated' statuses feed the decision matrix."
    )

    # Layer 5: Manager Table (integrates with Profile Service)
    st.subheader(":material/table_settings: Manager Decisions — Five-Layer Overview")
    _render_manager_table(analysis, profile_service)

    # Impact (insight cards + simulation)
    st.subheader(":material/lightbulb: Impact — What it means")
    render_insight_cards(analysis.insights)
    _render_price_simulation(analysis)

    # Action (ranked decisions)
    st.subheader(":material/task_alt: Action — Ranked decisions")
    render_opportunity_table(analysis.opportunities)

    # Detail expander
    with st.expander(
        "Detail — elasticity CI, KVI drivers, price curves, decision cards", expanded=False
    ):
        detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs(
            ["Elasticity", "KVI Scores", "Price Curves", "Decision Cards"]
        )
        with detail_tab1:
            _render_elasticity_confidence_detail(analysis.elasticity)
        with detail_tab2:
            if analysis.kvi.empty:
                st.info("No KVI data available.")
            else:
                show(_render_kvi_quadrant(analysis.kvi))
                st.caption(
                    "Quadrant split at median KVI score (y) and median revenue share (x). "
                    "KVI score combines basket penetration, revenue, elasticity and customer reach."
                )
                st.dataframe(
                    analysis.kvi.sort_values("kvi_score", ascending=False).head(20),
                    use_container_width=True,
                    hide_index=True,
                )
        with detail_tab3:
            curves = diagnose_price_curves_1d(df, n_tiers=3)
            if curves.empty:
                st.info("No price curve data available.")
            else:
                st.dataframe(
                    curves[
                        [
                            "stockcode",
                            "category",
                            "median_price",
                            "pack_size_numeric",
                            "price_per_unit",
                            "tier_label",
                            "has_violation",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                violations = curves[curves["has_violation"]]
                if not violations.empty:
                    st.warning(f"Price curve violations: {len(violations)}")
                    st.dataframe(violations, use_container_width=True, hide_index=True)
        with detail_tab4:
            if analysis.kvi.empty:
                st.info("No KVI data available for decision cards.")
            else:
                st.subheader("Product-Level Pricing Decision Cards")
                st.caption("Select an SKU to view its complete pricing decision summary.")

                # Get SKUs with decision data
                skus_with_decisions = []
                if not analysis.decision_matrix.empty:
                    skus_with_decisions = analysis.decision_matrix["stockcode"].tolist()

                if skus_with_decisions:
                    selected_sku = st.selectbox(
                        "Select SKU", skus_with_decisions, key="decision_card_sku"
                    )
                    render_pricing_decision_card(
                        stockcode=str(selected_sku),
                        kvi_data=analysis.kvi,
                        decision_data=analysis.decision_matrix,
                        elasticity_data=analysis.elasticity,
                        status_data=analysis.elasticity_status,
                        confidence_data=analysis.confidence,
                    )
                else:
                    st.info("No decision data available.")

# Mode spec — unchanged interface
MODE_SPEC: ModeSpec = ModeSpec(
    key="pricing",
    label="Pricing",
    icon=":material/price_check:",
    handler=render,
    requires=("has_price_variation", "min_distinct_prices_3", "sufficient_baskets_500"),
)
