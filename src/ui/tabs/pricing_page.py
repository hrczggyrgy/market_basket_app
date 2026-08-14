"""Pricing & Elasticity tab — decision-first layout.

Follows the Retail Decision Intelligence pattern:
Signal (scorecard) -> Evidence (coverage) -> Interpretation (KVI x elasticity
decision matrix, including the explicit "insufficient evidence" zone) ->
Impact (price scenario simulation) -> Action (insight cards + ranked
opportunities) -> Detail (elasticity CI, KVI, price curves).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.cache import cached_enrich_categories, run_cached_pricing_analysis
from src.analytics.pricing import diagnose_price_curves_1d
from src.ui.components import (
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


def _render_scorecard(analysis: object) -> None:
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
    """KVI importance x |elasticity| decision landscape with unknown zone."""
    fig = new_fig(height=480)
    if dm is None or dm.empty:
        return empty_state("No SKUs for the price decision matrix")

    estimable = dm[dm["decision"] != "insufficient_evidence"]
    unknown = dm[dm["decision"] == "insufficient_evidence"]

    if not estimable.empty:
        fig.add_trace(
            go.Scatter(
                x=estimable["abs_elasticity"],
                y=estimable["kvi_score"],
                mode="markers",
                text=estimable["stockcode"].astype(str),
                customdata=estimable[["decision", "elasticity_status"]].to_numpy(),
                hovertemplate=(
                    "%{text}<br>|e| %{x:.2f} | KVI %{y:.2f}<br>"
                    "decision: %{customdata[0]} (status: %{customdata[1]})<extra></extra>"
                ),
                marker={
                    "size": estimable["total_revenue"].rank(pct=True) * 24 + 6,
                    "color": estimable["decision"].map(DECISION_COLORS),
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
    new_qty = base_qty * (1 + elasticity * pct)
    new_price = base_price * (1 + pct)
    return new_qty, new_price, new_qty * new_price


def _render_price_simulation(analysis: object) -> None:
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

    st.subheader(":material/signal_cellular_alt: Signal")
    _render_scorecard(analysis)

    st.subheader(":material/science: Evidence — why we know what we know")
    show(_render_status_breakdown(analysis.elasticity_status))
    st.caption(
        "SKUs with constant or near-constant weekly prices, too few distinct price points, "
        "or too few weekly observations are excluded rather than silently reported as "
        "inelastic (0.0). Only 'estimated' statuses feed the decision matrix."
    )

    st.subheader(":material/interactive_space: Interpretation — Price Decision Matrix")
    show(_render_decision_matrix(analysis.decision_matrix))
    st.caption(
        "x = |own-price elasticity| (higher = more price-sensitive), y = KVI importance "
        "(higher = more strategic), bubble size = revenue, colour = decision. Crosses on the "
        "left are SKUs without usable elasticity evidence — shown explicitly, never "
        "assumed inelastic. Advocates are high-KVI, price-sensitive traffic drivers: defend "
        "their price (invest). Price levers are low-KVI, elastic SKUs — use as promotional "
        "levers. Protects carry margin safely; review them last."
    )

    st.subheader(":material/lightbulb: Impact — What it means")
    render_insight_cards(analysis.insights)
    _render_price_simulation(analysis)

    st.subheader(":material/task_alt: Action — Ranked decisions")
    render_opportunity_table(analysis.opportunities)

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
                        stockcode=selected_sku,
                        kvi_data=analysis.kvi,
                        decision_data=analysis.decision_matrix,
                        elasticity_data=analysis.elasticity,
                        status_data=analysis.elasticity_status,
                        confidence_data=analysis.confidence,
                    )
                else:
                    st.info("No decision data available.")


MODE_SPEC: ModeSpec = ModeSpec(
    key="pricing",
    label="Pricing",
    icon=":material/price_check:",
    handler=render,
)
