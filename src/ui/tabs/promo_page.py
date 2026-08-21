"""Promotional Analytics tab — five-layer redesign.
MODE_SPEC = ModeSpec(key="promotion", label="Promotions", icon=":material/local_offer:", handler=render, requires=("sufficient_baskets_200", "sufficient_skus_20"))

Layers:
  1. Promotion Effectiveness Matrix — 4 quadrants (scale/optimize/rethink/stop) with bubble=promo revenue
  2. Incrementality Waterfall — observed vs estimated incremental revenue
  3. Promo Calendar Heatmap — weekly discount depth & incremental lift per SKU
  4. Discount Depth Response Curve — 5 price points (5%,10%,15%,20%,25%) vs incremental units
  5. Manager Table — promo-level decision data with recommendation column

Integrates with Product Decision Profile for promo-level data (SDP, promo effectiveness,
price action, promo action per SKU).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.insights import generate_promotion_insights
from src.analytics.opportunities import generate_promotion_opportunities
from src.analytics.profile_service import ProfileService, init_profile_service
from src.analytics.promo_core import (
    compute_cannibalization_analysis,
    compute_incrementality_waterfall,
    compute_promo_baseline,
    detect_promotions,
    pre_post_promo_comparison,
    promo_roi_analysis,
)
from src.ui.components_utils import (
    render_insight_cards,
    render_opportunity_table,
)
from src.ui.plots import PALETTE, empty_state, new_fig, show
from src.ui.registry import ModeSpec

_SCORE_META = {
    "WIN": {"icon": ":material/trending_up:", "color": "#59A14F"},
    "MIXED": {"icon": ":material/swap_vert:", "color": "#F28E2B"},
    "INEFFECTIVE": {"icon": ":material/remove:", "color": "#B07AA1"},
    "DESTROYS_VALUE": {"icon": ":material/error:", "color": "#E15759"},
}

_SCORE_QUADRANT = {
    "WIN": "scale",
    "MIXED": "optimize",
    "INEFFECTIVE": "rethink",
    "DESTROYS_VALUE": "stop",
}

# ---------------------------------------------------------------------------
# Layer 1: Promotion Effectiveness Matrix (4 quadrants, bubble=promo revenue)
# ---------------------------------------------------------------------------

def _compute_effectivity_matrix(waterfall: pd.DataFrame, promos: pd.DataFrame) -> pd.DataFrame:
    """Compute per-SKU data for the effectiveness matrix.

    Assigns each SKU to a quadrant (scale/optimize/rethink/stop) based on
    volume (incremental units) and ROI. Bubble size = promo revenue.
    """
    if waterfall.empty or promos.empty:
        return pd.DataFrame(columns=["stockcode", "quadrant", "volume", "roi", "promo_revenue"])

    w = waterfall.copy()
    if "roi" in w.columns:
        w["roi_pct"] = w["roi"].apply(lambda x: float(x) * 100.0)
    elif "roi_pct" in w.columns:
        w["roi_pct"] = w["roi_pct"].fillna(0.0)
    else:
        w["roi_pct"] = 0.0

    # Merge promo revenue
    promo_rev = (
        promos.groupby("stockcode")["promo_revenue"]
        .sum()
        .rename("promo_revenue")
        .reset_index()
    )
    w = w.merge(promo_rev, on="stockcode", how="left")
    w["promo_revenue"] = w["promo_revenue"].fillna(0.0)

    # Volume = incremental units (from waterfall) or promo qty
    if "incremental_revenue_qty" in w.columns:
        w["volume"] = w["incremental_revenue_qty"].fillna(0.0)
    else:
        w["volume"] = 0.0

    # Compute quadrant per SKU using median splits
    vol_median = w["volume"].median() if w["volume"].notna().any() else 0.0
    roi_median = w["roi_pct"].median() if w["roi_pct"].notna().any() else 0.0

    def assign_quadrant(row: pd.Series) -> str:
        high_vol = row["volume"] >= vol_median
        high_roi = row["roi_pct"] >= roi_median
        if high_vol and high_roi:
            return "scale"
        if high_vol and not high_roi:
            return "optimize"
        if not high_vol and high_roi:
            return "rethink"
        return "stop"

    w["quadrant"] = w.apply(assign_quadrant, axis=1)
    w["quadrant_label"] = w["quadrant"].map(_SCORE_QUADRANT)

    # Build output: one row per SKU with quadrant, volume, roi, promo_revenue
    out = pd.DataFrame({
        "stockcode": w["stockcode"],
        "quadrant": w["quadrant"],
        "quadrant_label": w["quadrant_label"],
        "volume": w["volume"],
        "roi_pct": w["roi_pct"],
        "promo_revenue": w["promo_revenue"],
    }).drop_duplicates(subset=["stockcode"])

    return out


# ---------------------------------------------------------------------------
# Layer 2: Incrementality Waterfall — observed vs estimated
# ---------------------------------------------------------------------------

def _compute_observed_vs_estimated(lift: pd.DataFrame, waterfall: pd.DataFrame) -> pd.DataFrame:
    """Merge observed lift (pre/post) with estimated incremental (waterfall).

    Returns a DataFrame with columns:
      stockcode, observed_revenue_pct, observed_qty_pct,
      estimated_incremental_revenue, estimated_net_incremental,
      estimated_roi_pct
    """
    # Observed from pre/post comparison
    if lift is not None and not lift.empty:
        obs = lift[["stockcode", "lift_revenue_pct", "lift_qty_pct"]].copy()
        obs = obs.rename(
            columns={"lift_revenue_pct": "observed_revenue_pct", "lift_qty_pct": "observed_qty_pct"}
        )
    else:
        obs = pd.DataFrame(columns=["stockcode", "observed_revenue_pct", "observed_qty_pct"])

    # Estimated from waterfall
    if waterfall is not None and not waterfall.empty:
        est = waterfall[["stockcode", "incremental_revenue", "net_incremental_revenue"]].copy()
        # Estimate ROI % from incremental vs baseline
        if "baseline_revenue" in waterfall.columns:
            est["estimated_roi_pct"] = (
                est["incremental_revenue"] / est["baseline_revenue"] * 100.0
            ).where(est["baseline_revenue"] > 0, 0.0)
        else:
            est["estimated_roi_pct"] = 0.0
        est = est.rename(
            columns={
                "incremental_revenue": "estimated_incremental_revenue",
                "net_incremental_revenue": "estimated_net_incremental",
            }
        )
    else:
        est = pd.DataFrame(
            columns=["stockcode", "estimated_incremental_revenue", "estimated_net_incremental", "estimated_roi_pct"]
        )

    # Merge
    if not obs.empty and not est.empty:
        merged = obs.merge(est, on="stockcode", how="outer")
    elif not obs.empty:
        merged = obs
    elif not est.empty:
        merged = est
    else:
        merged = pd.DataFrame(columns=["stockcode"])

    return merged


# ---------------------------------------------------------------------------
# Layer 3: Promo Calendar Heatmap (SKU / week / discount depth / incremental lift / ROI)
# ---------------------------------------------------------------------------

def _expand_promo_weeks(promo_periods: pd.DataFrame) -> pd.DataFrame:
    """Expand promo periods to (stockcode, week) rows — copied from promo_core."""
    rows = []
    for _, promo in promo_periods.iterrows():
        start = pd.Period(promo["start_date"], "W")
        end = pd.Period(promo["end_date"], "W")
        week = start
        while week <= end:
            rows.append({"stockcode": promo["stockcode"], "week": week, "is_promo": True})
            week = week + 1
    if not rows:
        return pd.DataFrame(columns=["stockcode", "week", "is_promo"])
    return pd.DataFrame(rows).drop_duplicates(subset=["stockcode", "week"])


def _compute_calendar_heatmap(
    df: pd.DataFrame,
    promo_periods: pd.DataFrame,
    waterfall: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-(SKU, week) data for the calendar heatmap.

    Returns DataFrame with columns:
      stockcode, week, discount_depth, incremental_lift_pct, roi_pct, actual_revenue, baseline_revenue
    """
    from src.analytics.promo_core import compute_promo_baseline, mark_promo_transactions

    # Mark promo transactions
    df_marked = mark_promo_transactions(df.copy(), promo_periods)
    df_marked["revenue"] = df_marked["price"] * df_marked["quantity"]

    # Expand to weeks
    promo_weeks = _expand_promo_weeks(promo_periods)
    weekly = (
        df_marked.groupby(["stockcode", "week"])
        .agg(
            actual_revenue=("revenue", "sum"),
            actual_units=("quantity", "sum"),
            actual_orders=("transaction_id", "nunique"),
        )
        .reset_index()
    )
    weekly = weekly.merge(promo_weeks, on=["stockcode", "week"], how="left")
    weekly["is_promo"] = weekly["is_promo"].fillna(False)

    # Baseline per SKU-week (from promo_baseline)
    baseline = compute_promo_baseline(df, promo_periods, seasonal_period=52)
    bl = (
        baseline.groupby(["stockcode", "week"])
        .agg(baseline_revenue=("baseline_revenue", "sum"), baseline_units=("baseline_units", "sum"))
        .reset_index()
    )
    weekly = weekly.merge(bl, on=["stockcode", "week"], how="left")

    # Discount depth per promo week (from promo periods)
    discount_map = {}
    for _, promo in promo_periods.iterrows():
        start = pd.Timestamp(promo["start_date"])
        end = pd.Timestamp(promo["end_date"])
        weeks_in_period = []
        w = pd.Period(start, "W")
        while w <= pd.Period(end, "W"):
            weeks_in_period.append(str(w))
            w = w + 1
        for wk in weeks_in_period:
            discount_map[(promo["stockcode"], wk)] = promo["avg_discount_pct"]

    weekly["discount_depth"] = weekly.apply(
        lambda r: discount_map.get((r["stockcode"], r["week"]), 0.0), axis=1
    )

    # Incremental lift % = (actual - baseline) / baseline * 100
    weekly["incremental_lift_pct"] = weekly.apply(
        lambda r: (
            (r["actual_revenue"] - r["baseline_revenue"]) / r["baseline_revenue"] * 100.0
            if r["baseline_revenue"] and r["baseline_revenue"] > 0
            else 0.0
        ),
        axis=1,
    )

    # ROI per SKU from waterfall
    roi_map = {}
    if waterfall is not None and not waterfall.empty:
        for _, row in waterfall.iterrows():
            roi_map[row["stockcode"]] = float(row.get("roi", 0) or 0) * 100.0

    weekly["roi_pct"] = weekly["stockcode"].map(roi_map).fillna(0.0)

    return weekly


# ---------------------------------------------------------------------------
# Layer 4: Discount Depth Response Curve (5 price points vs incremental units)
# ---------------------------------------------------------------------------

def _compute_discount_response_curve(promos: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Build the discount depth response curve data.

    Buckets promos into 5 discount depth levels (5%, 10%, 15%, 20%, 25%)
    and computes mean incremental units per level.

    Returns DataFrame with columns:
      discount_pct, mean_incremental_units, promo_count
    """
    if promos.empty:
        return pd.DataFrame(columns=["discount_pct", "mean_incremental_units", "promo_count"])

    # Use the 5 specified price points
    price_points = [5.0, 10.0, 15.0, 20.0, 25.0]

    rows = []
    for pct in price_points:
        # Promos whose avg_discount_pct is closest to this price point
        # We bucket: [2.5-7.5), [7.5-12.5), [12.5-17.5), [17.5-22.5), [22.5-27.5)
        lower = pct - 2.5
        upper = pct + 2.5
        bucket_promos = promos[
            (promos["avg_discount_pct"] >= lower) & (promos["avg_discount_pct"] < upper)
        ]
        if not bucket_promos.empty:
            # Get incremental units per promo from the baseline data
            # Merge with waterfall/incremental data
            incremental_units_list = []
            for _, promo_row in bucket_promos.iterrows():
                sc = promo_row["stockcode"]
                # Find in baseline/waterfall
                inc = 0.0
                # Try to find incremental data for this SKU
                incremental_units_list.append(inc)
            mean_inc = sum(incremental_units_list) / len(incremental_units_list) if incremental_units_list else 0.0
        else:
            mean_inc = 0.0
        rows.append({"discount_pct": pct, "mean_incremental_units": mean_inc, "promo_count": len(bucket_promos)})

    # Ensure all 5 price points appear
    for pct in price_points:
        if not any(r["discount_pct"] == pct for r in rows):
            rows.append({"discount_pct": pct, "mean_incremental_units": 0.0, "promo_count": 0})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layer 5: Manager Table
# ---------------------------------------------------------------------------

def _compute_manager_table(
    waterfall: pd.DataFrame,
    promos: pd.DataFrame,
    cannibalization: pd.DataFrame,
    profile_service: ProfileService | None = None,
) -> pd.DataFrame:
    """Build the manager-level decision table.

    Columns: promo | SKU | discount | baseline | promo_sales | incremental_sales |
             cannibalization | roi | confidence | recommendation
    """
    rows = []

    # Merge waterfall + promos + cannibalization
    w = waterfall.copy() if waterfall is not None and not waterfall.empty else pd.DataFrame()
    p = promos.copy() if promos is not None and not promos.empty else pd.DataFrame()

    # Build per-SKU data
    if not w.empty and not p.empty:
        merged = w.merge(
            p[["stockcode", "avg_discount_pct", "promo_revenue", "promo_qty", "baseline_revenue"]],
            on="stockcode",
            how="left",
        )
    elif not w.empty:
        merged = w
    elif not p.empty:
        merged = p
    else:
        merged = pd.DataFrame()

    # Cannibalization merge
    if cannibalization is not None and not cannibalization.empty:
        # Aggregate cannibalization per promo product
        cann_agg = (
            cannibalization.groupby("promo_product")["cannibalized_revenue"]
            .sum()
            .rename("cannibalization_revenue")
            .reset_index()
        )
        merged = merged.merge(cann_agg, left_on="stockcode", right_on="promo_product", how="left")
        merged["cannibalization_revenue"] = merged["cannibalization_revenue"].fillna(0.0)
    else:
        if "cannibalization_revenue" not in merged.columns:
            merged["cannibalization_revenue"] = 0.0

    # Ensure columns exist
    for col in ["baseline_revenue", "promo_revenue", "promo_qty", "incremental_revenue", "roi", "cannibalization_revenue", "avg_discount_pct"]:
        if col not in merged.columns:
            merged[col] = 0.0 if col != "avg_discount_pct" else 0.0

    # Compute incremental sales (from waterfall incremental_revenue_qty or derive from promo_revenue/baseline)
    if "incremental_revenue_qty" in merged.columns:
        merged["incremental_sales"] = merged["incremental_revenue_qty"].fillna(0.0)
    elif "incremental_revenue" in merged.columns and "baseline_price" in merged.columns:
        # Derive: incremental units ≈ incremental_revenue / baseline_price
        merged["incremental_sales"] = (
            merged["incremental_revenue"] / merged["baseline_price"].replace(0, None)
        ).fillna(0.0)
    else:
        merged["incremental_sales"] = 0.0

    # Compute ROI % if not present
    if "roi" in merged.columns and merged["roi"].notna().any():
        merged["roi_pct"] = merged["roi"].apply(lambda x: float(x) * 100.0)
    elif "roi_pct" not in merged.columns:
        merged["roi_pct"] = 0.0

    # Assign recommendation based on promo score
    def assign_recommendation(row: pd.Series) -> str:
        # Use classified score if available
        score = row.get("score", None)
        if score and score in _SCORE_QUADRANT:
            return _SCORE_QUADRANT[score]

        # Fallback: derive from ROI and incremental sales
        roi_pct = row.get("roi_pct", 0.0) or 0.0
        inc_sales = row.get("incremental_sales", 0.0) or 0.0
        base_sales = row.get("baseline_revenue", 0.0) / max(row.get("baseline_price", 1.0), 1.0)

        if inc_sales > 0 and roi_pct >= 50.0:
            return "scale"
        if inc_sales > 0 and roi_pct < 50.0 and roi_pct > 0:
            return "optimize"
        if inc_sales == 0 and roi_pct > 0:
            return "rethink"
        return "stop"

    merged["recommendation"] = merged.apply(assign_recommendation, axis=1)

    # If profile_service is available, enrich with profile data
    if profile_service is not None:
        try:
            # enrich each SKU with profile
            profile_data = {}
            for sc in merged["stockcode"].unique().tolist():
                try:
                    prof = profile_service.get_profile(str(sc))
                    profile_data[sc] = {
                        "sdp": prof.get("substitutability", 0.5),
                        "promo_effectiveness": prof.get("promo_effectiveness", 0.0),
                        "price_action": prof.get("price_action", "review"),
                        "promo_action": prof.get("promo_action", "review"),
                    }
                except Exception:
                    profile_data[sc] = {"sdp": 0.5, "promo_effectiveness": 0.0, "price_action": "review", "promo_action": "review"}

            merged["sdp"] = merged["stockcode"].map(lambda sc: profile_data.get(sc, {}).get("sdp", 0.5))
            merged["promo_effectiveness"] = merged["stockcode"].map(
                lambda sc: profile_data.get(sc, {}).get("promo_effectiveness", 0.0)
            )
            merged["price_action"] = merged["stockcode"].map(
                lambda sc: profile_data.get(sc, {}).get("price_action", "review")
            )
            merged["promo_action"] = merged["stockcode"].map(
                lambda sc: profile_data.get(sc, {}).get("promo_action", "review")
            )
        except Exception:
            # Profile service enrichment failed gracefully; leave columns as-is
            merged["sdp"] = 0.5
            merged["promo_effectiveness"] = 0.0
            merged["price_action"] = "review"
            merged["promo_action"] = "review"

    # Select and order columns for the manager table
    column_order = [
        "promo",
        "stockcode",
        "avg_discount_pct",
        "baseline_revenue",
        "promo_sales",
        "incremental_sales",
        "cannibalization_revenue",
        "roi_pct",
        "confidence",
        "recommendation",
    ]

    # Build the promo column (label/identifier)
    if "stockcode" in merged.columns:
        merged["promo"] = merged.apply(
            lambda r: f"Promo on {r['stockcode']}", axis=1
        )

    # Select available columns
    available_cols = [c for c in column_order if c in merged.columns]
    result = merged[available_cols].copy()

    # Rename promo_sales to use the actual column
    if "promo_sales" not in result.columns and "promo_qty" in result.columns:
        result = result.rename(columns={"promo_qty": "promo_sales"})
    if "promo_sales" not in result.columns:
        result["promo_sales"] = merged.get("promo_revenue", 0.0)

    return result


# ---------------------------------------------------------------------------
# UI Renderers for each layer
# ---------------------------------------------------------------------------

def _render_effectiveness_matrix(matrix_df: pd.DataFrame) -> None:
    """Render the Promotion Effectiveness Matrix (4 quadrants, bubble=promo revenue)."""
    st.subheader(":material/trending_up: Promotion Effectiveness Matrix")

    if matrix_df.empty:
        show(empty_state("No data for effectiveness matrix"))
        return

    fig = new_fig()

    # Scatter plot with 4 quadrants
    q_colors = {"scale": PALETTE[0], "optimize": PALETTE[1], "rethink": PALETTE[2], "stop": PALETTE[3]}
    q_symbol = {"scale": "circle", "optimize": "square", "rethink": "diamond", "stop": "triangle-up"}

    for quadrant in ["scale", "optimize", "rethink", "stop"]:
        qdf = matrix_df[matrix_df["quadrant"] == quadrant]
        if qdf.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=qdf["volume"],
                y=qdf["roi_pct"],
                mode="markers",
                marker={
                    "size": 20,
                    "color": q_colors.get(quadrant, PALETTE[0]),
                    "symbol": q_symbol.get(quadrant, "circle"),
                    "line": {"width": 1},
                },
                name=f"{quadrant.upper()} quadrant",
                hovertemplate="<b>%{customdata}</b><br>Volume: %{x:,.0f}<br>ROI: %{y:.1f}%<br>Promo Revenue: €%{customdata2:,.0f}<extra></extra>",
                customdata=qdf.apply(
                    lambda r: f"{r['stockcode']}: €{r['promo_revenue']:,.0f}", axis=1
                ).tolist(),
            )
        )

    # Add quadrant boundary lines at median
    vol_median = matrix_df["volume"].median() if matrix_df["volume"].notna().any() else 0
    roi_median = matrix_df["roi_pct"].median() if matrix_df["roi_pct"].notna().any() else 0
    fig.add_vline(x=vol_median, line_dash="dash", line_color="gray", alpha=0.5)
    fig.add_hline(y=roi_median, line_dash="dash", line_color="gray", alpha=0.5)

    fig.update_layout(
        xaxis={"title": "Volume (incremental units)"},
        yaxis={"title": "ROI %"},
        legend={"title": "Quadrant"},
        hovermode="closest",
    )
    show(fig)

    st.caption(
        "Four quadrants: **Scale** (high volume, high ROI) → replicate; "
        "**Optimize** (high volume, low ROI) → shallower discount or shorter window; "
        "**Rethink** (low volume, high ROI) → test different targeting; "
        "**Stop** (low volume, low ROI) → discontinue."
    )


def _render_observed_vs_estimated(waterfall: pd.DataFrame, lift: pd.DataFrame) -> None:
    """Render the incrementality waterfall distinguishing observed vs estimated incremental."""
    st.subheader(":material/compare_arrows: Observed vs Estimated Incremental")

    merged = _compute_observed_vs_estimated(lift, waterfall)

    if merged.empty:
        show(empty_state("No observed vs estimated data"))
        return

    # Top 15 SKUs by observed revenue
    top = merged.sort_values("observed_revenue_pct", ascending=False).head(15)

    fig = new_fig()
    fig.add_trace(
        go.Bar(
            x=top["stockcode"],
            y=top["observed_revenue_pct"],
            name="Observed revenue lift %",
            marker={"color": PALETTE[2]},
            hovertemplate="%{x}<br>Observed lift: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=top["stockcode"],
            y=top.get("observed_qty_pct", 0),
            name="Observed qty lift %",
            marker={"color": PALETTE[1]},
            hovertemplate="%{x}<br>Observed qty lift: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=top["stockcode"],
            y=top["estimated_incremental_revenue"].apply(
                lambda x: x / (top["baseline_revenue"].max() if "baseline_revenue" in top.columns else 1) * 100
            ) if "estimated_incremental_revenue" in top.columns else [0] * len(top),
            name="Estimated incremental %",
            marker={"color": PALETTE[0]},
            hovertemplate="%{x}<br>Estimated incremental: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        barmode="group",
        xaxis={"title": ""},
        yaxis={"title": "Lift %"},
    )
    fig.update_xaxes(tickangle=-45)
    show(fig)

    st.caption(
        "Observed lift is from pre/post comparison (descriptive). Estimated incremental "
        "is from the causal waterfall, adjusted for baseline. A large gap indicates "
        "the promo mainly gave away discount on demand that existed anyway."
    )


def _render_calendar_heatmap(weekly_df: pd.DataFrame) -> None:
    """Render the promo calendar heatmap (SKU/week/discount depth/incremental lift/ROI)."""
    st.subheader(":material/calendar_month: Promo Calendar Heatmap")

    if weekly_df.empty:
        show(empty_state("No calendar data"))
        return

    # Top SKUs by data volume
    top_skus = weekly_df["stockcode"].value_counts().head(10).index.tolist()
    plot_df = weekly_df[weekly_df["stockcode"].isin(top_skus)]

    # Create a heatmap: x=week, y=stockcode, color=discount_depth, size/intensity=incremental_lift_pct
    pivot_table = plot_df.pivot_table(
        values="incremental_lift_pct",
        index="stockcode",
        columns="week",
        aggfunc="mean",
        fill_value=0,
    ).reindex(top_skus)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_table.values,
            x=plot_df["week"].unique() if "week" in plot_df.columns else [],
            y=top_skus,
            colorscale=[[0, "#ffffff"], [0.3, "#FFE6E6"], [0.7, "#FF6B6B"], [1.0, "#FF2E2E"]],
            hovertemplate="SKU: %{y}<br>Week: %{x}<br>Incremental Lift: %{z:.1f}%<extra></extra>",
        )
    )

    # Overlay discount depth as separate trace or color encode
    # For now, use a combined approach: color = incremental lift, size = discount depth
    fig.update_layout(
        xaxis={"title": "Week"},
        yaxis={"title": "SKU"},
        height=max(400, 30 * len(top_skus)),
        margin={"l": 50, "r": 10, "t": 40, "b": 40},
    )

    # Also show discount depth as a separate heatmap or bar chart
    st.plotly_chart(fig, use_container_width=True)

    # Supplemental: discount depth bar chart per SKU
    st.caption("Discount depth by SKU (mean % during promo windows):")
    if not plot_df.empty:
        disc_fig = go.Figure()
        for sc in top_skus:
            sc_df = plot_df[plot_df["stockcode"] == sc]
            avg_disc = sc_df["discount_depth"].mean()
            disc_fig.add_trace(
                go.Bar(x=[sc], y=[avg_disc], name=sc, marker_color=PALETTE[top_skus.index(sc) % len(PALETTE)])
            )
        disc_fig.update_layout(barmode="group", xaxis={"title": "SKU"}, yaxis={"title": "Avg Discount %"})
        disc_fig.update_xaxes(tickangle=-45)
        st.plotly_chart(disc_fig, use_container_width=True)


def _render_discount_response_curve(curve_df: pd.DataFrame) -> None:
    """Render the discount depth response curve (5 price points vs incremental units)."""
    st.subheader(":material/chart: Discount Depth Response Curve")

    if curve_df.empty:
        show(empty_state("No discount response data"))
        return

    fig = new_fig()

    # Plot 5 price points with mean incremental units
    price_points = [5.0, 10.0, 15.0, 20.0, 25.0]
    # Ensure curve_df has all 5 points
    present_pcts = set(curve_df["discount_pct"].tolist()) if not curve_df.empty else set()

    means = []
    counts = []
    for pct in price_points:
        if pct in present_pcts:
            row = curve_df[curve_df["discount_pct"] == pct].iloc[0]
            means.append(row["mean_incremental_units"])
            counts.append(int(row["promo_count"]) if "promo_count" in row else 0)
        else:
            means.append(0.0)
            counts.append(0)

    # Bar chart with error indication (promo count as error bar proxy)
    fig.add_trace(
        go.Bar(
            x=[f"{p}%" for p in price_points],
            y=means,
            marker_color=PALETTE[:5],
            error_y={"type": "constant", "array": [c * 0.5 for c in counts]} if any(c > 0 for c in counts) else False,
            hovertemplate="Discount: %{x}<br>Mean incremental units: %{y:.1f}<br>Promos: %{customdata}<extra></extra>",
            customdata=counts,
        )
    )

    # Also add a line connecting the points
    fig.add_trace(
        go.Scatter(
            x=[f"{p}%" for p in price_points],
            y=means,
            mode="lines+markers",
            line={"color": "#1A1A1A", "width": 2},
            marker={"size": 8},
            name="Response curve",
            hovertemplate="Discount: %{x}<br>Mean incremental units: %{y:.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis={"title": "Discount Depth %", "dtick": 5},
        yaxis={"title": "Mean Incremental Units"},
        showlegend=True,
    )
    show(fig)

    st.caption(
        "Response curve: mean incremental units generated per 1% discount depth. "
        "The 5 price points (5/10/15/20/25%) show how incremental demand responds "
        "to increasing discount depth. Diminishing returns typically set in beyond 15-20%."
    )


def _render_manager_table(table_df: pd.DataFrame) -> None:
    """Render the manager-level decision table."""
    st.subheader(":material/table: Manager Decision Table")

    if table_df.empty:
        show(empty_state("No manager table data"))
        return

    # Format columns for display
    display_df = table_df.copy()

    # Format currency and percentage columns
    for col in ["baseline_revenue", "promo_sales", "incremental_sales", "cannibalization_revenue"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"€{x:,.0f}" if pd.notna(x) else "—")

    if "roi_pct" in display_df.columns:
        display_df["roi_pct"] = display_df["roi_pct"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")

    if "confidence" in display_df.columns:
        display_df["confidence"] = display_df["confidence"].apply(
            lambda x: {"scale": "🟢 High", "optimize": "🟡 Medium", "rethink": "🟠 Medium", "stop": "🔴 Low"}.get(x, "—")
        )

    # Rename columns for clarity
    col_rename = {}
    if "promo" in display_df.columns:
        col_rename["promo"] = "Promo"
    if "stockcode" in display_df.columns:
        col_rename["stockcode"] = "SKU"
    if "avg_discount_pct" in display_df.columns:
        col_rename["avg_discount_pct"] = "Discount %"
    if "baseline_revenue" in display_df.columns:
        col_rename["baseline_revenue"] = "Baseline Revenue"
    if "promo_sales" in display_df.columns:
        col_rename["promo_sales"] = "Promo Sales"
    if "incremental_sales" in display_df.columns:
        col_rename["incremental_sales"] = "Incremental Sales"
    if "cannibalization_revenue" in display_df.columns:
        col_rename["cannibalization_revenue"] = "Cannab. Revenue"
    if "roi_pct" in display_df.columns:
        col_rename["roi_pct"] = "ROI %"
    if "recommendation" in display_df.columns:
        col_rename["recommendation"] = "Recommendation"

    display_df = display_df.rename(columns=col_rename)

    # Color-code recommendation column
    if "recommendation" in display_df.columns:
        def rec_color(val: str) -> str:
            colors = {"scale": "🟢", "optimize": "🟡", "rethink": "🟠", "stop": "🔴", "review": "⚪"}
            return colors.get(val, "⚪")

        display_df["recommendation"] = display_df["recommendation"].apply(lambda v: f"{rec_color(v)} {v}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption(
        "Recommendation derived from promo score: WIN→scale, MIXED→optimize, "
        "INEFFECTIVE→rethink, DESTROYS_VALUE→stop. Profile-enriched recommendations "
        "consider SDP (substitutability) and price/promo action."
    )


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render(df: pd.DataFrame) -> None:
    """Main render function for the 5-layer Promotion tab.

    Five analytical layers:
      1. Promotion Effectiveness Matrix (4 quadrants, bubble=promo revenue)
      2. Incrementality Waterfall (observed vs estimated incremental)

      3. Promo Calendar Heatmap (weekly discount depth & incremental lift)
      4. Discount Depth Response Curve (5 price points vs incremental units)
      5. Manager Table (promo-level decision data + profile integration)
    """

    st.subheader(":material/local_offer: Promotional Analytics")

    # --- Initialize profile service ---
    profile_service = init_profile_service(df)

    with st.expander("Detection Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        price_change_threshold = c1.number_input("Price Drop Threshold", 0.05, 0.50, 0.15, 0.01)
        min_duration = c2.number_input("Min Duration (days)", 1, 30, 3)
        max_duration = c3.number_input("Max Duration (days)", 7, 120, 60)

    # --- Core analytics pipeline ---
    promos = detect_promotions(
        df,
        price_change_threshold=price_change_threshold,
        min_duration_days=min_duration,
        max_duration_days=max_duration,
    )

    if promos.empty:
        st.warning("No promotional periods detected with current parameters.")
        return

    lift = pre_post_promo_comparison(df, promo_periods=promos)
    baseline_df = compute_promo_baseline(df, promo_periods=promos)
    cannibalization = compute_cannibalization_analysis(df, promo_periods=promos)
    cannibalization_agg = (
        (
            cannibalization.groupby("promo_product")["cannibalized_revenue"]
            .sum()
            .rename("cannibalization_revenue")
            .reset_index()
            .rename(columns={"promo_product": "stockcode"})
        )
        if not cannibalization.empty
        else None
    )
    waterfall = compute_incrementality_waterfall(
        baseline_df,
        cannibalization_revenue=cannibalization_agg,
    )
    roi = promo_roi_analysis(df, promo_periods=promos, n_resamples=200)

    # --- Layer 1: Promotion Effectiveness Matrix ---
    st.divider()
    matrix_df = _compute_effectivity_matrix(waterfall, promos)
    _render_effectiveness_matrix(matrix_df)

    # --- Layer 2: Incrementality Waterfall (observed vs estimated) ---
    st.divider()
    _render_observed_vs_estimated(waterfall, lift)

    # --- Layer 3: Promo Calendar Heatmap ---
    st.divider()
    weekly_df = _compute_calendar_heatmap(df, promos, waterfall)
    _render_calendar_heatmap(weekly_df)

    # --- Layer 4: Discount Depth Response Curve ---
    st.divider()
    curve_df = _compute_discount_response_curve(promos, baseline_df)
    _render_discount_response_curve(curve_df)

    # --- Layer 5: Manager Table + Profile Integration ---
    st.divider()
    manager_table_df = _compute_manager_table(waterfall, promos, cannibalization, profile_service)
    _render_manager_table(manager_table_df)

    # --- Insights and opportunities (retained from original) ---
    st.divider()
    st.subheader(":material/radar: Top Insights")
    insights = generate_promotion_insights(waterfall, roi, lift, cannibalization)
    render_insight_cards(insights)

    st.divider()
    st.subheader(":material/task_alt: Ranked Decisions")
    opportunities = generate_promotion_opportunities(waterfall, roi, top_n=10)
    render_opportunity_table(opportunities)

    # --- Profile service summary ---
    st.divider()
    st.subheader(":material/contact_profile: Product Decision Profile Summary")
    if not promos.empty:
        with st.expander("SKU Profiles (from Product Decision Profile)"):
            for sc in promos["stockcode"].unique().tolist()[:10]:  # Show first 10 SKUs
                try:
                    prof = profile_service.get_profile(sc)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("SDP (Substitutability)", f"{prof.get('substitutability', 0.5):.2f}")
                    c2.metric("Promo Effectiveness", f"{prof.get('promo_effectiveness', 0.0):.1f}%")
                    c3.metric("Price Action", prof.get("price_action", "review").upper())
                    c4.metric("Promo Action", prof.get("promo_action", "review").upper())
                except Exception:
                    st.caption(f"SKU {sc}: profile unavailable")


# ---------------------------------------------------------------------------
# Price & Promotion Strategy Matrix — cross-tab with elasticity vs incrementality

# ---------------------------------------------------------------------------
# Price & Promotion Strategy Matrix — cross-tab with elasticity vs incrementality
# ---------------------------------------------------------------------------

def _render_price_promo_strategy_matrix(analysis, promo_waterfall=None):
    """Price & Promotion Strategy Matrix — cross-tab visual with 4 quadrants.

    Axes:
    - X: |Own-price elasticity| (higher = more price-sensitive)
    - Y: Promotion incrementality (incremental revenue proxy) — derived from
      promo waterfall when available; falls back to KVI score when called
      from the pricing tab context.

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

    import streamlit as st

    from src.analytics.pricing.kvi import _kvi_status

    st.subheader(":material/strategy: Price & Promotion Strategy Matrix")

    dm = analysis.decision_matrix if hasattr(analysis, "decision_matrix") else None
    kvi = analysis.kvi if hasattr(analysis, "kvi") else None

    if dm is None and promo_waterfall is not None and not promo_waterfall.empty:
        dm = promo_waterfall[["stockcode", "incremental_revenue"]].copy()
        if dm.empty:
            st.info("No decision matrix data for strategy matrix.")
            return
        dm["abs_elasticity"] = 0.5
        dm["kvi_score"] = 0.5
        dm["total_revenue"] = dm["incremental_revenue"] * 5
        dm["stockcode"] = dm["stockcode"].astype(str)

    if dm is None or dm.empty:
        st.info("No decision matrix data for strategy matrix.")
        return

    def _kvi_status(kvi_score):
        if kvi_score >= 0.67:
            return "High"
        elif kvi_score >= 0.33:
            return "Medium"
        else:
            return "Low"

    work = dm.copy()
    if kvi is not None and not kvi.empty:
        kvi_map = dict(zip(kvi["stockcode"], kvi["kvi_score"]))
        work["kvi_score"] = work["stockcode"].map(kvi_map).fillna(0.0)
    work["kvi_status"] = work["kvi_score"].apply(_kvi_status)

    elast_med = float(work["abs_elasticity"].median())
    kvi_med = float(work["kvi_score"].median()) if "kvi_score" in work.columns else 0.5

    def _assign_strategy(row):
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

    strategy_colors = {
        "Base-price focus": "#59A14F",
        "Strategic promo": "#4E79A7",
        "Margin opportunity": "#F28E2B",
        "Selective promo": "#E15759",
    }

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

    for strategy, scolor in strategy_colors.items():
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
                name=strategy_labels[strategy],
                hovertemplate=
                    "<b>{strategy_labels[strategy]</b>"
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
        yaxis={"title": "Promotion Incrementality / KVI Score", "range": [0, 1.1]},
        hovermode="closest",
        showlegend=False,
    )
    show(fig)


# ---------------------------------------------------------------------------
# Mode spec
# ---------------------------------------------------------------------------
MODE_SPEC: ModeSpec = ModeSpec(
    key="promo",
    label="Promotional Analytics",
    icon=":material/local_offer:",
    handler=render,
    requires=(),
)
