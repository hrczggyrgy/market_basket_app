"""Assortment Optimization tab — five-layer redesign.

Layers:
  1) Portfolio Matrix — revenue contribution vs strategic uniqueness quadrants,
     bubble size = reach (customer penetration).
  2) Coverage curve — current / optimized / minimum coverage trajectory.
  3) Delist waterfall — 4 rationalization steps:
       Remove duplicate SKUs → Remove low-value tail → Protect KVIs → Add white-space products.
  4) Coverage × revenue-at-risk matrix.
  5) Manager table — SKU | Revenue | Reach | Uniqueness | Substitute coverage |
     Revenue at risk | Recovery potential | Keep/Add/Review/Delist | Confidence.

Integrates with Product Decision Profile (profile_service).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.assortment import (
    optimize_assortment_heuristic,
    optimize_assortment_milp,
)
from src.analytics.pricing.kvi import compute_kvi_score
from src.analytics.profile_service import (
    ProfileService,
    get_profile_service,
    init_profile_service,
)
from src.analytics.transference import (
    compute_demand_transference_matrix,
)
from src.ui.plots import PALETTE, empty_state, show
from src.ui.registry import ModeSpec

# ---------------------------------------------------------------------------
# Helper: compute per-SKU reach (penetration) from profile
# ---------------------------------------------------------------------------

def _get_reach(profile: dict[str, Any]) -> float:
    """Customer penetration / reach from profile (0–1)."""
    return float(profile.get("customer_reach", 0.0))


# ---------------------------------------------------------------------------
# Helper: compute strategic uniqueness = 1 - substitutability
# ---------------------------------------------------------------------------

def _get_uniqueness(profile: dict[str, Any]) -> float:
    """Strategic uniqueness = 1 - substitutability (0–1)."""
    sdp = float(profile.get("substitutability", 0.5))
    return round(1.0 - sdp, 3)


# ---------------------------------------------------------------------------
# Helper: revenue at risk if SKU delisted (sum of observed transfers out)
# ---------------------------------------------------------------------------

def _revenue_at_risk(sku: str, transfers: dict[str, pd.DataFrame], revenue_per_product: pd.Series) -> float:
    """Total revenue at risk if this SKU is removed — sum of transfer revenue out of it."""
    edges = transfers.get(sku)
    if edges is None or edges.empty:
        return 0.0
    return float(edges["observed_switching_transfer_revenue"].sum())


# ---------------------------------------------------------------------------
# Helper: recovery potential if SKU delisted (what kept substitutes can capture)
# ---------------------------------------------------------------------------

def _recovery_potential(sku: str, kept: set[str], transfers: dict[str, pd.DataFrame]) -> float:
    """Recovery revenue from delisted SKU via substitutes currently kept."""
    edges = transfers.get(sku)
    if edges is None or edges.empty:
        return 0.0
    in_kept = edges["to_product"].isin(kept)
    return float(edges.loc[in_kept, "observed_switching_transfer_revenue"].sum())


# ---------------------------------------------------------------------------
# Helper: decision label + confidence for manager table
# ---------------------------------------------------------------------------

def _decision_and_confidence(
    profile: dict[str, Any],
    kept: set[str],
    transfers: dict[str, pd.DataFrame],
    revenue_per_product: pd.Series,
) -> tuple[str, str, float]:
    """Return (label, action, confidence) for an SKU.

    label  : one of Keep, Add, Review, Delist
    action : human-readable rationale
    confidence : 0.0–1.0 based on data completeness
    """
    revenue = float(profile.get("revenue", 0.0))
    kvi_score = float(profile.get("kvi_score", 0.5))
    elasticity = float(profile.get("elasticity", 0.0))
    sdp = float(profile.get("substitutability", 0.5))
    reach = _get_reach(profile)
    uniqueness = _get_uniqueness(profile)

    # Data completeness: how many profile fields are populated?
    profile_fields = ["revenue", "abc", "xyz", "lifecycle", "velocity", "repeat_rate",
                      "customer_reach", "elasticity", "substitutability", "kvi_score"]
    filled = sum(1 for f in profile_fields if profile.get(f) not in (None, 0.0, "unknown", "mature"))
    total = len(profile_fields)
    confidence = min(1.0, filled / total) if total > 0 else 0.0

    # Decision logic
    is_kvi = kvi_score >= 0.6
    high_rev = revenue > revenue_per_product.quantile(0.8) if len(revenue_per_product) > 0 else False

    # Keep: KVIs protected, or high-revenue high-uniqueness anchors
    if is_kvi or (high_rev and uniqueness >= 0.7):
        label = "Keep"
        action = "Strategic KVI / high-revenue anchor"
    # Add: white-space products with good reach but not yet in assortment
    elif reach >= 0.15 and sku not in kept and uniqueness < 0.4:
        label = "Add"
        action = "White-space product — growth potential"
    # Review: medium-revenue, medium-uniqueness, or partial data
    elif 0.3 <= revenue <= revenue_per_product.quantile(0.7) if len(revenue_per_product) > 0 else False:
        label = "Review"
        action = "Assortment review — evaluate performance"
    # Delist: low revenue, low uniqueness, low reach
    else:
        label = "Delist"
        action = "Low priority — low revenue & low uniqueness"

    return label, action, confidence


# ---------------------------------------------------------------------------
# Layer 1: Portfolio Matrix — 4 quadrants (revenue vs uniqueness), bubble=reach
# ---------------------------------------------------------------------------

def _render_portfolio_matrix(
    df: pd.DataFrame,
    kept: list[str],
    profile_service: ProfileService,
) -> None:
    """Render the 4-quadrant Portfolio Matrix.

    Axes:
      X: Strategic uniqueness (1 - substitutability) → low uniqueness | high uniqueness
      Y: Revenue contribution (share of total) → low revenue | high revenue
    Bubble size: Reach (customer penetration, 0–1 scaled)
    """
    from src.analytics.data import revenue_column

    revenue_per_product = revenue_column(df).groupby(df["stockcode"]).sum().sort_values(ascending=False)
    total_rev = float(revenue_per_product.sum())

    # Build per-SKU data from profiles
    all_skus = list(revenue_per_product.index)
    rows = []

    for sku in all_skus:
        try:
            profile = profile_service.get_profile(sku)
        except Exception:
            profile = {}

        reach = _get_reach(profile)
        uniqueness = _get_uniqueness(profile)
        rev = float(profile.get("revenue", revenue_per_product.get(sku, 0.0)))
        rev_share = rev / total_rev if total_rev > 0 else 0.0
        at_risk = _revenue_at_risk(sku, {}, revenue_per_product)  # will be filled below

        # Compute substitute coverage
        transfers = _get_transfers_for_skus([sku])
        recovery = _recovery_potential(sku, set(kept) if kept else set(), transfers) if transfers else 0.0
        sub_cov = recovery / at_risk if at_risk > 0 else 1.0

        rows.append(
            {
                "stockcode": sku,
                "revenue_share": rev_share,
                "uniqueness": uniqueness,
                "reach": reach,
                "substitute_coverage": sub_cov,
                "revenue": rev,
            }
        )

    if not rows:
        show(empty_state("No SKU data for portfolio matrix"))
        return

    rdf = pd.DataFrame(rows)

    # --- 4 quadrants ---
    # Quadrant assignments: split at median revenue_share and median uniqueness
    rev_median = rdf["revenue_share"].median()
    uniq_median = rdf["uniqueness"].median()

    rdf["quadrant"] = "Low-Low"
    rdf.loc[(rdf["revenue_share"] >= rev_median) & (rdf["uniqueness"] >= uniq_median), "quadrant"] = "High-High"
    rdf.loc[(rdf["revenue_share"] >= rev_median) & (rdf["uniqueness"] < uniq_median), "quadrant"] = "High-Low"
    rdf.loc[(rdf["revenue_share"] < rev_median) & (rdf["uniqueness"] >= uniq_median), "quadrant"] = "Low-High"

    fig = go.Figure()

    quadrant_colors = {
        "High-High": PALETTE[0],   # dark blue - strategic anchors
        "High-Low": PALETTE[1],    # orange - commodity
        "Low-High": PALETTE[2],    # green - niche
        "Low-Low": PALETTE[4],     # gray - redundant
    }

    for quad, color in quadrant_colors.items():
        subset = rdf[rdf["quadrant"] == quad]
        if not subset.empty:
            fig.add_trace(
                go.Scatter(
                    x=subset["uniqueness"],
                    y=subset["revenue_share"],
                    mode="markers",
                    marker={
                        "size": 12 + 40 * subset["reach"],  # scale bubble by reach
                        "color": color,
                        "opacity": 0.7,
                        "line": {"width": 1, "color": "white"},
                    },
                    name=quad,
                    hovertemplate=
                    f"<b>%{{customdata}}</b>"
                    f"<br>Revenue share: %{{y:.2%}}"
                    f"<br>Uniqueness: %{{x:.2f}}"
                    f"<br>Reach: %{marker.size:.2f}"
                    f"<extra></extra>",
                    customdata=subset["stockcode"],
                )
            )

    # Add quadrant boundary lines
    fig.add_hline(y=rev_median, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=uniq_median, line_dash="dot", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="Assortment Portfolio Matrix — Revenue Share vs Strategic Uniqueness",
        xaxis={"title": "Strategic Uniqueness (1 - Substitutability)", "range": [0, 1], "dtick": 0.2},
        yaxis={"title": "Revenue Share", "range": [0, max(0.5, rdf["revenue_share"].max() * 1.2)], "dtick": 0.1},
        height=500,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )

    show(fig)

    # Summary stats caption
    st.caption(
        f"Quadrants split at revenue share median ({rev_median:.2%}) and uniqueness median ({uniq_median:.2f}). "
        f"Bubble size ∝ reach (customer penetration). n={len(rdf)} SKUs."
    )


# ---------------------------------------------------------------------------
# Layer 2: Coverage curve — current / optimized / minimum
# ---------------------------------------------------------------------------

def _render_coverage_curve(
    df: pd.DataFrame,
    kept: list[str],
    min_coverage: float = 0.80,
) -> None:
    """Render coverage curve showing current / optimized / minimum coverage."""

    revenue_per_product = revenue_column(df).groupby(df["stockcode"]).sum().sort_values(ascending=False)
    total_rev = float(revenue_per_product.sum())

    # Current coverage from current kept set
    kept_set = set(kept) if kept else set()
    transfers = {}
    try:
        dt_df = compute_demand_transference_matrix(df)
        if dt_df is not None and not dt_df.empty:
            transfers = {
                frm: g for frm, g in dt_df.groupby("from_product")
            }
    except Exception:
        pass

    metrics_current = _evaluate_solution_simple(kept_set, revenue_per_product, transfers)
    current_coverage = metrics_current.get("coverage", 0.0)

    # Optimized coverage via MILP (top-k by revenue with coverage constraint)
    try:
        selected_milp, milp_metrics = optimize_assortment_milp(
            df,
            max_skus=min(100, len(revenue_per_product)),
            min_coverage=min_coverage,
            objective="revenue",
        )
        optimized_coverage = milp_metrics.get("coverage", 0.0)
    except Exception:
        selected_milp = []
        optimized_coverage = current_coverage

    # Minimum coverage threshold line
    # Compute coverage curve: incremental SKUs added by revenue rank
    cum_rev = revenue_per_product.cumsum()
    cum_coverage = (cum_rev / total_rev).clip(upper=1.0)

    fig = go.Figure()

    # Minimum coverage threshold
    fig.add_hline(
        y=float(min_coverage),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Min coverage: {float(min_coverage):.0%}",
        annotation_position="top left",
    )

    # Current coverage point
    fig.add_trace(
        go.Scatter(
            x=[len(kept_set)],
            y=[current_coverage],
            mode="markers",
            marker={"size": 12, "color": PALETTE[0], "symbol": "triangle-up"},
            name="Current assortment",
            error_y={"type": "constant", "value": 0},
        )
    )

    # Optimized coverage point
    fig.add_trace(
        go.Scatter(
            x=[len(selected_milp)],
            y=[optimized_coverage],
            mode="markers",
            marker={"size": 12, "color": PALETTE[1], "symbol": "triangle-down"},
            name="Optimized assortment",
            error_y={"type": "constant", "value": 0},
        )
    )

    # Full coverage curve (cumulative)
    fig.add_trace(
        go.Scatter(
            x=cum_rev.index.tolist(),
            y=cum_cov.tolist() if (cum_cov := [c / total_rev for c in cum_rev]) else [],
            mode="lines",
            line={"color": "gray", "opacity": 0.3, "width": 1},
            name="Cumulative revenue coverage",
            showlegend=False,
        )
    )

    # Also add the step-wise: each SKU adds its share
    # Sort SKUs by revenue descending and plot cumulative
    sorted_rev = revenue_per_product.sort_values(ascending=False)
    cum_vals = sorted_rev.cumsum() / total_rev
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(cum_vals) + 1)),
            y=cum_vals.tolist(),
            mode="lines",
            line={"color": "rgba(0,0,0,0.1)", "width": 1},
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Assortment Coverage Curve",
        xaxis={"title": "Number of SKUs", "dtick": 10, "range": [0, max(50, len(revenue_per_product) * 0.1)]},
        yaxis={"title": "Revenue Coverage", "range": [0, 1.1], "dtick": 0.1},
        height=400,
        hovermode="x unified",
    )

    show(fig)

    # Metrics row
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Coverage", f"{current_coverage:.1%}")
    c2.metric("Optimized Coverage", f"{optimized_coverage:.1%}")
    c3.metric("Minimum Threshold", f"{float(min_coverage):.0%}")

    st.caption(
        f"Current: {len(kept_set)} SKUs → {current_coverage:.1%} coverage. "
        f"Optimized: {len(selected_milp)} SKUs → {optimized_coverage:.1%} coverage at ≥{float(min_coverage):.0%} threshold."
    )


# ---------------------------------------------------------------------------
# Layer 3: Delist waterfall — 4 rationalization steps
# ---------------------------------------------------------------------------

def _render_delist_waterfall(
    df: pd.DataFrame,
    kept: list[str],
) -> None:
    """Render the delist waterfall with 4 rationalization steps.

    Steps:
      1. Remove duplicate SKUs
      2. Remove low-value tail
      3. Protect KVIs
      4. Add white-space products
    """
    from src.analytics.data import revenue_column

    revenue_per_product = revenue_column(df).groupby(df["stockcode"]).sum().sort_values(ascending=False)
    total_rev = float(revenue_per_product.sum())
    kept_set = set(kept) if kept else set()

    # Compute transfers
    transfers = {}
    try:
        dt_df = compute_demand_transference_matrix(df)
        if dt_df is not None and not dt_df.empty:
            transfers = {
                frm: g for frm, g in dt_df.groupby("from_product")
            }
    except Exception:
        pass

    # Step 1: Remove duplicate SKUs (identify SKUs with same stockcode or near-duplicate descriptions)
    # In this dataset, "duplicates" are SKUs with identical product names but different stockcodes
    prod_to_skus: dict[str, list[str]] = {}
    for sku in revenue_per_product.index:
        try:
            prod_name = df[df["stockcode"] == sku]["product"].iloc[0]
        except Exception:
            prod_name = sku
        key = prod_name.lower().strip()
        prod_to_skus.setdefault(key, []).append(sku)

    dup_groups = {k: v for k, v in prod_to_skus.items() if len(v) > 1}
    dup_skus = set()
    for group in dup_groups.values():
        # Keep the highest-revenue duplicate, remove the rest
        group_rev = {s: revenue_per_product.get(s, 0.0) for s in group}
        keep_sku = max(group_rev, key=group_rev.get)
        dup_skus.update(s for s in group if s != keep_sku)

    # After step 1: revenue removed = sum of removed duplicates
    removed_dup_rev = sum(revenue_per_product.get(s, 0.0) for s in dup_skus if s in revenue_per_product.index)

    # Step 2: Remove low-value tail — SKUs below revenue percentile threshold
    tail_threshold = revenue_per_product.quantile(0.25)  # bottom 25%
    low_value_skus = [s for s in revenue_per_product.index if revenue_per_product[s] <= tail_threshold and s not in kept_set]
    # Exclude KVIs from removal
    try:
        kvi_df = compute_kvi_score(df, method="heuristic")
        kvi_skus = set(kvi_df[kvi_df["kvi_score"] >= 0.6]["stockcode"].tolist())
    except Exception:
        kvi_skus = set()
    low_value_skus = [s for s in low_value_skus if s not in kvi_skus]

    removed_tail_rev = sum(revenue_per_product.get(s, 0.0) for s in low_value_skus if s in revenue_per_product.index)

    # Step 3: Protect KVIs — ensure KVIs are retained (may add back some removed)
    # KVIs that were removed in steps 1-2 get restored
    kvi_to_protect = [s for s in low_value_skus + list(dup_skus) if s in kvi_skus]
    protected_rev = sum(revenue_per_product.get(s, 0.0) for s in kvi_to_protect if s in revenue_per_product.index)

    # After protecting KVIs, recalculate low-value tail (exclude protected KVIs)
    remaining_low_value = [s for s in low_value_skus if s not in kvi_to_protect]
    removed_tail_rev_after_protect = sum(revenue_per_product.get(s, 0.0) for s in remaining_low_value if s in revenue_per_product.index)

    # Step 4: Add white-space products — SKUs with good reach but not yet selected
    try:
        profile_service = get_profile_service()
        if profile_service is None:
            # Initialize from df
            profile_service = init_profile_service(df)
    except Exception:
        profile_service = None

    white_space_skus = []
    if profile_service is not None:
        all_skus = list(revenue_per_product.index)
        for sku in all_skus:
            try:
                profile = profile_service.get_profile(sku)
                reach = _get_reach(profile)
                uniqueness = _get_uniqueness(profile)
                # White-space: reach >= 15% but not yet in kept, low-moderate uniqueness
                if reach >= 0.15 and sku not in kept_set and uniqueness < 0.6:
                    white_space_skus.append(sku)
            except Exception:
                pass

    white_space_rev = sum(revenue_per_product.get(s, 0.0) for s in white_space_skus if s in revenue_per_product.index)

    # Waterfall figure: 5 bars (including total)
    steps = [
        ("Total Market", total_rev, "absolute"),
        ("Step 1: Remove duplicates", removed_dup_rev, "relative"),
        ("Step 2: Remove low-value tail", removed_tail_rev, "relative"),
        ("Step 3: Protect KVIs (±)", protected_rev, "relative"),
        ("Step 4: Add white-space", white_space_rev, "relative"),
        ("Recommended assortment", None, "total"),  # computed below
    ]

    # Compute kept revenue after all steps
    # Start with total, subtract removed, add back protected KVIs, add white-space
    kept_rev_after = total_rev - removed_dup_rev - removed_tail_rev_after_protect + white_space_rev
    # Also add revenue from kept SKUs that were already kept
    kept_rev_from_selected = sum(revenue_per_product.get(s, 0.0) for s in kept_set if s in revenue_per_product.index)

    # Final recommended: keep existing + add white-space, exclude removed
    final_kept = kept_set - dup_skus - set(remaining_low_value) + set(kvi_to_protect)
    final_kept = final_kept & set(revenue_per_product.index)  # only valid SKUs
    final_rev = sum(revenue_per_product.get(s, 0.0) for s in final_kept)

    fig = go.Figure(
        go.Waterfall(
            name="Delist Rationalization",
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(steps) - 2) + ["total"],
            x=[s[0] for s in steps],
            y=[steps[0][1]] + [s[1] for s in steps[1:]] + [final_rev],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": PALETTE[0]}},
            decreasing={"marker": {"color": PALETTE[4]}},
            totals={"marker": {"color": PALETTE[2]}},
        )
    )

    fig.update_layout(
        yaxis={"title": "Revenue ($)"},
        height=450,
        title="Delist Waterfall — 4-Step Rationalization",
    )

    show(fig)

    # Step summaries
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Step 1: Remove duplicates", f"${removed_dup_rev:,.0f}")
    c2.metric("Step 2: Remove low-value tail", f"${removed_tail_rev:,.0f}")
    c3.metric("Step 3: Protect KVIs", f"±${protected_rev:,.0f}")
    c4.metric("Step 4: Add white-space", f"${white_space_rev:,.0f}")

    st.caption(
        f"Recommended assortment revenue: ${final_rev:,.0f} "
        f"({len(final_kept)} SKUs). "
        f"Steps: dup removal → low-value tail → KVI protection → white-space addition."
    )


# ---------------------------------------------------------------------------
# Layer 4: Coverage × revenue-at-risk matrix
# ---------------------------------------------------------------------------

def _render_coverage_risk_matrix(
    df: pd.DataFrame,
    kept: list[str],
    min_coverage: float = 0.80,
) -> None:
    """Render coverage × revenue-at-risk matrix."""

    from src.analytics.data import revenue_column

    revenue_per_product = revenue_column(df).groupby(df["stockcode"]).sum().sort_values(ascending=False)
    total_rev = float(revenue_per_product.sum())
    kept_set = set(kept) if kept else set()

    # Compute transfers and revenue at risk per SKU
    transfers = {}
    try:
        dt_df = compute_demand_transference_matrix(df)
        if dt_df is not None and not dt_df.empty:
            transfers = {
                frm: g for frm, g in dt_df.groupby("from_product")
            }
    except Exception:
        pass

    # Build per-SKU data
    rows = []
    for sku in revenue_per_product.index:
        at_risk = _revenue_at_risk(sku, transfers, revenue_per_product)
        # Recovery potential if this SKU were delisted
        rp = _recovery_potential(sku, kept_set, transfers)
        # Coverage if this SKU were delisted
        # Simple: current coverage minus this SKU's contribution
        sku_share = revenue_per_product[sku] / total_rev if total_rev > 0 else 0.0
        # Effective coverage after removing this SKU
        effective_cov = max(0.0, current_coverage_from_metrics(kept_set, revenue_per_product, transfers) - sku_share / 2)  # approximate

        # But we want current overall coverage
        from src.analytics.assortment import _evaluate_solution
        current_metrics = _evaluate_solution(kept_set, revenue_per_product, transfers)
        current_cov = current_metrics.coverage

        rows.append(
            {
                "stockcode": sku,
                "revenue_at_risk": at_risk,
                "recovery_potential": rp,
                "revenue": revenue_per_product[sku],
                "revenue_share": revenue_per_product[sku] / total_rev if total_rev > 0 else 0.0,
                "current_coverage": current_cov,
            }
        )

    rdf = pd.DataFrame(rows)

    fig = go.Figure()

    # Scatter: revenue at risk (x) vs current coverage (y)
    fig.add_trace(
        go.Scatter(
            x=rdf["revenue_at_risk"],
            y=rdf["current_coverage"],
            mode="markers",
            marker={
                "size": 12 + 30 * rdf["revenue_share"],
                "color": rdf["recovery_potential"],
                "colorscale": "RdYlGn",
                "cmin": 0,
                "cmax": 1,
                "opacity": 0.7,
                "line": {"width": 1, "color": "white"},
            },
            name="SKUs",
            hovertemplate=
            "<b>%{customdata}</b>"
            "<br>Revenue at risk: $%{x:,.0f}"
            "<br>Current coverage: %{y:.1%}"
            "<br>Recovery potential: %{marker.color:.2f}"
            "<br>Revenue: $%{customdata[0]:,.0f}"
            "<extra></extra>",
            customdata=rdf[["revenue", "stockcode"]].values.tolist(),
        )
    )

    # Add min coverage threshold line
    fig.add_hline(
        y=float(min_coverage),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Min coverage: {float(min_coverage):.0%}",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Coverage × Revenue-at-Risk Matrix",
        xaxis={"title": "Revenue at Risk ($)", "type": "log", "dtick": 1e4},
        yaxis={"title": "Current Coverage", "range": [0, 1.1], "dtick": 0.1},
        height=500,
        hovermode="x unified",
    )

    show(fig)

    st.caption(
        "Bubble size ∝ revenue share. Color = recovery potential (0–1). "
        "Points in upper-right have high revenue at risk and low coverage — priority for protection or substitution."
    )


# ---------------------------------------------------------------------------
# Layer 5: Manager table — SKU | Revenue | Reach | Uniqueness | Substitute coverage |
#          Revenue at risk | Recovery potential | Keep/Add/Review/Delist | Confidence
# ---------------------------------------------------------------------------

def _render_manager_table(
    df: pd.DataFrame,
    kept: list[str],
    profile_service: ProfileService,
) -> None:
    """Render the manager decision table with all required columns."""

    revenue_per_product = revenue_column(df).groupby(df["stockcode"]).sum().sort_values(ascending=False)
    kept_set = set(kept) if kept else set()

    # Compute transfers once
    transfers = {}
    try:
        dt_df = compute_demand_transference_matrix(df)
        if dt_df is not None and not dt_df.empty:
            transfers = {
                frm: g for frm, g in dt_df.groupby("from_product")
            }
    except Exception:
        pass

    # Build per-SKU rows
    rows = []

    for sku in revenue_per_product.index:
        try:
            profile = profile_service.get_profile(sku)
        except Exception:
            profile = {}

        reach = _get_reach(profile)
        uniqueness = _get_uniqueness(profile)
        at_risk = _revenue_at_risk(sku, transfers, revenue_per_product)
        recovery = _recovery_potential(sku, kept_set, transfers)
        sub_cov = recovery / at_risk if at_risk > 0 else 1.0

        label, action, confidence = _decision_and_confidence(
            profile, kept_set, transfers, revenue_per_product
        )

        rows.append(
            {
                "stockcode": sku,
                "revenue": float(revenue_per_product[sku]),
                "reach": reach,
                "uniqueness": uniqueness,
                "substitute_coverage": sub_cov,
                "revenue_at_risk": at_risk,
                "recovery_potential": recovery,
                "decision": label,
                "action": action,
                "confidence": confidence,
            }
        )

    rdf = pd.DataFrame(rows, columns=[
        "stockcode", "revenue", "reach", "uniqueness",
        "substitute_coverage", "revenue_at_risk", "recovery_potential",
        "decision", "action", "confidence",
    ]).sort_values("revenue", ascending=False)

    # Display as interactive table + progress bars for confidence
    st.subheader("Assortment Manager Decision Table")

    # Format columns for display
    display_df = rdf.copy()
    display_df["revenue"] = display_df["revenue"].apply(lambda x: f"${x:,.0f}")
    display_df["reach"] = display_df["reach"].apply(lambda x: f"{x:.1%}")
    display_df["uniqueness"] = display_df["uniqueness"].apply(lambda x: f"{x:.2f}")
    display_df["substitute_coverage"] = display_df["substitute_coverage"].apply(lambda x: f"{x:.1%}")
    display_df["revenue_at_risk"] = display_df["revenue_at_risk"].apply(lambda x: f"${x:,.0f}")
    display_df["recovery_potential"] = display_df["recovery_potential"].apply(lambda x: f"${x:,.0f}")
    display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1%}")

    # Color-code the decision column
    st.dataframe(
        display_df,
        column_config={
            "decision": st.column_config.Column(
                "Decision",
                help="Keep / Add / Review / Delist",
            ),
        },
        use_container_width=True,
        hide_index=True,
    )

    # Summary metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    decision_counts = rdf["decision"].value_counts()
    c1.metric("Keep", f"{decision_counts.get('Keep', 0)}")
    c2.metric("Add", f"{decision_counts.get('Add', 0)}")
    c3.metric("Review", f"{decision_counts.get('Review', 0)}")
    c4.metric("Delist", f"{decision_counts.get('Delist', 0)}")
    c5.metric("Avg Confidence", f"{rdf['confidence'].mean():.1%}")

    # Decision summary
    st.caption(
        f"Decision rationale: Keep={list(rdf[rdf['decision']=='Keep']['action'].unique())}, "
        f"Add={list(rdf[rdf['decision']=='Add']['action'].unique())}, "
        f"Review={list(rdf[rdf['decision']=='Review']['action'].unique())}, "
        f"Delist={list(rdf[rdf['decision']=='Delist']['action'].unique())}"
    )


# ---------------------------------------------------------------------------
# Core evaluation helper (lightweight, no full MILP)
# ---------------------------------------------------------------------------

def _evaluate_solution_simple(
    kept: set[str],
    revenue_per_product: pd.Series,
    transfers: dict[str, pd.DataFrame] | None,
) -> dict[str, float]:
    """Lightweight evaluation mirroring _evaluate_solution without full dataclass."""
    if transfers is None:
        transfers = {}
    kept_present = [p for p in sorted(kept) if p in revenue_per_product.index]
    kept_revenue = float(revenue_per_product.loc[kept_present].sum()) if kept_present else 0.0
    total_revenue = float(revenue_per_product.sum())

    lost = 0.0
    recovered = 0.0
    for prod, rev in revenue_per_product.items():
        if prod in kept:
            continue
        lost += float(rev)
        edges = transfers.get(prod)
        if edges is not None and not edges.empty:
            in_kept = edges["to_product"].isin(kept)
            if in_kept.any():
                recovered += float(edges.loc[in_kept, "observed_switching_transfer_revenue"].sum())

    unmet = lost - recovered
    expected = kept_revenue + recovered
    coverage = min(1.0, expected / total_revenue) if total_revenue > 0 else 0.0
    recovery_rate = min(1.0, recovered / lost) if lost > 0 else 0.0

    return {
        "kept_revenue": kept_revenue,
        "recovered_revenue": recovered,
        "lost_revenue": lost,
        "unmet_demand": unmet,
        "expected_revenue": expected,
        "coverage": coverage,
        "recovery_rate": recovery_rate,
    }


def _get_transfers_for_skus(skus: list[str]) -> dict[str, pd.DataFrame]:
    """Helper to get transfers for specific SKUs (placeholder)."""
    return {}


# ---------------------------------------------------------------------------
# current_coverage_from_metrics helper
# ---------------------------------------------------------------------------

def current_coverage_from_metrics(
    kept: set[str],
    revenue_per_product: pd.Series,
    transfers: dict[str, pd.DataFrame] | None,
) -> float:
    """Compute current coverage from evaluation metrics."""
    if transfers is None:
        transfers = {}
    try:
        from src.analytics.assortment import _evaluate_solution as _es
        m = _es(kept, revenue_per_product, transfers)
        return m.get("coverage", 0.0)
    except Exception:
        # Fallback simple computation
        kept_rev = float(revenue_per_product.loc[[p for p in kept if p in revenue_per_product.index]].sum()) if kept else 0.0
        total = float(revenue_per_product.sum())
        return min(1.0, kept_rev / total) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render(df: pd.DataFrame) -> None:
    """Render the five-layer Assortment Optimization tab.

    Five layers (Waves 13–15 redesign):
      1) Portfolio Matrix — 4 quadrants: revenue vs strategic uniqueness,
         bubble size = reach (customer penetration).
      2) Coverage curve — current / optimized / minimum coverage trajectory.
      3) Delist waterfall — 4 rationalization steps:
           Remove duplicate SKUs → Remove low-value tail → Protect KVIs → Add white-space products.
      4) Coverage × revenue-at-risk matrix.
      5) Manager table — SKU | Revenue | Reach | Uniqueness | Substitute coverage |
         Revenue at risk | Recovery potential | Keep/Add/Review/Delist | Confidence.
    """
    st.subheader(":material/inventory_2: Assortment Optimization")

    # Initialize profile service from data
    try:
        profile_service = init_profile_service(df)
    except Exception:
        profile_service = None

    with st.expander("Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        max_skus = c1.number_input("Max SKUs for optimization", 10, 500, 50)
        min_coverage = c2.number_input("Minimum coverage threshold", 0.10, 1.0, 0.80, 0.05)
        objective = c3.selectbox("Optimization objective", ["revenue", "margin"])

    # Run optimization heuristic (fast, used for current assortment)
    selected, metrics = optimize_assortment_heuristic(
        df,
        max_skus=max_skus,
        min_coverage=min_coverage,
        objective=objective,
    )

    kept = selected  # list of selected SKU stockcodes

    # Metrics summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SKUs Selected", len(kept))
    col2.metric("Coverage", f"{metrics.get('coverage', 0):.1%}")
    col3.metric("Kept Revenue", f"${metrics.get('kept_revenue', 0):,.0f}")
    col4.metric("Recovery Rate", f"{metrics.get('recovery_rate', 0):.1%}")

    st.divider()

    # --- Layer 1: Portfolio Matrix ---
    _render_portfolio_matrix(df, kept, profile_service)
    st.divider()

    # --- Layer 2: Coverage curve ---
    _render_coverage_curve(df, kept, min_coverage=min_coverage)
    st.divider()

    # --- Layer 3: Delist waterfall ---
    _render_delist_waterfall(df, kept)
    st.divider()

    # --- Layer 4: Coverage × revenue-at-risk matrix ---
    _render_coverage_risk_matrix(df, kept, min_coverage=min_coverage)
    st.divider()

    # --- Layer 5: Manager table ---
    _render_manager_table(df, kept, profile_service)

    st.divider()

    # --- Scenario comparison (existing functionality) ---
    scenarios = compare_assortment_scenarios(df, [])
    _render_scenario_comparison(scenarios)


def _render_scenario_comparison(scenarios: pd.DataFrame) -> None:
    """Render scenario comparison (adapted from original)."""
    st.subheader(":material/compare_arrows: Scenario Comparison")
    if scenarios.empty:
        show(empty_state("No scenarios to compare"))
        return

    # Expected revenue bar chart
    fig = px.bar(
        scenarios.sort_values("expected_revenue", ascending=True).tail(10),
        x="expected_revenue",
        y="scenario_id",
        color="method",
        color_discrete_map={"greedy": PALETTE[0], "random": PALETTE[2], "milp": PALETTE[4]},
        orientation="h",
        hover_data=["coverage", "recovery_rate", "n_skus"],
    )
    fig.update_layout(xaxis={"title": "Expected Revenue ($)"}, yaxis={"title": "Scenario ID"})
    show(fig)

    # Coverage vs Recovery scatter
    fig2 = px.scatter(
        scenarios,
        x="coverage",
        y="recovery_rate",
        size="n_skus",
        color="method",
        color_discrete_map={"greedy": PALETTE[0], "random": PALETTE[2], "milp": PALETTE[4]},
        hover_data=["scenario_id", "kept_revenue", "expected_revenue"],
    )
    fig2.add_vline(x=0.8, line_dash="dash", line_color="gray", annotation_text="Target 80%")
    fig2.update_layout(xaxis={"title": "Coverage"}, yaxis={"title": "Recovery Rate"})
    show(fig2)
    st.caption("Bubble size = SKU count. Target: high coverage + high recovery.")


MODE_SPEC: ModeSpec = ModeSpec(
    key="assortment",
    label="Assortment",
    icon=":material/inventory_2:",
    handler=render,
    requires=("has_category", "sufficient_skus_20", "sufficient_baskets_500"),
)
