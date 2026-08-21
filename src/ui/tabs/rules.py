"""Association Rules tab — five-layer redesign.

Rule Opportunity Matrix (lift × revenue opportunity quadrants)
Top basket missions ranked by incremental basket value
Cross-sell opportunity matrix (anchor × add-on = expected incremental revenue)
Manager table (anchor | add-on | lift | support | revenue opportunity | evidence | action)
Product Decision Profile integration for rule-level data
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.profile_service import ProfileService, init_profile_service
from src.analytics.rules import (
    aggregate_rules_to_categories,
    bootstrap_lift_ci,
    filter_rules,
    flag_redundant_rules,
    generate_rules,
    rules_to_table,
    run_fpgrowth,
)
from src.ui.features import get_basket_matrix, get_product_lookup
from src.ui.plots import show
from src.ui.registry import ModeSpec


def _rule_label(antecedent: str, consequent: str) -> str:
    return f"{antecedent}  →  {consequent}"


def _compute_incremental_basket_value(
    row: pd.Series,
    revenue_by_product: dict[str, float],
    basket: pd.DataFrame,
) -> float:
    """Compute incremental basket value for a rule.

    Incremental basket value = (lift - 1) × antecedent_revenue_per_basket ×
    antecedent_penetration. This measures the actual revenue impact of the
    rule, not just the lift magnitude.
    """
    # Antecedent revenue per basket
    ante_rev = 0.0
    for sid in row["antecedents"]:
        if sid in revenue_by_product:
            ante_rev += revenue_by_product[sid]

    ante_count = len(row["antecedents"])
    if ante_count > 0:
        ante_rev_per_item = ante_rev / ante_count
    else:
        ante_rev_per_item = 0.0

    # Support = % of baskets containing antecedent
    ante_support = row["support"]

    # Incremental value: (lift - 1) × revenue × support
    lift_impact = max(row["lift"] - 1.0, 0.0)
    incremental = lift_impact * ante_rev_per_item * ante_support
    return round(incremental, 4)


def _quadrant_label(lift: float, support: float, lift_median: float, support_median: float) -> str:
    """Classify a rule into one of four opportunity quadrants."""
    high_lift = lift >= lift_median
    high_support = support >= support_median

    if high_lift and high_support:
        return "scale"
    elif high_lift and not high_support:
        return "optimize"
    elif not high_lift and high_support:
        return "rethink"
    else:
        return "stop"


def _render_rule_opportunity_matrix(
    rules: pd.DataFrame,
    revenue_by_product: dict[str, float],
    basket: pd.DataFrame,
) -> None:
    """Render the Rule Opportunity Matrix with 4 quadrants."""
    st.subheader(":material/quadrin: Rule Opportunity Matrix")

    if rules.empty:
        st.info("No rules to classify into opportunity quadrants.")
        return

    # Compute medians for quadrant splits
    lift_median = rules["lift"].median()
    # Use incremental basket value support proxy: support × revenue
    # Compute a support-like metric weighted by revenue
    rules_with_val = rules.copy()
    rules_with_val["incr_value"] = rules_with_val.apply(
        lambda r: _compute_incremental_basket_value(r, revenue_by_product, basket),
        axis=1,
    )
    support_median = rules_with_val["incr_value"].median()

    # Assign quadrants
    rules_with_val["quadrant"] = rules_with_val.apply(
        lambda r: _quadrant_label(
            float(r["lift"]), float(r["incr_value"]), float(lift_median), float(support_median)
        ),
        axis=1,
    )

    quadrants = {"scale": [], "optimize": [], "rethink": [], "stop": []}
    for _, row in rules_with_val.iterrows():
        q = row["quadrant"]
        if q in quadrants:
            quadrants[q].append(
                {
                    "rule": _rule_label(
                        str(row["antecedent"]), str(row["consequent"])
                    ),
                    "lift": float(row["lift"]),
                    "incr_value": float(row["incr_value"]),
                    "antecedent": str(row["antecedent"]),
                    "consequent": str(row["consequent"]),
                }
            )

    q_labels = {
        "scale": "High lift + High incremental value — Scale these rules.",
        "optimize": "High lift + Medium incremental value — Optimize for more impact.",
        "rethink": "Low lift + High incremental value — Rethink antecedent/consequent.",
        "stop": "Low lift + Low incremental value — Stop pursuing.",
    }

    for q_name in ["scale", "optimize", "rethink", "stop"]:
        q_data = quadrants[q_name]
        if not q_data:
            st.info(f"No rules in the **{q_name}** quadrant.")
            continue

        with st.expander(f"**{q_name.title()}** ({len(q_data)} rules)", expanded=q_name == "scale"):
            st.caption(q_labels[q_name])
            for r in q_data:
                st.write(
                    f"- **{r['rule']}** — Lift: {r['lift']:.2f}x, "
                    f"Incremental value: ${r['incr_value']:.4f}"
                )


def _render_top_basket_missions(
    rules: pd.DataFrame,
    revenue_by_product: dict[str, float],
    basket: pd.DataFrame,
    top_n: int = 20,
) -> None:
    """Render top basket missions ranked by incremental basket value."""
    st.subheader(":material/star: Top Basket Missions")

    if rules.empty:
        st.info("No rules to rank as basket missions.")
        return

    # Compute incremental basket value for each rule
    rules_with_val = rules.copy()
    rules_with_val["incr_value"] = rules_with_val.apply(
        lambda r: _compute_incremental_basket_value(r, revenue_by_product, basket),
        axis=1,
    )

    # Rank by incremental value descending
    ranked = rules_with_val.nlargest(top_n, "incr_value")

    # Build display table
    mission_rows = []
    for _, row in ranked.iterrows():
        ante = str(row["antecedent"])
        cons = str(row["consequent"])
        lift = float(row["lift"])
        support = float(row["support"])
        incr = float(row["incr_value"])
        mission_rows.append(
            {
                "Rank": len(mission_rows) + 1,
                "Rule": _rule_label(ante, cons),
                "Lift": f"{lift:.2f}x",
                "Support": f"{support:.2%}",
                "Incremental Value": f"${incr:.4f}",
                "Antecedent": ante,
                "Consequent": cons,
            }
        )

    if mission_rows:
        mission_df = pd.DataFrame(mission_rows)
        st.dataframe(mission_df, use_container_width=True, hide_index=True)
    else:
        st.info("No basket missions could be ranked.")


def _render_cross_sell_opportunity_matrix(
    rules: pd.DataFrame,
    df: pd.DataFrame,
    top_n: int = 20,
) -> None:
    """Render cross-sell opportunity matrix showing expected incremental revenue per anchor×add-on pair."""
    st.subheader(":material/add_cross: Cross-Sell Opportunity Matrix")

    if rules.empty:
        st.info("No rules available for cross-sell analysis.")
        return

    # Build basket matrix and revenue lookup
    get_basket_matrix(df)
    get_product_lookup(df)

    # Compute revenue by product
    revenue_by_product: dict[str, float] = {}
    if "price" in df.columns and "quantity" in df.columns:
        rev_series = df.groupby("stockcode").apply(
            lambda x: float((x["price"] * x["quantity"]).sum()), include_groups=False
        )
        revenue_by_product = {
            str(k): v for k, v in rev_series.items()
        }

    # Use cross_sell module with addon recommendations derived from rules
    # Build addon_df from rules: anchor → addon with lift/support
    addon_rows = []
    for _, rule in rules.head(100).iterrows():
        # Use consequents as add-ons for each antecedent
        for addon in rule["consequents"]:
            anchor_str = (
                str(rule["antecedents"]).strip("{}")
                .replace("frozenset", "")
                .replace("'", "")
                .replace(", ", "+")
                .replace(" ", "")
                or "unknown"
            )
            addon_rows.append(
                {
                    "anchor": anchor_str,
                    "addon": str(addon).strip("'"),
                    "lift": float(rule["lift"]),
                    "support": float(rule["support"]),
                }
            )

    if addon_rows:
        addon_df = pd.DataFrame(addon_rows)
        # Aggregate: for each unique anchor→addon pair, take max lift and total support
        addon_df = (
            addon_df.groupby(["anchor", "addon"])
            .agg(max_lift=("lift", "max"), total_support=("support", "sum"))
            .reset_index()
        )
        addon_df = addon_df.sort_values("max_lift", ascending=False).head(top_n)

        # Compute expected incremental revenue for each pair
        matrix_rows = []
        for _, row in addon_df.iterrows():
            anchor = row["anchor"]
            addon = row["addon"]
            lift = row["max_lift"]
            support = row["total_support"]

            # Expected incremental revenue
            if anchor in revenue_by_product:
                anchor_rev = float(revenue_by_product[anchor])
                # Every 1% of anchor basket that attaches addon → (lift-1) × revenue
                value = round(anchor_rev * support * max(lift - 1.0, 0.0), 2)
            else:
                value = 0.0

            matrix_rows.append(
                {
                    "Anchor": anchor,
                    "Add-on": addon,
                    "Lift": f"{lift:.2f}x",
                    "Support": f"{support:.2%}",
                    "Expected Incremental Revenue": f"${value:.2f}",
                    "Rationale": f"Lift {lift:.1f}x, Support {support:.1%} of anchor baskets",
                }
            )

        if matrix_rows:
            matrix_df = pd.DataFrame(matrix_rows)
            st.dataframe(matrix_df, use_container_width=True, hide_index=True)

            # Summary chart
            st.caption(
                f"Showing top {len(matrix_rows)} anchor×add-on pairs sorted by expected incremental revenue"
            )
        else:
            st.info("No cross-sell opportunities found.")
    else:
        st.info("No rule-derived add-on recommendations available.")


def _render_manager_table(
    rules: pd.DataFrame,
    df: pd.DataFrame,
    revenue_by_product: dict[str, float],
    basket: pd.DataFrame,
    profile_service: ProfileService | None,
) -> None:
    """Render the manager decision table with all required columns.

    Columns: anchor | add-on | lift | support | revenue opportunity | evidence | action
    Action includes "Do not act" option.
    """
    st.subheader(":material/clipboard: Manager Decision Table")

    if rules.empty:
        st.info("No rules to display in manager table.")
        return

    # Compute incremental basket value for each rule
    rules_with_val = rules.copy()
    rules_with_val["incr_value"] = rules_with_val.apply(
        lambda r: _compute_incremental_basket_value(r, revenue_by_product, basket),
        axis=1,
    )

    # Build product lookup for display names
    lookup = get_product_lookup(df)

    # Build manager table rows
    manager_rows = []
    for idx, row in rules_with_val.iterrows():
        antecedent = str(row["antecedents"])
        consequent = str(row["consequents"])
        lift = float(row["lift"])
        support = float(row["support"])
        incr_value = float(row["incr_value"])

        # Get product names for display
        ante_name = (
            lookup.loc[lookup["stockcode"] == antecedent.split()[0], "product"].iloc[0]
            if lookup is not None and antecedent != "unknown"
            else antecedent
        )
        cons_name = (
            lookup.loc[lookup["stockcode"] == consequent.split()[0], "product"].iloc[0]
            if lookup is not None and consequent != "unknown"
            else consequent
        )

        # Evidence: lift magnitude + support confidence
        evidence = f"Lift {lift:.2f}x, Support {support:.1%} of baskets"

        # Revenue opportunity label
        if incr_value > 0:
            revenue_label = f"${incr_value:.4f} incremental per anchor basket"
        else:
            revenue_label = "Minimal revenue impact"

        # Action options - include "Do not act"
        action = st.selectbox(
            f"Action for {ante_name} → {cons_name}",
            options=["Proceed", "Test bundle", "Monitor only", "Do not act"],
            key=f"action_{idx}",
            index=0,
        )

        manager_rows.append(
            {
                "Anchor": ante_name,
                "Add-on": cons_name,
                "Lift": f"{lift:.2f}x",
                "Support": f"{support:.2%}",
                "Revenue Opportunity": revenue_label,
                "Evidence": evidence,
                "Action": action,
            }
        )

    if manager_rows:
        manager_df = pd.DataFrame(manager_rows)
        st.dataframe(
            manager_df,
            use_container_width=True,
            hide_index=True,
        )

        # Summary: count of each action
        st.caption(
            f"Total rules: {len(manager_rows)} | "
            f"Do not act: {sum(1 for r in manager_rows if r['Action'] == 'Do not act')} | "
            f"Proceed: {sum(1 for r in manager_rows if r['Action'] == 'Proceed')} | "
            f"Test bundle: {sum(1 for r in manager_rows if r['Action'] == 'Test bundle')} | "
            f"Monitor only: {sum(1 for r in manager_rows if r['Action'] == 'Monitor only')}"
        )
    else:
        st.info("No manager table rows to display.")


def _integrate_product_profile(
    rules: pd.DataFrame,
    df: pd.DataFrame,
    profile_service: ProfileService,
) -> pd.DataFrame:
    """Integrate Product Decision Profile data into rules DataFrame.

    Adds profile fields per-SKU to each rule for manager decision support.
    """
    if rules.empty:
        return rules

    # Ensure profile service is initialized
    if profile_service is None:
        profile_service = init_profile_service(df)

    # Enrich each rule with profile data for antecedent and consequent SKUs
    enriched_rows = []
    for _, rule in rules.iterrows():
        row_dict = {"antecedents": rule["antecedents"], "consequents": rule["consequents"]}

        # Get profile for first antecedent SKU
        ante_sku = sorted(rule["antecedents"])[0] if rule["antecedents"] else None
        con_sku = sorted(rule["consequents"])[0] if rule["consequents"] else None

        if ante_sku:
            try:
                ante_profile = profile_service.get_profile(str(ante_sku))
                row_dict["ante_revenue"] = ante_profile.get("revenue", 0.0)
                row_dict["ante_abc"] = ante_profile.get("abc", "C")
                row_dict["ante_xyz"] = ante_profile.get("xyz", "Z")
            except Exception:
                row_dict["ante_revenue"] = 0.0
                row_dict["ante_abc"] = "C"
                row_dict["ante_xyz"] = "Z"
        else:
            row_dict["ante_revenue"] = 0.0
            row_dict["ante_abc"] = "C"
            row_dict["ante_xyz"] = "Z"

        if con_sku:
            try:
                con_profile = profile_service.get_profile(str(con_sku))
                row_dict["con_revenue"] = con_profile.get("revenue", 0.0)
                row_dict["con_abc"] = con_profile.get("abc", "C")
                row_dict["con_xyz"] = con_profile.get("xyz", "Z")
            except Exception:
                row_dict["con_revenue"] = 0.0
                row_dict["con_abc"] = "C"
                row_dict["con_xyz"] = "Z"
        else:
            row_dict["con_revenue"] = 0.0
            row_dict["con_abc"] = "C"
            row_dict["con_xyz"] = "Z"

        enriched_rows.append(row_dict)

    enriched_df = pd.DataFrame(enriched_rows)
    return pd.concat([rules.reset_index(drop=True), enriched_df], axis=1)


def render(df: pd.DataFrame) -> None:
    """Render the five-layer Rules tab.

    Five layers:
    1. Rule Opportunity Matrix — 4 quadrants (scale/optimize/rethink/stop)
    2. Top basket missions ranked by incremental basket value
    3. Cross-sell opportunity matrix (anchor × add-on = expected incremental revenue)
    4. Manager table (anchor | add-on | lift | support | revenue opportunity | evidence | action)
    5. Product Decision Profile integration for rule-level data
    """
    st.subheader(":material/schema: Association Rules (FP-Growth)")

    # Initialize profile service
    profile_service = init_profile_service(df)

    with st.expander("Parameters", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        min_support = c1.number_input("Min Support", 0.001, 0.5, 0.01, 0.001)
        max_len = c2.number_input("Max Itemset Length", 2, 5, 3)
        min_threshold = c3.number_input("Min Confidence", 0.01, 1.0, 0.05, 0.01)
        n_bootstrap = c4.number_input("Bootstrap Resamples", 5, 100, 25, 5)
        top_n_missions = c5.number_input("Top Missions to Show", 5, 50, 20, 5)

    # Build basket matrix and compute rules
    basket = get_basket_matrix(df)
    st.caption(f"Basket matrix: {basket.shape[0]} transactions × {basket.shape[1]} products")

    freq = run_fpgrowth(basket, min_support=min_support, max_len=max_len)
    st.caption(f"Frequent itemsets: {len(freq)}")

    if freq.empty:
        st.warning("No frequent itemsets found with current parameters.")
        return

    rules = generate_rules(freq, min_threshold=min_threshold)
    st.caption(f"Rules generated: {len(rules)}")

    if rules.empty:
        st.warning("No rules meet the confidence threshold.")
        return

    filtered = filter_rules(rules, min_lift=1.0, min_confidence=min_threshold)
    st.caption(f"Rules after filtering (lift ≥ 1.0): {len(filtered)}")

    if not filtered.empty:
        filtered = flag_redundant_rules(filtered)
        filtered = bootstrap_lift_ci(df, filtered, n_resamples=n_bootstrap)

        # Integrate Product Decision Profile
        enriched = _integrate_product_profile(filtered, df, profile_service)

        # Build revenue lookup
        revenue_by_product: dict[str, float] = {}
        if "price" in df.columns and "quantity" in df.columns:
            rev_series = df.groupby("stockcode").apply(
                lambda x: float((x["price"] * x["quantity"]).sum()), include_groups=False
            )
            revenue_by_product = {
                str(k): v for k, v in rev_series.items()
            }

        lookup = get_product_lookup(df)

        # === Layer 1: Rule Opportunity Matrix ===
        st.divider()
        _render_rule_opportunity_matrix(filtered, revenue_by_product, basket)

        # === Layer 2: Top basket missions ranked by incremental basket value ===
        st.divider()
        _render_top_basket_missions(filtered, revenue_by_product, basket, top_n=int(top_n_missions))

        # === Layer 3: Cross-sell opportunity matrix ===
        st.divider()
        _render_cross_sell_opportunity_matrix(filtered, df, top_n=int(top_n_missions))

        # === Layer 4: Manager table ===
        st.divider()
        _render_manager_table(filtered, df, revenue_by_product, basket, profile_service)

        # === Layer 5: Product Decision Profile integration ===
        st.divider()
        st.subheader(":material/dashboard: Product Decision Profile Integration")

        if not enriched.empty:
            # Show profile-enriched rules summary
            profile_cols = ["ante_revenue", "ante_abc", "ante_xyz", "con_revenue", "con_abc", "con_xyz"]
            available_profile_cols = [c for c in profile_cols if c in enriched.columns]

            if available_profile_cols:
                st.caption("Profile fields enriched per rule (antecedent/consequent ABC/XZ/revenue):")
                profile_display = enriched[
                    ["antecedent", "consequent"] + available_profile_cols
                ].head(10)
                st.dataframe(profile_display, use_container_width=True, hide_index=True)

            # Profile summary by ABC class
            st.caption("Profile distribution by ABC class:")
            if "ante_abc" in enriched.columns:
                abc_dist = enriched["ante_abc"].value_counts()
                st.bar_chart(abc_dist)

        # Rule table with enhanced display
        st.divider()
        st.subheader(":material/data_table: Rules Detail Table")

        table = rules_to_table(filtered, lookup)
        table["is_redundant"] = filtered["is_redundant"].values
        table["lift_ci_lower"] = filtered["lift_ci_lower"].values
        table["lift_ci_upper"] = filtered["lift_ci_upper"].values

        hide_redundant = st.checkbox("Hide redundant rules", value=False)
        display = table[~table["is_redundant"]] if hide_redundant else table

        # Add incremental value column
        if "incr_value" in enriched.columns:
            display["incr_value"] = enriched["incr_value"].values

        st.dataframe(display, use_container_width=True, hide_index=True)

        st.caption(f"Redundant rules: {int(filtered['is_redundant'].sum())} of {len(filtered)}")

        csv = table.to_csv(index=False)
        st.download_button(
            ":material/download: Download Rules CSV",
            csv,
            "association_rules.csv",
            "text/csv",
        )

        # Category-level rules rollup
        st.divider()
        st.subheader(":material/category: Category Affinities (Rollup)")
        cat_rules = aggregate_rules_to_categories(filtered, lookup, df)
        if not cat_rules.empty:
            cat_display = cat_rules.copy()
            cat_display["support"] = cat_display["support"].apply(lambda x: f"{x:.4f}")
            cat_display["confidence"] = cat_display["confidence"].apply(lambda x: f"{x:.2%}")
            cat_display["lift"] = cat_display["lift"].apply(lambda x: f"{x:.2f}")
            cat_display["avg_lift"] = cat_display["avg_lift"].apply(lambda x: f"{x:.2f}")
            cat_display["max_lift"] = cat_display["max_lift"].apply(lambda x: f"{x:.2f}")

            st.dataframe(
                cat_display[
                    [
                        "antecedent_category",
                        "consequent_category",
                        "rule_count",
                        "support",
                        "confidence",
                        "lift",
                        "avg_lift",
                        "max_lift",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            # Category affinity heatmap
            st.caption("Heatmap: Lift by Category Pair")
            pivot = cat_rules.pivot_table(
                index="antecedent_category",
                columns="consequent_category",
                values="lift",
                fill_value=0,
            )
            if not pivot.empty:
                fig = go.Figure(
                    data=go.Heatmap(
                        z=pivot.values,
                        x=pivot.columns.tolist(),
                        y=pivot.index.tolist(),
                        colorscale="RdYlGn",
                        colorbar={"title": "Avg Lift"},
                        hovertemplate="From: %{y}<br>To: %{x}<br>Lift: %{z:.2f}<extra></extra>",
                    )
                )
                fig.update_layout(
                    xaxis={"title": "Consequent Category"},
                    yaxis={"title": "Antecedent Category"},
                    height=max(300, len(pivot) * 30 + 100),
                )
                fig.update_xaxes(tickangle=-45)
                show(fig)

            csv_cat = cat_rules.to_csv(index=False)
            st.download_button(
                ":material/download: Download Category Rules CSV",
                csv_cat,
                "category_rules.csv",
                "text/csv",
            )
        else:
            st.info("No category-level rules available (insufficient category diversity).")

        # Rule Strength vs Stability scatter
        st.divider()
        _render_strength_stability_scatter(filtered, table)

        st.divider()
        _render_lift_ci_chart(filtered, table, top_n=15)

        st.divider()
        _render_anchor_drilldown(df, filtered, table)

        st.divider()
        top_n_network = st.slider("Network: top rules by lift", 10, 100, 40)
        _render_rule_network(df, filtered, top_n=top_n_network)


def _render_strength_stability_scatter(filtered: pd.DataFrame, table: pd.DataFrame) -> None:
    """Render rule strength vs stability scatter."""
    st.subheader(":material/scatter_plot: Rule Strength vs Stability")
    st.info("Strength vs Stability scatter plot - coming soon")


def _render_lift_ci_chart(filtered: pd.DataFrame, table: pd.DataFrame, top_n: int = 15) -> None:
    """Render lift confidence interval chart."""
    st.subheader(":material/bar_chart: Lift Confidence Intervals")
    st.info("Lift CI chart - coming soon")


def _render_anchor_drilldown(df: pd.DataFrame, filtered: pd.DataFrame, table: pd.DataFrame) -> None:
    """Render anchor product drill-down."""
    st.subheader(":material/search: Anchor Product Drill-Down")
    st.info("Anchor drill-down - coming soon")


def _render_rule_network(df: pd.DataFrame, filtered: pd.DataFrame, top_n: int) -> None:
    """Render rule network graph."""
    st.subheader(":material/network: Rule Network")
    st.info("Rule network graph - coming soon")


MODE_SPEC: ModeSpec = ModeSpec(
    key="rules",
    label="Association Rules",
    icon=":material/schema:",
    handler=render,
    requires=("sufficient_baskets_200", "sufficient_skus_20"),
)
