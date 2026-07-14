"""Product Performance Analytics Tab with persistent tab state."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.product_performance import compute_product_metrics
from src.ui.export import render_analytics_export
from src.ui.tabs import persistent_tabs


def render_product_performance_tab(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict
):
    """Render product performance analytics tab with persistent sub-tabs."""
    st.header("📦 Product Performance Analytics")

    if transactions_df.empty:
        st.warning("No transaction data available")
        return

    # Persistent sub-tabs
    tab_labels = [
        "📊 Product Dashboard",
        "🔄 Lifecycle Analysis",
        "📅 Seasonality",
        "🔗 Affinity & Cross-sell",
        "💰 Price Elasticity",
        "📈 Performance Comparison",
    ]
    selected = persistent_tabs(tab_labels, "product_perf_main_tabs", default_tab=0)

    if selected == 0:
        render_product_dashboard(transactions_df, product_lookup)
    elif selected == 1:
        render_lifecycle_analysis(transactions_df, product_lookup, params)
    elif selected == 2:
        render_seasonality_analysis(transactions_df, product_lookup, params)
    elif selected == 3:
        render_affinity_analysis(transactions_df, product_lookup, params)
    elif selected == 4:
        render_price_elasticity(transactions_df, product_lookup, params)
    elif selected == 5:
        render_performance_comparison(transactions_df, product_lookup, params)


def render_product_dashboard(transactions_df: pd.DataFrame, product_lookup: dict):
    """Render product performance dashboard."""
    st.subheader("Product Performance Overview")

    # Compute metrics
    with st.spinner("Computing product metrics..."):
        metrics = compute_product_metrics(transactions_df)

    if metrics.empty:
        st.warning("No product metrics available")
        return

    # Add product names
    metrics["product_name"] = metrics["stockcode"].map(product_lookup)

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    total_revenue = metrics["total_revenue"].sum()
    total_qty = metrics["total_quantity"].sum()
    n_products = len(metrics)

    with col1:
        st.metric("Total Products", f"{n_products:,}")
    with col2:
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    with col3:
        st.metric("Total Quantity", f"{total_qty:,.0f}")
    with col4:
        st.metric("Avg Price", f"${metrics['avg_price'].mean():.2f}")
    with col5:
        st.metric(
            "Revenue Concentration (Top 10%)",
            f"{metrics.nlargest(int(len(metrics) * 0.1), 'total_revenue')['total_revenue'].sum() / total_revenue * 100:.1f}%",
        )

    # Top products table
    st.subheader("Top Products by Revenue")

    top_n = st.slider("Top N Products", 5, 50, 20, key="perf_tab_top_n")

    display_cols = [
        "product_name",
        "total_revenue",
        "total_quantity",
        "avg_price",
        "avg_margin",
        "frequency",
        "unique_customers",
        "revenue_per_customer",
    ]
    display_cols = [c for c in display_cols if c in metrics.columns]

    top_products = metrics.nlargest(top_n, "total_revenue")[display_cols]

    st.dataframe(
        top_products.style.format(
            {
                "total_revenue": "${:,.2f}",
                "total_quantity": "{:,.0f}",
                "avg_price": "${:.2f}",
                "avg_margin": "{:.2%}",
                "frequency": "{:,.0f}",
                "unique_customers": "{:,.0f}",
                "revenue_per_customer": "${:,.2f}",
            }
        ).background_gradient(cmap="RdYlGn", subset=["total_revenue"]),
        width="stretch",
    )

    render_analytics_export(metrics, "Product_Performance_Metrics")

    # Visualizations
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        # Revenue vs Quantity scatter
        fig = px.scatter(
            metrics,
            x="total_quantity",
            y="total_revenue",
            size="unique_customers" if "unique_customers" in metrics.columns else None,
            color="avg_margin" if "avg_margin" in metrics.columns else None,
            hover_data=["product_name"],
            title="Product Revenue vs Quantity",
            labels={
                "total_quantity": "Total Quantity Sold",
                "total_revenue": "Total Revenue ($)",
            },
        )
        st.plotly_chart(fig, width="stretch")

    with viz_col2:
        # Revenue concentration (Pareto)
        sorted_rev = metrics["total_revenue"].sort_values(ascending=False).reset_index(drop=True)
        cum_pct = sorted_rev.cumsum() / sorted_rev.sum() * 100

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=list(range(1, len(sorted_rev) + 1)),
                y=sorted_rev.values,
                name="Revenue",
                marker_color="steelblue",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(sorted_rev) + 1)),
                y=cum_pct.values,
                name="Cumulative %",
                yaxis="y2",
                line=dict(color="red", width=2),
            )
        )
        fig.update_layout(
            title="Revenue Concentration (Pareto)",
            xaxis_title="Product Rank",
            yaxis_title="Revenue ($)",
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100]),
        )
        st.plotly_chart(fig, width="stretch")

    # Margin distribution
    if "avg_margin" in metrics.columns:
        st.subheader("Margin Distribution")
        fig = px.histogram(
            metrics,
            x="avg_margin",
            nbins=30,
            title="Average Margin Distribution",
            labels={"avg_margin": "Margin %", "count": "Number of Products"},
        )
        fig.add_vline(
            x=metrics["avg_margin"].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {metrics['avg_margin'].mean():.1%}",
        )
        st.plotly_chart(fig, width="stretch")


def render_lifecycle_analysis(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render product lifecycle analysis."""
    st.subheader("Product Lifecycle Analysis")

    top_n = st.slider("Top N Products by Revenue", 10, 200, 50, key="lifecycle_tab_top_n")

    with st.spinner("Analyzing product lifecycles..."):
        from src.analytics.product_performance import product_lifecycle_stage

        # First compute product metrics
        product_metrics = compute_product_metrics(transactions_df)
        lifecycle_df = product_lifecycle_stage(product_metrics, transactions_df, period="ME")

    if lifecycle_df.empty:
        st.warning("Insufficient data for lifecycle analysis")
        return

    # Filter to top N products
    lifecycle_df = lifecycle_df.nlargest(top_n, "total_revenue")

    # Add product names
    lifecycle_df["product_name"] = lifecycle_df["stockcode"].map(product_lookup)

    # Stage distribution
    st.subheader("Lifecycle Stage Distribution")

    stage_counts = lifecycle_df["stage"].value_counts().reset_index()
    stage_counts.columns = ["Stage", "Products"]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            stage_counts,
            values="Products",
            names="Stage",
            title="Product Lifecycle Stages",
            color_discrete_map={
                "Introduction": "#4CAF50",
                "Growth": "#2196F3",
                "Maturity": "#FF9800",
                "Decline": "#F44336",
                "Insufficient data": "#9E9E9E",
            },
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        # Revenue by stage
        stage_rev = (
            lifecycle_df.groupby("stage")
            .agg(
                products=("stockcode", "count"),
                total_revenue=("total_revenue", "sum"),
                avg_growth=("growth_rate", "mean"),
            )
            .reset_index()
        )

        fig = px.bar(
            stage_rev,
            x="stage",
            y="total_revenue",
            color="products",
            title="Revenue by Lifecycle Stage",
            labels={
                "stage": "Stage",
                "total_revenue": "Revenue ($)",
                "products": "Products",
            },
        )
        st.plotly_chart(fig, width="stretch")

    # Detailed table
    st.subheader("Product Lifecycle Details")

    display_cols = [
        "product_name",
        "stage",
        "trend",
        "growth_rate",
        "r_squared",
        "months_active",
        "total_revenue",
        "total_quantity",
        "avg_price",
    ]
    display_cols = [c for c in display_cols if c in lifecycle_df.columns]

    st.dataframe(
        lifecycle_df[display_cols]
        .style.format(
            {
                "growth_rate": "{:.2%}",
                "r_squared": "{:.3f}",
                "total_revenue": "${:,.2f}",
                "total_quantity": "{:,.0f}",
                "avg_price": "${:.2f}",
            }
        )
        .background_gradient(cmap="RdYlGn", subset=["growth_rate"]),
        width="stretch",
    )

    render_analytics_export(lifecycle_df, "Product_Lifecycle_Analysis")

    # Growth vs Revenue scatter
    st.subheader("Growth vs Revenue Positioning")
    fig = px.scatter(
        lifecycle_df.dropna(subset=["growth_rate", "total_revenue"]),
        x="total_revenue",
        y="growth_rate",
        color="stage",
        size="months_active",
        hover_data=["product_name"],
        title="Product Positioning: Revenue vs Growth",
        labels={
            "total_revenue": "Total Revenue ($)",
            "growth_rate": "Monthly Growth Rate",
        },
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, width="stretch")


def render_seasonality_analysis(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render seasonality analysis."""
    st.subheader("Product Seasonality Analysis")

    # Product selector
    products = transactions_df["stockcode"].unique()
    selected_product = st.selectbox(
        "Select Product for Seasonality Analysis",
        products,
        format_func=lambda x: product_lookup.get(x, x),
        key="seasonality_tab_product",
    )

    if selected_product:
        with st.spinner("Analyzing seasonality..."):
            from src.analytics.product_performance import product_seasonality

            seasonality = product_seasonality(transactions_df, selected_product)

        if not seasonality.get("has_seasonality", False):
            st.info(
                f"Seasonality not detected: {seasonality.get('message', 'Insufficient pattern')}"
            )

            # Still show monthly pattern if available
            if "monthly_pattern" in seasonality:
                month_data = pd.DataFrame(
                    {
                        "Month": list(seasonality["monthly_pattern"].keys()),
                        "Avg Quantity": list(seasonality["monthly_pattern"].values()),
                    }
                )
                month_data["Month"] = month_data["Month"].astype(int)
                month_data = month_data.sort_values("Month")

                fig = px.bar(
                    month_data,
                    x="Month",
                    y="Avg Quantity",
                    title=f"Monthly Pattern for {product_lookup.get(selected_product, selected_product)}",
                )
                st.plotly_chart(fig, width="stretch")
            return

        # Show seasonality metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Seasonal Strength", f"{seasonality['seasonal_strength']:.2%}")
        with col2:
            st.metric("CV Across Months", f"{seasonality['cv_across_months']:.2%}")
        with col3:
            st.metric("Peak Month", seasonality["peak_month"])
        with col4:
            st.metric(
                "Peak-to-Trough Ratio",
                f"{seasonality.get('peak_to_trough_ratio', 0):.1f}x",
            )

        # Monthly pattern chart
        if "monthly_pattern" in seasonality:
            month_data = pd.DataFrame(
                {
                    "Month": list(seasonality["monthly_pattern"].keys()),
                    "Avg Quantity": list(seasonality["monthly_pattern"].values()),
                }
            )
            month_data["Month"] = month_data["Month"].astype(int)
            month_data = month_data.sort_values("Month")

            month_names = {
                1: "Jan",
                2: "Feb",
                3: "Mar",
                4: "Apr",
                5: "May",
                6: "Jun",
                7: "Jul",
                8: "Aug",
                9: "Sep",
                10: "Oct",
                11: "Nov",
                12: "Dec",
            }
            month_data["Month_Name"] = month_data["Month"].map(month_names)

            fig = px.bar(
                month_data,
                x="Month_Name",
                y="Avg Quantity",
                title=f"Seasonal Pattern: {product_lookup.get(selected_product, selected_product)}",
                color="Avg Quantity",
                color_continuous_scale="RdYlBu",
            )
            st.plotly_chart(fig, width="stretch")

        # Seasonality interpretation
        st.subheader("Interpretation")
        if seasonality["seasonal_strength"] > 0.6:
            st.success(
                f"🔍 Strong seasonality detected! Peak demand in month {seasonality['peak_month']} "
                f"({['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][seasonality['peak_month'] - 1]}). "
                f"Plan inventory and promotions accordingly."
            )
        elif seasonality["seasonal_strength"] > 0.3:
            st.info(
                f"📊 Moderate seasonality. Peak in month {seasonality['peak_month']}. "
                f"Consider seasonal adjustments."
            )
        else:
            st.info("📊 Weak seasonality. Demand relatively stable across months.")


def render_affinity_analysis(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render product affinity and cross-sell analysis."""
    st.subheader("Product Affinity & Cross-Sell Opportunities")

    # Product selector for affinity
    products = transactions_df["stockcode"].unique()
    selected_product = st.selectbox(
        "Select Anchor Product for Affinity Analysis",
        products,
        format_func=lambda x: product_lookup.get(x, x),
        key="affinity_tab_product",
    )

    if selected_product:
        min_support = st.slider(
            "Min Support", 0.0005, 0.01, 0.001, 0.0005, key="affinity_tab_min_support"
        )

        with st.spinner("Computing affinity scores..."):
            from src.analytics.product_performance import product_affinity_score

            affinity = product_affinity_score(
                transactions_df, selected_product, min_support=min_support
            )

        if affinity.empty:
            st.warning(
                f"No strong affinities found for {product_lookup.get(selected_product, selected_product)}"
            )
        else:
            st.subheader(
                f"Top Affinities for {product_lookup.get(selected_product, selected_product)}"
            )

            affinity["product_name"] = affinity["product"].map(product_lookup)

            display_cols = ["product_name", "support", "confidence", "lift", "leverage"]
            st.dataframe(
                affinity[display_cols]
                .head(20)
                .style.format(
                    {
                        "support": "{:.4f}",
                        "confidence": "{:.2%}",
                        "lift": "{:.2f}",
                        "leverage": "{:.4f}",
                    }
                )
                .background_gradient(cmap="RdYlGn", subset=["lift", "confidence"]),
                width="stretch",
            )

            # Visualization
            top_affinity = affinity.head(15)
            fig = px.bar(
                top_affinity,
                x="lift",
                y="product_name",
                orientation="h",
                color="confidence",
                title=f"Top Cross-Sell Opportunities for {product_lookup.get(selected_product, selected_product)}",
                labels={
                    "product_name": "Product",
                    "lift": "Lift",
                    "confidence": "Confidence",
                },
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width="stretch")

            render_analytics_export(affinity, f"Affinity_{selected_product}")

    # Cross-sell opportunity matrix
    st.subheader("Cross-Sell Opportunity Matrix")

    matrix_n = st.slider("Top N Products for Matrix", 10, 50, 30, key="affinity_tab_matrix_n")

    with st.spinner("Computing cross-sell matrix..."):
        from src.analytics.product_performance import cross_sell_opportunity_matrix

        matrix = cross_sell_opportunity_matrix(transactions_df, top_n=matrix_n)

    if not matrix.empty:
        # Add product names to index/columns for display
        display_matrix = matrix.copy()
        display_matrix.index = [product_lookup.get(x, x)[:30] for x in matrix.index]
        display_matrix.columns = [product_lookup.get(x, x)[:30] for x in matrix.columns]

        fig = px.imshow(
            display_matrix.values,
            x=display_matrix.columns,
            y=display_matrix.index,
            color_continuous_scale="RdYlGn",
            aspect="auto",
            title="Cross-Sell Opportunity Matrix (Lift)",
            labels={"color": "Lift"},
        )
        fig.update_layout(height=600, xaxis_tickangle=45)
        st.plotly_chart(fig, width="stretch")

        # Show top cross-sell pairs
        st.subheader("Top Cross-Sell Pairs")
        pairs = []
        for i, prod_a in enumerate(matrix.index):
            for j, prod_b in enumerate(matrix.columns):
                if i < j and matrix.iloc[i, j] > 1.5:
                    pairs.append(
                        {
                            "Product A": product_lookup.get(prod_a, prod_a),
                            "Product B": product_lookup.get(prod_b, prod_b),
                            "Lift": matrix.iloc[i, j],
                        }
                    )

        if pairs:
            pairs_df = pd.DataFrame(pairs).sort_values("Lift", ascending=False).head(20)
            st.dataframe(
                pairs_df.style.format({"Lift": "{:.2f}"}).background_gradient(
                    cmap="RdYlGn", subset=["Lift"]
                ),
                width="stretch",
            )
            render_analytics_export(pairs_df, "Top_Cross_Sell_Pairs")


def render_price_elasticity(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render price elasticity analysis."""
    st.subheader("Price Elasticity Analysis")

    products = transactions_df["stockcode"].unique()
    selected_product = st.selectbox(
        "Select Product for Elasticity Analysis",
        products,
        format_func=lambda x: product_lookup.get(x, x),
        key="elasticity_tab_product",
    )

    if selected_product:
        min_periods = st.slider("Min Weekly Periods", 5, 50, 10, key="elasticity_tab_min_periods")

        with st.spinner("Estimating price elasticity..."):
            from src.analytics.product_performance import price_elasticity_analysis

            elasticity = price_elasticity_analysis(
                transactions_df, selected_product, min_periods=min_periods
            )

        if elasticity.get("elasticity") is None:
            st.warning(
                f"Cannot estimate elasticity: {elasticity.get('message', 'Insufficient data')}"
            )
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Elasticity", f"{elasticity['elasticity']:.3f}")
            with col2:
                st.metric("R²", f"{elasticity['r_squared']:.3f}")
            with col3:
                st.metric("P-value", f"{elasticity['p_value']:.4f}")
            with col4:
                st.metric("Observations", elasticity["n_observations"])

            st.info(f"**Interpretation:** {elasticity['interpretation']}")

            # Show price-quantity scatter
            prod_df = transactions_df[transactions_df["stockcode"] == selected_product].copy()
            prod_df["date"] = pd.to_datetime(prod_df["date"])
            prod_df["revenue"] = prod_df["price"] * prod_df["quantity"]

            weekly = (
                prod_df.set_index("date")
                .groupby(pd.Grouper(freq="W"))
                .agg(avg_price=("price", "mean"), total_qty=("quantity", "sum"))
                .dropna()
            )

            if len(weekly) > 0:
                fig = px.scatter(
                    weekly,
                    x="avg_price",
                    y="total_qty",
                    trendline="ols",
                    title=f"Price vs Quantity (Weekly): {product_lookup.get(selected_product, selected_product)}",
                    labels={
                        "avg_price": "Average Price ($)",
                        "total_qty": "Total Quantity",
                    },
                )
                st.plotly_chart(fig, width="stretch")

                # Elasticity interpretation
                if elasticity["elasticity"] < -1:
                    st.success(
                        "💡 **Pricing Opportunity**: Demand is elastic. Price reductions could increase revenue."
                    )
                elif elasticity["elasticity"] < 0:
                    st.info(
                        "📊 **Stable Pricing**: Demand is inelastic. Price increases may increase revenue without losing much volume."
                    )
                else:
                    st.warning(
                        "⚠️ **Unusual Pattern**: Positive elasticity detected. Investigate data quality."
                    )


def render_performance_comparison(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict
):
    """Render product performance comparison."""
    st.subheader("Product Performance Comparison")

    # Select products to compare
    products = transactions_df["stockcode"].unique()
    selected_products = st.multiselect(
        "Select Products to Compare (max 10)",
        products,
        format_func=lambda x: product_lookup.get(x, x),
        max_selections=10,
        default=list(products[:5]) if len(products) >= 5 else list(products),
        key="compare_tab_products",
    )

    if not selected_products:
        st.info("Select products to compare")
        return

    # Compute metrics for selected products
    all_metrics = compute_product_metrics(transactions_df)
    comparison = all_metrics[all_metrics["stockcode"].isin(selected_products)].copy()
    comparison["product_name"] = comparison["stockcode"].map(product_lookup)

    if comparison.empty:
        st.warning("No metrics available for selected products")
        return

    # Metric selection
    available_metrics = [
        c
        for c in [
            "total_revenue",
            "total_quantity",
            "avg_price",
            "avg_margin",
            "frequency",
            "unique_customers",
            "revenue_per_customer",
        ]
        if c in comparison.columns
    ]

    metric = st.selectbox(
        "Metric to Compare",
        available_metrics,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    # Bar chart comparison
    fig = px.bar(
        comparison,
        x="product_name",
        y=metric,
        color="stockcode",
        title=f"Product Comparison: {metric.replace('_', ' ').title()}",
        labels={"product_name": "Product", metric: metric.replace("_", " ").title()},
    )
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, width="stretch")

    # Radar chart
    if len(selected_products) >= 2:
        st.subheader("Multi-Metric Radar Comparison")

        radar_metrics = [
            "total_revenue",
            "total_quantity",
            "avg_price",
            "frequency",
            "unique_customers",
        ]
        radar_metrics = [m for m in radar_metrics if m in comparison.columns]

        if radar_metrics:
            # Normalize metrics for radar
            radar_data = comparison[["product_name"] + radar_metrics].copy()
            for m in radar_metrics:
                max_val = radar_data[m].max()
                if max_val > 0:
                    radar_data[m] = radar_data[m] / max_val

            fig = go.Figure()
            for _, row in radar_data.iterrows():
                fig.add_trace(
                    go.Scatterpolar(
                        r=row[radar_metrics].values,
                        theta=[m.replace("_", " ").title() for m in radar_metrics],
                        fill="toself",
                        name=row["product_name"],
                    )
                )
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Normalized Product Comparison",
            )
            st.plotly_chart(fig, width="stretch")

    # Detailed comparison table
    st.subheader("Detailed Comparison")
    st.dataframe(
        comparison[["product_name"] + available_metrics]
        .style.format(
            {
                "total_revenue": "${:,.2f}",
                "total_quantity": "{:,.0f}",
                "avg_price": "${:.2f}",
                "avg_margin": "{:.2%}",
                "frequency": "{:,.0f}",
                "unique_customers": "{:,.0f}",
                "revenue_per_customer": "${:,.2f}",
            }
        )
        .background_gradient(cmap="RdYlGn"),
        width="stretch",
    )

    render_analytics_export(comparison, "Product_Comparison")
