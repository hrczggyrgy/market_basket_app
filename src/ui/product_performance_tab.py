"""Product Performance Analytics Tab with persistent tab state.

Transaction-only metrics: revenue, demand, loyalty, penetration, stability, switching.
No margin/COGS visuals.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.product_performance import (
    compute_basket_uplift,
    compute_product_dashboard_metrics,
    compute_product_metrics,
    compute_repeat_rate,
    compute_switching_gain_loss,
    compute_time_to_second_purchase,
    product_lifecycle_stage,
)
from src.ui.export import render_analytics_export
from src.ui.tabs import persistent_tabs


def render_product_performance_tab(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict
):
    """Render product performance analytics tab with persistent sub-tabs."""
    st.header(" Product Performance Analytics")

    if transactions_df.empty:
        st.warning("No transaction data available")
        return

    # Persistent sub-tabs
    tab_labels = [
        " Dashboard",
        " Opportunity Scatter",
        " Monthly Trends",
        " Loyalty & Repeat",
        " Switching & Uplift",
        " Lifecycle",
        " Seasonality",
        " Affinity & Cross-sell",
        " Price Elasticity",
    ]
    selected = persistent_tabs(tab_labels, "product_perf_main_tabs", default_tab=0)

    if selected == 0:
        render_product_dashboard(transactions_df, product_lookup)
    elif selected == 1:
        render_opportunity_scatter(transactions_df, product_lookup)
    elif selected == 2:
        render_monthly_trends(transactions_df, product_lookup)
    elif selected == 3:
        render_loyalty_repeat(transactions_df, product_lookup)
    elif selected == 4:
        render_switching_uplift(transactions_df, product_lookup)
    elif selected == 5:
        render_lifecycle_analysis(transactions_df, product_lookup, params)
    elif selected == 6:
        render_seasonality_analysis(transactions_df, product_lookup, params)
    elif selected == 7:
        render_affinity_analysis(transactions_df, product_lookup, params)
    elif selected == 8:
        render_price_elasticity(transactions_df, product_lookup, params)


def render_product_dashboard(transactions_df: pd.DataFrame, product_lookup: dict):
    """Render comprehensive product dashboard with transaction-only KPIs."""
    st.subheader("Product Performance Overview")

    # Compute comprehensive metrics
    with st.spinner("Computing product dashboard metrics..."):
        metrics = compute_product_dashboard_metrics(transactions_df)

    if metrics.empty:
        st.warning("No product metrics available")
        return

    # Add product names
    metrics["product_name"] = metrics["stockcode"].map(product_lookup)

    # ============================================================
    # KPI TILES
    # ============================================================
    st.subheader("Key Metrics")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    total_revenue = metrics["total_revenue"].sum()
    total_qty = metrics["total_quantity"].sum()
    n_products = len(metrics)
    avg_pen = metrics["basket_penetration"].mean() if "basket_penetration" in metrics else 0
    avg_repeat = metrics["repeat_rate"].mean() if "repeat_rate" in metrics else 0
    avg_vel = metrics["velocity"].mean() if "velocity" in metrics else 0

    with col1:
        st.metric("Products", f"{n_products:,}")
    with col2:
        st.metric("Revenue", f"${total_revenue:,.0f}")
    with col3:
        st.metric("Units Sold", f"{total_qty:,.0f}")
    with col4:
        st.metric("Avg Basket Penetration", f"{avg_pen:.1%}")
    with col5:
        st.metric("Avg Repeat Rate", f"{avg_repeat:.1%}")
    with col6:
        st.metric("Avg Velocity (units/mo)", f"{avg_vel:.1f}")

    # ============================================================
    # TOP PRODUCTS TABLE
    # ============================================================
    st.subheader("Top Products by Revenue")

    top_n = st.slider("Top N Products", 5, 100, 20, key="perf_tab_top_n")

    display_cols = [
        "product_name",
        "total_revenue",
        "total_quantity",
        "avg_price",
        "unique_customers",
        "revenue_per_customer",
        "basket_penetration",
        "repeat_rate",
        "velocity",
        "price_index",
    ]
    display_cols = [c for c in display_cols if c in metrics.columns]

    top_products = metrics.nlargest(top_n, "total_revenue")[display_cols]

    st.dataframe(
        top_products.style.format(
            {
                "total_revenue": "${:,.2f}",
                "total_quantity": "{:,.0f}",
                "avg_price": "${:.2f}",
                "unique_customers": "{:,.0f}",
                "revenue_per_customer": "${:,.2f}",
                "basket_penetration": "{:.1%}",
                "repeat_rate": "{:.1%}",
                "velocity": "{:.1f}",
                "price_index": "{:.2f}",
            }
        ).background_gradient(cmap="RdYlGn", subset=["total_revenue", "basket_penetration", "repeat_rate", "velocity"]),
        width="stretch",
    )

    render_analytics_export(metrics, "Product_Dashboard_Metrics")

    # ============================================================
    # QUICK VISUALIZATIONS
    # ============================================================
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
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
            height=350,
        )
        st.plotly_chart(fig, width="stretch")

    with viz_col2:
        # Price positioning distribution
        if "price_index" in metrics.columns:
            fig = px.histogram(
                metrics,
                x="price_index",
                nbins=30,
                title="Price Positioning Index Distribution",
                labels={"price_index": "Price Index (vs Category Median)"},
            )
            fig.add_vline(
                x=1.0,
                line_dash="dash",
                line_color="red",
                annotation_text="Category Median",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, width="stretch")


def render_opportunity_scatter(transactions_df: pd.DataFrame, product_lookup: dict):
    """Render product opportunity scatter: penetration vs velocity."""
    st.subheader("Product Opportunity Scatter")
    st.caption("X = Basket Penetration (reach) | Y = Velocity (demand intensity) | Bubble = Revenue | Color = Repeat Rate")

    with st.spinner("Computing opportunity metrics..."):
        metrics = compute_product_dashboard_metrics(transactions_df)

    if metrics.empty:
        st.warning("No metrics available")
        return

    metrics["product_name"] = metrics["stockcode"].map(product_lookup)

    # Filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        min_revenue = st.number_input("Min Revenue", 0, int(metrics["total_revenue"].max()), 0, key="opp_min_rev")
    with col2:
        max_points = st.slider("Max Points", 20, 200, 100, key="opp_max_points")

    plot_df = metrics[metrics["total_revenue"] >= min_revenue].copy()
    plot_df = plot_df.nlargest(max_points, "total_revenue")

    # Ensure required columns exist
    x_col = "basket_penetration" if "basket_penetration" in plot_df.columns else "unique_shopper_penetration"
    y_col = "velocity" if "velocity" in plot_df.columns else "sell_through_rate"
    color_col = "repeat_rate" if "repeat_rate" in plot_df.columns else "revenue_per_customer"
    size_col = "total_revenue"

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        size=size_col,
        color=color_col,
        hover_data=["product_name", "total_revenue", "total_quantity", "avg_price", "unique_customers", "repeat_rate", "velocity", "price_index"],
        title="Product Opportunity Map: Penetration vs Velocity",
        labels={
            x_col: "Basket Penetration (Reach)",
            y_col: "Velocity (Units/Active Period)",
            color_col: "Repeat Rate",
            size_col: "Revenue",
        },
        color_continuous_scale="RdYlGn",
        size_max=50,
    )

    # Add quadrant lines
    if x_col in plot_df.columns and y_col in plot_df.columns:
        x_mid = plot_df[x_col].median()
        y_mid = plot_df[y_col].median()
        fig.add_vline(x=x_mid, line_dash="dash", line_color="gray", annotation_text="Median Penetration")
        fig.add_hline(y=y_mid, line_dash="dash", line_color="gray", annotation_text="Median Velocity")

    fig.update_layout(height=600)
    st.plotly_chart(fig, width="stretch")

    # Quadrant interpretation
    st.caption("""
    **Quadrant Guide:**
    - **High Pen + High Vel** = Stars (broad reach, strong demand)
    - **High Pen + Low Vel** = Cash Cows (broad reach, steady demand)
    - **Low Pen + High Vel** = Niche Gems (deep loyalty, growth potential)
    - **Low Pen + Low Vel** = Dogs / New Products (investigate or sunset)
    """)


def render_monthly_trends(transactions_df: pd.DataFrame, product_lookup: dict):
    """Render monthly trend sparklines for top products."""
    st.subheader("Monthly Trend Sparklines")

    with st.spinner("Computing monthly trends..."):
        from src.analytics.basket_metrics import basket_penetration_over_time
        pen_trends = basket_penetration_over_time(transactions_df)

    if pen_trends.empty:
        st.warning("No trend data available")
        return

    # Get top products by revenue
    metrics = compute_product_dashboard_metrics(transactions_df)
    top_products = metrics.nlargest(20, "total_revenue")["stockcode"].tolist()

    # Filter trends to top products
    top_trends = pen_trends[pen_trends["stockcode"].isin(top_products)].copy()
    top_trends["product_name"] = top_trends["stockcode"].map(product_lookup)

    if top_trends.empty:
        st.warning("No trend data for top products")
        return

    # Period to datetime for plotting
    top_trends["period_dt"] = top_trends["period"].dt.to_timestamp()

    # Sparkline chart - small multiples
    selected_products = st.multiselect(
        "Select Products",
        top_products,
        default=top_products[:8],
        format_func=lambda x: product_lookup.get(x, x),
        key="trends_product_select",
    )

    if not selected_products:
        st.info("Select products to view trends")
        return

    filtered = top_trends[top_trends["stockcode"].isin(selected_products)]

    fig = px.line(
        filtered,
        x="period_dt",
        y="basket_penetration",
        color="product_name",
        title="Monthly Basket Penetration Trends",
        labels={"period_dt": "Month", "basket_penetration": "Basket Penetration"},
    )
    fig.update_layout(height=500, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    # Revenue trend
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["price"] * df["quantity"]
    df["period"] = df["date"].dt.to_period("M")

    rev_trends = df[df["stockcode"].isin(selected_products)].groupby(["stockcode", "period"])["revenue"].sum().reset_index()
    rev_trends["period_dt"] = rev_trends["period"].dt.to_timestamp()
    rev_trends["product_name"] = rev_trends["stockcode"].map(product_lookup)

    fig2 = px.line(
        rev_trends,
        x="period_dt",
        y="revenue",
        color="product_name",
        title="Monthly Revenue Trends",
        labels={"period_dt": "Month", "revenue": "Revenue ($)"},
    )
    fig2.update_layout(height=500, hovermode="x unified")
    st.plotly_chart(fig2, width="stretch")


def render_loyalty_repeat(transactions_df: pd.DataFrame, product_lookup: dict):
    """Render repeat rate, time-to-second-purchase, loyalty metrics."""
    st.subheader("Loyalty & Repeat Purchase Metrics")

    with st.spinner("Computing loyalty metrics..."):
        repeat_rate = compute_repeat_rate(transactions_df)
        tt2p = compute_time_to_second_purchase(transactions_df)

    if repeat_rate.empty:
        st.warning("No loyalty data available")
        return

    repeat_rate["product_name"] = repeat_rate["stockcode"].map(product_lookup)
    tt2p["product_name"] = tt2p["stockcode"].map(product_lookup)

    # Merge
    loyalty = repeat_rate.merge(
        tt2p[["stockcode", "median_days_to_second", "p25_days_to_second", "p75_days_to_second"]],
        on="stockcode",
        how="left",
    )

    # KPI tiles
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Repeat Rate", f"{loyalty['repeat_rate'].mean():.1%}")
    with col2:
        st.metric("Products with >50% Repeat", f"{(loyalty['repeat_rate'] > 0.5).sum()}")
    with col3:
        st.metric("Median Time to 2nd Purchase", f"{loyalty['median_days_to_second'].median():.0f} days")
    with col4:
        st.metric("Products with Fast Loyalty (<30d)", f"{(loyalty['median_days_to_second'] < 30).sum()}")

    # Scatter: Repeat Rate vs Time to 2nd Purchase
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            loyalty.dropna(subset=["repeat_rate", "median_days_to_second"]),
            x="median_days_to_second",
            y="repeat_rate",
            size="total_buyers",
            color="repeat_rate",
            hover_data=["product_name", "repeat_buyers", "total_buyers"],
            title="Repeat Rate vs Time to 2nd Purchase",
            labels={
                "median_days_to_second": "Median Days to 2nd Purchase",
                "repeat_rate": "Repeat Rate",
                "total_buyers": "Total Buyers",
            },
            color_continuous_scale="RdYlGn",
        )
        fig.add_vline(x=30, line_dash="dash", line_color="gray", annotation_text="30 days")
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50% Repeat")
        fig.update_layout(height=450)
        st.plotly_chart(fig, width="stretch")

    with col2:
        # Distribution of time to 2nd purchase
        fig = px.histogram(
            loyalty.dropna(subset=["median_days_to_second"]),
            x="median_days_to_second",
            nbins=30,
            title="Distribution: Median Time to 2nd Purchase",
            labels={"median_days_to_second": "Median Days to 2nd Purchase"},
        )
        fig.add_vline(
            x=loyalty["median_days_to_second"].median(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: {loyalty['median_days_to_second'].median():.0f}d",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, width="stretch")

    # Table
    st.subheader("Product Loyalty Details")
    display_cols = [
        "product_name",
        "total_buyers",
        "repeat_buyers",
        "repeat_rate",
        "median_days_to_second",
        "p25_days_to_second",
        "p75_days_to_second",
    ]
    display_cols = [c for c in display_cols if c in loyalty.columns]

    st.dataframe(
        loyalty[display_cols]
        .style.format(
            {
                "repeat_rate": "{:.1%}",
                "median_days_to_second": "{:.0f}",
                "p25_days_to_second": "{:.0f}",
                "p75_days_to_second": "{:.0f}",
            }
        ).background_gradient(cmap="RdYlGn", subset=["repeat_rate"]),
        width="stretch",
    )

    render_analytics_export(loyalty, "Product_Loyalty_Metrics")


def render_switching_uplift(transactions_df: pd.DataFrame, product_lookup: dict):
    """Render switching gain/loss and basket uplift metrics."""
    st.subheader("Switching Gain/Loss & Basket Uplift")

    tab1, tab2 = st.tabs([" Switching Gain/Loss", " Basket Uplift"])

    with tab1:
        st.subheader("Switching Gain/Loss")
        st.caption("Gain = customers switching TO this product | Loss = customers switching FROM this product")

        with st.spinner("Computing switching metrics..."):
            switching = compute_switching_gain_loss(transactions_df)
            switching["product_name"] = switching["stockcode"].map(product_lookup)

        if switching.empty:
            st.warning("No switching data available")
        else:
            # KPI tiles
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Gain Customers", f"{switching['gain_customers'].sum():,}")
            with col2:
                st.metric("Total Loss Customers", f"{switching['loss_customers'].sum():,}")
            with col3:
                st.metric("Avg Net Gain Rate", f"{switching['gain_rate'].mean():.1%}")
            with col4:
                st.metric("Products with Net Gain", f"{(switching['net_gain'] > 0).sum()}")

            # Scatter: Gain vs Loss
            fig = px.scatter(
                switching,
                x="loss_customers",
                y="gain_customers",
                size="total_customers",
                color="net_gain",
                hover_data=["product_name", "gain_customers", "loss_customers", "net_gain", "gain_rate", "loss_rate"],
                title="Switching Gain vs Loss",
                labels={
                    "loss_customers": "Customers Switching FROM",
                    "gain_customers": "Customers Switching TO",
                    "net_gain": "Net Gain",
                },
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
            )
            fig.add_shape(
                type="line",
                x0=0, y0=0, x1=switching["loss_customers"].max(), y1=switching["gain_customers"].max(),
                line=dict(dash="dash", color="gray"),
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, width="stretch")

            # Table
            display_cols = ["product_name", "gain_customers", "loss_customers", "net_gain", "gain_rate", "loss_rate"]
            st.dataframe(
                switching[display_cols]
                .style.format(
                    {
                        "gain_rate": "{:.1%}",
                        "loss_rate": "{:.1%}",
                    }
                ).background_gradient(cmap="RdYlGn", subset=["net_gain", "gain_rate"]),
                width="stretch",
            )

            render_analytics_export(switching, "Switching_Gain_Loss")

    with tab2:
        st.subheader("Basket Value Uplift")
        st.caption("Halo effect: avg basket value WHEN product is present vs WHEN absent")

        with st.spinner("Computing basket uplift..."):
            uplift = compute_basket_uplift(transactions_df)
            uplift["product_name"] = uplift["stockcode"].map(product_lookup)

        if uplift.empty:
            st.warning("No uplift data available")
        else:
            # KPI tiles
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Uplift", f"${uplift['basket_value_uplift'].mean():.2f}")
            with col2:
                st.metric("Avg Uplift %", f"{uplift['basket_value_uplift_pct'].mean():.1f}%")
            with col3:
                st.metric("Products with Positive Uplift", f"{(uplift['basket_value_uplift'] > 0).sum()}")
            with col4:
                st.metric("Max Uplift", f"${uplift['basket_value_uplift'].max():.2f}")

            # Top uplift products
            st.subheader("Top Basket Uplift Products")
            top_uplift = uplift.nlargest(20, "basket_value_uplift_pct")

            fig = px.bar(
                top_uplift,
                x="basket_value_uplift_pct",
                y="product_name",
                orientation="h",
                color="basket_value_uplift_pct",
                title="Top Products by Basket Value Uplift %",
                labels={"basket_value_uplift_pct": "Uplift %", "product_name": "Product"},
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
            st.plotly_chart(fig, width="stretch")

            # Table
            display_cols = ["product_name", "baskets_with", "avg_basket_value_with", "avg_basket_value_without", "basket_value_uplift", "basket_value_uplift_pct"]
            display_cols = [c for c in display_cols if c in uplift.columns]

            st.dataframe(
                top_uplift[display_cols]
                .style.format(
                    {
                        "avg_basket_value_with": "${:.2f}",
                        "avg_basket_value_without": "${:.2f}",
                        "basket_value_uplift": "${:.2f}",
                        "basket_value_uplift_pct": "{:.1f}%",
                    }
                ).background_gradient(cmap="RdYlGn", subset=["basket_value_uplift_pct"]),
                width="stretch",
            )

            render_analytics_export(uplift, "Basket_Uplift")


def render_lifecycle_analysis(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render product lifecycle analysis."""
    st.subheader("Product Lifecycle Analysis")

    top_n = st.slider("Top N Products by Revenue", 10, 200, 50, key="lifecycle_tab_top_n")

    with st.spinner("Analyzing product lifecycles..."):
        product_metrics = compute_product_metrics(transactions_df)
        lifecycle_df = product_lifecycle_stage(product_metrics, transactions_df, period="M")

    if lifecycle_df.empty:
        st.warning("Insufficient data for lifecycle analysis")
        return

    lifecycle_df = lifecycle_df.nlargest(top_n, "total_revenue")
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
            color="Stage",
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
            labels={"stage": "Stage", "total_revenue": "Revenue ($)", "products": "Products"},
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
        ).background_gradient(cmap="RdYlGn", subset=["growth_rate"]),
        width="stretch",
    )

    render_analytics_export(lifecycle_df, "Product_Lifecycle_Analysis")


def render_seasonality_analysis(transactions_df: pd.DataFrame, product_lookup: dict, params: dict):
    """Render product seasonality analysis."""
    st.subheader("Product Seasonality Analysis")

    products = transactions_df["stockcode"].unique()
    selected_product = st.selectbox(
        "Select Product for Seasonality Analysis",
        products,
        format_func=lambda x: product_lookup.get(x, x),
        key="seasonality_tab_product",
    )

    if not selected_product:
        st.info("Select a product to analyze")
        return

    with st.spinner("Analyzing seasonality..."):
        from src.analytics.product_performance import product_seasonality

        seasonality = product_seasonality(transactions_df, selected_product)

    if not seasonality.get("has_seasonality", False):
        st.info(f"Seasonality not detected: {seasonality.get('message', 'Insufficient pattern')}")

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

            month_names = {
                1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
            }
            month_data["Month_Name"] = month_data["Month"].map(month_names)

            fig = px.bar(
                month_data,
                x="Month_Name",
                y="Avg Quantity",
                title=f"Monthly Pattern for {product_lookup.get(selected_product, selected_product)}",
                color="Avg Quantity",
                color_continuous_scale="RdYlBu",
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
        st.metric("Peak-to-Trough Ratio", f"{seasonality.get('peak_to_trough_ratio', 0):.1f}x")

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
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
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

    # Interpretation
    st.subheader("Interpretation")
    if seasonality["seasonal_strength"] > 0.6:
        st.success(
            f" **Strong seasonality detected!** Peak demand in month {seasonality['peak_month']} "
            f"({['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][seasonality['peak_month'] - 1]}). "
            f"Plan inventory and promotions accordingly."
        )
    elif seasonality["seasonal_strength"] > 0.3:
        st.info(
            f" **Moderate seasonality.** Peak in month {seasonality['peak_month']}. "
            f"Consider seasonal adjustments."
        )
    else:
        st.info(" **Weak seasonality.** Demand relatively stable across months.")


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

            affinity = product_affinity_score(transactions_df, selected_product, min_support=min_support)

        if affinity.empty:
            st.warning(f"No strong affinities found for {product_lookup.get(selected_product, selected_product)}")
        else:
            st.subheader(f"Top Affinities for {product_lookup.get(selected_product, selected_product)}")

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
                ).background_gradient(cmap="RdYlGn", subset=["lift", "confidence"]),
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
                labels={"lift": "Lift", "confidence": "Confidence", "product_name": "Product"},
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
        fig.update_layout(height=700, xaxis_tickangle=45)
        st.plotly_chart(fig, width="stretch")

        # Top cross-sell pairs
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
                pairs_df.style.format({"Lift": "{:.2f}"}).background_gradient(cmap="RdYlGn", subset=["Lift"]),
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

    if not selected_product:
        st.info("Select a product to analyze")
        return

    min_periods = st.slider("Min Weekly Periods", 5, 50, 10, key="elasticity_tab_min_periods")

    with st.spinner("Estimating price elasticity..."):
        from src.analytics.product_performance import price_elasticity_analysis

        elasticity = price_elasticity_analysis(transactions_df, selected_product, min_periods=min_periods)

    if elasticity.get("elasticity") is None:
        st.warning(f"Cannot estimate elasticity: {elasticity.get('message', 'Insufficient data')}")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Elasticity", f"{elasticity['elasticity']:.3f}")
    with col2:
        st.metric("R²", f"{elasticity['r_squared']:.3f}")
    with col3:
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
        import numpy as np

        fig = px.scatter(
            weekly,
            x="avg_price",
            y="total_qty",
            title=f"Price vs Quantity (Weekly): {product_lookup.get(selected_product, selected_product)}",
            labels={"avg_price": "Average Price ($)", "total_qty": "Total Quantity"},
        )
        slope, intercept = np.polyfit(weekly["avg_price"], weekly["total_qty"], 1)
        x_range = np.linspace(weekly["avg_price"].min(), weekly["avg_price"].max(), 2)
        fig.add_scatter(
            x=x_range,
            y=slope * x_range + intercept,
            mode="lines",
            name=f"Trend (slope={slope:.2f})",
            line=dict(dash="dash", color="red"),
        )
        st.plotly_chart(fig, width="stretch")

        # Elasticity interpretation
        if elasticity["elasticity"] < -1:
            st.success(
                " **Pricing Opportunity**: Demand is elastic. Price reductions could increase revenue."
            )
        elif elasticity["elasticity"] < -0.05:
            st.info(
                " **Stable Pricing**: Demand is inelastic. Price increases may increase revenue without losing much volume."
            )
        else:
            st.warning(
                " **Unusual Pattern**: Positive elasticity detected. Investigate data quality."
            )


def render_performance_comparison(
    transactions_df: pd.DataFrame, product_lookup: dict, params: dict
):
    """Render product performance comparison."""
    st.subheader("Product Performance Comparison")

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
    all_metrics = compute_product_dashboard_metrics(transactions_df)
    comparison = all_metrics[all_metrics["stockcode"].isin(selected_products)].copy()
    comparison["product_name"] = comparison["stockcode"].map(product_lookup)

    if comparison.empty:
        st.warning("No metrics available for selected products")
        return

    # Metric selection
    available_metrics = [
        c for c in [
            "total_revenue",
            "total_quantity",
            "avg_price",
            "basket_penetration",
            "unique_shopper_penetration",
            "repeat_rate",
            "velocity",
            "revenue_per_customer",
            "unique_customers",
            "price_index",
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
            m for m in [
                "total_revenue",
                "total_quantity",
                "avg_price",
                "basket_penetration",
                "repeat_rate",
                "velocity",
            ] if m in comparison.columns
        ]

        if radar_metrics:
            # Normalize
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
                "basket_penetration": "{:.1%}",
                "unique_shopper_penetration": "{:.1%}",
                "repeat_rate": "{:.1%}",
                "velocity": "{:.1f}",
                "revenue_per_customer": "${:,.2f}",
                "unique_customers": "{:,.0f}",
                "price_index": "{:.2f}",
            }
        ).background_gradient(cmap="RdYlGn"),
        width="stretch",
    )

    render_analytics_export(comparison, "Product_Comparison")
