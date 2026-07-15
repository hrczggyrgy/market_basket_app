"""Product switching / brand switching analysis."""

import numpy as np
import pandas as pd



def compute_switching_matrix(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    customer_col: str = "customer_id",
    date_col: str = "date",
    window_days: int = 90,
    min_transactions: int = 2,
) -> pd.DataFrame:
    """
    Compute product-to-product switching matrix.

    Returns:
        DataFrame with columns: from_product, to_product, switch_count, switch_rate
        plus asymmetry-oriented metrics useful in directed switching analysis.
    """
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Filter customers with minimum transactions
    cust_counts = df.groupby(customer_col).size()
    valid_customers = cust_counts[cust_counts >= min_transactions].index
    df = df[df[customer_col].isin(valid_customers)]

    df = df.sort_values([customer_col, date_col])

    switches = []

    for customer, group in df.groupby(customer_col):
        group = group.sort_values(date_col)
        products = group[product_col].values
        dates = pd.to_datetime(group[date_col]).values
        txn_ids = group["transaction_id"].values if "transaction_id" in group.columns else None

        for i in range(len(products) - 1):
            days_diff = (pd.Timestamp(dates[i + 1]) - pd.Timestamp(dates[i])).days
            same_basket = (
                txn_ids is not None
                and txn_ids[i] is not None
                and txn_ids[i + 1] is not None
                and txn_ids[i] == txn_ids[i + 1]
            )
            if days_diff <= window_days and products[i] != products[i + 1] and not same_basket:
                switches.append(
                    {
                        "from_product": products[i],
                        "to_product": products[i + 1],
                        "customer_id": customer,
                        "days_between": days_diff,
                    }
                )

    if not switches:
        return pd.DataFrame(
            columns=[
                "from_product",
                "to_product",
                "switch_count",
                "switch_rate",
                "avg_days_between",
                "unique_customers",
                "total_switches_from",
                "reverse_count",
                "asymmetry_ratio",
                "switches_per_customer",
            ]
        )

    switch_df = pd.DataFrame(switches)

    # Aggregate
    switch_matrix = (
        switch_df.groupby(["from_product", "to_product"])
        .agg(
            switch_count=("customer_id", "count"),
            avg_days_between=("days_between", "mean"),
            unique_customers=("customer_id", "nunique"),
        )
        .reset_index()
    )

    # Add total switches from each product for rate calculation
    from_totals = switch_matrix.groupby("from_product")["switch_count"].sum().reset_index()
    from_totals.columns = ["from_product", "total_switches_from"]

    switch_matrix = switch_matrix.merge(from_totals, on="from_product")
    switch_matrix["switch_rate"] = (
        switch_matrix["switch_count"] / switch_matrix["total_switches_from"]
    )

    # Add reverse flow count to detect asymmetry / net winning direction
    reverse = switch_matrix.rename(
        columns={
            "from_product": "to_product",
            "to_product": "from_product",
            "switch_count": "reverse_count",
        }
    )[["from_product", "to_product", "reverse_count"]]

    switch_matrix = switch_matrix.merge(reverse, on=["from_product", "to_product"], how="left")
    switch_matrix["reverse_count"] = switch_matrix["reverse_count"].fillna(0)
    switch_matrix["asymmetry_ratio"] = (
        (switch_matrix["switch_count"] - switch_matrix["reverse_count"])
        / (switch_matrix["switch_count"] + switch_matrix["reverse_count"] + 1e-9)
    )
    switch_matrix["switches_per_customer"] = (
        switch_matrix["switch_count"] / switch_matrix["unique_customers"].replace(0, np.nan)
    ).fillna(0)

    # Sort by switch count
    switch_matrix = switch_matrix.sort_values("switch_count", ascending=False).reset_index(
        drop=True
    )

    return switch_matrix



def get_customer_loyalty_metrics(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    customer_col: str = "customer_id",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Compute customer loyalty metrics.

    Returns:
        DataFrame with loyalty metrics per customer
    """
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    metrics = []

    for customer, group in df.groupby(customer_col):
        group = group.sort_values(date_col)
        products = group[product_col].values
        n_purchases = len(products)

        # Unique products purchased
        unique_products = len(set(products))

        # Repeat purchase rate (same product more than once)
        product_counts = pd.Series(products).value_counts()
        repeat_products = (product_counts > 1).sum()
        repeat_rate = repeat_products / unique_products if unique_products > 0 else 0

        # Most purchased product
        top_product = product_counts.index[0] if len(product_counts) > 0 else None
        top_product_share = product_counts.iloc[0] / n_purchases if n_purchases > 0 else 0

        # Brand concentration (HHI) — how concentrated purchases are across products
        shares = product_counts.values / n_purchases
        concentration_hhi = (shares**2).sum()

        # Switching count (exclude same-basket consecutive items)
        txn_ids = group["transaction_id"].values if "transaction_id" in group.columns else None
        switches = 0
        for i in range(n_purchases - 1):
            same_basket = (
                txn_ids is not None
                and txn_ids[i] is not None
                and txn_ids[i + 1] is not None
                and txn_ids[i] == txn_ids[i + 1]
            )
            if products[i] != products[i + 1] and not same_basket:
                switches += 1

        # Purchase frequency
        days_span = (group[date_col].max() - group[date_col].min()).days
        lifespan_days = max(days_span, 1)
        freq = n_purchases / lifespan_days * 30  # per month

        metrics.append(
            {
                "customer_id": customer,
                "transaction_count": n_purchases,
                "unique_products": unique_products,
                "repeat_rate": repeat_rate,
                "favorite_product": top_product,
                "favorite_share": top_product_share,
                "concentration_hhi": concentration_hhi,
                "switch_count": switches,
                "switch_rate": (switches / (n_purchases - 1) if n_purchases > 1 else 0),
                "purchase_frequency_per_month": freq,
                "customer_lifespan_days": lifespan_days,
            }
        )

    loyalty_df = pd.DataFrame(metrics)

    # Derive loyalty segments
    repeat_quantiles = loyalty_df["repeat_rate"].quantile([0.33, 0.66])
    freq_quantiles = loyalty_df["purchase_frequency_per_month"].quantile([0.33, 0.66])

    repeat_66 = repeat_quantiles.iloc[1] if len(repeat_quantiles) > 1 else 0.5
    repeat_33 = repeat_quantiles.iloc[0] if len(repeat_quantiles) > 0 else 0.25
    freq_66 = freq_quantiles.iloc[1] if len(freq_quantiles) > 1 else 1.0
    freq_33 = freq_quantiles.iloc[0] if len(freq_quantiles) > 0 else 0.5

    conditions = [
        (loyalty_df["repeat_rate"] >= repeat_66)
        & (loyalty_df["purchase_frequency_per_month"] >= freq_66),
        (loyalty_df["repeat_rate"] >= repeat_33)
        | (loyalty_df["purchase_frequency_per_month"] >= freq_33),
        (loyalty_df["repeat_rate"] == 0) & (loyalty_df["transaction_count"] == 1),
    ]
    choices = ["Loyal", "Regular", "New"]
    loyalty_df["loyalty_segment"] = np.select(conditions, choices, default="At Risk")

    return loyalty_df



def compute_transition_matrix(
    transactions_df: pd.DataFrame,
    product_col: str = "stockcode",
    customer_col: str = "customer_id",
    date_col: str = "date",
    top_n: int = 15,
) -> pd.DataFrame:
    """Compute a simple first-order Markov transition matrix across top products."""
    if transactions_df.empty or product_col not in transactions_df.columns:
        return pd.DataFrame()

    top_products = transactions_df[product_col].value_counts().head(top_n).index
    df = transactions_df[transactions_df[product_col].isin(top_products)].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([customer_col, date_col])
    df["next_product"] = df.groupby(customer_col)[product_col].shift(-1)
    df = df[df["next_product"].notna()]
    df = df[df["next_product"].isin(top_products)]

    if df.empty:
        return pd.DataFrame()

    transitions = df.groupby([product_col, "next_product"]).size().unstack(fill_value=0)
    row_sums = transitions.sum(axis=1).replace(0, np.nan)
    transition_probs = transitions.div(row_sums, axis=0).fillna(0)

    return transition_probs



def detect_brand_switching(
    transactions_df: pd.DataFrame,
    brand_col: str = "brand",
    category_col: str = "category",
    customer_col: str = "customer_id",
    date_col: str = "date",
    window_days: int = 90,
) -> pd.DataFrame:
    """
    Detect brand switching within the same category.

    Returns:
        DataFrame with brand switching events
    """
    required_cols = [brand_col, category_col]
    if not all(c in transactions_df.columns for c in required_cols):
        return pd.DataFrame(
            columns=[
                "customer_id",
                "category",
                "from_brand",
                "to_brand",
                "from_date",
                "to_date",
                "days_between",
            ]
        )

    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([customer_col, date_col])

    # Vectorised brand switching detection
    df["next_date"] = df.groupby(customer_col)[date_col].shift(-1)
    df["next_brand"] = df.groupby(customer_col)[brand_col].shift(-1)
    df["next_category"] = df.groupby(customer_col)[category_col].shift(-1)
    df["days_between"] = (df["next_date"] - df[date_col]).dt.days

    switched = (
        (df["days_between"] <= window_days)
        & (df[category_col] == df["next_category"])
        & (df[brand_col] != df["next_brand"])
        & df["next_date"].notna()
    )

    result = df[switched].copy()
    if result.empty:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "category",
                "from_brand",
                "to_brand",
                "from_date",
                "to_date",
                "days_between",
            ]
        )

    result = result.rename(
        columns={
            customer_col: "customer_id",
            category_col: "category",
            brand_col: "from_brand",
            "next_brand": "to_brand",
            date_col: "from_date",
            "next_date": "to_date",
        }
    )
    return result[
        [
            "customer_id",
            "category",
            "from_brand",
            "to_brand",
            "from_date",
            "to_date",
            "days_between",
        ]
    ]



def get_top_switching_paths(
    transactions_df: pd.DataFrame, min_switches: int = 5, top_n: int = 20
) -> pd.DataFrame:
    """Get top product-to-product switching paths."""
    switch_matrix = compute_switching_matrix(transactions_df)

    if switch_matrix.empty:
        return switch_matrix

    filtered = switch_matrix[switch_matrix["switch_count"] >= min_switches]
    return filtered.head(top_n)



def get_switching_heatmap_data(
    transactions_df: pd.DataFrame, top_n_products: int = 30
) -> pd.DataFrame:
    """Get matrix data for switching heatmap visualization."""
    switch_matrix = compute_switching_matrix(transactions_df)

    if switch_matrix.empty:
        return pd.DataFrame()

    # Filter to top products by total switch involvement
    from_counts = switch_matrix.groupby("from_product")["switch_count"].sum()
    to_counts = switch_matrix.groupby("to_product")["switch_count"].sum()
    total_counts = from_counts.add(to_counts, fill_value=0)
    top_products = total_counts.nlargest(top_n_products).index.tolist()

    switch_matrix = switch_matrix[
        switch_matrix["from_product"].isin(top_products)
        & switch_matrix["to_product"].isin(top_products)
    ]

    # Pivot to matrix
    matrix = switch_matrix.pivot(
        index="from_product", columns="to_product", values="switch_count"
    ).fillna(0)

    return matrix
