"""Observed Product Switching Analysis.

Tracks consecutive purchases per customer: when a customer buys product B in
a later transaction after having bought A earlier (within a window), that is
an observed switch A -> B.

WARNING: This is an OBSERVED correlation, NOT a causal counterfactual estimate.
It assumes switching behavior is invariant to delisting (no strategic response).

Estimability Status:
- estimated:                 usable switching patterns available.
- insufficient_customers:    fewer than min_customers with switching behavior.
- insufficient_transitions:  fewer than min_transitions observed.
- insufficient_observations:  insufficient transaction history.
- no_switching_observed:     no switching patterns detected (not an error).
- unavailable:               no switching data available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.schemas import (
    CATEGORY_SWITCHING,
    LOYALTY_METRICS,
    SWITCHING_MATRIX,
    SWITCHING_STATUS,
    check,
)


def _customer_sequences(
    df: pd.DataFrame,
    window_days: int,
    min_transactions: int,
    seasonal_adjustment: bool = False,
) -> pd.DataFrame:
    """Extract customer purchase sequences with optional seasonal adjustment.

    Args:
        df: Transaction data
        window_days: Maximum gap between consecutive purchases to consider a switch
        min_transactions: Minimum transactions per customer to include
        seasonal_adjustment: If True, adjusts window based on day-of-week patterns
    """
    df = df.sort_values(["customer_id", "date", "transaction_id"])

    # Add seasonal features if adjustment is enabled
    if seasonal_adjustment:
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"] >= 5
        # Weekend purchases get longer windows (people shop less frequently on weekends)
        df["seasonal_window"] = window_days * (1.5 if df["is_weekend"].any() else 1.0)

    seq = (
        df.groupby(["customer_id", "transaction_id"])
        .agg(
            date=("date", "first"),
            products=("stockcode", lambda s: ",".join(sorted(set(s)))),
        )
        .reset_index()
    )
    # Sort by customer_id and date to ensure correct sequence order within each customer
    seq = seq.sort_values(["customer_id", "date"]).reset_index(drop=True)
    if seasonal_adjustment:
        seq["is_weekend"] = seq.groupby("customer_id")["is_weekend"].transform("first")
    else:
        seq["is_weekend"] = False
    seq["prev_date"] = seq.groupby("customer_id")["date"].shift(1)
    seq["prev_products"] = seq.groupby("customer_id")["products"].shift(1)
    seq = seq.dropna(subset=["prev_products"])
    seq["gap_days"] = (seq["date"] - seq["prev_date"]).dt.days

    # Apply seasonal window adjustment if enabled
    if seasonal_adjustment:
        seq["adjusted_window"] = window_days * seq["is_weekend"].apply(lambda x: 1.5 if x else 1.0)
        seq = seq[seq["gap_days"].le(seq["adjusted_window"])]
    else:
        seq = seq[seq["gap_days"].le(window_days)]

    counts = seq.groupby("customer_id").size().reset_index(name="n")
    keep = counts[counts["n"].ge(min_transactions - 1)]["customer_id"]
    return seq[seq["customer_id"].isin(keep)]


def compute_switching_matrix(
    df: pd.DataFrame,
    window_days: int = 90,
    min_transactions: int = 3,
    seasonal_adjustment: bool = False,
    switching_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Observed transition counts between consecutive purchases: A -> B.

    WARNING: This is an OBSERVED correlation, NOT a causal counterfactual estimate.
    It assumes switching behavior is invariant to delisting (no strategic response).

    Args:
        df: Transaction data
        window_days: Maximum gap between consecutive purchases to consider a switch
        min_transactions: Minimum transactions per customer to include
        seasonal_adjustment: If True, adjusts window based on day-of-week patterns
        switching_df: Precomputed switching matrix with columns ['from_product', 'to_product', 'count'].
                      If provided, the function will normalize this matrix and return it.
                      If None, the switching matrix is computed from df.

    Returns:
        DataFrame with from_product, to_product, count, pct columns.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2024-01-01', periods=20, freq='W'),
        ...     'customer_id': ['C1'] * 10 + ['C2'] * 10,
        ...     'stockcode': ['A', 'B'] * 10,
        ...     'transaction_id': [f'T{i}' for i in range(20)],
        ... })
        >>> matrix = compute_switching_matrix(df, window_days=90, min_transactions=2)
        >>> 'from_product' in matrix.columns
        True
    """
    if switching_df is not None:
        if switching_df.empty:
            return check(
                pd.DataFrame(columns=list(SWITCHING_MATRIX.columns)),
                SWITCHING_MATRIX,
                allow_empty=True,
            )
        # Aggregate the switching matrix by from_product and to_product, summing counts
        matrix = switching_df.groupby(["from_product", "to_product"], as_index=False)["count"].sum()
        # Normalize the provided switching matrix
        matrix["pct"] = matrix.groupby("from_product")["count"].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 0
        )
        # When switching_df is provided, the count column contains the normalized values
        matrix["count"] = matrix["pct"]
        return check(matrix, SWITCHING_MATRIX)

    seq = _customer_sequences(df, window_days, min_transactions, seasonal_adjustment)
    if seq.empty:
        return check(
            pd.DataFrame(columns=list(SWITCHING_MATRIX.columns)), SWITCHING_MATRIX, allow_empty=True
        )

    # Vectorized: expand products into lists, then create all cross pairs
    seq = seq.copy()
    seq["prev_list"] = seq["prev_products"].str.split(",")
    seq["cur_list"] = seq["products"].str.split(",")

    # Explode to get all from->to combinations per row
    exploded = seq.explode("prev_list").explode("cur_list")
    exploded = exploded.rename(columns={"prev_list": "from_product", "cur_list": "to_product"})

    # Keep only actual switches (different products)
    switched = exploded[exploded["from_product"] != exploded["to_product"]]

    if switched.empty:
        return check(
            pd.DataFrame(columns=list(SWITCHING_MATRIX.columns)), SWITCHING_MATRIX, allow_empty=True
        )

    matrix = switched.groupby(["from_product", "to_product"]).size().reset_index(name="count")
    matrix["pct"] = matrix.groupby("from_product")["count"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0
    )
    return check(matrix, SWITCHING_MATRIX)


def compute_transition_matrix(
    df: pd.DataFrame,
    window_days: int = 90,
    min_transactions: int = 3,
    normalize: bool = True,
    switching_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Square from->to observed transition probability matrix (rows sum to 1).

    WARNING: This is an OBSERVED correlation, NOT a causal counterfactual estimate.
    It assumes switching behavior is invariant to delisting (no strategic response).

    If normalize=True (default), returns P(to | from) - probability of switching to B given A was switched from.
    If normalize=False, returns raw counts.

    Adds an absorbing "no_switch" state representing customers who don't switch away.
    Args:
        df: Transaction data
        window_days: Maximum gap between consecutive purchases to consider a switch
        min_transactions: Minimum transactions per customer to include
        normalize: If True, returns P(to | from) - probability of switching to B given A was switched from.
                  If False, returns raw counts.
        switching_df: Precomputed switching matrix with columns ['from_product', 'to_product', 'count'].
                      If provided, the function will use this matrix (and normalize it if normalize=True).
                      If None, the switching matrix is computed from df.
    """
    matrix = compute_switching_matrix(df, window_days, min_transactions, switching_df=switching_df)
    if matrix.empty:
        return pd.DataFrame()
    pivot = matrix.pivot(index="from_product", columns="to_product", values="count").fillna(0)

    if normalize:
        # Add "no_switch" absorbing state - represents probability of not switching
        # For customers who switch away from A, P(to | from) is given by normalized counts
        # We add an explicit "no_switch" column to represent staying with the same product
        row_sums = pivot.sum(axis=1)
        # Normalize to get P(to | from switched)
        pivot = pivot.div(row_sums.replace(0, 1), axis=0).fillna(0)
        # Add absorbing state for "no switch" - probability of not switching
        pivot["no_switch"] = 1.0 - pivot.sum(axis=1)
        pivot["no_switch"] = pivot["no_switch"].clip(lower=0)

    return pivot


# def build_event_slices(
#     df: pd.DataFrame,
def build_event_slices(
    df: pd.DataFrame,
    events: pd.DataFrame | list[dict],
    pre_days: int = 30,
    post_days: int = 30,
) -> dict[str, pd.DataFrame]:
    """Slice a transaction frame into pre-event / event / post-event windows.

    "pre"   = [event start - pre_days, event start)   (exclusive of start)
    "event" = [event start, event end]                (inclusive)
    "post"  = (event end, event end + post_days]      (exclusive of end)

    ``events`` may be a DataFrame with ``start_date`` / ``end_date`` columns
    (e.g. PROMO_PERIODS) or a list of dicts with the same keys. Slices are
    clamped to the frame's date range; empty slices are omitted from the dict.

    Returns:
        Dict phase -> sliced dataframe ("pre", "event", "post" when non-empty).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if isinstance(events, pd.DataFrame):
        event_rows = events[["start_date", "end_date"]].to_dict("records")
    else:
        event_rows = [
            {
                "start_date": pd.Timestamp(e.get("start_date")),
                "end_date": pd.Timestamp(e.get("end_date", e.get("start_date"))),
            }
            for e in events
        ]
    if not event_rows:
        return {}

    slices: dict[str, pd.DataFrame] = {}
    for label in ("pre", "event", "post"):
        mask = pd.Series(False, index=df.index)
        for ev in event_rows:
            start = pd.Timestamp(ev["start_date"])
            end = pd.Timestamp(ev["end_date"])
            if start > end:
                start, end = end, start
            if label == "pre":
                lo, hi = start - pd.Timedelta(days=pre_days), start - pd.Timedelta(days=1)
            elif label == "event":
                lo, hi = start, end
            else:
                lo, hi = end + pd.Timedelta(days=1), end + pd.Timedelta(days=post_days)
            mask |= df["date"].between(lo, hi)
        sliced = df[mask]
        if not sliced.empty:
            slices[label] = sliced
    return slices


def compute_category_switching_by_phase(
    df: pd.DataFrame,
    events: dict,
    pre_days: int = 30,
    post_days: int = 30,
    window_days: int = 90,
    min_transactions: int = 3,
    product_lookup: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Category-level switching per time phase (pre / event / post).

    Reuses the same product->category rollup as
    ``compute_category_switching_matrix`` on each sliced frame. Produces a
    dict keyed by phase, each validated against CATEGORY_SWITCHING (an empty
    DataFrame when a phase slice has no switching).

    Returns:
        Dict phase -> dataframe (contract-validated, may be empty).
    """
    slices = build_event_slices(df, events, pre_days=pre_days, post_days=post_days)
    out: dict[str, pd.DataFrame] = {}
    for label, sliced in slices.items():
        cat_matrix = compute_category_switching_matrix(
            sliced,
            window_days=window_days,
            min_transactions=min_transactions,
            product_lookup=product_lookup,
        )
        out[label] = cat_matrix
    return out


def compute_category_switching_matrix(
    df: pd.DataFrame,
    window_days: int = 90,
    min_transactions: int = 3,
    product_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate product-level switching to category-level transitions.

    Maps each SKU to its category (via ``product_lookup`` when supplied, else
    the ``category`` column) and rolls product->product switches up to
    category->category counts. Also records how many distinct product pairs
    contribute to each category transition.

    Returns:
        DataFrame validated against CATEGORY_SWITCHING.
    """
    matrix = compute_switching_matrix(df, window_days, min_transactions)
    if matrix.empty:
        return check(
            pd.DataFrame(columns=list(CATEGORY_SWITCHING.columns)),
            CATEGORY_SWITCHING,
            allow_empty=True,
        )

    if product_lookup is not None and not product_lookup.empty:
        cat_map = product_lookup.set_index("stockcode")["category"].to_dict()
    else:
        cat_map = (
            df.groupby("stockcode")["category"].first().to_dict()
            if "category" in df.columns
            else {}
        )

    def cat(item: str) -> str:
        return str(cat_map.get(item, "Unknown"))

    cat_matrix = matrix.copy()
    cat_matrix["from_category"] = cat_matrix["from_product"].map(cat)
    cat_matrix["to_category"] = cat_matrix["to_product"].map(cat)

    # Drop same-category self-switches (no category movement), but keep a
    # distinct-product-pair count for intra-category fidelity reporting.
    agg = (
        cat_matrix.groupby(["from_category", "to_category"])
        .agg(count=("count", "sum"), product_pairs=("count", "size"))
        .reset_index()
    )
    total = agg["count"].sum()
    agg["pct"] = agg["count"] / total if total > 0 else 0.0
    agg = agg.sort_values("count", ascending=False).reset_index(drop=True)
    return check(agg, CATEGORY_SWITCHING)


def get_top_switching_paths(
    df: pd.DataFrame, top_n: int = 20, window_days: int = 90, min_transactions: int = 3
) -> pd.DataFrame:
    """Most common switching paths."""
    matrix = compute_switching_matrix(df, window_days, min_transactions)
    if matrix.empty:
        return check(
            pd.DataFrame(columns=list(SWITCHING_MATRIX.columns)), SWITCHING_MATRIX, allow_empty=True
        )
    top = matrix.sort_values("count", ascending=False).head(top_n).reset_index(drop=True)
    return check(top, SWITCHING_MATRIX)


def get_customer_loyalty_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-customer repeat-purchase and switching behavior."""
    txn = df.groupby(["customer_id", "transaction_id"])["stockcode"].agg(set).reset_index()
    txn["basket_size"] = txn["stockcode"].map(len)
    cust_txn = txn.groupby("customer_id").agg(
        n_transactions=("transaction_id", "nunique"),
        avg_basket_size=("basket_size", "mean"),
    )
    repeat = (
        df.groupby(["customer_id", "stockcode"])
        .size()
        .reset_index(name="purchases")
        .groupby("customer_id")["purchases"]
        .apply(lambda s: (s > 1).sum() / len(s))
        .rename("repeat_purchase_rate")
    )
    n_products = df.groupby("customer_id")["stockcode"].nunique().rename("n_distinct_products")
    seq = _customer_sequences(df, 90, 2)
    if seq.empty:
        switch_count = pd.Series(0, index=df["customer_id"].unique(), name="switching_count")
    else:
        switched = seq.apply(
            lambda row: set(row["prev_products"].split(",")) != set(row["products"].split(",")),
            axis=1,
        )
        switch_count = seq.loc[switched].groupby("customer_id").size().rename("switching_count")
    metrics = cust_txn.join(repeat, how="left").join(n_products, how="left")
    metrics["switching_count"] = switch_count.reindex(metrics.index).fillna(0)
    metrics["switching_rate"] = (
        (metrics["switching_count"] / (metrics["n_transactions"] - 1)).clip(upper=1.0).fillna(0.0)
    )
    metrics = metrics.reset_index()
    metrics = metrics.fillna(
        {"repeat_purchase_rate": 0.0, "n_distinct_products": 1, "switching_count": 0}
    )
    return check(metrics, LOYALTY_METRICS)


def compute_switching_status(
    df: pd.DataFrame,
    window_days: int = 90,
    min_transactions: int = 3,
    min_customers: int = 10,
    min_transitions: int = 5,
) -> pd.DataFrame:
    """Per-product switching estimability status.

    Unlike compute_switching_matrix (which returns only products with switching),
    this returns one row per product with an explicit switching_status so callers
    can answer "why is this product missing switching data?" instead of silently
    dropping it or misreading it as "no switching occurs."

    Status values:
    - estimated:                 usable switching patterns available.
    - insufficient_customers:    fewer than min_customers with switching behavior.
    - insufficient_transitions:  fewer than min_transitions observed.
    - insufficient_observations:  insufficient transaction history.
    - no_switching_observed:     no switching patterns detected (not an error).
    - unavailable:               no switching data available.

    Args:
        df: Transaction data
        window_days: Maximum gap for switching detection
        min_transactions: Minimum transactions per customer
        min_customers: Minimum customers with switching behavior
        min_transitions: Minimum switching transitions required

    Returns:
        DataFrame with stockcode, switching_status, n_switchers, n_transitions, n_observations.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2024-01-01', periods=50, freq='W'),
        ...     'customer_id': ['C1'] * 25 + ['C2'] * 25,
        ...     'stockcode': ['A', 'B', 'C'] * 16 + ['A', 'B'] * 2,
        ...     'transaction_id': [f'T{i}' for i in range(50)],
        ... })
        >>> status = compute_switching_status(df, min_customers=2)
        >>> 'switching_status' in status.columns
        True
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Get all products
    all_products = df["stockcode"].unique()

    # Compute switching matrix
    matrix = compute_switching_matrix(df, window_days, min_transactions)

    rows = []
    for product in all_products:
        # Count observations for this product
        product_obs = df[df["stockcode"] == product]
        n_observations = len(product_obs)
        n_customers = product_obs["customer_id"].nunique()

        # Count switching transitions involving this product
        if not matrix.empty:
            out_transitions = matrix[matrix["from_product"] == product]["count"].sum()
            in_transitions = matrix[matrix["to_product"] == product]["count"].sum()
            n_transitions = int(out_transitions + in_transitions)
            n_switchers = len(
                set(matrix[matrix["from_product"] == product]["to_product"].tolist())
                | set(matrix[matrix["to_product"] == product]["from_product"].tolist())
            )
        else:
            n_transitions = 0
            n_switchers = 0

        # Determine status
        if n_observations < min_transactions:
            status = "insufficient_observations"
        elif n_customers < min_customers:
            status = "insufficient_customers"
        elif n_transitions == 0:
            status = "no_switching_observed"
        elif n_transitions < min_transitions:
            status = "insufficient_transitions"
        else:
            status = "estimated"

        rows.append(
            {
                "stockcode": product,
                "switching_status": status,
                "n_switchers": n_switchers,
                "n_transitions": n_transitions,
                "n_observations": n_observations,
                "n_customers": n_customers,
            }
        )

    status_df = pd.DataFrame(rows)
    return check(status_df, SWITCHING_STATUS)


def compute_switch_in_out_rates(
    matrix: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute switch-in and switch-out rates per product.

    Switch-out rate: proportion of a product's customers who switch away
    Switch-in rate: proportion of new customers who come from other products

    Args:
        matrix: Switching matrix from compute_switching_matrix
        df: Transaction data for customer counts

    Returns:
        DataFrame with stockcode, switch_out_rate, switch_in_rate, net_rate, n_switchers_out, n_switchers_in

    Example:
        >>> matrix = pd.DataFrame({
        ...     'from_product': ['A', 'B'],
        ...     'to_product': ['B', 'A'],
        ...     'count': [10, 5]
        ... })
        >>> rates = compute_switch_in_out_rates(matrix, df)
        >>> 'switch_out_rate' in rates.columns
        True
    """
    if matrix.empty:
        return pd.DataFrame(
            columns=[
                "stockcode",
                "switch_out_rate",
                "switch_in_rate",
                "net_rate",
                "n_switchers_out",
                "n_switchers_in",
            ]
        )

    # Get customer counts per product
    product_customers = df.groupby("stockcode")["customer_id"].nunique()

    # Count switchers out and in
    switchers_out = matrix.groupby("from_product").size().rename("n_switchers_out")
    switchers_in = matrix.groupby("to_product").size().rename("n_switchers_in")

    # Combine
    all_products = set(product_customers.index) | set(switchers_out.index) | set(switchers_in.index)

    rows = []
    for product in all_products:
        n_customers = product_customers.get(product, 0)
        n_out = switchers_out.get(product, 0)
        n_in = switchers_in.get(product, 0)

        switch_out_rate = n_out / n_customers if n_customers > 0 else 0.0
        switch_in_rate = n_in / n_customers if n_customers > 0 else 0.0
        net_rate = switch_in_rate - switch_out_rate

        rows.append(
            {
                "stockcode": product,
                "switch_out_rate": switch_out_rate,
                "switch_in_rate": switch_in_rate,
                "net_rate": net_rate,
                "n_switchers_out": n_out,
                "n_switchers_in": n_in,
            }
        )

    return pd.DataFrame(rows)


def _bootstrap_switching_matrix(
    df: pd.DataFrame,
    n_resamples: int = 200,
    window_days: int = 90,
    min_transactions: int = 3,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap confidence intervals for switching rates, resampling at customer level."""
    rng = np.random.default_rng(random_seed)
    customers = df["customer_id"].unique()
    cust_groups = {c: g for c, g in df.groupby("customer_id")}

    point_matrix = compute_switching_matrix(df, window_days, min_transactions)
    if point_matrix.empty:
        return pd.DataFrame()

    # For each unique from->to pair, collect bootstrap replicates
    pairs = point_matrix[["from_product", "to_product"]].drop_duplicates()
    replicates_dict = {}

    for _, pair_row in pairs.iterrows():
        frm, to = pair_row["from_product"], pair_row["to_product"]
        replicates = []

        for _ in range(n_resamples):
            # Resample customers with replacement
            cust_idx = rng.integers(0, len(customers), size=len(customers))
            frames = [cust_groups[c] for c in customers[cust_idx]]
            if not frames:
                continue
            resample = pd.concat(frames, ignore_index=True)
            if resample.empty:
                continue

            boot_matrix = compute_switching_matrix(resample, window_days, min_transactions)
            if not boot_matrix.empty:
                match = boot_matrix[
                    (boot_matrix["from_product"] == frm) & (boot_matrix["to_product"] == to)
                ]
                if not match.empty:
                    replicates.append(float(match["pct"].iloc[0]))

        if replicates:
            arr = np.asarray(replicates)
            replicates_dict[(frm, to)] = {
                "ci_lower": float(np.percentile(arr, 2.5)),
                "ci_upper": float(np.percentile(arr, 97.5)),
                "std_error": float(arr.std(ddof=1)),
                "n_resamples": len(arr),
            }

    # Merge back to point matrix
    result = point_matrix.copy()
    result["ci_lower"] = result.apply(
        lambda r: replicates_dict.get((r["from_product"], r["to_product"]), {}).get(
            "ci_lower", 0.0
        ),
        axis=1,
    )
    result["ci_upper"] = result.apply(
        lambda r: replicates_dict.get((r["from_product"], r["to_product"]), {}).get(
            "ci_upper", 0.0
        ),
        axis=1,
    )
    result["std_error"] = result.apply(
        lambda r: replicates_dict.get((r["from_product"], r["to_product"]), {}).get(
            "std_error", 0.0
        ),
        axis=1,
    )
    result["n_resamples"] = result.apply(
        lambda r: replicates_dict.get((r["from_product"], r["to_product"]), {}).get(
            "n_resamples", n_resamples
        ),
        axis=1,
    )

    return result


def compute_substitution_strength(
    demand_transference_df: pd.DataFrame,
    df: pd.DataFrame,
    sdp_df: pd.DataFrame,
    bootstrap_switching_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Classify switching patterns as movement vs true substitution vs loyalty.

    Uses switch rate confidence, revenue concentration, and SDP to determine
    whether observed switching represents true substitution behavior.

    Args:
        demand_transference_df: Output from compute_demand_transference_matrix
        df: Transaction data
        sdp_df: SDP scores from compute_substitutable_demand_percentage
        bootstrap_switching_df: optional pre-computed bootstrap switching matrix
            with columns ci_lower, ci_upper. If not provided, a bootstrap is
            computed internally (default 100 resamples).

    Returns:
        DataFrame validated against SWITCHING_SUBSTITUTION contract.
    """
    from src.analytics.schemas import SWITCHING_SUBSTITUTION, check

    if demand_transference_df.empty:
        return check(
            pd.DataFrame(columns=list(SWITCHING_SUBSTITUTION.columns)),
            SWITCHING_SUBSTITUTION,
            allow_empty=True,
        )

    # Total revenue per product
    rev = (
        df.groupby("stockcode")
        .apply(lambda x: (x["price"] * x["quantity"]).sum())
        .rename("revenue")
    )

    # Merge with SDP
    sdp_lookup = (
        sdp_df.set_index("stockcode")["sdp"] if not sdp_df.empty else pd.Series(dtype=float)
    )

    # Prepare bootstrap CI lookup if needed
    ci_lookup = {}
    if bootstrap_switching_df is None:
        # Compute bootstrap CI internally
        boot_df = _bootstrap_switching_matrix(df, n_resamples=100, random_seed=42)
        if not boot_df.empty:
            for _, r in boot_df.iterrows():
                ci_lookup[(r["from_product"], r["to_product"])] = {
                    "ci_lower": r["ci_lower"],
                    "ci_upper": r["ci_upper"],
                }
    else:
        if not bootstrap_switching_df.empty:
            for _, r in bootstrap_switching_df.iterrows():
                ci_lookup[(r["from_product"], r["to_product"])] = {
                    "ci_lower": r["ci_lower"],
                    "ci_upper": r["ci_upper"],
                }

    # Build result per from->to pair
    rows = []
    for _, row in demand_transference_df.iterrows():
        frm, to = row["from_product"], row["to_product"]
        switch_rate = row["switch_rate"]
        revenue_at_risk = row["observed_switching_transfer_revenue"]
        recovery_proxy = revenue_at_risk

        # Get SDP for from_product (substitutability of demand being lost)
        sdp_from = sdp_lookup.get(frm, 0.5)
        rev_from = rev.get(frm, 0.0)

        # Classification logic
        # Movement: low switch rate, low revenue at risk, high SDP (customers just browsing)
        # Substitution: significant switch rate, meaningful revenue, CI doesn't include 0
        # Loyalty: high revenue, low switch rate, low SDP (unique demand)

        # Use revenue share of switching
        rev_share = revenue_at_risk / rev_from if rev_from > 0 else 0.0

        # Substitution strength based on switch rate and revenue concentration
        if switch_rate >= 0.3 and rev_share >= 0.1:
            strength = "strong"
        elif switch_rate >= 0.15 and rev_share >= 0.05:
            strength = "moderate"
        elif switch_rate > 0:
            strength = "weak"
        else:
            strength = "dominant"  # This shouldn't happen but handle it

        # Classification
        if strength in ("moderate", "strong") and rev_share >= 0.1:
            classification = "substitution"
        elif strength == "weak" and sdp_from >= 0.8:
            classification = "movement"
        elif rev_share < 0.05 and sdp_from < 0.2:
            classification = "loyalty"
        else:
            classification = "insufficient_evidence"

        # Confidence based on switch rate magnitude and revenue
        if strength == "strong" and rev_share >= 0.15:
            confidence = "high"
        elif strength == "moderate" and rev_share >= 0.05:
            confidence = "medium"
        else:
            confidence = "low"

        ci_lower = ci_lookup.get((frm, to), {}).get("ci_lower", 0.0)
        ci_upper = ci_lookup.get((frm, to), {}).get("ci_upper", 0.0)

        rows.append(
            {
                "from_product": frm,
                "to_product": to,
                "switch_rate": switch_rate,
                "switch_rate_ci_lower": ci_lower,
                "switch_rate_ci_upper": ci_upper,
                "revenue_at_risk": revenue_at_risk,
                "recovery_proxy": recovery_proxy,
                "substitution_strength": strength,
                "classification": classification,
                "confidence": confidence,
            }
        )

    table = pd.DataFrame(rows, columns=list(SWITCHING_SUBSTITUTION.columns))
    return check(table, SWITCHING_SUBSTITUTION, allow_empty=True)


def compute_high_value_switching(
    df: pd.DataFrame,
    demand_transference_df: pd.DataFrame,
    clv_customer_df: pd.DataFrame,
    top_n_segments: int = 3,
) -> pd.DataFrame:
    """Analyze switching among high-CLV customers.

    Identifies which products are losing/winning the most valuable customers.

    Args:
        df: Transaction data
        demand_transference_df: Output from compute_demand_transference_matrix
        clv_customer_df: CLV_CUSTOMER output with predicted_clv and p_alive
        top_n_segments: Number of high-value segments to analyze

    Returns:
        DataFrame validated against HIGH_VALUE_SWITCHING contract.
    """
    from src.analytics.schemas import HIGH_VALUE_SWITCHING, check

    if clv_customer_df.empty or demand_transference_df.empty:
        return check(
            pd.DataFrame(columns=list(HIGH_VALUE_SWITCHING.columns)),
            HIGH_VALUE_SWITCHING,
            allow_empty=True,
        )

    # Identify high-value customers (top decile by predicted CLV)
    clv_threshold = clv_customer_df["predicted_clv"].quantile(0.9)
    high_value_customers = set(
        clv_customer_df[clv_customer_df["predicted_clv"] >= clv_threshold]["customer_id"]
    )

    if not high_value_customers:
        return check(
            pd.DataFrame(columns=list(HIGH_VALUE_SWITCHING.columns)),
            HIGH_VALUE_SWITCHING,
            allow_empty=True,
        )

    # Filter transactions to high-value customers
    hv_df = df[df["customer_id"].isin(high_value_customers)].copy()

    # Compute switching matrix for high-value customers
    hv_matrix = compute_switching_matrix(hv_df, window_days=90, min_transactions=2)
    if hv_matrix.empty:
        return check(
            pd.DataFrame(columns=list(HIGH_VALUE_SWITCHING.columns)),
            HIGH_VALUE_SWITCHING,
            allow_empty=True,
        )

    # Compute per-segment analysis if segments available
    rows = []
    if "clv_segment" in clv_customer_df.columns:
        for segment in clv_customer_df["clv_segment"].unique()[:top_n_segments]:
            seg_customers = set(
                clv_customer_df[clv_customer_df["clv_segment"] == segment]["customer_id"]
            )
            seg_hv = hv_df[hv_df["customer_id"].isin(seg_customers)]
            if seg_hv.empty:
                continue

            seg_matrix = compute_switching_matrix(seg_hv, window_days=90, min_transactions=2)
            if seg_matrix.empty:
                continue

            for _, row in seg_matrix.iterrows():
                frm, to = row["from_product"], row["to_product"]
                high_value_customers_switched = int(row["count"])
                # Revenue at risk for this segment
                seg_rev = seg_hv.groupby("stockcode").apply(
                    lambda x: (x["price"] * x["quantity"]).sum()
                )
                rev_at_risk = seg_rev.get(frm, 0.0) * (
                    row["count"]
                    / max(seg_matrix[seg_matrix["from_product"] == frm]["count"].sum(), 1)
                )

                # Switch rate within segment
                seg_total_switches = seg_matrix[seg_matrix["from_product"] == frm]["count"].sum()
                switch_rate = row["count"] / max(seg_total_switches, 1)

                # Avg CLV of switchers
                avg_clv = clv_customer_df[clv_customer_df["customer_id"].isin(seg_customers)][
                    "predicted_clv"
                ].mean()

                rows.append(
                    {
                        "from_product": frm,
                        "to_product": to,
                        "high_value_customers_switched": high_value_customers_switched,
                        "high_value_revenue_at_risk": rev_at_risk,
                        "high_value_switch_rate": switch_rate,
                        "segment": segment,
                        "avg_clv_of_switchers": float(avg_clv) if pd.notna(avg_clv) else 0.0,
                    }
                )
    else:
        # Aggregate high-value switching
        for _, row in hv_matrix.iterrows():
            frm, to = row["from_product"], row["to_product"]
            high_value_customers_switched = int(row["count"])
            hv_rev = hv_df.groupby("stockcode").apply(lambda x: (x["price"] * x["quantity"]).sum())
            rev_at_risk = hv_rev.get(frm, 0.0) * (
                row["count"] / max(hv_matrix[hv_matrix["from_product"] == frm]["count"].sum(), 1)
            )
            seg_total_switches = hv_matrix[hv_matrix["from_product"] == frm]["count"].sum()
            switch_rate = row["count"] / max(seg_total_switches, 1)
            avg_clv = clv_customer_df[clv_customer_df["customer_id"].isin(high_value_customers)][
                "predicted_clv"
            ].mean()

            rows.append(
                {
                    "from_product": frm,
                    "to_product": to,
                    "high_value_customers_switched": high_value_customers_switched,
                    "high_value_revenue_at_risk": rev_at_risk,
                    "high_value_switch_rate": switch_rate,
                    "segment": "high_value",
                    "avg_clv_of_switchers": float(avg_clv) if pd.notna(avg_clv) else 0.0,
                }
            )

    table = pd.DataFrame(rows, columns=list(HIGH_VALUE_SWITCHING.columns))
    return check(table, HIGH_VALUE_SWITCHING, allow_empty=True)


def generate_switching_opportunity_matrix(
    demand_transference_df: pd.DataFrame,
    substitution_df: pd.DataFrame,
    high_value_df: pd.DataFrame,
    sdp_df: pd.DataFrame,
    delist_impact_df: pd.DataFrame,
    revenue_by_product: pd.Series,
    top_n: int = 10,
) -> pd.DataFrame:
    """Generate prioritized switching opportunities with actions.

    Combines substitution strength, high-value customer impact, SDP, and delist impact
    into a ranked opportunity matrix.

    Args:
        demand_transference_df: DEMAND_TRANSFERENCE output
        substitution_df: SWITCHING_SUBSTITUTION output
        high_value_df: HIGH_VALUE_SWITCHING output
        sdp_df: SDP_SCORES output
        delist_impact_df: DELIST_IMPACT output
        revenue_by_product: Per-product revenue Series
        top_n: Max opportunities to return

    Returns:
        DataFrame validated against SWITCHING_OPPORTUNITY contract.
    """
    from src.analytics.schemas import SWITCHING_OPPORTUNITY, check

    if demand_transference_df.empty:
        return check(
            pd.DataFrame(columns=list(SWITCHING_OPPORTUNITY.columns)),
            SWITCHING_OPPORTUNITY,
            allow_empty=True,
        )

    # Build lookup dictionaries
    sub_lookup = {}
    if not substitution_df.empty:
        for _, row in substitution_df.iterrows():
            sub_lookup[(row["from_product"], row["to_product"])] = row

    hv_lookup = {}
    if not high_value_df.empty:
        for _, row in high_value_df.iterrows():
            hv_lookup[(row["from_product"], row["to_product"])] = row

    sdp_lookup = sdp_df.set_index("stockcode")["sdp"].to_dict() if not sdp_df.empty else {}
    delist_lookup = (
        delist_impact_df.set_index("stockcode")["net_revenue_impact"].to_dict()
        if not delist_impact_df.empty
        else {}
    )

    opportunities = []

    for _, row in demand_transference_df.iterrows():
        frm, to = row["from_product"], row["to_product"]
        revenue_at_risk = row["observed_switching_transfer_revenue"]
        recovery_proxy = revenue_at_risk

        sub_row = sub_lookup.get((frm, to))
        hv_row = hv_lookup.get((frm, to))

        classification = (
            sub_row["classification"] if sub_row is not None else "insufficient_evidence"
        )
        strength = sub_row["substitution_strength"] if sub_row is not None else "weak"
        confidence = sub_row["confidence"] if sub_row is not None else "low"

        # High-value impact
        hv_customers = hv_row["high_value_customers_switched"] if hv_row is not None else 0
        hv_revenue = hv_row["high_value_revenue_at_risk"] if hv_row is not None else 0.0

        # SDP for from product
        sdp_from = sdp_lookup.get(frm, 0.5)

        # Determine opportunity type and action
        if classification == "substitution" and strength in ("moderate", "strong"):
            if hv_customers > 0:
                opportunity_type = "protect"
                action = f"Protect {frm} from switching to {to} — {hv_customers} high-value customers at risk"
                rationale = f"Strong substitution ({strength}) with {hv_customers} high-value customers switching (€{hv_revenue:,.0f})."
            else:
                opportunity_type = "consolidate"
                action = f"Consolidate {frm} into {to} — demand naturally transfers"
                rationale = f"Customers substitute from {frm} to {to} naturally (rate: {row['switch_rate']:.0%})."
        elif classification == "movement":
            opportunity_type = "steal_share"
            action = f"Target {to} buyers with {frm} promotions — low loyalty, high browsing"
            rationale = f"Movement pattern: customers browse but don't deeply substitute (SDP={sdp_from:.0%})."
        elif classification == "loyalty":
            opportunity_type = "protect"
            action = f"Protect {frm} — unique demand driver (SDP={sdp_from:.0%})"
            rationale = f"Low switching, unique demand: losing {frm} would leak revenue."
        else:
            # Check delist impact
            delist_impact = delist_lookup.get(frm)
            if delist_impact is not None and delist_impact > 0:
                opportunity_type = "delist_candidate"
                action = f"Consider delisting {frm} — positive net impact (€{delist_impact:,.0f})"
                rationale = (
                    f"Simulated delist recovers more than it loses: net +€{delist_impact:,.0f}."
                )
            else:
                opportunity_type = "protect"
                action = f"Monitor {frm} -> {to} — insufficient evidence for action"
                rationale = "Switching observed but classification unclear."

        net_impact = recovery_proxy - revenue_at_risk

        opportunities.append(
            {
                "from_product": frm,
                "to_product": to,
                "opportunity_type": opportunity_type,
                "revenue_at_risk": revenue_at_risk,
                "recoverable_revenue": recovery_proxy,
                "net_impact": net_impact,
                "action": action,
                "confidence": confidence,
                "rationale": rationale,
            }
        )

    # Sort by revenue at risk descending, then by opportunity priority
    priority_order = {
        "protect": 0,
        "win_back": 1,
        "steal_share": 2,
        "consolidate": 3,
        "delist_candidate": 4,
    }
    opp_df = pd.DataFrame(opportunities)
    opp_df["priority"] = opp_df["opportunity_type"].map(priority_order).fillna(5)
    opp_df = opp_df.sort_values(["priority", "revenue_at_risk"], ascending=[True, False]).head(
        top_n
    )

    return check(
        opp_df[list(SWITCHING_OPPORTUNITY.columns)], SWITCHING_OPPORTUNITY, allow_empty=True
    )
