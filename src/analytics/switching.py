"""Product switching analysis.

Tracks consecutive purchases per customer: when a customer buys product B in
a later transaction after having bought A earlier (within a window), that is
a switch A -> B.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.schemas import CATEGORY_SWITCHING, LOYALTY_METRICS, SWITCHING_MATRIX, check


def _customer_sequences(
    df: pd.DataFrame,
    window_days: int,
    min_transactions: int,
) -> pd.DataFrame:
    df = df.sort_values(["customer_id", "date", "transaction_id"])
    seq = (
        df.groupby(["customer_id", "transaction_id"])
        .agg(
            date=("date", "first"),
            products=("stockcode", lambda s: ",".join(sorted(set(s)))),
        )
        .reset_index()
    )
    seq["prev_date"] = seq.groupby("customer_id")["date"].shift(1)
    seq["prev_products"] = seq.groupby("customer_id")["products"].shift(1)
    seq = seq.dropna(subset=["prev_products"])
    seq["gap_days"] = (seq["date"] - seq["prev_date"]).dt.days
    seq = seq[seq["gap_days"].le(window_days)]
    counts = seq.groupby("customer_id").size().reset_index(name="n")
    keep = counts[counts["n"].ge(min_transactions - 1)]["customer_id"]
    return seq[seq["customer_id"].isin(keep)]


def compute_switching_matrix(
    df: pd.DataFrame,
    window_days: int = 90,
    min_transactions: int = 3,
) -> pd.DataFrame:
    """Transition counts between consecutive purchases: A -> B."""
    seq = _customer_sequences(df, window_days, min_transactions)
    if seq.empty:
        return check(pd.DataFrame(columns=list(SWITCHING_MATRIX.columns)), SWITCHING_MATRIX, allow_empty=True)

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
        return check(pd.DataFrame(columns=list(SWITCHING_MATRIX.columns)), SWITCHING_MATRIX, allow_empty=True)

    matrix = (
        switched.groupby(["from_product", "to_product"])
        .size()
        .reset_index(name="count")
    )
    matrix["pct"] = matrix["count"] / matrix["count"].sum()
    return check(matrix, SWITCHING_MATRIX)


def compute_transition_matrix(df: pd.DataFrame, window_days: int = 90, min_transactions: int = 3) -> pd.DataFrame:
    """Square from->to transition probability matrix (rows sum to 1)."""
    matrix = compute_switching_matrix(df, window_days, min_transactions)
    if matrix.empty:
        return pd.DataFrame()
    pivot = matrix.pivot(index="from_product", columns="to_product", values="count").fillna(0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    return pivot


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
            {"start_date": pd.Timestamp(e.get("start_date")), "end_date": pd.Timestamp(e.get("end_date", e.get("start_date")))}
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
        return check(pd.DataFrame(columns=list(CATEGORY_SWITCHING.columns)), CATEGORY_SWITCHING, allow_empty=True)

    if product_lookup is not None and not product_lookup.empty:
        cat_map = product_lookup.set_index("stockcode")["category"].to_dict()
    else:
        cat_map = df.groupby("stockcode")["category"].first().to_dict() if "category" in df.columns else {}

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
        return check(pd.DataFrame(columns=list(SWITCHING_MATRIX.columns)), SWITCHING_MATRIX, allow_empty=True)
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
        metrics["switching_count"] / (metrics["n_transactions"] - 1)
    ).clip(upper=1.0).fillna(0.0)
    metrics = metrics.reset_index()
    metrics = metrics.fillna(
        {"repeat_purchase_rate": 0.0, "n_distinct_products": 1, "switching_count": 0}
    )
    return check(metrics, LOYALTY_METRICS)
