"""Co-purchase affinity analysis.

Affinity = phi coefficient (via sklearn.metrics.matthews_corrcoef, which is
exactly phi for binary vectors) between product purchase vectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.rules import create_basket_matrix
from src.analytics.schemas import AFFINITY_PAIRS, check


def _filter_df(
    df: pd.DataFrame,
    segment_col: str | None = None,
    segment_val: str | None = None,
    mission_col: str | None = None,
    mission_val: str | None = None,
) -> pd.DataFrame:
    """Filter dataframe by segment and/or mission."""
    result = df.copy()
    if segment_col and segment_val and segment_col in result.columns:
        result = result[result[segment_col] == segment_val]
    if mission_col and mission_val and mission_col in result.columns:
        result = result[result[mission_col] == mission_val]
    return result


def compute_cooccurrence_matrix(
    df: pd.DataFrame,
    top_n_products: int | None = None,
    segment_col: str | None = None,
    segment_val: str | None = None,
    mission_col: str | None = None,
    mission_val: str | None = None,
) -> pd.DataFrame:
    """Raw count of shared transactions per stockcode pair."""
    df = _filter_df(df, segment_col, segment_val, mission_col, mission_val)
    basket = create_basket_matrix(df)
    if top_n_products is not None:
        counts = basket.sum().sort_values(ascending=False)
        basket = basket[counts.head(top_n_products).index]
    matrix = basket.to_numpy(dtype=np.int64)
    cooccurrence = (matrix.T @ matrix).astype(int)
    return pd.DataFrame(cooccurrence, index=basket.columns, columns=basket.columns)


def compute_pair_trend(
    df: pd.DataFrame,
    product_a: str,
    product_b: str,
    period: str = "W",
    segment_col: str | None = None,
    segment_val: str | None = None,
    mission_col: str | None = None,
    mission_val: str | None = None,
) -> pd.DataFrame:
    """Per-period co-occurrence count for a product pair."""
    df = _filter_df(df, segment_col, segment_val, mission_col, mission_val)
    both = df[df["stockcode"].isin([product_a, product_b])]
    if both.empty:
        return pd.DataFrame(columns=["period", "cooccurrence"])
    pivot = both.pivot_table(
        index="transaction_id",
        columns="stockcode",
        values="quantity",
        aggfunc="count",
        fill_value=0,
    ).reset_index()
    if not {product_a, product_b}.issubset(pivot.columns):
        return pd.DataFrame(columns=["period", "cooccurrence"])
    shared = pivot[(pivot[product_a] > 0) & (pivot[product_b] > 0)]
    if shared.empty:
        return pd.DataFrame(columns=["period", "cooccurrence"])
    periods = both[both["transaction_id"].isin(shared["transaction_id"])].set_index(
        "transaction_id"
    )
    trend = (
        periods["date"]
        .groupby(periods.index)
        .first()
        .dt.to_period(period)
        .astype(str)
        .value_counts()
        .sort_index()
    )
    return pd.DataFrame({"period": trend.index, "cooccurrence": trend.values})


def compute_pair_centrality(
    df: pd.DataFrame,
    top_n_products: int = 100,
    min_cooccurrence: int = 5,
    segment_col: str | None = None,
    segment_val: str | None = None,
    mission_col: str | None = None,
    mission_val: str | None = None,
) -> pd.DataFrame:
    """PageRank centrality over the weighted co-occurrence graph."""
    import networkx as nx

    df = _filter_df(df, segment_col, segment_val, mission_col, mission_val)
    cooccurrence = compute_cooccurrence_matrix(df, top_n_products=top_n_products)
    graph = nx.Graph()
    products = cooccurrence.index
    values = cooccurrence.to_numpy()
    for i in range(len(products)):
        for j in range(i + 1, len(products)):
            if values[i, j] >= min_cooccurrence:
                graph.add_edge(products[i], products[j], weight=float(values[i, j]))

    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["stockcode", "pagerank", "betweenness", "degree"])

    pagerank = nx.pagerank(graph, weight="weight")
    betweenness = nx.betweenness_centrality(graph, weight="weight")
    return (
        pd.DataFrame(
            {
                "stockcode": list(graph.nodes()),
                "pagerank": [pagerank.get(n, 0.0) for n in graph.nodes()],
                "betweenness": [betweenness.get(n, 0.0) for n in graph.nodes()],
                "degree": [graph.degree(n) for n in graph.nodes()],
            }
        )
        .sort_values("pagerank", ascending=False)
        .reset_index(drop=True)
    )


def _affinity_and_cooccurrence(
    df: pd.DataFrame,
    min_cooccurrence: int,
    top_n_products: int | None,
    segment_col: str | None = None,
    segment_val: str | None = None,
    mission_col: str | None = None,
    mission_val: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    df = _filter_df(df, segment_col, segment_val, mission_col, mission_val)
    basket = create_basket_matrix(df)
    if top_n_products is not None:
        counts = basket.sum().sort_values(ascending=False)
        basket = basket[counts.head(top_n_products).index]
    M = basket.to_numpy(dtype=bool)
    n = M.shape[0]
    cooccur = (M.T @ M.astype(np.int64)).astype(int)
    counts = M.sum(axis=0).astype(float)
    numerator = cooccur * n - np.outer(counts, counts)
    denominator = np.sqrt(np.outer(counts, n - counts) * np.outer(n - counts, counts))
    phi = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )
    phi = np.where(cooccur >= min_cooccurrence, phi, np.nan)
    np.fill_diagonal(phi, 1.0)
    products = basket.columns.tolist()
    support = counts / n
    return (
        pd.DataFrame(phi, index=products, columns=products),
        pd.DataFrame(cooccur / n, index=products, columns=products),
        support,
    )


def compute_affinity_matrix(
    df: pd.DataFrame,
    min_cooccurrence: int = 5,
    top_n_products: int | None = None,
    segment_col: str | None = None,
    segment_val: str | None = None,
    mission_col: str | None = None,
    mission_val: str | None = None,
) -> pd.DataFrame:
    """Product x product phi-coefficient affinity matrix (NaN below min co-occurrence)."""
    affinity, _, _ = _affinity_and_cooccurrence(
        df, min_cooccurrence, top_n_products, segment_col, segment_val, mission_col, mission_val
    )
    return affinity


def get_top_affinity_pairs(
    df: pd.DataFrame,
    top_n: int = 20,
    min_cooccurrence: int = 5,
    min_affinity: float = 0.0,
    top_n_products: int | None = 200,
    segment_col: str | None = None,
    segment_val: str | None = None,
    mission_col: str | None = None,
    mission_val: str | None = None,
) -> pd.DataFrame:
    """Highest-affinity product pairs with co-occurrence rates.

    `top_n_products` limits the candidate pool to the most-purchased products,
    which keeps pair enumeration tractable on large catalogs.
    """
    affinity, cooccur, support = _affinity_and_cooccurrence(
        df, min_cooccurrence, top_n_products, segment_col, segment_val, mission_col, mission_val
    )
    products = affinity.columns.tolist()
    rows = []
    for i in range(len(products)):
        for j in range(i + 1, len(products)):
            value = affinity.iloc[i, j]
            if np.isnan(value):
                continue
            rows.append(
                {
                    "product_a": products[i],
                    "product_b": products[j],
                    "affinity": float(value),
                    "cooccurrence": float(cooccur.iloc[i, j]),
                    "support_a": float(support[i]),
                    "support_b": float(support[j]),
                }
            )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return check(
            pd.DataFrame(columns=list(AFFINITY_PAIRS.columns)), AFFINITY_PAIRS, allow_empty=True
        )
    pairs = pairs[pairs["affinity"].ge(min_affinity)]
    pairs = pairs.sort_values("affinity", ascending=False).head(top_n).reset_index(drop=True)
    return check(pairs, AFFINITY_PAIRS)


def get_product_affinity_profile(
    df: pd.DataFrame,
    product: str,
    top_n: int = 10,
    segment_col: str | None = None,
    segment_val: str | None = None,
    mission_col: str | None = None,
    mission_val: str | None = None,
) -> pd.DataFrame:
    """Top co-purchase partners for a single product."""
    affinity = compute_affinity_matrix(
        df,
        min_cooccurrence=2,
        segment_col=segment_col,
        segment_val=segment_val,
        mission_col=mission_col,
        mission_val=mission_val,
    )
    if product not in affinity.index:
        return check(
            pd.DataFrame(columns=list(AFFINITY_PAIRS.columns)), AFFINITY_PAIRS, allow_empty=True
        )
    row = affinity.loc[product].drop(labels=[product]).dropna().sort_values(ascending=False)
    partners = row.head(top_n).index.tolist()
    pairs = get_top_affinity_pairs(
        df,
        top_n=10_000,
        min_cooccurrence=2,
        segment_col=segment_col,
        segment_val=segment_val,
        mission_col=mission_col,
        mission_val=mission_val,
    )
    mask = ((pairs["product_a"] == product) & pairs["product_b"].isin(partners)) | (
        (pairs["product_b"] == product) & pairs["product_a"].isin(partners)
    )
    return check(pairs.loc[mask].reset_index(drop=True), AFFINITY_PAIRS, allow_empty=True)
