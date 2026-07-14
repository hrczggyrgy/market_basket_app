"""FP-Growth algorithm wrapper using mlxtend - maintained for backwards compatibility."""

from src.algorithms.frequent_itemsets import (
    create_basket_matrix,
    get_product_lookup,
    run_fpgrowth,
)

__all__ = ["run_fpgrowth", "create_basket_matrix", "get_product_lookup"]
