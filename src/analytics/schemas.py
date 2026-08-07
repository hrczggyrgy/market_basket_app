"""Declared output contracts for every analytics DataFrame.

Every analytics function that returns a DataFrame with a fixed set of columns
must validate its own output against one of these contracts before returning,
and every UI function consuming that output must validate the input against
the same contract before rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


class SchemaError(ValueError):
    """Raised when a DataFrame does not satisfy a declared contract."""


class EmptyResult(pd.DataFrame):
    """Sentinel for intentionally empty results with diagnostic context.

    Use instead of returning an empty DataFrame that would silently pass
    validation with allow_empty=True. Carries the reason for emptiness
    so callers can distinguish "no data" from "error" from "legitimately empty".
    """

    def __init__(
        self,
        reason: str = "No data meeting criteria",
        contract_name: str = "",
        diagnostics: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self._reason = reason
        self._contract_name = contract_name
        self._diagnostics = diagnostics or {}

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def contract_name(self) -> str:
        return self._contract_name

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self._diagnostics

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"EmptyResult(reason={self._reason!r}, contract={self._contract_name!r})"


def is_empty_result(obj: Any) -> bool:
    """Check if an object is an EmptyResult sentinel."""
    return isinstance(obj, EmptyResult)


@dataclass(frozen=True)
class ValueValidator:
    """Semantic validation rule for a column."""

    column: str
    check_fn: Callable[[pd.Series], pd.Series]  # returns boolean mask of valid rows
    error_msg: str
    severity: str = "error"  # "error" or "warning"


@dataclass(frozen=True)
class DataContract:
    """A declared set of required columns for a DataFrame with semantic validators."""

    name: str
    columns: tuple[str, ...]
    validators: tuple[ValueValidator, ...] = field(default_factory=tuple)

    def validate(
        self,
        df: pd.DataFrame,
        *,
        allow_empty: bool = False,
        check_values: bool = True,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Validate structure and optionally values.

        Returns:
            (validated_df, warnings_list)
        Raises:
            SchemaError: if structure validation fails
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise SchemaError(f"{self.name}: expected a DataFrame, got {type(df).__name__}")

        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise SchemaError(f"{self.name}: missing required columns {missing}")

        if not allow_empty and len(df) == 0:
            raise SchemaError(f"{self.name}: expected non-empty DataFrame")

        warnings = []
        if check_values and len(df) > 0:
            for validator in self.validators:
                if validator.column not in df.columns:
                    continue
                mask = validator.check_fn(df[validator.column])
                invalid_count = int((~mask).sum())
                if invalid_count > 0:
                    msg = f"{self.name}.{validator.column}: {invalid_count} rows violate '{validator.error_msg}'"
                    if validator.severity == "error":
                        raise SchemaError(msg)
                    else:
                        warnings.append(msg)

        return df, warnings


def contract(
    *columns: str,
    validators: Optional[list[ValueValidator]] = None,
) -> DataContract:
    """Build a DataContract from column names and optional validators."""
    return DataContract(
        name="unnamed",
        columns=tuple(columns),
        validators=tuple(validators or []),
    )


def check(
    df: pd.DataFrame,
    c: DataContract,
    *,
    allow_empty: bool = False,
    check_values: bool = True,
) -> pd.DataFrame:
    """Validate `df` against `c` and return it unchanged (for output self-checks)."""
    validated, warnings = c.validate(df, allow_empty=allow_empty, check_values=check_values)
    if warnings:
        import warnings as _warnings
        for w in warnings:
            _warnings.warn(w, UserWarning, stacklevel=2)
    return validated


def make_empty_result(
    contract: DataContract,
    reason: str = "No data meeting criteria",
    diagnostics: Optional[dict[str, Any]] = None,
) -> EmptyResult:
    """Create an EmptyResult sentinel for a contract."""
    return EmptyResult(reason=reason, contract_name=contract.name, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Cross-contract referential integrity validators
# ---------------------------------------------------------------------------

def validate_referential_integrity(
    outputs: dict[str, pd.DataFrame],
    contracts: dict[str, DataContract],
) -> list[str]:
    """Validate foreign-key-like relationships across contract outputs.

    Args:
        outputs: Mapping of contract_name -> DataFrame (or EmptyResult)
        contracts: Mapping of contract_name -> DataContract

    Returns:
        List of warning messages (does not raise on failure)
    """
    warnings = []

    # Build lookup sets
    stockcodes = set()
    if "transactions" in outputs and not is_empty_result(outputs["transactions"]):
        stockcodes = set(outputs["transactions"]["stockcode"].unique())

    customer_ids = set()
    if "transactions" in outputs and not is_empty_result(outputs["transactions"]):
        customer_ids = set(outputs["transactions"]["customer_id"].unique())

    # Validate transference references
    if "demand_transference" in outputs and not is_empty_result(outputs["demand_transference"]):
        dt = outputs["demand_transference"]
        missing_from = set(dt["from_product"].unique()) - stockcodes
        missing_to = set(dt["to_product"].unique()) - stockcodes
        if missing_from:
            warnings.append(f"demand_transference: {len(missing_from)} from_products not in transactions")
        if missing_to:
            warnings.append(f"demand_transference: {len(missing_to)} to_products not in transactions")

    # Validate switching references
    if "switching_matrix" in outputs and not is_empty_result(outputs["switching_matrix"]):
        sw = outputs["switching_matrix"]
        missing_from = set(sw["from_product"].unique()) - stockcodes
        missing_to = set(sw["to_product"].unique()) - stockcodes
        if missing_from:
            warnings.append(f"switching_matrix: {len(missing_from)} from_products not in transactions")
        if missing_to:
            warnings.append(f"switching_matrix: {len(missing_to)} to_products not in transactions")

    # Validate affinity references
    if "affinity_pairs" in outputs and not is_empty_result(outputs["affinity_pairs"]):
        ap = outputs["affinity_pairs"]
        missing_a = set(ap["product_a"].unique()) - stockcodes
        missing_b = set(ap["product_b"].unique()) - stockcodes
        if missing_a:
            warnings.append(f"affinity_pairs: {len(missing_a)} product_a not in transactions")
        if missing_b:
            warnings.append(f"affinity_pairs: {len(missing_b)} product_b not in transactions")

    # Validate CDT tree products
    if "cdt_tree_products" in outputs and not is_empty_result(outputs["cdt_tree_products"]):
        tp = outputs["cdt_tree_products"]
        missing = set(tp["stockcode"].unique()) - stockcodes
        if missing:
            warnings.append(f"cdt_tree_products: {len(missing)} products not in transactions")

    # Validate customer-level outputs
    for contract_name in ("customer_entropy", "rfm_features", "rfm_segments", "behavioral_features",
                          "behavioral_segments", "clv_customer", "uplift_scores"):
        if contract_name in outputs and not is_empty_result(outputs[contract_name]):
            df = outputs[contract_name]
            if "customer_id" in df.columns:
                missing = set(df["customer_id"].unique()) - customer_ids
                if missing:
                    warnings.append(f"{contract_name}: {len(missing)} customer_ids not in transactions")

    return warnings


# ---------------------------------------------------------------------------
# Data layer contracts
# ---------------------------------------------------------------------------

TRANSACTIONS = DataContract(
    name="transactions",
    columns=("date", "transaction_id", "stockcode", "product", "customer_id", "price", "quantity"),
    validators=(
        ValueValidator("price", lambda s: s > 0, "price must be positive"),
        ValueValidator("quantity", lambda s: s > 0, "quantity must be positive"),
    ),
)

# ---------------------------------------------------------------------------
# Association rules / FP-Growth contracts
# ---------------------------------------------------------------------------

FREQUENT_ITEMSETS = DataContract(
    name="frequent_itemsets",
    columns=("support", "itemsets", "length"),
    validators=(
        ValueValidator("support", lambda s: (s > 0) & (s <= 1), "support must be in (0, 1]"),
    ),
)

RULES = DataContract(
    name="association_rules",
    columns=(
        "antecedents",
        "consequents",
        "antecedent support",
        "consequent support",
        "support",
        "confidence",
        "lift",
        "lift_ci_lower",
        "lift_ci_upper",
        "leverage",
        "conviction",
        "zhangs_metric",
        "q_value",
        "is_redundant",
    ),
    validators=(
        ValueValidator("support", lambda s: (s > 0) & (s <= 1), "support must be in (0, 1]"),
        ValueValidator("confidence", lambda s: (s >= 0) & (s <= 1), "confidence must be in [0, 1]"),
        ValueValidator("lift", lambda s: s > 0, "lift must be positive"),
        ValueValidator("lift_ci_lower", lambda s: s.isna() | (s > 0), "lift CI lower must be positive or NaN"),
        ValueValidator("lift_ci_upper", lambda s: s.isna() | (s > 0), "lift CI upper must be positive or NaN"),
    ),
)

RULES_TABLE = DataContract(
    name="rules_table",
    columns=(
        "antecedent",
        "consequent",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
        "zhangs_metric",
    ),
    validators=(
        ValueValidator("support", lambda s: (s > 0) & (s <= 1), "support must be in (0, 1]"),
        ValueValidator("confidence", lambda s: (s >= 0) & (s <= 1), "confidence must be in [0, 1]"),
        ValueValidator("lift", lambda s: s > 0, "lift must be positive"),
    ),
)

# Category-level rules rollup
CATEGORY_RULES = DataContract(
    name="category_rules",
    columns=(
        "antecedent_category",
        "consequent_category",
        "support",
        "confidence",
        "lift",
        "rule_count",
        "avg_lift",
        "max_lift",
    ),
    validators=(
        ValueValidator("support", lambda s: (s > 0) & (s <= 1), "support must be in (0, 1]"),
        ValueValidator("confidence", lambda s: (s >= 0) & (s <= 1), "confidence must be in [0, 1]"),
        ValueValidator("lift", lambda s: s > 0, "lift must be positive"),
        ValueValidator("rule_count", lambda s: s >= 1, "rule_count must be >= 1"),
    ),
)

# ---------------------------------------------------------------------------
# Co-purchase / add-on contracts
# ---------------------------------------------------------------------------

AFFINITY_PAIRS = DataContract(
    name="affinity_pairs",
    columns=("product_a", "product_b", "affinity", "cooccurrence", "support_a", "support_b"),
    validators=(
        ValueValidator("affinity", lambda s: s > 0, "affinity must be positive"),
        ValueValidator("cooccurrence", lambda s: s >= 0, "cooccurrence must be non-negative"),
        ValueValidator("support_a", lambda s: (s > 0) & (s <= 1), "support_a must be in (0, 1]"),
        ValueValidator("support_b", lambda s: (s > 0) & (s <= 1), "support_b must be in (0, 1]"),
    ),
)

ADDON_RECS = DataContract(
    name="addon_recommendations",
    columns=("anchor", "addon", "support", "confidence", "lift", "cooccurrence"),
    validators=(
        ValueValidator("support", lambda s: (s > 0) & (s <= 1), "support must be in (0, 1]"),
        ValueValidator("confidence", lambda s: (s >= 0) & (s <= 1), "confidence must be in [0, 1]"),
        ValueValidator("lift", lambda s: s > 0, "lift must be positive"),
    ),
)

# ---------------------------------------------------------------------------
# Switching contracts
# ---------------------------------------------------------------------------

SWITCHING_MATRIX = DataContract(
    name="switching_matrix",
    columns=("from_product", "to_product", "count", "pct"),
    validators=(
        ValueValidator("count", lambda s: s >= 0, "count must be non-negative"),
        ValueValidator("pct", lambda s: (s >= 0) & (s <= 1), "pct must be in [0, 1]"),
    ),
)

LOYALTY_METRICS = DataContract(
    name="customer_loyalty_metrics",
    columns=(
        "customer_id",
        "n_transactions",
        "n_distinct_products",
        "repeat_purchase_rate",
        "avg_basket_size",
        "switching_count",
        "switching_rate",
    ),
    validators=(
        ValueValidator("n_transactions", lambda s: s > 0, "n_transactions must be positive"),
        ValueValidator("repeat_purchase_rate", lambda s: (s >= 0) & (s <= 1), "repeat_purchase_rate must be in [0, 1]"),
        ValueValidator("switching_rate", lambda s: (s >= 0) & (s <= 1), "switching_rate must be in [0, 1]"),
    ),
)

# ---------------------------------------------------------------------------
# Basket metrics contracts
# ---------------------------------------------------------------------------

BASKET_PENETRATION = DataContract(
    name="basket_penetration",
    columns=("stockcode", "basket_count", "penetration", "revenue_share"),
    validators=(
        ValueValidator("basket_count", lambda s: s > 0, "basket_count must be positive"),
        ValueValidator("penetration", lambda s: (s >= 0) & (s <= 1), "penetration must be in [0, 1]"),
        ValueValidator("revenue_share", lambda s: (s >= 0) & (s <= 1), "revenue_share must be in [0, 1]"),
    ),
)

BASKET_OVER_TIME = DataContract(
    name="basket_over_time",
    columns=("period", "n_baskets", "avg_basket_size", "avg_basket_value"),
    validators=(
        ValueValidator("n_baskets", lambda s: s >= 0, "n_baskets must be non-negative"),
        ValueValidator("avg_basket_size", lambda s: s > 0, "avg_basket_size must be positive"),
        ValueValidator("avg_basket_value", lambda s: s >= 0, "avg_basket_value must be non-negative"),
    ),
)

BASKET_COMPOSITION = DataContract(
    name="basket_composition",
    columns=("basket_size", "n_baskets", "pct"),
    validators=(
        ValueValidator("basket_size", lambda s: s > 0, "basket_size must be positive"),
        ValueValidator("n_baskets", lambda s: s >= 0, "n_baskets must be non-negative"),
        ValueValidator("pct", lambda s: (s >= 0) & (s <= 1), "pct must be in [0, 1]"),
    ),
)

REVENUE_SPC = DataContract(
    name="revenue_spc",
    columns=("period", "revenue", "center", "ucl", "lcl", "anomaly", "rule"),
    validators=(
        ValueValidator("revenue", lambda s: s >= 0, "revenue must be non-negative"),
        ValueValidator("center", lambda s: s.isna() | (s >= 0), "center (control mean) must be non-negative or NaN"),
        ValueValidator("anomaly", lambda s: s.isin({False, True}), "anomaly must be a boolean"),
    ),
)

CUSTOMER_ENTROPY = DataContract(
    name="customer_entropy",
    columns=("customer_id", "n_distinct_products", "n_purchases", "entropy", "normalized_entropy"),
    validators=(
        ValueValidator("n_distinct_products", lambda s: s > 0, "n_distinct_products must be positive"),
        ValueValidator("n_purchases", lambda s: s > 0, "n_purchases must be positive"),
        ValueValidator("entropy", lambda s: s >= 0, "entropy must be non-negative"),
        ValueValidator("normalized_entropy", lambda s: (s >= 0) & (s <= 1), "normalized_entropy must be in [0, 1]"),
    ),
)

IPT_CV = DataContract(
    name="ipt_cv",
    columns=("stockcode", "mean_ipt", "std_ipt", "cv_ipt", "n_transactions"),
    validators=(
        ValueValidator("mean_ipt", lambda s: s > 0, "mean_ipt must be positive"),
        ValueValidator("std_ipt", lambda s: s >= 0, "std_ipt must be non-negative"),
        ValueValidator("cv_ipt", lambda s: s >= 0, "cv_ipt must be non-negative"),
        ValueValidator("n_transactions", lambda s: s > 0, "n_transactions must be positive"),
    ),
)

# ---------------------------------------------------------------------------
# Cohort contracts
# ---------------------------------------------------------------------------

COHORT_RETENTION = DataContract(
    name="cohort_retention",
    columns=("cohort", "period_index", "retained", "cohort_size", "retention_rate"),
    validators=(
        ValueValidator("retained", lambda s: s >= 0, "retained must be non-negative"),
        ValueValidator("cohort_size", lambda s: s > 0, "cohort_size must be positive"),
        ValueValidator("retention_rate", lambda s: (s >= 0) & (s <= 1), "retention_rate must be in [0, 1]"),
    ),
)

COHORT_SIZES = DataContract(
    name="cohort_sizes",
    columns=("cohort", "n_customers", "n_transactions", "revenue"),
    validators=(
        ValueValidator("n_customers", lambda s: s > 0, "n_customers must be positive"),
        ValueValidator("n_transactions", lambda s: s >= 0, "n_transactions must be non-negative"),
        ValueValidator("revenue", lambda s: s >= 0, "revenue must be non-negative"),
    ),
)

POP_COMPARISON = DataContract(
    name="period_over_period",
    columns=("period", "revenue", "transactions", "customers", "aov", "revenue_growth", "aov_growth"),
    validators=(
        ValueValidator("revenue", lambda s: s >= 0, "revenue must be non-negative"),
        ValueValidator("transactions", lambda s: s >= 0, "transactions must be non-negative"),
        ValueValidator("customers", lambda s: s >= 0, "customers must be non-negative"),
        ValueValidator("aov", lambda s: s >= 0, "aov must be non-negative"),
    ),
)

YOY_COMPARISON = DataContract(
    name="year_over_year",
    columns=(
        "year",
        "week",
        "revenue",
        "transactions",
        "customers",
        "aov",
        "prior_revenue",
        "prior_aov",
        "revenue_yoy_growth",
        "aov_yoy_growth",
    ),
    validators=(
        ValueValidator("revenue", lambda s: s >= 0, "revenue must be non-negative"),
        ValueValidator("transactions", lambda s: s >= 0, "transactions must be non-negative"),
        ValueValidator("customers", lambda s: s >= 0, "customers must be non-negative"),
        ValueValidator("aov", lambda s: s >= 0, "aov must be non-negative"),
    ),
)

COHORT_LTV = DataContract(
    name="cohort_ltv_curve",
    columns=("cohort", "period_index", "cumulative_revenue", "ltv_per_customer"),
    validators=(
        ValueValidator("cumulative_revenue", lambda s: s >= 0, "cumulative_revenue must be non-negative"),
        ValueValidator("ltv_per_customer", lambda s: s >= 0, "ltv_per_customer must be non-negative"),
    ),
)

COHORT_DECAY = DataContract(
    name="cohort_decay",
    columns=("cohort", "decay_rate"),
    validators=(
        ValueValidator("decay_rate", lambda s: s.notna(), "decay_rate must not be NaN (negative implies retention growth)"),
    ),
)

# ---------------------------------------------------------------------------
# Category / performance contracts
# ---------------------------------------------------------------------------

CATEGORY_KPIS = DataContract(
    name="category_kpis",
    columns=("category", "revenue", "transactions", "customers", "penetration", "aov", "revenue_share", "growth_pct"),
    validators=(
        ValueValidator("revenue", lambda s: s >= 0, "revenue must be non-negative"),
        ValueValidator("transactions", lambda s: s >= 0, "transactions must be non-negative"),
        ValueValidator("customers", lambda s: s >= 0, "customers must be non-negative"),
        ValueValidator("penetration", lambda s: (s >= 0) & (s <= 1), "penetration must be in [0, 1]"),
        ValueValidator("aov", lambda s: s >= 0, "aov must be non-negative"),
        ValueValidator("revenue_share", lambda s: (s >= 0) & (s <= 1), "revenue_share must be in [0, 1]"),
    ),
)

CATEGORY_SCORECARD = DataContract(
    name="category_scorecard",
    columns=("category", "revenue", "revenue_share", "transactions", "customers", "aov", "growth_pct", "role", "rag"),
    validators=(
        ValueValidator("revenue", lambda s: s >= 0, "revenue must be non-negative"),
        ValueValidator("revenue_share", lambda s: (s >= 0) & (s <= 1), "revenue_share must be in [0, 1]"),
        ValueValidator("transactions", lambda s: s >= 0, "transactions must be non-negative"),
        ValueValidator("customers", lambda s: s >= 0, "customers must be non-negative"),
        ValueValidator("aov", lambda s: s >= 0, "aov must be non-negative"),
    ),
)

INFERRED_CATEGORIES = DataContract(
    name="inferred_categories",
    columns=("stockcode", "product", "inferred_category"),
)

CATEGORY_TREND = DataContract(
    name="category_trend",
    columns=("category", "period", "revenue", "basket_penetration"),
    validators=(
        ValueValidator("revenue", lambda s: s >= 0, "revenue must be non-negative"),
        ValueValidator("basket_penetration", lambda s: (s >= 0) & (s <= 1), "basket_penetration must be in [0, 1]"),
    ),
)

CATEGORY_MANAGER_SCORECARD = DataContract(
    name="category_manager_scorecard",
    columns=(
        "category",
        "role",
        "total_revenue",
        "revenue_yoy_growth",
        "basket_penetration",
        "repeat_purchase_rate",
        "sku_share",
        "revenue_share",
        "kvi_count",
        "kvi_share",
    ),
    validators=(
        ValueValidator("total_revenue", lambda s: s >= 0, "total_revenue must be non-negative"),
        ValueValidator("basket_penetration", lambda s: (s >= 0) & (s <= 1), "basket_penetration must be in [0, 1]"),
        ValueValidator("repeat_purchase_rate", lambda s: (s >= 0) & (s <= 1), "repeat_purchase_rate must be in [0, 1]"),
        ValueValidator("sku_share", lambda s: (s >= 0) & (s <= 1), "sku_share must be in [0, 1]"),
        ValueValidator("revenue_share", lambda s: (s >= 0) & (s <= 1), "revenue_share must be in [0, 1]"),
        ValueValidator("kvi_count", lambda s: s >= 0, "kvi_count must be non-negative"),
        ValueValidator("kvi_share", lambda s: (s >= 0) & (s <= 1), "kvi_share must be in [0, 1]"),
    ),
)

ASSORTMENT_EFFICIENCY = DataContract(
    name="assortment_efficiency",
    columns=(
        "category",
        "role",
        "sku_share",
        "revenue_share",
        "total_revenue",
        "efficiency_index",
        "efficiency_label",
    ),
    validators=(
        ValueValidator("sku_share", lambda s: (s >= 0) & (s <= 1), "sku_share must be in [0, 1]"),
        ValueValidator("revenue_share", lambda s: (s >= 0) & (s <= 1), "revenue_share must be in [0, 1]"),
        ValueValidator("total_revenue", lambda s: s >= 0, "total_revenue must be non-negative"),
        ValueValidator("efficiency_index", lambda s: s >= 0, "efficiency_index must be non-negative"),
    ),
)

CATEGORY_GROWTH_MATRIX = DataContract(
    name="category_growth_matrix",
    columns=(
        "category",
        "role",
        "revenue_share",
        "growth_pct",
        "total_revenue",
        "quadrant",
    ),
    validators=(
        ValueValidator("revenue_share", lambda s: (s >= 0) & (s <= 1), "revenue_share must be in [0, 1]"),
        ValueValidator("total_revenue", lambda s: s >= 0, "total_revenue must be non-negative"),
        ValueValidator("quadrant", lambda s: s.isin({"star", "cash_cow", "question_mark", "dog"}), "quadrant must be star/cash_cow/question_mark/avoid"),
    ),
)

CATEGORY_ROLES = DataContract(
    name="category_roles",
    columns=(
        "category",
        "role",
        "trip_generation_rate",
        "demand_cv",
        "seasonality_amplitude",
        "attachment_rate",
        "destination_categories",
        "category_source",
    ),
    validators=(
        ValueValidator("trip_generation_rate", lambda s: (s >= 0) & (s <= 1), "trip_generation_rate must be in [0, 1]"),
        ValueValidator("demand_cv", lambda s: s >= 0, "demand_cv must be non-negative"),
        ValueValidator("seasonality_amplitude", lambda s: s >= 0, "seasonality_amplitude must be non-negative"),
        ValueValidator("attachment_rate", lambda s: (s >= 0) & (s <= 1), "attachment_rate must be in [0, 1]"),
    ),
)

PRODUCT_METRICS = DataContract(
    name="product_metrics",
    columns=("stockcode", "revenue", "units", "transactions", "customers", "avg_price", "penetration"),
    validators=(
        ValueValidator("revenue", lambda s: s >= 0, "revenue must be non-negative"),
        ValueValidator("units", lambda s: s >= 0, "units must be non-negative"),
        ValueValidator("transactions", lambda s: s >= 0, "transactions must be non-negative"),
        ValueValidator("customers", lambda s: s >= 0, "customers must be non-negative"),
        ValueValidator("avg_price", lambda s: s > 0, "avg_price must be positive"),
        ValueValidator("penetration", lambda s: (s >= 0) & (s <= 1), "penetration must be in [0, 1]"),
    ),
)

ABC_CLASSES = DataContract(
    name="abc_classes",
    columns=("stockcode", "revenue", "cumulative_share", "abc_class"),
    validators=(
        ValueValidator("revenue", lambda s: s >= 0, "revenue must be non-negative"),
        ValueValidator("cumulative_share", lambda s: (s >= 0) & (s <= 1), "cumulative_share must be in [0, 1]"),
    ),
)

XYZ_CLASSES = DataContract(
    name="xyz_classes",
    columns=("stockcode", "revenue", "cv", "xyz_class"),
    validators=(
        ValueValidator("revenue", lambda s: s >= 0, "revenue must be non-negative"),
        ValueValidator("cv", lambda s: s >= 0, "cv must be non-negative"),
    ),
)

LIFECYCLE = DataContract(
    name="product_lifecycle",
    columns=("stockcode", "recent_revenue", "prior_revenue", "growth_pct", "stage"),
    validators=(
        ValueValidator("recent_revenue", lambda s: s >= 0, "recent_revenue must be non-negative"),
        ValueValidator("prior_revenue", lambda s: s >= 0, "prior_revenue must be non-negative"),
    ),
)

PRODUCT_VELOCITY = DataContract(
    name="product_velocity",
    columns=("stockcode", "units", "active_days", "velocity"),
    validators=(
        ValueValidator("units", lambda s: s >= 0, "units must be non-negative"),
        ValueValidator("active_days", lambda s: s > 0, "active_days must be positive"),
        ValueValidator("velocity", lambda s: s >= 0, "velocity must be non-negative"),
    ),
)

REPEAT_RATE = DataContract(
    name="repeat_rate",
    columns=("stockcode", "n_customers", "repeat_customers", "repeat_rate"),
    validators=(
        ValueValidator("n_customers", lambda s: s > 0, "n_customers must be positive"),
        ValueValidator("repeat_customers", lambda s: s >= 0, "repeat_customers must be non-negative"),
        ValueValidator("repeat_rate", lambda s: (s >= 0) & (s <= 1), "repeat_rate must be in [0, 1]"),
    ),
)

SECOND_PURCHASE = DataContract(
    name="second_purchase",
    columns=("stockcode", "n_second_purchasers", "median_days_to_second", "mean_days_to_second"),
    validators=(
        ValueValidator("n_second_purchasers", lambda s: s >= 0, "n_second_purchasers must be non-negative"),
        ValueValidator("median_days_to_second", lambda s: s >= 0, "median_days_to_second must be non-negative"),
        ValueValidator("mean_days_to_second", lambda s: s >= 0, "mean_days_to_second must be non-negative"),
    ),
)

SKU_RATIONALIZATION = DataContract(
    name="sku_rationalization",
    columns=("stockcode", "revenue", "abc_class", "xyz_class", "velocity", "repeat_rate", "action"),
    validators=(
        ValueValidator("revenue", lambda s: s >= 0, "revenue must be non-negative"),
        ValueValidator("velocity", lambda s: s >= 0, "velocity must be non-negative"),
        ValueValidator("repeat_rate", lambda s: (s >= 0) & (s <= 1), "repeat_rate must be in [0, 1]"),
    ),
)

CUSTOMER_FEATURES = DataContract(
    name="customer_features",
    columns=(
        "customer_id",
        "recency_days",
        "frequency",
        "monetary",
        "n_baskets",
        "avg_basket_size",
        "n_distinct_products",
        "favorite_category",
        "target_product",
    ),
    validators=(
        ValueValidator("recency_days", lambda s: s >= 0, "recency_days must be non-negative"),
        ValueValidator("frequency", lambda s: s > 0, "frequency must be positive"),
        ValueValidator("monetary", lambda s: s >= 0, "monetary must be non-negative"),
        ValueValidator("n_baskets", lambda s: s > 0, "n_baskets must be positive"),
        ValueValidator("avg_basket_size", lambda s: s > 0, "avg_basket_size must be positive"),
    ),
)

MODEL_METRICS = DataContract(
    name="model_metrics",
    columns=("metric", "value"),
)

TREE_RULES = DataContract(
    name="tree_rules",
    columns=("rule_index", "rule_path", "n_samples", "purity", "target_class"),
    validators=(
        ValueValidator("n_samples", lambda s: s > 0, "n_samples must be positive"),
        ValueValidator("purity", lambda s: (s >= 0) & (s <= 1), "purity must be in [0, 1]"),
    ),
)

# ---------------------------------------------------------------------------
# Promotional analytics contracts
# ---------------------------------------------------------------------------

CATEGORY_PROMO_TIMELINE = DataContract(
    name="category_promo_timeline",
    columns=(
        "category",
        "period",
        "promo_revenue",
        "non_promo_revenue",
        "n_promos",
        "avg_discount_pct",
    ),
    validators=(
        ValueValidator("promo_revenue", lambda s: s >= 0, "promo_revenue must be non-negative"),
        ValueValidator("non_promo_revenue", lambda s: s >= 0, "non_promo_revenue must be non-negative"),
        ValueValidator("n_promos", lambda s: s >= 0, "n_promos must be non-negative"),
        ValueValidator("avg_discount_pct", lambda s: (s >= 0) & (s <= 100), "avg_discount_pct must be in [0, 100]"),
    ),
)

PROMO_PERIODS = DataContract(
    name="promo_periods",
    columns=(
        "stockcode",
        "product_name",
        "start_date",
        "end_date",
        "duration_days",
        "avg_discount_pct",
        "promo_revenue",
        "baseline_revenue",
        "promo_qty",
        "baseline_qty",
        "promo_orders",
        "baseline_orders",
        "promo_customers",
        "baseline_customers",
        "qty_lift",
        "revenue_lift",
        "avg_promo_price",
        "avg_baseline_price",
    ),
    validators=(
        ValueValidator("duration_days", lambda s: s > 0, "duration_days must be positive"),
        ValueValidator("avg_discount_pct", lambda s: s >= 0, "avg_discount_pct must be non-negative"),
        ValueValidator("promo_revenue", lambda s: s >= 0, "promo_revenue must be non-negative"),
        ValueValidator("baseline_revenue", lambda s: s >= 0, "baseline_revenue must be non-negative"),
        ValueValidator("promo_qty", lambda s: s >= 0, "promo_qty must be non-negative"),
        ValueValidator("baseline_qty", lambda s: s >= 0, "baseline_qty must be non-negative"),
        ValueValidator("avg_promo_price", lambda s: s > 0, "avg_promo_price must be positive"),
        ValueValidator("avg_baseline_price", lambda s: s > 0, "avg_baseline_price must be positive"),
    ),
)

PROMO_BASELINE = DataContract(
    name="promo_baseline",
    columns=(
        "stockcode",
        "week",
        "actual_units",
        "actual_revenue",
        "avg_price",
        "is_promo",
        "baseline_units",
        "baseline_revenue",
        "incremental_units",
        "incremental_revenue",
        "incrementality_pct",
    ),
    validators=(
        ValueValidator("actual_units", lambda s: s >= 0, "actual_units must be non-negative"),
        ValueValidator("actual_revenue", lambda s: s >= 0, "actual_revenue must be non-negative"),
        ValueValidator("avg_price", lambda s: s > 0, "avg_price must be positive"),
        ValueValidator("baseline_units", lambda s: s >= 0, "baseline_units must be non-negative"),
        ValueValidator("baseline_revenue", lambda s: s >= 0, "baseline_revenue must be non-negative"),
    ),
)

PROMO_LIFT = DataContract(
    name="promo_lift",
    columns=(
        "promo_id",
        "stockcode",
        "start_date",
        "end_date",
        "treated_revenue",
        "control_revenue",
        "treated_qty",
        "control_qty",
        "treated_orders",
        "control_orders",
        "lift_revenue_pct",
        "lift_qty_pct",
        "lift_orders_pct",
        "p_value",
        "significant",
    ),
    validators=(
        ValueValidator("treated_revenue", lambda s: s >= 0, "treated_revenue must be non-negative"),
        ValueValidator("control_revenue", lambda s: s >= 0, "control_revenue must be non-negative"),
        ValueValidator("treated_qty", lambda s: s >= 0, "treated_qty must be non-negative"),
        ValueValidator("control_qty", lambda s: s >= 0, "control_qty must be non-negative"),
        ValueValidator("treated_orders", lambda s: s >= 0, "treated_orders must be non-negative"),
        ValueValidator("control_orders", lambda s: s >= 0, "control_orders must be non-negative"),
    ),
)

PROMO_WATERFALL = DataContract(
    name="promo_waterfall",
    columns=(
        "stockcode",
        "baseline_revenue",
        "incremental_revenue",
        "acceleration_revenue",
        "switching_revenue",
        "stockpiling_revenue",
        "net_incremental_revenue",
        "roi",
    ),
    validators=(
        ValueValidator("baseline_revenue", lambda s: s >= 0, "baseline_revenue must be non-negative"),
    ),
)

PROMO_ROI = DataContract(
    name="promo_roi",
    columns=(
        "stockcode",
        "incremental_revenue",
        "ci_low",
        "ci_high",
        "incremental_profit",
        "promo_cost",
        "roi_pct",
    ),
    validators=(
        ValueValidator("promo_cost", lambda s: s >= 0, "promo_cost must be non-negative"),
    ),
)

PROMO_TIMING_DOW = DataContract(
    name="promo_timing_dow",
    columns=(
        "dow",
        "day_name",
        "promo_revenue",
        "base_revenue",
        "promo_orders",
        "base_orders",
        "revenue_lift",
    ),
    validators=(
        ValueValidator("promo_revenue", lambda s: s >= 0, "promo_revenue must be non-negative"),
        ValueValidator("base_revenue", lambda s: s >= 0, "base_revenue must be non-negative"),
        ValueValidator("promo_orders", lambda s: s >= 0, "promo_orders must be non-negative"),
        ValueValidator("base_orders", lambda s: s >= 0, "base_orders must be non-negative"),
    ),
)

PROMO_TIMING_MONTH = DataContract(
    name="promo_timing_month",
    columns=("month", "month_name", "promo_revenue", "base_revenue", "promo_orders", "base_orders", "revenue_lift"),
    validators=(
        ValueValidator("promo_revenue", lambda s: s >= 0, "promo_revenue must be non-negative"),
        ValueValidator("base_revenue", lambda s: s >= 0, "base_revenue must be non-negative"),
        ValueValidator("promo_orders", lambda s: s >= 0, "promo_orders must be non-negative"),
        ValueValidator("base_orders", lambda s: s >= 0, "base_orders must be non-negative"),
    ),
)

PROMO_HALO = DataContract(
    name="promo_halo",
    columns=("promo_product", "halo_product", "halo_revenue", "base_revenue", "halo_orders", "base_orders", "revenue_lift"),
    validators=(
        ValueValidator("halo_revenue", lambda s: s >= 0, "halo_revenue must be non-negative"),
        ValueValidator("base_revenue", lambda s: s >= 0, "base_revenue must be non-negative"),
        ValueValidator("halo_orders", lambda s: s >= 0, "halo_orders must be non-negative"),
        ValueValidator("base_orders", lambda s: s >= 0, "base_orders must be non-negative"),
    ),
)

PROMO_CANNIBALIZATION = DataContract(
    name="promo_cannibalization",
    columns=(
        "promo_product",
        "peer_product",
        "category",
        "promo_revenue",
        "base_revenue",
        "promo_orders",
        "base_orders",
        "cannibalized_revenue",
        "cannibalization_index",
    ),
    validators=(
        ValueValidator("promo_revenue", lambda s: s >= 0, "promo_revenue must be non-negative"),
        ValueValidator("base_revenue", lambda s: s >= 0, "base_revenue must be non-negative"),
        ValueValidator("promo_orders", lambda s: s >= 0, "promo_orders must be non-negative"),
        ValueValidator("base_orders", lambda s: s >= 0, "base_orders must be non-negative"),
        ValueValidator("cannibalized_revenue", lambda s: s >= 0, "cannibalized_revenue must be non-negative"),
    ),
)

UPLIFT_PROPENSITY = DataContract(
    name="uplift_propensity",
    columns=("customer_id", "propensity", "treatment"),
    validators=(
        ValueValidator("propensity", lambda s: (s >= 0) & (s <= 1), "propensity must be in [0, 1]"),
    ),
)

UPLIFT_METRICS = DataContract(
    name="uplift_metrics",
    columns=("metric", "value"),
)

UPLIFT_SCORES = DataContract(
    name="uplift_scores",
    columns=(
        "customer_id",
        "uplift",
        "ci_lower",
        "ci_upper",
        "propensity_score",
        "treatment_flag",
        "acceleration_uplift",
        "switching_uplift",
        "stockpiling_uplift",
    ),
    validators=(
        ValueValidator("propensity_score", lambda s: s.isna() | ((s >= 0) & (s <= 1)), "propensity_score must be in [0, 1] or NaN"),
    ),
)

QINI_CURVE = DataContract(
    name="qini_curve",
    columns=("x", "qini_y", "random_y", "ci_lower", "ci_upper", "qini_coefficient", "auuc"),
    validators=(
        ValueValidator("x", lambda s: (s >= 0) & (s <= 1), "x must be in [0, 1]"),
    ),
)

# ---------------------------------------------------------------------------
# CLV contracts
# ---------------------------------------------------------------------------

CLV_PREDICTIONS = DataContract(
    name="clv_predictions",
    columns=(
        "customer_id",
        "frequency",
        "recency",
        "T",
        "monetary_value",
        "predicted_purchases",
        "expected_avg_value",
        "predicted_clv",
        "ci_lower",
        "ci_upper",
        "p_alive",
        "clv_segment",
    ),
    validators=(
        ValueValidator("frequency", lambda s: s >= 0, "frequency must be non-negative"),
        ValueValidator("recency", lambda s: s >= 0, "recency must be non-negative"),
        ValueValidator("T", lambda s: s > 0, "T must be positive"),
        ValueValidator("monetary_value", lambda s: s >= 0, "monetary_value must be non-negative"),
        ValueValidator("predicted_purchases", lambda s: s >= 0, "predicted_purchases must be non-negative"),
        ValueValidator("expected_avg_value", lambda s: s >= 0, "expected_avg_value must be non-negative"),
        ValueValidator("predicted_clv", lambda s: s >= 0, "predicted_clv must be non-negative"),
        ValueValidator("ci_lower", lambda s: s >= 0, "ci_lower must be non-negative"),
        ValueValidator("p_alive", lambda s: (s >= 0) & (s <= 1), "p_alive must be in [0, 1]"),
    ),
)

CLV_DIAGNOSTICS = DataContract(
    name="clv_diagnostics",
    columns=("metric", "value"),
)

CLV_CUSTOMER = DataContract(
    name="clv_customer",
    columns=(
        "customer_id",
        "frequency",
        "recency_days",
        "customer_lifetime_days",
        "total_revenue",
        "avg_order_value",
        "p_alive",
        "predicted_purchases",
        "expected_avg_value",
        "predicted_clv",
        "clv_12m",
        "clv_12m_discounted",
        "clv_segment",
        "entropy",
        "normalized_entropy",
    ),
    validators=(
        ValueValidator("frequency", lambda s: s >= 0, "frequency must be non-negative"),
        ValueValidator("recency_days", lambda s: s >= 0, "recency_days must be non-negative"),
        ValueValidator("total_revenue", lambda s: s >= 0, "total_revenue must be non-negative"),
        ValueValidator("avg_order_value", lambda s: s >= 0, "avg_order_value must be non-negative"),
        ValueValidator("p_alive", lambda s: (s >= 0) & (s <= 1), "p_alive must be in [0, 1]"),
        ValueValidator("predicted_purchases", lambda s: s >= 0, "predicted_purchases must be non-negative"),
        ValueValidator("expected_avg_value", lambda s: s >= 0, "expected_avg_value must be non-negative"),
        ValueValidator("predicted_clv", lambda s: s >= 0, "predicted_clv must be non-negative"),
        ValueValidator("clv_12m", lambda s: s >= 0, "clv_12m must be non-negative"),
        ValueValidator("normalized_entropy", lambda s: (s >= 0) & (s <= 1), "normalized_entropy must be in [0, 1]"),
    ),
)

# ---------------------------------------------------------------------------
# Demand transference contracts
# ---------------------------------------------------------------------------

DEMAND_TRANSFERENCE = DataContract(
    name="demand_transference",
    columns=(
        "from_product",
        "to_product",
        "switch_rate",
        "revenue_share_from",
        "demand_transference",
        "revenue_at_risk",
    ),
    validators=(
        ValueValidator("switch_rate", lambda s: (s >= 0) & (s <= 1), "switch_rate must be in [0, 1]"),
        ValueValidator("revenue_share_from", lambda s: (s >= 0) & (s <= 1), "revenue_share_from must be in [0, 1]"),
        ValueValidator("demand_transference", lambda s: s >= 0, "demand_transference must be non-negative"),
        ValueValidator("revenue_at_risk", lambda s: s >= 0, "revenue_at_risk must be non-negative"),
    ),
)

SDP_SCORES = DataContract(
    name="substitutable_demand_percentage",
    columns=("stockcode", "sdp"),
    validators=(
        ValueValidator("sdp", lambda s: (s >= 0) & (s <= 1), "sdp must be in [0, 1]"),
    ),
)

DELIST_IMPACT = DataContract(
    name="delist_impact",
    columns=(
        "stockcode",
        "product_revenue",
        "estimated_revenue_recovered",
        "net_revenue_impact",
        "recovery_rate",
    ),
    validators=(
        ValueValidator("product_revenue", lambda s: s >= 0, "product_revenue must be non-negative"),
        ValueValidator("estimated_revenue_recovered", lambda s: s >= 0, "estimated_revenue_recovered must be non-negative"),
    ),
)

NODE_DELIST_IMPACT = DataContract(
    name="node_delist_impact",
    columns=(
        "node_id",
        "n_products",
        "total_node_revenue",
        "internal_recovery",
        "external_leakage",
        "node_sdp",
    ),
    validators=(
        ValueValidator("n_products", lambda s: s > 0, "n_products must be positive"),
        ValueValidator("total_node_revenue", lambda s: s >= 0, "total_node_revenue must be non-negative"),
        ValueValidator("internal_recovery", lambda s: s >= 0, "internal_recovery must be non-negative"),
        ValueValidator("external_leakage", lambda s: s >= 0, "external_leakage must be non-negative"),
        ValueValidator("node_sdp", lambda s: s.isna() | ((s >= 0) & (s <= 1)), "node_sdp must be in [0, 1] or NaN"),
    ),
)

CROSS_ELASTICITY = DataContract(
    name="cross_elasticity",
    columns=(
        "product_a",
        "product_b",
        "own_elasticity",
        "own_elasticity_se",
        "own_elasticity_p",
        "cross_elasticity",
        "cross_elasticity_se",
        "cross_elasticity_p",
        "r_squared",
        "n_obs",
        "avg_price_a",
        "avg_price_b",
    ),
    validators=(
        ValueValidator("own_elasticity_se", lambda s: s >= 0, "own_elasticity_se must be non-negative"),
        ValueValidator("cross_elasticity_se", lambda s: s >= 0, "cross_elasticity_se must be non-negative"),
        ValueValidator("r_squared", lambda s: (s >= 0) & (s <= 1), "r_squared must be in [0, 1]"),
        ValueValidator("n_obs", lambda s: s > 0, "n_obs must be positive"),
        ValueValidator("avg_price_a", lambda s: s > 0, "avg_price_a must be positive"),
        ValueValidator("avg_price_b", lambda s: s > 0, "avg_price_b must be positive"),
    ),
)

RECOVERY_HHI = DataContract(
    name="recovery_hhi",
    columns=(
        "delisted_product",
        "recovery_hhi",
        "n_substitutes",
        "total_revenue_at_risk",
        "top_substitute",
        "top_share",
    ),
    validators=(
        ValueValidator("recovery_hhi", lambda s: (s >= 0) & (s <= 1), "recovery_hhi must be in [0, 1]"),
        ValueValidator("n_substitutes", lambda s: s > 0, "n_substitutes must be positive"),
        ValueValidator("total_revenue_at_risk", lambda s: s >= 0, "total_revenue_at_risk must be non-negative"),
        ValueValidator("top_share", lambda s: (s >= 0) & (s <= 1), "top_share must be in [0, 1]"),
    ),
)

TRANSFERENCE_CI = DataContract(
    name="transference_bootstrap_ci",
    columns=("pair", "estimate", "lower", "upper", "std_error", "n_resamples"),
    validators=(
        ValueValidator("estimate", lambda s: s >= 0, "estimate must be non-negative"),
        ValueValidator("lower", lambda s: s >= 0, "lower must be non-negative"),
        ValueValidator("upper", lambda s: s >= 0, "upper must be non-negative"),
        ValueValidator("std_error", lambda s: s >= 0, "std_error must be non-negative"),
        ValueValidator("n_resamples", lambda s: s > 0, "n_resamples must be positive"),
    ),
)

# ---------------------------------------------------------------------------
# Assortment optimization contracts
# ---------------------------------------------------------------------------

ASSORTMENT_SOLUTION = DataContract(
    name="assortment_solution",
    columns=("stockcode", "selected", "revenue", "rank"),
    validators=(
        ValueValidator("revenue", lambda s: s >= 0, "revenue must be non-negative"),
        ValueValidator("rank", lambda s: s >= 0, "rank must be non-negative"),
    ),
)

ASSORTMENT_SCENARIO = DataContract(
    name="assortment_scenario",
    columns=(
        "scenario_id",
        "method",
        "n_skus",
        "kept_revenue",
        "recovered_revenue",
        "lost_revenue",
        "unmet_demand",
        "expected_revenue",
        "coverage",
        "recovery_rate",
    ),
    validators=(
        ValueValidator("n_skus", lambda s: s > 0, "n_skus must be positive"),
        ValueValidator("kept_revenue", lambda s: s >= 0, "kept_revenue must be non-negative"),
        ValueValidator("recovered_revenue", lambda s: s >= 0, "recovered_revenue must be non-negative"),
        ValueValidator("lost_revenue", lambda s: s >= 0, "lost_revenue must be non-negative"),
        ValueValidator("unmet_demand", lambda s: s >= 0, "unmet_demand must be non-negative"),
        ValueValidator("expected_revenue", lambda s: s >= 0, "expected_revenue must be non-negative"),
        ValueValidator("coverage", lambda s: (s >= 0) & (s <= 1), "coverage must be in [0, 1]"),
        ValueValidator("recovery_rate", lambda s: (s >= 0) & (s <= 1), "recovery_rate must be in [0, 1]"),
    ),
)

ASSORTMENT_EVALUATION = DataContract(
    name="assortment_evaluation",
    columns=(
        "scenario_id",
        "method",
        "selected_skus",
        "kept_revenue",
        "recovered_revenue",
        "lost_revenue",
        "unmet_demand",
        "expected_revenue",
        "coverage",
        "recovery_rate",
        "n_categories_covered",
        "n_categories_total",
    ),
    validators=(
        ValueValidator("selected_skus", lambda s: s > 0, "selected_skus must be positive"),
        ValueValidator("kept_revenue", lambda s: s >= 0, "kept_revenue must be non-negative"),
        ValueValidator("recovered_revenue", lambda s: s >= 0, "recovered_revenue must be non-negative"),
        ValueValidator("lost_revenue", lambda s: s >= 0, "lost_revenue must be non-negative"),
        ValueValidator("unmet_demand", lambda s: s >= 0, "unmet_demand must be non-negative"),
        ValueValidator("expected_revenue", lambda s: s >= 0, "expected_revenue must be non-negative"),
        ValueValidator("coverage", lambda s: (s >= 0) & (s <= 1), "coverage must be in [0, 1]"),
        ValueValidator("recovery_rate", lambda s: (s >= 0) & (s <= 1), "recovery_rate must be in [0, 1]"),
        ValueValidator("n_categories_covered", lambda s: s >= 0, "n_categories_covered must be non-negative"),
        ValueValidator("n_categories_total", lambda s: s >= 0, "n_categories_total must be non-negative"),
    ),
)

# ---------------------------------------------------------------------------
# CDT (Customer Decision Tree) contracts
# ---------------------------------------------------------------------------

CDT_ATTRIBUTES = DataContract(
    name="cdt_attributes",
    columns=(
        "stockcode",
        "price_tier",
        "velocity_tier",
        "seasonality_class",
        "basket_size_affinity",
        "substitution_tier",
    ),
)

CDT_ASSIGNMENTS = DataContract(
    name="cdt_cluster_assignments",
    columns=("stockcode", "cluster"),
)

CDT_QUALITY = DataContract(
    name="cdt_cluster_quality",
    columns=("cluster", "size", "within_similarity", "across_similarity"),
    validators=(
        ValueValidator("size", lambda s: s > 0, "size must be positive"),
        ValueValidator("within_similarity", lambda s: s.isna() | ((s >= -1) & (s <= 1)), "within_similarity must be in [-1, 1] or NaN"),
        ValueValidator("across_similarity", lambda s: s.isna() | ((s >= -1) & (s <= 1)), "across_similarity must be in [-1, 1] or NaN"),
    ),
)

CDT_OPTIMAL_K = DataContract(
    name="cdt_optimal_k",
    columns=("n_clusters", "silhouette"),
    validators=(
        ValueValidator("n_clusters", lambda s: s > 0, "n_clusters must be positive"),
        ValueValidator("silhouette", lambda s: (s >= -1) & (s <= 1), "silhouette must be in [-1, 1]"),
    ),
)

CDT_COMMUNITY = DataContract(
    name="cdt_communities",
    columns=("stockcode", "community"),
)

CDT_TREE_NODES = DataContract(
    name="cdt_tree_nodes",
    columns=("node_id", "name", "attribute", "attribute_value", "size", "is_leaf", "similarity_within", "parent_id"),
    validators=(
        ValueValidator("size", lambda s: s >= 0, "size must be non-negative"),
        ValueValidator("is_leaf", lambda s: s.isin([0, 1]), "is_leaf must be boolean (0/1)"),
        ValueValidator("similarity_within", lambda s: s.isna() | ((s >= 0) & (s <= 1)), "similarity_within must be in [0, 1] or NaN"),
    ),
)

CDT_TREE_PRODUCTS = DataContract(
    name="cdt_tree_products",
    columns=("node_id", "stockcode"),
)

CDT_VALIDATION = DataContract(
    name="cdt_validation",
    columns=("method", "metric", "value"),
)

CDT_TREE_SCORE = DataContract(
    name="cdt_tree_score",
    columns=("metric", "value"),
)

# ---------------------------------------------------------------------------
# Segmentation contracts
# ---------------------------------------------------------------------------

RFM_FEATURES = DataContract(
    name="rfm_features",
    columns=(
        "customer_id",
        "recency_days",
        "frequency",
        "monetary",
        "avg_order_value",
        "max_order_value",
        "n_items",
        "n_unique_products",
        "n_unique_categories",
        "first_purchase",
        "last_purchase",
        "avg_price_paid",
        "std_order_value",
        "customer_lifetime_days",
        "purchase_interval",
        "items_per_order",
        "revenue_per_item",
        "order_value_cv",
        "recency_segment",
        "frequency_segment",
        "monetary_segment",
    ),
    validators=(
        ValueValidator("recency_days", lambda s: s >= 0, "recency_days must be non-negative"),
        ValueValidator("frequency", lambda s: s > 0, "frequency must be positive"),
        ValueValidator("monetary", lambda s: s >= 0, "monetary must be non-negative"),
        ValueValidator("avg_order_value", lambda s: s >= 0, "avg_order_value must be non-negative"),
        ValueValidator("max_order_value", lambda s: s >= 0, "max_order_value must be non-negative"),
        ValueValidator("n_items", lambda s: s > 0, "n_items must be positive"),
        ValueValidator("n_unique_products", lambda s: s > 0, "n_unique_products must be positive"),
        ValueValidator("customer_lifetime_days", lambda s: s >= 0, "customer_lifetime_days must be non-negative"),
        ValueValidator("avg_price_paid", lambda s: s > 0, "avg_price_paid must be positive"),
        ValueValidator("order_value_cv", lambda s: s >= 0, "order_value_cv must be non-negative"),
    ),
)

RFM_SEGMENTS = DataContract(
    name="rfm_segments",
    columns=(
        "customer_id",
        "recency_days",
        "frequency",
        "monetary",
        "recency_score",
        "frequency_score",
        "monetary_score",
        "rfm_score",
        "segment",
        "cluster",
    ),
    validators=(
        ValueValidator("recency_days", lambda s: s >= 0, "recency_days must be non-negative"),
        ValueValidator("frequency", lambda s: s > 0, "frequency must be positive"),
        ValueValidator("monetary", lambda s: s >= 0, "monetary must be non-negative"),
        ValueValidator("recency_score", lambda s: s.isin([1, 2, 3, 4]), "recency_score must be in {1,2,3,4}"),
        ValueValidator("frequency_score", lambda s: s.isin([1, 2, 3, 4]), "frequency_score must be in {1,2,3,4}"),
        ValueValidator("monetary_score", lambda s: s.isin([1, 2, 3, 4]), "monetary_score must be in {1,2,3,4}"),
    ),
)

BEHAVIORAL_FEATURES = DataContract(
    name="behavioral_features",
    columns=(
        "customer_id",
        "days_active",
        "purchase_frequency",
        "avg_days_between",
        "total_revenue",
        "avg_order_value",
        "revenue_std",
        "n_products",
        "n_categories",
        "avg_basket_size",
        "max_basket_size",
        "avg_price",
        "price_cv",
        "weekend_ratio",
    ),
    validators=(
        ValueValidator("days_active", lambda s: s >= 0, "days_active must be non-negative"),
        ValueValidator("purchase_frequency", lambda s: s >= 0, "purchase_frequency must be non-negative"),
        ValueValidator("total_revenue", lambda s: s >= 0, "total_revenue must be non-negative"),
        ValueValidator("avg_order_value", lambda s: s >= 0, "avg_order_value must be non-negative"),
        ValueValidator("n_products", lambda s: s > 0, "n_products must be positive"),
        ValueValidator("n_categories", lambda s: s > 0, "n_categories must be positive"),
        ValueValidator("avg_basket_size", lambda s: s > 0, "avg_basket_size must be positive"),
        ValueValidator("avg_price", lambda s: s > 0, "avg_price must be positive"),
        ValueValidator("price_cv", lambda s: s >= 0, "price_cv must be non-negative"),
        ValueValidator("weekend_ratio", lambda s: (s >= 0) & (s <= 1), "weekend_ratio must be in [0, 1]"),
    ),
)

BEHAVIORAL_SEGMENTS = DataContract(
    name="behavioral_segments",
    columns=(
        "customer_id",
        "cluster",
        "segment",
        "cluster_distance",
        "cluster_confidence",
    ),
    validators=(
        ValueValidator("cluster_distance", lambda s: s >= 0, "cluster_distance must be non-negative"),
        ValueValidator("cluster_confidence", lambda s: (s >= 0) & (s <= 1), "cluster_confidence must be in [0, 1]"),
    ),
)

SURVIVAL_PREDICTIONS = DataContract(
    name="survival_predictions",
    columns=("customer_id", "survival_prob", "churn_risk"),
    validators=(
        ValueValidator("survival_prob", lambda s: (s >= 0) & (s <= 1), "survival_prob must be in [0, 1]"),
        ValueValidator("churn_risk", lambda s: (s >= 0) & (s <= 1), "churn_risk must be in [0, 1]"),
    ),
)

SURVIVAL_DIAGNOSTICS = DataContract(
    name="survival_diagnostics",
    columns=("metric", "value"),
)

VALUE_BASED_SEGMENTS = DataContract(
    name="value_based_segments",
    columns=(
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "avg_order",
        "n_products",
        "lifetime_days",
        "future_revenue",
        "predicted_clv",
        "value_segment",
    ),
    validators=(
        ValueValidator("recency", lambda s: s >= 0, "recency must be non-negative"),
        ValueValidator("frequency", lambda s: s >= 0, "frequency must be non-negative"),
        ValueValidator("monetary", lambda s: s >= 0, "monetary must be non-negative"),
        ValueValidator("avg_order", lambda s: s >= 0, "avg_order must be non-negative"),
        ValueValidator("n_products", lambda s: s >= 0, "n_products must be non-negative"),
        ValueValidator("lifetime_days", lambda s: s >= 0, "lifetime_days must be non-negative"),
        ValueValidator("future_revenue", lambda s: s >= 0, "future_revenue must be non-negative"),
        ValueValidator("predicted_clv", lambda s: s >= 0, "predicted_clv must be non-negative"),
    ),
)

CLUSTER_QUALITY = DataContract(
    name="cluster_quality",
    columns=(
        "metric",
        "value",
    ),
)

CLUSTER_STABILITY = DataContract(
    name="cluster_stability",
    columns=(
        "metric",
        "value",
    ),
)

# ---------------------------------------------------------------------------
# Pricing contracts
# ---------------------------------------------------------------------------

ELASTICITY = DataContract(
    name="elasticity",
    columns=(
        "stockcode",
        "elasticity",
        "r_squared",
        "p_value",
        "std_err",
        "ci_lower",
        "ci_upper",
        "n_obs",
        "avg_price",
        "avg_weekly_qty",
        "price_cv",
    ),
    validators=(
        ValueValidator("r_squared", lambda s: (s >= 0) & (s <= 1), "r_squared must be in [0, 1]"),
        ValueValidator("p_value", lambda s: (s >= 0) & (s <= 1), "p_value must be in [0, 1]"),
        ValueValidator("std_err", lambda s: s >= 0, "std_err must be non-negative"),
        ValueValidator("n_obs", lambda s: s > 0, "n_obs must be positive"),
        ValueValidator("avg_price", lambda s: s > 0, "avg_price must be positive"),
        ValueValidator("avg_weekly_qty", lambda s: s >= 0, "avg_weekly_qty must be non-negative"),
        ValueValidator("price_cv", lambda s: s >= 0, "price_cv must be non-negative"),
    ),
)

HIERARCHICAL_ELASTICITY = DataContract(
    name="hierarchical_elasticity",
    columns=(
        "stockcode",
        "category",
        "elasticity_ols",
        "elasticity_cat",
        "elasticity_shrunk",
        "shrink_weight",
        "r_squared",
        "p_value",
        "n_obs",
        "avg_price",
        "std_err",
    ),
    validators=(
        ValueValidator("shrink_weight", lambda s: (s >= 0) & (s <= 1), "shrink_weight must be in [0, 1]"),
        ValueValidator("r_squared", lambda s: (s >= 0) & (s <= 1), "r_squared must be in [0, 1]"),
        ValueValidator("p_value", lambda s: (s >= 0) & (s <= 1), "p_value must be in [0, 1]"),
        ValueValidator("n_obs", lambda s: s > 0, "n_obs must be positive"),
        ValueValidator("avg_price", lambda s: s > 0, "avg_price must be positive"),
        ValueValidator("std_err", lambda s: s >= 0, "std_err must be non-negative"),
    ),
)

CROSS_ELASTICITY = DataContract(
    name="cross_elasticity",
    columns=(
        "product_a",
        "product_b",
        "own_elasticity",
        "own_elasticity_se",
        "own_elasticity_p",
        "cross_elasticity",
        "cross_elasticity_se",
        "cross_elasticity_p",
        "r_squared",
        "n_obs",
        "avg_price_a",
        "avg_price_b",
    ),
    validators=(
        ValueValidator("own_elasticity_se", lambda s: s >= 0, "own_elasticity_se must be non-negative"),
        ValueValidator("cross_elasticity_se", lambda s: s >= 0, "cross_elasticity_se must be non-negative"),
        ValueValidator("r_squared", lambda s: (s >= 0) & (s <= 1), "r_squared must be in [0, 1]"),
        ValueValidator("n_obs", lambda s: s > 0, "n_obs must be positive"),
        ValueValidator("avg_price_a", lambda s: s > 0, "avg_price_a must be positive"),
        ValueValidator("avg_price_b", lambda s: s > 0, "avg_price_b must be positive"),
    ),
)

KVI_SCORES = DataContract(
    name="kvi_scores",
    columns=(
        "stockcode",
        "kvi_score",
        "category",
        "total_revenue",
        "basket_penetration",
        "trip_incidence",
        "abs_elasticity",
    ),
    validators=(
        ValueValidator("kvi_score", lambda s: s >= 0, "kvi_score must be non-negative"),
        ValueValidator("total_revenue", lambda s: s >= 0, "total_revenue must be non-negative"),
        ValueValidator("basket_penetration", lambda s: (s >= 0) & (s <= 1), "basket_penetration must be in [0, 1]"),
        ValueValidator("trip_incidence", lambda s: (s >= 0) & (s <= 1), "trip_incidence must be in [0, 1]"),
        ValueValidator("abs_elasticity", lambda s: s >= 0, "abs_elasticity must be non-negative"),
    ),
)

PRICE_CURVE_1D = DataContract(
    name="price_curve_1d",
    columns=(
        "stockcode",
        "product_name",
        "category",
        "brand",
        "median_price",
        "pack_size_numeric",
        "price_per_unit",
        "tier",
        "tier_label",
        "has_violation",
    ),
    validators=(
        ValueValidator("median_price", lambda s: s > 0, "median_price must be positive"),
        ValueValidator("pack_size_numeric", lambda s: s > 0, "pack_size_numeric must be positive"),
        ValueValidator("price_per_unit", lambda s: s > 0, "price_per_unit must be positive"),
        ValueValidator("has_violation", lambda s: s.isin([True, False]), "has_violation must be boolean"),
    ),
)

PRICE_CURVE_MULTI = DataContract(
    name="price_curve_multi",
    columns=(
        "stockcode",
        "product_name",
        "category",
        "brand",
        "median_price",
        "pack_size_numeric",
        "price_per_unit",
        "basket_penetration",
        "trip_incidence",
        "elasticity",
        "margin_per_unit",
        "tier",
        "tier_label",
        "has_violation",
    ),
    validators=(
        ValueValidator("median_price", lambda s: s > 0, "median_price must be positive"),
        ValueValidator("pack_size_numeric", lambda s: s > 0, "pack_size_numeric must be positive"),
        ValueValidator("price_per_unit", lambda s: s > 0, "price_per_unit must be positive"),
        ValueValidator("basket_penetration", lambda s: (s >= 0) & (s <= 1), "basket_penetration must be in [0, 1]"),
        ValueValidator("trip_incidence", lambda s: (s >= 0) & (s <= 1), "trip_incidence must be in [0, 1]"),
        ValueValidator("has_violation", lambda s: s.isin([True, False]), "has_violation must be boolean"),
    ),
)

IV_ELASTICITY = DataContract(
    name="iv_elasticity",
    columns=(
        "stockcode",
        "iv_elasticity",
        "iv_elasticity_se",
        "iv_elasticity_p",
        "iv_r_squared",
        "first_stage_f",
        "weak_instrument",
        "n_obs",
        "avg_price",
        "avg_weekly_qty",
        "avg_instrument",
    ),
    validators=(
        ValueValidator("iv_elasticity_se", lambda s: s >= 0, "iv_elasticity_se must be non-negative"),
        ValueValidator("iv_elasticity_p", lambda s: (s >= 0) & (s <= 1), "iv_elasticity_p must be in [0, 1]"),
        ValueValidator("iv_r_squared", lambda s: (s >= 0) & (s <= 1), "iv_r_squared must be in [0, 1]"),
        ValueValidator("first_stage_f", lambda s: s >= 0, "first_stage_f must be non-negative"),
        ValueValidator("weak_instrument", lambda s: s.isin([True, False]), "weak_instrument must be boolean"),
        ValueValidator("n_obs", lambda s: s > 0, "n_obs must be positive"),
        ValueValidator("avg_price", lambda s: s > 0, "avg_price must be positive"),
        ValueValidator("avg_weekly_qty", lambda s: s >= 0, "avg_weekly_qty must be non-negative"),
    ),
)

RDD_ELASTICITY = DataContract(
    name="rdd_elasticity",
    columns=(
        "product_a",
        "product_b",
        "threshold_price",
        "cross_elasticity",
        "n_obs",
        "bandwidth",
    ),
    validators=(
        ValueValidator("threshold_price", lambda s: s > 0, "threshold_price must be positive"),
        ValueValidator("n_obs", lambda s: s > 0, "n_obs must be positive"),
        ValueValidator("bandwidth", lambda s: s > 0, "bandwidth must be positive"),
    ),
)

SYNTHETIC_CONTROL = DataContract(
    name="synthetic_control",
    columns=(
        "treatment_product",
        "metric",
        "value",
    ),
)

CAUSAL_UPLIFT = DataContract(
    name="causal_uplift",
    columns=(
        "customer_id",
        "uplift",
        "treatment",
        "propensity",
    ),
    validators=(
        ValueValidator("treatment", lambda s: s.isin([0, 1]), "treatment must be binary (0/1)"),
        ValueValidator("propensity", lambda s: (s >= 0) & (s <= 1), "propensity must be in [0, 1]"),
    ),
)
