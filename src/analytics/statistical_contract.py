"""Statistical Contract Layer.

Provides a unified interface for declaring and validating statistical properties
of analytical outputs beyond basic DataContracts. Ensures every result carries:
- Estimate + CI + n + coverage + stability + assumptions + limitations + action
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional

import pandas as pd


class StatisticalClaimType(Enum):
    """Type of statistical claim being made."""
    DESCRIPTIVE = "descriptive"           # Simple summary statistics
    OBSERVATIONAL = "observational"       # Correlational/associational
    CAUSAL = "causal"                     # Causal inference claim
    PREDICTIVE = "predictive"             # Forecast/prediction


class ReliabilityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


@dataclass
class StatisticalAssumption:
    """A statistical assumption with validation."""
    name: str
    description: str
    check_fn: Optional[Callable[[Any], bool]] = None
    severity: Literal["critical", "warning", "info"] = "warning"
    is_satisfied: Optional[bool] = None
    evidence: Optional[str] = None


@dataclass
class StatisticalContract:
    """Complete statistical contract for an analytical output.

    Every analytical function should declare its contract and validate
    its output against it before returning.
    """
    # Core identification
    function_name: str
    claim_type: StatisticalClaimType

    # Estimate specification
    estimate_name: str
    estimate_type: Literal["point", "interval", "distribution"]
    estimate_unit: str  # e.g., "elasticity", "revenue", "probability"

    # Statistical properties (what the output MUST contain)
    required_columns: List[str]
    column_descriptions: Dict[str, str]

    # Validity conditions
    assumptions: List[StatisticalAssumption] = field(default_factory=list)

    # Limitations and scope
    limitations: List[str] = field(default_factory=list)
    scope_conditions: List[str] = field(default_factory=list)  # e.g., "requires n > 100"

    # Actionability
    recommended_action: Optional[str] = None
    action_conditions: Dict[str, Any] = field(default_factory=dict)

    # Reliability
    reliability_requirements: Dict[str, Any] = field(default_factory=dict)

    def validate_output(self, output: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        """Validate output against this contract.

        Returns:
            (validated_df, warnings_list)
        """
        warnings = []

        # Check required columns
        missing = [c for c in self.required_columns if c not in output.columns]
        if missing:
            raise ValueError(f"{self.function_name}: Missing required columns: {missing}")

        # Check column types and ranges
        for col in self.required_columns:
            if col not in output.columns:
                continue

            # Basic validation based on column name patterns
            if col.endswith("_ci_lower") or col.endswith("_ci_upper"):
                # CI should be numeric and non-negative (for positive quantities)
                if not pd.api.types.is_numeric_dtype(output[col]):
                    warnings.append(f"{col}: Expected numeric type")
            elif col.endswith("_ci_status"):
                valid = {"valid", "insufficient_resamples", "fit_failed", "unknown"}
                invalid = set(output[col].unique()) - valid
                if invalid:
                    warnings.append(f"{col}: Invalid status values: {invalid}")
            elif col.endswith("_p_value") or col.endswith("_pvalue"):
                # p-values should be in [0, 1]
                invalid = output[col][(output[col] < 0) | (output[col] > 1)].notna().any()
                if invalid:
                    warnings.append(f"{col}: p-values outside [0,1]")
            elif col.endswith("_r_squared") or col.endswith("_rsquared"):
                invalid = output[col][(output[col] < 0) | (output[col] > 1)].notna().any()
                if invalid:
                    warnings.append(f"{col}: R-squared outside [0,1]")

        # Check assumptions
        for assumption in self.assumptions:
            if assumption.check_fn is not None:
                try:
                    satisfied = assumption.check_fn()
                    assumption.is_satisfied = bool(satisfied)
                    if not satisfied and assumption.severity == "critical":
                        raise ValueError(f"Critical assumption violated: {assumption.name}")
                    elif not satisfied and assumption.severity == "warning":
                        warnings.append(f"Assumption violated ({assumption.severity}): {assumption.name} - {assumption.description}")
                except Exception as e:
                    warnings.append(f"Assumption check failed ({assumption.name}): {e}")

        # Check scope conditions (informational, not warnings)
        scope_info = []
        for condition in self.scope_conditions:
            scope_info.append(f"Scope condition: {condition}")

        return output, warnings, scope_info

    def to_summary(self) -> Dict[str, Any]:
        """Generate a human-readable summary of this contract."""
        return {
            "function": self.function_name,
            "claim_type": self.claim_type.value,
            "estimate": f"{self.estimate_name} ({self.estimate_type}, {self.estimate_unit})",
            "required_columns": self.required_columns,
            "assumptions": [a.name for a in self.assumptions],
            "limitations": self.limitations,
            "scope_conditions": self.scope_conditions,
            "recommended_action": self.recommended_action,
        }


# Common statistical contracts that can be reused

ELASTICITY_CONTRACT = StatisticalContract(
    function_name="estimate_loglog_elasticity",
    claim_type=StatisticalClaimType.OBSERVATIONAL,
    estimate_name="price_elasticity",
    estimate_type="point",
    estimate_unit="elasticity",
    required_columns=[
        "stockcode", "elasticity", "std_err", "p_value", "r_squared",
        "ci_lower", "ci_upper", "n_obs", "avg_price", "avg_weekly_qty", "price_cv"
    ],
    column_descriptions={
        "elasticity": "Point estimate of price elasticity (observed, NOT causal)",
        "std_err": "Heteroskedasticity-robust standard error (HC3)",
        "p_value": "p-value for null elasticity = 0",
        "ci_lower": "Lower bound of 95% CI",
        "ci_upper": "Upper bound of 95% CI",
        "n_obs": "Number of weekly observations used",
    },
    assumptions=[
        StatisticalAssumption(
            name="exogeneity",
            description="Price is exogenous (uncorrelated with demand shocks). VIOLATED in practice.",
            severity="critical",
        ),
        StatisticalAssumption(
            name="log_linearity",
            description="Log-log linear relationship between price and quantity",
            severity="warning",
        ),
        StatisticalAssumption(
            name="no_simultaneity",
            description="No simultaneous determination of price and quantity",
            severity="critical",
        ),
    ],
    limitations=[
        "Does NOT estimate causal elasticity. Price endogeneity not addressed.",
        "Requires sufficient price variation (CV > threshold).",
        "Time fixed effects included but may not capture all confounders.",
        "Small sample bias possible for low n_obs.",
    ],
    scope_conditions=[
        "Requires min_periods weekly observations (default 10)",
        "Requires price_cv > min_price_variation (default 0.05)",
        "Only valid for products with sufficient price variation",
    ],
    recommended_action="Use ONLY for descriptive price response analysis. For causal inference, use IV/RDD with valid instruments.",
)


CLV_CONTRACT = StatisticalContract(
    function_name="predict_clv_bg_nbd",
    claim_type=StatisticalClaimType.PREDICTIVE,
    estimate_name="predicted_clv",
    estimate_type="point",
    estimate_unit="currency",
    required_columns=[
        "customer_id", "frequency", "recency", "T", "monetary_value",
        "predicted_purchases", "expected_avg_value", "predicted_clv",
        "ci_lower", "ci_upper", "ci_status", "p_alive", "clv_segment"
    ],
    column_descriptions={
        "predicted_clv": "Point estimate of CLV over prediction horizon",
        "ci_lower": "Lower bound of 95% bootstrap CI",
        "ci_upper": "Upper bound of 95% bootstrap CI",
        "ci_status": "CI validity status: valid/insufficient_resamples/fit_failed",
        "p_alive": "Probability customer is still active",
    },
    assumptions=[
        StatisticalAssumption(
            name="stationarity",
            description="Purchase process and spend distribution are stationary over time",
            severity="warning",
        ),
        StatisticalAssumption(
            name="independent_customers",
            description="Customer behaviors are independent",
            severity="warning",
        ),
        StatisticalAssumption(
            name="gamma_gamma_independence",
            description="Frequency and monetary value are independent (Gamma-Gamma assumption)",
            severity="critical",
        ),
    ],
    limitations=[
        "BG/NBD assumes no covariates; all customers share same parameters",
        "Bootstrap CI requires >=15 successful resamples per customer",
        "Gamma-Gamma independence assumption often violated in practice",
        "Predictions degrade for very long horizons",
    ],
    scope_conditions=[
        "Requires >=10 repeat customers for model fitting",
        "Requires monetary_value > 0 for Gamma-Gamma",
        "Prediction horizon should be <= observation window",
    ],
    recommended_action="Use for customer prioritization and cohort analysis. Validate against holdout.",
)


ASSORTMENT_CONTRACT = StatisticalContract(
    function_name="optimize_assortment_milp",
    claim_type=StatisticalClaimType.PREDICTIVE,
    estimate_name="optimal_assortment",
    estimate_type="set",
    estimate_unit="SKUs",
    required_columns=[
        "stockcode", "selected", "revenue", "rank"
    ],
    column_descriptions={
        "selected": "1 if SKU is in optimal assortment, 0 otherwise",
        "revenue": "SKU's revenue contribution",
        "rank": "Revenue rank among selected SKUs",
    },
    assumptions=[
        StatisticalAssumption(
            name="demand_transference_known",
            description="Transference matrix accurately captures demand reallocation",
            severity="critical",
        ),
        StatisticalAssumption(
            name="constant_revenue",
            description="Per-SKU revenue remains constant after assortment change",
            severity="critical",
        ),
        StatisticalAssumption(
            name="recovery_margin_known",
            description="Recovery margin (recovery_margin parameter) is known and constant",
            severity="warning",
        ),
    ],
    limitations=[
        "Uses observed switching as proxy for demand transference (NOT causal)",
        "Recovery_margin is a heuristic, not estimated from data",
        "Does not account for inventory costs, cannibalization, or long-term effects",
        "MILP may not find global optimum for large problems (time limit)",
    ],
    scope_conditions=[
        "Requires sufficient transactions for transference estimation",
        "Max SKUs constraint must be feasible with category coverage",
    ],
    recommended_action="Use as decision support, not automated decision. Validate with simulation.",
)


# Utility function to attach contract to output
def attach_contract(
    output: pd.DataFrame,
    contract: StatisticalContract,
    reliability: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Attach contract metadata to output DataFrame."""
    import json
    result = output.copy()
    meta = {
        "statistical_contract": contract.function_name,
        "claim_type": contract.claim_type.value,
        "estimate": contract.estimate_name,
        "limitations": contract.limitations,
        "scope_conditions": contract.scope_conditions,
        "recommended_action": contract.recommended_action,
        "reliability": reliability or {},
    }
    result["_statistical_contract"] = [json.dumps(meta)] * len(result)
    return result


# Registry of all contracts
CONTRACT_REGISTRY: Dict[str, StatisticalContract] = {
    "estimate_loglog_elasticity": ELASTICITY_CONTRACT,
    "predict_clv_bg_nbd": CLV_CONTRACT,
    "optimize_assortment_milp": ASSORTMENT_CONTRACT,
}
def get_contract(name: str) -> Optional[StatisticalContract]:
    """Get a contract by name."""
    return CONTRACT_REGISTRY.get(name)


def register_contract(contract: StatisticalContract) -> None:
    """Register a new contract."""
    CONTRACT_REGISTRY[contract.function_name] = contract


def validate_against_contract(
    output: pd.DataFrame,
    contract_name: str,
) -> tuple[pd.DataFrame, List[str], List[str]]:
    """Validate output against a registered contract.

    Returns:
        (validated_df, warnings, scope_info)
    """
    contract = get_contract(contract_name)
    if contract is None:
        raise ValueError(f"No contract registered for {contract_name}")
    return contract.validate_output(output)
