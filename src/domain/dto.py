"""Domain data transfer objects for the Market Basket Analysis application."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Dict, List, Optional

import pandas as pd


class AnalysisMode(StrEnum):
    """Supported analysis modes."""

    ASSOCIATION_RULES = "association_rules"
    COPURCHASE = "copurchase"
    ADDON = "addon"
    SWITCHING = "switching"
    CHOICE_PREDICTION = "choice_prediction"
    CDT_BUILDER = "cdt_builder"
    CDT_BENCHMARK = "cdt_benchmark"
    DEMAND_TRANSFERENCE = "demand_transference"
    ASSORTMENT_OPTIMIZER = "assortment_optimizer"
    CUSTOMER_SEGMENTATION = "customer_segmentation"
    PRODUCT_PERFORMANCE = "product_performance"
    COHORT_ANALYSIS = "cohort_analysis"
    PROMOTIONAL_ANALYTICS = "promotional_analytics"
    ELASTICITY_ANALYSIS = "elasticity_analysis"
    KVI_IDENTIFICATION = "kvi_identification"
    PRICE_CURVE_DIAGNOSTICS = "price_curve_diagnostics"
    PROMO_UPLIFT_MODELING = "promo_uplift_modeling"
    ELASTICITY_BENCHMARK = "elasticity_benchmark"


class PipelineStage(StrEnum):
    """Pipeline execution stages."""

    DATA_LOAD = "data_load"
    DATA_VALIDATE = "data_validate"
    BASKET_MATRIX = "basket_matrix"
    FREQUENT_ITEMSETS = "frequent_itemsets"
    ASSOCIATION_RULES = "association_rules"
    SIMILARITY_MATRIX = "similarity_matrix"
    HIERARCHICAL_CLUSTERING = "hierarchical_clustering"
    CDT_TREE_BUILD = "cdt_tree_build"
    BEHAVIORAL_MATRICES = "behavioral_matrices"
    CUSTOMER_FEATURES = "customer_features"
    SEGMENTATION = "segmentation"
    CLV_PREDICTION = "clv_prediction"
    ELASTICITY_ESTIMATION = "elasticity_estimation"
    KVI_SCORING = "kvi_scoring"
    PRICE_TIERS = "price_tiers"
    PROMO_DETECTION = "promo_detection"
    UPLIFT_MODELING = "uplift_modeling"
    DEMAND_TRANSFERENCE = "demand_transference"
    ASSORTMENT_OPTIMIZATION = "assortment_optimization"


@dataclass
class DataSummary:
    """Summary statistics for transaction data."""

    n_transactions: int
    n_customers: int
    n_products: int
    date_range: tuple
    total_revenue: float
    avg_basket_size: float
    avg_basket_value: float

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "DataSummary":
        """Create summary from transaction DataFrame."""
        df = df.copy()
        df["revenue"] = df["price"] * df["quantity"]
        basket_revenue = df.groupby("transaction_id")["revenue"].sum()
        basket_size = df.groupby("transaction_id")["quantity"].sum()

        return cls(
            n_transactions=df["transaction_id"].nunique(),
            n_customers=df["customer_id"].nunique(),
            n_products=df["stockcode"].nunique(),
            date_range=(df["date"].min(), df["date"].max()),
            total_revenue=df["revenue"].sum(),
            avg_basket_size=basket_size.mean(),
            avg_basket_value=basket_revenue.mean(),
        )


@dataclass
class PipelineResult:
    """Result of a pipeline stage execution."""

    stage: PipelineStage
    success: bool
    data: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_empty(self) -> bool:
        """Check if result data is empty."""
        if self.data is None:
            return True
        if isinstance(self.data, pd.DataFrame):
            return self.data.empty
        if isinstance(self.data, (list, dict, set)):
            return len(self.data) == 0
        return False


@dataclass
class AnalysisRequest:
    """Request to run an analysis."""

    mode: AnalysisMode
    transactions_df: pd.DataFrame
    config: Dict[str, Any]
    product_lookup: Dict[str, str]
    previous_results: Optional[Dict[PipelineStage, PipelineResult]] = None

    def __post_init__(self):
        if self.previous_results is None:
            self.previous_results = {}


@dataclass
class AnalysisResponse:
    """Response from an analysis execution."""

    mode: AnalysisMode
    results: Dict[PipelineStage, PipelineResult]
    summary: Dict[str, Any]
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    exports: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0

    def get_result(self, stage: PipelineStage) -> Optional[PipelineResult]:
        """Get result for a specific stage."""
        return self.results.get(stage)

    def get_data(self, stage: PipelineStage) -> Any:
        """Get data for a specific stage."""
        result = self.get_result(stage)
        return result.data if result else None


@dataclass
class ValidationResult:
    """Data validation result."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add an error (makes validation invalid)."""
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning."""
        self.warnings.append(message)

    def add_info(self, message: str) -> None:
        """Add an info message."""
        self.info.append(message)


@dataclass
class ColumnMapping:
    """Column mapping for data loading."""

    date: str
    transaction_id: str
    stockcode: str
    product: str
    customer_id: str
    price: str
    quantity: str
    category: Optional[str] = None
    brand: Optional[str] = None
    size: Optional[str] = None
    flavor: Optional[str] = None


REQUIRED_COLUMNS = [
    "date",
    "transaction_id",
    "stockcode",
    "product",
    "customer_id",
    "price",
    "quantity",
]


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    size_bytes: int
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = datetime.now()
