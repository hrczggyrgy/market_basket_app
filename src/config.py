"""Configuration using Pydantic Settings with environment variable support."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FPGrowthConfig(BaseSettings):
    """FP-Growth algorithm parameters."""

    model_config = SettingsConfigDict(env_prefix="FPGROWTH_")

    min_support: float = Field(default=0.002, gt=0, lt=1, description="Minimum support threshold")
    min_confidence: float = Field(
        default=0.1, gt=0, lt=1, description="Minimum confidence threshold"
    )
    max_itemset_len: int = Field(default=3, ge=2, le=6, description="Maximum itemset length")
    min_lift: float = Field(default=1.2, gt=0, description="Minimum lift threshold")


class CDTConfig(BaseSettings):
    """Customer Decision Tree parameters."""

    model_config = SettingsConfigDict(env_prefix="CDT_")

    min_cluster_size: int = Field(
        default=3, ge=2, le=10, description="Minimum products per cluster"
    )
    quality_threshold: float = Field(
        default=0.6, gt=0, lt=1, description="Quality threshold vs unconstrained baseline"
    )
    split_criterion: Literal["mutual_info", "gini", "entropy", "mixed"] = Field(
        default="mutual_info", description="Attribute split criterion"
    )
    split_alpha: float = Field(
        default=0.5, ge=0, le=1, description="Entropy weight for mixed criterion"
    )
    min_cooccurrence: int = Field(
        default=5, ge=2, le=20, description="Minimum co-occurrence for similarity"
    )
    linkage_method: Literal["average", "complete", "single"] = Field(
        default="average", description="Hierarchical clustering linkage"
    )
    min_k: int = Field(default=2, ge=2, le=10, description="Minimum clusters for silhouette search")
    max_k: int = Field(
        default=15, ge=3, le=20, description="Maximum clusters for silhouette search"
    )
    community_method: Literal["none", "label_propagation", "louvain", "leiden"] = Field(
        default="louvain", description="Community detection method"
    )
    community_resolution: float = Field(
        default=1.0, ge=0.5, le=2.0, description="Community resolution"
    )
    graph_min_weight: float = Field(
        default=0.1, ge=0, le=0.5, description="Minimum edge weight for product graph"
    )
    graph_max_degree: int = Field(default=50, ge=10, le=100, description="Max edges per node")
    top_n_products: int = Field(
        default=50, ge=20, le=200, description="Top N products for large catalogs"
    )
    min_lift_bundling: float = Field(
        default=1.2, ge=1.0, le=3.0, description="Min lift for bundling"
    )
    max_substitution: float = Field(
        default=0.3, ge=0, le=0.5, description="Max substitution for bundling"
    )
    extract_from_text: bool = Field(
        default=False, description="Extract attributes from product text"
    )


class ElasticityConfig(BaseSettings):
    """Price elasticity estimation parameters."""

    model_config = SettingsConfigDict(env_prefix="ELASTICITY_")

    method: Literal["loglog_ols", "hierarchical_eb", "bayesian_hierarchical", "xgb"] = Field(
        default="loglog_ols", description="Elasticity estimation method"
    )
    min_periods: int = Field(
        default=10, ge=5, le=50, description="Minimum time periods per product"
    )
    min_price_variation: float = Field(
        default=0.05, gt=0, le=0.5, description="Minimum price coefficient of variation"
    )
    show_shap: bool = Field(default=False, description="Show SHAP values for XGBoost method")
    bayesian_mode: Literal["fast (ADVI)", "full (NUTS)"] = Field(
        default="fast (ADVI)", description="Bayesian sampling mode"
    )


class KVIConfig(BaseSettings):
    """Key Value Item identification parameters."""

    model_config = SettingsConfigDict(env_prefix="KVI_")

    method: Literal["xgb_importance", "rfm_elasticity"] = Field(
        default="xgb_importance", description="KVI scoring method"
    )
    top_k: int = Field(default=20, ge=10, le=100, description="Top K KVI products")
    margin_weighted: bool = Field(
        default=False, description="Weight by margin if cost data available"
    )


class PromoUpliftConfig(BaseSettings):
    """Promotional uplift modeling parameters."""

    model_config = SettingsConfigDict(env_prefix="PROMO_")

    drop_threshold: float = Field(
        default=0.15, ge=0.05, le=0.5, description="Price drop threshold for promo detection"
    )
    baseline_window: int = Field(default=28, ge=14, le=90, description="Baseline window in days")
    uplift_method: Literal["t_learner", "s_learner"] = Field(
        default="t_learner", description="Causal estimation method"
    )
    base_n_estimators: int = Field(
        default=200, ge=50, le=500, description="Base learner n_estimators"
    )
    base_max_depth: int = Field(default=5, ge=3, le=10, description="Base learner max_depth")
    propensity_stratification: bool = Field(default=True, description="Propensity stratification")


class PriceCurveConfig(BaseSettings):
    """Price curve diagnostics parameters."""

    model_config = SettingsConfigDict(env_prefix="PRICE_CURVE_")

    clustering_method: Literal["kmeans", "gmm"] = Field(
        default="kmeans", description="Clustering method"
    )
    n_tiers: int = Field(default=3, ge=2, le=5, description="Number of price tiers")
    multivariate: bool = Field(default=False, description="Use multivariate clustering")


class SegmentationConfig(BaseSettings):
    """Customer segmentation parameters."""

    model_config = SettingsConfigDict(env_prefix="SEGMENT_")

    method: Literal["behavioral", "rfm_quantile", "rfm_kmeans"] = Field(
        default="behavioral", description="Segmentation method"
    )
    n_segments: int = Field(default=8, ge=3, le=12, description="Number of segments")
    behavioral_clusters: int = Field(default=6, ge=3, le=10, description="Behavioral clusters")
    value_horizon: int = Field(default=90, ge=30, le=365, description="CLV horizon in days")


class PerformanceConfig(BaseSettings):
    """Performance and caching parameters."""

    model_config = SettingsConfigDict(env_prefix="PERF_")

    cache_ttl_seconds: int = Field(default=3600, ge=60, description="Cache TTL in seconds")
    max_cache_size_mb: int = Field(default=500, ge=50, description="Maximum cache size in MB")
    parallel_jobs: int = Field(default=-1, ge=-1, description="Parallel jobs (-1 = all cores)")
    use_polars: bool = Field(default=True, description="Use Polars for accelerated groupby")
    use_sparse_matrices: bool = Field(
        default=True, description="Use sparse matrices for basket/similarity"
    )
    chunk_size: int = Field(
        default=10000, ge=1000, description="Chunk size for large data processing"
    )


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG_")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    json_format: bool = Field(default=True)
    include_correlation_id: bool = Field(default=True)


class AppConfig(BaseSettings):
    """Main application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="allow",
    )

    # Sub-configs
    fpgrowth: FPGrowthConfig = Field(default_factory=FPGrowthConfig)
    cdt: CDTConfig = Field(default_factory=CDTConfig)
    elasticity: ElasticityConfig = Field(default_factory=ElasticityConfig)
    kvi: KVIConfig = Field(default_factory=KVIConfig)
    promo_uplift: PromoUpliftConfig = Field(default_factory=PromoUpliftConfig)
    price_curve: PriceCurveConfig = Field(default_factory=PriceCurveConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Global
    config_version: str = Field(default="1.0.0", description="Config version")
    data_dir: Path = Field(default=Path("data"))
    output_dir: Path = Field(default=Path("output"))
    temp_dir: Path = Field(default=Path("/tmp/market_basket_app"))

    # Sample data
    sample_n_customers: int = 200
    sample_seed: int = 42

    def ensure_dirs(self) -> None:
        """Create required directories."""
        for dir_path in [self.data_dir, self.output_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def to_legacy_dict(self) -> dict:
        """Convert to legacy flat dict for backward compatibility."""
        return {
            "min_support": self.fpgrowth.min_support,
            "min_confidence": self.fpgrowth.min_confidence,
            "max_itemset_len": self.fpgrowth.max_itemset_len,
            "min_lift": self.fpgrowth.min_lift,
            "analysis_mode": "Association Rules",
            "analysis_params": {},
            "run_analysis": False,
        }


# Backward compatibility alias
Config = AppConfig

# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get global config instance (singleton)."""
    global _config
    if _config is None:
        _config = AppConfig()
        _config.ensure_dirs()
    return _config


def reset_config() -> None:
    """Reset config (for testing)."""
    global _config
    _config = None
