"""Domain exceptions hierarchy for structured error handling."""

from typing import Any, Dict, Optional


class MarketBasketError(Exception):
    """Base exception for all market basket analysis errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "MARKET_BASKET_ERROR",
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.recoverable = recoverable


class DataError(MarketBasketError):
    """Data loading, validation, or transformation errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "DATA_ERROR",
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
    ):
        super().__init__(message, code=code, details=details, recoverable=recoverable)


class DataLoadingError(DataError):
    """Failed to load data from source."""

    def __init__(
        self,
        message: str,
        *,
        source: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            code="DATA_LOADING_ERROR",
            details={**(details or {}), "source": source},
            recoverable=True,
        )


class DataValidationError(DataError):
    """Data validation failed."""

    def __init__(
        self,
        message: str,
        *,
        missing_columns: Optional[list] = None,
        invalid_rows: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            code="DATA_VALIDATION_ERROR",
            details={
                **(details or {}),
                "missing_columns": missing_columns,
                "invalid_rows": invalid_rows,
            },
            recoverable=True,
        )


class InsufficientDataError(DataError):
    """Not enough data for reliable analysis."""

    def __init__(
        self,
        message: str,
        *,
        metric: str,
        actual: float,
        required: float,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            code="INSUFFICIENT_DATA",
            details={
                **(details or {}),
                "metric": metric,
                "actual": actual,
                "required": required,
            },
            recoverable=False,
        )


class ConfigurationError(MarketBasketError):
    """Invalid or missing configuration."""

    def __init__(
        self,
        message: str,
        *,
        param: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            code="CONFIGURATION_ERROR",
            details={**(details or {}), "param": param},
            recoverable=False,
        )


class AnalysisError(MarketBasketError):
    """Analysis computation errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "ANALYSIS_ERROR",
        analysis_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = False,
    ):
        super().__init__(
            message,
            code=code,
            details={**(details or {}), "analysis_type": analysis_type},
            recoverable=recoverable,
        )


class ComputationError(AnalysisError):
    """Mathematical or statistical computation failed."""

    def __init__(
        self,
        message: str,
        *,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            code="COMPUTATION_ERROR",
            analysis_type=operation,
            details=details,
            recoverable=False,
        )


class ConvergenceError(AnalysisError):
    """Model failed to converge."""

    def __init__(
        self,
        message: str,
        *,
        model: Optional[str] = None,
        iterations: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            code="CONVERGENCE_ERROR",
            analysis_type=model,
            details={**(details or {}), "iterations": iterations},
            recoverable=True,
        )


class PipelineError(MarketBasketError):
    """Pipeline state management errors."""

    def __init__(
        self,
        message: str,
        *,
        stage: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            code="PIPELINE_ERROR",
            details={**(details or {}), "stage": stage},
            recoverable=True,
        )


class CacheError(MarketBasketError):
    """Caching related errors."""

    def __init__(
        self,
        message: str,
        *,
        key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            code="CACHE_ERROR",
            details={**(details or {}), "key": key},
            recoverable=True,
        )


class VisualizationError(MarketBasketError):
    """Chart/visualization rendering errors."""

    def __init__(
        self,
        message: str,
        *,
        chart_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            code="VISUALIZATION_ERROR",
            details={**(details or {}), "chart_type": chart_type},
            recoverable=True,
        )


class ExternalServiceError(MarketBasketError):
    """External service (DB, API, MLflow) errors."""

    def __init__(
        self,
        message: str,
        *,
        service: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            code="EXTERNAL_SERVICE_ERROR",
            details={**(details or {}), "service": service},
            recoverable=True,
        )
