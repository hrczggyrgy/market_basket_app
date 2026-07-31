"""Structured logging configuration."""

import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import structlog
from structlog.processors import JSONRenderer


def configure_logging(
    level: str = "INFO",
    json_output: bool = True,
    include_timestamp: bool = True,
) -> None:
    """Configure structured logging with structlog."""

    # Base processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]
        ),
    ]

    if include_timestamp:
        processors.append(structlog.processors.TimeStamper(fmt="iso", utc=True))

    if json_output:
        processors.append(JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Set log level for noisy libraries
    logging.getLogger("streamlit").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class AnalysisLogger:
    """Context-aware logger for analysis operations."""

    def __init__(
        self,
        analysis_name: str,
        analysis_id: str,
        logger: Optional[structlog.stdlib.BoundLogger] = None,
    ):
        self.analysis_name = analysis_name
        self.analysis_id = analysis_id
        self.logger = logger or get_logger("analysis")
        self._start_time: Optional[datetime] = None

    def __enter__(self) -> "AnalysisLogger":
        self._start_time = datetime.now()
        self.logger.info(
            "analysis_started",
            analysis_name=self.analysis_name,
            analysis_id=self.analysis_id,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
        if exc_type is None:
            self.logger.info(
                "analysis_completed",
                analysis_name=self.analysis_name,
                analysis_id=self.analysis_id,
                duration_seconds=duration,
            )
        else:
            self.logger.error(
                "analysis_failed",
                analysis_name=self.analysis_name,
                analysis_id=self.analysis_id,
                duration_seconds=duration,
                error_type=exc_type.__name__ if exc_type else "Unknown",
                error_message=str(exc_val) if exc_val else "Unknown",
            )

    def info(self, event: str, **kwargs) -> None:
        """Log info event with analysis context."""
        self.logger.info(
            event,
            analysis_name=self.analysis_name,
            analysis_id=self.analysis_id,
            **kwargs,
        )

    def warning(self, event: str, **kwargs) -> None:
        """Log warning event with analysis context."""
        self.logger.warning(
            event,
            analysis_name=self.analysis_name,
            analysis_id=self.analysis_id,
            **kwargs,
        )

    def error(self, event: str, **kwargs) -> None:
        """Log error event with analysis context."""
        self.logger.error(
            event,
            analysis_name=self.analysis_name,
            analysis_id=self.analysis_id,
            **kwargs,
        )

    def debug(self, event: str, **kwargs) -> None:
        """Log debug event with analysis context."""
        self.logger.debug(
            event,
            analysis_name=self.analysis_name,
            analysis_id=self.analysis_id,
            **kwargs,
        )

    def step(self, step_name: str, **kwargs) -> None:
        """Log a pipeline step."""
        self.info(f"step_{step_name}", step=step_name, **kwargs)

    def metric(self, name: str, value: float, **kwargs) -> None:
        """Log a metric."""
        self.info(f"metric_{name}", metric_name=name, metric_value=value, **kwargs)


def log_dataframe_info(
    logger: structlog.stdlib.BoundLogger,
    df_name: str,
    df,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Log DataFrame metadata."""
    if hasattr(df, "shape"):
        rows, cols = df.shape
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        logger.info(
            "dataframe_info",
            name=df_name,
            rows=rows,
            columns=cols,
            memory_mb=round(memory_mb, 2),
            dtypes={str(k): str(v) for k, v in df.dtypes.items()},
            **(extra or {}),
        )


def log_pipeline_stage(
    logger: structlog.stdlib.BoundLogger,
    stage: str,
    status: str,
    **kwargs,
) -> None:
    """Log pipeline stage transition."""
    logger.info(
        f"pipeline_stage_{status}",
        stage=stage,
        status=status,
        **kwargs,
    )
