"""Tests for export metadata inclusion."""

import json
import pytest
import pandas as pd
from datetime import datetime

from src.ui.export import (
    _build_export_metadata,
    _inject_metadata_csv,
    _inject_metadata_json,
    render_analytics_export,
)


class TestExportMetadata:
    """Test export metadata generation and inclusion."""

    def test_build_export_metadata_complete(self):
        """Metadata includes all required fields."""
        metadata = _build_export_metadata(
            analysis_name="test_analysis",
            date_range=(datetime(2024, 1, 1), datetime(2024, 12, 31)),
            filters={"category": "Food", "min_revenue": 1000},
            method_params={"method": "loglog_ols", "min_periods": 10},
            sample_sizes={"n_transactions": 1000, "n_customers": 100},
            readiness_status="directional",
            limitation="Observational only",
        )

        assert metadata["analysis"] == "test_analysis"
        assert metadata["date_range"]["start"] == "2024-01-01T00:00:00"
        assert metadata["date_range"]["end"] == "2024-12-31T00:00:00"
        assert metadata["filters"]["category"] == "Food"
        assert metadata["method_parameters"]["method"] == "loglog_ols"
        assert metadata["sample_sizes"]["n_transactions"] == 1000
        assert metadata["readiness_confidence"] == "directional"
        assert metadata["limitation_caveat"] == "Observational only"
        assert "export_timestamp" in metadata

    def test_build_export_metadata_minimal(self):
        """Minimal metadata works."""
        metadata = _build_export_metadata(analysis_name="minimal")
        assert metadata["analysis"] == "minimal"
        assert metadata["date_range"] is None
        assert metadata["filters"] == {}
        assert metadata["method_parameters"] == {}
        assert metadata["readiness_confidence"] == "unknown"

    def test_inject_metadata_csv(self):
        """CSV export includes metadata as comments."""
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        metadata = {"analysis": "test", "export_timestamp": "2024-01-01T00:00:00Z"}

        csv_output = _inject_metadata_csv(df, metadata)
        lines = csv_output.split("\n")

        # First lines should be metadata comments
        assert lines[0] == "# Metadata"
        assert "# analysis: test" in lines
        assert "# export_timestamp: 2024-01-01T00:00:00Z" in lines
        assert "# End Metadata" in lines
        # Data should follow
        assert "col1,col2" in csv_output
        assert "1,a" in csv_output

    def test_inject_metadata_json(self):
        """JSON export includes metadata envelope."""
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        metadata = {"analysis": "test", "export_timestamp": "2024-01-01T00:00:00Z"}

        json_output = _inject_metadata_json(df, metadata)
        parsed = json.loads(json_output)

        assert "metadata" in parsed
        assert parsed["metadata"]["analysis"] == "test"
        assert "data" in parsed
        assert len(parsed["data"]) == 2
        assert parsed["data"][0]["col1"] == 1

    def test_inject_metadata_json_handles_numpy(self):
        """JSON export handles numpy types."""
        df = pd.DataFrame({"col1": [1.5, 2.5]})
        metadata = {"analysis": "test"}

        json_output = _inject_metadata_json(df, metadata)
        parsed = json.loads(json_output)
        # Should not raise TypeError for numpy float64
        assert parsed["data"][0]["col1"] == 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])