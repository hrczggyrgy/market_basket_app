"""Tests for canonical metric definitions."""

import pytest

from src.analytics.metric_definitions import (
    METRIC_DEFINITIONS,
    get_metric_definition,
    get_metric_help,
    get_metric_caveat,
    get_metric_label,
    list_metrics_by_group,
    all_metric_keys,
)


class TestMetricDefinitions:
    """Test canonical metric definitions registry."""

    def test_all_metrics_have_required_fields(self):
        """Every metric must have label, definition, denominator, caveat, help."""
        required = {"label", "definition", "denominator", "caveat", "help"}
        for key, defn in METRIC_DEFINITIONS.items():
            missing = required - set(defn.keys())
            assert not missing, f"Metric '{key}' missing fields: {missing}"

    def test_get_metric_definition_known(self):
        """Known metric returns full definition."""
        defn = get_metric_definition("revenue")
        assert defn["label"] == "Revenue"
        assert "price × quantity" in defn["definition"]
        assert defn["denominator"] == "N/A (absolute sum)"

    def test_get_metric_definition_unknown(self):
        """Unknown metric returns safe defaults."""
        defn = get_metric_definition("nonexistent_metric")
        assert defn["label"] == "Nonexistent Metric"
        assert "not in canonical registry" in defn["caveat"]

    def test_get_metric_help(self):
        """Help text is non-empty for known metrics."""
        help_text = get_metric_help("basket_penetration")
        assert "shopping trips" in help_text.lower()

    def test_get_metric_caveat(self):
        """Caveat text is non-empty for known metrics."""
        caveat = get_metric_caveat("elasticity")
        assert "observational" in caveat.lower()

    def test_get_metric_label(self):
        """Label is formatted for display."""
        label = get_metric_label("shopper_penetration")
        assert label == "Shopper Penetration"

    def test_list_metrics_by_group(self):
        """Groups contain expected metrics."""
        core = list_metrics_by_group("core")
        assert "revenue" in core
        assert "basket" in core
        assert "basket_penetration" in core

    def test_all_metric_keys(self):
        """Registry keys match METRIC_DEFINITIONS."""
        keys = all_metric_keys()
        assert set(keys) == set(METRIC_DEFINITIONS.keys())
        assert len(keys) > 10  # reasonable number of metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])