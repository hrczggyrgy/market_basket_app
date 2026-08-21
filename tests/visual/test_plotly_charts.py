"""Visual regression tests for shared Plotly chart components."""

from __future__ import annotations

import pandas as pd
import pytest
import plotly.graph_objects as go

from src.ui.plots import PALETTE, new_fig, show, empty_state, render_bar_with_ci, render_line_with_ci


class TestSharedPlotComponents:
    """Test shared Plotly helpers produce consistent visual output."""

    def test_new_fig_basic(self, assert_snapshot):
        """Test basic new_fig() creates consistent figure."""
        fig = new_fig()
        fig.add_trace(
            go.Scatter(
                x=[1, 2, 3, 4],
                y=[10, 15, 13, 17],
                mode="lines+markers",
                name="Test Series",
            )
        )
        fig.update_layout(title="Basic Line Chart", xaxis_title="X", yaxis_title="Y")
        assert_snapshot(fig, "new_fig_basic")

    def test_new_fig_with_palette_colors(self, assert_snapshot):
        """Test new_fig() with PALETTE colors."""
        fig = new_fig()
        for i, color in enumerate(PALETTE[:5]):
            fig.add_trace(
                go.Bar(
                    x=[f"Category {i}"],
                    y=[10 + i * 5],
                    marker_color=color,
                    name=f"Series {i}",
                )
            )
        fig.update_layout(title="Palette Colors", barmode="group")
        assert_snapshot(fig, "new_fig_palette_colors")

    def test_empty_state(self, assert_snapshot):
        """Test empty_state() renders consistent message."""
        fig = empty_state("No data available for this selection")
        assert_snapshot(fig, "empty_state")

    def test_render_bar_with_ci(self, assert_snapshot, sample_df):
        """Test render_bar_with_ci() with sample data."""
        # Create test data with CI columns
        test_df = pd.DataFrame({
            "category": ["A", "B", "C", "D"],
            "value": [100, 150, 120, 180],
            "ci_lower": [90, 140, 110, 170],
            "ci_upper": [110, 160, 130, 190],
        })
        fig = render_bar_with_ci(
            test_df,
            x_col="category",
            y_col="value",
            ci_lower_col="ci_lower",
            ci_upper_col="ci_upper",
            title="Bar Chart with CI",
        )
        assert_snapshot(fig, "render_bar_with_ci")

    def test_render_line_with_ci(self, assert_snapshot):
        """Test render_line_with_ci() with time series data."""
        test_df = pd.DataFrame({
            "date": [str(d) for d in pd.date_range("2024-01-01", periods=12, freq="ME")],
            "value": [100, 110, 105, 115, 120, 125, 130, 128, 135, 140, 145, 150],
            "ci_lower": [95, 105, 100, 110, 115, 120, 125, 123, 130, 135, 140, 145],
            "ci_upper": [105, 115, 110, 120, 125, 130, 135, 133, 140, 145, 150, 155],
        })
        fig = render_line_with_ci(
            test_df,
            x_col="date",
            y_col="value",
            ci_lower_col="ci_lower",
            ci_upper_col="ci_upper",
            title="Line Chart with CI Band",
        )
        assert_snapshot(fig, "render_line_with_ci")


class TestPaletteConsistency:
    """Test PALETTE colors are consistent."""

    def test_palette_length(self):
        """PALETTE should have expected number of colors."""
        assert len(PALETTE) >= 5, "PALETTE should have at least 5 colors"

    def test_palette_format(self):
        """All PALETTE colors should be valid hex."""
        for color in PALETTE:
            assert color.startswith("#"), f"Color {color} should start with #"
            assert len(color) == 7, f"Color {color} should be 7 chars (#RRGGBB)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])