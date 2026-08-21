"""Visual regression test fixtures for Plotly chart snapshots."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import pytest
from pixelmatch import pixelmatch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)

# Tolerance for pixel differences (0.01 = 1%)
PIXEL_TOLERANCE = float(os.getenv("VISUAL_TOLERANCE", "0.01"))
# Threshold for pixelmatch (0-1, lower = more sensitive)
PIXELMATCH_THRESHOLD = float(os.getenv("VISUAL_THRESHOLD", "0.1"))


def _fig_to_png_bytes(fig: Any) -> bytes:
    """Convert Plotly figure to PNG bytes using kaleido."""
    return fig.to_image(format="png", engine="kaleido", scale=2)


def _save_snapshot(name: str, png_bytes: bytes) -> Path:
    """Save snapshot to disk."""
    path = SNAPSHOT_DIR / f"{name}.png"
    path.write_bytes(png_bytes)
    return path


def _load_snapshot(name: str) -> bytes | None:
    """Load existing snapshot from disk."""
    path = SNAPSHOT_DIR / f"{name}.png"
    if path.exists():
        return path.read_bytes()
    return None


def _compare_images(baseline: bytes, current: bytes, threshold: float = PIXELMATCH_THRESHOLD) -> tuple[bool, float, bytes | None]:
    """Compare two PNG images using pixelmatch.
    
    Returns:
        (match: bool, diff_ratio: float, diff_image: bytes | None)
    """
    baseline_img = Image.open(io.BytesIO(baseline)).convert("RGBA")
    current_img = Image.open(io.BytesIO(current)).convert("RGBA")
    
    # Ensure same size
    if baseline_img.size != current_img.size:
        current_img = current_img.resize(baseline_img.size, Image.Resampling.LANCZOS)
    
    width, height = baseline_img.size
    
    # Convert to raw pixel data
    baseline_data = list(baseline_img.tobytes())
    current_data = list(current_img.tobytes())
    diff_data = [0] * len(baseline_data)
    
    mismatch = pixelmatch(
        baseline_data,
        current_data,
        width,
        height,
        diff_data,
        threshold=threshold,
        includeAA=True,
    )
    
    total_pixels = width * height
    diff_ratio = mismatch / total_pixels
    
    # Save diff image if there's a mismatch
    diff_bytes = None
    if diff_ratio > PIXEL_TOLERANCE:
        diff_img = Image.frombytes("RGBA", (width, height), bytes(diff_data))
        buf = io.BytesIO()
        diff_img.save(buf, format="PNG")
        diff_bytes = buf.getvalue()
    
    return diff_ratio <= PIXEL_TOLERANCE, diff_ratio, diff_bytes


@pytest.fixture
def assert_snapshot():
    """Fixture for visual regression testing of Plotly figures.
    
    Usage:
        def test_my_chart(assert_snapshot, sample_df):
            fig = render_my_chart(sample_df)
            assert_snapshot(fig, "my_chart")
    
    On first run, saves golden image. On subsequent runs, compares against golden.
    Set VISUAL_TOLERANCE env var to adjust pixel tolerance (default 1%).
    Set VISUAL_THRESHOLD env var to adjust pixelmatch threshold (default 0.1).
    """
    def _assert(fig: Any, name: str, tolerance: float | None = None):
        current_png = _fig_to_png_bytes(fig)
        baseline_png = _load_snapshot(name)
        
        if baseline_png is None:
            # First run - save as golden image
            _save_snapshot(name, current_png)
            pytest.skip(f"Created golden snapshot: {name}.png")
        
        # Compare
        match, diff_ratio, diff_bytes = _compare_images(baseline_png, current_png)
        
        if not match:
            # Save diff for debugging
            if diff_bytes:
                diff_path = SNAPSHOT_DIR / f"{name}.diff.png"
                diff_path.write_bytes(diff_bytes)
            
            pytest.fail(
                f"Visual regression detected for '{name}': "
                f"{diff_ratio:.2%} pixels differ (tolerance: {PIXEL_TOLERANCE:.2%}). "
                f"Diff saved to {name}.diff.png"
            )
    
    return _assert


@pytest.fixture(scope="session")
def sample_df():
    """Re-export sample_df from root conftest."""
    from tests.conftest import sample_df as _sample_df
    return _sample_df


@pytest.fixture(scope="session")
def app_path():
    """Re-export app_path from root conftest."""
    from tests.conftest import app_path as _app_path
    return _app_path