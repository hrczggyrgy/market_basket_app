"""AppTest: Segmentation tab renders RFM / Behavioral (+ radar) / Value-Based."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.slow


def _text(at: AppTest) -> str:
    bits: list[str] = []
    for el in at.main:
        try:
            v = el.value
        except (KeyError, AttributeError):
            continue
        if isinstance(v, str):
            bits.append(v)
    return " ".join(bits)


def test_segmentation_tab_renders_radar(app_path) -> None:
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    at.sidebar.radio[1].set_value("segmentation")
    at.run()
    exceptions = at.exception
    assert len(exceptions) == 0, f"Segmentation tab raised: {[str(e.value) for e in exceptions]}"

    body_text = _text(at)
    assert "Customer Segmentation" in body_text
    assert "Segment Radar" in body_text
    assert "Segment Migration" in body_text
    assert "Segment Distribution" in body_text
    assert "n_clusters" not in body_text