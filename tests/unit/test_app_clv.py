"""AppTest: CLV tab renders violin and histogram."""

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


def test_clv_tab_renders_violin(app_path) -> None:
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    at.sidebar.radio[1].set_value("clv")
    at.run()
    exceptions = at.exception
    assert len(exceptions) == 0, f"CLV tab raised: {[str(e.value) for e in exceptions]}"

    body_text = _text(at)
    assert "CLV Distribution" in body_text
    assert "Violin" in body_text
    assert "Histogram" in body_text
    assert "Predicted CLV" in body_text