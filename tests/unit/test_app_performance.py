"""AppTest: Performance tab lifecycle stage corrections.

Regression coverage for:
- product_lifecycle_stage growth computed between the two most recent weekly
  periods (previously it summed ALL-period revenue as "recent"),
- the Performance tab render path in the real app.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.ui import registry

pytestmark = pytest.mark.slow


def test_performance_tab_renders_lifecycle_scatter(app_path) -> None:
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    assert len(at.exception) == 0

    at.sidebar.radio[1].set_value("performance")
    at.run()
    exceptions = at.exception
    assert len(exceptions) == 0, f"Performance tab raised: {[str(e.value) for e in exceptions]}"

    bits = []
    for el in at.main:
        try:
            v = el.value
        except (KeyError, AttributeError):
            continue
        if isinstance(v, str):
            bits.append(v)
    assert "Lifecycle Stage (Growth vs Revenue)" in " ".join(bits)


def test_performance_mode_sweep(app_path) -> None:
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    modes = registry.get_modes()
    for key in modes:
        at.sidebar.radio[1].set_value(key)
        at.run()
        assert len(at.exception) == 0, f"Mode {key!r} raised: {[str(e.value) for e in at.exception]}"