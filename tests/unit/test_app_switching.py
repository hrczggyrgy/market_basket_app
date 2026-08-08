"""AppTest: Switching tab with category-level Sankey.

Boots the real app with the built-in sample data source and verifies:
- the Switching tab renders without exceptions,
- the category-level Sankey section renders,
- no other registered tab breaks.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.ui import registry

pytestmark = pytest.mark.slow


def _sweep_modes(at: AppTest) -> None:
    """Select each registered mode and assert the app renders without exception."""
    modes = registry.get_modes()
    for key in modes:
        at.sidebar.radio[1].set_value(key)
        at.run()
        exceptions = at.exception
        assert len(exceptions) == 0, f"Mode {key!r} raised: {[str(e.value) for e in exceptions]}"


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


def test_switching_tab_renders_category_sankey(app_path) -> None:
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    assert len(at.exception) == 0, f"Boot raised: {[str(e.value) for e in at.exception]}"

    at.sidebar.radio[1].set_value("switching")
    at.run()
    exceptions = at.exception
    assert len(exceptions) == 0, f"Switching tab raised: {[str(e.value) for e in exceptions]}"

    body_text = _text(at)
    assert "Category Switching Flow" in body_text


def test_switching_mode_sweep(app_path) -> None:
    """Switching away from and back to Switching must not break other tabs."""
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    _sweep_modes(at)


def test_switching_time_slice_section_renders(app_path) -> None:
    """Time-sliced switching section must render (or degrade gracefully) with event metadata."""
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    assert len(at.exception) == 0

    at.sidebar.radio[1].set_value("switching")
    at.run()
    exceptions = at.exception
    assert len(exceptions) == 0, f"Switching tab raised: {[str(e.value) for e in exceptions]}"

    body_text = _text(at)
    assert "Time-Sliced Category Switching" in body_text
