"""AppTest: Cohorts tab with role-based retention.

Boots the real app with the built-in sample data source and verifies:
- the Cohorts tab renders without exceptions,
- the "Retention by Category Role" section renders,
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


def test_cohorts_tab_renders_role_retention(app_path) -> None:
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    assert len(at.exception) == 0, f"Boot raised: {[str(e.value) for e in at.exception]}"

    at.sidebar.radio[1].set_value("cohorts")
    at.run()
    exceptions = at.exception
    assert len(exceptions) == 0, f"Cohorts tab raised: {[str(e.value) for e in exceptions]}"

    body_text = _text(at)
    assert "Retention by Category Role" in body_text


def test_cohorts_mode_sweep(app_path) -> None:
    """Switching away from and back to Cohorts must not break other tabs."""
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    _sweep_modes(at)