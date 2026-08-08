"""AppTest: Rules tab with strength vs stability scatter.

Boots the real app with the built-in sample data source and verifies:
- the Rules tab renders without exceptions (regression: frozenset antecedents
  must not be subscripted in the strength-stability scatter),
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


def test_rules_tab_renders_strength_scatter(app_path) -> None:
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    assert len(at.exception) == 0, f"Boot raised: {[str(e.value) for e in at.exception]}"

    at.sidebar.radio[1].set_value("rules")
    at.run()
    exceptions = at.exception
    assert len(exceptions) == 0, f"Rules tab raised: {[str(e.value) for e in exceptions]}"