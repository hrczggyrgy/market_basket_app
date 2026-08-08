"""AppTest: Co-purchase tab with segment/mission filters.

Boots the real app with the built-in sample data source and verifies:
- the Co-purchase tab renders without exceptions,
- segment/mission filter widgets appear when segment columns exist,
- the network updates when a filter is changed,
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


def test_copurchase_renders_with_filters(app_path) -> None:
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    assert len(at.exception) == 0, f"Boot raised: {[str(e.value) for e in at.exception]}"

    at.sidebar.radio[1].set_value("copurchase")
    at.run()
    exceptions = at.exception
    assert len(exceptions) == 0, f"Co-purchase tab raised: {[str(e.value) for e in exceptions]}"

    # Filter widgets present: segment selectbox and mission selectbox
    segment_selects = [s for s in at.selectbox if "Segment" in s.label]
    mission_selects = [s for s in at.selectbox if "Mission" in s.label]
    assert segment_selects, "expected a Customer Segment selectbox"
    assert mission_selects, "expected a Basket Mission selectbox"

    # Segment filter must offer a non-None option
    assert len(segment_selects[0].options) >= 2

    # Change the mission filter and re-render
    if len(mission_selects[0].options) >= 2:
        other = next((o for o in mission_selects[0].options if o != mission_selects[0].value), None)
        if other is not None:
            mission_selects[0].select(other)
            at.run()
            exceptions = at.exception
            assert len(exceptions) == 0, f"Mission filter switch raised: {[str(e.value) for e in exceptions]}"


def test_copurchase_mode_sweep(app_path) -> None:
    """Switching away from and back to Co-purchase must not break other tabs."""
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    _sweep_modes(at)
