"""AppTest mode sweep: every registered tab must render without exceptions.

Boots the real app (app.py) twice:
1. with the built-in sample data source,
2. with an uploaded Online-Retail-shaped CSV (no category column).

Each run sweeps every registered sidebar mode and asserts zero exceptions.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.ui import registry

pytestmark = pytest.mark.slow


def _online_retail_csv() -> bytes:
    """Online-Retail-shaped upload: no category column, fixed price per SKU."""
    rng = np.random.default_rng(7)
    skus = {f"S{i}": round(rng.uniform(0.5, 20), 2) for i in range(40)}
    rows = []
    for i in range(1200):
        sku = rng.choice(list(skus))
        rows.append(
            {
                "date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(rng.integers(0, 365))),
                "transaction_id": f"I{i}",
                "stockcode": sku,
                "product": f"Product {sku}",
                "customer_id": f"C{i % 200}",
                "price": skus[sku],
                "quantity": int(rng.integers(1, 10)),
            }
        )
    return pd.DataFrame(rows).to_csv(index=False).encode()


def _sweep_modes(at: AppTest) -> None:
    """Select each registered mode and assert the app renders without exception."""
    modes = registry.get_modes()
    for key in modes:
        at.sidebar.radio[1].set_value(key)
        at.run()
        exceptions = at.exception
        assert len(exceptions) == 0, f"Mode {key!r} raised: {[str(e.value) for e in exceptions]}"


def test_mode_sweep_sample_data(app_path) -> None:
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    assert len(at.exception) == 0, f"Boot raised: {[str(e.value) for e in at.exception]}"
    _sweep_modes(at)
    # The new tab must be registered
    assert "category" in registry.get_modes()


def test_mode_sweep_uploaded_dataset(app_path) -> None:
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    uploader = at.sidebar.file_uploader[0]
    uploader.upload(
        filename="online_retail.csv",
        content=_online_retail_csv(),
        mime_type="text/csv",
    )
    at.run()
    assert len(at.exception) == 0, f"Upload boot raised: {[str(e.value) for e in at.exception]}"
    _sweep_modes(at)


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

def test_category_tab_renders_scorecard(app_path) -> None:
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    at.sidebar.radio[1].set_value("category")
    at.run()
    exceptions = at.exception
    assert len(exceptions) == 0, f"Category tab raised: {[str(e.value) for e in exceptions]}"

    body_text = _text(at)
    assert "Category Scorecard" in body_text
    assert "Category Role Treemap" in body_text
    assert "Assortment Efficiency" in body_text
    assert "Category Growth Matrix" in body_text
    assert "Scenario Grid" in body_text
    assert "Category Promo Timeline" in body_text
    assert "Category Cannibalization" in body_text
    assert "Category Drill-down" in body_text


def test_category_drilldown_select_updates(app_path) -> None:
    """Selecting a different category rerenders the drill-down without exceptions."""
    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    at.sidebar.radio[1].set_value("category")
    at.run()
    assert len(at.exception) == 0

    assert len(at.selectbox) >= 1, "expected a category drill-down selectbox"
    first = at.selectbox[0].value
    opts = at.selectbox[0].options
    assert len(opts) >= 2

    other = next((o for o in opts if o != first), None)
    assert other is not None
    at.selectbox[0].select(other)
    at.run()
    exceptions = at.exception
    assert len(exceptions) == 0, f"Drill-down after switch raised: {[str(e.value) for e in exceptions]}"
    assert at.selectbox[0].value == other
