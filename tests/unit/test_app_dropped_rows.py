"""Regression test: sidebar 'Dropped N rows' message with drop-inducing data.

load_transactions() returns the dropped count as an int, but app.py used to call
len(dropped) which raises TypeError when the data actually triggers a drop.
This test boots the real app (app.py) with an uploaded CSV containing invalid
rows and asserts the sidebar renders without exception.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from src.analytics.data import load_transactions

pytestmark = pytest.mark.slow


def _csv_with_dropped_rows() -> bytes:
    """CSV guaranteed to drop >= 1 row during cleaning (null price)."""
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "transaction_id": ["T1", "T2", "T3", "T4"],
            "stockcode": ["SKU_A", "SKU_B", "SKU_C", "SKU_D"],
            "product": ["A", "B", "C", "D"],
            "customer_id": ["C1", "C2", "C3", "C4"],
            "price": [1.0, 2.0, None, 4.0],
            "quantity": [1, 1, 1, 1],
        }
    )
    return raw.to_csv(index=False).encode()


def test_load_transactions_returns_int_dropped_count() -> None:
    df, warning, dropped, quality = load_transactions(io.BytesIO(_csv_with_dropped_rows()))
    assert isinstance(dropped, int)
    assert dropped == 1
    assert "missing/invalid" in warning


def test_app_sidebar_renders_with_dropped_rows(app_path, tmp_path) -> None:
    from streamlit.testing.v1 import AppTest

    csv_path = tmp_path / "with_nulls.csv"
    csv_path.write_bytes(_csv_with_dropped_rows())

    at = AppTest.from_file(str(app_path), default_timeout=120)
    at.run()
    uploader = at.sidebar.file_uploader[0]
    uploader.upload(
        filename="with_nulls.csv",
        content=_csv_with_dropped_rows(),
        mime_type="text/csv",
    )
    at.run()

    exceptions = at.exception
    assert len(exceptions) == 0, f"App raised: {exceptions}"

    sidebar_text = " ".join(str(el.value) for el in at.sidebar if hasattr(el, "value"))
    assert "Dropped 1 rows during cleaning" in sidebar_text
