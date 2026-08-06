"""Schema-hardening regression baseline for the analytics validation harness.

Runs the full ValidationHarness over the sample dataset and asserts the
outputs still match the committed baseline (columns, shapes, success flags).
Use `python -m src.analytics.validation`-style export to refresh the baseline
after intentional contract changes.
"""

import pytest

from src.analytics.data import load_transactions
from src.analytics.validation import ValidationHarness, assert_validation

BASELINE_PATH = "tests/fixtures/analytics_baseline.json"


@pytest.mark.slow
def test_analytics_baseline_unchanged() -> None:
    df, warning, dropped, quality = load_transactions("sample_data/sample_transactions.csv")
    harness = ValidationHarness(df)
    harness.run_all()

    summary = harness.summary()
    failed = summary[summary["success"] == False]  # noqa: E712
    assert len(failed) == 0, f"{len(failed)} analytics functions failed:\n{failed[['module', 'function', 'error']]}"

    discrepancies = harness.validate_cross_contracts()
    assert len(discrepancies) == 0, f"cross-contract violations:\n{discrepancies}"

    diffs = assert_validation(harness, BASELINE_PATH)
    assert not diffs, f"baseline drift detected:\n{diffs}"


@pytest.mark.slow
def test_analytics_baseline_detects_drift(tmp_path) -> None:
    import json

    df, warning, dropped, quality = load_transactions("sample_data/sample_transactions.csv")
    harness = ValidationHarness(df)
    harness.run_all()

    with open(BASELINE_PATH) as fh:
        baseline = json.load(fh)
    first_key = next(iter(baseline))
    baseline[first_key]["columns"] = ["DEFINITELY_WRONG_COLUMN"]
    drifted = tmp_path / "drifted_baseline.json"
    drifted.write_text(json.dumps(baseline))

    diffs = assert_validation(harness, str(drifted))
    assert any(first_key in d for d in diffs), f"drift not detected: {diffs}"
