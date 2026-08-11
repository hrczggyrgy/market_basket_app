# Market Basket App — Agent Instructions

## Quick Commands

```bash
# Run app
streamlit run app.py

# Run tests (fast, skips slow marked)
.venv/bin/python -m pytest tests/unit -x -q -k "not slow"

# Run specific test module
.venv/bin/python -m pytest tests/unit/test_pricing.py -v

# Run with coverage
.venv/bin/python -m pytest tests/unit --cov=src --cov-report=term-missing

# Lint
.venv/bin/python -m ruff check .

# Typecheck
.venv/bin/python -m mypy src

# Validate analytics contracts (slow)
.venv/bin/python -m pytest tests/unit/test_validation_baseline.py -x -q

# Regenerate analytics baseline after intentional schema changes
python -c "
from src.analytics.validation import ValidationHarness, export_baseline
from src.analytics.sample_data import generate_transactions
df = generate_transactions()
harness = ValidationHarness(df)
harness.run_all()
export_baseline(harness, 'tests/fixtures/analytics_baseline.json')
"
```

## Architecture Notes

### Mode Registry Pattern (Critical)
- Tabs register via `MODE_SPEC` in `src/ui/tabs/<tab>.py`
- Import in `app.py` → registry auto-registers
- Handler signature: `render(df: pd.DataFrame) -> None`
- The app loads data once, dispatches selected mode handler with transaction DataFrame

### Shared Plot Helpers (`src/ui/plots.py`)
```python
PALETTE = ["#006D77", "#83C5BE", "#EDF6F9", "#FFDDD2", "#E29578"]
new_fig()          # Creates go.Figure with dark template + consistent styling
show(fig)          # fig.update_layout + st.plotly_chart(fig, use_container_width=True)
empty_state(msg)   # Returns Figure with centered message for empty data
render_bar_with_ci # Bar chart with error bars
```

### Data Contracts (`src/analytics/schemas.py`)
All analytics outputs validated against `DataContract` schemas with semantic validators. Validation at function return via `check(df, CONTRACT)`. **Never bypass this** — it's the primary correctness guarantee.

### Adding a New Tab
1. Create `src/ui/tabs/new_tab.py` with `render(df: pd.DataFrame) -> None`
2. Add `MODE_SPEC = ModeSpec(key="new_tab", label="Label", icon=":material/icon:", handler=render)`
3. Import in `app.py` (registry auto-registers)

## Test Conventions

- Fixtures: `tests/unit/pricing_fixtures.py` (sample data builders)
- `conftest.py` defines `sample_df` fixture (loads `sample_data/sample_transactions.csv`)
- Slow tests marked `@pytest.mark.slow` — skip with `-k "not slow"`
- AppTest integration tests: `test_app_*.py` — run full mode sweep
- Analytics validation baseline: `tests/fixtures/analytics_baseline.json` — regenerate with `export_baseline()` after intentional contract changes

## Key Analytics Modules

| Module | Purpose | Key Output |
|--------|---------|------------|
| `pricing/elasticity.py` | Log-log OLS, hierarchical EB, cross-price | `ELASTICITY`, `ELASTICITY_STATUS` |
| `pricing/kvi.py` | Key Value Item scoring | `KVI_SCORES` |
| `clv.py` | BG/NBD + Gamma-Gamma CLV | `CLV_CUSTOMER`, `CLV_PREDICTIONS` |
| `promo.py` | Promo detection, lift, waterfall, ROI | `PROMO_PERIODS`, `PROMO_WATERFALL`, `PROMO_ROI` |
| `switching.py` + `transference.py` | Product switching, demand recovery | `SWITCHING_MATRIX`, `DEMAND_TRANSFERENCE` |
| `performance.py` | ABC/XYZ, velocity, lifecycle | `ABC_CLASSES`, `XYZ_CLASSES`, `SKU_RATIONALIZATION` |
| `cdt/` | Customer Decision Tree | `CDT_TREE_NODES`, `CDT_ASSIGNMENTS` |
| `segmentation/` | RFM, behavioral, value-based | `RFM_SEGMENTS`, `BEHAVIORAL_SEGMENTS` |

## Common Pitfalls

1. **Don't forget `check()`** — every analytics function must validate its output against a schema
2. **Schema changes need baseline regen** — run `export_baseline()` and commit new `analytics_baseline.json`
3. **Session state for heavy objects** — CLV model fits cached via `st.session_state` in tabs
4. **Large CSV warning** — `transactions2.csv` and `large_sample.csv` are 100MB+; don't load in tests
5. **NumPy 2.4.6 + linearmodels** — warning at import but works; don't downgrade numpy
   The `linearmodels` package emits a numpy 1.x ABI warning on numpy 2.4.6; it's non-fatal and doesn't affect functionality.
6. **Streamlit width="stretch"** — `show(fig)` handles this, don't use `st.plotly_chart(fig, width="stretch")` directly
7. **Promo module rename** — `src/analytics/promo.py` → `promo_core.py`; imports via `from src.analytics.promo import ...` work via `__init__.py` re-export

## Environment

- Python 3.10+ (uses `.venv` by default)
- Dependencies in `pyproject.toml` (single source of truth)
- Install with `pip install -e .` or `pip install -r requirements.txt`
- Dev deps: `pytest`, `pytest-timeout`, `ruff`, `mypy`

## Validation Commands

```bash
# CDT validation (ARI, NMI)
python -c "from src.analytics.cdt.validation import run_cdt_validation; print(run_cdt_validation())"

# Segmentation validation
python -c "from src.analytics.segmentation.core import run_segmentation_validation; print(run_segmentation_validation())"

# Elasticity validation
python -c "from src.analytics.pricing.elasticity import run_validation; print(run_validation())"

# Promo detection validation
python -c "from src.analytics.promo import run_promo_validation; print(run_promo_validation())"

# Assortment validation
python -c "from src.analytics.assortment import run_assortment_validation; print(run_assortment_validation())"
```