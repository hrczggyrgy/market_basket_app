# Rollback Strategy Documentation

This document describes how to roll back each phase of the market-basket-performance-optimization plan if a regression is detected.

## Phase 0: Profiling Baseline & uv Migration

### What was changed
- Added `@measure_analysis` decorator to all cache functions
- Created `src/performance/profiler.py` and `src/performance/memory.py`
- Migrated from pip to uv with `uv.lock`
- Pruned optional dependencies to `advanced` extra
- Created benchmark script `scripts/benchmark.py`

### Rollback Procedure
```bash
# 1. Revert uv changes
git checkout HEAD -- pyproject.toml uv.lock
# 2. Remove profiler decorator from cache functions
# Edit src/analytics/cache.py and remove @measure_analysis from all functions
# 3. Delete performance modules
rm src/performance/profiler.py src/performance/memory.py
rmdir src/performance
# 4. Delete benchmark script
rm scripts/benchmark.py
# 5. Re-run tests to verify
uv run pytest tests/unit -x -q
```

### Verification
- Tests pass without performance modules
- No import errors for `@measure_analysis`
- Cache functions work without profiler

---

## Phase 1: Data Platform (DuckDB + Parquet + Arrow)

### What was changed
- Created `src/data/duckdb_manager.py`, `src/data/ingestion.py`, `src/data/fingerprint.py`, `src/data/normalization.py`
- Modified `app.py` to use new ingestion pipeline
- Created migration scripts for sample/user data

### Rollback Procedure
```bash
# 1. Revert app.py data loading
git checkout HEAD~1 -- app.py
# 2. Delete data platform modules
rm src/data/duckdb_manager.py src/data/ingestion.py src/data/fingerprint.py src/data/normalization.py
rm src/data/quality.py src/data/schemas.py
# 3. Delete migration scripts
rm scripts/migrate_sample_to_parquet.py scripts/migrate_user_csv_to_parquet.py
# 4. Verify old load_transactions works
uv run python -c "from src.analytics.data import load_transactions; print('OK')"
```

### Verification
- `load_transactions` works as before
- No DuckDB dependencies in data loading path
- Sample data loads without Parquet

---

## Phase 2: FeatureStore (Canonical Fact Tables)

### What was changed
- Created `src/features/registry.py`, `src/features/product.py`, `src/features/basket.py`, etc.
- Created 6 canonical fact tables in DuckDB

### Rollback Procedure
```bash
# 1. Delete FeatureStore modules
rm src/features/registry.py src/features/product.py src/features/basket.py
rm src/features/customer.py src/features/switching.py src/features/co_purchase.py
rm src/features/weekly.py
# 2. Revert DuckDBManager to not create fact views
git checkout HEAD~1 -- src/data/duckdb_manager.py
```

### Verification
- Analytics engines work without FeatureStore
- No references to fact tables in analytics code

---

## Phase 3: Result Store & Analysis Registry (Clean-room transition)

### What was changed
- Removed all `@st.cache_data` decorators from `src/analytics/cache.py`
- Created orchestration modules: `result_store.py`, `analysis_registry.py`, `analysis_executor.py`, `dependencies.py`, `readiness.py`
- Added compatibility shims in `cache.py` redirecting to ResultStore

### Rollback Procedure
```bash
# 1. Restore @st.cache_data decorators
git checkout HEAD~1 -- src/analytics/cache.py
# 2. Delete orchestration modules
rm -rf src/orchestration
# 3. Verify cache.py works with Streamlit caching
uv run python -c "import streamlit as st; from src.analytics.cache import cached_basket_metrics; print('OK')"
```

### Verification
- All cache functions work with `@st.cache_data`
- No import errors from orchestration modules

---

## Phase 4: Decision Center Decoupling

### What was changed
- Refactored `decision_center.py` to ONLY read from ResultStore
- Removed direct engine calls and `include_clv`/`include_assortment` parameters
- Added "Advanced Intelligence" section with explicit "Run" buttons

### Rollback Procedure
```bash
# 1. Restore old decision_center.py
git checkout HEAD~1 -- src/analytics/decision_center.py
# 2. Revert UI tab for decision center
git checkout HEAD~1 -- src/ui/tabs/decision_center.py
```

### Verification
- Decision Center works with direct engine calls
- `include_clv` and `include_assortment` parameters functional

---

## Phase 5: Switching/Transference Consolidation

### What was changed
- Created `SwitchingEngine` class in `src/analytics/switching_engine.py`
- Refactored `transference.py` to consume `switching_edges` from SwitchingEngine
- Removed duplicate `compute_switching_matrix` call in transference

### Rollback Procedure
```bash
# 1. Restore transference.py
git checkout HEAD~1 -- src/analytics/transference.py
# 2. Delete SwitchingEngine
rm src/analytics/switching_engine.py
# 3. Verify transference computes switching independently
uv run python -c "from src.analytics.transference import compute_demand_transference_matrix; print('OK')"
```

### Verification
- Transference works independently
- Switching computed separately when needed

---

## Phase 6: Co-purchase/FP-Growth Optimization

### What was changed
- Added `top_n_products`, `min_cooccurrence`, `max_pairs` bounds to `get_top_affinity_pairs`
- Made FP-Growth lazy (Tier C) with `cached_rules()`
- Moved bootstrap CI to `cached_rules_bootstrap()` (Tier C)

### Rollback Procedure
```bash
# 1. Restore copurchase.py bounds
git checkout HEAD~1 -- src/analytics/copurchase.py
# 2. Restore cache.py copurchase/rules functions
git checkout HEAD~1 -- src/analytics/cache.py
```

### Verification
- Co-purchase works without candidate bounds
- FP-Growth runs in cached_copurchase (Tier B)
- Bootstrap CI runs automatically

---

## Phase 7: Pricing/Promotion Optimization

### What was changed
- Refactored `run_pricing_analysis` to accept `weekly_panel` from FeatureStore
- Split promo into `cached_promotion` (fast) and `cached_promotion_advanced` (Tier C)

### Rollback Procedure
```bash
# 1. Restore pricing pipeline
git checkout HEAD~1 -- src/analytics/pricing/pipeline.py
# 2. Restore cache.py promotion functions
git checkout HEAD~1 -- src/analytics/cache.py
```

### Verification
- Pricing works with raw transactions only
- Promo runs as single cached function

---

## Phase 8: Advanced Engine Isolation

### What was changed
- Created `CLVEngine`, `CDTEngine`, `AssortmentEngine`, `NetworkEngine`, `MarkovEngine`
- Added graceful degradation for missing optional dependencies
- Updated cache.py to use engine factory functions

### Rollback Procedure
```bash
# 1. Restore cache.py for CLV, CDT, Assortment
git checkout HEAD~1 -- src/analytics/cache.py
# 2. Delete engine files
rm src/analytics/clv.py src/analytics/cdt/tree.py src/analytics/assortment.py
rm src/analytics/network.py src/analytics/markov.py
# 3. Restore original module implementations
git checkout HEAD~1 -- src/analytics/clv.py src/analytics/cdt/ src/analytics/assortment.py
```

### Verification
- CLV, CDT, Assortment work as before
- No engine factory functions needed

---

## Phase 9: Runtime Optimization & Hardening

### What was changed
- Added categorical dtypes, float32, Arrow interchange in data loading
- Lazy imports for heavy modules
- Memory guardrails and cold-start handler

### Rollback Procedure
```bash
# 1. Restore data.py loading
git checkout HEAD~1 -- src/analytics/data.py
# 2. Remove performance optimizations
git checkout HEAD~1 -- src/performance/memory.py
# 3. Verify no categorical/float32 in data loading
uv run python -c "from src.analytics.data import load_transactions; df,_,_,_ = load_transactions('sample_data/sample_transactions.csv'); print(df.dtypes)"
```

### Verification
- Data loading uses default dtypes
- No memory guardrails in execution path
- Cold-start handler not invoked

---

## Phase 10: Production Hardening & CI

### What was changed
- Added GitHub Actions CI workflow
- Performance regression tests
- Architecture and rollback documentation
- Error boundaries and dataset size limits

### Rollback Procedure
```bash
# 1. Remove CI workflow
rm .github/workflows/ci.yml
# 2. Remove performance check script
rm scripts/check_performance.py
# 3. Remove documentation
rm docs/architecture.md docs/rollback.md
```

### Verification
- No CI pipeline on push/PR
- No performance checks in CI
- Standard test/lint workflow only

---

## Full Rollback (All Phases)

If a catastrophic regression requires complete rollback:

```bash
# 1. Find the commit before Phase 0 started
git log --oneline | grep "Phase 0" | head -1
# 2. Hard reset to that commit
git reset --hard <commit-hash>
# 3. Force push if needed (with team coordination)
git push --force-with-lease origin main
```

## Rollback Decision Matrix

| Symptom | Likely Phase | Immediate Action |
|---------|--------------|------------------|
| ImportError on orchestration | Phase 3 | Rollback Phase 3 |
| Decision Center shows no data | Phase 4 | Rollback Phase 4 |
| Switching computed twice | Phase 5 | Rollback Phase 5 |
| FP-Growth too slow | Phase 6 | Rollback Phase 6 |
| Pricing fails on weekly panel | Phase 7 | Rollback Phase 7 |
| CLV/CDT missing dependencies | Phase 8 | Rollback Phase 8 |
| Memory warnings in logs | Phase 9 | Rollback Phase 9 |
| CI fails on performance | Phase 10 | Rollback Phase 10 |

## Testing After Rollback

After any rollback, run:
```bash
# 1. Full test suite
uv run pytest tests/unit -x -q

# 2. Import verification
uv run python -c "
from src.analytics import *
from src.ui import *
print('All imports OK')
"

# 3. End-to-end smoke test
uv run python -c "
import streamlit as st
from src.analytics.data import load_transactions
df, warn, dropped, qr = load_transactions('sample_data/sample_transactions.csv')
print(f'Loaded {len(df)} rows, {dropped} dropped')
"
```

## Contact

For rollback assistance, contact the platform team with:
1. Phase that caused the issue
2. Error messages / performance metrics
3. Whether partial or full rollback is needed