# Promotional Analytics Improvement Plan

## Executive Summary

**Current State**: The promotional analytics module provides sophisticated descriptive analytics (STL baseline, uplift modeling, Qini curves, cannibalization/halo analysis) but **does not deliver valid causal incrementality estimates**. The waterfall combines effects that aren't on the same causal footing.

**Target State**: Two distinct analytical layers:
1. **Descriptive Layer** — "Promotional Revenue Decomposition" (what happened relative to expected)
2. **Causal Layer** — "Causal Incrementality Engine" (what additional demand was caused by promotions)

---

## Phase 1: Fix Descriptive Baseline (Weeks 1-2)

### 1.1 Fix STL Baseline Interpolation Leakage
**Problem**: Lines 260-268 in `compute_promo_baseline()` interpolate across promo weeks using post-promo data, contaminating the baseline for promo weeks.

**Solution**: 
- Fit STL **only on non-promo weeks** (no interpolation across treatment periods)
- For promo weeks, use **forward-fill** from last observed non-promo trend/seasonal
- Or use **structural time series** (e.g., `statsmodels.tsa.statespace`) that can handle missing observations natively

```python
# Current (leaky):
units_interp = np.interp(np.arange(len(sku)), np.where(non_promo_mask)[0], units_non_promo)
# Then fit STL on interpolated series

# Proposed:
# 1. Create series with NaN for promo weeks
# 2. Fit STL/structural model only on observed (non-promo) points
# 3. Forecast/impute promo weeks from model fitted ONLY on non-promo data
```

### 1.2 Separate Quantity and Price Effects
**Problem**: `compute_promo_baseline()` computes revenue baseline directly, mixing quantity and price effects.

**Solution**: 
- Baseline **quantity** and **price** separately
- Compute `incremental_units = actual_units - baseline_units`
- Compute `incremental_revenue_from_qty = incremental_units * baseline_price`
- Compute `price_effect = actual_revenue - actual_units * baseline_price`
- Waterfall shows: `baseline_revenue + qty_effect + price_effect = actual_revenue`

### 1.3 Fix Promo Detection Documentation
- Rename `detect_promotions()` → `detect_price_based_promotions()` 
- Add explicit warning: "Heuristic price-drop detection, not promotion verification"

---

## Phase 2: Causal Incrementality Engine (Weeks 3-5)

### 2.1 SKU-Week Panel Construction
Build a proper panel dataset at (stockcode, week) level:
```python
# Columns: stockcode, week, units, revenue, avg_price, is_promo, 
#          category, n_customers, n_orders, competitors_avg_price (if available)
```

### 2.2 Two-Way Fixed Effects (TWFE) Model
Primary causal estimator:
```python
# log(units) ~ promo + sku_fe + week_fe + controls
# Using linearmodels.PanelOLS with entity/time effects
# Clustered SEs at SKU level
```

### 2.3 Event Study Design
Validate parallel trends and estimate dynamic effects:
```python
# Event study around promo start:
# log(units) ~ promo_leads/lags + sku_fe + week_fe
# Plot coefficients: -4 to +4 weeks around promo start
# Pre-trend validation: all pre-promo coefficients ≈ 0
```

### 2.4 Event-Study Based Waterfall Components
Replace descriptive proxies with causal estimates:
| Component | Current (Descriptive) | New (Causal) |
|-----------|----------------------|--------------|
| Direct SKU Effect | STL incremental | TWFE `promo` coefficient × promo weeks |
| Halo | Basket co-occurrence lift | Separate TWFE on peer SKUs in promo baskets |
| Cannibalization | Pre/post peer comparison | TWFE on peer SKUs with promo indicator |
| Stockpiling | N/A | Post-promo dip estimation (negative lags) |

---

## Phase 3: Waterfall Redesign (Week 6)

### 4.1 Two-Layer Waterfall Architecture

**Layer 1: Descriptive Decomposition** (Existing, relabeled)
```
Observed Promo Revenue
├── Baseline Revenue (STL descriptive)
└── Revenue Above Baseline (descriptive)
    ├── Quantity Effect
    ├── Price Effect
    └── Residual
```
**Label**: "Descriptive — Not Causal"

### 4.2 Causal Incrementality Waterfall (New)
```
Causal Incremental Revenue (TWFE estimate)
├── Direct SKU Effect (promo coeff × promo weeks)
├── Halo Effect (causal peer lift in promo baskets)
├── Cannibalization (causal peer substitution)
├── Stockpiling Effect (post-promo dip estimation)
└── Net Incremental Revenue
```
**Label**: "Causal — Requires Parallel Trends Assumption"

### 4.3 UI Presentation
```
┌─ Descriptive Layer ────────────────────────────────────┐
│ Baseline: $X    |  Above Baseline: $Y  | Uplift: Z%   │
│ [Explicit: NOT causal incrementality]                  │
└────────────────────────────────────────────────────────┘

┌─ Causal Incrementality (Requires Parallel Trends) ─────┐
│ Direct SKU: $A  |  Halo: $B  |  Cannibalization: -$C  │
│ Stockpiling: -$D  |  NET INCREMENTAL: $E             │
│ [Parallel trends: ✓/✗  |  Event study: link]         │
└────────────────────────────────────────────────────────┘
```

---

## Phase 3: Promo Detection & Data Quality (Week 7)

### 5.1 Improved Promo Detection
- **Option A**: Require explicit `promo_flag` column in input data (preferred)
- **Option B**: If heuristic only, require manual review/approval of detected periods
- **Option C**: Accept external promo calendar (CSV upload)

### 5.2 Data Quality Checks for Promo Analysis
- Minimum pre-period length for each promo (≥ 4 weeks)
- Minimum post-period length (≥ 2 weeks)
- No overlapping promos for same SKU
- Sufficient pre-period price variation
- Parallel trends pre-test (joint F-test on pre-promo dummies)

---

## Phase 4: Causal Assumptions & Sensitivity (Week 8)

### 6.1 Explicit Assumption Documentation
For each causal estimate, auto-generate assumption checklist:
- [ ] Parallel trends (event study pre-trends p > 0.05)
- [ ] No spillover (halo/cannibalization measured)
- [ ] Stable treatment effect (no effect modification by time)
- [ ] No anticipation effects (pre-trends flat)
- [ ] Stable unit treatment value (SUTVA)

### 6.2 Sensitivity Analysis
- **Rosenbaum bounds**: How strong must unobserved confounder be to explain away effect?
- **Placebo tests**: Randomly shift promo dates, re-estimate (should be ~0)
- **Alternative specs**: Different fixed effects, controls, functional forms

---

## Technical Implementation Details

### Files to Modify
| File | Changes |
|------|---------|
| `src/analytics/promo.py` | Core changes (baseline, waterfall, detection) |
| `src/analytics/schemas.py` | New contracts for causal panel, event study, causal waterfall |
| `src/ui/tabs/promo_page.py` | Two-layer UI, causal warnings |
| `src/analytics/causal.py` | New TWFE/event-study functions |
| `tests/unit/test_promo.py` | Unit tests for new causal estimators |

### New Files
| File | Purpose |
|------|---------|
| `src/analytics/promo/causal.py` | TWFE, event study, causal waterfall |
| `src/analytics/promo/descriptive.py` | Renamed descriptive baseline/waterfall |
| `tests/unit/test_promo_causal.py` | Causal estimator tests with synthetic data |

### New Schema Contracts
```python
# New contracts in schemas.py
PROMO_CAUSAL_PANEL       # SKU-week panel with treatment
PROMO_TWFE_RESULT        # TWFE coefficients, SEs, diagnostics
PROMO_EVENT_STUDY        # Event study coefficients + pre-trend test
PROMO_CAUSAL_WATERFALL   # Causal waterfall components
PROMO_PARALLEL_TRENDS    # Pre-trend test results
```

---

## Validation Strategy

### Synthetic Data Testing
Generate data with known ground truth:
```python
def generate_promo_data(n_skus=50, n_weeks=52, promo_effect=0.2, 
                        halo_effect=0.05, cannibalization=0.03):
    # Generate data with known causal effects
    # Test: Does estimator recover promo_effect ≈ 0.2?
    # Test: Does waterfall recover true components?
```

### Benchmark Tests
| Test | Expected |
|------|----------|
| No promo → zero effect | ✓ |
| Known promo effect | Estimate within 95% CI |
| No cannibalization → waterfall cannibalization ≈ 0 | ✓ |
| Parallel trends violated → pre-trend test fails | ✓ |

---

## Migration Strategy

### Backward Compatibility
- Keep all existing functions (deprecated with warnings)
- New functions use `_causal` suffix: `compute_causal_promo_baseline()`
- UI defaults to descriptive layer; causal layer opt-in

### Deprecation Path
| Function | Status |
|----------|--------|
| `compute_promo_baseline()` | Deprecated → `compute_descriptive_promo_baseline()` |
| `pre_post_promo_lift()` | Deprecated → `pre_post_promo_comparison()` |
| `compute_incrementality_waterfall()` | Deprecated → `compute_descriptive_waterfall()` |
| New: `compute_causal_waterfall()` | New causal engine |

---

## Resource Estimates

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1: Descriptive Fixes | 2 weeks | None |
| Phase 2: Causal Engine | 3 weeks | `linearmodels` (PanelOLS) |
| Phase 3: Waterfall Redesign | 1 week | Phase 2 |
| Phase 4: Detection/Quality | 1 week | None |
| Phase 5: Sensitivity/Assumptions | 1 week | Phase 2 |
| **Total** | **8 weeks** | |

---

## Questions for Clarification

Before finalizing the plan, I'd like your input on:

1. **Data Availability**: Do you have access to competitor prices, store-level data, or marketing calendars? (Affects causal identification)

2. **Causal Priority**: Is causal incrementality a hard requirement for v2.1, or can descriptive layer suffice for v2.1 with causal in v2.2?

3. **External Instruments**: Do you have cost data, tax changes, or supply shocks for IV estimation?

4. **Stakeholder Communication**: How will you communicate "descriptive vs causal" to business users?

5. **Validation Data**: Do you have holdout periods or A/B test results for validation?

---

## Appendix: Key Code Locations to Modify

| Function | File | Lines | Priority |
|----------|------|-------|----------|
| `compute_promo_baseline` | `promo.py` | 226-287 | P1 - Fix interpolation leakage |
| `compute_incrementality_waterfall` | `promo.py` | 364-396 | P1 - Separate qty/price, add causal layer |
| `detect_promotions` | `promo.py` | 66-129 | P2 - Add explicit warnings, improve detection |
| `pre_post_promo_comparison` | `promo.py` | 290-350 | P2 - Already has warning, ensure UI reflects |
| `promo_roi_analysis` | `promo.py` | 403-463 | P2 - Use causal incremental revenue |
| `compute_incrementality_waterfall` | `promo.py` | 364-396 | P1 - Separate descriptive/causal |
| `halo_effect_analysis` | `promo.py` | 494-518 | P2 - Replace with causal halo |
| `compute_cannibalization_analysis` | `promo.py` | 521-602 | P2 - Replace with causal cannibalization |
| `build_uplift_dataset` | `promo.py` | 716-764 | P2 - Add causal features |
| `promo_roi_analysis` | `promo.py` | 403-463 | P2 - Use causal incremental revenue |
| UI rendering | `promo_page.py` | All | P1 - Two-layer waterfall display |

---

## Next Steps

1. **Confirm priority**: Which phase(s) for v2.1 vs v2.2?
2. **Confirm data access**: What additional data is available?
3. **Review plan**: Any architectural concerns?
4. **Assign ownership**: Who implements each phase?
5. **Set milestones**: Target dates for each phase

---

*This plan is ready for review. No code changes have been made — all analysis and planning only.*