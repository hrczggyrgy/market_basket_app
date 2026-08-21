# Market Basket App - Architecture Documentation

## Overview

This document describes the architecture of the Market Basket Intelligence application, a Streamlit-based Customer Decision Intelligence platform for retail analytics.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit App                            │
├─────────────────────────────────────────────────────────────────┤
│  app.py  ──►  Data Loading  ──►  Mode Registry  ──►  Tabs      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  AnalysisExecutor  │  ResultStore  │  AnalysisRegistry        │
│  ReadinessEngine   │  Dependencies  │  DecisionCenter          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Analytics Engines                           │
├─────────────────────────────────────────────────────────────────┤
│ Tier A (Instant)    │ Tier B (Cached)  │ Tier C (On-Demand)    │
│ ─────────────────   │ ───────────────  │ ────────────────────  │
│ Overview            │ Basket           │ CDT                    │
│ Performance         │ Pricing          │ CLV                    │
│                     │ Switching        │ Assortment             │
│                     │ Promotion        │ Network                │
│                     │ Segmentation     │ Markov                 │
│                     │ Cross-sell       │ Rules                  │
│                     │ Cohorts          │ Promo Advanced         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Platform                              │
├─────────────────────────────────────────────────────────────────┤
│  DuckDB Manager  │  FeatureStore  │  Ingestion Pipeline       │
│  Parquet/Arrow   │  Fact Tables   │  Fingerprinting            │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Data Platform (Phase 1)
- **DuckDB Manager**: Read-only connection pool for analytical queries
- **FeatureStore**: Canonical fact tables (product, customer, basket, weekly panels)
- **Ingestion Pipeline**: CSV → validate → canonicalize → fingerprint → Parquet → DuckDB
- **Dataset Fingerprinting**: SHA-256 based stable dataset IDs for cache namespace

### 2. Orchestration Layer (Phases 3-4)
- **AnalysisRegistry**: Declarative specs for all 15+ analyses with tier, deps, budgets
- **ResultStore**: Versioned cache keyed by (dataset_id, analysis_key, version, param_hash)
- **AnalysisExecutor**: Dependency resolution, cache checking, engine execution
- **ReadinessEngine**: Status computation (READY/CACHED/ADVANCED/NOT_AVAILABLE)
- **DecisionCenter**: Cross-domain signal aggregation from ResultStore only

### 3. Analytics Engines (Phases 5-8)

#### Tier A: Instant Analyses (recompute every time)
- **Overview**: KPIs, revenue trends, data quality
- **Performance**: ABC/XYZ classification, velocity, lifecycle

#### Tier B: Lazy Cached Analyses (cached between runs)
- **Basket**: Penetration, composition, entropy
- **Pricing**: Elasticity from weekly panels, KVI scoring
- **Switching**: Transition matrices, status, category switching
- **Promotion**: Fast layer (periods, baseline, ROI)
- **Segmentation**: RFM, behavioral, value-based
- **Cross-sell**: Co-purchase affinity, rules
- **Cohorts**: Retention, LTV curves, YoY

#### Tier C: On-Demand Analyses (explicit user action)
- **CDT**: Similarity, clustering, decision trees
- **CLV**: BG/NBD + Gamma-Gamma with bootstrap CI
- **Assortment**: Heuristic/MILP optimization, scenarios
- **Network**: Co-purchase & switching networks
- **Markov**: Transition matrices, steady state, absorption
- **Rules**: FP-Growth + bootstrap lift CI
- **Promo Advanced**: TWFE, event study, cross-SKU effects

### 4. Performance Optimizations (Phase 9)
- **Arrow Interchange**: DuckDB → Arrow → Pandas/Plotly zero-copy
- **Categorical dtypes**: stockcode, customer_id, transaction_id, product
- **Float32**: Price/revenue columns for memory efficiency
- **Lazy imports**: Heavy modules loaded on demand
- **Memory guardrails**: Pre-analysis memory checks, threshold warnings
- **Cold-start handler**: Ephemeral filesystem management for Streamlit Cloud

## Data Flow

```
User CSV Upload
      │
      ▼
┌─────────────────┐
│ Ingestion       │  ──► Canonical df + dataset_id + quality_report
│ Pipeline        │
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ DuckDB Manager  │  ──► Parquet files registered as tables
│ + FeatureStore  │      Materialized views for fact tables
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Orchestration   │  ──► AnalysisExecutor runs dependency graph
│ Layer           │      Results cached in ResultStore
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Analytics       │  ──► Engines produce contract-validated outputs
│ Engines         │
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ UI Tabs         │  ──► ReadinessEngine shows status
│ + Decision      │      DecisionCenter aggregates signals
│ Center          │
└─────────────────┘
```

## Performance Budgets

| Analysis | Budget (100k rows) |
|----------|-------------------|
| Overview | < 1s |
| Product | < 1s |
| Basket | < 2s |
| Pricing | < 3s |
| Switching | < 4s |
| Promotion | < 4s |
| Peak Memory | < 900MB |

## Technology Stack

- **Python**: 3.13+
- **Web Framework**: Streamlit 1.59+
- **Data**: Pandas 2.3+, DuckDB, PyArrow
- **Analytics**: SciPy, scikit-learn, statsmodels, lifetimes, mlxtend, networkx
- **Visualization**: Plotly 6.9+
- **Package Manager**: uv
- **Testing**: pytest, pytest-xdist
- **Linting**: ruff, mypy

## Deployment

### Streamlit Cloud
- Ephemeral filesystem handled by ColdStartHandler
- Dataset re-upload required on each session
- uv.lock committed for reproducible builds

### Local Development
```bash
uv sync --extra dev --extra advanced
streamlit run app.py
```

## Rollback Strategy

Each phase can be independently rolled back:
1. **Phase 0-1**: Revert to CSV/Pandas data loading
2. **Phase 2-3**: Disable ResultStore, re-enable @st.cache_data
3. **Phase 4**: Re-enable direct engine calls in DecisionCenter
4. **Phase 5-8**: Re-enable Tier C analyses as Tier B
5. **Phase 9-10**: Remove performance optimizations, revert CI

See `docs/rollback.md` for detailed procedures.

## Version Compatibility

- **Schema Version**: 1.0.0 (data contract)
- **Feature Version**: 1.0.0 (feature store)
- **Analytics Version**: 1.0.0 (engine implementations)
- Bump versions when contracts change to invalidate caches