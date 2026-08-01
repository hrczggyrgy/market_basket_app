# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Fixed
- `#29` Price Curve Diagnostics scatter crash — replaced stale `avg_price` hover reference with `median_price`, which is the actual column emitted by `diagnose_price_curves_1d()` and `diagnose_price_curves_multivariate()`.
- `#28` *(previous)* Various open issues resolved on `fix/all-open-issues` branch.

### Changed
- `.gitignore` hardened: virtual-env directories (`.venv*/`), large data files (`*.csv`, `*.parquet`), stray pip artefacts (`=*`), secrets, and model/cache files now properly excluded.
- Added `CHANGELOG.md` (this file), `SECURITY.md`, and `.github/PULL_REQUEST_TEMPLATE.md`.

---

## [1.3.0] — 2026-07

### Added
- Multivariate Price Curve Diagnostics (Price + Elasticity + Penetration + Margin).
- Bayesian Hierarchical elasticity estimation (NUTS / ADVI) with trace diagnostics.
- KVI Composite Score using the NielsenIQ 4-signal framework.
- Demand Transference: delist simulation with substitutable-demand waterfall charts.
- Assortment Optimizer: heuristic + MILP solver (OR-Tools) with scenario comparison.
- Promo Uplift Modeling: T-learner / S-learner causal inference with Qini curves.
- `run_validation()` benchmark for elasticity methods (RMSE / bias / HDI coverage).

### Changed
- `_estimate_all_elasticities_vectorized` design matrix corrected to `(N, 2·k)` shape (fix #25).
- Basket-segment ASP aggregation pre-computes `revenue` before groupby (fix #21).
- `_run_bayesian_elasticity_cached` renames `elasticity_mean → elasticity` (fix #24).

---

## [1.0.0] — 2025-Q4

### Added
- Initial public release.
- Association Rules (FP-Growth), Co-purchase, Add-on/Impulse, Switching.
- Customer Decision Tree (CDT): hierarchical clustering + attribute labelling.
- Customer Segmentation (RFM quantile/k-means, behavioural clustering, CLV).
- Product Performance (lifecycle curves, ABC/XYZ).
- Cohort Analysis (retention heatmaps, revenue per customer).
- Promotional Analytics (promo detection, lift decomposition).
- CDT Builder advanced mode: ensemble similarity, community detection (Louvain/Leiden).
- Streamlit UI with full CSV upload, column auto-detection, and export (CSV/JSON/HTML).
