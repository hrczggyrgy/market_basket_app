# Market Basket Analysis — Customer Decision Intelligence

> **A full-featured Streamlit application for market basket analysis, customer choice modeling, and Customer Decision Tree (CDT) construction.**
>
> **Live app:** [marketbasketapp.streamlit.app](https://marketbasketapp.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip or uv

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd market_basket_app

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies (pyproject.toml is the canonical source)
pip install -e .

# Run the app
streamlit run app.py
```

### Alternative Installation (using requirements.txt)

```bash
pip install -r requirements.txt
```

> **Note:** `pyproject.toml` is the canonical dependency specification. `requirements.txt` is generated from it for environments that require a flat dependency list (e.g., Streamlit Cloud, Heroku). To regenerate `requirements.txt`:
> ```bash
> pip install pip-tools
> pip-compile pyproject.toml -o requirements.txt
> ```

### Quick Test (No UI)

```bash
# Run end-to-end pipeline test
python -c "
from src.data.generator import generate_transactions
from src.analytics import (
    build_customer_sequences, build_similarity_matrix,
    perform_hierarchical_clustering, find_optimal_clusters,
    get_cluster_assignments, build_cdt, extract_product_attributes
)
df = generate_transactions(n_transactions=2000, n_customers=300, n_products=50, seed=42)
seqs = build_customer_sequences(df)
sim = build_similarity_matrix(df, method='yules_q', min_cooccurrence=3)
linkage, labels = perform_hierarchical_clustering(sim)
k, scores = find_optimal_clusters(linkage, sim, min_clusters=2, max_clusters=15)
assignments = get_cluster_assignments(linkage, sim, n_clusters=k)
attrs = extract_product_attributes(df, attribute_cols=['category','brand','size','flavor'])
root, meta = build_cdt(sim, assignments, attrs, min_cluster_size=3)
print(f'CDT: {meta[\"n_nodes\"]} nodes, {meta[\"n_leaves\"]} leaves, depth {meta[\"max_depth\"]}')
print(f'Quality: {meta[\"quality_ratio\"]:.1%} vs baseline')
"
```

---

## Project Structure

```
market_basket_app/
├── app.py                          # Main Streamlit entry point
├── requirements.txt                # Python dependencies (generated from pyproject.toml)
├── pyproject.toml                  # Canonical dependency specification
├── README.md                       # This file
├── data/
│   └── sample_transactions.csv     # Sample data (generated on first run)
├── src/
│   ├── algorithms/
│   │   └── fpgrowth.py             # FP-Growth frequent itemset mining
│   ├── analytics/                  # Core analytics modules
│   │   ├── addon.py                # Add-on / impulse analysis
│   │   ├── assortment_opt.py       # Assortment optimization (MILP/heuristic)
│   │   ├── basket_metrics.py       # Basket penetration & value metrics
│   │   ├── cdt_attributes.py       # CDT attribute extraction & enrichment
│   │   ├── cdt_behavioral.py       # CDT behavioral matrices (switching, substitution, bundling)
│   │   ├── cdt_clustering.py       # CDT hierarchical clustering & dendrogram
│   │   ├── cdt_community.py        # CDT community detection (louvain, leiden, label propagation)
│   │   ├── cdt_similarity.py       # CDT similarity matrices (Phi, Jaccard, PMI, TF-IDF)
│   │   ├── cdt_tree_builder.py     # CDT tree construction & scoring
│   │   ├── cohort.py               # Cohort analysis
│   │   ├── copurchase.py           # Co-purchase analysis
│   │   ├── demand_transference.py  # Delist simulation & substitution demand
│   │   ├── pricing.py             # Price elasticity, KVI, curve diagnostics
│   │   ├── product_performance.py  # Product lifecycle & performance
│   │   ├── promo_uplift.py        # Causal uplift modeling (T-learner, S-learner)
│   │   ├── promotional.py          # Promotional analytics (legacy)
│   │   ├── segmentation.py         # Customer segmentation (RFM, behavioral)
│   │   └── switching.py            # Brand/product switching
│   ├── application/                # Application services layer
│   │   ├── pipeline.py             # Pipeline service
│   │   ├── cdt_service.py          # CDT orchestration service
│   │   └── services.py             # Shared application services
│   ├── config.py                   # Configuration (Pydantic Settings)
│   ├── data/
│   │   ├── loader.py               # CSV loading & validation
│   │   └── generator.py            # Synthetic data generator
│   ├── domain/                     # Domain models & exceptions
│   │   ├── dto.py                  # Data transfer objects
│   │   └── exceptions.py           # Custom exceptions
│   ├── infrastructure/             # Infrastructure concerns
│   │   └── logging.py              # Structured logging (structlog)
│   ├── models/
│   │   └── decision_tree.py        # Supervised choice prediction
│   ├── rules/
│   │   └── generator.py            # Association rule generation
│   ├── ui/                         # Streamlit UI tabs
│   │   ├── sidebar.py              # Sidebar configuration
│   │   ├── cdt_unified_tab.py      # Unified CDT Builder, Demand Transference, Assortment Optimizer
│   │   ├── cdt_tab.py              # Legacy Decision Tree & Patterns tab (CDT)
│   │   ├── pricing_tab.py          # Elasticity, KVI, Price Curves, Promo Uplift
│   │   ├── rules_tab.py            # Association Rules tab
│   │   ├── copurchase_tab.py       # Co-purchase tab
│   │   ├── addon_tab.py            # Add-on tab
│   │   ├── switching_tab.py        # Switching tab
│   │   ├── tree_tab.py             # Choice Prediction Model tab
│   │   ├── segmentation_tab.py     # Customer Segmentation tab
│   │   ├── product_performance_tab.py
│   │   ├── cohort_tab.py           # Cohort Analysis tab
│   │   ├── promotional_tab.py      # Promotional Analytics tab
│   │   ├── clv_tab.py              # CLV Analytics tab
│   │   ├── category_overview_tab.py # Category Overview tab
│   │   └── export.py               # Export utilities
│   ├── utils/                      # Utility modules
│   │   ├── pipeline.py             # Global pipeline state management
│   │   ├── session.py              # Session state helpers
│   │   └── cache.py                # Caching utilities
│   └── viz/                        # Plotly visualizations
│       ├── cdt_viz.py              # CDT visualizations (dendrogram, sunburst, treemap)
│       ├── decision_tree.py        # Decision tree visualizations
│       ├── network.py              # Network graphs
│       └── heatmap.py              # Heatmaps & scatter plots
```

---

## Features

| Category | Module | Description |
|----------|--------|-------------|
| **Association Rules** | `rules_tab` | FP-Growth frequent itemsets → association rules with lift/confidence filters; network graph, heatmap, parallel coordinates |
| **Co-purchase** | `copurchase_tab` | Product affinity matrix; symmetric co-purchase heatmap; bundle candidate detection |
| **Add-on / Impulse** | `addon_tab` | Anchor → add-on recommendations; lift-ranked impulse items |
| **Switching** | `switching_tab` | Brand/product switching flows; Sankey diagrams; defector/loyalist identification |
| **Choice Prediction** | `tree_tab` | Supervised decision tree: predicts next product choice from customer history |
| **Decision Tree & Patterns (CDT)** | `cdt_tab` | **Customer Decision Tree**: unsupervised hierarchical clustering → attribute-labeled tree; substitution/bundle detection; dendrogram, sunburst, treemap |
| **Customer Segmentation** | `segmentation_tab` | RFM (quantile/k-means), behavioral clustering, CLV estimation |
| **Product Performance** | `product_performance_tab` | Lifecycle curves, price elasticity, ABC/XYZ classification |
| **Cohort Analysis** | `cohort_tab` | Retention heatmaps, revenue per customer, AOV by cohort |
| **Promotional Analytics** | `promotional_tab` | Promo detection, lift decomposition (incrementality vs. forward-buy vs. substitution) |
| **CDT Builder** (advanced) | `cdt_assortment_tab` | Ensemble similarity (Phi/Jaccard/PMI/TF-IDF), community detection (louvain/leiden), multi-method clustering, attribute-enriched tree with configurable split criterion |
| **Demand Transference** | `cdt_assortment_tab` | Delist simulation: compute substitutable demand, cannibalization analysis, waterfall impact charts |
| **Assortment Optimizer** | `cdt_assortment_tab` | SKU rationalization via heuristic MILP, scenario comparison, revenue/coverage trade-offs |
| **Elasticity Analysis** | `pricing_tab` | Log-log OLS, hierarchical empirical Bayes, XGBoost elasticity estimation with SHAP |
| **KVI Identification** | `pricing_tab` | Key Value Item scoring via XGBoost importance or RFM + elasticity hybrid |
| **Price Curve Diagnostics** | `pricing_tab` | K-Means/GMM price tier clustering, tier violation detection |
| **Promo Uplift Modeling** | `pricing_tab` | Causal T-learner / S-learner, Qini curves, uplift by customer segment, propensity stratification |

---

## Data Requirements

### Required CSV Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | datetime | Transaction date | `2024-01-15` |
| `transaction_id` | string | Unique transaction ID | `INV000026` |
| `stockcode` | string | Product SKU/code | `85050` |
| `product` | string | Product description | `CHOCOLATE BAR` |
| `customer_id` | string | Unique customer ID | `CUST0157` |
| `price` | float | Unit price | `19.88` |
| `quantity` | int | Quantity purchased | `1` |

### Optional Columns (Auto-detected)

| Column | Use Case |
|--------|----------|
| `category` | CDT attribute labeling, product hierarchy |
| `brand` | CDT attribute labeling, switching analysis |
| `size` | CDT attribute splits |
| `flavor` / `variant` | CDT attribute splits |
| `promo_flag` | Promotional analytics |
| `is_online` | Channel analysis |

### Sample Data Format

```csv
date,transaction_id,stockcode,product,customer_id,price,quantity,category,brand
2024-01-01,INV000026,85050,CHOCOLATE BAR,CUST0157,19.88,1,Confectionery,BrandA
2024-01-01,INV000026,22093,RETROSPOT PENCIL CASE,CUST0157,3.35,2,Stationery,BrandB
```

> **Data Quality Requirements:**
> - **Minimum viable**: 500+ transactions, 30+ products, 100+ customers
> - **Recommended**: 2,000–5,000 transactions, 50–100 products, 200–500 customers
> - **Production**: 10,000+ transactions, 100+ products, 500+ customers
> - **Critical**: Category must have **frequent repurchase** (weekly groceries). Durables/single-purchase categories won't yield switching signals.

---

## Configuration Reference

### Global FP-Growth Parameters (Sidebar)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Min Support | 0.002 | 0.0005–0.05 | Fraction of transactions containing itemset |
| Min Confidence | 0.10 | 0.01–0.50 | Minimum rule confidence |
| Max Itemset Length | 3 | 2–6 | Max items per frequent itemset |
| Min Lift | 1.2 | 0.5–5.0 | Minimum lift for rule filtering |

### CDT Builder Parameters (CDT & Assortment Category)

#### Similarity
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Similarity Methods | phi | phi, jaccard, pmi, cosine_tfidf | Methods for ensemble similarity |
| Min Co-occurrence | 5 | 2–20 | Min customers buying both products |

#### Community Detection
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Community Method | louvain | none / label_propagation / louvain / leiden | Product graph community detection |
| Resolution | 1.0 | 0.5–2.0 | Community resolution parameter |
| Graph Min Weight | 0.1 | 0.0–0.5 | Minimum edge weight for product graph |
| Graph Max Degree | 50 | 10–100 | Max edges per node |

#### Clustering
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Linkage Method | average | average / complete / single | Agglomerative clustering linkage |
| Min Clusters (k) | 2 | 2–10 | Silhouette search floor |
| Max Clusters (k) | 15 | 3–20 | Silhouette search ceiling |

#### Tree Building
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Min Cluster Size | 3 | 2–10 | Min products per tree node |
| Quality Threshold | 60% | 40–80% | Tree quality vs. unconstrained baseline |
| Split Criterion | mutual_info | mutual_info / gini / entropy / mixed | Attribute split scoring method |
| Split Alpha | 0.5 | 0.0–1.0 | Entropy weight for mixed criterion |

#### Behavioral
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Top N Products | 50 | 20–200 | Limit for large catalogs |
| Min Lift (bundling) | 1.2 | 1.0–3.0 | Co-purchase strength floor |
| Max Substitution | 0.3 | 0.0–0.5 | Substitutability ceiling for bundles |

### Elasticity Analysis Parameters (Pricing & Promotions)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Method | loglog_ols | loglog_ols / hierarchical_eb / xgb | Elasticity estimation method |
| Min Periods | 10 | 5–50 | Min time periods per product |
| Min Price Variation | 5% | 1–50% | Min price coefficient of variation |
| Show SHAP Values | false | — | Enable SHAP for XGBoost method |

### KVI Identification Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Method | xgb_importance | xgb_importance / rfm_elasticity | KVI scoring method |
| Top K KVI | 20 | 10–100 | Number of KVIs to identify |
| Margin-Weighted | No | — | Weight by margin if cost data available |

### Promo Uplift Modeling Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Promo Drop Threshold | 15% | 5–50% | Price drop to flag as promo |
| Baseline Window | 28 days | 14–90 | Historical window for baseline demand |
| Uplift Method | t_learner | t_learner / s_learner | Causal estimation method |
| Base n_estimators | 200 | 50–500 | XGBoost n_estimators |
| Base max_depth | 5 | 3–10 | XGBoost max_depth |
| Propensity Stratification | Yes | — | IPW-based propensity weighting |

---

## Analysis Modes Walkthrough

### 1. Association Rules
Classic market basket analysis. FP-Growth finds frequent itemsets → association rules with lift/confidence. Visualizations: network graph, heatmap, parallel coordinates, rule table with filters.

### 2. Co-purchase Analysis
Symmetric product affinity matrix. Identifies products frequently bought together in same transaction. Outputs: heatmap, top pairs table, bundle recommendations.

### 3. Add-on / Impulse Analysis
Anchor product → add-on recommendations. Ranks by lift to find true impulse items vs. staples. Useful for checkout placement, "frequently bought together" widgets.

### 4. Switching Analysis
Tracks customer brand/product transitions over time. Sankey diagrams show flow; identifies defectors, loyalists, switchers. Configurable time window & min transactions.

### 5. Customer Choice Modelling (Supervised)
Trains a decision tree to predict **next product choice** from customer history (RFM, favorite categories, recency). Outputs: tree visualization, feature importance, prediction accuracy.

### 6. Decision Tree & Patterns — **Customer Decision Tree (CDT)**
**Enterprise-grade unsupervised hierarchy:**
1. **Similarity Matrix** — Yule's Q / Jaccard on co-purchase
2. **Hierarchical Clustering** — Agglomerative with silhouette optimization
3. **Attribute Labeling** — Mutual information splits on category/brand/size/flavor
4. **Tree Construction** — Recursive partitioning with quality threshold
5. **Behavioral Extraction** — Substitution pairs, bundle candidates, cross-sell edges

**Outputs:** Interactive dendrogram, sunburst, treemap, substitution matrix, bundle table, quality metrics.

### 7. Customer Segmentation
- **RFM Quantile** (classic 4×4×4)
- **RFM K-Means** (configurable k)
- **Behavioral Clustering** (purchase patterns)
- **CLV Estimation** (BG/NBD + Gamma-Gamma optional)

### 8. Product Performance
- Lifecycle curves (intro/growth/maturity/decline)
- Price elasticity estimation
- ABC (revenue) + XYZ (volatility) classification
- New product launch tracking

### 9. Cohort Analysis
- Retention heatmaps (weekly/monthly/quarterly)
- Revenue per customer, AOV by cohort
- Configurable periods & metrics

### 10. Promotional Analytics
- Automatic promo detection (price drop thresholds)
- Lift decomposition: **True Incrementality** vs. **Forward Buy** vs. **Substitution**
- Cannibalization & halo effects
- Promo ROI estimation

### 11. CDT Builder (Advanced)
**Enhanced CDT with community detection and ensemble similarity:**
1. **Similarity Ensemble** — Combine Phi, Jaccard, PMI, and Cosine TF-IDF into a single similarity matrix
2. **Community Detection** — Label propagation, Louvain, or Leiden clustering on the product graph
3. **Multi-method Clustering** — Cluster within communities, merge dendrograms
4. **Attribute-enriched Tree** — Split criterion selection (mutual info, Gini, entropy, mixed) with configurable alpha

**Outputs:** Community summary, silhouette analysis, similarity comparison across methods, interactive dendrogram/sunburst/treemap.

### 12. Demand Transference
**Simulate the impact of delisting products:**
- Compute **substitutable demand** between product pairs using switching or CDT-derived substitution matrices
- Estimate **recovery rate**: what fraction of delisted demand transfers to remaining products
- **Cannibalization** analysis: identify which products lose demand when a new product is introduced
- **Waterfall charts** showing net revenue impact per scenario

**Use case:** Planogram optimization, discontinuation decisions, "what-if" delist simulations.

### 13. Assortment Optimizer
**SKU rationalization using constrained optimization:**
- **Heuristic solver** — greedy item selection based on revenue/margin contribution
- **MILP solver** — Mixed Integer Linear Programming with OR-Tools for exact optimization
- **Constraints**: max SKUs, minimum category coverage, substitution bounds
- **Scenario comparison**: compare multiple assortment configurations side-by-side

**Outputs:** Revenue vs. coverage Pareto frontier, scenario comparison table, exportable recommendations.

### 14. Elasticity Analysis
**Price elasticity estimation with three methods:**
- **Log-log OLS** — Classic log-log regression: `log(quantity) = α + β * log(price)`
- **Hierarchical Empirical Bayes** — Partial pooling across product categories for stable estimates with limited data
- **XGBoost** — Non-parametric elasticity: predict demand from price + features, derive elasticity via partial dependence

**Outputs:** Elasticity distribution histogram, category-level comparison, product-level elasticity table with confidence intervals.

### 15. KVI (Key Value Item) Identification
**Score products by their price sensitivity and strategic importance:**
- **XGBoost Importance** — Train demand model with price as feature, derive KVI score from price feature importance
- **RFM + Elasticity Hybrid** — Combine elasticity estimates with RFM-based customer value metrics
- Configurable top-K, margin-weighting option

**Use case:** Identify items where price changes have outsized impact on customer perception and traffic.

### 16. Price Curve Diagnostics
**Detect price tier violations and optimize pricing structure:**
- **K-Means** or **Gaussian Mixture Model** clustering to identify natural price tiers within each category
- **Tier violation detection**: products priced outside their expected tier (overpriced/underpriced relative to comparable items)
- Configurable number of tiers (2–5)

**Outputs:** Tier assignment per product, violation flags, category-level price distribution plots.

### 17. Promo Uplift Modeling
**Causal inference for promotion effectiveness:**
- **T-learner** — Train separate treatment/control models: `E[Y|X, T=1] - E[Y|X, T=0]`
- **S-learner** — Single model with treatment as feature
- **Qini curves** — Evaluate uplift model performance across population segments
- **Propensity stratification** — Control for selection bias in observational data
- **Segment-level uplift** — Identify which customer segments respond best to promotions

**Outputs:** Uplift distribution, Qini curve, segment-level treatment effects, feature importance (SHAP).

---

## UI Overview

### Sidebar Layout
```
Data Upload
  ├── File uploader (CSV)
  ├── Column auto-detection / manual mapping
  └── Use Sample Data checkbox

FP-Growth Parameters
  ├── Min Support, Min Confidence, Max Itemset Length, Min Lift

Analysis Category (radio)
  ├── Association Rules
  │   └── Association Rules / Co-purchase / Add-on / Switching
  ├── CDT & Assortment
  │   ├── CDT Builder (similarity methods, community detection, clustering, tree building, behavioral)
  │   ├── Demand Transference (substitution source, delist products, recovery constraint)
  │   └── Assortment Optimizer (max SKUs, coverage, objective, solver, time limit)
  ├── Pricing & Promotions
  │   ├── Elasticity Analysis (method, min periods, SHAP toggle)
  │   ├── KVI Identification (method, top-K, margin-weighting)
  │   ├── Price Curve Diagnostics (clustering method, n tiers)
  │   └── Promo Uplift Modeling (drop threshold, baseline window, T/S learner, propensity)
  ├── Customer Segmentation
  ├── Product Performance
  ├── Cohort Analysis
  └── Promotional Analytics (legacy)

Run Analysis Button
```

### Main Tabs (Dynamic)
Tabs render based on selected analysis mode. Each tab includes:
- Interactive Plotly visualizations
- Configurable parameters
- Export buttons (CSV, JSON, HTML)

---

## Export Capabilities

| Format | Use Case |
|--------|----------|
| **CSV** | Rule tables, segment assignments, cohort matrices, product metrics |
| **JSON** | CDT tree structure, rule networks, model parameters |
| **HTML** | Interactive Plotly charts (standalone) |
| **PNG/PDF** | Static chart exports via Kaleido |

---

## Testing & Development

### Run Tests (if available)
```bash
pytest tests/ -v
```

### Lint & Format
```bash
# Using ruff (fast)
ruff check src/
ruff format src/

# Or black/isort
black src/
isort src/
```

### Type Check
```bash
mypy src/
```

### Generate Sample Data
```python
from src.data.generator import generate_transactions
df = generate_transactions(n_transactions=5000, n_customers=500, n_products=100, seed=42)
df.to_csv("data/my_sample.csv", index=False)
```

---

## Troubleshooting / FAQ

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in activated venv |
| `ValueError: Missing required columns` | Check CSV has all 7 required columns (see Data Requirements) |
| `MemoryError` on large datasets | Reduce `Top N Products`, increase `Min Support`, or sample data |
| Empty rules / no clusters | Lower `Min Support` (try 0.001), lower `Min Co-occurrence` (try 2) |
| Streamlit won't start | Check port 8501 free; try `streamlit run app.py --server.port 8502` |
| Slow CDT clustering | Reduce `Top N Products` to 50; increase `Min Co-occurrence` |

### Performance Tips

| Dataset Size | Recommended Settings |
|--------------|---------------------|
| < 2K transactions | Default settings OK |
| 2K–10K | Min Support 0.002, Top N Products 50 |
| 10K–50K | Min Support 0.005, Top N Products 30, sample for CDT |
| 50K+ | Use sampled subset for exploration; full run for production |

### Data Quality Checks

```python
# Quick data health check
from src.data.loader import load_transactions, get_data_summary
df = load_transactions("your_file.csv")
summary = get_data_summary(df)
print(f"Transactions: {summary['n_transactions']:,}")
print(f"Customers: {summary['n_customers']:,}")
print(f"Products: {summary['n_products']:,}")
print(f"Date range: {summary['date_range']}")
print(f"Avg basket size: {summary['avg_basket_size']:.1f}")
```

---

## Methodology References

| Method | Reference |
|--------|-----------|
| **FP-Growth** | Han et al., "Mining Frequent Patterns without Candidate Generation" (2000) |
| **Yule's Q** | Yule, "On the Association of Attributes in Statistics" (1912) |
| **Hierarchical Clustering** | SciPy `linkage(metric='precomputed')` |
| **Silhouette Analysis** | Rousseeuw, "Silhouettes" (1987) |
| **Mutual Information Splits** | Quinlan, "Induction of Decision Trees" (1986) |
| **CDT** | Customer Decision Tree Science (public docs) |
| **RFM** | Hughes, "Strategic Database Marketing" (1994) |
| **BG/NBD CLV** | Fader et al., "Counting Your Customers" (2005) |
| **Adjusted Rand Index** | Hubert & Arabie, "Comparing Partitions" (1985) |
| **Normalized Mutual Information** | Strehl & Ghosh, "Cluster Ensembles" (2002) |
| **XGBoost** | Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System" (2016) |
| **Changepoint Detection** | Killick et al., "Optimal Detection of Changepoints" (2012) |
| **SHAP** | Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (2017) |
| **Qini Curve** | Radcliffe & Surry, "Real-World Uplift Modelling" (2011) |
| **MILP (OR-Tools)** | Perron et al., "OR-Tools User's Manual" (2024) |

---

## Validation & Benchmarks

The project includes synthetic ground-truth validation modules for each major analytics pipeline. Each validation generates data with known parameters, runs the analytical method, and reports recovery metrics.

| Module | Function | Metrics | Purpose |
|--------|----------|---------|---------|
| `cdt_validation.py` | `generate_synthetic_cluster_data()` / `run_cdt_validation()` | ARI, NMI, #clusters | Validates hierarchical clustering + tree building against known cluster structure |
| `segmentation_validation.py` | `generate_synthetic_customer_segments()` / `run_segmentation_validation()` | ARI, NMI, #segments found | Validates RFM quantile, RFM K-Means, Behavioral clustering against true segments |
| `promotional_validation.py` | `generate_synthetic_promo_data()` / `run_promo_detection_validation()` | Precision, Recall, F1 | Validates adaptive promo detection (rolling z-score) against injected promo periods |
| `assortment_validation.py` | `generate_synthetic_assortment_instance()` / `run_assortment_validation()` | Objective value, coverage, optimality gap | Validates heuristic vs. local search vs. MILP assortment solvers |
| `validation.py` | `generate_synthetic_elasticity_data()` / `run_validation()` | RMSE, Bias, 94% HDI Coverage | Validates OLS, Hierarchical EB, XGBoost, NUTS Bayesian elasticity methods |

**Running Benchmarks:**
```bash
# CDT validation
python -c "from src.analytics.cdt_validation import run_cdt_validation; print(run_cdt_validation())"

# Segmentation validation
python -c "from src.analytics.segmentation_validation import run_segmentation_validation; print(run_segmentation_validation())"

# Promo validation
python -c "from src.analytics.promotional_validation import run_promo_detection_validation; print(run_promo_detection_validation())"

# Elasticity validation
python -c "from src.analytics.validation import run_validation; print(run_validation())"

# All validations run automatically in CI (see .github/workflows/ci.yml)
```

---

## CDT & Pricing Modes Reference

The sidebar organizes analysis modes into categories. Here's the mapping between sidebar modes and their implementation:

### CDT & Assortment Modes (Unified via `render_cdt_tab`)

| Sidebar Mode | `render_cdt_tab` mode | Description |
|--------------|----------------------|-------------|
| **Decision Tree & Patterns** | `"cdt"` | Legacy/simple CDT with basic hierarchical clustering |
| **CDT Builder** | `"cdt"` | Advanced CDT with ensemble similarity, community detection, configurable clustering |
| **Demand Transference** | `"transference"` | Delist simulation & substitution demand analysis |
| **Assortment Optimizer** | `"assortment"` | SKU rationalization via MILP/heuristic |
| **CDT Benchmark** | `"benchmark"` | Synthetic validation with ARI/NMI metrics |

### Pricing & Promotions Modes (Unified via `render_pricing_tab`)

| Sidebar Mode | `render_pricing_tab` mode | Description |
|--------------|--------------------------|-------------|
| **Elasticity Analysis** | `"elasticity"` | Log-log OLS, Hierarchical EB, XGBoost, Bayesian |
| **KVI Identification** | `"kvi"` | Key Value Item scoring via XGBoost importance |
| **KVI Composite** | `"kvi_composite"` | NielsenIQ 4-signal framework (Elasticity + Penetration + Frequency + Price Recall) |
| **Price Ladder** | `"price_ladder"` | ASP tier chart with violation detection |
| **Price Curve Diagnostics** | `"price_curves"` | K-Means/GMM price tier clustering |
| **Promo Uplift Modeling** | `"promo_uplift"` | Causal T-learner/S-learner with Qini curves |
| **Elasticity Benchmark** | `"benchmark"` | Synthetic elasticity validation |

## Extending the App

### Add a New Analysis Tab

1. Create `src/ui/new_analysis_tab.py` with `render_new_analysis_tab(df, lookup, params)`
2. Add import in `app.py`
3. Add mode to `sidebar.py` analysis categories
4. Add case in `app.py:run_analysis()`

### Add a New Visualization

1. Create function in `src/viz/` (e.g., `custom_viz.py`)
2. Use Plotly `go.Figure` or `px.*`
3. Return `fig` for `st.plotly_chart(fig, use_container_width=True)`

### Custom Similarity Metric

Add to `src/analytics/cdt_similarity.py`:
```python
def my_similarity(matrix: pd.DataFrame) -> pd.DataFrame:
    # matrix: customer x product binary
    # return: product x product similarity
    pass
```

---

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open Pull Request

### Code Style
- Follow existing patterns in `src/`
- Type hints required for new functions
- Docstrings for public functions
- Keep tabs focused (single responsibility)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **SciPy / scikit-learn** — Clustering, metrics, tree algorithms
- **Plotly** — Interactive visualizations
- **Streamlit** — Rapid UI framework
- **FP-Growth** — Adapted from standard implementations
- **Online Retail Dataset** (UCI) — Inspiration for sample data schema

---

## Support

- **Issues**: GitHub Issues for bugs & feature requests
- **Discussions**: GitHub Discussions for questions
- **Documentation**: This README + inline code docs

---

*Built for category managers, data scientists, and retail analysts who need advanced decision intelligence.*# Force rebuild Wed Aug  5 10:52:51 PM CEST 2026
# Deploy trigger Thu Aug  6 09:07:39 AM CEST 2026
