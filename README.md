# Market Basket Analysis — Customer Decision Intelligence

> **A full-featured Streamlit application for market basket analysis, customer choice modeling, and Customer Decision Tree (CDT) construction.**

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

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

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
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   └── sample_transactions.csv     # Sample data (generated on first run)
├── src/
│   ├── algorithms/
│   │   └── fpgrowth.py             # FP-Growth frequent itemset mining
│   ├── analytics/                  # Core analytics modules
│   │   ├── copurchase.py           # Co-purchase analysis
│   │   ├── cdt_*.py                # Customer Decision Tree modules
│   │   ├── cohort.py               # Cohort analysis
│   │   ├── promotional.py          # Promotional analytics
│   │   ├── segmentation.py         # Customer segmentation (RFM, behavioral)
│   │   ├── switching.py            # Brand/product switching
│   │   ├── addon.py                # Add-on / impulse analysis
│   │   └── product_performance.py  # Product lifecycle & performance
│   ├── data/
│   │   ├── loader.py               # CSV loading & validation
│   │   └── generator.py            # Synthetic data generator
│   ├── models/
│   │   └── decision_tree.py        # Supervised choice prediction
│   ├── rules/
│   │   └── generator.py            # Association rule generation
│   ├── ui/                         # Streamlit UI tabs
│   │   ├── sidebar.py              # Sidebar configuration
│   │   ├── rules_tab.py            # Association Rules tab
│   │   ├── copurchase_tab.py       # Co-purchase tab
│   │   ├── addon_tab.py            # Add-on tab
│   │   ├── switching_tab.py        # Switching tab
│   │   ├── tree_tab.py             # Choice Prediction Model tab
│   │   ├── cdt_tab.py              # Decision Tree & Patterns tab (CDT)
│   │   ├── segmentation_tab.py     # Customer Segmentation tab
│   │   ├── product_performance_tab.py
│   │   ├── cohort_tab.py           # Cohort Analysis tab
│   │   ├── promotional_tab.py      # Promotional Analytics tab
│   │   └── export.py               # Export utilities
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

### CDT-Specific Parameters (Decision Tree & Patterns Tab)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Similarity Method | Yule's Q | Yule's Q / Jaccard | Pairwise product similarity coefficient |
| Min Co-occurrence | 5 | 2–20 | Min customers buying both products |
| Linkage Method | average | avg/complete/single | Agglomerative clustering linkage |
| Min Clusters (k) | 2 | 2–10 | Silhouette search floor |
| Max Clusters (k) | 15 | 3–20 | Silhouette search ceiling |
| Min Cluster Size | 3 | 2–10 | Min products per tree node |
| Quality Threshold | 60% | 40–80% | Tree quality vs. unconstrained baseline |
| Top N Products | 50 | 20–200 | Limit for large catalogs |
| Min Lift (bundling) | 1.2 | 1.0–3.0 | Co-purchase strength floor |
| Max Substitution | 0.3 | 0.0–0.5 | Substitutability ceiling for bundles |

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

Analysis Options
  ├── Analysis Category (6 categories)
  │   └── Analysis Mode (sub-modes per category)
  └── Mode-specific parameters

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

---

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

*Built for category managers, data scientists, and retail analysts who need advanced decision intelligence.*