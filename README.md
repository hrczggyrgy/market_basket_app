# Market Basket Intelligence

> **A full-featured Streamlit application for market basket analysis, customer choice modeling, and Customer Decision Intelligence.**

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
pip install -e .

# Run the app
streamlit run app.py
```

---

## Project Structure

```
market_basket_app/
├── app.py                          # Main Streamlit entry point
├── pyproject.toml                  # Canonical dependency specification
├── README.md                       # This file
├── sample_data/
│   └── sample_transactions.csv     # Sample transaction data
├── src/
│   ├── analytics/                  # Core analytics modules
│   │   ├── assortment.py           # Assortment optimization (MILP/heuristic)
│   │   ├── basket_metrics.py       # Basket penetration & value metrics
│   │   ├── cdt/                    # Customer Decision Tree package
│   │   │   ├── attributes.py       # Product attribute derivation
│   │   │   ├── clustering.py       # Hierarchical clustering
│   │   │   ├── community.py        # Community detection
│   │   │   ├── similarity.py       # Similarity matrices (phi, Jaccard, PMI, TF-IDF)
│   │   │   ├── tree.py             # CDT tree construction
│   │   │   └── validation.py       # Synthetic validation
│   │   ├── clv.py                  # Customer Lifetime Value (BG/NBD + Gamma-Gamma)
│   │   ├── cohort.py               # Cohort analysis
│   │   ├── copurchase.py           # Co-purchase affinity
│   │   ├── performance.py          # Product performance (ABC/XYZ, velocity, lifecycle)
│   │   ├── pricing/                # Pricing analytics package
│   │   │   ├── elasticity.py       # Log-log OLS, hierarchical EB, cross-price
│   │   │   ├── causal.py           # IV elasticity, RDD, synthetic control
│   │   │   └── kvi.py              # Key Value Item scoring
│   │   ├── promo.py                # Promotional analytics (detection, lift, waterfall, ROI)
│   │   ├── schemas.py              # Data contracts & validators
│   │   ├── segmentation/           # Customer segmentation
│   │   │   ├── core.py             # RFM features, clustering
│   │   │   └── rfm.py              # RFM feature computation
│   │   ├── switching.py            # Product switching analysis
│   │   └── transference.py         # Demand transference / demand recovery
│   ├── config.py                   # Configuration (Pydantic Settings)
│   ├── data/
│   │   └── loader.py               # CSV loading & validation
│   ├── ui/                         # Streamlit UI layer
│   │   ├── plots.py                # Shared Plotly helpers (palette, new_fig, show, empty_state)
│   │   ├── registry.py             # Mode registry (ModeSpec, register_mode, dispatch)
│   │   └── tabs/                   # Individual analysis tabs
│   │       ├── overview.py         # KPIs, revenue trend, new/returning, Pareto
│   │       ├── rules.py            # Association rules with lift CI, redundancy, network
│   │       ├── copurchase.py       # Co-occurrence heatmap, PageRank network, pair trends
│   │       ├── switching.py        # Sankey, switcher/loyalist, monthly net, transition heatmap
│   │       ├── cohorts.py          # Retention/revenue heatmaps, AOV curves, YoY
│   │       ├── performance.py      # ABC Pareto, XYZ scatter, lifecycle, velocity vs repeat
│   │       ├── pricing_page.py     # Elasticity + shrink, KVI, price curves
│   │       ├── segmentation.py     # RFM, behavioral, value-based segments
│   │       ├── cdt_page.py         # Similarity heatmap, dendrogram, Sankey tree, split importance
│   │       ├── assortment_page.py  # Coverage gauge, waterfall, category coverage, scenarios
│   │       ├── clv_page.py         # CLV distribution, segments, freq vs recency, diagnostics
│   │       └── promo_page.py       # Promo periods, lift DiD, waterfall, ROI+CI, timing
│   └── utils/
│       └── session.py              # Session state helpers
└── tests/
    └── unit/                       # Unit tests (216 passing)
```

---

## Features

### 12 Analysis Tabs

| Tab | Key Visuals | Analytics |
|-----|-------------|-----------|
| **Overview** | Revenue trend + rolling avg, New/Returning stacked area, Basket histogram, Pareto | KPIs, data quality |
| **Rules** | Lift CI error bars, Redundancy toggle, Anchor drill-down, Rule network | FP-Growth, bootstrap lift CI, redundant rule detection |
| **Co-purchase** | Co-occurrence heatmap, PageRank centrality network, Pair trends, Affinity profile | Phi coefficient affinity, co-occurrence matrix |
| **Switching** | Sankey flow, Switcher/Loyalist bars, Monthly net flow, Transition heatmap | Customer sequence analysis, transition matrices |
| **Cohorts** | Retention heatmap, Revenue heatmap, AOV curves, YoY bars + growth | Retention tables, LTV curves, period-over-period |
| **Performance** | ABC Pareto, XYZ scatter, Lifecycle scatter, Velocity vs Repeat, SKU rationalization | ABC/XYZ classification, velocity, lifecycle stages |
| **Pricing** | Elasticity table + shrink, KVI scores, Price curves | Log-log OLS, hierarchical EB, cross-price, curve diagnostics |
| **Segmentation** | RFM (k-means/quantile), Behavioral, Value-based with BG/NBD CLV | RFM features, k-means, survival modeling |
| **CDT** | Similarity heatmap, Dendrogram, Sankey tree, Split importance | Ensemble similarity, hierarchical clustering, decision tree |
| **Assortment** | Coverage gauge, Revenue waterfall, Category coverage, Scenario scatter, SKU treemap | Heuristic/MILP optimization, demand transference |
| **CLV** | CLV histogram, Segment bars, Freq vs Recency, P(Alive) vs CLV, Diagnostics | BG/NBD + Gamma-Gamma, bootstrap CI |
| **Promotions** | Promo timeline, Lift DiD bars, Waterfall, ROI with CI, Timing by DoW/Month | Promo detection, DiD lift, incrementality waterfall, bootstrap ROI |

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
| `cost` | Margin calculations |

---

## Configuration Reference

### CDT Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Similarity Methods | `phi` | phi, jaccard, pmi, cosine_tfidf, ensemble | Methods for similarity matrix |
| Min Co-occurrence | 5 | 2–20 | Min customers buying both products |
| Community Method | `none` | none, label_propagation, louvain, leiden | Product graph community detection |
| Resolution | 1.0 | 0.5–2.0 | Community resolution parameter |
| Linkage Method | `ward` | ward, average, complete, single | Agglomerative clustering linkage |
| Min Clusters | 2 | 2–10 | Silhouette search floor |
| Max Clusters | 15 | 3–20 | Silhouette search ceiling |
| Min Cluster Size | 3 | 2–10 | Min products per tree node |
| Quality Threshold | 60% | 40–80% | Tree quality vs baseline |
| Split Criterion | `mutual_info` | mutual_info, gini, entropy, mixed | Attribute split scoring method |

### Pricing / Elasticity Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Method | `robust` | robust, linregress | OLS estimator (HC3 vs scipy linregress) |
| Min Periods | 10 | 5–50 | Min time periods per SKU |
| Min Price Variation | 5% | 1–50% | Min price coefficient of variation |
| Min Distinct Prices | 3 | — | Minimum distinct price points per SKU |

### Promotions Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Price Drop Threshold | 15% | 5–50% | Price drop to flag as promo |
| Min Duration | 3 days | 1–30 | Minimum promo duration |
| Max Duration | 60 days | 7–120 | Maximum promo duration |
| Baseline Window | 30 days | 14–90 | Window for DiD lift calculation |

---

## Architecture

### Mode Registry Pattern

Tabs register themselves via `MODE_SPEC` in `src/ui/registry.py`:

```python
from src.ui.registry import ModeSpec, register_mode
from src.ui.plots import PALETTE, new_fig, show

MODE_SPEC = ModeSpec(
    key="overview",
    label="Overview",
    icon=":material/analytics:",
    handler=render,
)
register_mode(MODE_SPEC)
```

The app loads data once, then dispatches to the selected mode's handler with the transaction DataFrame.

### Shared Plot Helpers (`src/ui/plots.py`)

```python
PALETTE = ["#006D77", "#83C5BE", "#EDF6F9", "#FFDDD2", "#E29578"]
new_fig()          # Creates go.Figure with dark template + consistent styling
show(fig)          # fig.update_layout + st.plotly_chart(fig, width="stretch")
empty_state(msg)   # Returns a Figure with centered message for empty data
```

### Data Contracts (`src/analytics/schemas.py`)

All analytics outputs are validated against `DataContract` schemas with column requirements and semantic validators (e.g., `p_value in [0,1]`, `elasticity <= 0` for demand). Validation happens at function return via `check(df, CONTRACT)`.

---

## Testing

```bash
# Run full test suite
.venv/bin/python -m pytest tests/unit -x -q -k "not slow"

# Run specific test module
.venv/bin/python -m pytest tests/unit/test_pricing.py -v

# Run with coverage
.venv/bin/python -m pytest tests/unit --cov=src --cov-report=term-missing
```

**Current status:** 216 tests passing, 4 deselected.

---

## Methodology References

| Method | Reference |
|--------|-----------|
| **FP-Growth** | Han et al., "Mining Frequent Patterns without Candidate Generation" (2000) |
| **Phi Coefficient** | Yule, "On the Association of Attributes in Statistics" (1912) |
| **Hierarchical Clustering** | SciPy `linkage(metric='precomputed')` |
| **Silhouette Analysis** | Rousseeuw, "Silhouettes" (1987) |
| **Mutual Information Splits** | Quinlan, "Induction of Decision Trees" (1986) |
| **CDT** | Customer Decision Tree methodology |
| **RFM** | Hughes, "Strategic Database Marketing" (1994) |
| **BG/NBD CLV** | Fader et al., "Counting Your Customers" (2005) |
| **Empirical Bayes Elasticity** | Rossi et al., "Bayesian Statistics and Marketing" (2005) |
| **Causal Inference (T/S-learner)** | Künzel et al., "Meta-learners for Estimating Heterogeneous Treatment Effects" (2019) |
| **Qini Curve** | Radcliffe & Surry, "Real-World Uplift Modelling" (2011) |
| **MILP (SciPy)** | SciPy `milp` with `LinearConstraint` |
| **PageRank** | Page et al., "The PageRank Citation Ranking" (1999) |

---

## Validation & Benchmarks

Each analytics module includes synthetic validation:

```bash
# CDT validation (ARI, NMI)
python -c "from src.analytics.cdt.validation import run_cdt_validation; print(run_cdt_validation())"

# Segmentation validation (ARI, NMI)
python -c "from src.analytics.segmentation.core import run_segmentation_validation; print(run_segmentation_validation())"

# Elasticity validation (RMSE, bias, HDI coverage)
python -c "from src.analytics.pricing.elasticity import run_validation; print(run_validation())"

# Promo detection validation (Precision, Recall, F1)
python -c "from src.analytics.promo import run_promo_validation; print(run_promo_validation())"

# Assortment validation (objective, coverage, optimality gap)
python -c "from src.analytics.assortment import run_assortment_validation; print(run_assortment_validation())"
```

---

## Extending the App

### Add a New Analysis Tab

1. Create `src/ui/tabs/new_tab.py` with `render(df: pd.DataFrame) -> None`
2. Add `MODE_SPEC = ModeSpec(key="new_tab", ...)` at bottom
3. Import in `app.py` and registry registers automatically

### Add a Visualization

```python
# In src/ui/plots.py or directly in tab
fig = new_fig()
fig.add_trace(go.Bar(x=..., y=..., marker_color=PALETTE[0]))
show(fig)  # Uses width="stretch"
```

### Custom Similarity Metric

Add to `src/analytics/cdt/similarity.py`:
```python
def my_similarity(df: pd.DataFrame) -> pd.DataFrame:
    # df: customer x product binary matrix
    # return: product x product similarity
    pass
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **SciPy / scikit-learn** — Clustering, metrics, tree algorithms
- **Plotly** — Interactive visualizations
- **Streamlit** — Rapid UI framework
- **lifetimes** — BG/NBD + Gamma-Gamma CLV
- **networkx** — Graph algorithms for CDT/co-purchase
- **Online Retail Dataset (UCI)** — Inspiration for sample data schema

---

*Built for category managers, data scientists, and retail analysts who need advanced Customer Decision Intelligence.*