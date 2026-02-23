# TrustingSyntheticData

[![Tests](https://github.com/MichaelKoo21/TrustingSyntheticData/actions/workflows/test.yml/badge.svg)](https://github.com/MichaelKoo21/TrustingSyntheticData/actions/workflows/test.yml)

A Multiattribute Decision Analysis (MADA) framework for evaluating and selecting synthetic data generation methods.

## Overview

This project provides a systematic approach to comparing synthetic data generation methods across multiple quality dimensions:

- **Fidelity**: Statistical similarity to original data (Propensity Score AUC)
- **Privacy**: Protection against re-identification (Distance to Closest Record)
- **Utility**: Downstream ML model performance (Train-Synthetic-Test-Real)
- **Fairness**: Equitable outcomes across groups (Demographic Parity Gap)

## Key Finding

**No single method dominates across all metrics.** Method selection depends on user priorities:

| Use Case | Recommended Method | Why |
|----------|-------------------|-----|
| Healthcare/Finance (privacy-critical) | DP-BN (PrivBayes) | Best privacy + fidelity |
| ML Model Training | DP-BN (PrivBayes) | Strong utility + fidelity |
| Bias Auditing | Synthpop | Best fairness preservation |
| Academic Benchmarks | DP-BN (PrivBayes) | Best statistical similarity |

## Installation

```bash
# Clone repository
git clone https://github.com/MichaelKoo21/TrustingSyntheticData.git
cd TrustingSyntheticData

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

## Data Setup

The analysis commands (`tsd recommend`, `tsd verify`, `tsd analyze`) work out of the box using the pre-computed results in `results/experiments/all_results.csv`. No raw data download is needed.

To regenerate synthetic data from scratch, download the ACS PUMS source data:

```bash
bash scripts/download_data.sh
```

This downloads the 2024 ACS PUMS 1-Year California person file (~278MB) to `data/raw/`.

### Optional Dependencies

**Synthpop** requires R and the `synthpop` R package:

```bash
# Install R, then from an R console:
install.packages("synthpop")
```

**GReaT** requires a GPU. Use the provided Colab notebook (`notebooks/great_colab_t4.ipynb`) with a T4 runtime.

**Visualizations** require the viz extras:

```bash
pip install -e ".[viz]"
```

## Quick Start

### Run Statistical Analysis

```bash
tsd analyze --results results/experiments/all_results.csv
```

### Get Method Recommendation

```bash
# Available profiles: privacy_first, utility_first, balanced
tsd recommend --results results/experiments/all_results.csv --profile balanced
```

Or via Python:

```python
from tsd.analysis import generate_recommendation, load_results, PROFILES

df = load_results("results/experiments/all_results.csv")
profile = PROFILES["privacy_first"]
recommendation = generate_recommendation(df, profile.weights, profile.name)
print(recommendation)
```

### Verify Results

```bash
tsd verify --results results/experiments/all_results.csv
```

## Methods Evaluated

*Raw metric means across 5 replicates. Fidelity AUC: closer to 0.5 = better. Privacy DCR: higher = better. Utility TSTR: higher = better. Fairness Gap: lower = better.*

| Method | Type | Fidelity AUC | Privacy DCR | Utility TSTR | Fairness Gap |
|--------|------|:------------:|:-----------:|:------------:|:------------:|
| DP-BN (PrivBayes) | Bayesian Network | 0.85 | 0.11 | 0.92 | 0.06 |
| CTGAN | Deep Learning | 0.80 | 0.08 | 0.94 | 0.06 |
| Indep. Marginals | Baseline | 0.88 | 0.14 | 0.04 | 0.08 |
| GReaT | Transformer/LLM | 0.73 | 0.05 | 0.89 | 0.08 |
| Synthpop | Statistical/CART | 0.48 | 0.01 | 1.00 | 0.03 |

**Note on Membership Inference:** The membership inference attack metric (`privacy_mi_auc`) was evaluated but the attack classifier could not distinguish train from test records on this dataset (AUC = 0.50, essentially random). This is expected for a large public survey dataset with random train/test splits. Privacy is measured by **Distance to Closest Record (DCR)**, which is the operative privacy metric in our analysis.

### GReaT Generator Notes

GReaT uses language models (GPT-2) for tabular data generation:
- Requires GPU (use Google Colab)
- Use `notebooks/great_colab_t4.ipynb` on Google Colab with a T4 GPU
- Local wrapper: `tsd/generators/great_generator.py`

## Project Structure

```
TrustingSyntheticData/
├── tsd/                 # Python package
│   ├── generators/      # 5 synthetic data generators
│   ├── measures/        # 5 evaluation measures
│   ├── preprocessing/   # Data loading utilities
│   ├── analysis/        # MADA framework, VOI, figures, verification
│   ├── cli.py           # CLI entry point
│   └── config.py        # Dataset configuration
├── tests/               # Test suite
├── results/
│   └── experiments/     # Results and figures
└── data/                # Input datasets
```

## Results

Experiment results are in `results/experiments/`:

- `all_results.csv` - Raw data (25 runs: 5 methods × 5 replicates)
- `summary.csv` - Method summary statistics
- `figures/` - Publication-ready visualizations

### Generated Figures

| Figure | Description |
|--------|-------------|
| `method_comparison_bars.png` | Bar chart comparison across metrics |
| `tradeoff_scatter.png` | Trade-off analysis plots |
| `replicate_boxplots.png` | Variance across replicates |
| `correlation_heatmap.png` | Metric correlations |
| `radar_comparison.png` | Multi-dimensional comparison |
| `pareto_frontier.png` | Pareto optimal methods |
| `mada_all_scenarios.png` | MADA framework scenarios |

## Reproducing Paper Results

All claims in the paper can be verified from the committed results:

```bash
# Verify statistical claims match paper tables
tsd verify

# View MADA recommendations for all stakeholder profiles
tsd recommend --profile balanced

# Regenerate all figures (requires viz extras)
tsd analyze
```

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov

# Format code
black .
ruff check --fix .
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{trustingsyntheticdata2026,
  title = {TrustingSyntheticData: A MADA Framework for Synthetic Data Method Selection},
  year = {2026},
  url = {https://github.com/MichaelKoo21/TrustingSyntheticData}
}
```

## License

MIT License
