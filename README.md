# TrustingSyntheticData

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
git clone https://github.com/username/TrustingSyntheticData.git
cd TrustingSyntheticData

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

## Quick Start

### Run Statistical Analysis

```python
from src.analysis import run_analysis

report, results = run_analysis("results/experiments/all_results_complete.csv")
print(report)
```

### Get Method Recommendation

```python
from src.analysis import generate_recommendation, load_results

df = load_results("results/experiments/all_results_complete.csv")

# Custom weights for your use case
weights = {
    "privacy": 0.45,   # High priority
    "utility": 0.30,
    "fidelity": 0.15,
    "fairness": 0.10,
}

recommendation = generate_recommendation(df, weights, "My Scenario")
print(recommendation)
```

### Use Pre-defined Profiles

```python
from src.analysis import PROFILES, generate_recommendation, load_results

df = load_results("results/experiments/all_results_complete.csv")

# Available profiles: privacy_focused, utility_focused, balanced,
#                    fidelity_focused, fairness_focused
profile = PROFILES["privacy_focused"]
recommendation = generate_recommendation(df, profile.weights, profile.name)
```

### Generate Visualizations

```python
from src.analysis import generate_all_visualizations

figures = generate_all_visualizations(
    "results/experiments/all_results_complete.csv",
    output_dir="results/experiments/figures"
)
```

## Methods Evaluated

| Method | Type | Fidelity | Privacy | Utility | Best For |
|--------|------|----------|---------|---------|----------|
| DP-BN (PrivBayes) | Bayesian Network | 0.95 | 0.56 | 0.85 | Privacy-critical apps |
| CTGAN | Deep Learning | 0.94 | 0.38 | 0.34 | Complex patterns |
| Indep. Marginals | Baseline | 0.88 | 0.33 | 0.63 | Quick baseline |
| GReaT | Transformer/LLM | 0.76 | 0.15 | 0.96 | High utility needs |
| Synthpop | Statistical/CART | 0.63 | 0.08 | 0.99 | Fairness + utility |

### GReaT Generator Notes

GReaT uses language models (GPT-2) for tabular data generation:
- Requires GPU (use Google Colab)
- Use `src/generators/great_colab_notebook_fixed.ipynb`
- Local wrapper: `src/generators/great_generator.py`

## Project Structure

```
TrustingSyntheticData/
├── src/
│   ├── generators/      # 5 synthetic data generators
│   ├── measures/        # 5 evaluation measures
│   ├── preprocessing/   # Data loading utilities
│   └── analysis/        # Statistical analysis & MADA framework
├── tests/               # Test suite
├── results/
│   └── experiments/     # Results and figures
└── data/                # Input datasets
```

## Results

Experiment results are in `results/experiments/`:

- `all_results_complete.csv` - Raw data (25 runs: 5 methods × 5 replicates)
- `statistical_analysis_report.txt` - Full statistical report
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
  url = {https://github.com/username/TrustingSyntheticData}
}
```

## License

MIT License
