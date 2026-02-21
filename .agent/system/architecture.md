# System Architecture

> TrustingSyntheticData - MADA Framework for Synthetic Data Method Selection

**Last Updated**: 2026-01-23

---

## Overview

This project implements a Multiattribute Decision Analysis (MADA) framework for evaluating and comparing synthetic data generation methods. The architecture supports:

1. **Data Generation**: 5 synthetic data generation methods
2. **Evaluation**: 5 quality measures across privacy, utility, fidelity, fairness
3. **Decision Support**: MADA framework with user profiles and recommendations
4. **Value Analysis**: VOI (Value of Information) analysis for decision-making

---

## Module Architecture

```
tsd/
├── generators/           # Synthetic data generation
│   ├── ctgan_generator.py
│   ├── dpbn_generator.py (PrivBayes)
│   ├── independent_marginals.py
│   ├── synthpop_generator.py
│   ├── great_generator.py (wrapper)
│   └── great_colab_notebook_fixed.ipynb
│
├── measures/             # Evaluation metrics
│   ├── fidelity.py       # Propensity Score AUC
│   ├── privacy.py        # DCR + Membership Inference
│   ├── utility.py        # TSTR F1 Ratio
│   └── fairness.py       # Demographic Parity Gap
│
├── preprocessing/        # Data loading & preparation
│   └── data_loader.py
│
├── analysis/             # Statistical analysis & decision support
│   ├── statistical_analysis.py  # ANOVA, effect sizes, pairwise
│   ├── visualizations.py        # 6 publication plots
│   ├── mada_framework.py        # Decision support system
│   └── voi_analysis.py          # Value of Information
│
└── utils/                # Helpers
    └── ...
```

---

## Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Raw Data   │────▶│  Generator  │────▶│  Synthetic  │
│  (psam_p06) │     │  (5 types)  │     │    Data     │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Measures   │
                                        │ (5 metrics) │
                                        └──────┬──────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    MADA     │◀────│  Analysis   │◀────│   Results   │
│  Framework  │     │ (stats/viz) │     │    CSV      │
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## Key Components

### 1. Generators

| Generator | Library | Notes |
|-----------|---------|-------|
| CTGAN | `sdv` | Deep learning GAN |
| DP-BN | `DataSynthesizer` | PrivBayes with ε=1.0 |
| Independent Marginals | Custom | Baseline (no correlations) |
| Synthpop | `synthpop` (R) | CART-based |
| GReaT | `be-great` | LLM fine-tuning, GPU required |

### 2. Measures

| Measure | Range | Better | Description |
|---------|-------|--------|-------------|
| Fidelity | [0,1] | Higher | 1 - Propensity AUC (closer to 0.5 = better) |
| Privacy (DCR) | [0,∞) | Higher | 5th percentile distance to closest record |
| Utility | [0,1] | Higher | TSTR F1 / TRTR F1 ratio |
| Fairness | [0,1] | Lower | Max demographic parity gap across groups |

### 3. MADA Framework

**User Profiles**:
- `privacy_focused`: w = {privacy: 0.45, utility: 0.30, fidelity: 0.15, fairness: 0.10}
- `utility_focused`: w = {utility: 0.40, fidelity: 0.30, privacy: 0.15, fairness: 0.15}
- `balanced`: w = {all: 0.25}
- `fidelity_focused`: w = {fidelity: 0.45, ...}
- `fairness_focused`: w = {fairness: 0.50, ...}

**Scoring**:
```python
score = Σ (w_i × normalized_value_i)
```

### 4. VOI Analysis

Implements Keisler (2004) framework:
- Strategy comparison (Random, Heuristic, Rough, Full)
- VOI decomposition (Prioritization vs Information)
- Decision rules derivation

---

## Results Files

| File | Purpose |
|------|---------|
| `all_results_complete.csv` | Raw experimental data (25 runs) |
| `statistical_analysis_report.txt` | Full statistical report |
| `voi_analysis_results.txt` | Decision analysis results |
| `method_summary.csv` | Method rankings |
| `reproducibility_check.json` | Validation status |
| `figures/*.png` | All visualizations |

---

## External Dependencies

**Python**:
- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: ML evaluation, propensity scoring
- `scipy`: Statistical tests
- `sdv`: CTGAN generator
- `DataSynthesizer`: PrivBayes generator
- `matplotlib`, `seaborn`: Visualization

**R** (via subprocess):
- `synthpop`: Synthpop generator

**GPU** (optional, for GReaT):
- `be-great`: LLM-based generator
- `torch`: PyTorch backend

---

## Configuration

**Experiment Settings** (from `run_experiments.py`):
- N_SAMPLES: 35,000 records
- N_REPLICATES: 5 per method
- RANDOM_SEEDS: [42, 123, 456, 789, 1011]
- DP_EPSILON: 1.0 (for PrivBayes)

**MADA Settings** (from `mada_framework.py`):
- Normalization: Min-max scaling
- Aggregation: Weighted sum
- Sensitivity: ±20% weight perturbation
