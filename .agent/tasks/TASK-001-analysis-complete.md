# TASK-001: Statistical Analysis & MADA Framework

**Status**: COMPLETED
**Date**: 2026-01-15
**Duration**: Day 5 of project

---

## Objective

Implement comprehensive statistical analysis of experiment results and create a Multiattribute Decision Analysis (MADA) framework for synthetic data method selection.

---

## Deliverables

### 1. Statistical Analysis Module (`tsd/analysis/statistical_analysis.py`)

| Feature | Description |
|---------|-------------|
| Descriptive Statistics | Mean, std, 95% CI for each method-metric combination |
| Normality Testing | Shapiro-Wilk tests for assumption checking |
| Homogeneity of Variance | Levene's test |
| Omnibus Tests | ANOVA and Kruskal-Wallis for overall differences |
| Effect Sizes | Eta-squared (η²) for practical significance |
| Pairwise Comparisons | t-tests and Mann-Whitney with Bonferroni correction |
| Cohen's d | Effect size for pairwise differences |
| Correlation Analysis | Spearman correlations for trade-off detection |
| Data Quality Checks | Detect constant values, missing data |

### 2. Visualization Module (`tsd/analysis/visualizations.py`)

| Visualization | Purpose |
|--------------|---------|
| Bar Chart Comparison | Compare methods across all metrics with error bars |
| Trade-off Scatter Plots | Show fidelity-utility and privacy-utility trade-offs |
| Box Plots | Display variance across replicates |
| Correlation Heatmap | Visualize metric relationships |
| Radar Chart | Multi-dimensional method profiles |
| Pareto Frontier | Identify Pareto-optimal methods |

### 3. MADA Framework (`tsd/analysis/mada_framework.py`)

| Feature | Description |
|---------|-------------|
| Score Normalization | Min-max and z-score methods |
| Weighted Aggregation | Custom attribute weights |
| Pre-defined Profiles | 5 user archetypes (privacy, utility, balanced, fidelity, fairness) |
| Sensitivity Analysis | How weight changes affect rankings |
| Recommendation Reports | Detailed reports with trade-off warnings |
| Scenario Demos | Healthcare, ML training, bias auditing, publication benchmark |

---

## Key Findings

### Statistical Results

- **All metrics show significant differences** between methods (ANOVA p < 0.001)
- **Large effect sizes** across all metrics (η² > 0.38)
- **Key trade-off confirmed**: Fidelity/Privacy vs Utility (ρ = -0.7)

### Method Rankings by Metric

| Metric | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| Fidelity | DP-BN (0.95) | CTGAN (0.94) | Indep. Marginals (0.88) |
| Privacy | DP-BN (0.56) | CTGAN (0.38) | Indep. Marginals (0.33) |
| Utility | Synthpop (0.99) | GReaT (0.96) | DP-BN (0.85) |
| Fairness | Synthpop (0.03) | Indep. Marginals (0.04) | CTGAN (0.05) |

### MADA Recommendations

| Scenario | Weights | Recommended |
|----------|---------|-------------|
| Healthcare (privacy 45%) | P:45, U:30, F:15, Fa:10 | DP-BN |
| ML Training (utility 55%) | U:55, F:25, P:10, Fa:10 | DP-BN |
| Bias Auditing (fairness 50%) | Fa:50, U:25, F:15, P:10 | Synthpop |
| Publication (fidelity 45%) | F:45, U:30, P:15, Fa:10 | DP-BN |

---

## Files Created

```
tsd/analysis/
├── __init__.py                 # Module exports (updated)
├── statistical_analysis.py     # NEW: Statistical tests
├── visualizations.py           # NEW: Plotting functions
└── mada_framework.py           # NEW: Decision support

results/experiments/
├── statistical_analysis_report.txt  # NEW: Full report
└── figures/
    ├── method_comparison_bars.png   # NEW
    ├── tradeoff_scatter.png         # NEW
    ├── replicate_boxplots.png       # NEW
    ├── correlation_heatmap.png      # NEW
    ├── radar_comparison.png         # NEW
    ├── pareto_frontier.png          # NEW
    ├── mada_profile_comparison.png  # NEW
    ├── mada_healthcare_scenario.png # NEW
    └── mada_all_scenarios.png       # NEW
```

---

## Data Quality Issues Identified

1. ~~**GReaT method**: Identical values across all 5 replicates - likely generator bug~~ **FIXED in TASK-002**
2. **Membership Inference**: Constant AUC (0.54) - metric may not be discriminative

---

## Usage Examples

### Quick Statistical Report
```bash
python -m tsd.analysis.statistical_analysis results/experiments/all_results_complete.csv
```

### Generate All Visualizations
```bash
python -m tsd.analysis.visualizations results/experiments/all_results_complete.csv
```

### Run MADA Demo
```bash
python -m tsd.analysis.mada_framework results/experiments/all_results_complete.csv
```

### Python API
```python
from tsd.analysis import (
    run_analysis,
    generate_all_visualizations,
    generate_recommendation,
    PROFILES,
)

# Full analysis
report, results = run_analysis("results/experiments/all_results_complete.csv")

# Custom recommendation
df = load_results("results/experiments/all_results_complete.csv")
weights = {"privacy": 0.5, "utility": 0.3, "fidelity": 0.1, "fairness": 0.1}
print(generate_recommendation(df, weights, "Custom Scenario"))
```

---

## Next Steps

1. ~~**Fix GReaT generator** - Investigate constant output issue~~ **COMPLETED - see TASK-002**
2. **Paper writing** - Use visualizations and findings for publication
3. **Additional datasets** - Validate findings on other datasets
4. **Web interface** - Build interactive MADA tool for researchers

---

## References

- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
- Keeney, R. L., & Raiffa, H. (1993). Decisions with Multiple Objectives
