# TrustingSyntheticData - Navigator

> Multiattribute Decision Analysis Framework for Synthetic Data Generation Method Selection

**Tech Stack**: Python, pandas, numpy, scikit-learn, scipy, SDV (CTGAN), DataSynthesizer (PrivBayes)

**Last Updated**: 2026-01-17

---

## Quick Navigation

### Project Structure
```
TrustingSyntheticData/
├── tsd/                    # Python package
│   ├── generators/         # 5 synthetic data generators
│   ├── measures/           # 5 evaluation measures
│   ├── preprocessing/      # Data loading utilities
│   └── analysis/           # Statistical analysis & MADA framework
├── tests/                  # Test suite
├── data/                   # Data files
├── results/                # Generated results
│   └── experiments/        # Experiment outputs & figures
├── .agent/                 # Navigator documentation
│   ├── tasks/              # Implementation plans
│   ├── system/             # Architecture docs
│   └── sops/               # Standard procedures
└── pyproject.toml          # Project configuration
```

### Documentation Index

| Document | Location | When to Load |
|----------|----------|--------------|
| Task Plans | `.agent/tasks/*.md` | Starting feature work |
| System Architecture | `.agent/system/*.md` | Understanding components |
| SOPs | `.agent/sops/**/*.md` | Following procedures |
| Analysis Completion | `.agent/tasks/TASK-001-analysis-complete.md` | Review analysis work |

---

## Current Focus

**Status**: ANALYSIS COMPLETE - Project Ready for Use

**Completed Milestones**:
- Day 7: GReaT GENERATOR FIX COMPLETE (2026-01-17)
  - Fixed seeding bug (results now vary across replicates)
  - Added disk cleanup to Colab notebook
  - Re-ran all 5 GReaT experiments with proper seeding
  - Updated all analysis and visualizations
- Day 5: ANALYSIS & MADA FRAMEWORK COMPLETE (2026-01-15)
  - Statistical analysis with ANOVA, effect sizes, pairwise comparisons
  - 6 publication-ready visualizations
  - MADA decision support system with 4 scenario demos
- Day 4: FULL EXPERIMENTS COMPLETE (2026-01-15)
  - 25 runs (5 methods × 5 replicates)
  - Results: `results/experiments/all_results_complete.csv`
- Day 3: Integration testing PASSED (2026-01-14)
- Day 2: All 5 generators working (2026-01-14)
- Day 1: All 5 measures implemented (2026-01-12)

**Key Findings**:
- DP-BN (PrivBayes) wins 3/4 use cases (privacy, utility, fidelity focused)
- Synthpop wins for fairness-focused applications
- GReaT excels at utility (2nd best) but has lower fidelity/privacy
- Confirmed trade-off: Fidelity/Privacy vs Utility (ρ = -0.7)

**Next Up**: Paper writing, additional datasets, or production deployment

---

## Key Patterns

### Code Organization
- Source code in `tsd/` package
- Tests in `tests/` directory
- Data processing utilities and synthetic data generation methods

### Development Workflow
1. Activate virtual environment: `source .venv/bin/activate`
2. Install dependencies: `pip install -e ".[dev]"`
3. Run tests: `pytest`
4. Format code: `black .` and `ruff check --fix .`

### Testing
- pytest for unit tests
- Test files: `tests/test_*.py`
- Run coverage: `pytest --cov`

---

## Load-On-Demand Sections

### When Starting a New Feature
Load: `.agent/tasks/TASK-XXX.md` (if exists)

### When Debugging
Load: `.agent/sops/debugging/` for relevant SOPs

### When Working with Synthetic Data Generation
Key components:
- SDV library for CTGAN
- DataSynthesizer for PrivBayes (DP Bayesian Network)
- be-great library for GReaT (requires GPU - use Colab)
- scikit-learn for evaluation metrics

**GReaT Generator Notes**:
- Requires GPU for reasonable training time
- Use `tsd/generators/great_colab_notebook_fixed.ipynb` for Colab
- Local wrapper: `tsd/generators/great_generator.py`
- Must set seeds via `torch.manual_seed()` AND TrainingArguments

### When Using the Analysis Module
```python
from tsd.analysis import (
    # Statistical analysis
    run_analysis,              # Full statistical report
    descriptive_statistics,    # Mean, std, CI by method
    omnibus_tests,             # ANOVA, Kruskal-Wallis
    pairwise_comparisons,      # Post-hoc with Bonferroni

    # Visualizations
    generate_all_visualizations,  # All 6 plots

    # MADA Framework
    PROFILES,                  # Pre-defined user profiles
    generate_recommendation,   # Get method recommendation
    sensitivity_analysis,      # Weight sensitivity
)
```

### Key Results Files
| File | Description |
|------|-------------|
| `results/experiments/all_results_complete.csv` | Raw experiment data |
| `results/experiments/statistical_analysis_report.txt` | Full stats report |
| `results/experiments/figures/` | All visualizations |

---

## Navigator Commands Reference

| Command | Purpose |
|---------|---------|
| `/nav:start` | Start session, load navigator |
| `/nav:task` | Create/manage task documentation |
| `/nav:sop` | Create standard operating procedures |
| `/nav:compact` | Save context and clear for new work |
| `/nav:marker` | Create context checkpoint |

---

## Notes

_Add project-specific notes here as you work._
