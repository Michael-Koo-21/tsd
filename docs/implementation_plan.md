# Implementation Plan: Multiattribute Decision Analysis for Synthetic Data Generation

**Project:** TrustingSyntheticData  
**Authors:** Michael Koo & Alfonso Berumen  
**Date:** January 5, 2026  
**Target Journal:** INFORMS Decision Analysis

---

## Executive Summary

This implementation plan details the technical execution of an empirical study comparing five synthetic data generation methods using multiattribute decision analysis. The study uses California ACS PUMS 2024 person-level microdata (N=50,000 sample from 322,055 eligible adult records) and evaluates methods across fidelity, privacy, utility, fairness, and efficiency objectives.

**Estimated Total Duration:** 4-5 weeks  
**Primary Risk:** GReaT method GPU requirements  
**Recommended Approach:** Phased implementation with early feasibility validation

---

## 1. Data Understanding

### 1.1 Dataset Overview

| Attribute | Value |
|-----------|-------|
| Source File | `data/raw/psam_p06.csv` |
| Total Records | 393,725 |
| Total Columns | 286 |
| Adults (AGEP ≥ 18) | 322,055 |
| Geography | California only (`STATE` unique value = 06) |
| Adults with Numeric PINCP | 322,055 (after filtering adults; note raw file includes non-numeric “N/A” codes for children) |
| Target Sample | 50,000 (15.5% of eligible) |

### 1.2 Target Variables Identified

All specified variables are **present and available** in the dataset:

#### Continuous Variables (4)

| Variable | Code | Description | Missing % | Range | Notes |
|----------|------|-------------|-----------|-------|-------|
| Age | `AGEP` | Person's age | 0% | 0-94 | Integer, top-coded at 94 |
| Income | `PINCP` | Total person's income | 0%* | -$19,998 to $4.2M | *After filtering adults; raw file contains “N/A” codes for children (<15) |
| Hours Worked | `WKHP` | Usual hours/week | ~37%** | 1-99 | **“N/A” for non-workers (ACS universe restrictions) |
| Commute Time | `JWMNP` | Travel time to work | ~52%** | 1-141 min | **“N/A” for non-commuters (ACS universe restrictions) |

**⚠️ Discrepancy / Design Choice Note:** WKHP and JWMNP are not “missing at random”; they are structurally **Not in Universe** for non-workers / non-commuters. For publication-quality defensibility, explicitly choose one of:
1. **Primary analysis (recommended):** Keep WKHP/JWMNP and model their “N/A” status explicitly (missing indicators + consistent handling across methods).
2. **Worker-only analysis:** Restrict to records with valid employment/commute universes; report reduced N and changed target distribution.
3. **Reduced-variable robustness check:** Drop WKHP/JWMNP and re-run the full pipeline to test whether rankings are robust to variable inclusion.

#### Categorical Variables (8)

| Variable | Code | Description | Missing % | Cardinality | Recode Needed |
|----------|------|-------------|-----------|-------------|---------------|
| Sex | `SEX` | Sex | 0% | 2 | None |
| Race | `RAC1P` | Recoded race | 0% | 9 | Yes → 5 categories |
| Education | `SCHL` | Educational attainment | 0%* | 24 | Yes → 5 categories |
| Marital Status | `MAR` | Marital status | 0% | 5 | None |
| Employment | `ESR` | Employment status | ~0%* | 6 | None |
| Worker Class | `COW` | Class of worker | ~27%** | 9 | “N/A” for non-workers |
| Occupation | `OCCP` | Occupation code | ~27%** | 300+ | Yes → 10-15 groups; “N/A” for non-workers |
| Birth Place | `POBP` | Place of birth | 0% | 149 | Yes → 2 (US/Foreign) |
| Hispanic | `HISP` | Hispanic origin | 0% | 24 | Used for race recoding |

*Empirical missingness rates above are computed among adults (AGEP ≥ 18) in the provided file; rates differ if children are included.

### 1.3 Variable Code Mappings

**Important parsing note (publication-critical):** Many ACS codes are stored as zero-padded strings in the official dictionary (e.g., `POBP` is `C,3`), but may be read as integers by default CSV parsers. For reproducibility and correct recoding, treat code variables as strings in preprocessing (with zero-padding where applicable), and only convert truly continuous measures to numeric.

#### RAC1P (Race) Recoding

| Original Code | Original Label | Recode |
|---------------|----------------|--------|
| 1 | White alone | 1 = White |
| 2 | Black or African American alone | 2 = Black |
| 3 | American Indian alone | 5 = Other |
| 4 | Alaska Native alone | 5 = Other |
| 5 | American Indian and Alaska Native | 5 = Other |
| 6 | Asian alone | 3 = Asian |
| 7 | Native Hawaiian and Pacific Islander | 5 = Other |
| 8 | Some Other Race alone | 5 = Other |
| 9 | Two or More Races | 5 = Other |

**Hispanic Ethnicity:** Use HISP > 1 to identify Hispanic (overrides RAC1P → 4 = Hispanic)

#### SCHL (Education) Recoding

| Original Codes | Original Labels | Recode |
|----------------|-----------------|--------|
| 01-15 | No schooling through Grade 12 (no diploma) | 1 = Less than HS |
| 16-17 | High school diploma or GED | 2 = High School |
| 18-20 | Some college, 1+ years, Associate's | 3 = Some College |
| 21 | Bachelor's degree | 4 = Bachelor's |
| 22-24 | Master's, Professional, Doctorate | 5 = Graduate |

#### POBP (Place of Birth) Recoding

| Original Codes | Recode |
|----------------|--------|
| 001-072 (US states + territories) | 1 = US-Born |
| 100+ (Foreign countries) | 2 = Foreign-Born |

### 1.4 Target Variable Definition

```
HIGH_INCOME = 1 if PINCP > 50000 else 0
```

**Class Distribution:**
- HIGH_INCOME = 1: 39.2% (126,197 records)
- HIGH_INCOME = 0: 60.8% (195,858 records)

This is reasonably balanced (not severely imbalanced).

### 1.5 Recommended Variable Selection

Given missingness patterns, **recommended final variable set:**

| Variable | Type | Cardinality | Include |
|----------|------|-------------|---------|
| AGEP | Continuous | - | ✓ |
| PINCP | Continuous | - | ✓ |
| SEX | Categorical | 2 | ✓ |
| RAC1P_RECODED | Categorical | 5 | ✓ (with Hispanic from HISP) |
| SCHL_RECODED | Categorical | 5 | ✓ |
| MAR | Categorical | 5 | ✓ |
| ESR | Categorical | 6 | ✓ |
| POBP_RECODED | Categorical | 2 | ✓ |
| HIGH_INCOME | Binary Target | 2 | ✓ (derived) |

**Total Variables:** 8 predictors + 1 target = **9 columns**

**Rationale for exclusions:**
- WKHP, JWMNP: often structurally “Not in Universe” (e.g., not working / not in labor force) → including them would require restricting the analytic population (e.g., workers only) or treating “N/A” as an explicit state.
- COW, OCCP: similarly “Not in Universe” for non-workers, plus additional harmonization/recoding complexity.

---

## 2. Technical Feasibility Assessment

### 2.1 Method Overview

| Method | Package (validated) | Language | GPU Required | Est. Runtime (N=35K) |
|--------|---------|----------|--------------|----------------------|
| Independent Marginals | Custom | Python | No | < 1 min |
| Synthpop | `synthpop` (CRAN; current release series 1.9-x) | R | No | 5-30 min |
| CTGAN | `sdv` (actively maintained; current release series 1.3x) | Python | Optional | 30-90 min |
| DP Bayesian network (“PrivBayes”) | `DataSynthesizer` 0.1.13 | Python | No | 5-30 min |
| GReaT | `be-great` 0.0.9 | Python | Optional (recommended) | 2-6 hours (GPU), much longer on CPU |

### 2.2 Detailed Method Specifications

#### 2.2.1 Independent Marginals (Baseline)

```
Package: None (custom implementation)
Python Version: 3.9+
Dependencies: pandas, numpy
Installation: pip install pandas numpy
Runtime Estimate: < 1 minute
Known Issues: None
Status: ✅ Ready
```

**Implementation:**
```python
def independent_marginals(df_train, n_synthetic):
    synthetic = pd.DataFrame()
    for col in df_train.columns:
        synthetic[col] = df_train[col].sample(n=n_synthetic, replace=True).values
    return synthetic
```

#### 2.2.2 Synthpop

```
Package: synthpop (CRAN; current release series 1.9-x)
Language: R 4.0+
Dependencies: R, rpy2 (Python bridge)
Installation: 
  - R: install.packages("synthpop")
  - Python: pip install rpy2
Runtime Estimate: 5-15 minutes
Known Issues: R/Python interop can be tricky
Status: ⚠️ Requires R environment setup
```

**Setup Requirements:**
1. Install R (version 4.0 or later)
2. Install synthpop package in R
3. Install rpy2 for Python-R bridge
4. Alternative: Run R script separately, save output as CSV

**Recommended Approach:** Create standalone R script, call via subprocess

#### 2.2.3 CTGAN

```
Package: sdv 1.32+
Python Version: >= 3.9
Dependencies: torch, pandas, numpy, scikit-learn
Installation: pip install sdv
Runtime Estimate: 30-60 minutes (CPU), 10-20 minutes (GPU)
Known Issues: 
  - Memory intensive for large datasets
  - Training can be unstable (use fixed seeds)
  - Version compatibility with numpy/pandas
Status: ✅ Ready (standard Python install)
```

**Key Parameters:**
```python
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df_train)

synthesizer = CTGANSynthesizer(
    metadata,
    epochs=300,          # Set per study design; SDV defaults may differ
    verbose=True,
    enable_gpu=True      # Enable if CUDA is available
)
synthesizer.fit(data=df_train)
synthetic = synthesizer.sample(num_rows=n_synthetic)

assert len(synthetic) == n_synthetic
```

#### 2.2.4 DP Bayesian Network (“PrivBayes”)

```
Package: DataSynthesizer 0.1.13
Python Version: 3.9+
Dependencies: pandas, numpy, scipy
Installation: pip install DataSynthesizer
Runtime Estimate: 5-20 minutes
Known Issues:
  - Requires numeric encoding of categoricals
  - epsilon parameter critically affects output quality
Status: ✅ Ready (standard Python install)
```

**Terminology note:** For brevity, we refer to this method as **DP Bayesian network (“PrivBayes”)** throughout. Under the hood it uses DataSynthesizer’s “correlated attribute mode”, which is closely related in spirit to PrivBayes-style DP Bayesian networks. In the manuscript, describe it precisely as “DataSynthesizer correlated-mode DP Bayesian network synthesis (PrivBayes-inspired)” unless you explicitly verify algorithmic equivalence to PrivBayes.

**Key Parameters:**
```python
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator

epsilon = 1.0  # Privacy budget (standard moderate privacy)
degree_of_bayesian_network = 2  # Max parents per node

describer = DataDescriber(category_threshold=20)
describer.describe_dataset_in_correlated_attribute_mode(
  dataset_file=df_train,
    epsilon=epsilon,
    k=degree_of_bayesian_network
)
```

#### 2.2.5 GReaT

```
Package: be-great 0.0.9
Python Version: >= 3.9
Dependencies: torch, transformers, pandas
Installation: pip install be-great
Runtime Estimate: 2-6 hours (GPU), substantially longer on CPU
Known Issues:
  - GPU strongly recommended for practical turnaround (CPU fine-tuning can be prohibitively slow)
  - Hyperparameter sensitivity; may require smaller LLM and/or guided sampling
  - Numeric formatting/precision can affect training stability (use float precision controls where appropriate)
Status: ⚠️ Feasible but highest-compute method
```

**Key Parameters:**
```python
from be_great import GReaT

model = GReaT(
    llm='distilgpt2',    # Smaller model for tractability
    epochs=50,           # Reduced from typical 100+
    batch_size=32,       # Adjust based on GPU memory
  fp16=True,            # Enable if GPU supports half precision
  report_to=[]
)
```

**⚠️ GPU Availability Check Required:**
- If GPU available: Proceed with GReaT
- If no GPU: Consider cloud compute (Google Colab, AWS) or drop method

### 2.3 Dependency Matrix

```
Core Dependencies:
├── Python 3.9-3.11
├── pandas >= 2.0
├── numpy >= 1.24
├── scikit-learn >= 1.3
├── scipy >= 1.10
└── matplotlib >= 3.7

Method-Specific:
├── sdv >= 1.32 (CTGAN)
├── DataSynthesizer == 0.1.13 (DP Bayesian network; correlated mode; “PrivBayes-inspired”)
├── be-great == 0.0.9 (GReaT)
├── torch >= 2.0 (CTGAN, GReaT)
├── transformers >= 4.30 (GReaT)
└── rpy2 >= 3.5 (Synthpop bridge)

R Requirements:
├── R >= 4.0
└── synthpop >= 1.8-0
```

### 2.4 Hardware Requirements

| Configuration | CPU | RAM | GPU | Suitable For |
|---------------|-----|-----|-----|--------------|
| Minimum | 4 cores | 16 GB | None | All except GReaT |
| Recommended | 8 cores | 32 GB | NVIDIA 8GB+ | All methods |
| Cloud Option | - | - | T4/V100 | GReaT specifically |

---

## 3. Implementation Architecture

### 3.1 Project Structure

```
TrustingSyntheticData/
├── README.md
├── requirements.txt
├── environment.yml                 # Conda environment
├── config/
│   ├── experiment_config.yaml      # All experiment parameters
│   └── variable_mappings.yaml      # Variable recoding rules
├── data/
│   ├── raw/
│   │   ├── psam_p06.csv           # Original ACS PUMS
│   │   └── PUMS_Data_Dictionary_2024.csv
│   ├── processed/
│   │   ├── train.parquet          # Training set (N=35,000)
│   │   ├── validation.parquet     # Validation set (N=7,500)
│   │   └── test.parquet           # Test set (N=7,500)
│   └── synthetic/
│       ├── independent_marginals/
│       ├── synthpop/
│       ├── ctgan/
│       ├── privbayes/
│       └── great/
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration loader
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── load_data.py           # Data loading utilities
│   │   ├── recode_variables.py    # Variable transformations
│   │   └── split_data.py          # Train/val/test splitting
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract base class
│   │   ├── independent_marginals.py
│   │   ├── synthpop_wrapper.py
│   │   ├── ctgan_wrapper.py
│   │   ├── privbayes_wrapper.py
│   │   └── great_wrapper.py
│   ├── measures/
│   │   ├── __init__.py
│   │   ├── fidelity.py            # Propensity AUC, KS, TVD
│   │   ├── privacy.py             # DCR computation
│   │   ├── utility.py             # TSTR F1/AUC
│   │   ├── fairness.py            # Subgroup gap
│   │   └── efficiency.py          # Timing utilities
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── value_functions.py     # v(x) transformations
│   │   ├── maut.py                # Weighted value aggregation
│   │   ├── sensitivity.py         # Weight sensitivity
│   │   └── voi.py                 # Value of information
│   └── visualization/
│       ├── __init__.py
│       ├── bar_charts.py          # Stacked value charts
│       ├── tornado.py             # Sensitivity diagrams
│       └── tables.py              # Result tables
├── scripts/
│   ├── 01_preprocess.py           # Run preprocessing
│   ├── 02_generate_synthetic.py   # Generate all synthetic data
│   ├── 03_compute_measures.py     # Evaluate all measures
│   ├── 04_value_analysis.py       # MAUT analysis
│   └── 05_sensitivity.py          # Sensitivity analysis
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_method_validation.ipynb
│   └── 03_results_analysis.ipynb
├── R/
│   └── synthpop_generate.R        # Standalone R script
├── results/
│   ├── raw_measures.csv
│   ├── value_scores.csv
│   ├── figures/
│   └── tables/
└── docs/
    ├── study_design_spec.md
    ├── implementation_plan.md
    └── paper_draft.md
```

### 3.2 Interface Definitions

#### Generator Base Class

```python
# src/generators/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class GenerationResult:
    synthetic_data: pd.DataFrame
    training_time: float      # seconds
    generation_time: float    # seconds
    metadata: dict            # method-specific info

class SyntheticGenerator(ABC):
    """Abstract base class for all synthetic data generators."""
    
    def __init__(self, config: dict, seed: int):
        self.config = config
        self.seed = seed
        self._set_random_state()
    
    @abstractmethod
    def _set_random_state(self) -> None:
        """Set random seeds for reproducibility."""
        pass
    
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Fit the generator on training data."""
        pass
    
    @abstractmethod
    def generate(self, n: int) -> pd.DataFrame:
        """Generate n synthetic records."""
        pass
    
    def fit_generate(self, df: pd.DataFrame, n: int) -> GenerationResult:
        """Fit and generate with timing."""
        import time
        
        start_fit = time.time()
        self.fit(df)
        fit_time = time.time() - start_fit
        
        start_gen = time.time()
        synthetic = self.generate(n)
        gen_time = time.time() - start_gen
        
        return GenerationResult(
            synthetic_data=synthetic,
            training_time=fit_time,
            generation_time=gen_time,
            metadata=self._get_metadata()
        )
    
    @abstractmethod
    def _get_metadata(self) -> dict:
        """Return method-specific metadata."""
        pass
```

#### Measure Interface

```python
# src/measures/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd

@dataclass
class MeasureResult:
    value: float
    name: str
    direction: str  # 'higher_better' or 'lower_better'
    details: dict   # Additional info (e.g., per-variable breakdown)

class Measure(ABC):
    """Abstract base class for evaluation measures."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def direction(self) -> str:
        pass
    
    @abstractmethod
    def compute(
        self,
        df_real_train: pd.DataFrame,
        df_synthetic: pd.DataFrame,
        df_real_test: pd.DataFrame,
        **kwargs
    ) -> MeasureResult:
        pass
```

### 3.3 Configuration Schema

```yaml
# config/experiment_config.yaml
data:
  source_file: "data/raw/psam_p06.csv"
  sample_size: 50000
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  filter_adults: true
  min_age: 18
  
variables:
  continuous:
    - AGEP
    - PINCP
  categorical:
    - SEX
    - RAC1P_RECODED
    - SCHL_RECODED
    - MAR
    - ESR
    - POBP_RECODED
  target: HIGH_INCOME
  target_threshold: 50000
  fairness_group: RAC1P_RECODED
  
methods:
  independent_marginals:
    enabled: true
  synthpop:
    enabled: true
    method: "cart"
    cart_minbucket: 5
  ctgan:
    enabled: true
    epochs: 300
    batch_size: 500
  privbayes:  # DataSynthesizer correlated DP Bayesian network (PrivBayes-inspired)
    enabled: true
    epsilon: 1.0
    degree: 2
  great:
    enabled: true  # Set false if no GPU
    llm: "distilgpt2"
    epochs: 50
    batch_size: 32

experiment:
  n_replicates: 5
  random_seeds: [42, 123, 456, 789, 1011]
  
evaluation:
  min_subgroup_size: 100
  propensity_classifier: "logistic_regression"
  utility_classifier: "gradient_boosting"
  
value_functions:
  fidelity:
    x_best: 0.50
    x_worst: null  # Determined empirically
    form: "linear"
  privacy:
    x_best: null   # Determined empirically
    x_worst: 0
    form: "linear"
  utility:
    x_best: 1.0
    x_worst: null  # Determined empirically
    form: "linear"
  fairness:
    x_best: 0
    x_worst: null  # Determined empirically
    form: "linear"
  efficiency:
    x_best: null   # Determined empirically
    x_worst: 480   # 8 hours
    form: "logarithmic"

weights:
  privacy_first:
    fidelity: 0.25
    privacy: 0.45
    utility: 0.15
    fairness: 0.10
    efficiency: 0.05
  utility_first:
    fidelity: 0.30
    privacy: 0.10
    utility: 0.40
    fairness: 0.10
    efficiency: 0.10
  balanced:
    fidelity: 0.25
    privacy: 0.25
    utility: 0.25
    fairness: 0.15
    efficiency: 0.10
```

### 3.4 Shared Utilities

```python
# src/utils.py

import logging
import random
import numpy as np
import pandas as pd
from pathlib import Path

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure project-wide logging."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("synth_da")

def set_all_seeds(seed: int) -> None:
    """Set random seeds for all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def save_checkpoint(data: pd.DataFrame, path: Path, name: str) -> Path:
    """Save intermediate results with timestamp."""
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / f"{name}.parquet"
    data.to_parquet(filepath, index=False)
    return filepath

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
```

---

## 4. Execution Plan

### 4.1 Phase 0: Environment Setup (Day 1)

| Task | Time | Validation |
|------|------|------------|
| Create conda environment | 30 min | `conda activate synth_da` |
| Install Python dependencies | 30 min | `pip list` shows all packages |
| Install R + synthpop | 30 min | R can load synthpop |
| Verify GPU availability | 15 min | `torch.cuda.is_available()` |
| Create project structure | 30 min | All directories exist |

**Checkpoint:** Run `python -c "import sdv; import DataSynthesizer; print('OK')"`

### 4.2 Phase 1: Data Preprocessing (Days 1-2)

| Task | Time | Deliverable |
|------|------|-------------|
| Implement data loading | 2 hrs | `src/preprocessing/load_data.py` |
| Implement variable recoding | 3 hrs | `src/preprocessing/recode_variables.py` |
| Implement train/val/test split | 1 hr | `src/preprocessing/split_data.py` |
| Create preprocessing script | 1 hr | `scripts/01_preprocess.py` |
| Run preprocessing | 30 min | `data/processed/{train,val,test}.parquet` |
| Validate outputs | 1 hr | EDA notebook confirming distributions |

**Checkpoint:** 
- Train set: 35,000 rows, 9 columns
- Test set: 7,500 rows, correct HIGH_INCOME distribution
- All recoded variables have expected cardinalities

### 4.3 Phase 2: Method Validation (Days 3-4)

| Task | Time | Validation |
|------|------|------------|
| Implement Independent Marginals | 1 hr | Generates correct schema |
| Test Synthpop on 1K sample | 2 hrs | R script runs, output matches schema |
| Test CTGAN on 1K sample | 2 hrs | Training converges, output valid |
| Test DP BN (“PrivBayes”) on 1K sample | 2 hrs | Generates under ε=1.0 |
| Test GReaT on 1K sample | 4 hrs | Fine-tuning completes |
| Document any issues | 1 hr | Issues log updated |

**Checkpoint:** All 5 methods produce 1,000 synthetic records matching input schema

### 4.4 Phase 3: Measure Implementation (Days 5-7)

| Task | Time | Deliverable |
|------|------|------------|
| Implement Propensity AUC | 2 hrs | `src/measures/fidelity.py` |
| Implement KS/TVD (secondary) | 2 hrs | Same file |
| Implement DCR 5th percentile | 3 hrs | `src/measures/privacy.py` |
| Implement TSTR F1 ratio | 2 hrs | `src/measures/utility.py` |
| Implement subgroup gap | 2 hrs | `src/measures/fairness.py` |
| Implement timing wrapper | 1 hr | `src/measures/efficiency.py` |
| Unit tests for all measures | 2 hrs | Tests pass on dummy data |

**Checkpoint:** All measures compute correctly on 1K synthetic samples

### 4.5 Phase 4: Full Experiments (Days 8-14)

| Task | Est. Time | Notes |
|------|-----------|-------|
| Independent Marginals (5 replicates) | 10 min | Trivial |
| Synthpop (5 replicates) | 1-2 hrs | R overhead |
| CTGAN (5 replicates) | 3-5 hrs | GPU recommended |
| DP BN (“PrivBayes”) (5 replicates) | 1-2 hrs | CPU only |
| GReaT (5 replicates) | 10-20 hrs | GPU strongly recommended |
| Compute all measures | 4-6 hrs | Parallelizable |

**Total Experiment Runtime:** ~20-35 hours (with GPU; expect substantially longer if GReaT runs on CPU)

**Checkpoint:** 25 synthetic datasets saved, all measures computed

### 4.6 Phase 5: Value Analysis (Days 15-18)

| Task | Time | Deliverable |
|------|------|------------|
| Determine empirical anchors | 1 hr | Update config with x_best/x_worst values |
| Implement value functions | 2 hrs | `src/analysis/value_functions.py` |
| Implement MAUT aggregation | 2 hrs | `src/analysis/maut.py` |
| Compute values for all archetypes | 1 hr | `results/value_scores.csv` |
| Generate stacked bar charts | 2 hrs | Figures 3-4 |
| Implement sensitivity analysis | 3 hrs | `src/analysis/sensitivity.py` |
| Run sensitivity analysis | 2 hrs | Tornado diagrams |

**Decision Analysis computation details (aligns with study design spec):**

1. **Anchor derivation (benchmark-relative):**
  - Compute each primary measure per replicate.
  - Derive anchors using *method-level means* (to avoid anchors being set by a single outlier replicate):
    - Fidelity (propensity AUC): $x_{best}=0.50$; $x_{worst}=$ Independent Marginals mean AUC.
    - Utility (TSTR F1 ratio): $x_{best}=1.0$; $x_{worst}=$ Independent Marginals mean ratio.
    - Fairness (max subgroup utility gap): $x_{best}=0$; $x_{worst}=$ Independent Marginals mean gap.
    - Privacy (DCR 5th percentile): $x_{best}=$ max method mean DCR; $x_{worst}=0$.
    - Efficiency (total time): $x_{best}=$ min method mean time; $x_{worst}=480$ minutes.
  - For fidelity/utility/fairness, exclude Independent Marginals from the candidate set when computing any empirical “best”.

2. **Expected value over stochastic replicates:**
  - Apply single-attribute value functions to each replicate’s raw measures, then aggregate (mean ± SD) at the value level.
  - This avoids conflating $v(\mathbb{E}[X])$ with $\mathbb{E}[v(X)]$ when value functions are nonlinear (notably efficiency).

3. **MAUT aggregation and assumptions (paper-facing):**
  - Use an additive model $V=\sum_i w_i v_i(x_i)$.
  - Explicitly state preferential independence and additive-form assumptions (and any threshold-based robustness checks) in the manuscript.

4. **Sensitivity analyses to report (minimum set):**
  - Single-weight sensitivity: vary each weight ±0.15 with proportional renormalization.
  - Equal-weights reference case ($w_i=0.20$).
  - Dominance analysis: identify weight regions where a method remains optimal.
  - Value-function shape robustness: replace linear (non-efficiency) value functions with a simple piecewise-linear diminishing-returns alternative and re-check rank stability.

**Checkpoint:** Complete value scores for all methods × archetypes

### 4.7 Phase 6: VOI Analysis & Documentation (Days 19-21)

| Task | Time | Deliverable |
|------|------|------------|
| Implement VOI framework | 4 hrs | `src/analysis/voi.py` |
| Derive decision rules | 2 hrs | Documented heuristics |
| Generate all figures | 3 hrs | `results/figures/` |
| Generate all tables | 2 hrs | `results/tables/` |
| Code documentation | 2 hrs | Docstrings, README |

### 4.8 Timeline Summary

```
Week 1: Setup + Preprocessing + Method Validation
  Mon: Environment setup, project structure
  Tue: Data preprocessing
  Wed: Method validation (IndepMarg, Synthpop)
  Thu: Method validation (CTGAN, DP BN “PrivBayes”)
  Fri: Method validation (GReaT), measure implementation starts

Week 2: Measures + Full Experiments
  Mon: Complete measure implementation
  Tue: Unit tests, debugging
  Wed: Begin full experiments (IndepMarg, Synthpop)
  Thu: Full experiments (CTGAN, DP BN “PrivBayes”)
  Fri: Full experiments (GReaT - may extend to weekend)

Week 3: Analysis + Visualization
  Mon: Measure computation for all datasets
  Tue: Value function implementation
  Wed: MAUT analysis, stacked bar charts
  Thu: Sensitivity analysis
  Fri: VOI analysis

Week 4: Documentation + Paper
  Mon: Generate final figures/tables
  Tue: Code cleanup, documentation
  Wed-Fri: Paper writing (results section)
```

---

## 5. Risk Assessment

**Scope & threats to external validity (paper-facing):** Results are conditional on the chosen ACS PUMS extract (California, adults; specific variables and preprocessing rules), the selected synthesis hyperparameter ranges (including ε for DP methods), and the evaluation design (replicate-based Monte Carlo with the specified classifiers and privacy proxy). The contribution is comparative performance under a transparent, pre-registered-like protocol; generalization to other states, years, variable sets, or downstream tasks should be treated as an empirical question rather than assumed.

### 5.1 Technical Risks

| Risk | Probability | Impact | Mitigation | Fallback |
|------|-------------|--------|------------|----------|
| **GReaT fails (no GPU)** | Medium | Low | Use cloud compute (Colab) | Drop GReaT; 4 methods still sufficient |
| **Synthpop R/Python bridge issues** | Medium | Medium | Use standalone R script | Run R separately, save CSV |
| **CTGAN training unstable** | Low | Low | Use multiple seeds, early stopping | Increase epochs, report variability |
| **DP BN (“PrivBayes”) poor quality at ε=1** | Low | Medium | Test multiple ε values | Document as finding (privacy cost) |
| **DCR computation slow** | Medium | Medium | Sample-based approximation | Use 10K subset for DCR |
| **Memory issues with N=35K** | Low | High | Process in batches | Reduce to N=25K |

### 5.2 Data Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Insufficient subgroup sizes** | Low | Medium | Pre-check: smallest subgroup (Black) has ~14K adults |
| **Class imbalance in HIGH_INCOME** | Low | Low | 40/60 split is reasonable; no resampling needed |
| **Structural “N/A” handling inconsistent** | Medium | Medium | Treat “N/A” as explicit state (indicators + harmonized preprocessing across methods) |

### 5.3 Timeline Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **GReaT takes longer than expected** | High | Medium | Start GReaT early; run overnight |
| **Debugging consumes time** | Medium | Medium | Build incrementally; test each component |
| **Results don't show tradeoffs** | Low | Medium | This is a finding; reframe contribution |

### 5.4 Priority for Early Validation

**Build and test in this order to fail fast:**

1. **Data preprocessing** - Validates data availability
2. **Independent Marginals** - Simplest method, validates pipeline
3. **Propensity AUC measure** - Core fidelity metric
4. **CTGAN** - Most used deep learning method
5. **GReaT** - Highest risk; validate GPU early
6. **Other measures** - Build after core pipeline works
7. **Synthpop, DP BN (“PrivBayes”)** - Lower risk; can run in parallel

---

## 6. Validation Checkpoints

### 6.1 Checkpoint 1: Data Ready (End of Day 2)

```
□ Preprocessed data saved as parquet files
□ Train: 35,000 rows, 9 columns
□ Test: 7,500 rows, same columns
□ HIGH_INCOME distribution: ~40% positive
□ RAC1P_RECODED has 5 categories
□ All values in expected ranges
```

### 6.2 Checkpoint 2: Methods Validated (End of Day 4)

```
□ All 5 methods run on 1K sample without error
□ Output schema matches input (same columns, types)
□ GReaT GPU status confirmed
□ Synthpop R bridge working (or standalone script ready)
□ Timing estimates updated based on 1K results
```

### 6.3 Checkpoint 3: Measures Working (End of Day 7)

```
□ Propensity AUC returns value in [0.5, 1.0]
□ DCR 5th percentile returns positive value
□ TSTR F1 ratio returns value in [0, 1]
□ Max subgroup gap returns value ≥ 0
□ All measures produce reasonable values on test data
```

### 6.4 Checkpoint 4: Experiments Complete (End of Day 14)

```
□ 25 synthetic datasets generated (5 methods × 5 replicates)
□ All datasets saved with consistent naming
□ Raw measures computed for all datasets
□ results/raw_measures.csv exists with 25 rows
□ No missing values in measure results
```

### 6.5 Checkpoint 5: Analysis Complete (End of Day 18)

```
□ Value functions applied correctly (values in [0, 1])
□ Overall values computed for all archetypes
□ Rankings differ across archetypes (validating the contribution)
□ Sensitivity analysis identifies threshold weights
□ All figures generated
```

### 6.6 Publication-Quality Reporting Checklist

Use this as a “Definition of Done” for a Decision Analysis–quality methods section and reproducibility supplement (clear assumptions, auditable computations, and decision-relevant interpretation):

1. **Data provenance & universe restrictions**
  - State the exact PUMS extract (year, geography, file name), and the final analytic population (e.g., adults AGEP ≥ 18).
  - Explicitly distinguish true missingness from ACS “Not in Universe” codes (WKHP/JWMNP/COW/OCCP).
  - Document all recodes (especially race/HISP override) and any grouping decisions (OCCP bins).

2. **Reproducibility & compute disclosure**
  - Pin package versions (lockfile preferred) and report OS/Python version.
  - Record seeds for every replicate and method.
  - Report hardware used (CPU model, RAM, GPU model/VRAM if applicable) and wall-clock runtime per method.

3. **Method specification**
  - Report all hyperparameters used (CTGAN epochs/batch size, DP BN epsilon/k, GReaT llm/epochs/batch size/fp16).
  - If the DP BN method is referred to as “PrivBayes”, describe it precisely as “DataSynthesizer correlated-mode DP Bayesian network synthesis (PrivBayes-inspired)” unless you validate algorithmic equivalence.

4. **Metric definitions & uncertainty**
  - Define each metric unambiguously (what model is used for propensity, how DCR is computed, how subgroup gaps are defined).
  - Report variability across replicates (mean ± SD and range). Emphasize whether observed differences exceed replicate variability.

5. **Decision analysis transparency**
  - Provide value function forms/anchors and archetype weights, and state explicitly that archetypes are illustrative preference profiles (not elicited from a specific stakeholder).
  - State modeling assumptions explicitly (preferential independence; additive aggregation; value-function curvature choices).
  - Report sensitivity analyses (weight perturbations; dominance regions; alternate value-function shapes; optional worker-only vs full-adult population).

### 6.7 Key References (Methods & Software)

- **Sequential synthesis / synthpop**: Nowok, B., Raab, G. M., & Dibben, C. (2016). *synthpop: Bespoke Creation of Synthetic Data in R*. *Journal of Statistical Software*, 74(11). https://doi.org/10.18637/jss.v074.i11
- **CTGAN algorithm**: Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). *Modeling Tabular data using Conditional GAN*. NeurIPS 2019. https://arxiv.org/abs/1907.00503
- **SDV software**: Patki, N., Wedge, R., & Veeramachaneni, K. (2016). *The Synthetic Data Vault*. IEEE DSAA 2016. (Citation provided in SDV project materials)
- **GReaT**: Borisov, V., Sessler, K., Leemann, T., Pawelczyk, M., & Kasneci, G. (2023). *Language Models are Realistic Tabular Data Generators*. ICLR 2023. https://openreview.net/forum?id=cEygmQNOeI
- **DataSynthesizer**: *DataSynthesizer: Privacy-Preserving Synthetic Datasets* (project paper linked from the DataSynthesizer distribution). Also note documented assumptions and correlated-attribute-mode demo notebooks.

**Licensing note (reproducibility):** SDV is distributed under the Business Source License per its project documentation; ensure you can legally redistribute code/artifacts as needed for your intended publication and supplementary materials.

---

## 7. Appendices

### Appendix A: Requirements Files

**requirements.txt:**
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
scipy>=1.10
matplotlib>=3.7
seaborn>=0.12
pyyaml>=6.0
pyarrow>=12.0
tqdm>=4.65
sdv>=1.32
DataSynthesizer==0.1.13
be-great==0.0.9
torch>=2.0
transformers>=4.30
rpy2>=3.5
jupyter>=1.0
```

**environment.yml:**
```yaml
name: synth_da
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pandas>=2.0
  - numpy>=1.24
  - scikit-learn>=1.3
  - scipy>=1.10
  - matplotlib>=3.7
  - seaborn>=0.12
  - pyyaml>=6.0
  - pyarrow>=12.0
  - tqdm>=4.65
  - pytorch>=2.0
  - transformers>=4.30
  - jupyter
  - r-base>=4.0
  - rpy2>=3.5
  - pip
  - pip:
    - sdv>=1.32
    - DataSynthesizer==0.1.13
    - be-great==0.0.9
```

### Appendix B: Quick Start Commands

```bash
# Setup
conda env create -f environment.yml
conda activate synth_da

# Install R package (run in R console)
R -e "install.packages('synthpop', repos='https://cloud.r-project.org')"

# Run pipeline
python scripts/01_preprocess.py
python scripts/02_generate_synthetic.py --method all --replicates 5
python scripts/03_compute_measures.py
python scripts/04_value_analysis.py
python scripts/05_sensitivity.py
```

### Appendix C: Variable Recoding Reference

| Variable | Original Values | Recoded Values |
|----------|-----------------|----------------|
| RAC1P | 1-9 | 1=White, 2=Black, 3=Asian, 4=Hispanic*, 5=Other |
| SCHL | 01-24 | 1=<HS, 2=HS, 3=Some College, 4=Bachelor's, 5=Graduate |
| POBP | 001-554 | 1=US-Born (001-072), 2=Foreign-Born (100+) |

*Hispanic determined by HISP > 1, overrides RAC1P

### Appendix D: Key Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Keep universe-restricted vars (WKHP/JWMNP/COW/OCCP) | Yes | Encode “N/A” explicitly; preserves realism of work/commute structure |
| Use HISP for Hispanic | Yes | More accurate than RAC1P=8 (Some Other) |
| Binary POBP recode | Yes | Simplifies interpretation |
| GReaT epochs = 50 | Yes | Balance quality vs. runtime |
| DP BN (“PrivBayes”) ε = 1.0 | Yes | Starting point; test sensitivity across ε |
| Min subgroup n = 100 | Yes | Statistical stability |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-05 | Initial implementation plan |

---

**Next Step:** Review this plan and confirm:
1. Variable handling strategy for “Not in Universe” codes (keep + explicit encoding vs worker-only vs drop as robustness)
2. GReaT inclusion (depends on GPU availability)
3. Timeline feasibility for your schedule
