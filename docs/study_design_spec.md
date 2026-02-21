# Empirical Study Design Specification

## A Multiattribute Decision Analysis Framework for Synthetic Data Generation Method Selection

**Authors:** Michael Koo & Alfonso Berumen
**Document Version:** 1.0
**Date:** December 31, 2024
**Purpose:** Formalized decisions for empirical study design

---

## 1. Contribution Statement

### 1.1 Paper Type
**Applied Decision Analysis Paper**

This paper demonstrates that multiattribute value modeling produces actionable insights for synthetic data generation method selection that existing benchmarking studies cannot provide.

### 1.2 Core Contribution Statement

> We develop the first multiattribute decision analysis framework for synthetic data generation method selection, integrating value-focused objectives hierarchy, single-dimensional value functions, stakeholder archetypes, and value of information analysis. Through proof-of-concept application to American Community Survey Public Use Microdata (N=50,000), we demonstrate that optimal method choice is preference-dependent, with no method dominating across fidelity, privacy, utility, fairness, and efficiency objectives. The framework provides practitioners systematic, auditable guidance for method selection tailored to organizational priorities. While our empirical findings are dataset-specific, the framework is domain-general and provides a reusable template for organizations to systematically evaluate methods in their own context.

### 1.3 Supporting Contributions

1. **Framework Development**: First comprehensive decision analysis framework for synthetic data method selection (VF-SDSF: Value-Focused Synthetic Data Selection Framework), integrating objectives hierarchy, value functions, decision-focused transformation, and value of information analysis. The framework is domain-general and can be applied to any tabular data context.

2. **Empirical Demonstration**: Proof-of-concept application using California ACS PUMS data (N=50,000) demonstrating that the framework is implementable and reveals preference-dependent method rankings. This shows the framework produces actionable insights not apparent from raw metric inspection.

3. **Practical Decision Rules**: Value of information analysis identifying when comprehensive benchmarking is warranted versus when simple heuristics suffice, providing guidance for resource allocation in method evaluation.

---

## 2. Dataset Specification

### 2.1 Primary Dataset

**American Community Survey (ACS) Public Use Microdata Sample (PUMS)**

| Attribute | Specification |
|-----------|---------------|
| Source | U.S. Census Bureau, 2024 1-Year PUMS |
| Geographic Scope | Single state (California used in the empirical application; STATE=06) |
| Target Sample Size | 50,000 records (randomly sampled if source larger) |
| Download Source | https://www.census.gov/programs-surveys/acs/microdata.html |

### 2.2 Variable Selection

Select variables to ensure mixed types, demographic coverage, and meaningful prediction tasks.

**Primary analysis variable set (FINALIZED):** Use 8 predictor variables that avoid ambiguity around ACS "Not in Universe" codes (structural missingness for non-workers/non-commuters). Universe-restricted variables (WKHP, JWMNP, COW, OCCP) are excluded for simplicity, cleaner interpretation, and methodological defensibility.

**Continuous Variables (2):**
| Variable | ACS Code | Description |
|----------|----------|-------------|
| AGEP | AGEP | Age |
| PINCP | PINCP | Total person's income |

**Categorical Variables (6):**
| Variable | ACS Code | Description | Cardinality |
|----------|----------|-------------|-------------|
| SEX | SEX | Sex | 2 |
| RAC1P_RECODED | RAC1P (+ HISP) | Race (recoded with Hispanic override) | 5 |
| SCHL_RECODED | SCHL | Educational attainment (recoded) | 5 |
| MAR | MAR | Marital status | 5 |
| ESR | ESR | Employment status | 6 |
| POBP_RECODED | POBP | Place of birth (recoded: US-born vs. foreign-born) | 2 |

**Total Variables:** 8 predictors + 1 target = 9 columns

**Robustness extension (optional, if time permits):** Re-run analysis with universe-restricted variables (WKHP, JWMNP, COW, OCCP) to test ranking stability across variable sets. Report whether method rankings remain stable.

**Target Variable for Utility Task:**
| Variable | Definition | Type |
|----------|------------|------|
| HIGH_INCOME | PINCP > $50,000 | Binary classification |

### 2.3 Data Preprocessing

```
1. Filter to adults (AGEP >= 18)
2. Filter to records with non-missing income (PINCP not null)
3. Recode categoricals:
    - SCHL: Collapse to 5 levels (< HS, HS, Some college, Bachelor's, Graduate)
    - Race: Collapse RAC1P to {White, Black, Asian, Other} and set Hispanic using HISP > 1
    - POBP: Collapse to 2 levels (US-born vs. foreign-born)
4. Handle missing/universe restrictions:
    - For the primary analysis variable set above, missingness should be minimal among adults; document any exclusions.
    - For robustness analyses that include universe-restricted variables (e.g., WKHP/JWMNP), either encode “Not in Universe” explicitly or restrict the analytic population and report the resulting N.
5. Create binary target: HIGH_INCOME = 1 if PINCP > 50000, else 0
6. Random sample to N = 50,000 if source larger
```

### 2.4 Data Partitioning

| Partition | Proportion | Size (N=35,000) | Purpose |
|-----------|------------|-----------------|---------|
| Training | 70% | 24,500 | Train generative models |
| Test | 30% | 10,500 | TSTR evaluation (never seen by generators) |

**Note on Sample Size (Updated):** Due to data availability constraints after filtering to adults with valid income, the final sample size is N=35,000 (rather than the originally planned N=50,000). This provides 24,500 training records and 10,500 test records. The 70/30 split (rather than the originally planned 70/15/15) was adopted because hyperparameter tuning was not performed systematically; instead, default or literature-recommended parameters were used for all methods.

**Critical:** Test set is held out entirely. Synthetic data is generated from training set only. TSTR models are trained on synthetic data and evaluated on real test set.

### 2.5 Secondary Dataset (Optional Extension)

If time and resources permit, add one healthcare dataset to demonstrate context-dependence. Candidate:

| Dataset | Source | Access |
|---------|--------|--------|
| UCI Heart Disease | UCI ML Repository | Public, no restrictions |
| Diabetes 130-US Hospitals | UCI ML Repository | Public, no restrictions |

**Decision:** Begin with ACS PUMS only. Add secondary dataset only after primary analysis complete.

---

## 3. Synthetic Data Generation Methods

### 3.1 Selected Methods (5)

| Method | Type | Purpose in Study | Implementation |
|--------|------|------------------|----------------|
| **Independent Marginals** | Trivial baseline | Floor for comparison; defines x_worst | Custom (sample each column independently) |
| **Synthpop** | Statistical | Established practitioner standard | R package `synthpop` |
| **CTGAN** | Deep learning (GAN) | Most common deep generative approach | Python `sdv` library |
| **DP Bayesian network (PrivBayes-inspired)** | Differentially private | Formal privacy guarantees (under stated assumptions) | Python `DataSynthesizer` (correlated mode; PrivBayes-inspired) |
| **GReaT** | Language model | Emerging LLM-based approach | Python `be-great` library |

### 3.2 Method Specifications

#### 3.2.1 Independent Marginals (Baseline)

**Description:** Sample each column independently from its empirical marginal distribution. Preserves marginals perfectly but destroys all correlations.

**Implementation:**
```python
def independent_marginals(df_train, n_synthetic):
    synthetic = pd.DataFrame()
    for col in df_train.columns:
        synthetic[col] = df_train[col].sample(n=n_synthetic, replace=True).values
    return synthetic
```

**Hyperparameters:** None.

#### 3.2.2 Synthpop

**Description:** Sequential CART-based synthesis. Variables synthesized one at a time, conditional on previously synthesized variables, using classification/regression trees.

**Implementation:** R package `synthpop` (CRAN; current release series 1.9-x).

**Hyperparameters:** Use defaults.
```r
syn(df_train, method = "cart", cart.minbucket = 5)
```

**Note:** Requires R environment. Can call from Python via `rpy2` or run separately.

#### 3.2.3 CTGAN

**Description:** Conditional GAN for tabular data with mode-specific normalization and training-by-sampling.

**Implementation:** Python `sdv` library (Synthetic Data Vault), current 1.3x release series (pin exact version in environment files).

**Hyperparameters:** Use defaults.
```python
from sdv.single_table import CTGANSynthesizer
synthesizer = CTGANSynthesizer(metadata, epochs=300)
synthesizer.fit(df_train)
synthetic = synthesizer.sample(num_rows=n_synthetic)
```

#### 3.2.4 DP Bayesian Network (“PrivBayes”)

**Description:** Differentially private Bayesian-network-based synthesis using DataSynthesizer’s correlated-attribute mode (PrivBayes-inspired). Provides formal ε-differential privacy guarantees under documented assumptions.

**Implementation:** Python `DataSynthesizer` library (pin 0.1.13).

**Hyperparameters:**
```python
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator

epsilon = 1.0  # Privacy budget — document this choice
degree_of_bayesian_network = 2

describer = DataDescriber()
describer.describe_dataset_in_correlated_attribute_mode(
    df_train, 
    epsilon=epsilon,
    k=degree_of_bayesian_network
)
generator = DataGenerator()
generator.generate_dataset_in_correlated_attribute_mode(n_synthetic, describer)
```

**Implementation note:** DataSynthesizer’s public examples use notebook workflows and may require file-based inputs depending on version. For the empirical study, we will follow the project’s provided demos/notebooks and standardize I/O via intermediate CSV/parquet exports.

**Privacy Parameter:** ε = 1.0 (standard "moderate privacy" setting). Document in paper.

#### 3.2.5 GReaT

**Description:** Fine-tunes GPT-2 on textual representations of tabular rows, generates new rows via text completion.

**Implementation:** Python `be-great` library (pin 0.0.9; Python >= 3.9).

**Hyperparameters:** Use defaults, but limit epochs for tractability.
```python
from be_great import GReaT
model = GReaT(llm='distilgpt2', epochs=50, batch_size=32)
model.fit(df_train)
synthetic = model.sample(n_samples=n_synthetic)
```

**Note:** GPU strongly recommended for tractable runtimes. If unavailable, consider dropping this method or using cloud compute.

### 3.3 Pre-Experiment Method Validation

**Required before proceeding:**

For each method, confirm:
- [ ] Can install package/dependencies
- [ ] Runs successfully on 1,000-row subset of ACS PUMS
- [ ] Completes in < 1 hour on test subset
- [ ] Output has correct schema (same columns, appropriate types)

Document any failures or modifications required.

### 3.4 Synthetic Data Generation Protocol

For each method:
1. Train/fit on training partition (N = 35,000)
2. Generate synthetic dataset of size N = 35,000 (matching training size)
3. Repeat 5 times (5 independent replicates per method)
4. Record training time and generation time for each replicate
5. Save all synthetic datasets for measure computation

**Total synthetic datasets:** 5 methods × 5 replicates = 25 datasets

---

## 4. Value Measures

### 4.1 Measure Summary

| Objective | Primary Measure | Scale Direction | Secondary Measures |
|-----------|-----------------|-----------------|-------------------|
| Fidelity | Propensity Score AUC | Lower is better (closer to 0.5) | Avg KS (continuous), Avg TVD (categorical) |
| Privacy | Membership Inference Attack Success Rate | Lower is better (closer to 0.5) | DCR 5th Percentile (Appendix) |
| Utility | TSTR F1 Ratio | Higher is better (closer to 1.0) | TSTR AUC Ratio |
| Fairness | Max Subgroup Utility Gap | Lower is better (closer to 0) | — |
| Efficiency | Total Time (minutes) | Lower is better | — |

### 4.2 Measure Specifications

#### 4.2.1 Fidelity: Propensity Score AUC

**Definition:** Train a classifier to distinguish real from synthetic records. AUC near 0.5 indicates synthetic data is indistinguishable from real.

**Computation:**
```python
def propensity_auc(df_real, df_synthetic):
    # Label real as 1, synthetic as 0
    df_real['label'] = 1
    df_synthetic['label'] = 0
    combined = pd.concat([df_real, df_synthetic])
    
    X = combined.drop('label', axis=1)
    y = combined['label']
    
    # Encode categoricals
    X_encoded = pd.get_dummies(X)
    
    # Train-test split for classifier evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3)
    
    # Train classifier (logistic regression for simplicity)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Compute AUC
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    
    return auc  # Best = 0.5, Worst = 1.0
```

**Interpretation:**
- AUC = 0.5: Classifier cannot distinguish → perfect fidelity
- AUC = 1.0: Classifier perfectly distinguishes → zero fidelity

**Secondary Measures:**

*Average KS Statistic (continuous variables):*
```python
def avg_ks(df_real, df_synthetic, continuous_cols):
    ks_stats = []
    for col in continuous_cols:
        stat, _ = ks_2samp(df_real[col], df_synthetic[col])
        ks_stats.append(stat)
    return np.mean(ks_stats)  # Best = 0, Worst = 1
```

*Average TVD (categorical variables):*
```python
def avg_tvd(df_real, df_synthetic, categorical_cols):
    tvd_stats = []
    for col in categorical_cols:
        p_real = df_real[col].value_counts(normalize=True)
        p_syn = df_synthetic[col].value_counts(normalize=True)
        # Align indices
        all_cats = set(p_real.index) | set(p_syn.index)
        tvd = 0.5 * sum(abs(p_real.get(c, 0) - p_syn.get(c, 0)) for c in all_cats)
        tvd_stats.append(tvd)
    return np.mean(tvd_stats)  # Best = 0, Worst = 1
```

#### 4.2.2 Privacy: DCR 5th Percentile (PRIMARY)

**Definition:** Distance from each synthetic record to its nearest neighbor in training data.
Lower DCR = higher memorization risk.

**Computation:** For each synthetic record, compute Euclidean distance (after standardization)
to all training records. Report 5th percentile of minimum distances.

**Why DCR Instead of Membership Inference:**
Validation pilot (N=5K) showed membership inference attack achieved only AUC=0.540 on
real data (below 0.6 threshold for validity). An attack classifier that cannot distinguish
training members from non-members in real data cannot meaningfully evaluate synthetic
data privacy. DCR provides robust distance-based privacy assessment.

**Interpretation:**
- High DCR (>1.0): Synthetic records far from training data (good privacy)
- Medium DCR (0.3-1.0): Some records close to training but not copies
- Low DCR (<0.3): Many records suspiciously close to training (poor privacy)

**Secondary Measure (Appendix):**
Report membership inference for completeness but flag as "attack too weak for primary analysis."

#### 4.2.3 Utility: TSTR F1 Ratio

**Definition:** Train a classifier on synthetic data, evaluate on real test set. Compare F1 score to classifier trained on real training data.

**Computation:**
```python
def tstr_f1_ratio(df_real_train, df_synthetic, df_real_test, target_col):
    X_real_train = df_real_train.drop(target_col, axis=1)
    y_real_train = df_real_train[target_col]
    
    X_synthetic = df_synthetic.drop(target_col, axis=1)
    y_synthetic = df_synthetic[target_col]
    
    X_test = df_real_test.drop(target_col, axis=1)
    y_test = df_real_test[target_col]
    
    # Encode features
    encoder = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), continuous_cols)
    ])
    
    X_real_train_enc = encoder.fit_transform(X_real_train)
    X_synthetic_enc = encoder.transform(X_synthetic)
    X_test_enc = encoder.transform(X_test)
    
    # Train on real
    clf_real = GradientBoostingClassifier(n_estimators=100)
    clf_real.fit(X_real_train_enc, y_real_train)
    f1_real = f1_score(y_test, clf_real.predict(X_test_enc))
    
    # Train on synthetic
    clf_syn = GradientBoostingClassifier(n_estimators=100)
    clf_syn.fit(X_synthetic_enc, y_synthetic)
    f1_syn = f1_score(y_test, clf_syn.predict(X_test_enc))
    
    # Ratio
    return f1_syn / f1_real  # Best = 1.0, Worst = 0
```

**Target Variable:** HIGH_INCOME (binary: income > $50,000)

**Secondary Measure:** TSTR AUC Ratio (same computation, replace F1 with AUC)

#### 4.2.4 Fairness: Maximum Subgroup Utility Gap

**Definition:** Compute TSTR F1 ratio separately for each demographic subgroup. Report the maximum absolute difference between any subgroup's ratio and the overall ratio.

**Computation:**
```python
def max_subgroup_utility_gap(df_real_train, df_synthetic, df_real_test, target_col, group_col):
    # Compute overall TSTR F1 ratio
    overall_ratio = tstr_f1_ratio(df_real_train, df_synthetic, df_real_test, target_col)
    
    # Compute per-subgroup ratios
    subgroup_ratios = {}
    for group in df_real_test[group_col].unique():
        test_subset = df_real_test[df_real_test[group_col] == group]
        if len(test_subset) >= 100:  # Minimum subgroup size
            # Note: Still train on full data, but evaluate on subgroup
            ratio = tstr_f1_ratio_on_subset(df_real_train, df_synthetic, test_subset, target_col)
            subgroup_ratios[group] = ratio
    
    # Max gap
    gaps = [abs(r - overall_ratio) for r in subgroup_ratios.values()]
    return max(gaps)  # Best = 0, Worst = large positive
```

**Subgroup Definition:** RAC1P (race), collapsed to 5 categories.

**Minimum Subgroup Size:** n ≥ 100 in test set for inclusion.

#### 4.2.5 Efficiency: Total Time (Minutes)

**Definition:** Wall-clock time for training the generative model plus generating synthetic dataset.

**Computation:**
```python
import time

def measure_time(fit_func, sample_func, df_train, n_synthetic):
    start_train = time.time()
    model = fit_func(df_train)
    end_train = time.time()
    
    start_gen = time.time()
    synthetic = sample_func(model, n_synthetic)
    end_gen = time.time()
    
    total_minutes = (end_train - start_train + end_gen - start_gen) / 60
    return total_minutes  # Lower is better
```

**Measurement Conditions:**
- Document hardware specifications (CPU, RAM, GPU if used)
- Run on consistent hardware across all methods
- If method requires GPU and others don't, note this limitation

---

## 5. Value Functions

### 5.1 Anchor Point Derivation Approach

**Method:** Benchmark-relative anchoring

- **x_best:** Best performance observed across all methods (excluding independent marginals for fidelity/utility/fairness)
- **x_worst:** Performance of Independent Marginals baseline (for fidelity, utility, fairness) OR reasonable worst-case (for privacy, efficiency)

This approach:
- Avoids arbitrary threshold selection
- Ensures the value scale spans the actual observed range
- Uses the trivial baseline to define "floor" performance

### 5.2 Value Function Specifications

#### 5.2.1 Fidelity (Propensity AUC)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| x_best | 0.50 | Theoretical optimum (indistinguishable) |
| x_worst | Observed AUC of Independent Marginals | Empirical floor |
| Functional Form | Linear | No strong preference for nonlinearity |

**Value Function:**
```
v_fidelity(x) = (x_worst - x) / (x_worst - 0.50)

where x = observed propensity AUC
```

**Note:** If Independent Marginals achieves AUC = 0.85, then:
- AUC = 0.50 → v = 1.0
- AUC = 0.85 → v = 0.0
- AUC = 0.675 → v = 0.5

#### 5.2.2 Privacy (Membership Inference Attack Success Rate)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| x_best | 0.50 | Theoretical optimum (random guessing - cannot distinguish members) |
| x_worst | Observed maximum attack success rate across methods | Empirical worst case |
| Functional Form | Linear | No strong preference for nonlinearity |

**Value Function:**
```
v_privacy(x) = (x_worst - x) / (x_worst - 0.50)

where x = observed membership inference attack success rate
```

**Note:** Lower attack success rate is better (closer to 0.50 = random guessing).
- Attack rate = 0.50 → v = 1.0 (perfect privacy)
- Attack rate = x_worst → v = 0.0 (worst observed)
- Attack rate = midpoint → v = 0.5

#### 5.2.3 Utility (TSTR F1 Ratio)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| x_best | 1.0 | Theoretical optimum (parity with real data) |
| x_worst | Observed ratio of Independent Marginals | Empirical floor |
| Functional Form | Linear | No strong preference for nonlinearity |

**Value Function:**
```
v_utility(x) = (x - x_worst) / (1.0 - x_worst)

where x = observed TSTR F1 ratio
```

**Note:** If Independent Marginals achieves ratio = 0.40, then:
- Ratio = 1.0 → v = 1.0
- Ratio = 0.40 → v = 0.0
- Ratio = 0.70 → v = 0.5

#### 5.2.4 Fairness (Max Subgroup Utility Gap)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| x_best | 0 | Theoretical optimum (no gap) |
| x_worst | Observed gap of Independent Marginals | Empirical floor |
| Functional Form | Linear | No strong preference for nonlinearity |

**Value Function:**
```
v_fairness(x) = (x_worst - x) / x_worst

where x = observed max subgroup utility gap
```

#### 5.2.5 Efficiency (Total Time)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| x_best | Observed min time across methods | Empirical ceiling |
| x_worst | 480 minutes (8 hours) | Practical limit for iteration |
| Functional Form | Logarithmic | Diminishing sensitivity at longer durations |

**Value Function:**
```
v_efficiency(x) = 1 - log(x / x_best) / log(480 / x_best)

where x = observed total time in minutes
```

**Note:** Logarithmic form means:
- Going from 1 min to 10 min is more consequential than going from 100 min to 110 min
- Capped at 480 minutes (if method exceeds, v = 0)

### 5.3 Value Function Summary Table

| Objective | x_best | x_worst | Form | Direction |
|-----------|--------|---------|------|-----------|
| Fidelity | 0.50 | Empirical (baseline) | Linear | Lower raw → Higher value |
| Privacy | 0.50 (random guessing) | Empirical (max attack rate) | Linear | Lower raw → Higher value |
| Utility | 1.0 | Empirical (baseline) | Linear | Higher raw → Higher value |
| Fairness | 0 | Empirical (baseline) | Linear | Lower raw → Higher value |
| Efficiency | Empirical (min) | Empirical (max) | Logarithmic | Lower raw → Higher value |

---

## 6. Swing Weights

### 6.1 Approach: Stakeholder Archetypes

Rather than eliciting weights from a specific individual (limiting generalizability) or deriving from literature patterns (methodologically unsound), we define three stakeholder archetypes representing distinct but plausible preference profiles.

### 6.2 Archetype Definitions

#### Archetype 1: Privacy-First Practitioner

**Profile:** Works in healthcare, finance, or government where regulatory compliance and individual protection are paramount. Willing to sacrifice some fidelity and utility for strong privacy guarantees.

| Objective | Weight | Rationale |
|-----------|--------|-----------|
| Privacy | 0.45 | Primary concern |
| Fidelity | 0.25 | Must preserve key patterns |
| Utility | 0.15 | Secondary to privacy |
| Fairness | 0.10 | Important but not primary |
| Efficiency | 0.05 | Will invest time for privacy |
| **Total** | **1.00** | |

#### Archetype 2: Utility-First Practitioner

**Profile:** Machine learning engineer focused on model training. Needs synthetic data that supports equivalent predictive performance. Privacy concerns are secondary (perhaps already addressed through access controls).

| Objective | Weight | Rationale |
|-----------|--------|-----------|
| Utility | 0.40 | Primary concern |
| Fidelity | 0.30 | Supports utility |
| Privacy | 0.10 | Secondary concern |
| Fairness | 0.10 | Model fairness matters |
| Efficiency | 0.10 | Values fast iteration |
| **Total** | **1.00** | |

#### Archetype 3: Balanced Practitioner

**Profile:** General-purpose data scientist seeking reasonable performance across all dimensions. No single objective dominates. Represents "typical" user without extreme preferences.

| Objective | Weight | Rationale |
|-----------|--------|-----------|
| Fidelity | 0.25 | Foundational requirement |
| Privacy | 0.25 | Important safeguard |
| Utility | 0.25 | Must support tasks |
| Fairness | 0.15 | Emerging priority |
| Efficiency | 0.10 | Practical constraint |
| **Total** | **1.00** | |

### 6.3 Weight Summary Matrix

| Objective | Privacy-First | Utility-First | Balanced |
|-----------|---------------|---------------|----------|
| Fidelity | 0.25 | 0.30 | 0.25 |
| Privacy | 0.45 | 0.10 | 0.25 |
| Utility | 0.15 | 0.40 | 0.25 |
| Fairness | 0.10 | 0.10 | 0.15 |
| Efficiency | 0.05 | 0.10 | 0.10 |

### 6.4 Sensitivity Analysis Plan

In addition to archetype-specific results, conduct:

1. **Single-weight sensitivity:** Vary each weight ±0.15 while proportionally adjusting others. Identify threshold weights at which rank reversals occur.

2. **Dominance analysis:** Identify weight ranges (if any) over which a method is preferred regardless of other weights.

3. **Equal weights comparison:** Report results with all weights = 0.20 as a reference point.

---

## 7. Model Assumptions

### 7.1 Preferential Independence

**Assumption (modeling):** Preferences over one objective are (approximately) independent of the achieved level on other objectives, enabling an additive value model.

**Role in the paper:** This is a standard Decision Analysis modeling assumption rather than an empirically tested behavioral claim in this study.

**Why it is defensible here:** The objective is to provide *decision support* for method selection under heterogeneous priorities. For many practitioners, marginal improvements in fidelity, privacy, utility, fairness, and efficiency can be meaningfully traded off without requiring explicit interaction terms.

**Where it may fail (limitations):**
- **Hard constraints / vetoes:** If privacy is unacceptably low (e.g., extreme disclosure risk), fidelity/utility improvements may no longer be relevant.
- **Strong complementarity:** If utility depends on fidelity only above some threshold (or similar interactions), an additive model may misstate preferences.

**How we address this (robustness and transparency):**
- Report sensitivity analyses and explicitly discuss “veto” scenarios as decision rules (e.g., exclude any method failing a minimum privacy screen).
- Present results as conditional on this modeling stance, and interpret rank reversals qualitatively (why they occur) rather than treating small differences as definitive.

### 7.2 Additive Value Function

**Assumption (aggregation):** Overall value is represented as a weighted sum of single-attribute value functions.

**Status:** Adopted as the primary model, consistent with preferential independence (Keeney & Raiffa, 1976).

**Alternative considered:** A multiplicative model would impose strong interaction effects (e.g., zero value on one dimension collapsing overall value), which is not appropriate as a default for this decision context. We instead treat “must-have” requirements (if any) as explicit screening rules.

### 7.3 Risk Neutrality

**Assumption (value curvature):** Linear value functions (except efficiency) are used as a transparent baseline, implying constant marginal value of improvement over the observed range.

**Status:** Baseline specification; not presented as a behavioral truth.

**Sensitivity / robustness:** Re-estimate results with a simple piecewise-linear diminishing-returns alternative and report whether method rankings are stable.

---

## 8. Prediction Task Specification

### 8.1 Primary Classification Task

| Specification | Value |
|---------------|-------|
| Target Variable | HIGH_INCOME |
| Definition | 1 if PINCP > $50,000, else 0 |
| Task Type | Binary classification |
| Evaluation Metric | F1 Score (primary), AUC (secondary) |
| Class Balance | Approximately 40-60% (varies by sample) |

### 8.2 Classifier Specification

Two classifiers are used for different purposes (and must be reported distinctly):

1) **TSTR classifier (utility and subgroup gap):** Gradient Boosting (sklearn)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Algorithm | Gradient Boosting (sklearn) | Strong baseline, handles mixed types |
| n_estimators | 100 | Reasonable default |
| max_depth | 5 | Prevent overfitting |
| Other parameters | Defaults | Avoid hyperparameter sensitivity |

2) **Propensity classifier (fidelity):** Logistic regression (as specified in the propensity AUC definition)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Algorithm | Logistic Regression (sklearn) | Simple, fast, and standard for distinguishability |
| max_iter | 1000 | Ensures convergence |
| Other parameters | Defaults | Avoid hyperparameter sensitivity |

### 8.3 Cross-Validation

For TSTR evaluation:
- Train classifier on full synthetic dataset
- Evaluate on held-out real test set (N ≈ 7,500)
- No cross-validation within synthetic (would complicate interpretation)

For propensity score:
- 70/30 train/test split of combined real + synthetic
- Single split (not CV) for consistency

---

## 9. Fairness Subgroup Specification

### 9.1 Primary Subgroup Variable

| Specification | Value |
|---------------|-------|
| Variable | RAC1P (race) |
| Original Categories | 9 |
| Recoded Categories | 5 |

### 9.2 Subgroup Recoding

| Code | Label | Original RAC1P Values |
|------|-------|----------------------|
| 1 | White | 1 |
| 2 | Black | 2 |
| 3 | Asian | 6 |
| 4 | Hispanic | (Use HISP variable if available, else Other) |
| 5 | Other | 3, 4, 5, 7, 8, 9 |

**Note:** If Hispanic ethnicity variable (HISP) is available, use it to define Hispanic group. Otherwise, aggregate remaining categories into "Other."

### 9.3 Minimum Subgroup Size

| Requirement | Value | Rationale |
|-------------|-------|-----------|
| Minimum n in test set | 100 | Statistical stability |
| Action if below | Exclude from gap calculation | Document which groups excluded |

### 9.4 Secondary Subgroup Analysis (Optional)

If time permits, repeat analysis with:
- SEX (2 categories)
- SCHL recoded (education, 3 categories: < Bachelor's, Bachelor's, Graduate)

---

## 10. Replication Protocol

### 10.1 Number of Replicates

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Replicates per method | 5 | Enables variability assessment |
| Random seeds | 42, 123, 456, 789, 1011 | Fixed for reproducibility |

### 10.2 Reporting

For each measure, report:
- Mean across 5 replicates
- Standard deviation across 5 replicates
- Range (min, max)

### 10.3 Statistical Comparisons

When comparing methods:
- Report mean ± SD
- For key comparisons, conduct paired t-tests or Wilcoxon signed-rank tests
- Note whether differences exceed variability

---

## 11. Experimental Workflow

### 11.1 Phase 1: Setup and Validation

```
Week 1:
□ Download and preprocess ACS PUMS data
□ Implement all measure computation functions
□ Test each method on 1,000-row subset
□ Verify all methods run successfully
□ Document any required modifications
```

### 11.2 Phase 2: Full Experiments

```
Week 2:
□ Run Independent Marginals (5 replicates)
□ Run Synthpop (5 replicates)
□ Run CTGAN (5 replicates)
□ Run DP BN (“PrivBayes”) (5 replicates)
□ Run GReaT (5 replicates) — may require more time
□ Save all synthetic datasets
```

### 11.3 Phase 3: Measure Computation

```
Week 3:
□ Compute fidelity measures for all 25 datasets
□ Compute privacy measures for all 25 datasets
□ Compute utility measures for all 25 datasets
□ Compute fairness measures for all 25 datasets
□ Record efficiency measures (already captured during generation)
□ Compile results into master matrix
```

### 11.4 Phase 4: Value Analysis

```
Week 4:
□ Determine empirical anchor points from results
□ Apply value functions to all measures
□ Compute overall value for each archetype
□ Generate visualizations (stacked bar charts)
□ Conduct sensitivity analysis
□ Apply Dees et al. decision-focused transformation
```

### 11.5 Phase 5: VOI Analysis and Writing

```
Week 5+:
□ Implement VOI analysis (comparing decision strategies)
□ Derive practical decision rules
□ Write results sections
□ Complete discussion and conclusions
```

---

## 12. Expected Outputs

### 12.1 Data Artifacts

| Artifact | Description | Storage |
|----------|-------------|---------|
| Preprocessed ACS PUMS | Train/val/test splits | `data/processed/` |
| Synthetic datasets | 25 datasets (5 methods × 5 replicates) | `data/synthetic/` |
| Raw measures | CSV with all measure values | `results/raw_measures.csv` |
| Value scores | CSV with transformed values | `results/value_scores.csv` |

### 12.2 Tables

| Table | Content |
|-------|---------|
| Table 1 | Dataset characteristics (N, variables, types) |
| Table 2 | Method specifications (parameters, implementation) |
| Table 3 | Raw performance matrix (methods × measures, mean ± SD) |
| Table 4 | Value scores by archetype (methods × archetypes) |
| Table 5 | Sensitivity analysis thresholds |
| Table 6 | VOI analysis results |

### 12.3 Figures

| Figure | Content |
|--------|---------|
| Figure 1 | Objectives hierarchy diagram |
| Figure 2 | Value functions (5 panels) |
| Figure 3 | Stacked bar chart: value by objective for each method (Balanced archetype) |
| Figure 4 | Trellis: stacked bar charts for all three archetypes |
| Figure 5 | Decision-focused transformation comparison |
| Figure 6 | Tornado diagram (weight sensitivity) |
| Figure 7 | VOI decision tree / heuristic |

---

## 13. Risk Mitigation

### 13.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GReaT fails to run (GPU) | Medium | Low | Drop GReaT; still have 4 methods |
| DP BN (“PrivBayes”) produces poor results | Low | Medium | Document as finding; formal privacy has cost |
| CTGAN training unstable | Low | Low | Use multiple seeds; report variability |
| Insufficient subgroup sizes | Low | Medium | Aggregate smaller groups; document |

### 13.2 Timeline Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Experiments take longer than expected | Medium | High | Start experiments immediately; parallelize |
| Results don't show interesting tradeoffs | Low | High | This is itself a finding; reframe contribution |
| Privacy measures computationally expensive | Medium | Medium | Sample-based approximation if needed |

---

## 14. Documentation and Reproducibility

### 14.1 Code Repository Structure

```
synthetic-data-da/
├── README.md
├── requirements.txt
├── config/
│   └── experiment_config.yaml    # All parameters from this document
├── data/
│   ├── raw/                      # Original ACS PUMS download
│   ├── processed/                # Preprocessed splits
│   └── synthetic/                # Generated synthetic datasets
├── tsd/
│   ├── preprocessing.py          # Data loading and preprocessing
│   ├── generators/               # Method wrappers
│   │   ├── baseline.py           # Independent marginals
│   │   ├── synthpop_wrapper.py
│   │   ├── ctgan_wrapper.py
│   │   ├── privbayes_wrapper.py
│   │   └── great_wrapper.py
│   ├── measures/                 # Measure computation
│   │   ├── fidelity.py
│   │   ├── privacy.py
│   │   ├── utility.py
│   │   ├── fairness.py
│   │   └── efficiency.py
│   ├── analysis/                 # Value modeling
│   │   ├── value_functions.py
│   │   ├── maut.py
│   │   └── sensitivity.py
│   └── visualization/            # Plotting
├── notebooks/                    # Exploratory analysis
├── results/                      # Output tables and figures
└── paper/                        # LaTeX source
```

### 14.2 Reproducibility Checklist

- [ ] All random seeds documented and fixed
- [ ] Package versions recorded in requirements.txt
- [ ] Hardware specifications documented
- [ ] All preprocessing steps scripted (no manual steps)
- [ ] Raw data preserved; processed data regenerable
- [ ] Analysis pipeline runnable end-to-end

---

## Appendix A: Decision Checklist (Completed)

```
CONTRIBUTION
[X] Paper type: Applied DA paper demonstrating framework produces insights
[X] One-sentence contribution: We demonstrate that optimal synthetic data 
    generation method selection depends critically on practitioner objectives,
    with no method dominating across all criteria.

DATASETS
[X] Primary: ACS PUMS 2024 (California extract, N=50,000)
[X] Secondary: None initially; optional extension
[ ] Data acquired and preprocessed: PENDING

METHODS
[X] Independent marginals (baseline)
[X] Synthpop
[X] CTGAN
[X] DP BN ("PrivBayes")
[X] GReaT
[ ] All methods tested on small sample: PENDING

MEASURES
Fidelity:    Primary: Propensity AUC       Secondary: Avg KS, Avg TVD
Privacy:     Primary: DCR 5th percentile   Secondary: None
Utility:     Primary: TSTR F1 Ratio        Secondary: TSTR AUC Ratio
Fairness:    Primary: Max Subgroup Gap     Secondary: None
Efficiency:  Primary: Total Time (min)     Secondary: None

VALUE FUNCTIONS
  Fidelity:   x_best=0.50     x_worst=empirical    Form: linear
  Privacy:    x_best=empirical x_worst=0           Form: linear
  Utility:    x_best=1.0      x_worst=empirical    Form: linear
  Fairness:   x_best=0        x_worst=empirical    Form: linear
  Efficiency: x_best=empirical x_worst=480         Form: logarithmic

WEIGHTS
[X] Approach: Stakeholder Archetypes + Sensitivity Analysis
[X] Archetypes defined:
    Privacy-First:  Fid=0.25 Priv=0.45 Util=0.15 Fair=0.10 Eff=0.05
    Utility-First:  Fid=0.30 Priv=0.10 Util=0.40 Fair=0.10 Eff=0.10
    Balanced:       Fid=0.25 Priv=0.25 Util=0.25 Fair=0.15 Eff=0.10

INDEPENDENCE ASSUMPTION
[X] Assumed (stated as limitation in paper)

PREDICTION TASK
[X] Target variable: HIGH_INCOME (PINCP > $50,000)
[X] Task type: Binary Classification

FAIRNESS SUBGROUPS
[X] Groups: White, Black, Asian, Hispanic, Other (from RAC1P)
[X] Minimum subgroup size: n > 100

REPLICATES
[X] Number per method: 5
[X] Random seeds: 42, 123, 456, 789, 1011
```

---

## Appendix B: Quick Reference Card

**One-Page Summary for Running Experiments**

| Item | Specification |
|------|---------------|
| **Data** | ACS PUMS 2024 (California extract), N=50,000, 70/15/15 split |
| **Methods** | Indep. Marginals, Synthpop, CTGAN, DP BN (“PrivBayes”), GReaT |
| **Replicates** | 5 per method, seeds: 42, 123, 456, 789, 1011 |
| **Fidelity** | Propensity AUC (target: 0.5) |
| **Privacy** | DCR 5th percentile (higher = better) |
| **Utility** | TSTR F1 ratio (target: 1.0) |
| **Fairness** | Max subgroup utility gap (target: 0) |
| **Efficiency** | Total time in minutes |
| **Target** | HIGH_INCOME = PINCP > $50K |
| **Subgroups** | Race (5 categories), min n=100 |
| **Archetypes** | Privacy-First, Utility-First, Balanced |

---
