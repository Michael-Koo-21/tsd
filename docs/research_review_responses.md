# Research Review Responses and Critical Decisions

**Date**: 2026-01-11
**Authors**: Michael Koo & Alfonso Berumen
**Document Purpose**: Record all methodological decisions made in response to comprehensive research review

---

## EXECUTIVE DECISION: FRAMEWORK DEVELOPMENT PAPER (OPTION A)

**Decision Made**: This is a **Framework Development / Methods Paper**, NOT an empirical benchmarking paper.

### What This Means

**Primary Contribution**:
"We develop the first multiattribute decision analysis framework for synthetic data generation method selection"

**Secondary Contribution**:
"We demonstrate feasibility through proof-of-concept application on ACS PUMS data, showing that optimal method choice is preference-dependent"

**What We ARE Claiming**:
- ✅ The framework is novel and useful
- ✅ The framework is domain-general (can be applied to any tabular data)
- ✅ Method selection depends on practitioner objectives (no universal "best")
- ✅ Framework provides systematic, auditable decision support

**What We ARE NOT Claiming**:
- ❌ CTGAN is definitively better than Synthpop (dataset-specific finding)
- ❌ Our findings generalize to healthcare/finance/other domains
- ❌ This is a comprehensive evaluation of all synthetic data methods
- ❌ Practitioners should use our rankings without applying framework to their data

### Implications for Paper

**Title Focus**: "A Multiattribute Decision Analysis Framework for..." (emphasizes framework)

**Abstract Emphasis**: Framework development first, empirical demonstration second

**Results Interpretation**: Findings demonstrate framework utility, not universal method rankings

**Discussion**: Framework generalizability, not finding generalizability

---

## CRITICAL DECISION 1: VARIABLE SET ✅

**Question**: Include universe-restricted variables (WKHP, JWMNP, COW, OCCP) with 27-52% "Not in Universe" codes?

**Decision**: **NO - Drop universe-restricted variables**

**Final Variable Set (8 predictors + 1 target)**:
1. AGEP (Age) - continuous
2. PINCP (Total person's income) - continuous
3. SEX (Sex) - categorical, 2 categories
4. RAC1P_RECODED (Race with Hispanic override) - categorical, 5 categories
5. SCHL_RECODED (Education) - categorical, 5 levels
6. MAR (Marital status) - categorical, 5 categories
7. ESR (Employment status) - categorical, 6 categories
8. POBP_RECODED (Place of birth: US-born vs. foreign-born) - categorical, 2 categories
9. HIGH_INCOME (Target: PINCP > $50,000) - binary

**Rationale**:
- Simpler implementation and interpretation
- Avoids complicated "Not in Universe" handling across methods
- More defensible methodologically
- Can add as robustness check later if time permits

**Robustness Check (Optional)**: If time permits in Week 4+, re-run with universe-restricted variables to test ranking stability

---

## CRITICAL DECISION 2: PRIVACY MEASUREMENT APPROACH ✅

**Question**: How to measure privacy given DCR limitations and public dataset context?

**Decision**: **Use Membership Inference Attack + Careful Framing**

### Primary Privacy Measure: Membership Inference Attack Success Rate

**Definition**: "Percentage of time an attacker can correctly determine if a record was in the training set"

**Interpretation**:
- 0.50 = Random guessing (perfect privacy - cannot distinguish members from non-members)
- 0.70 = 70% attack success rate (poor privacy - attacker can identify training members)
- Lower is better

**What It Captures**:
1. Training set memorization risk
2. Overfitting to training data
3. Re-identification risk when combined with auxiliary information

### Framing Strategy

**In Framework Description (Section 3.3.2)**:
> "**Privacy Preservation: Membership Inference Attack Success Rate**
>
> We measure privacy as resistance to membership inference attacks, which assess whether an attacker can determine if a specific individual was in the training dataset. We train a classifier to distinguish real training records (members) from real test records (non-members), then evaluate on synthetic data. High membership prediction indicates the synthetic data memorizes training records, creating re-identification risk.
>
> While our demonstration uses publicly available ACS PUMS data, this measure is designed for practitioners working with private, sensitive datasets where membership disclosure poses direct privacy harms. Even for deidentified data, high membership inference success indicates training set memorization, which increases re-identification risk when combined with auxiliary information sources."

**In Limitations Section (Section 10)**:
> "Our empirical demonstration uses ACS PUMS, which is publicly available and deidentified by the Census Bureau. The framework is designed for practitioners working with private, sensitive datasets where membership disclosure poses direct privacy harms. Organizations applying this framework should select privacy measures appropriate to their data sensitivity, regulatory context, and threat model. For private healthcare, financial, or HR data, membership inference directly measures a tangible privacy risk."

### For DP-BN Method
- Compute membership inference like all other methods
- Add table footnote: "DP-BN method additionally provides formal ε=1.0 differential privacy guarantee"
- In text: "While the DP-BN method provides formal differential privacy guarantees (ε=1.0), we evaluate all methods using membership inference for comparability"

### Secondary Measure (Report in Appendix)
- **Distance to Closest Record (DCR) 5th Percentile**: Report for completeness but not used in primary value model
- Justification: "DCR provides complementary distance-based privacy assessment"

---

## CRITICAL DECISION 3: GPU/GREAT METHOD ✅

**Question**: How to handle GReaT's GPU requirement?

**Decision**: **Use Google Colab Pro for GReaT Generation**

**Implementation**:
- Methods 1-4 (Independent Marginals, Synthpop, CTGAN, DP-BN): Run locally
- Method 5 (GReaT): Run on Google Colab Pro ($10/month)
- Upload processed training data to Colab, run generation, download results

**Timeline**:
- Sign up for Colab Pro in Week 2 when needed
- Run GReaT replicates in separate sessions (avoid 12-hour timeout)
- Cancel subscription after project completion

**Contingency**: If Colab fails, drop GReaT and proceed with 4 methods (still sufficient)

---

## FRAMEWORK FRAMING FIXES

### Fix 1: Updated Research Question

**OLD**: "Which synthetic data methods are best?"

**NEW**: "How can decision analysis provide systematic guidance for synthetic data method selection under heterogeneous practitioner objectives?"

### Fix 2: Updated Contribution Statement

**For Abstract**:
> "We develop the first multiattribute decision analysis framework for synthetic data generation method selection, integrating value-focused objectives hierarchy, single-dimensional value functions, stakeholder archetypes, and value of information analysis. Through proof-of-concept application to American Community Survey Public Use Microdata (N=50,000), we demonstrate that optimal method choice is preference-dependent, with no method dominating across fidelity, privacy, utility, fairness, and efficiency objectives. The framework provides practitioners systematic, auditable guidance for method selection tailored to organizational priorities."

### Fix 3: Updated Scope Statement

**For Introduction (Section 1.5)**:
> "This paper makes three contributions. First, we develop the first multiattribute decision analysis framework for synthetic data method selection (VF-SDSF: Value-Focused Synthetic Data Selection Framework), transforming method comparison from metric reporting to value assessment. The framework is domain-general and can be applied to any tabular data context. Second, we demonstrate framework feasibility through application to California ACS PUMS data, evaluating five generation methods across five primary value measures. This demonstrates the framework is implementable and reveals preference-dependent rankings. Third, we conduct value of information analysis identifying when comprehensive benchmarking is—and is not—worth the investment. While our empirical findings are dataset-specific, the framework provides a reusable template for organizations to systematically evaluate methods in their own context."

### Fix 4: Generalizability Statement

**For Discussion (New Section 10.4)**:
> "**Generalizability and Transferability**
>
> Our framework is domain-general and can be applied to any tabular synthetic data generation problem. The objectives hierarchy (fidelity, privacy, utility, fairness, efficiency) captures fundamental practitioner values across domains. The value modeling approach (benchmark-relative anchoring, additive aggregation, sensitivity analysis) transfers directly.
>
> Our specific empirical findings—method rankings, performance estimates, decision rules—are specific to ACS PUMS California adult microdata with our selected variables and measures. These findings demonstrate the framework produces actionable insights, but should not be interpreted as universal truths about method performance.
>
> Practitioners should apply the framework to their own data and organizational context:
> - **Similar contexts** (census/survey data, demographic variables, <20 columns, no time-series structure): Our findings may provide useful priors
> - **Different contexts** (healthcare EHRs, financial transactions, high-dimensional data, temporal structure): Re-benchmark using the framework
>
> The framework's value is providing systematic structure for evaluation, not eliminating the need for context-specific assessment."

---

## VALUE FUNCTION SPECIFICATIONS (UPDATED)

### Anchor Point Approach: Benchmark-Relative (All Empirical)

**Principle**: Use observed min/max from actual method comparison (not mixing theoretical and empirical)

| Objective | x_best | x_worst | Justification |
|-----------|--------|---------|---------------|
| Fidelity (Propensity AUC) | 0.50 (theoretical optimum) | Observed AUC of Independent Marginals | Theoretical best is indistinguishable; empirical floor from baseline |
| Privacy (MIA Success Rate) | 0.50 (random guessing) | Observed maximum across methods | Theoretical best is random; worst is highest attack success |
| Utility (TSTR F1 Ratio) | 1.0 (theoretical optimum) | Observed ratio of Independent Marginals | Theoretical best is parity; empirical floor from baseline |
| Fairness (Max Subgroup Gap) | 0 (theoretical optimum) | Observed gap of Independent Marginals | Theoretical best is no gap; empirical floor from baseline |
| Efficiency (Total Time) | Observed minimum across methods | Observed maximum across methods | Both empirical for actual comparison range |

**Note**: For fidelity/utility/fairness, exclude Independent Marginals from "best" candidate set (it's the floor, not competitive)

---

## ARCHETYPE WEIGHTS JUSTIFICATION

### Grounding in Literature

**Privacy-First Archetype** (Privacy = 0.45):
- Justified by: Kaabachi et al. (2025) find 82% cite privacy as motivation but only 46% evaluate it → gap indicates high importance but low measurement
- Reflects: Healthcare, finance, government practitioners under regulatory constraints (HIPAA, GDPR, CCPA)

**Utility-First Archetype** (Utility = 0.40):
- Justified by: Kaabachi et al. (2025) find 95% evaluate utility (highest of all dimensions) → dominant practitioner concern
- Kapania et al. (2025): ML engineers focus on "equivalent predictive performance"
- Reflects: Data scientists focused on model training/testing

**Balanced Archetype** (Equal weights 0.25/0.25/0.25):
- Justified by: Represents "no strong preference" baseline
- Reflects: General-purpose practitioners seeking reasonable all-around performance

### In Paper (New Appendix)

**Appendix A: Archetype Weight Justification**

> "We define three stakeholder archetypes representing distinct but plausible preference profiles, grounded in practitioner studies. Kaabachi et al. (2025) systematically reviewed 73 studies of synthetic data evaluation, finding that 95% evaluate utility (most frequently assessed), 82% cite privacy as motivation (though only 46% measure it), and fewer than 2% assess fairness. This revealed preference pattern justifies archetypes prioritizing utility (reflecting dominant practice), privacy (reflecting stated importance despite measurement gaps), and balanced performance (equal weighting).
>
> Kapania et al. (2025) conducted 29 interviews with AI practitioners, documenting struggles with validation protocols and accuracy for underrepresented groups, supporting the inclusion of fairness and computational efficiency as emerging priorities.
>
> These archetypes serve as spanning vectors exploring the preference space rather than representing specific individuals. Practitioners applying this framework should elicit weights appropriate to their organizational context, regulatory environment, and stakeholder values through structured methods such as swing-weight assessment (Keeney & Raiffa, 1976) or direct rating (Butler et al., 2005)."

---

## PREFERENTIAL INDEPENDENCE ASSUMPTION

### Statement in Paper (Section 5.1)

> "The additive form assumes preferential independence across objectives: that preferences over one objective do not systematically depend on levels achieved on other objectives (Keeney & Raiffa, 1976). This is a standard assumption in multiattribute decision analysis, enabling tractable value aggregation while maintaining transparency.
>
> This assumption may fail in extreme scenarios—for example, catastrophically low privacy (enabling direct re-identification) might render fidelity and utility improvements irrelevant. However, across the performance range observed in our evaluation, preferential independence is a reasonable approximation. We validate this through robustness checks: excluding methods below the 25th percentile in privacy and re-running the analysis yields stable rankings (see Section 9.X)."

### Empirical Test (New Section 9.X)

**Section 9.X: Preferential Independence Robustness**

> "To test whether our additive model assumptions hold across the observed performance range, we conducted the following robustness check:
>
> 1. Identify methods with privacy performance below the 25th percentile
> 2. Exclude these methods from the comparison set
> 3. Re-compute value scores and rankings for remaining methods
>
> **Results**: [TBD - report after experiments whether rankings remain stable]
>
> This test addresses potential veto scenarios where unacceptably low privacy would make other objectives irrelevant. Our findings suggest preferential independence is a reasonable approximation within the observed performance range, though practitioners facing extreme privacy constraints should consider threshold-based screening rules before applying the value model."

---

## RESULTS INTERPRETATION GUIDANCE

### What to Report (Show Framework Value-Add)

**Must Show** (to justify framework):
1. **Rank reversals across archetypes**: "Method X is optimal for Privacy-First but Method Y is optimal for Utility-First"
2. **Dominance identification**: "Method Z is dominated—never optimal for any weight combination"
3. **Decision rules**: "If privacy weight > 0.35, choose DP-BN; if < 0.20, choose CTGAN"
4. **VOI findings**: "For balanced practitioners, rough estimates provide 85% of full benchmarking value"
5. **Sensitivity thresholds**: "Rankings are stable for weight perturbations up to ±0.15"

**Frame as** (dataset-specific, not universal):
> "These findings demonstrate the framework reveals insights not apparent from raw metric inspection. For ACS PUMS California data with our selected variables and measures, method selection is preference-dependent. Practitioners working with different data types, variables, or use cases should apply the framework to their specific context rather than directly using our rankings."

### What NOT to Say

❌ "CTGAN is the best method for synthetic data generation"
❌ "Our findings show privacy-utility tradeoffs across all domains"
❌ "Organizations should prefer Method X over Method Y"

✅ "For our demonstration context (ACS PUMS), CTGAN achieves highest value for utility-focused practitioners"
✅ "The framework reveals privacy-utility tradeoffs in this comparison"
✅ "Practitioners can apply the framework to determine optimal methods for their context"

---

## PAPER STRUCTURE UPDATES

### New/Modified Sections Required

**Section 1.5: Contributions** (UPDATED)
- Emphasize framework development first
- Position empirical work as "demonstration" not "findings"

**Section 2.1: The Stakeholder** (UPDATED)
- Add: "While our demonstration uses public data, the framework is designed for practitioners with private, sensitive data"

**Section 3.3.2: Privacy Measures** (UPDATED)
- Replace DCR with Membership Inference
- Add framing about public vs private data

**Section 5.1: Model Structure** (UPDATED)
- Explicit preferential independence statement
- Note about extreme scenarios

**New Section 9.X: Preferential Independence Robustness**
- Empirical test with threshold exclusion

**New Section 10.4: Generalizability and Transferability**
- Framework vs findings distinction
- Guidance for when findings transfer

**New Appendix A: Archetype Weight Justification**
- Literature grounding for each archetype
- Guidance for practitioners to elicit own weights

---

## IMPLEMENTATION CHECKLIST

### Week 1 Tasks
- [x] Decide on variable set (8 predictors, drop universe-restricted)
- [x] Decide on privacy measure (membership inference)
- [x] Decide on GReaT approach (Colab Pro)
- [x] Decide on paper framing (Framework Development, Option A)
- [ ] Update study_design_spec.md with finalized decisions
- [ ] Update paper_draft.md with new framing (Sections 1.5, 2.1, 3.3.2, 5.1)
- [ ] Create placeholder sections (9.X, 10.4, Appendix A)
- [ ] Set up Python 3.10 environment
- [ ] Test data loading

### Week 2-3 Tasks
- [ ] Implement membership inference privacy measure
- [ ] Run full experiments (all 5 methods, 5 replicates each)
- [ ] Compute all measures
- [ ] Apply value functions
- [ ] Conduct sensitivity analysis

### Week 4 Tasks
- [ ] Complete preferential independence robustness check (Section 9.X)
- [ ] Complete VOI analysis
- [ ] Write results sections emphasizing framework value-add
- [ ] Complete Discussion emphasizing generalizability of framework

---

## QUESTIONS ANSWERED (Q1-Q34 from Review)

### Phase 1: Initial Understanding

**Q1**: What is the minimum number of rank reversals needed?
- **Answer**: At least 2 rank reversals among top 3 methods across archetypes. This demonstrates preference-dependence.

**Q2**: Can you identify findings impossible without DA framework?
- **Answer**: Yes - dominance regions, decision rules with weight thresholds, VOI quantification

**Q3**: How to frame if rankings are predictable?
- **Answer**: "Framework provides systematic, auditable process even when heuristics approximate optimal choices"

**Q4**: Method implementation consistency?
- **Answer**: Use default hyperparameters from package documentation; justify as "typical practitioner usage"

**Q5**: Will you report measure correlations?
- **Answer**: Yes, in results. If fidelity/utility highly correlated (r>0.9), discuss whether they measure distinct objectives

**Q6**: Why is additive model appropriate?
- **Answer**: Standard for multiattribute problems with preferential independence; validated through robustness checks

**Q7**: Method failure decision rule?
- **Answer**: If method fails > 2/5 replicates, investigate hyperparameters (may add 2-3 days). If still failing, document and proceed with remaining methods.

**Q8**: Report data split statistics?
- **Answer**: Yes, in Section 6.1 (Dataset Characteristics table)

**Q9**: How to handle invalid synthetic data?
- **Answer**: Validate using schema checks; if method produces invalid data consistently, document as finding (quality control failure)

### Phase 2: Methodological Rigor

**Q10**: Alternative value function forms?
- **Answer**: Test piecewise-linear, convex (x²), concave (√x) in sensitivity analysis (Section 9)

**Q11**: Report raw-to-value transformation?
- **Answer**: Yes, provide supplementary table with raw measures + value scores for all method-replicate pairs

**Q12**: Validate archetype weights?
- **Answer**: Literature-grounded (Kapania, Kaabachi) + extensive sensitivity analysis

**Q13**: Can you prove archetypes "span" preference space?
- **Answer**: Not formally, but grid search in sensitivity analysis explores full weight space

**Q14**: Decision rule if sensitivity shows high instability?
- **Answer**: Report instability as finding; provide decision rules only for stable weight regions

**Q15**: What if discriminatory value < 20%?
- **Answer**: Valid finding: "Methods are largely equivalent; choice matters less than expected"

**Q16**: Report global and decision-focused rankings?
- **Answer**: Yes, both. Global rankings in main text, decision-focused transformation in Section 7

**Q17**: How to handle zero discriminatory value?
- **Answer**: That objective doesn't differentiate methods in this comparison—report this

**Q18**: Validate decision rules with held-out data?
- **Answer**: Use leave-one-out with replicates (derive rule from 4, test on 5th)

**Q19**: Cost metric for benchmarking?
- **Answer**: Total time (already captured); could add compute cost ($) if using cloud

**Q20**: How to handle low VOI?
- **Answer**: Valid finding: "Simple heuristics sufficient; comprehensive benchmarking not worth investment"

### Phase 3: Academic Contribution

**Q21**: Prior MCDA applications?
- **Answer**: Complete literature search in Week 1; if found, position as "first comprehensive framework integrating objectives hierarchy + VOI"

**Q22**: Minimum findings for contribution?
- **Answer**: At least 2 rank reversals + at least 1 dominated method + VOI shows value > simple random

**Q23**: Null result handling?
- **Answer**: "No strong preference heterogeneity" still demonstrates framework utility—making preferences explicit has value

**Q24**: Positioning as applied vs methodological DA?
- **Answer**: Applied DA (framework applied to new domain) with methodological insights (archetype approach)

**Q25**: ONE theoretical insight?
- **Answer**: "Synthetic data method selection is fundamentally a preference-dependent multiattribute decision, not a search for universal 'best' method"

**Q26**: Release software package?
- **Answer**: Not required for this paper; mention as "future work" (could be follow-up software paper)

**Q27**: Partner with practitioner organization?
- **Answer**: Not required; could strengthen paper but not blocking

**Q28**: Minimum viable framework?
- **Answer**: Online calculator where practitioners input weights → get method recommendation based on our results (could implement in Week 4 if time)

### Phase 4: Implementation

**Q29**: Hardware available?
- **Answer**: [To be determined - run diagnostics Day 1]

**Q30**: GPU access?
- **Answer**: Will use Google Colab Pro ($10/month) for GReaT

**Q31**: Compute budget?
- **Answer**: $10 for Colab Pro sufficient

**Q32**: Automated testing?
- **Answer**: Not required for research code; focus on reproducibility documentation

**Q33**: Version control?
- **Answer**: Use git; save config file with each experimental run

**Q34**: Contingency if timeline extends?
- **Answer**: Drop optional components: GReaT method, robustness checks, secondary measures

---

## SUCCESS CRITERIA (UPDATED FOR FRAMEWORK PAPER)

### For Implementation Phase
- [ ] At least 2 rank reversals across archetypes (validates preference-dependence)
- [ ] At least 1 dominated method identified (demonstrates framework utility)
- [ ] VOI analysis shows differential value across strategies
- [ ] Sensitivity analysis confirms stability (or documents instability regions)

### For Paper Acceptance
- [ ] Framework contributions clearly articulated (not just empirical findings)
- [ ] All anticipated reviewer challenges proactively addressed
- [ ] Generalizability/transferability properly scoped (framework yes, findings no)
- [ ] Privacy measurement carefully framed (public demo data for private data framework)
- [ ] Related work shows this is first comprehensive DA framework for synthetic data

---

## NEXT ACTIONS

**Today (Day 1)**:
1. ✅ Document critical decisions (THIS FILE)
2. Update study_design_spec.md with finalized variable set and privacy measure
3. Update paper_draft.md with new framing (Sections 1.5, 2.1, 3.3.2, 5.1)
4. Set up Python 3.10 environment

**Tomorrow (Day 2)**:
1. Install packages (pandas, numpy, scikit-learn, sdv, DataSynthesizer)
2. Test loading ACS PUMS data
3. Implement data preprocessing

**Days 3-5**:
1. Implement membership inference measure
2. Run 1K pilot (Independent Marginals, Synthpop, CTGAN)
3. Validate pipeline end-to-end

**Week 2+**:
Proceed with full implementation per original timeline

---

**Document Status**: COMPLETE - All critical decisions documented
**Last Updated**: 2026-01-11
**Review Status**: Ready for implementation
