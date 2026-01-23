# Pre-Registration: Expected Results & Hypotheses
**Date:** 2026-01-12
**Authors:** Michael Koo & Alfonso Berumen
**Purpose:** Document expected results BEFORE running full experiments to protect against HARKing

---

## 1. Purpose of Pre-Registration

This document records our predictions about experimental outcomes before analyzing results. This:
- Protects against HARKing (Hypothesizing After Results are Known)
- Distinguishes confirmatory from exploratory findings
- Strengthens methodological rigor for publication
- Provides audit trail for reviewers

**IMPORTANT:** This document is locked after validation pilot completes. Any deviations from these expectations must be documented in results section as "unexpected findings."

---

## 2. Expected Method Rankings

### 2.1 Privacy-First Archetype (w_privacy = 0.45)

**Expected Ranking (best to worst):**
1. **DP Bayesian Network** - Formal ε-DP guarantee (ε=1.0)
2. **Independent Marginals** - No correlation preservation = no memorization
3. **Synthpop** - CART-based, moderate memorization risk
4. **CTGAN** - Deep learning, higher memorization risk
5. **GReaT** - LLM fine-tuning, highest memorization risk

**Rationale:**
- Methods with formal privacy guarantees or simpler architectures should rank higher
- Deep learning methods (CTGAN, GReaT) likely memorize training data more
- Literature: Zhang et al. (2021) show GANs can memorize training examples

**Expected Value Range:** V_privacy_first = [0.35, 0.75]

### 2.2 Utility-First Archetype (w_utility = 0.40)

**Expected Ranking (best to worst):**
1. **CTGAN** - Best distributional fidelity in literature
2. **GReaT** - LLM-based, emerging strong performer
3. **Synthpop** - Established, reliable performance
4. **DP Bayesian Network** - Privacy-utility tradeoff hurts utility
5. **Independent Marginals** - No correlations = poor utility

**Rationale:**
- Methods that preserve correlations should support better ML utility
- DP methods sacrifice utility for privacy (established tradeoff)
- Independent Marginals is floor (no joint distribution)

**Expected Value Range:** V_utility_first = [0.25, 0.85]

### 2.3 Balanced Archetype (w_i = 0.20-0.25)

**Expected Ranking (best to worst):**
1. **CTGAN** or **Synthpop** - Good all-around performance
2. **DP Bayesian Network** - Strong privacy compensates for utility loss
3. **GReaT** - High cost hurts overall value
4. **Independent Marginals** - Dominated baseline

**Rationale:**
- Balanced weights favor methods without extreme weaknesses
- Efficiency becomes more important in balanced view
- GReaT's computational cost is significant drag

**Expected Value Range:** V_balanced = [0.35, 0.70]

---

## 3. Expected Measure Values

### 3.1 Fidelity (Propensity AUC) - Lower is Better

| Method | Expected AUC | Range | Justification |
|--------|--------------|-------|---------------|
| Independent Marginals | 0.90 | [0.85, 0.95] | Worst fidelity - no correlations |
| Synthpop | 0.65 | [0.60, 0.75] | Moderate - CART preserves some structure |
| CTGAN | 0.58 | [0.52, 0.65] | Good - GAN trained for distributional match |
| DP-BN | 0.70 | [0.65, 0.80] | Privacy noise degrades fidelity |
| GReaT | 0.62 | [0.55, 0.70] | Good - LLM captures patterns |

**Expected Discriminatory Value:** (0.90 - 0.58) / 0.69 = **0.46** (moderate-high discrimination)

### 3.2 Privacy (DCR 5th Percentile) - Higher is Better

**Note:** Using DCR as primary measure based on validation pilot. If membership inference becomes valid at N=35K, report both.

| Method | Expected DCR | Range | Justification |
|--------|--------------|-------|---------------|
| Independent Marginals | High | [0.8, 1.2] | No memorization - synthetic far from training |
| DP-BN | High | [0.7, 1.0] | Formal privacy adds noise |
| Synthpop | Medium | [0.4, 0.7] | Some close matches from CART leaves |
| CTGAN | Low | [0.2, 0.5] | Deep model may generate near-copies |
| GReaT | Low | [0.1, 0.4] | LLM fine-tuning = high memorization risk |

**Expected Discriminatory Value:** (1.0 - 0.25) / 0.55 = **1.36** (high discrimination)

### 3.3 Utility (TSTR F1 Ratio) - Higher is Better

**Note:** This measure is not yet implemented. Expected values based on literature.

| Method | Expected Ratio | Range | Justification |
|--------|----------------|-------|---------------|
| Independent Marginals | 0.45 | [0.35, 0.55] | Floor - no correlations |
| Synthpop | 0.78 | [0.70, 0.85] | Good - preserves key relationships |
| CTGAN | 0.85 | [0.78, 0.92] | Best - GANs excel at utility |
| DP-BN | 0.62 | [0.50, 0.70] | Privacy hurts utility |
| GReaT | 0.80 | [0.72, 0.88] | Good - LLM captures patterns |

**Expected Discriminatory Value:** (0.85 - 0.45) / 0.70 = **0.57** (high discrimination)

### 3.4 Fairness (Max Subgroup Utility Gap) - Lower is Better

**Note:** This measure is not yet implemented. Expected values are speculative.

| Method | Expected Gap | Range | Justification |
|--------|--------------|-------|---------------|
| Independent Marginals | 0.15 | [0.10, 0.20] | Random sampling may amplify group differences |
| Synthpop | 0.08 | [0.05, 0.12] | CART may preserve group patterns |
| CTGAN | 0.10 | [0.06, 0.15] | GANs can amplify biases |
| DP-BN | 0.06 | [0.03, 0.10] | DP noise may reduce group differences |
| GReaT | 0.12 | [0.08, 0.18] | LLM biases from pre-training |

**Expected Discriminatory Value:** (0.15 - 0.06) / 0.10 = **0.90** (high discrimination)

### 3.5 Efficiency (Total Time, Minutes) - Lower is Better

| Method | Expected Time | Range | Justification |
|--------|--------------|-------|---------------|
| Independent Marginals | 0.5 min | [0.1, 1] | Trivial - just sampling |
| Synthpop | 15 min | [10, 25] | R overhead + CART training |
| CTGAN | 45 min | [30, 90] | GAN training (CPU) |
| DP-BN | 10 min | [5, 20] | Bayesian network learning |
| GReaT | 240 min | [120, 360] | LLM fine-tuning (GPU) |

**Expected Discriminatory Value:** (240 - 0.5) / 60 = **3.99** (very high discrimination)

---

## 4. Expected Measure Correlations

### 4.1 Pre-Registered Correlation Matrix

| Measure Pair | Expected r | Range | Interpretation |
|--------------|------------|-------|----------------|
| Fidelity ↔ Utility | +0.60 | [0.4, 0.8] | Moderate positive - better match → better utility |
| Fidelity ↔ Privacy | -0.40 | [-0.6, -0.2] | Moderate negative - better match → more memorization |
| Utility ↔ Privacy | -0.50 | [-0.7, -0.3] | Moderate negative - classic tradeoff |
| Fidelity ↔ Fairness | +0.20 | [-0.1, 0.5] | Weak positive - preserving patterns may preserve biases |
| Efficiency ↔ Fidelity | -0.30 | [-0.5, 0.0] | Weak negative - complex methods take longer |

### 4.2 Validation Criteria

**If observed |r| > 0.85 for any pair:**
- Flag as high correlation in results
- Consider combining into single objective
- Discuss implications for multiattribute framework

**If observed r differs from expected by > 0.4:**
- Report as unexpected finding
- Investigate reasons for discrepancy
- Discuss in results section

---

## 5. Expected Key Findings

### 5.1 Primary Hypotheses (Confirmatory)

**H1: Preference-Dependent Rankings**
- **Prediction:** At least 2 rank reversals among top 3 methods across archetypes
- **Test:** Compare rankings for Privacy-First vs Utility-First
- **Expected:** DP-BN best for Privacy-First, CTGAN best for Utility-First

**H2: No Dominant Method**
- **Prediction:** No single method is Pareto-optimal across all objectives
- **Test:** Check if any method achieves top 2 ranking on all 5 objectives
- **Expected:** Each method has at least one dimension where it ranks bottom-3

**H3: Privacy-Utility Tradeoff**
- **Prediction:** Methods with better privacy have worse utility (r = -0.5 ± 0.2)
- **Test:** Correlation between privacy scores and utility scores
- **Expected:** Negative correlation confirms classic tradeoff

### 5.2 Secondary Hypotheses (Exploratory)

**H4: Efficiency-Quality Tradeoff**
- **Prediction:** Faster methods have lower quality (weak negative correlation)
- **Test:** Correlate efficiency with average of fidelity+utility scores
- **Expected:** r = -0.3 ± 0.3

**H5: Method Stability Across Replicates**
- **Prediction:** CV (coefficient of variation) < 0.3 for all measures
- **Test:** Compute std/mean across 5 replicates per method
- **Expected:** Stable results justify N=5 replicates

**H6: Independent Marginals as Effective Anchor**
- **Prediction:** Independent Marginals consistently ranks worst or 2nd-worst
- **Test:** Count how many objectives it ranks bottom-2
- **Expected:** ≥ 3 out of 5 objectives

---

## 6. Decision Rules & Value of Information

### 6.1 Expected VOI Patterns

**Strategy Comparison:**
1. **Full Benchmark** (our approach): Evaluate all 5 methods on all 5 measures
2. **Heuristic 1** (fidelity-only): Choose method with best propensity AUC
3. **Heuristic 2** (privacy-only): Choose method with best DCR
4. **Default Strategy**: Always use CTGAN (practitioner default)

**Expected VOI:**
- Full benchmark vs Fidelity-only: **VOI = 0.15** (15% improvement for privacy-focused users)
- Full benchmark vs Privacy-only: **VOI = 0.20** (20% improvement for utility-focused users)
- Full benchmark vs Default (CTGAN): **VOI = 0.25** (25% improvement for privacy-focused users)

**Interpretation:**
- If VOI > 10%: "Comprehensive benchmarking is worth the investment"
- If VOI < 5%: "Simple heuristics suffice for most users"

### 6.2 Expected Decision Rules

Based on expected rankings, we anticipate deriving rules like:

```
IF privacy_weight > 0.35:
    THEN prefer DP-BN or Independent Marginals

IF utility_weight > 0.35:
    THEN prefer CTGAN or GReaT

IF efficiency_weight > 0.15:
    THEN avoid GReaT (too slow)

IF balanced_preferences (all weights ≈ 0.20):
    THEN prefer CTGAN or Synthpop
```

---

## 7. Scenarios That Would Invalidate Expectations

### 7.1 "Boring Results" Scenarios

**Scenario 1: No Rank Reversals**
- **What:** All archetypes prefer same method (e.g., CTGAN dominates)
- **Implication:** Framework still valid, but tradeoffs less pronounced
- **Paper Impact:** Reframe as "unexpected dominance of one method"

**Scenario 2: All Measures Correlate Highly**
- **What:** All pairwise |r| > 0.85
- **Implication:** Methods lie on single quality dimension
- **Paper Impact:** Still publishable; contribution is systematic framework

**Scenario 3: Low Discrimination**
- **What:** All methods score within ±10% on all measures
- **Implication:** Methods are more similar than expected
- **Paper Impact:** Weaker empirical contribution, but framework remains valid

### 7.2 "Surprising Results" Scenarios

**Scenario 1: Privacy-Utility Positive Correlation**
- **What:** Better privacy correlates with better utility (r > 0.3)
- **Implication:** No tradeoff exists for these methods/data
- **Paper Impact:** Interesting finding; discuss why tradeoff doesn't hold

**Scenario 2: GReaT Outperforms on Privacy**
- **What:** GReaT has best privacy scores despite being LLM
- **Implication:** Our intuition about memorization is wrong
- **Paper Impact:** Investigate why; report as unexpected finding

**Scenario 3: DP-BN Achieves Best Utility**
- **What:** Formal DP doesn't hurt utility (or even helps)
- **Implication:** DP noise acts as regularizer
- **Paper Impact:** Very interesting finding; emphasize in discussion

---

## 8. Validation Pilot Results

**Date:** 2026-01-12

**Privacy Measure Selected:**
- [X] DCR 5th Percentile (primary)
- [ ] Membership Inference (attack AUC = 0.560 < 0.6 threshold)

**N=5K Pilot Findings:**
- [X] DCR discriminates adequately between methods.
- [X] Membership Inference Attack is NOT valid (AUC < 0.6), confirming the need to switch the primary privacy metric.
- [X] All methods and measures in the pilot run successfully.
- [X] No other critical methodological issues were detected.

**Decision:**
A **GO** decision was made to proceed with the full N=35K experiments.

**Adjustments Made Based on Pilot:**
1.  **Use DCR as primary privacy measure:** The validation pilot confirmed that the membership inference attack is too weak to be a reliable measure for this dataset. DCR will be used as the primary privacy metric for the main experiments, as anticipated.
2.  **Proceed with confidence:** The pilot was successful, and the core components of the experimental pipeline are validated.

---

## 9. Post-Experiment Validation

**To be completed after full N=35K experiments:**

### 9.1 Hypothesis Testing Results

| Hypothesis | Expected | Observed | Match? | Notes |
|------------|----------|----------|--------|-------|
| H1: Rank reversals | ≥2 reversals | [TBD] | [TBD] | |
| H2: No dominance | No Pareto-optimal | [TBD] | [TBD] | |
| H3: Privacy-utility tradeoff | r ≈ -0.5 | [TBD] | [TBD] | |
| H4: Efficiency-quality tradeoff | r ≈ -0.3 | [TBD] | [TBD] | |
| H5: Method stability | CV < 0.3 | [TBD] | [TBD] | |

### 9.2 Unexpected Findings

List any results that deviate substantially from pre-registered expectations:

1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

For each unexpected finding:
- Describe the deviation
- Propose explanation
- Discuss implications
- Flag as exploratory (not confirmatory)

---

## 10. Document Lock & Audit Trail

**Pre-Registration Locked:** 2026-01-12 21:30:00 PST

**Locked By:** Michael Koo

**Git Commit Hash:** `29338c495cc4a5cdccf711e646702dfee6b911b5`

**Commit Date:** 2026-01-23

**Commit Message:** "Pre-registration: Lock hypotheses before full analysis - includes pre_registration.md with expected rankings and measures"

**Verification Command:** `git log --oneline 29338c4 -1`

**Subsequent Changes:**
All changes after lock must be documented here with justification:

| Date | Change | Reason |
|------|--------|--------|
| 2026-01-23 | Added git commit hash | Initial repo setup with pre-registration lock |

---

## References

- Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. PNAS, 115(11), 2600-2606.
- Zhang, J., Cormode, G., Procopiuc, C. M., Srivastava, D., & Xiao, X. (2021). PrivBayes: Private Data Release via Bayesian Networks. ACM TODS, 42(4), 1-41.
- Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling Tabular data using Conditional GAN. NeurIPS 2019.

---

**Document Status:** 🔒 LOCKED - Pre-registration complete
**Locked Date:** 2026-01-12 21:30:00 PST
**Next Review:** After full N=35K experiments complete
