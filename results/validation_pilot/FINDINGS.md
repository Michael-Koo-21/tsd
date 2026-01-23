# Validation Pilot Key Findings

**Date:** 2026-01-12
**Sample Size:** N=5K

## Unexpected Finding: CTGAN Fidelity

**Observation:** CTGAN has worse propensity AUC (0.920) than Independent Marginals (0.865).
This is counterintuitive since GANs should preserve distributions better.

**Likely Explanations:**
1. Small sample size - CTGAN needs more data (N=35K should improve this)
2. Insufficient training - 100 epochs may not be enough for convergence
3. Mode collapse - GAN training instability at small N

**Implication for Full Experiments:**
- This suggests discrimination will INCREASE at full scale (good for framework)
- CTGAN performance should improve with N=35K and more epochs
- Document this as exploratory finding in pre-registration

**Action:** Proceed to full experiments. If CTGAN still underperforms on fidelity,
this is an interesting finding worth discussing in paper.