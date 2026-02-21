# Validation Pilot Checklist
**Date:** 2026-01-12
**Purpose:** Step-by-step guide for running validation pilot before full experiments

---

## Quick Summary

**Time Required:** ~1 hour (mostly computation)
**Purpose:** Validate privacy measures work correctly before investing 2-3 weeks in full experiments
**Output:** Go/No-Go decision for full N=35K experiments

---

## What We Fixed

✅ **DCR Privacy Measure** - Implemented as backup for membership inference
✅ **Membership Inference Validity Checks** - Now warns if attack is too weak
✅ **Validation Pilot Script** - Tests both measures on N=5K sample
✅ **Pre-Registration Document** - Records expected results before experiments

---

## Step-by-Step Instructions

### Step 1: Review What Was Created (5 min)

New files created:
- `tsd/measures/dcr_privacy.py` - Distance to Closest Record privacy measure
- `tsd/measures/membership_inference.py` - Updated with validity checks
- `tsd/validation_pilot.py` - N=5K validation script
- `docs/pre_registration.md` - Expected results (to prevent HARKing)
- `docs/methodological_validation.md` - Full methodological review

**Action:** Skim through these files to understand what was added.

### Step 2: Test DCR Measure (Optional, 5 min)

Verify DCR implementation works:

```bash
cd /Users/michaelkoo/Projects/TrustingSyntheticData
python tsd/measures/dcr_privacy.py
```

**Expected:** Should compute DCR for Independent Marginals and CTGAN on N=2K sample.

### Step 3: Run N=5K Validation Pilot (30-45 min)

Run the validation pilot:

```bash
cd /Users/michaelkoo/Projects/TrustingSyntheticData
python tsd/validation_pilot.py
```

**What it does:**
1. Loads N=5K sample from ACS PUMS
2. Generates synthetic data (Independent Marginals + CTGAN)
3. Computes all measures (Fidelity, Privacy MI, Privacy DCR)
4. Checks validity and discrimination
5. Provides GO/NO-GO decision

**Expected output:**
```
[Phase 1] Data Preprocessing
...
[Phase 2] Synthetic Data Generation
...
[Phase 3] Comprehensive Evaluation
...
[Phase 4] Methodological Validation
>>> Check 1: Privacy Measure Validity
>>> Check 2: Measure Discrimination
>>> Check 3: Method Scaling
...
[Phase 5] GO/NO-GO Decision
✅ GO: Proceed with full N=35K experiments
```

### Step 4: Review Results (10 min)

Check output files in `results/validation_pilot/`:
- `validation_results.csv` - All measure values
- `decision_summary.json` - GO/NO-GO decision with reasons

**Key things to check:**

1. **Primary Privacy Measure:**
   - If `"primary_privacy_measure": "Membership Inference"` → Great! Attack works.
   - If `"primary_privacy_measure": "DCR"` → Expected. Use DCR going forward.

2. **GO Decision:**
   - If `"go_decision": true` → Proceed to full experiments
   - If `"go_decision": false` → Fix issues first

3. **Measure Discrimination:**
   - Check that fidelity and privacy discriminate between methods (not all identical)

### Step 5: Lock Pre-Registration (2 min)

After validation pilot succeeds:

```bash
cd /Users/michaelkoo/Projects/TrustingSyntheticData

# Update pre-registration with validation results
# Add git commit to lock it
git add docs/pre_registration.md
git commit -m "Lock pre-registration after validation pilot"

# Record commit hash in pre-registration.md Section 10
git log -1 --format="%H"
```

This creates audit trail proving you documented expectations before seeing results.

### Step 6: Update Study Design (5 min)

Based on validation results, update `docs/study_design_spec.md`:

**If using DCR as primary:**
Update Section 4.2.2:
```markdown
### 4.2.2 Privacy: DCR 5th Percentile (PRIMARY)

**Definition:** Distance from each synthetic record to its nearest training record.
Lower DCR indicates potential memorization risk.

**Note:** Membership inference was tested but attack classifier achieved AUC < 0.6
on real data (too weak to be valid). DCR is used as primary privacy measure.

[Keep existing secondary measure note about membership inference]
```

**If using Membership Inference as primary:**
No changes needed - proceed as originally specified.

---

## Expected Outcomes

### Scenario A: GO Decision (Expected)

**What you'll see:**
```
✅ GO: Proceed with full N=35K experiments

Validation Results:
  ✓ Primary privacy measure: DCR
  ✓ All measures compute successfully
  ✓ Methods show discriminatory differences
```

**Next steps:**
1. ✅ Lock pre-registration
2. ✅ Update study_design_spec.md if needed
3. ✅ Proceed to full experiments (Week 2 plan)

### Scenario B: NO-GO Decision (Unlikely)

**What you'll see:**
```
🛑 NO-GO: Address critical issues before full experiments

Critical Issues:
  ❌ [Issue description]
```

**Next steps:**
1. Read `results/validation_pilot/decision_summary.json` for details
2. Fix identified issues
3. Re-run validation pilot
4. Contact Claude for help debugging

---

## Decision Matrix

| Finding | Action | Impact on Timeline |
|---------|--------|-------------------|
| DCR works, MI doesn't | Use DCR as primary | No delay |
| Both work | Use MI as primary, DCR as secondary | No delay |
| Neither works | Debug privacy measures | +2-3 days |
| Low discrimination | Add more diverse method OR proceed | +1 week OR no delay |
| Methods identical | Investigate / proceed with framework focus | No delay |

---

## What Happens After GO Decision

Once validation pilot passes:

### Immediate (Week 2):
1. Implement remaining generators:
   - Synthpop (R wrapper)
   - GReaT (via Google Colab Pro)

2. Implement remaining measures:
   - TSTR utility (F1 ratio)
   - Fairness (subgroup gap)

3. Run full experiments:
   - 5 methods × 5 replicates = 25 datasets
   - Expected runtime: 20-35 hours

### Week 3:
1. Implement value functions
2. MAUT aggregation
3. Sensitivity analysis

### Week 4+:
1. VOI analysis
2. Generate figures/tables
3. Write paper

---

## Troubleshooting

### Problem: DCR takes too long on N=5K

**Solution:** Already batched - should complete in ~1-2 minutes per method

### Problem: CTGAN fails with memory error

**Solution:** Reduce batch_size in validation_pilot.py:
```python
df_synth_ctgan = generate_ctgan(
    df_train=df_train,
    n_samples=len(df_train),
    epochs=100,
    batch_size=100,  # Reduce from 200
    ...
)
```

### Problem: Validation pilot crashes

**Solution:** Check Python environment:
```bash
python --version  # Should be 3.13
pip list | grep -E "(scipy|pandas|sklearn)"
```

If missing scipy:
```bash
pip install scipy
```

### Problem: Both privacy measures show warning

**Rare scenario - likely need N=35K for stable results.**

**Solution:**
1. Check `results/validation_pilot/validation_results.csv`
2. If DCR values look reasonable (not all 0 or NaN), proceed
3. Note in pre-registration that privacy discrimination is lower than expected
4. Document as limitation in paper

---

## Contact / Support

If you encounter issues:

1. **First:** Check `results/validation_pilot/decision_summary.json` for specific error messages
2. **Second:** Review `docs/methodological_validation.md` Section matching your issue
3. **Third:** Ask Claude for help with specific error message

---

## Summary: What You Need to Do NOW

**Minimum viable validation (45 min):**
1. ✅ Run `python tsd/validation_pilot.py`
2. ✅ Review `results/validation_pilot/decision_summary.json`
3. ✅ If GO decision: Lock pre-registration and proceed to Week 2

**That's it!** Everything else is already implemented and ready.

---

**Document Created:** 2026-01-12
**Last Updated:** 2026-01-12
**Status:** Ready to execute
