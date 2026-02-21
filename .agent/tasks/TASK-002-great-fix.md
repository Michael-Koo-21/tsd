# TASK-002: GReaT Generator Seeding Fix

**Status**: COMPLETED
**Date**: 2026-01-17
**Duration**: 1 session

---

## Problem

GReaT synthetic data files had **identical values across all 5 replicates**:

```
MD5 (great_rep1.csv) = a02dbd21ebb3962a206eaaf42d09fd11
MD5 (great_rep2.csv) = a02dbd21ebb3962a206eaaf42d09fd11  # SAME!
MD5 (great_rep3.csv) = a02dbd21ebb3962a206eaaf42d09fd11  # SAME!
MD5 (great_rep4.csv) = a02dbd21ebb3962a206eaaf42d09fd11  # SAME!
MD5 (great_rep5.csv) = a02dbd21ebb3962a206eaaf42d09fd11  # SAME!
```

This caused statistical analysis to show zero variance for GReaT method.

---

## Root Cause

The original Colab notebook defined seeds but **never used them**:

```python
# BUGGY CODE
for rep, seed in enumerate(seeds, 1):
    model = GReaT(
        llm='distilgpt2',
        epochs=100,
        batch_size=32
        # seed parameter MISSING!
    )
```

The GReaT library requires:
1. `seed` parameter in constructor (passed to HuggingFace TrainingArguments)
2. Manual seeding of `torch`, `numpy`, and `random` before training/generation

---

## Solution

### 1. Created Local Generator Wrapper

**File**: `tsd/generators/great_generator.py`

```python
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

# Usage
gen = GReaTGenerator(llm='distilgpt2', epochs=100, random_state=42)
gen.fit(df_train)
df_synth = gen.generate(n_samples=1000)
```

### 2. Fixed Colab Notebook

**File**: `tsd/generators/great_colab_notebook_fixed.ipynb`

Changes:
- Added `set_all_seeds(seed)` before each training run
- Pass `seed` and `data_seed` to GReaT constructor
- Set seeds again before generation (with offset)
- Added disk cleanup after each replicate (deleted checkpoints)
- Added hash verification to confirm files are different

### 3. Updated Generators Module

**File**: `tsd/generators/__init__.py`

Added exports:
- `GReaTGenerator`
- `generate_great`

---

## Results After Fix

All 5 files now have **different hashes**:

```
MD5 (great_rep1.csv) = bebae9fc6be9abcd3e33754b50489891
MD5 (great_rep2.csv) = a45b04bf9865fb0fe4e6ed67603bb34c
MD5 (great_rep3.csv) = 019272d5c1788969c936e7b05cb027e1
MD5 (great_rep4.csv) = f4278cadcdf1da1e523055c790d55855
MD5 (great_rep5.csv) = b32517e212fec7d1d4603f4bc0f747d3
```

### Updated GReaT Metrics

| Metric | OLD (constant) | NEW (mean ± std) |
|--------|---------------|------------------|
| Fidelity | 0.748 ± 0.000 | 0.758 ± 0.008 |
| Privacy DCR | 0.122 ± 0.000 | 0.150 ± 0.007 |
| Utility TSTR | 0.957 ± 0.000 | 0.964 ± 0.015 |
| Fairness Gap | 0.062 ± 0.000 | 0.055 ± 0.017 |

### GReaT Method Profile

- **Strength**: Excellent utility (2nd best at 0.96, behind only Synthpop)
- **Weakness**: Lower fidelity (0.76) and privacy (0.15) than deep learning methods
- **Use case**: Utility-focused applications where privacy is less critical

---

## Files Created/Modified

| File | Action |
|------|--------|
| `tsd/generators/great_generator.py` | NEW - Local wrapper with seeding |
| `tsd/generators/great_colab_notebook_fixed.ipynb` | NEW - Fixed notebook |
| `tsd/generators/__init__.py` | UPDATED - Added GReaT exports |
| `results/experiments/synthetic_data/great_rep*.csv` | REPLACED - New data |
| `results/experiments/all_results_complete.csv` | UPDATED - New GReaT measures |
| `results/experiments/statistical_analysis_report.txt` | UPDATED |
| `results/experiments/figures/*.png` | UPDATED - All visualizations |

---

## Colab Disk Space Issue

Initial fix attempt hit disk space limits after 3 replicates. Solution:

```python
def cleanup_disk():
    # Delete checkpoint directories
    for d in os.listdir('.'):
        if d.startswith('great_rep') and os.path.isdir(d):
            shutil.rmtree(d)

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache
    torch.cuda.empty_cache()
```

---

## Lessons Learned

1. **Always verify variance** - Zero variance in replicates indicates a seeding bug
2. **Check file hashes** - Quick way to detect identical outputs
3. **Seed everything** - PyTorch models need `torch.manual_seed()` AND library-specific seeds
4. **Clean up checkpoints** - Large model checkpoints can fill Colab disk quickly

---

## References

- be-great library: https://github.com/kathrinse/be_great
- HuggingFace TrainingArguments: https://huggingface.co/docs/transformers/main_classes/trainer
