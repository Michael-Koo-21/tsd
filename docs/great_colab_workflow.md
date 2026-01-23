# GReaT Colab Batch Workflow

Instructions for running 5 replicates of GReaT on Google Colab.

## Prerequisites

- Google Colab Pro account (or any GPU runtime)
- Training data file: `great_train_data_5k.csv` (3,500 training records)

**IMPORTANT:** All methods in the experiment use n_samples=5000 (70/30 split → n_train=3,500).
GReaT must be trained on the same 3,500 records for fair comparison.

## Step 1: Export Training Data

Run this locally to create the training data file (if not already present):

```bash
cd /Users/michaelkoo/Projects/TrustingSyntheticData
source .venv/bin/activate

# Create training data with n_samples=5000 → n_train=3500
python -c "
from src.preprocessing.load_data import preprocess_acs_data
df_train, df_test = preprocess_acs_data(sample_size=5000)
df_train.to_csv('great_train_data_5k.csv', index=False)
print(f'Saved {len(df_train)} training records to great_train_data_5k.csv')
print(f'This matches n_train for other methods (CTGAN, DPBN, Synthpop)')
"
```

## Step 2: Create Colab Notebook

1. Go to https://colab.research.google.com/
2. Create a new notebook
3. Change runtime to GPU: Runtime → Change runtime type → GPU (T4)

## Step 3: Run This Code in Colab

### Cell 1: Setup
```python
!pip install be-great pandas -q
print("✓ Dependencies installed")
```

### Cell 2: Upload Data
```python
from google.colab import files
uploaded = files.upload()
# Click "Choose Files" and select great_train_data_5k.csv
```

### Cell 3: Generate All Replicates
```python
import pandas as pd
from be_great import GReaT
import warnings
warnings.filterwarnings('ignore')

# Load training data
df_train = pd.read_csv('great_train_data_5k.csv')
print(f"Loaded {len(df_train)} training records")

# Random seeds (must match local experiments)
seeds = [42, 123, 456, 789, 1011]

# Generate 5 replicates
for rep, seed in enumerate(seeds, 1):
    print(f"\n{'='*50}")
    print(f"REPLICATE {rep}/5 (seed={seed})")
    print(f"{'='*50}")

    # Train model
    print("Training GReaT model...")
    model = GReaT(
        llm='distilgpt2',
        epochs=100,
        batch_size=32
    )
    model.fit(df_train)
    print("✓ Training complete")

    # Generate synthetic data
    print(f"Generating {len(df_train)} synthetic records...")
    df_synth = model.sample(n_samples=len(df_train))

    # Save
    filename = f'great_rep{rep}.csv'
    df_synth.to_csv(filename, index=False)
    print(f"✓ Saved to {filename}")

print(f"\n{'='*50}")
print("ALL 5 REPLICATES COMPLETE!")
print(f"{'='*50}")
```

### Cell 4: Download All Files
```python
from google.colab import files

for rep in range(1, 6):
    files.download(f'great_rep{rep}.csv')
    print(f"Downloaded great_rep{rep}.csv")
```

## Step 4: Copy Files to Project

Move downloaded files to the experiments directory:

```bash
# From your Downloads folder
mv great_rep*.csv /Users/michaelkoo/Projects/TrustingSyntheticData/results/experiments/synthetic_data/
```

## Step 5: Compute GReaT Measures

Run locally to compute measures for GReaT:

```bash
cd /Users/michaelkoo/Projects/TrustingSyntheticData
source .venv/bin/activate

python << 'EOF'
import pandas as pd
import sys
sys.path.insert(0, '.')

from src.preprocessing.load_data import preprocess_acs_data
from src.measures import propensity_auc, dcr_privacy, tstr_utility, fairness_gap

# Load data
df_train, df_test = preprocess_acs_data(sample_size=5000)
median_income = df_train['PINCP'].median()

# Prepare TSTR data
df_train_tstr = df_train.copy()
df_test_tstr = df_test.copy()
df_train_tstr['HIGH_INCOME'] = (df_train_tstr['PINCP'] > median_income).astype(int)
df_test_tstr['HIGH_INCOME'] = (df_test_tstr['PINCP'] > median_income).astype(int)
df_train_tstr = df_train_tstr.drop('PINCP', axis=1)
df_test_tstr = df_test_tstr.drop('PINCP', axis=1)

results = []
seeds = [42, 123, 456, 789, 1011]

for rep in range(1, 6):
    print(f"\nProcessing great_rep{rep}...")
    df_synth = pd.read_csv(f'results/experiments/synthetic_data/great_rep{rep}.csv')

    df_synth_tstr = df_synth.copy()
    df_synth_tstr['HIGH_INCOME'] = (df_synth_tstr['PINCP'] > median_income).astype(int)
    df_synth_tstr = df_synth_tstr.drop('PINCP', axis=1)

    fidelity = propensity_auc(df_train, df_synth)['auc']
    dcr = dcr_privacy(df_train, df_synth)['dcr_percentile']
    tstr = tstr_utility(df_train_tstr, df_synth_tstr, df_test_tstr, target_col='HIGH_INCOME')['f1_ratio']
    fair = fairness_gap(df_train_tstr, df_synth_tstr, df_test_tstr, target_col='HIGH_INCOME', subgroup_col='SEX')['max_gap']

    results.append({
        'method': 'great',
        'replicate': rep,
        'seed': seeds[rep-1],
        'fidelity_auc': fidelity,
        'privacy_dcr': dcr,
        'utility_tstr': tstr,
        'fairness_gap': fair
    })
    print(f"  Fidelity: {fidelity:.4f}, DCR: {dcr:.4f}, TSTR: {tstr:.4f}, Fairness: {fair:.4f}")

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv('results/experiments/great_results.csv', index=False)
print("\n✓ Results saved to results/experiments/great_results.csv")

# Merge with main results
df_all = pd.read_csv('results/experiments/all_results.csv')
df_combined = pd.concat([df_all, df_results], ignore_index=True)
df_combined.to_csv('results/experiments/all_results_with_great.csv', index=False)
print("✓ Combined results saved to results/experiments/all_results_with_great.csv")
EOF
```

## Expected Runtime

- Training: ~10-15 min per replicate
- Generation: ~2-3 min per replicate
- Total: ~60-90 min for all 5 replicates

## Troubleshooting

### Colab Disconnects
- Save after each replicate
- Use Colab Pro for longer runtimes
- Run replicates in separate sessions if needed

### Memory Issues
- Restart runtime between replicates
- Reduce batch_size to 16

### Out of GPU
- Check GPU availability in Colab
- Wait a few minutes and retry
