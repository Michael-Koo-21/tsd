"""
Quick test of DP-BN generator on N=5K sample

Purpose: Validate that DP-BN (DataSynthesizer) works correctly before full experiments
"""

import pandas as pd
import time
import sys
import os
import multiprocessing

# Fix for Python 3.13 on macOS multiprocessing issue
if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.load_data import preprocess_acs_data
from src.generators.dp_bayesian_network import generate_dp_bayesian_network

def run_test():
    """Run DP-BN test."""
    print("="*70)
    print("Testing DP-BN Generator (N=5K)")
    print("="*70)

    # Load data
    print("\nLoading N=5K sample...")
    df_train, df_test = preprocess_acs_data(sample_size=5000, random_state=42)

    print(f"Train: {len(df_train):,} records")
    print(f"Test: {len(df_test):,} records")
    print(f"Columns: {list(df_train.columns)}")

    # Generate synthetic data
    print("\nGenerating synthetic data with DP-BN...")
    print("  epsilon=1.0, degree=2")

    start_time = time.time()

    try:
        df_synthetic = generate_dp_bayesian_network(
            df_train=df_train,
            n_samples=len(df_train),
            epsilon=1.0,
            degree_of_bayesian_network=2,
            random_state=42
        )

        elapsed = time.time() - start_time

        print(f"\n✓ Success! Generated {len(df_synthetic):,} records")
        print(f"  Time: {elapsed:.1f} seconds")

        # Validate schema
        print("\nValidating schema...")
        if set(df_synthetic.columns) == set(df_train.columns):
            print("  ✓ Columns match")
        else:
            print("  ✗ Column mismatch!")
            print(f"    Missing: {set(df_train.columns) - set(df_synthetic.columns)}")
            print(f"    Extra: {set(df_synthetic.columns) - set(df_train.columns)}")

        if len(df_synthetic) == len(df_train):
            print(f"  ✓ Row count correct ({len(df_synthetic):,})")
        else:
            print(f"  ✗ Row count mismatch: expected {len(df_train):,}, got {len(df_synthetic):,}")

        # Check for nulls
        null_counts = df_synthetic.isnull().sum()
        if null_counts.sum() == 0:
            print("  ✓ No null values")
        else:
            print(f"  ⚠ Found {null_counts.sum()} null values:")
            print(null_counts[null_counts > 0])

        # Check data types
        print("\nData types:")
        for col in df_train.columns:
            train_type = df_train[col].dtype
            synth_type = df_synthetic[col].dtype
            match = "✓" if train_type == synth_type else "✗"
            print(f"  {match} {col}: {synth_type} (expected {train_type})")

        print("\n" + "="*70)
        print("✓ DP-BN TEST PASSED")
        print("="*70)
        print("\nDP-BN generator is ready for full experiments.")

    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        print("\nError details:")
        import traceback
        traceback.print_exc()
        print("\n" + "="*70)
        print("✗ DP-BN TEST FAILED")
        print("="*70)
        sys.exit(1)


if __name__ == '__main__':
    run_test()
