"""
Pilot Test Script

Run end-to-end pipeline on small sample (1K records) to validate:
1. Data preprocessing
2. Synthetic data generation (Independent Marginals + CTGAN)
3. Evaluation measures (Propensity AUC + Membership Inference)
4. Results storage

This pilot validates the pipeline before running full experiments.
"""

import pandas as pd
import numpy as np
import sys
import os
import time
import multiprocessing
from pathlib import Path

# Fix for Python 3.13 on macOS multiprocessing issue
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.load_data import preprocess_acs_data
from src.generators.independent_marginals import generate_independent_marginals
from src.generators.ctgan_generator import generate_ctgan
from src.measures.propensity_auc import propensity_auc
from src.measures.membership_inference import membership_inference_privacy


def run_pilot_test(
    sample_size: int = 1000,
    ctgan_epochs: int = 50,
    random_state: int = 42,
    output_dir: str = "results/pilot"
):
    """
    Run pilot test with 2 methods on small sample.

    Args:
        sample_size: Number of records to sample from ACS PUMS
        ctgan_epochs: Number of CTGAN training epochs (reduced for pilot)
        random_state: Random seed
        output_dir: Directory to save results
    """
    print("="*80)
    print("PILOT TEST - SYNTHETIC DATA GENERATION PIPELINE")
    print("="*80)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ===== Phase 1: Data Preprocessing =====
    print("\n[Phase 1] Data Preprocessing")
    print("-"*80)

    df_train, df_test = preprocess_acs_data(
        sample_size=sample_size,
        train_size=0.7,
        random_state=random_state
    )

    print(f"\nFinal data splits:")
    print(f"  Training: {len(df_train):,} records")
    print(f"  Test: {len(df_test):,} records")

    # Save preprocessed data
    df_train.to_csv(f"{output_dir}/train_data.csv", index=False)
    df_test.to_csv(f"{output_dir}/test_data.csv", index=False)

    # ===== Phase 2: Synthetic Data Generation =====
    print("\n[Phase 2] Synthetic Data Generation")
    print("-"*80)

    results = []

    # Method 1: Independent Marginals
    print("\n>>> Method 1: Independent Marginals")
    start_time = time.time()

    df_synth_indep = generate_independent_marginals(
        df_train=df_train,
        n_samples=len(df_train),
        random_state=random_state
    )

    time_indep = time.time() - start_time

    df_synth_indep.to_csv(f"{output_dir}/synthetic_independent_marginals.csv", index=False)
    print(f"✓ Generation time: {time_indep:.2f} seconds")

    # Method 2: CTGAN
    print("\n>>> Method 2: CTGAN")
    start_time = time.time()

    df_synth_ctgan = generate_ctgan(
        df_train=df_train,
        n_samples=len(df_train),
        epochs=ctgan_epochs,
        batch_size=100,
        verbose=False,
        random_state=random_state
    )

    time_ctgan = time.time() - start_time

    df_synth_ctgan.to_csv(f"{output_dir}/synthetic_ctgan.csv", index=False)
    print(f"✓ Generation time: {time_ctgan:.2f} seconds")

    # ===== Phase 3: Evaluation =====
    print("\n[Phase 3] Evaluation")
    print("-"*80)

    # Evaluate Independent Marginals
    print("\n>>> Evaluating Independent Marginals")

    # Fidelity: Propensity AUC
    fidelity_indep = propensity_auc(df_train, df_synth_indep, random_state=random_state)
    print(f"  Fidelity (Propensity AUC): {fidelity_indep['auc']:.4f} (score: {fidelity_indep['auc_fidelity']:.4f})")

    # Privacy: Membership Inference
    privacy_indep = membership_inference_privacy(
        df_real_train=df_train,
        df_real_test=df_test,
        df_synthetic=df_synth_indep,
        random_state=random_state
    )
    print(f"  Privacy (Attack Success): {privacy_indep['attack_success_rate']:.4f} (score: {privacy_indep['privacy_score']:.4f})")

    results.append({
        'method': 'Independent Marginals',
        'generation_time_sec': time_indep,
        'fidelity_auc': fidelity_indep['auc'],
        'fidelity_score': fidelity_indep['auc_fidelity'],
        'privacy_attack_success': privacy_indep['attack_success_rate'],
        'privacy_score': privacy_indep['privacy_score'],
        'privacy_attack_auc': privacy_indep['attack_auc']
    })

    # Evaluate CTGAN
    print("\n>>> Evaluating CTGAN")

    # Fidelity: Propensity AUC
    fidelity_ctgan = propensity_auc(df_train, df_synth_ctgan, random_state=random_state)
    print(f"  Fidelity (Propensity AUC): {fidelity_ctgan['auc']:.4f} (score: {fidelity_ctgan['auc_fidelity']:.4f})")

    # Privacy: Membership Inference
    privacy_ctgan = membership_inference_privacy(
        df_real_train=df_train,
        df_real_test=df_test,
        df_synthetic=df_synth_ctgan,
        random_state=random_state
    )
    print(f"  Privacy (Attack Success): {privacy_ctgan['attack_success_rate']:.4f} (score: {privacy_ctgan['privacy_score']:.4f})")

    results.append({
        'method': 'CTGAN',
        'generation_time_sec': time_ctgan,
        'fidelity_auc': fidelity_ctgan['auc'],
        'fidelity_score': fidelity_ctgan['auc_fidelity'],
        'privacy_attack_success': privacy_ctgan['attack_success_rate'],
        'privacy_score': privacy_ctgan['privacy_score'],
        'privacy_attack_auc': privacy_ctgan['attack_auc']
    })

    # ===== Phase 4: Results Summary =====
    print("\n[Phase 4] Results Summary")
    print("-"*80)

    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False))

    # Save results
    df_results.to_csv(f"{output_dir}/pilot_results.csv", index=False)

    print(f"\n✓ All results saved to {output_dir}/")

    # ===== Validation Checks =====
    print("\n[Validation Checks]")
    print("-"*80)

    checks_passed = []
    checks_failed = []

    # Check 1: Synthetic data has correct schema
    for method_name, df_synth in [("IndepMarg", df_synth_indep), ("CTGAN", df_synth_ctgan)]:
        if set(df_synth.columns) == set(df_train.columns):
            checks_passed.append(f"✓ {method_name}: Schema matches training data")
        else:
            checks_failed.append(f"✗ {method_name}: Schema mismatch")

    # Check 2: Synthetic data has correct size
    for method_name, df_synth in [("IndepMarg", df_synth_indep), ("CTGAN", df_synth_ctgan)]:
        if len(df_synth) == len(df_train):
            checks_passed.append(f"✓ {method_name}: Generated {len(df_synth)} records as expected")
        else:
            checks_failed.append(f"✗ {method_name}: Expected {len(df_train)}, got {len(df_synth)}")

    # Check 3: Fidelity scores are computed
    for result in results:
        if 0 <= result['fidelity_score'] <= 1:
            checks_passed.append(f"✓ {result['method']}: Fidelity score in valid range [0,1]")
        else:
            checks_failed.append(f"✗ {result['method']}: Invalid fidelity score")

    # Check 4: Privacy scores are computed
    for result in results:
        if 0 <= result['privacy_score'] <= 1:
            checks_passed.append(f"✓ {result['method']}: Privacy score in valid range [0,1]")
        else:
            checks_failed.append(f"✗ {result['method']}: Invalid privacy score")

    print("\nPassed checks:")
    for check in checks_passed:
        print(f"  {check}")

    if checks_failed:
        print("\nFailed checks:")
        for check in checks_failed:
            print(f"  {check}")

    # Final status
    print("\n" + "="*80)
    if not checks_failed:
        print("✓ PILOT TEST PASSED - Pipeline is working correctly!")
    else:
        print(f"⚠ PILOT TEST PARTIAL - {len(checks_failed)} checks failed")
    print("="*80)

    return df_results


if __name__ == "__main__":
    # Run pilot test
    results = run_pilot_test(
        sample_size=1000,
        ctgan_epochs=50,  # Reduced for speed
        random_state=42,
        output_dir="results/pilot"
    )

    print("\nPilot test complete. Review results in results/pilot/")
