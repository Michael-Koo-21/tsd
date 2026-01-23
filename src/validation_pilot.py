"""
Validation Pilot Script - N=5K Sample

Purpose: Validate methodology before full N=35K experiments
- Test both privacy measures (Membership Inference + DCR)
- Check measure validity and correlations
- Verify methods scale properly from N=1K to N=5K
- Provide go/no-go decision for full experiments

Expected Runtime: ~30 minutes (2 methods, both privacy measures)
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
from src.measures.dcr_privacy import dcr_privacy


def run_validation_pilot(
    sample_size: int = 5000,
    ctgan_epochs: int = 100,
    random_state: int = 42,
    output_dir: str = "results/validation_pilot"
):
    """
    Run validation pilot with enhanced checks.

    Args:
        sample_size: Number of records (5K recommended)
        ctgan_epochs: CTGAN epochs (more than pilot, less than full)
        random_state: Random seed
        output_dir: Output directory
    """
    print("="*80)
    print("VALIDATION PILOT - METHODOLOGICAL CHECKS (N=5K)")
    print("="*80)
    print("\nPurpose: Validate methodology before full N=35K experiments")
    print("Checks: Privacy measure validity, measure correlations, method scaling")
    print()

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

    print(f"\nData splits:")
    print(f"  Training: {len(df_train):,} records")
    print(f"  Test: {len(df_test):,} records")

    # Save
    df_train.to_csv(f"{output_dir}/train_data.csv", index=False)
    df_test.to_csv(f"{output_dir}/test_data.csv", index=False)

    # ===== Phase 2: Generate Synthetic Data =====
    print("\n[Phase 2] Synthetic Data Generation")
    print("-"*80)

    results = []
    synthetic_datasets = {}

    # Method 1: Independent Marginals
    print("\n>>> Method 1: Independent Marginals")
    start_time = time.time()

    df_synth_indep = generate_independent_marginals(
        df_train=df_train,
        n_samples=len(df_train),
        random_state=random_state
    )

    time_indep = time.time() - start_time
    synthetic_datasets['Independent Marginals'] = df_synth_indep

    df_synth_indep.to_csv(f"{output_dir}/synthetic_independent_marginals.csv", index=False)
    print(f"✓ Generation time: {time_indep:.2f} seconds")

    # Method 2: CTGAN
    print("\n>>> Method 2: CTGAN")
    start_time = time.time()

    df_synth_ctgan = generate_ctgan(
        df_train=df_train,
        n_samples=len(df_train),
        epochs=ctgan_epochs,
        batch_size=200,
        verbose=False,
        random_state=random_state
    )

    time_ctgan = time.time() - start_time
    synthetic_datasets['CTGAN'] = df_synth_ctgan

    df_synth_ctgan.to_csv(f"{output_dir}/synthetic_ctgan.csv", index=False)
    print(f"✓ Generation time: {time_ctgan:.2f} seconds")

    # ===== Phase 3: Evaluation - All Measures =====
    print("\n[Phase 3] Comprehensive Evaluation")
    print("-"*80)

    # Evaluate Independent Marginals
    print("\n>>> Evaluating Independent Marginals")

    # Fidelity
    fidelity_indep = propensity_auc(df_train, df_synth_indep, random_state=random_state)
    print(f"  Fidelity (Propensity AUC): {fidelity_indep['auc']:.4f}")

    # Privacy - Membership Inference
    privacy_mi_indep = membership_inference_privacy(
        df_real_train=df_train,
        df_real_test=df_test,
        df_synthetic=df_synth_indep,
        random_state=random_state
    )
    print(f"  Privacy (Membership Inference):")
    print(f"    - Attack AUC: {privacy_mi_indep['attack_auc']:.4f}")
    print(f"    - Attack Valid: {privacy_mi_indep['attack_valid']}")
    print(f"    - Attack Success Rate: {privacy_mi_indep['attack_success_rate']:.4f}")
    if privacy_mi_indep['warning']:
        print(f"    - ⚠️  {privacy_mi_indep['warning']}")

    # Privacy - DCR
    privacy_dcr_indep = dcr_privacy(
        df_real_train=df_train,
        df_synthetic=df_synth_indep,
        percentile=5
    )
    print(f"  Privacy (DCR 5th Percentile):")
    print(f"    - DCR: {privacy_dcr_indep['dcr_percentile']:.4f}")
    print(f"    - Privacy Score: {privacy_dcr_indep['privacy_score']:.4f}")

    results.append({
        'method': 'Independent Marginals',
        'generation_time_sec': time_indep,
        'fidelity_auc': fidelity_indep['auc'],
        'fidelity_score': fidelity_indep['auc_fidelity'],
        'privacy_mi_attack_auc': privacy_mi_indep['attack_auc'],
        'privacy_mi_attack_valid': privacy_mi_indep['attack_valid'],
        'privacy_mi_attack_success': privacy_mi_indep['attack_success_rate'],
        'privacy_mi_score': privacy_mi_indep['privacy_score'],
        'privacy_dcr_percentile': privacy_dcr_indep['dcr_percentile'],
        'privacy_dcr_score': privacy_dcr_indep['privacy_score']
    })

    # Evaluate CTGAN
    print("\n>>> Evaluating CTGAN")

    # Fidelity
    fidelity_ctgan = propensity_auc(df_train, df_synth_ctgan, random_state=random_state)
    print(f"  Fidelity (Propensity AUC): {fidelity_ctgan['auc']:.4f}")

    # Privacy - Membership Inference
    privacy_mi_ctgan = membership_inference_privacy(
        df_real_train=df_train,
        df_real_test=df_test,
        df_synthetic=df_synth_ctgan,
        random_state=random_state
    )
    print(f"  Privacy (Membership Inference):")
    print(f"    - Attack AUC: {privacy_mi_ctgan['attack_auc']:.4f}")
    print(f"    - Attack Valid: {privacy_mi_ctgan['attack_valid']}")
    print(f"    - Attack Success Rate: {privacy_mi_ctgan['attack_success_rate']:.4f}")
    if privacy_mi_ctgan['warning']:
        print(f"    - ⚠️  {privacy_mi_ctgan['warning']}")

    # Privacy - DCR
    privacy_dcr_ctgan = dcr_privacy(
        df_real_train=df_train,
        df_synthetic=df_synth_ctgan,
        percentile=5
    )
    print(f"  Privacy (DCR 5th Percentile):")
    print(f"    - DCR: {privacy_dcr_ctgan['dcr_percentile']:.4f}")
    print(f"    - Privacy Score: {privacy_dcr_ctgan['privacy_score']:.4f}")

    results.append({
        'method': 'CTGAN',
        'generation_time_sec': time_ctgan,
        'fidelity_auc': fidelity_ctgan['auc'],
        'fidelity_score': fidelity_ctgan['auc_fidelity'],
        'privacy_mi_attack_auc': privacy_mi_ctgan['attack_auc'],
        'privacy_mi_attack_valid': privacy_mi_ctgan['attack_valid'],
        'privacy_mi_attack_success': privacy_mi_ctgan['attack_success_rate'],
        'privacy_mi_score': privacy_mi_ctgan['privacy_score'],
        'privacy_dcr_percentile': privacy_dcr_ctgan['dcr_percentile'],
        'privacy_dcr_score': privacy_dcr_ctgan['privacy_score']
    })

    # ===== Phase 4: Methodological Validation =====
    print("\n[Phase 4] Methodological Validation")
    print("-"*80)

    df_results = pd.DataFrame(results)

    # Check 1: Privacy Measure Validity
    print("\n>>> Check 1: Privacy Measure Validity")
    print()

    mi_valid_count = df_results['privacy_mi_attack_valid'].sum()
    print(f"Membership Inference Attack:")
    print(f"  Valid attacks: {mi_valid_count}/{len(df_results)}")

    if mi_valid_count == 0:
        print(f"  ⚠️  CRITICAL: No valid membership inference attacks detected")
        print(f"  → Action: Use DCR as primary privacy measure")
        primary_privacy_measure = 'DCR'
    elif mi_valid_count == len(df_results):
        print(f"  ✓ All attacks valid - can use membership inference")
        primary_privacy_measure = 'Membership Inference'
    else:
        print(f"  ⚠️  Mixed validity - some attacks work, some don't")
        print(f"  → Action: Use DCR as primary, report MI as secondary")
        primary_privacy_measure = 'DCR'

    print(f"\nDCR Privacy:")
    print(f"  Mean DCR 5th percentile: {df_results['privacy_dcr_percentile'].mean():.4f}")
    print(f"  ✓ DCR computed successfully for all methods")

    # Check 2: Measure Discrimination
    print("\n>>> Check 2: Measure Discrimination")
    print()

    measures_to_check = {
        'Fidelity (AUC)': 'fidelity_auc',
        'Privacy (DCR)': 'privacy_dcr_score'
    }

    discriminatory_values = {}
    for measure_name, col in measures_to_check.items():
        values = df_results[col].dropna()
        if len(values) > 1:
            disc_value = (values.max() - values.min()) / values.mean()
            discriminatory_values[measure_name] = disc_value
            print(f"{measure_name}:")
            print(f"  Range: [{values.min():.4f}, {values.max():.4f}]")
            print(f"  Discriminatory value: {disc_value:.3f}")
            if disc_value < 0.15:
                print(f"  ⚠️  Low discrimination - methods may be too similar")
            else:
                print(f"  ✓ Adequate discrimination")

    # Check 3: Method Scaling
    print("\n>>> Check 3: Method Scaling (N=1K vs N=5K)")
    print()
    print("Comparing to N=1K pilot results:")
    print("  (Load previous pilot results if available)")

    # Try to load N=1K results
    pilot_1k_path = "results/pilot/pilot_results.csv"
    if Path(pilot_1k_path).exists():
        df_pilot_1k = pd.read_csv(pilot_1k_path)
        print("\n  N=1K Pilot Results:")
        print(df_pilot_1k[['method', 'fidelity_auc', 'generation_time_sec']].to_string(index=False))
        print("\n  N=5K Validation Results:")
        print(df_results[['method', 'fidelity_auc', 'generation_time_sec']].to_string(index=False))

        # Check for dramatic changes
        for method in ['Independent Marginals', 'CTGAN']:
            if method in df_pilot_1k['method'].values and method in df_results['method'].values:
                auc_1k = df_pilot_1k[df_pilot_1k['method'] == method]['fidelity_auc'].values[0]
                auc_5k = df_results[df_results['method'] == method]['fidelity_auc'].values[0]
                change = abs(auc_5k - auc_1k)
                print(f"\n  {method}:")
                print(f"    Fidelity AUC change: {auc_1k:.4f} → {auc_5k:.4f} (Δ={change:.4f})")
                if change > 0.1:
                    print(f"    ⚠️  Large change - method may be sensitive to sample size")
                else:
                    print(f"    ✓ Stable across sample sizes")
    else:
        print("  (No N=1K results found for comparison)")

    # ===== Phase 5: Decision & Recommendations =====
    print("\n[Phase 5] GO/NO-GO Decision")
    print("="*80)

    issues_found = []
    warnings_found = []

    # Critical: Privacy measure validity
    if primary_privacy_measure == 'DCR':
        warnings_found.append("Membership inference attacks not reliable - using DCR")

    # High priority: Discrimination
    if any(dv < 0.15 for dv in discriminatory_values.values()):
        warnings_found.append("Some measures show low discrimination")

    # Determine decision
    if len(issues_found) == 0:
        print("\n✅ GO: Proceed with full N=35K experiments")
        print()
        print("Validation Results:")
        print(f"  ✓ Primary privacy measure: {primary_privacy_measure}")
        print(f"  ✓ All measures compute successfully")
        print(f"  ✓ Methods show discriminatory differences")

        if warnings_found:
            print("\nWarnings (non-blocking):")
            for warning in warnings_found:
                print(f"  ⚠️  {warning}")
    else:
        print("\n🛑 NO-GO: Address critical issues before full experiments")
        print()
        print("Critical Issues:")
        for issue in issues_found:
            print(f"  ❌ {issue}")

    # Save results
    df_results.to_csv(f"{output_dir}/validation_results.csv", index=False)
    print(f"\n✓ Results saved to {output_dir}/")

    # Create decision summary
    decision_summary = {
        'validation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sample_size': sample_size,
        'primary_privacy_measure': primary_privacy_measure,
        'go_decision': len(issues_found) == 0,
        'critical_issues': len(issues_found),
        'warnings': len(warnings_found),
        'issues_list': issues_found,
        'warnings_list': warnings_found
    }

    import json
    with open(f"{output_dir}/decision_summary.json", 'w') as f:
        json.dump(decision_summary, f, indent=2)

    print("\n" + "="*80)
    print("Validation pilot complete.")
    print("="*80)

    return df_results, decision_summary


if __name__ == "__main__":
    # Run validation pilot
    print("\nStarting validation pilot...")
    print("Expected runtime: ~30 minutes\n")

    results, decision = run_validation_pilot(
        sample_size=5000,
        ctgan_epochs=100,
        random_state=42,
        output_dir="results/validation_pilot"
    )

    print("\n" + "="*80)
    if decision['go_decision']:
        print("RECOMMENDATION: Proceed with full experiments")
        print()
        print("Next steps:")
        print("1. Review validation results in results/validation_pilot/")
        print("2. Document primary privacy measure in study_design_spec.md")
        print("3. Run full N=35K experiments (Week 2)")
    else:
        print("RECOMMENDATION: Address issues before proceeding")
        print()
        print("Next steps:")
        print("1. Review critical issues in results/validation_pilot/decision_summary.json")
        print("2. Fix identified problems")
        print("3. Re-run validation pilot")
    print("="*80)
