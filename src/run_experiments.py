#!/usr/bin/env python3
"""
Experiment Runner for Synthetic Data Generation Study

Runs all generators and measures across multiple replicates.
Supports checkpointing and result aggregation.

Usage:
    python src/run_experiments.py --n-samples 35000 --n-replicates 5
    python src/run_experiments.py --n-samples 1000 --n-replicates 1 --methods independent_marginals  # Test run
"""

import argparse
import pandas as pd
import numpy as np
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import generators
from src.generators.independent_marginals import generate_independent_marginals
from src.generators.ctgan_generator import generate_ctgan
from src.generators.dp_bayesian_network import generate_dp_bayesian_network
from src.generators.synthpop_wrapper import generate_synthpop

# Import measures
from src.measures import (
    propensity_auc,
    dcr_privacy,
    membership_inference_privacy,
    tstr_utility,
    fairness_gap
)

# Import data loader
from src.preprocessing.load_data import preprocess_acs_data

# Configuration
RANDOM_SEEDS = [42, 123, 456, 789, 1011]
RESULTS_DIR = Path("results/experiments")
SYNTHETIC_DIR = RESULTS_DIR / "synthetic_data"
MEASURES_DIR = RESULTS_DIR / "measures"

# Generator functions mapping
GENERATORS = {
    'independent_marginals': lambda df, n, seed: generate_independent_marginals(df, n, random_state=seed),
    'ctgan': lambda df, n, seed: generate_ctgan(df, n, epochs=300, random_state=seed),
    'dpbn': lambda df, n, seed: generate_dp_bayesian_network(df, n, epsilon=1.0, random_state=seed),
    'synthpop': lambda df, n, seed: generate_synthpop(df, n, random_state=seed),
}


def load_checkpoint(checkpoint_file: Path) -> dict:
    """Load checkpoint if it exists."""
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {'completed': [], 'results': []}


def save_checkpoint(checkpoint_file: Path, checkpoint: dict):
    """Save checkpoint."""
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def compute_all_measures(df_train, df_test, df_synth, median_income):
    """Compute all measures for a synthetic dataset.
    
    Privacy Measure Note:
    DCR (Distance to Closest Record) is used as the PRIMARY privacy measure.
    Membership inference is computed but typically invalid for this data
    (attack AUC < 0.6 threshold). See methodological_validation.md for details.
    """
    results = {}

    # Prepare TSTR data (binarize PINCP)
    df_train_tstr = df_train.copy()
    df_test_tstr = df_test.copy()
    df_synth_tstr = df_synth.copy()

    df_train_tstr['HIGH_INCOME'] = (df_train_tstr['PINCP'] > median_income).astype(int)
    df_test_tstr['HIGH_INCOME'] = (df_test_tstr['PINCP'] > median_income).astype(int)
    df_synth_tstr['HIGH_INCOME'] = (df_synth_tstr['PINCP'] > median_income).astype(int)

    df_train_tstr = df_train_tstr.drop('PINCP', axis=1)
    df_test_tstr = df_test_tstr.drop('PINCP', axis=1)
    df_synth_tstr = df_synth_tstr.drop('PINCP', axis=1)

    # 1. Propensity AUC (Fidelity)
    try:
        result = propensity_auc(df_train, df_synth)
        results['fidelity_auc'] = result['auc']
    except Exception as e:
        results['fidelity_auc'] = None
        print(f"    Warning: Propensity AUC failed - {e}")

    # 2. DCR Privacy (PRIMARY privacy measure)
    try:
        result = dcr_privacy(df_train, df_synth)
        results['privacy_dcr'] = result['dcr_percentile']
    except Exception as e:
        results['privacy_dcr'] = None
        print(f"    Warning: DCR failed - {e}")

    # 3. Membership Inference (SECONDARY - typically invalid, see docs)
    # Computed for completeness but not used in main analysis when attack_valid=False
    try:
        result = membership_inference_privacy(df_train, df_test, df_synth)
        results['privacy_mi_auc'] = result['attack_auc']
        results['privacy_mi_valid'] = result['attack_valid']
        if not result['attack_valid']:
            print(f"    Note: MI attack invalid (AUC={result['attack_auc']:.3f} < 0.6). Using DCR as primary.")
    except Exception as e:
        results['privacy_mi_auc'] = None
        results['privacy_mi_valid'] = None
        print(f"    Warning: MI failed - {e}")

    # 4. TSTR Utility
    try:
        result = tstr_utility(df_train_tstr, df_synth_tstr, df_test_tstr, target_col='HIGH_INCOME')
        results['utility_tstr'] = result['f1_ratio']
    except Exception as e:
        results['utility_tstr'] = None
        print(f"    Warning: TSTR failed - {e}")

    # 5. Fairness Gap (using RAC1P_RECODED as subgroup per study design)
    try:
        result = fairness_gap(df_train_tstr, df_synth_tstr, df_test_tstr,
                             target_col='HIGH_INCOME', subgroup_col='RAC1P_RECODED')
        results['fairness_gap'] = result['max_gap']
    except Exception as e:
        results['fairness_gap'] = None
        print(f"    Warning: Fairness failed - {e}")

    return results


def run_experiments(n_samples: int, n_replicates: int, methods: list, skip_measures: bool = False):
    """Run experiments for specified methods and replicates."""

    print("=" * 70)
    print("SYNTHETIC DATA GENERATION EXPERIMENTS")
    print("=" * 70)
    print(f"N samples: {n_samples:,}")
    print(f"N replicates: {n_replicates}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Random seeds: {RANDOM_SEEDS[:n_replicates]}")
    print("=" * 70)

    # Create directories
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    MEASURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint_file = RESULTS_DIR / "checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_file)

    # Load data
    print("\nLoading data...")
    df_train, df_test = preprocess_acs_data(sample_size=n_samples)
    median_income = df_train['PINCP'].median()
    print(f"Train: {len(df_train):,}, Test: {len(df_test):,}")

    # Track all results
    all_results = checkpoint['results'].copy()

    # Run experiments
    total_runs = len(methods) * n_replicates
    current_run = 0

    for method in methods:
        if method not in GENERATORS:
            print(f"\nWarning: Unknown method '{method}', skipping...")
            continue

        print(f"\n{'='*70}")
        print(f"METHOD: {method.upper()}")
        print(f"{'='*70}")

        for rep in range(n_replicates):
            current_run += 1
            seed = RANDOM_SEEDS[rep]
            run_id = f"{method}_rep{rep+1}"

            # Check if already completed
            if run_id in checkpoint['completed']:
                print(f"\n[{current_run}/{total_runs}] {run_id} - Already completed, skipping...")
                continue

            print(f"\n[{current_run}/{total_runs}] {run_id} (seed={seed})")

            # Generate synthetic data
            synth_file = SYNTHETIC_DIR / f"{run_id}.csv"

            if synth_file.exists():
                print(f"  Loading existing synthetic data from {synth_file}")
                df_synth = pd.read_csv(synth_file)
            else:
                print(f"  Generating synthetic data...")
                start_time = time.time()
                try:
                    df_synth = GENERATORS[method](df_train, len(df_train), seed)
                    elapsed = time.time() - start_time
                    print(f"  Generated {len(df_synth):,} records in {elapsed:.1f}s")

                    # Save synthetic data
                    df_synth.to_csv(synth_file, index=False)
                    print(f"  Saved to {synth_file}")
                except Exception as e:
                    print(f"  ERROR: Generation failed - {e}")
                    continue

            # Compute measures
            if not skip_measures:
                print(f"  Computing measures...")
                start_time = time.time()
                measures = compute_all_measures(df_train, df_test, df_synth, median_income)
                elapsed = time.time() - start_time
                print(f"  Measures computed in {elapsed:.1f}s")

                # Record results
                result = {
                    'method': method,
                    'replicate': rep + 1,
                    'seed': seed,
                    'n_samples': len(df_train),
                    **measures,
                    'timestamp': datetime.now().isoformat()
                }
                all_results.append(result)

                # Print summary
                print(f"    Fidelity: {measures.get('fidelity_auc', 'N/A'):.4f}" if measures.get('fidelity_auc') else "    Fidelity: N/A")
                print(f"    DCR:      {measures.get('privacy_dcr', 'N/A'):.4f}" if measures.get('privacy_dcr') else "    DCR: N/A")
                print(f"    TSTR:     {measures.get('utility_tstr', 'N/A'):.4f}" if measures.get('utility_tstr') else "    TSTR: N/A")
                print(f"    Fairness: {measures.get('fairness_gap', 'N/A'):.4f}" if measures.get('fairness_gap') else "    Fairness: N/A")

            # Update checkpoint
            checkpoint['completed'].append(run_id)
            checkpoint['results'] = all_results
            save_checkpoint(checkpoint_file, checkpoint)

    # Save final results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_file = RESULTS_DIR / "all_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n{'='*70}")
        print(f"Results saved to {results_file}")

        # Print summary
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}")

        summary = results_df.groupby('method').agg({
            'fidelity_auc': ['mean', 'std'],
            'privacy_dcr': ['mean', 'std'],
            'utility_tstr': ['mean', 'std'],
            'fairness_gap': ['mean', 'std']
        }).round(4)
        print(summary)

        # Save summary
        summary_file = RESULTS_DIR / "summary.csv"
        summary.to_csv(summary_file)
        print(f"\nSummary saved to {summary_file}")

    print(f"\n{'='*70}")
    print("EXPERIMENTS COMPLETE")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Run synthetic data experiments')
    parser.add_argument('--n-samples', type=int, default=35000,
                        help='Number of samples (default: 35000)')
    parser.add_argument('--n-replicates', type=int, default=5,
                        help='Number of replicates (default: 5)')
    parser.add_argument('--methods', nargs='+',
                        default=['independent_marginals', 'ctgan', 'dpbn', 'synthpop'],
                        help='Methods to run (default: all local methods)')
    parser.add_argument('--skip-measures', action='store_true',
                        help='Only generate synthetic data, skip measures')
    parser.add_argument('--reset', action='store_true',
                        help='Reset checkpoint and start fresh')

    args = parser.parse_args()

    # Reset checkpoint if requested
    if args.reset:
        checkpoint_file = RESULTS_DIR / "checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("Checkpoint reset.")

    run_experiments(
        n_samples=args.n_samples,
        n_replicates=args.n_replicates,
        methods=args.methods,
        skip_measures=args.skip_measures
    )


if __name__ == '__main__':
    main()
