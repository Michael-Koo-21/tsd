#!/usr/bin/env python3
"""
Experiment Runner for Synthetic Data Generation Study

Runs all generators and measures across multiple replicates.
Supports checkpointing and result aggregation.

Usage:
    tsd run --n-samples 35000 --n-replicates 5
    tsd run --n-samples 1000 --n-replicates 1 --methods independent_marginals  # Test run
"""

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

from tsd.config import DatasetConfig, default_config_path, load_config
from tsd.generators.ctgan_generator import generate_ctgan
from tsd.generators.dp_bayesian_network import generate_dp_bayesian_network
from tsd.generators.great_generator import generate_great
from tsd.generators.independent_marginals import generate_independent_marginals
from tsd.generators.synthpop_wrapper import generate_synthpop
from tsd.measures import (
    dcr_privacy,
    fairness_gap,
    membership_inference_privacy,
    propensity_auc,
    tstr_utility,
)
from tsd.preprocessing.load_data import load_and_preprocess

warnings.filterwarnings("ignore")

# Configuration
RANDOM_SEEDS = [42, 123, 456, 789, 1011]
RESULTS_DIR = Path("results/experiments")
SYNTHETIC_DIR = RESULTS_DIR / "synthetic_data"
MEASURES_DIR = RESULTS_DIR / "measures"


def _build_generators(config: DatasetConfig) -> dict:
    """Build generator functions, passing categorical_columns from config to CTGAN."""
    cat_cols = config.categorical_columns or None
    return {
        "independent_marginals": lambda df, n, seed: generate_independent_marginals(
            df, n, random_state=seed
        ),
        "ctgan": lambda df, n, seed: generate_ctgan(
            df, n, epochs=300, random_state=seed, categorical_columns=cat_cols
        ),
        "dpbn": lambda df, n, seed: generate_dp_bayesian_network(
            df, n, epsilon=1.0, random_state=seed
        ),
        "synthpop": lambda df, n, seed: generate_synthpop(df, n, random_state=seed),
        "great": lambda df, n, seed: generate_great(
            df, n, epochs=15, batch_size=64, random_state=seed, device="auto"
        ),
    }


def load_checkpoint(checkpoint_file: Path) -> dict:
    """Load checkpoint if it exists."""
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    return {"completed": [], "results": []}


def save_checkpoint(checkpoint_file: Path, checkpoint: dict):
    """Save checkpoint."""
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint, f, indent=2)


def compute_all_measures(df_train, df_test, df_synth, config: DatasetConfig):
    """Compute all measures for a synthetic dataset.

    Args:
        df_train: Training data
        df_test: Test data
        df_synth: Synthetic data
        config: Dataset configuration (provides target_variable, subgroup_column, etc.)

    Privacy Measure Note:
    DCR (Distance to Closest Record) is used as the PRIMARY privacy measure.
    Membership inference is computed but typically invalid for this data
    (attack AUC < 0.6 threshold). See methodological_validation.md for details.
    """
    results = {}

    target_col = config.target_variable
    subgroup_col = config.subgroup_column

    # Prepare TSTR data: binarize target if needed
    df_train_tstr = df_train.copy()
    df_test_tstr = df_test.copy()
    df_synth_tstr = df_synth.copy()

    if config.target_binarize:
        src = config.target_binarize["source_column"]
        threshold = config.target_binarize["threshold"]
        for df in (df_train_tstr, df_test_tstr, df_synth_tstr):
            df[target_col] = (df[src] > threshold).astype(int)
            if src != target_col and src in df.columns:
                df.drop(columns=[src], inplace=True)

    # 1. Propensity AUC (Fidelity)
    try:
        result = propensity_auc(df_train, df_synth)
        results["fidelity_auc"] = result["auc"]
    except Exception as e:
        results["fidelity_auc"] = None
        print(f"    Warning: Propensity AUC failed - {e}")

    # 2. DCR Privacy (PRIMARY privacy measure)
    try:
        result = dcr_privacy(df_train, df_synth)
        results["privacy_dcr"] = result["dcr_percentile"]
    except Exception as e:
        results["privacy_dcr"] = None
        print(f"    Warning: DCR failed - {e}")

    # 3. Membership Inference (SECONDARY - typically invalid, see docs)
    try:
        result = membership_inference_privacy(df_train, df_test, df_synth)
        results["privacy_mi_auc"] = result["attack_auc"]
        results["privacy_mi_valid"] = result["attack_valid"]
        if not result["attack_valid"]:
            print(f"    Note: MI attack invalid (AUC={result['attack_auc']:.3f} < 0.6). Using DCR.")
    except Exception as e:
        results["privacy_mi_auc"] = None
        results["privacy_mi_valid"] = None
        print(f"    Warning: MI failed - {e}")

    # 4. TSTR Utility
    try:
        result = tstr_utility(df_train_tstr, df_synth_tstr, df_test_tstr, target_col=target_col)
        results["utility_tstr"] = result["f1_ratio"]
    except Exception as e:
        results["utility_tstr"] = None
        print(f"    Warning: TSTR failed - {e}")

    # 5. Fairness Gap
    try:
        result = fairness_gap(
            df_train_tstr,
            df_synth_tstr,
            df_test_tstr,
            target_col=target_col,
            subgroup_col=subgroup_col,
        )
        results["fairness_gap"] = result["max_gap"]
    except Exception as e:
        results["fairness_gap"] = None
        print(f"    Warning: Fairness failed - {e}")

    return results


def run_experiments(
    config: DatasetConfig,
    n_samples: int,
    n_replicates: int,
    methods: list,
    skip_measures: bool = False,
):
    """Run experiments for specified methods and replicates."""

    print("=" * 70)
    print("SYNTHETIC DATA GENERATION EXPERIMENTS")
    print("=" * 70)
    print(f"Dataset: {config.name}")
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
    config.sample_size = n_samples
    df_train, df_test = load_and_preprocess(config)
    print(f"Train: {len(df_train):,}, Test: {len(df_test):,}")

    generators = _build_generators(config)

    # Track all results
    all_results = checkpoint["results"].copy()

    # Run experiments
    total_runs = len(methods) * n_replicates
    current_run = 0

    for method in methods:
        if method not in generators:
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
            if run_id in checkpoint["completed"]:
                print(f"\n[{current_run}/{total_runs}] {run_id} - Already completed, skipping...")
                continue

            print(f"\n[{current_run}/{total_runs}] {run_id} (seed={seed})")

            # Generate synthetic data
            synth_file = SYNTHETIC_DIR / f"{run_id}.csv"

            if synth_file.exists():
                print(f"  Loading existing synthetic data from {synth_file}")
                df_synth = pd.read_csv(synth_file)
            else:
                print("  Generating synthetic data...")
                start_time = time.time()
                try:
                    df_synth = generators[method](df_train, len(df_train), seed)
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
                print("  Computing measures...")
                start_time = time.time()
                measures = compute_all_measures(df_train, df_test, df_synth, config)
                elapsed = time.time() - start_time
                print(f"  Measures computed in {elapsed:.1f}s")

                # Record results
                result = {
                    "method": method,
                    "replicate": rep + 1,
                    "seed": seed,
                    "n_samples": len(df_train),
                    **measures,
                    "timestamp": datetime.now().isoformat(),
                }
                all_results.append(result)

                # Print summary
                for metric_key, label in [
                    ("fidelity_auc", "Fidelity"),
                    ("privacy_dcr", "DCR"),
                    ("utility_tstr", "TSTR"),
                    ("fairness_gap", "Fairness"),
                ]:
                    val = measures.get(metric_key)
                    print(f"    {label}: {val:.4f}" if val is not None else f"    {label}: N/A")

            # Update checkpoint
            checkpoint["completed"].append(run_id)
            checkpoint["results"] = all_results
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

        summary = (
            results_df.groupby("method")
            .agg(
                {
                    "fidelity_auc": ["mean", "std"],
                    "privacy_dcr": ["mean", "std"],
                    "utility_tstr": ["mean", "std"],
                    "fairness_gap": ["mean", "std"],
                }
            )
            .round(4)
        )
        print(summary)

        # Save summary
        summary_file = RESULTS_DIR / "summary.csv"
        summary.to_csv(summary_file)
        print(f"\nSummary saved to {summary_file}")

    print(f"\n{'='*70}")
    print("EXPERIMENTS COMPLETE")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Run synthetic data experiments")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to dataset config YAML (default: bundled acs_pums.yaml)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=35000, help="Number of samples (default: 35000)"
    )
    parser.add_argument(
        "--n-replicates", type=int, default=5, help="Number of replicates (default: 5)"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["independent_marginals", "ctgan", "dpbn", "synthpop", "great"],
        help="Methods to run (default: all 5 methods)",
    )
    parser.add_argument(
        "--skip-measures",
        action="store_true",
        help="Only generate synthetic data, skip measures",
    )
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint and start fresh")

    args = parser.parse_args()

    # Load config
    config_path = args.config if args.config else default_config_path()
    config = load_config(config_path)

    # Reset checkpoint if requested
    if args.reset:
        checkpoint_file = RESULTS_DIR / "checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("Checkpoint reset.")

    run_experiments(
        config=config,
        n_samples=args.n_samples,
        n_replicates=args.n_replicates,
        methods=args.methods,
        skip_measures=args.skip_measures,
    )


if __name__ == "__main__":
    main()
