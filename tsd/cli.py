"""
TSD command-line interface.

Subcommands:
    tsd run        Run synthetic data experiments
    tsd analyze    Run MADA analysis and generate figures
    tsd recommend  Print method recommendation for a given profile
    tsd verify     Verify paper claims against results
"""

import argparse
import json
import sys
from pathlib import Path


def cmd_run(args):
    """Run experiments."""
    from tsd.config import default_config_path, load_config
    from tsd.run_experiments import RESULTS_DIR, run_experiments

    config_path = args.config if args.config else default_config_path()
    config = load_config(config_path)

    output_dir = Path(args.output_dir) if args.output_dir else None
    results_dir = output_dir if output_dir is not None else RESULTS_DIR

    if args.reset:
        checkpoint_file = results_dir / "checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("Checkpoint reset.")

    # Resolve sample size: --sample-size takes priority, --n-samples is deprecated alias
    sample_size = args.sample_size if args.sample_size is not None else args.n_samples

    run_experiments(
        config=config,
        sample_size=sample_size,
        n_replicates=args.n_replicates,
        methods=args.methods,
        skip_measures=args.skip_measures,
        output_dir=output_dir,
        n_synthetic=args.n_synthetic,
    )


def _parse_value_refs(raw):
    """Parse --value-refs argument: JSON string or 'auto'."""
    if raw is None:
        return None
    if raw == "auto":
        return "auto"
    return json.loads(raw)


def cmd_analyze(args):
    """Run MADA analysis and generate figures."""
    from tsd.analysis.figures import regenerate_paper_figures

    results_path = Path(args.results)
    output_dir = Path(args.output_dir)

    if not results_path.exists():
        print(f"Error: results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    value_refs = _parse_value_refs(args.value_refs)
    regenerate_paper_figures(results_path, output_dir, value_refs=value_refs)


def cmd_recommend(args):
    """Print recommendation for a given profile."""
    from tsd.analysis.mada_framework import (
        generate_recommendation,
        load_results,
    )
    from tsd.constants import ARCHETYPES

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    df = load_results(results_path)

    if args.weights:
        weights = json.loads(args.weights)
        profile_name = "Custom"
    else:
        profile = args.profile
        if profile not in ARCHETYPES:
            print(
                f"Error: unknown profile '{profile}'. " f"Choose from: {', '.join(ARCHETYPES)}",
                file=sys.stderr,
            )
            sys.exit(1)
        weights = ARCHETYPES[profile]["weights"]
        profile_name = ARCHETYPES[profile]["name"]

    report = generate_recommendation(df, weights, profile_name)
    print(report)


def cmd_verify(args):
    """Verify paper claims."""
    from tsd.analysis.verify import verify_claims

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    print(verify_claims(results_path))


def main():
    parser = argparse.ArgumentParser(
        prog="tsd",
        description="TSD: Multiattribute Decision Analysis for Synthetic Data Method Selection",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── tsd run ───────────────────────────────────────────────────────────
    p_run = subparsers.add_parser("run", help="Run synthetic data experiments")
    p_run.add_argument(
        "--config", type=str, default=None, help="Dataset config YAML (default: acs_pums)"
    )
    p_run.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Max rows to sample from real data (default: use config value or all rows)",
    )
    p_run.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # deprecated alias for --sample-size
    )
    p_run.add_argument(
        "--n-synthetic",
        type=int,
        default=None,
        help="Number of synthetic rows to generate (default: match training set size)",
    )
    p_run.add_argument("--n-replicates", type=int, default=5)
    p_run.add_argument(
        "--methods",
        nargs="+",
        default=["independent_marginals", "ctgan", "dpbn", "synthpop", "great"],
    )
    p_run.add_argument("--skip-measures", action="store_true")
    p_run.add_argument("--reset", action="store_true")
    p_run.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for results (default: results/experiments)",
    )
    p_run.set_defaults(func=cmd_run)

    # ── tsd analyze ───────────────────────────────────────────────────────
    p_analyze = subparsers.add_parser("analyze", help="Run MADA analysis and generate figures")
    p_analyze.add_argument(
        "--results",
        type=str,
        default="results/experiments/all_results.csv",
    )
    p_analyze.add_argument(
        "--output-dir",
        type=str,
        default="results/experiments/figures",
    )
    p_analyze.add_argument(
        "--value-refs",
        type=str,
        default=None,
        help="Value function reference points: 'auto' to derive from data, "
        'or JSON e.g. \'{"fidelity": {"x_worst": 0.9}}\'',
    )
    p_analyze.set_defaults(func=cmd_analyze)

    # ── tsd recommend ─────────────────────────────────────────────────────
    p_rec = subparsers.add_parser("recommend", help="Print method recommendation")
    p_rec.add_argument(
        "--results",
        type=str,
        default="results/experiments/all_results.csv",
    )
    p_rec.add_argument(
        "--profile",
        type=str,
        default="balanced",
        help="Archetype profile (privacy_first, utility_first, balanced)",
    )
    p_rec.add_argument(
        "--weights",
        type=str,
        default=None,
        help='Custom weights as JSON, e.g. \'{"fidelity":0.2,"privacy":0.3,...}\'',
    )
    p_rec.set_defaults(func=cmd_recommend)

    # ── tsd verify ────────────────────────────────────────────────────────
    p_verify = subparsers.add_parser("verify", help="Verify paper claims against results")
    p_verify.add_argument(
        "--results",
        type=str,
        default="results/experiments/all_results.csv",
    )
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
