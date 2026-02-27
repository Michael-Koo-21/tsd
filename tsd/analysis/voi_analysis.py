"""
Value of Information (VOI) Analysis for Synthetic Data Method Selection.

Implements the Keisler (2004) framework to decompose the value of
benchmarking into prioritization vs. information components.

This module computes:
1. Strategy values (S1-S4) for each stakeholder archetype
2. Value of Prioritization: V(S3) - V(S1)
3. Value of Information: V(S4) - V(S3)
4. EVPI (Expected Value of Perfect Information)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from tsd.constants import ARCHETYPES, NORMALIZED_EFFICIENCY

# VOI analysis uses the 4 core metrics (no efficiency in raw data)
_VOI_METRIC_MAP = {
    "fidelity_auc": "fidelity",
    "privacy_dcr": "privacy",
    "utility_tstr": "utility",
    "fairness_gap": "fairness",
}

_VOI_METRIC_DIRECTION = {
    "fidelity_auc": False,
    "privacy_dcr": True,
    "utility_tstr": True,
    "fairness_gap": False,
}


@dataclass
class VOIResults:
    """Container for VOI analysis results."""

    archetype: str
    s1_random: float
    s2_heuristic: float
    s2_informed: float
    s3_rough: float
    s4_benchmark: float
    oracle: float
    value_of_prioritization: float
    value_of_information: float
    total_improvement: float
    pct_from_prioritization: float
    pct_from_information: float


def load_results(filepath: str | Path) -> pd.DataFrame:
    """Load experiment results from CSV."""
    return pd.read_csv(filepath)


def get_method_performance(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract mean and std performance for each method.

    Returns:
        means: DataFrame of mean performance by method
        stds: DataFrame of std performance by method
    """
    metrics = list(_VOI_METRIC_MAP.keys())
    means = df.groupby("method")[metrics].mean()
    stds = df.groupby("method")[metrics].std()
    return means, stds


def normalize_to_value(means: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw metrics to [0,1] value scale.

    Uses min-max normalization with direction handling.
    """
    values = means.copy()

    for metric in means.columns:
        col = means[metric]
        min_val, max_val = col.min(), col.max()

        if max_val > min_val:
            values[metric] = (col - min_val) / (max_val - min_val)
        else:
            values[metric] = 0.5

        # Invert if lower is better
        if not _VOI_METRIC_DIRECTION[metric]:
            values[metric] = 1 - values[metric]

    return values


def compute_method_value(values: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Compute aggregate value for each method given weights.
    """
    method_values = pd.Series(0.0, index=values.index)

    for metric, attr in _VOI_METRIC_MAP.items():
        if metric in values.columns and attr in weights:
            method_values += values[metric] * weights[attr]

    # Add efficiency (not in _VOI_METRIC_MAP but in weights).
    # Uses NORMALIZED_EFFICIENCY from constants.py (single source of truth).
    if "efficiency" in weights:
        for method in method_values.index:
            if method in NORMALIZED_EFFICIENCY:
                method_values[method] += NORMALIZED_EFFICIENCY[method] * weights["efficiency"]

    return method_values


def strategy_s1_random(method_values: pd.Series) -> float:
    """
    S1: Random selection - expected value of uniform random choice.
    """
    return method_values.mean()


def strategy_s2_heuristic(
    method_values: pd.Series, archetype: str, informed: bool = False
) -> float:
    """
    S2: Simple heuristic based on archetype.

    Default (informed=False):
        Privacy-first: Choose DP BN
        Utility-first: Choose CTGAN (common practitioner belief)
        Balanced: Choose GReaT

    Informed variant (informed=True):
        Privacy-first: Choose DP BN
        Utility-first: Choose Synthpop
        Balanced: Choose Synthpop
    """
    if informed:
        heuristic_choice = {
            "privacy_first": "dpbn",
            "utility_first": "synthpop",
            "balanced": "synthpop",
        }
    else:
        heuristic_choice = {
            "privacy_first": "dpbn",
            "utility_first": "ctgan",  # Common but potentially suboptimal
            "balanced": "great",
        }

    choice = heuristic_choice.get(archetype, "synthpop")
    if choice in method_values.index:
        return method_values[choice]
    return method_values.mean()


def strategy_s3_rough_estimates(
    method_values: pd.Series,
    stds: pd.DataFrame,
    weights: Dict[str, float],
    n_simulations: int = 10000,
    prior_noise: float = 0.15,
    seed: int = 42,
) -> float:
    """
    S3: Rough estimates with prior uncertainty.

    Models practitioner prior beliefs as noisy estimates of true method value.
    Default prior_noise=0.15 represents moderate but imperfect prior knowledge.
    """
    rng = np.random.default_rng(seed)

    # Simulate prior beliefs and decisions
    achieved_values = []

    for _ in range(n_simulations):
        # Add noise to true values to simulate imperfect prior knowledge
        noisy_values = method_values + rng.normal(0, prior_noise, len(method_values))

        # Choose method with highest estimated value
        chosen_method = noisy_values.idxmax()

        # Record actual value of chosen method
        achieved_values.append(method_values[chosen_method])

    return np.mean(achieved_values)


def strategy_s4_benchmark(method_values: pd.Series) -> float:
    """
    S4: Full benchmarking - select the best method.
    """
    return method_values.max()


def compute_oracle(
    means: pd.DataFrame,
    stds: pd.DataFrame,
    weights: Dict[str, float],
    n_simulations: int = 10000,
    uncertainty_inflation: float = 1.5,
    seed: int = 42,
) -> float:
    """
    Compute oracle value (EVPI bound) - expected value with perfect information.

    Simulates performance uncertainty and computes expected value of
    always choosing the best method for each realization.

    Uses fixed normalization anchors (min/max from observed means) so the
    transformation is linear in θ, guaranteeing EVPI >= V(S4) by Jensen's
    inequality. Values are NOT clamped to [0,1] — noise may push normalized
    values beyond this range, which is correct: it reflects the possibility
    that true performance lies outside the observed range.
    """
    rng = np.random.default_rng(seed)

    # Pre-compute fixed normalization anchors from observed means
    norm_min = {}
    norm_max = {}
    for metric in means.columns:
        norm_min[metric] = means[metric].min()
        norm_max[metric] = means[metric].max()

    oracle_values = []

    for _ in range(n_simulations):
        # Sample "true" performance with inflated uncertainty
        sampled_means = means.copy()
        for metric in means.columns:
            noise = rng.normal(0, stds[metric].values * uncertainty_inflation)
            sampled_means[metric] = means[metric] + noise

        # Normalize using FIXED anchors (linear in θ, preserves Jensen's)
        values = sampled_means.copy()
        for metric in sampled_means.columns:
            mn, mx = norm_min[metric], norm_max[metric]
            if mx > mn:
                values[metric] = (sampled_means[metric] - mn) / (mx - mn)
            else:
                values[metric] = 0.5
            # Invert if lower is better
            if not _VOI_METRIC_DIRECTION[metric]:
                values[metric] = 1 - values[metric]

        method_values = compute_method_value(values, weights)

        # Oracle always picks the best
        oracle_values.append(method_values.max())

    return np.mean(oracle_values)


def run_voi_analysis(
    df: pd.DataFrame,
    n_simulations: int = 50000,
    prior_noise: float = 0.15,
    uncertainty_inflation: float = 1.5,
    seed: int = 42,
) -> List[VOIResults]:
    """
    Run complete VOI analysis for all archetypes.

    Returns list of VOIResults for each archetype.
    """
    means, stds = get_method_performance(df)
    results = []

    for arch_id, arch_config in ARCHETYPES.items():
        weights = arch_config["weights"]

        # Compute value scores for this archetype
        values = normalize_to_value(means)
        method_values = compute_method_value(values, weights)

        # Compute each strategy's expected value
        s1 = strategy_s1_random(method_values)
        s2 = strategy_s2_heuristic(method_values, arch_id)
        s2_informed = strategy_s2_heuristic(method_values, arch_id, informed=True)
        s3 = strategy_s3_rough_estimates(
            method_values, stds, weights, n_simulations, prior_noise, seed
        )
        s4 = strategy_s4_benchmark(method_values)
        oracle_raw = compute_oracle(
            means, stds, weights, n_simulations, uncertainty_inflation, seed
        )
        # Enforce Oracle >= S4: Jensen's inequality guarantees E[max V(θ)] >= max V(E[θ])
        # for convex max over linear value functions with fixed normalization anchors.
        # Finite-sample MC estimates may violate this; the constraint is theoretically exact.
        oracle = max(oracle_raw, s4)

        # Compute VOI decomposition
        vop = s3 - s1  # Value of Prioritization
        voi = s4 - s3  # Value of Information
        total = s4 - s1

        # Percentages (handle edge cases)
        if total > 0.001:
            pct_prior = (vop / total) * 100
            pct_info = (voi / total) * 100
        else:
            pct_prior = 50.0
            pct_info = 50.0

        results.append(
            VOIResults(
                archetype=arch_config["name"],
                s1_random=s1,
                s2_heuristic=s2,
                s2_informed=s2_informed,
                s3_rough=s3,
                s4_benchmark=s4,
                oracle=oracle,
                value_of_prioritization=vop,
                value_of_information=voi,
                total_improvement=total,
                pct_from_prioritization=pct_prior,
                pct_from_information=pct_info,
            )
        )

    return results


def inflation_factor_sensitivity(
    df: pd.DataFrame,
    factors: List[float] | None = None,
    n_simulations: int = 10000,
    prior_noise: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Sensitivity analysis of VOI results to the uncertainty inflation factor.

    The inflation factor (default 1.5x) scales replicate standard deviations to
    approximate cross-dataset variability. This parameter directly affects EVPI
    and the VOI decomposition, but its value is asserted without calibration.
    This function reports VOI results across a range of inflation factors.

    Returns dict with per-factor results and a summary of how the prioritization
    percentage varies.
    """
    if factors is None:
        factors = [1.0, 1.25, 1.5, 1.75, 2.0]

    factor_results = {}

    for factor in factors:
        results = run_voi_analysis(
            df,
            n_simulations=n_simulations,
            prior_noise=prior_noise,
            uncertainty_inflation=factor,
            seed=seed,
        )

        pct_prioritization = [r.pct_from_prioritization for r in results]
        pct_information = [r.pct_from_information for r in results]
        oracle_values = [r.oracle for r in results]
        s4_values = [r.s4_benchmark for r in results]

        factor_results[factor] = {
            "results": results,
            "mean_pct_prioritization": np.mean(pct_prioritization),
            "mean_pct_information": np.mean(pct_information),
            "mean_oracle": np.mean(oracle_values),
            "mean_s4": np.mean(s4_values),
            "mean_evpi_gap": np.mean([o - s for o, s in zip(oracle_values, s4_values)]),
            "per_archetype": {
                r.archetype: {
                    "pct_prioritization": r.pct_from_prioritization,
                    "pct_information": r.pct_from_information,
                    "oracle": r.oracle,
                    "evpi_gap": r.oracle - r.s4_benchmark,
                }
                for r in results
            },
        }

    # Summary across factors
    pct_range = [factor_results[f]["mean_pct_prioritization"] for f in factors]
    summary = {
        "factors_tested": factors,
        "mean_pct_prioritization_range": (min(pct_range), max(pct_range)),
        "baseline_factor": 1.5,
        "baseline_pct": factor_results.get(1.5, {}).get("mean_pct_prioritization"),
        "interpretation": (
            f"Across inflation factors {min(factors):.1f}--{max(factors):.1f}, "
            f"mean prioritization percentage ranges from "
            f"{min(pct_range):.0f}% to {max(pct_range):.0f}%. "
            f"The qualitative finding that prioritization accounts for the majority "
            f"of value is {'robust' if min(pct_range) > 50 else 'sensitive'} "
            f"to the inflation factor choice."
        ),
    }

    return {"by_factor": factor_results, "summary": summary}


def format_results_tables(results: List[VOIResults]) -> Tuple[str, str]:
    """
    Format results as LaTeX-ready tables.

    Returns:
        strategy_table: Strategy outcomes table
        voi_table: VOI decomposition table
    """
    # Strategy outcomes table
    strategy_rows = []
    strategy_rows.append("Strategy & Privacy-First & Utility-First & Balanced & Mean \\\\")
    strategy_rows.append("\\midrule")

    # Compute means
    s1_mean = np.mean([r.s1_random for r in results])
    s2_mean = np.mean([r.s2_heuristic for r in results])
    s2_inf_mean = np.mean([r.s2_informed for r in results])
    s3_mean = np.mean([r.s3_rough for r in results])
    s4_mean = np.mean([r.s4_benchmark for r in results])
    oracle_mean = np.mean([r.oracle for r in results])

    strategy_rows.append(
        f"S1 (Random) & {results[0].s1_random:.3f} & {results[1].s1_random:.3f} & {results[2].s1_random:.3f} & {s1_mean:.3f} \\\\"
    )
    strategy_rows.append(
        f"S2 (Heuristic) & {results[0].s2_heuristic:.3f} & {results[1].s2_heuristic:.3f} & {results[2].s2_heuristic:.3f} & {s2_mean:.3f} \\\\"
    )
    strategy_rows.append(
        f"S2' (Informed Heuristic) & {results[0].s2_informed:.3f} & {results[1].s2_informed:.3f} & {results[2].s2_informed:.3f} & {s2_inf_mean:.3f} \\\\"
    )
    strategy_rows.append(
        f"S3 (Rough Estimates) & {results[0].s3_rough:.3f} & {results[1].s3_rough:.3f} & {results[2].s3_rough:.3f} & {s3_mean:.3f} \\\\"
    )
    strategy_rows.append(
        f"S4 (Full Benchmark) & {results[0].s4_benchmark:.3f} & {results[1].s4_benchmark:.3f} & {results[2].s4_benchmark:.3f} & {s4_mean:.3f} \\\\"
    )
    strategy_rows.append("\\midrule")
    strategy_rows.append(
        f"Oracle (EVPI bound) & {results[0].oracle:.3f} & {results[1].oracle:.3f} & {results[2].oracle:.3f} & {oracle_mean:.3f} \\\\"
    )

    strategy_table = "\n".join(strategy_rows)

    # VOI decomposition table
    voi_rows = []
    voi_rows.append("Component & Privacy-First & Utility-First & Balanced & Mean \\\\")
    voi_rows.append("\\midrule")

    vop_mean = np.mean([r.value_of_prioritization for r in results])
    voi_mean = np.mean([r.value_of_information for r in results])
    total_mean = np.mean([r.total_improvement for r in results])
    pct_prior_mean = np.mean([r.pct_from_prioritization for r in results])
    pct_info_mean = np.mean([r.pct_from_information for r in results])

    voi_rows.append(
        f"Value of Prioritization $V(\\text{{S3}})-V(\\text{{S1}})$ & {results[0].value_of_prioritization:.3f} & {results[1].value_of_prioritization:.3f} & {results[2].value_of_prioritization:.3f} & {vop_mean:.3f} \\\\"
    )
    voi_rows.append(
        f"Value of Information $V(\\text{{S4}})-V(\\text{{S3}})$ & {results[0].value_of_information:.3f} & {results[1].value_of_information:.3f} & {results[2].value_of_information:.3f} & {voi_mean:.3f} \\\\"
    )
    voi_rows.append(
        f"Total Improvement $V(\\text{{S4}})-V(\\text{{S1}})$ & {results[0].total_improvement:.3f} & {results[1].total_improvement:.3f} & {results[2].total_improvement:.3f} & {total_mean:.3f} \\\\"
    )
    voi_rows.append("\\midrule")
    voi_rows.append(
        f"\\% from Prioritization & {results[0].pct_from_prioritization:.0f}\\% & {results[1].pct_from_prioritization:.0f}\\% & {results[2].pct_from_prioritization:.0f}\\% & \\textbf{{{pct_prior_mean:.0f}\\%}} \\\\"
    )
    voi_rows.append(
        f"\\% from Information & {results[0].pct_from_information:.0f}\\% & {results[1].pct_from_information:.0f}\\% & {results[2].pct_from_information:.0f}\\% & {pct_info_mean:.0f}\\% \\\\"
    )

    voi_table = "\n".join(voi_rows)

    return strategy_table, voi_table


def print_report(results: List[VOIResults]) -> None:
    """Print a human-readable VOI analysis report."""
    print("=" * 70)
    print("VALUE OF INFORMATION (VOI) ANALYSIS")
    print("Following Keisler (2004) Framework")
    print("=" * 70)
    print()

    # Strategy outcomes
    print("STRATEGY OUTCOMES (Expected Value)")
    print("-" * 70)
    print(
        f"{'Strategy':<25} {'Privacy-First':>12} {'Utility-First':>14} {'Balanced':>10} {'Mean':>8}"
    )
    print("-" * 70)

    s1_vals = [r.s1_random for r in results]
    s2_vals = [r.s2_heuristic for r in results]
    s2_inf_vals = [r.s2_informed for r in results]
    s3_vals = [r.s3_rough for r in results]
    s4_vals = [r.s4_benchmark for r in results]
    oracle_vals = [r.oracle for r in results]

    print(
        f"{'S1 (Random)':<25} {s1_vals[0]:>12.3f} {s1_vals[1]:>14.3f} {s1_vals[2]:>10.3f} {np.mean(s1_vals):>8.3f}"
    )
    print(
        f"{'S2 (Heuristic)':<25} {s2_vals[0]:>12.3f} {s2_vals[1]:>14.3f} {s2_vals[2]:>10.3f} {np.mean(s2_vals):>8.3f}"
    )
    print(
        f"{'S2 (Informed)':<25} {s2_inf_vals[0]:>12.3f} {s2_inf_vals[1]:>14.3f} {s2_inf_vals[2]:>10.3f} {np.mean(s2_inf_vals):>8.3f}"
    )
    print(
        f"{'S3 (Rough Estimates)':<25} {s3_vals[0]:>12.3f} {s3_vals[1]:>14.3f} {s3_vals[2]:>10.3f} {np.mean(s3_vals):>8.3f}"
    )
    print(
        f"{'S4 (Full Benchmark)':<25} {s4_vals[0]:>12.3f} {s4_vals[1]:>14.3f} {s4_vals[2]:>10.3f} {np.mean(s4_vals):>8.3f}"
    )
    print("-" * 70)
    print(
        f"{'Oracle (EVPI bound)':<25} {oracle_vals[0]:>12.3f} {oracle_vals[1]:>14.3f} {oracle_vals[2]:>10.3f} {np.mean(oracle_vals):>8.3f}"
    )
    print()

    # VOI decomposition
    print("VOI DECOMPOSITION")
    print("-" * 70)
    print(
        f"{'Component':<35} {'Privacy-First':>12} {'Utility-First':>14} {'Balanced':>10} {'Mean':>8}"
    )
    print("-" * 70)

    vop_vals = [r.value_of_prioritization for r in results]
    voi_vals = [r.value_of_information for r in results]
    total_vals = [r.total_improvement for r in results]
    pct_prior_vals = [r.pct_from_prioritization for r in results]
    pct_info_vals = [r.pct_from_information for r in results]

    print(
        f"{'Value of Prioritization':<35} {vop_vals[0]:>12.3f} {vop_vals[1]:>14.3f} {vop_vals[2]:>10.3f} {np.mean(vop_vals):>8.3f}"
    )
    print(
        f"{'Value of Information':<35} {voi_vals[0]:>12.3f} {voi_vals[1]:>14.3f} {voi_vals[2]:>10.3f} {np.mean(voi_vals):>8.3f}"
    )
    print(
        f"{'Total Improvement':<35} {total_vals[0]:>12.3f} {total_vals[1]:>14.3f} {total_vals[2]:>10.3f} {np.mean(total_vals):>8.3f}"
    )
    print("-" * 70)
    print(
        f"{'% from Prioritization':<35} {pct_prior_vals[0]:>11.0f}% {pct_prior_vals[1]:>13.0f}% {pct_prior_vals[2]:>9.0f}% {np.mean(pct_prior_vals):>7.0f}%"
    )
    print(
        f"{'% from Information':<35} {pct_info_vals[0]:>11.0f}% {pct_info_vals[1]:>13.0f}% {pct_info_vals[2]:>9.0f}% {np.mean(pct_info_vals):>7.0f}%"
    )
    print()

    # Key findings
    print("KEY FINDINGS")
    print("-" * 70)
    mean_pct_prior = np.mean(pct_prior_vals)
    print(f"1. Prioritization accounts for {mean_pct_prior:.0f}% of achievable value improvement")
    print("   (Keisler 2004 found ~71% in portfolio decision analysis)")
    print()
    print("2. VOI varies by archetype:")
    for r in results:
        print(f"   - {r.archetype}: VOI = {r.value_of_information:.3f}")
    print()
    print("3. Heuristics perform well when priorities are clear:")
    for r in results:
        heuristic_gap = r.s4_benchmark - r.s2_heuristic
        print(f"   - {r.archetype}: S2 vs S4 gap = {heuristic_gap:.3f}")
    print()
    print("=" * 70)


def main(filepath: str | Path = None):
    """Run VOI analysis and print results."""
    if filepath is None:
        filepath = Path(__file__).parent.parent.parent / "results/experiments/all_results.csv"

    df = load_results(filepath)
    results = run_voi_analysis(df)
    print_report(results)

    # Also print LaTeX tables
    print("\nLATEX TABLES FOR PAPER")
    print("=" * 70)
    strategy_table, voi_table = format_results_tables(results)
    print("\nStrategy Table:")
    print(strategy_table)
    print("\nVOI Table:")
    print(voi_table)

    return results
