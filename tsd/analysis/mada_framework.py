"""
Multiattribute Decision Analysis (MADA) Framework for Synthetic Data Method Selection.

This module implements a decision support system that helps practitioners select
the most appropriate synthetic data generation method based on their specific
requirements and priorities.

Five Objectives (matching paper Section 4):
- Fidelity: Statistical similarity to original data (Propensity AUC)
- Privacy: Protection against re-identification (DCR 5th percentile)
- Utility: Downstream ML model performance (TSTR F1 ratio)
- Fairness: Equitable outcomes across demographic groups (Max subgroup gap)
- Efficiency: Computational cost and runtime (Total time in minutes)

Features:
- Weighted scoring with customizable objective weights (5 objectives)
- Multiple normalization methods (min-max, z-score)
- Sensitivity analysis for weight changes
- Pre-defined stakeholder archetypes (Privacy-First, Utility-First, Balanced)
- Visual comparison and recommendation reports
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from tsd.constants import (
    DEFAULT_RUNTIME_MINUTES,
    METHOD_LABELS,
    METRIC_DIRECTION,
    METRIC_MAP,
    PROFILES,
)


def load_results(filepath: str | Path) -> pd.DataFrame:
    """Load experiment results from CSV."""
    return pd.read_csv(filepath)


def add_efficiency_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add efficiency_time metric to results DataFrame.

    If 'efficiency_time' or 'generation_time_min' column exists, use it.
    Otherwise, use DEFAULT_RUNTIME_MINUTES based on method.
    """
    df = df.copy()

    # Check for existing time columns
    if "efficiency_time" in df.columns:
        return df
    elif "generation_time_min" in df.columns:
        df["efficiency_time"] = df["generation_time_min"]
    elif "generation_time_sec" in df.columns:
        df["efficiency_time"] = df["generation_time_sec"] / 60.0
    else:
        # Use default runtime values based on method
        df["efficiency_time"] = df["method"].map(DEFAULT_RUNTIME_MINUTES)

    return df


def get_method_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean scores for each method, including efficiency."""
    # Ensure efficiency data is present
    df = add_efficiency_data(df)

    metrics = list(METRIC_MAP.keys())
    # Filter to only metrics present in the dataframe
    available_metrics = [m for m in metrics if m in df.columns]
    return df.groupby("method")[available_metrics].mean()


def normalize_scores(
    scores: pd.DataFrame, method: Literal["minmax", "zscore"] = "minmax"
) -> pd.DataFrame:
    """
    Normalize scores to 0-1 scale.

    Handles direction (lower is better for fairness_gap, efficiency_time).
    """
    normalized = scores.copy()

    for metric in scores.columns:
        if metric not in METRIC_DIRECTION:
            continue  # Skip unknown metrics

        values = scores[metric]

        if method == "minmax":
            min_val, max_val = values.min(), values.max()
            if max_val > min_val:
                normalized[metric] = (values - min_val) / (max_val - min_val)
            else:
                normalized[metric] = 0.5
        elif method == "zscore":
            mean, std = values.mean(), values.std()
            if std > 0:
                normalized[metric] = (values - mean) / std
                # Convert to 0-1 range using sigmoid-like transformation
                normalized[metric] = 1 / (1 + np.exp(-normalized[metric]))
            else:
                normalized[metric] = 0.5

        # Invert if lower is better
        if not METRIC_DIRECTION[metric]:
            normalized[metric] = 1 - normalized[metric]

    return normalized


def calculate_weighted_scores(normalized: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """
    Calculate weighted aggregate score for each method.

    Args:
        normalized: Normalized scores DataFrame
        weights: Dict mapping attributes to weights (must sum to 1.0)

    Returns:
        Series with weighted scores per method
    """
    # Validate weights sum to 1
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

    weighted_scores = pd.Series(0.0, index=normalized.index)

    for metric, attribute in METRIC_MAP.items():
        if metric not in normalized.columns:
            continue  # Skip metrics not in data
        weight = weights.get(attribute, 0)
        weighted_scores += normalized[metric] * weight

    return weighted_scores


def rank_methods(weighted_scores: pd.Series) -> pd.DataFrame:
    """Rank methods by weighted score."""
    ranking = pd.DataFrame(
        {
            "method": weighted_scores.index,
            "score": weighted_scores.values,
            "rank": weighted_scores.rank(ascending=False).astype(int),
        }
    )
    return ranking.sort_values("rank")


def sensitivity_analysis(
    normalized: pd.DataFrame,
    base_weights: dict[str, float],
    attribute: str,
    weight_range: tuple[float, float] = (0.0, 0.8),
    steps: int = 20,
) -> pd.DataFrame:
    """
    Analyze how changing one attribute's weight affects rankings.

    Redistributes weight changes proportionally among other attributes.
    """
    results = []
    other_attrs = [a for a in base_weights.keys() if a != attribute]
    other_total = sum(base_weights[a] for a in other_attrs)

    for w in np.linspace(weight_range[0], weight_range[1], steps):
        # Redistribute remaining weight proportionally
        remaining = 1.0 - w
        new_weights = {attribute: w}

        for a in other_attrs:
            if other_total > 0:
                new_weights[a] = base_weights[a] / other_total * remaining
            else:
                new_weights[a] = remaining / len(other_attrs)

        scores = calculate_weighted_scores(normalized, new_weights)

        for method, score in scores.items():
            results.append(
                {
                    "weight": w,
                    "method": method,
                    "score": score,
                }
            )

    return pd.DataFrame(results)


def monte_carlo_optimality(
    df: pd.DataFrame,
    weights: dict[str, float],
    n_simulations: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute probability of optimality for each method via Monte Carlo simulation.

    Samples performance metrics from Normal(mean, std) estimated from replicates,
    normalizes, computes weighted value, and records which method is optimal.

    Args:
        df: Raw results DataFrame with 'method' column and metric columns.
        weights: Dict mapping attributes to weights (must sum to 1.0).
        n_simulations: Number of Monte Carlo draws.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: method, p_optimal, mean_rank, rank_ci_low, rank_ci_high.
    """
    rng = np.random.default_rng(seed)

    metrics = [m for m in METRIC_MAP.keys() if m in df.columns]
    means = df.groupby("method")[metrics].mean()
    stds = df.groupby("method")[metrics].std().fillna(0)
    methods = means.index.tolist()

    optimal_counts = {m: 0 for m in methods}
    all_ranks = {m: [] for m in methods}

    for _ in range(n_simulations):
        # Sample performance for each method/metric
        sampled = means.copy()
        for metric in metrics:
            noise = rng.normal(0, stds[metric].values)
            sampled[metric] = means[metric].values + noise

        # Normalize (min-max with direction handling)
        normalized = sampled.copy()
        for metric in metrics:
            col = sampled[metric]
            mn, mx = col.min(), col.max()
            if mx > mn:
                normalized[metric] = (col - mn) / (mx - mn)
            else:
                normalized[metric] = 0.5
            if not METRIC_DIRECTION.get(metric, True):
                normalized[metric] = 1 - normalized[metric]

        # Add efficiency if in weights
        if "efficiency" in weights:
            normalized["efficiency_time"] = pd.Series(
                {m: DEFAULT_RUNTIME_MINUTES.get(m, 1.0) for m in methods}
            )
            # Efficiency normalization (lower time = better)
            col = normalized["efficiency_time"]
            mn, mx = col.min(), col.max()
            if mx > mn:
                normalized["efficiency_time"] = 1 - (col - mn) / (mx - mn)
            else:
                normalized["efficiency_time"] = 0.5

        # Compute weighted scores
        scores = calculate_weighted_scores(normalized, weights)

        # Record optimal and ranks
        ranks = scores.rank(ascending=False)
        optimal_method = scores.idxmax()
        optimal_counts[optimal_method] += 1
        for m in methods:
            all_ranks[m].append(ranks[m])

    # Compile results
    results = []
    for m in methods:
        ranks_arr = np.array(all_ranks[m])
        results.append(
            {
                "method": m,
                "p_optimal": optimal_counts[m] / n_simulations,
                "mean_rank": np.mean(ranks_arr),
                "rank_ci_low": int(np.percentile(ranks_arr, 2.5)),
                "rank_ci_high": int(np.percentile(ranks_arr, 97.5)),
            }
        )

    return pd.DataFrame(results).sort_values("p_optimal", ascending=False)


def value_function_sensitivity(
    df: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    """
    Test robustness to alternative value function forms.

    Evaluates five value function specifications:
    - linear: standard min-max (v = x_norm)
    - concave: diminishing returns (v = sqrt(x_norm))
    - convex: increasing returns (v = x_norm^2)
    - logistic: S-curve (v = 1/(1+exp(-10*(x_norm - 0.5))))
    - step: threshold (v = 1 if x_norm > 0.5 else 0)

    Args:
        df: Raw results DataFrame.
        weights: Dict mapping attributes to weights.

    Returns:
        DataFrame with columns: method, vf_form, value, rank.
    """
    scores = get_method_scores(df)
    # Base min-max normalization
    base_norm = normalize_scores(scores)

    def apply_vf(normalized: pd.DataFrame, form: str) -> pd.DataFrame:
        result = normalized.copy()
        metrics = [m for m in METRIC_MAP.keys() if m in result.columns]
        for metric in metrics:
            x = normalized[metric].values
            if form == "linear":
                pass  # Already linear
            elif form == "concave":
                result[metric] = np.sqrt(x)
            elif form == "convex":
                result[metric] = x**2
            elif form == "logistic":
                result[metric] = 1.0 / (1.0 + np.exp(-10 * (x - 0.5)))
            elif form == "step":
                result[metric] = np.where(x > 0.5, 1.0, 0.0)
        return result

    results = []
    for form in ["linear", "concave", "convex", "logistic", "step"]:
        transformed = apply_vf(base_norm, form)
        weighted = calculate_weighted_scores(transformed, weights)
        ranking = weighted.rank(ascending=False)
        for method in weighted.index:
            results.append(
                {
                    "method": method,
                    "vf_form": form,
                    "value": weighted[method],
                    "rank": int(ranking[method]),
                }
            )

    return pd.DataFrame(results)


def generate_recommendation(
    df: pd.DataFrame, weights: dict[str, float], profile_name: str = "Custom"
) -> str:
    """Generate a detailed recommendation report."""
    scores = get_method_scores(df)
    normalized = normalize_scores(scores)
    weighted = calculate_weighted_scores(normalized, weights)
    ranking = rank_methods(weighted)

    report = []
    report.append("=" * 70)
    report.append("MADA RECOMMENDATION REPORT")
    report.append("Synthetic Data Generation Method Selection (5 Objectives)")
    report.append("=" * 70)
    report.append("")

    # Profile info
    report.append(f"User Profile: {profile_name}")
    report.append("-" * 40)
    report.append("Objective Weights:")
    for attr, weight in sorted(weights.items(), key=lambda x: -x[1]):
        bar = "█" * int(weight * 20) + "░" * (20 - int(weight * 20))
        report.append(f"  {attr.capitalize():<12} {bar} {weight:.0%}")
    report.append("")

    # Rankings
    report.append("METHOD RANKINGS")
    report.append("-" * 40)
    report.append(f"{'Rank':<6} {'Method':<25} {'Score':>10}")
    report.append("-" * 45)

    for _, row in ranking.iterrows():
        method_label = METHOD_LABELS.get(row["method"], row["method"])
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(row["rank"], "  ")
        report.append(f"{medal} {row['rank']:<3} {method_label:<25} {row['score']:>10.4f}")

    report.append("")

    # Top recommendation
    top_method = ranking.iloc[0]["method"]
    top_score = ranking.iloc[0]["score"]
    second_method = ranking.iloc[1]["method"]
    second_score = ranking.iloc[1]["score"]

    report.append("RECOMMENDATION")
    report.append("-" * 40)
    report.append(f"Primary: {METHOD_LABELS.get(top_method, top_method)}")
    report.append(
        f"Score advantage: +{(top_score - second_score):.4f} over {METHOD_LABELS.get(second_method, second_method)}"
    )
    report.append("")

    # Method strengths/weaknesses (5 objectives)
    report.append("TOP METHOD PROFILE (5 Objectives)")
    report.append("-" * 40)

    method_normalized = normalized.loc[top_method]
    for metric, attr in METRIC_MAP.items():
        if metric not in method_normalized.index:
            continue
        val = method_normalized[metric]
        strength = "Strong" if val > 0.7 else "Moderate" if val > 0.4 else "Weak"
        bar = "█" * int(val * 15) + "░" * (15 - int(val * 15))
        report.append(f"  {attr.capitalize():<12} {bar} {val:.2f} ({strength})")

    report.append("")

    # Trade-off warning
    report.append("TRADE-OFF CONSIDERATIONS")
    report.append("-" * 40)

    weaknesses = []
    for metric, attr in METRIC_MAP.items():
        if metric in method_normalized.index and method_normalized[metric] < 0.4:
            weaknesses.append(attr)

    if weaknesses:
        report.append(f"The recommended method has lower scores in: {', '.join(weaknesses)}")
        report.append("Consider alternative methods if these attributes are critical.")
    else:
        report.append("The recommended method has balanced performance across all attributes.")

    report.append("")
    report.append("=" * 70)

    return "\n".join(report)


def plot_comparison(df: pd.DataFrame, weights: dict[str, float], output_path: Path = None):
    """Create visual comparison of methods with current weights."""
    import matplotlib.pyplot as plt

    scores = get_method_scores(df)
    normalized = normalize_scores(scores)
    weighted = calculate_weighted_scores(normalized, weights)
    ranking = rank_methods(weighted)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Stacked bar showing weighted contribution
    ax1 = axes[0]
    methods = ranking["method"].values

    bottom = np.zeros(len(methods))
    # Color palette for 5 objectives
    colors = {
        "fidelity": "#4C72B0",
        "privacy": "#55A868",
        "utility": "#C44E52",
        "fairness": "#8172B3",
        "efficiency": "#CCB974",
    }

    for metric, attr in METRIC_MAP.items():
        if metric not in normalized.columns:
            continue
        values = [normalized.loc[m, metric] * weights.get(attr, 0) for m in methods]
        ax1.barh(
            range(len(methods)),
            values,
            left=bottom,
            label=f"{attr.capitalize()} ({weights.get(attr, 0):.0%})",
            color=colors.get(attr, "#999999"),
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += values

    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods])
    ax1.set_xlabel("Weighted Score")
    ax1.set_title("Weighted Score Breakdown by Objective (5 Objectives)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_xlim(0, 1)

    # Add total scores
    for i, (_m, s) in enumerate(zip(methods, ranking["score"])):
        ax1.text(s + 0.02, i, f"{s:.3f}", va="center", fontsize=10, fontweight="bold")

    # Right: Weight sensitivity
    ax2 = axes[1]

    # Show sensitivity for the most important attribute
    top_attr = max(weights.items(), key=lambda x: x[1])[0]
    sens = sensitivity_analysis(normalized, weights, top_attr)

    for method in methods:
        method_data = sens[sens["method"] == method]
        ax2.plot(
            method_data["weight"],
            method_data["score"],
            label=METHOD_LABELS.get(method, method),
            linewidth=2,
        )

    ax2.axvline(
        x=weights[top_attr],
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Current ({weights[top_attr]:.0%})",
    )
    ax2.set_xlabel(f"{top_attr.capitalize()} Weight")
    ax2.set_ylabel("Weighted Score")
    ax2.set_title(f"Sensitivity Analysis: {top_attr.capitalize()} Weight")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")

    return fig


def demo_all_profiles(df: pd.DataFrame, output_dir: Path = None) -> dict:
    """
    Run MADA analysis for all predefined profiles.

    Returns dict with reports and creates comparison visualization.
    """
    import matplotlib.pyplot as plt

    results = {}

    print("=" * 70)
    print("MADA FRAMEWORK DEMONSTRATION (5 Objectives)")
    print("Comparing recommendations across stakeholder archetypes")
    print("=" * 70)
    print()

    scores = get_method_scores(df)
    normalized = normalize_scores(scores)

    # Use the 3 primary profiles matching the paper
    primary_profiles = ["privacy_first", "utility_first", "balanced"]

    # Summary table
    summary_data = []

    for profile_id in primary_profiles:
        if profile_id not in PROFILES:
            continue
        profile = PROFILES[profile_id]
        weighted = calculate_weighted_scores(normalized, profile.weights)
        ranking = rank_methods(weighted)
        top_method = ranking.iloc[0]["method"]
        top_score = ranking.iloc[0]["score"]

        summary_data.append(
            {
                "profile": profile.name,
                "top_method": METHOD_LABELS.get(top_method, top_method),
                "score": top_score,
                "weights": profile.weights,
            }
        )

        results[profile_id] = {
            "profile": profile,
            "ranking": ranking,
            "report": generate_recommendation(df, profile.weights, profile.name),
        }

    # Print summary
    print("PROFILE COMPARISON SUMMARY (Paper Table 4)")
    print("-" * 70)
    print(f"{'Profile':<30} {'Recommended Method':<25} {'Score':>10}")
    print("-" * 70)

    for row in summary_data:
        print(f"{row['profile']:<30} {row['top_method']:<25} {row['score']:>10.4f}")

    print()
    print("-" * 70)
    print("KEY INSIGHT: Method selection depends heavily on practitioner priorities!")
    print("-" * 70)
    print()

    # Create comparison visualization
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes = axes.flatten()

        for idx, profile_id in enumerate(primary_profiles):
            if profile_id not in PROFILES:
                continue
            profile = PROFILES[profile_id]
            ax = axes[idx]

            weighted = calculate_weighted_scores(normalized, profile.weights)
            ranking = rank_methods(weighted)

            methods = ranking["method"].values
            scores_vals = ranking["score"].values

            colors = [
                "gold" if i == 0 else "silver" if i == 1 else "peru" if i == 2 else "lightgray"
                for i in range(len(methods))
            ]

            bars = ax.barh(
                range(len(methods)), scores_vals, color=colors, edgecolor="black", linewidth=0.5
            )

            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=9)
            ax.set_xlabel("Weighted Score")
            ax.set_title(f"{profile.name}\n{profile.description}", fontsize=10, fontweight="bold")
            ax.set_xlim(0, 1)

            # Add score labels
            for i, (_bar, score) in enumerate(zip(bars, scores_vals)):
                ax.text(score + 0.02, i, f"{score:.3f}", va="center", fontsize=9)

        plt.suptitle(
            "MADA Framework: Method Rankings by Stakeholder Archetype (5 Objectives)\n(Gold = Recommended)",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        fig_path = output_dir / "mada_profile_comparison.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"Comparison visualization saved to: {fig_path}")

    return results


def interactive_demo(filepath: str | Path):
    """
    Interactive demonstration of the MADA framework (5 Objectives).
    """
    df = load_results(filepath)

    print("\n" + "=" * 70)
    print("MULTIATTRIBUTE DECISION ANALYSIS (MADA) FRAMEWORK")
    print("Interactive Method Selection Demo (5 Objectives)")
    print("=" * 70 + "\n")

    # Show raw scores first (including efficiency)
    scores = get_method_scores(df)
    print("RAW METHOD SCORES (Mean across 5 replicates, including efficiency)")
    print("Note: efficiency_time is in minutes (lower = better)")
    print("-" * 60)
    print(scores.round(4).to_string())
    print()

    # Show normalized scores
    normalized = normalize_scores(scores)
    print("NORMALIZED SCORES (0-1 scale, higher = better)")
    print("-" * 60)
    print(normalized.round(4).to_string())
    print()

    # Demo each profile
    output_dir = Path(filepath).parent / "figures"
    results = demo_all_profiles(df, output_dir)

    # Print detailed report for balanced profile
    print("\n" + "=" * 70)
    print("DETAILED REPORT: Balanced Profile")
    print("=" * 70 + "\n")
    print(results["balanced"]["report"])

    return results
