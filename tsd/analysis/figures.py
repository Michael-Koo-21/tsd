"""
Regenerate MADA framework figures with value functions matching the paper.

Uses the exact value functions from Section 5.2 of the paper
to ensure figures match Table 4 values.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsd.constants import ARCHETYPES, COLORS, DEFAULT_RUNTIME_MINUTES, METHOD_LABELS


def v_fidelity(x):
    """Lower AUC = better. AUC=0.5 is ideal, x_worst = Ind. Marginals AUC"""
    x_worst = 0.880
    v = (x_worst - x) / (x_worst - 0.50)
    return max(0, min(1, v))


def v_privacy(x):
    """Higher DCR = better privacy"""
    x_worst = 0.007  # Synthpop (worst)
    x_best = 0.140  # Ind. Marginals (best)
    v = (x - x_worst) / (x_best - x_worst)
    return max(0, min(1, v))


def v_utility(x):
    """Higher TSTR = better. 1.0 is ideal, x_worst = Ind. Marginals"""
    x_worst = 0.037
    v = (x - x_worst) / (1.0 - x_worst)
    return max(0, min(1, v))


def v_fairness(x):
    """Lower gap = better. x_worst = Ind. Marginals gap"""
    x_worst = 0.080
    v = (x_worst - x) / x_worst
    return max(0, min(1, v))


def v_efficiency(x):
    """Logarithmic. x_best ~ 0.5 min, ceiling at 480 min"""
    x_best = 0.5
    if x <= x_best:
        return 1.0
    v = 1 - np.log(x / x_best) / np.log(480 / x_best)
    return max(0, min(1, v))


def load_and_compute_values(filepath):
    """Load results and compute value scores using paper's value functions."""
    df = pd.read_csv(filepath)

    summary = df.groupby("method").agg(
        {
            "fidelity_auc": "mean",
            "privacy_dcr": "mean",
            "utility_tstr": "mean",
            "fairness_gap": "mean",
        }
    )

    value_scores = {}
    for method in summary.index:
        efficiency = DEFAULT_RUNTIME_MINUTES.get(method, 1.0)
        value_scores[method] = {
            "fidelity": v_fidelity(summary.loc[method, "fidelity_auc"]),
            "privacy": v_privacy(summary.loc[method, "privacy_dcr"]),
            "utility": v_utility(summary.loc[method, "utility_tstr"]),
            "fairness": v_fairness(summary.loc[method, "fairness_gap"]),
            "efficiency": v_efficiency(efficiency),
        }

    return value_scores


def calculate_weighted_score(value_scores, method, weights):
    """Calculate weighted aggregate score."""
    vs = value_scores[method]
    return sum(weights[k] * vs[k] for k in weights)


def generate_mada_profile_comparison(value_scores, output_dir):
    """Generate the MADA profile comparison figure."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    methods = list(value_scores.keys())

    for idx, (_arch_id, arch) in enumerate(ARCHETYPES.items()):
        ax = axes[idx]
        weights = arch["weights"]

        scores = [(m, calculate_weighted_score(value_scores, m, weights)) for m in methods]
        scores.sort(key=lambda x: x[1], reverse=True)

        methods_sorted = [s[0] for s in scores]
        scores_vals = [s[1] for s in scores]

        colors = [
            "gold" if i == 0 else "silver" if i == 1 else "peru" if i == 2 else "lightgray"
            for i in range(len(methods_sorted))
        ]

        bars = ax.barh(
            range(len(methods_sorted)),
            scores_vals,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_yticks(range(len(methods_sorted)))
        ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods_sorted], fontsize=9)
        ax.set_xlabel("Weighted Score")
        ax.set_title(f"{arch['name']}\n{arch['description']}", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1)

        for i, (_bar, score) in enumerate(zip(bars, scores_vals)):
            ax.text(score + 0.02, i, f"{score:.3f}", va="center", fontsize=9)

    plt.suptitle(
        "MADA Framework: Method Rankings by Stakeholder Archetype (5 Objectives)\n"
        "(Gold = Recommended)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    fig_path = Path(output_dir) / "mada_profile_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {fig_path}")
    return fig_path


def generate_pareto_frontier(df, output_dir):
    """Generate the Pareto frontier figure (Fidelity vs Utility)."""
    plt.style.use("seaborn-v0_8-whitegrid")
    summary = df.groupby("method").agg({"fidelity_auc": "mean", "utility_tstr": "mean"})

    fig, ax = plt.subplots(figsize=(10, 8))

    for method in summary.index:
        fid = summary.loc[method, "fidelity_auc"]
        util = summary.loc[method, "utility_tstr"]

        ax.scatter(fid, util, s=200, c=COLORS[method], edgecolors="black", linewidth=1.5, zorder=3)

        offset = (10, 10) if method != "synthpop" else (-60, 10)
        ax.annotate(
            METHOD_LABELS[method],
            (fid, util),
            textcoords="offset points",
            xytext=offset,
            fontsize=10,
            fontweight="bold",
        )

    # Identify Pareto-optimal methods
    pareto_methods = []
    for method in summary.index:
        fid = summary.loc[method, "fidelity_auc"]
        util = summary.loc[method, "utility_tstr"]
        dominated = False
        for other in summary.index:
            if other == method:
                continue
            o_fid = summary.loc[other, "fidelity_auc"]
            o_util = summary.loc[other, "utility_tstr"]
            if o_fid <= fid and o_util >= util and (o_fid < fid or o_util > util):
                dominated = True
                break
        if not dominated:
            pareto_methods.append((fid, util, method))

    pareto_methods.sort(key=lambda x: x[0])

    for fid, util, _method in pareto_methods:
        ax.scatter(fid, util, s=350, facecolors="none", edgecolors="gold", linewidth=3, zorder=2)

    if len(pareto_methods) > 1:
        fids = [p[0] for p in pareto_methods]
        utils = [p[1] for p in pareto_methods]
        ax.plot(fids, utils, "k--", linewidth=2, alpha=0.5, zorder=1)

    ax.set_xlabel("Fidelity (Propensity AUC) \u2192", fontsize=12)
    ax.set_ylabel("Utility (TSTR) \u2192", fontsize=12)
    ax.set_title(
        "Pareto Frontier: Fidelity vs Utility\n(Gold circles = Pareto optimal)",
        fontsize=14,
        fontweight="bold",
    )

    for m in summary.index:
        ax.scatter([], [], c=COLORS[m], s=100, label=METHOD_LABELS[m])
    ax.legend(loc="lower left", fontsize=9)

    ax.axhline(y=0.7, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=0.85, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlim(0.4, 0.95)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = Path(output_dir) / "pareto_frontier.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {fig_path}")
    return fig_path


def regenerate_paper_figures(results_path: str | Path, output_dir: str | Path):
    """
    Regenerate all paper figures.

    Args:
        results_path: Path to all_results.csv
        output_dir: Directory to write figure PNGs
    """
    results_path = Path(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_path)
    value_scores = load_and_compute_values(results_path)

    generate_mada_profile_comparison(value_scores, output_dir)
    generate_pareto_frontier(df, output_dir)

    # Also generate all standard visualizations
    from tsd.analysis.visualizations import generate_all_visualizations

    generate_all_visualizations(results_path, output_dir)

    print("\nAll figures regenerated.")


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "results/experiments/all_results.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "results/experiments/figures"
    regenerate_paper_figures(path, out)
