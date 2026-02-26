"""
Regenerate MADA framework figures with value functions matching the paper.

Uses the exact value functions from Section 5.2 of the paper
to ensure figures match Table 4 values.
"""

from pathlib import Path

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


def load_and_compute_values(filepath, normalization="reference"):
    """Load results and compute value scores.

    Args:
        filepath: Path to all_results.csv
        normalization: "reference" for reference-anchored value functions
            (used in decision-focused analysis), or "minmax" for pure
            min-max normalization (used in VOI analysis and Table 5).
    """
    df = pd.read_csv(filepath)

    summary = df.groupby("method").agg(
        {
            "fidelity_auc": "mean",
            "privacy_dcr": "mean",
            "utility_tstr": "mean",
            "fairness_gap": "mean",
        }
    )

    if normalization == "minmax":
        return _compute_minmax_values(summary)

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


# Direction: True = higher is better, False = lower is better
_METRIC_DIRECTION = {
    "fidelity_auc": False,
    "privacy_dcr": True,
    "utility_tstr": True,
    "fairness_gap": False,
}

_METRIC_TO_OBJ = {
    "fidelity_auc": "fidelity",
    "privacy_dcr": "privacy",
    "utility_tstr": "utility",
    "fairness_gap": "fairness",
}


def _compute_minmax_values(summary):
    """Min-max normalization matching VOI analysis and paper Table 5."""
    value_scores = {}

    for method in summary.index:
        value_scores[method] = {}

    for metric, obj in _METRIC_TO_OBJ.items():
        col = summary[metric]
        min_val, max_val = col.min(), col.max()
        for method in summary.index:
            if max_val > min_val:
                v = (col[method] - min_val) / (max_val - min_val)
            else:
                v = 0.5
            if not _METRIC_DIRECTION[metric]:
                v = 1 - v
            value_scores[method][obj] = v

    for method in summary.index:
        efficiency = DEFAULT_RUNTIME_MINUTES.get(method, 1.0)
        value_scores[method]["efficiency"] = v_efficiency(efficiency)

    return value_scores


def calculate_weighted_score(value_scores, method, weights):
    """Calculate weighted aggregate score."""
    vs = value_scores[method]
    return sum(weights[k] * vs[k] for k in weights)


def generate_mada_profile_comparison(value_scores, output_dir):
    """Generate the MADA profile comparison figure."""
    import matplotlib.pyplot as plt

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
    """Generate the Pareto frontier figure (Privacy vs Fidelity).

    Both axes use higher = better:
      x-axis: Privacy (DCR 5th Percentile) — higher DCR = better privacy
      y-axis: Fidelity (1 - Propensity AUC) — higher value = better fidelity
    """
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    summary = df.groupby("method").agg({"fidelity_auc": "mean", "privacy_dcr": "mean"})

    fig, ax = plt.subplots(figsize=(10, 8))

    for method in summary.index:
        priv = summary.loc[method, "privacy_dcr"]
        fid = 1 - summary.loc[method, "fidelity_auc"]

        ax.scatter(
            priv, fid, s=200, c=COLORS[method], edgecolors="black", linewidth=1.5, zorder=3
        )

        offset = (10, 10) if method != "independent" else (10, -15)
        ax.annotate(
            METHOD_LABELS[method],
            (priv, fid),
            textcoords="offset points",
            xytext=offset,
            fontsize=10,
            fontweight="bold",
        )

    # Identify Pareto-optimal methods (higher on both axes = better)
    pareto_methods = []
    for method in summary.index:
        priv = summary.loc[method, "privacy_dcr"]
        fid = 1 - summary.loc[method, "fidelity_auc"]
        dominated = False
        for other in summary.index:
            if other == method:
                continue
            o_priv = summary.loc[other, "privacy_dcr"]
            o_fid = 1 - summary.loc[other, "fidelity_auc"]
            if o_priv >= priv and o_fid >= fid and (o_priv > priv or o_fid > fid):
                dominated = True
                break
        if not dominated:
            pareto_methods.append((priv, fid, method))

    pareto_methods.sort(key=lambda x: x[0])

    for priv, fid, _method in pareto_methods:
        ax.scatter(priv, fid, s=350, facecolors="none", edgecolors="gold", linewidth=3, zorder=2)

    if len(pareto_methods) > 1:
        privs = [p[0] for p in pareto_methods]
        fids = [p[1] for p in pareto_methods]
        ax.plot(privs, fids, "k--", linewidth=2, alpha=0.5, zorder=1)

    ax.set_xlabel("Privacy (DCR 5th Percentile) \u2192", fontsize=12)
    ax.set_ylabel("Fidelity (1 \u2212 Propensity AUC) \u2192", fontsize=12)
    ax.set_title(
        "Pareto Frontier: Privacy vs Fidelity\n(Gold circles = Pareto optimal)",
        fontsize=14,
        fontweight="bold",
    )

    for m in summary.index:
        ax.scatter([], [], c=COLORS[m], s=100, label=METHOD_LABELS[m])
    ax.legend(loc="center right", fontsize=9)

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
    # Profile comparison uses min-max normalization to match paper Table 5
    value_scores_minmax = load_and_compute_values(results_path, normalization="minmax")

    generate_mada_profile_comparison(value_scores_minmax, output_dir)
    generate_pareto_frontier(df, output_dir)

    # Also generate all standard visualizations
    from tsd.analysis.visualizations import generate_all_visualizations

    generate_all_visualizations(results_path, output_dir)

    print("\nAll figures regenerated.")
