"""
Regenerate MADA framework figures with value functions matching the paper.

Uses the exact value functions from Section 5.2 of the paper
to ensure figures match Table 4 values.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from tsd.constants import ARCHETYPES, COLORS, DEFAULT_RUNTIME_MINUTES, METHOD_LABELS


def v_fidelity(x, x_worst=0.880, x_best=0.50):
    """Lower AUC = better. Defaults from ACS PUMS paper."""
    v = (x_worst - x) / (x_worst - x_best)
    return max(0, min(1, v))


def v_privacy(x, x_worst=0.007, x_best=0.140):
    """Higher DCR = better privacy. Defaults from ACS PUMS paper."""
    v = (x - x_worst) / (x_best - x_worst)
    return max(0, min(1, v))


def v_utility(x, x_worst=0.037, x_best=1.0):
    """Higher TSTR = better. Defaults from ACS PUMS paper."""
    v = (x - x_worst) / (x_best - x_worst)
    return max(0, min(1, v))


def v_fairness(x, x_worst=0.080, x_best=0.0):
    """Lower gap = better. Defaults from ACS PUMS paper."""
    v = (x_worst - x) / (x_worst - x_best)
    return max(0, min(1, v))


def v_efficiency(x, x_best=0.5, x_ceiling=480):
    """Logarithmic. Defaults from ACS PUMS paper."""
    if x <= x_best:
        return 1.0
    v = 1 - np.log(x / x_best) / np.log(x_ceiling / x_best)
    return max(0, min(1, v))


def load_and_compute_values(filepath, normalization="reference", value_refs=None):
    """Load results and compute value scores.

    Args:
        filepath: Path to all_results.csv
        normalization: "reference" for reference-anchored value functions
            (used in decision-focused analysis), "minmax" for pure
            min-max normalization (used in VOI analysis and Table 5),
            or "auto" for reference-anchored with data-derived reference points.
        value_refs: Optional dict of reference points for value functions,
            e.g. {"fidelity": {"x_worst": 0.9}, "privacy": {"x_best": 0.2}}.
            Only used with "reference" normalization. Unspecified refs use
            paper defaults.
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

    if normalization == "auto":
        value_refs = _derive_value_refs(summary)

    refs = value_refs or {}
    value_scores = {}
    for method in summary.index:
        efficiency = DEFAULT_RUNTIME_MINUTES.get(method, 1.0)
        value_scores[method] = {
            "fidelity": v_fidelity(summary.loc[method, "fidelity_auc"], **refs.get("fidelity", {})),
            "privacy": v_privacy(summary.loc[method, "privacy_dcr"], **refs.get("privacy", {})),
            "utility": v_utility(summary.loc[method, "utility_tstr"], **refs.get("utility", {})),
            "fairness": v_fairness(summary.loc[method, "fairness_gap"], **refs.get("fairness", {})),
            "efficiency": v_efficiency(efficiency, **refs.get("efficiency", {})),
        }

    return value_scores


def _derive_value_refs(summary):
    """Derive value function reference points from actual data ranges."""
    return {
        "fidelity": {
            "x_worst": summary["fidelity_auc"].max(),
            "x_best": 0.50,
        },
        "privacy": {
            "x_worst": summary["privacy_dcr"].min(),
            "x_best": summary["privacy_dcr"].max(),
        },
        "utility": {
            "x_worst": summary["utility_tstr"].min(),
            "x_best": 1.0,
        },
        "fairness": {
            "x_worst": summary["fairness_gap"].max(),
            "x_best": 0.0,
        },
    }


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

        ax.scatter(priv, fid, s=200, c=COLORS[method], edgecolors="black", linewidth=1.5, zorder=3)

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


def correlation_adjusted_analysis(value_scores):
    """Run MADA with fidelity and privacy collapsed into a single dimension.

    Merges fidelity and privacy into 'data_similarity' (simple average of the
    two value scores) and remaps archetype weights accordingly, normalizing
    to sum to 1.0. Reports whether the Synthpop vs DP-BN recommendation and
    the 0.40 privacy threshold survive dimension reduction.

    Args:
        value_scores: Dict of {method: {objective: value}} from load_and_compute_values.

    Returns:
        Dict with collapsed value scores, remapped archetype results, and
        breakeven analysis.
    """
    methods = list(value_scores.keys())

    # Collapse fidelity + privacy into data_similarity (average)
    collapsed_scores = {}
    for method in methods:
        vs = value_scores[method]
        collapsed_scores[method] = {
            "data_similarity": (vs["fidelity"] + vs["privacy"]) / 2.0,
            "utility": vs["utility"],
            "fairness": vs["fairness"],
            "efficiency": vs["efficiency"],
        }

    # Remap archetype weights: merge fidelity + privacy weights into data_similarity
    results = {}
    for arch_id, arch in ARCHETYPES.items():
        w = arch["weights"]
        collapsed_w = {
            "data_similarity": w["fidelity"] + w["privacy"],
            "utility": w["utility"],
            "fairness": w["fairness"],
            "efficiency": w["efficiency"],
        }

        # Compute weighted scores
        method_vals = {}
        for method in methods:
            cs = collapsed_scores[method]
            method_vals[method] = sum(collapsed_w[k] * cs[k] for k in collapsed_w)

        ranking = sorted(method_vals.items(), key=lambda x: x[1], reverse=True)

        results[arch_id] = {
            "name": arch["name"],
            "collapsed_weights": collapsed_w,
            "method_values": method_vals,
            "ranking": ranking,
            "top_method": ranking[0][0],
            "top_score": ranking[0][1],
        }

    # Breakeven analysis: find privacy threshold where DP-BN overtakes Synthpop
    # Vary data_similarity weight from 0.05 to 0.80 (representing combined fidelity+privacy)
    import numpy as np

    breakeven_threshold = None
    balanced_w = ARCHETYPES["balanced"]["weights"]
    other_objs = ["utility", "fairness", "efficiency"]
    other_total = sum(balanced_w[o] for o in other_objs)

    for ds_w in np.arange(0.05, 0.85, 0.01):
        remaining = 1.0 - ds_w
        weights = {"data_similarity": ds_w}
        for o in other_objs:
            weights[o] = balanced_w[o] / other_total * remaining

        method_vals = {}
        for method in methods:
            cs = collapsed_scores[method]
            method_vals[method] = sum(weights[k] * cs[k] for k in weights)

        top = max(method_vals, key=method_vals.get)
        if top == "dpbn" and breakeven_threshold is None:
            breakeven_threshold = ds_w

    return {
        "collapsed_scores": collapsed_scores,
        "archetype_results": results,
        "breakeven_data_similarity_weight": breakeven_threshold,
        "original_breakeven_privacy_weight": 0.40,
    }


def generate_tornado_sensitivity(value_scores, output_dir, baseline_weights=None):
    """Generate a one-way sensitivity tornado plot (Balanced Baseline).

    For each objective, varies its weight from 0.05 to 0.50 with proportional
    rescaling of remaining weights, and records the range of the top achievable
    value. Bars are sorted by impact magnitude (widest at top).

    Args:
        value_scores: Dict of {method: {objective: value}} from load_and_compute_values.
        output_dir: Directory to write the figure.
        baseline_weights: Baseline weight dict. Defaults to Balanced archetype.
    """
    import matplotlib.pyplot as plt

    if baseline_weights is None:
        baseline_weights = ARCHETYPES["balanced"]["weights"]

    objectives = list(baseline_weights.keys())
    methods = list(value_scores.keys())

    weight_lo, weight_hi = 0.05, 0.50

    # For each objective, compute the top method value at lo and hi weights
    results = []
    for obj in objectives:
        other_objs = [o for o in objectives if o != obj]
        other_total = sum(baseline_weights[o] for o in other_objs)

        values_at = {}
        for w in [weight_lo, weight_hi]:
            remaining = 1.0 - w
            weights = {obj: w}
            for o in other_objs:
                if other_total > 0:
                    weights[o] = baseline_weights[o] / other_total * remaining
                else:
                    weights[o] = remaining / len(other_objs)
            # Compute best method value at this weight
            method_vals = {m: calculate_weighted_score(value_scores, m, weights) for m in methods}
            values_at[w] = max(method_vals.values())

        lo_val = values_at[weight_lo]
        hi_val = values_at[weight_hi]
        results.append(
            {
                "objective": obj,
                "lo": min(lo_val, hi_val),
                "hi": max(lo_val, hi_val),
                "range": abs(hi_val - lo_val),
            }
        )

    # Sort by range (widest at top)
    results.sort(key=lambda r: r["range"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    y_positions = range(len(results))
    # Compute baseline value for the center line
    baseline_val = max(calculate_weighted_score(value_scores, m, baseline_weights) for m in methods)

    obj_labels = {
        "fidelity": "Fidelity",
        "privacy": "Privacy",
        "utility": "Utility",
        "fairness": "Fairness",
        "efficiency": "Efficiency",
    }
    obj_colors = {
        "fidelity": "#4C72B0",
        "privacy": "#55A868",
        "utility": "#C44E52",
        "fairness": "#8172B3",
        "efficiency": "#CCB974",
    }

    for i, r in enumerate(results):
        obj = r["objective"]
        ax.barh(
            i,
            r["hi"] - r["lo"],
            left=r["lo"],
            height=0.6,
            color=obj_colors.get(obj, "#999999"),
            edgecolor="black",
            linewidth=0.5,
        )
        # Annotate range
        ax.text(
            r["hi"] + 0.005,
            i,
            f"\u0394={r['range']:.3f}",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([obj_labels.get(r["objective"], r["objective"]) for r in results])
    ax.axvline(x=baseline_val, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Top Achievable Value")
    ax.set_title(
        "One-Way Sensitivity Tornado Plot (Balanced Baseline)\n"
        "Weight varied from 0.05 to 0.50; remaining weights rescaled proportionally",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlim(
        min(r["lo"] for r in results) - 0.02,
        max(r["hi"] for r in results) + 0.04,
    )

    plt.tight_layout()
    fig_path = Path(output_dir) / "tornado_sensitivity.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved: {fig_path}")
    return fig_path


def regenerate_paper_figures(results_path: str | Path, output_dir: str | Path, value_refs=None):
    """
    Regenerate all paper figures.

    Args:
        results_path: Path to all_results.csv
        output_dir: Directory to write figure PNGs
        value_refs: Optional value function reference points. Pass "auto" to
            derive from data, a dict to override specific refs, or None for
            paper defaults. Only affects reference-anchored normalization.
    """
    results_path = Path(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_path)
    normalization = "auto" if value_refs == "auto" else "minmax"
    norm_kwargs = {}
    if isinstance(value_refs, dict):
        normalization = "reference"
        norm_kwargs["value_refs"] = value_refs
    # Profile comparison uses min-max normalization to match paper Table 5
    value_scores_minmax = load_and_compute_values(
        results_path, normalization=normalization, **norm_kwargs
    )

    generate_mada_profile_comparison(value_scores_minmax, output_dir)
    generate_tornado_sensitivity(value_scores_minmax, output_dir)
    generate_pareto_frontier(df, output_dir)

    # Also generate all standard visualizations
    from tsd.analysis.visualizations import generate_all_visualizations

    generate_all_visualizations(results_path, output_dir)

    print("\nAll figures regenerated.")
