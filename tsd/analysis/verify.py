"""
Verify that paper claims match actual data.

Checks raw metrics, normalized values, weighted scores, and ranking
consistency against the paper tables.
"""

from pathlib import Path

import pandas as pd

from tsd.constants import ARCHETYPES, CORE_METRICS


def verify_claims(results_path: str | Path) -> str:
    """
    Run all verification checks and return a printable report.

    Args:
        results_path: Path to all_results.csv

    Returns:
        Multi-line string report.
    """
    df = pd.read_csv(results_path)
    lines: list[str] = []

    def p(text: str = ""):
        lines.append(text)

    means = df.groupby("method")[CORE_METRICS].mean()

    p("=" * 70)
    p("VERIFICATION OF PAPER CLAIMS")
    p("=" * 70)

    # 1. Raw means
    p("\n1. RAW MEANS (Table 5 in paper):")
    p("-" * 70)
    p(means.round(4).to_string())

    # 2. Specific claims
    p("\n2. VERIFICATION OF SPECIFIC CLAIMS:")
    p("-" * 70)

    checks = [
        ("synthpop", "fidelity_auc", 0.484, "Synthpop AUC"),
        ("synthpop", "utility_tstr", 1.0, "Synthpop TSTR"),
        ("independent_marginals", "privacy_dcr", 0.14, "Ind. Marg DCR"),
    ]
    for method, metric, expected, label in checks:
        actual = means.loc[method, metric]
        ok = abs(actual - expected) < 0.01
        p(f"   {label}: {actual:.4f} (expected ~{expected}) - {'MATCH' if ok else 'MISMATCH'}")

    # 3. Normalization check
    p("\n3. NORMALIZATION CHECK:")
    p("-" * 70)
    directions = {
        "fidelity_auc": False,
        "privacy_dcr": True,
        "utility_tstr": True,
        "fairness_gap": False,
    }
    normalized = means.copy()
    for metric in CORE_METRICS:
        col = means[metric]
        mn, mx = col.min(), col.max()
        if mx > mn:
            normalized[metric] = (col - mn) / (mx - mn)
        else:
            normalized[metric] = 0.5
        if not directions[metric]:
            normalized[metric] = 1 - normalized[metric]

    p("Normalized values [0-1, higher = better]:")
    p(normalized.round(4).to_string())

    # 4. Weighted scores
    p("\n4. WEIGHTED SCORES VERIFICATION (Table 4):")
    p("-" * 70)

    # Add efficiency estimates for weighted scoring
    efficiency = {
        "independent_marginals": 1.0,
        "synthpop": 1.0,
        "ctgan": 0.95,
        "dpbn": 1.0,
        "great": 0.24,
    }
    norm_with_eff = normalized.copy()
    norm_with_eff["efficiency"] = pd.Series(efficiency)

    metric_map_with_eff = {
        "fidelity_auc": "fidelity",
        "privacy_dcr": "privacy",
        "utility_tstr": "utility",
        "fairness_gap": "fairness",
        "efficiency": "efficiency",
    }

    for arch_name, arch in ARCHETYPES.items():
        weights = arch["weights"]
        scores = pd.Series(0.0, index=norm_with_eff.index)
        for metric, attr in metric_map_with_eff.items():
            if metric in norm_with_eff.columns:
                scores += norm_with_eff[metric] * weights.get(attr, 0)

        p(f"\n{arch_name.upper()}:")
        for method, score in scores.sort_values(ascending=False).items():
            p(f"   {method}: {score:.2f}")

    # 5. Pre-registration deviation
    p("\n5. PRE-REGISTRATION DEVIATION CHECK:")
    p("-" * 70)
    utility_ranking = means["utility_tstr"].sort_values(ascending=False)
    p(f"Utility ranking: {list(utility_ranking.index)}")
    p(f"Top method: {utility_ranking.index[0]}")

    # 6. Suspicious results
    p("\n6. SUSPICIOUS RESULTS CHECK:")
    p("-" * 70)
    synth_tstr = df[df["method"] == "synthpop"]["utility_tstr"].values
    p(f"Synthpop TSTR replicates: {synth_tstr.round(4)}")
    exceeds = int(sum(synth_tstr > 1.0))
    p(f"Replicates where TSTR > 1.0: {exceeds}/5")

    p("\n" + "=" * 70)
    p("VERIFICATION COMPLETE")
    p("=" * 70)

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "results/experiments/all_results.csv"
    print(verify_claims(path))
