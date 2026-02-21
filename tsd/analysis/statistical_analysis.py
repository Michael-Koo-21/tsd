"""
Statistical analysis of synthetic data generation experiment results.

Performs:
- Descriptive statistics (mean, std, 95% CI) by method
- ANOVA/Kruskal-Wallis tests for significant differences
- Post-hoc pairwise comparisons with multiple testing correction
- Effect size calculations
- Correlation analysis between metrics (trade-off analysis)
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_results(filepath: str | Path) -> pd.DataFrame:
    """Load experiment results from CSV."""
    df = pd.read_csv(filepath)
    return df


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate descriptive statistics for each method and metric.

    Returns DataFrame with mean, std, 95% CI, min, max for each metric by method.
    """
    metrics = ["fidelity_auc", "privacy_dcr", "utility_tstr", "fairness_gap"]

    results = []
    for method in df["method"].unique():
        method_data = df[df["method"] == method]

        for metric in metrics:
            values = method_data[metric].dropna()
            if len(values) == 0:
                continue

            mean = values.mean()
            std = values.std(ddof=1)
            se = std / np.sqrt(len(values))
            ci_lower = mean - 1.96 * se
            ci_upper = mean + 1.96 * se

            results.append(
                {
                    "method": method,
                    "metric": metric,
                    "n": len(values),
                    "mean": mean,
                    "std": std,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "min": values.min(),
                    "max": values.max(),
                }
            )

    return pd.DataFrame(results)


def check_normality(df: pd.DataFrame) -> dict:
    """
    Check normality assumption using Shapiro-Wilk test.

    Returns dict with test results for each method-metric combination.
    """
    metrics = ["fidelity_auc", "privacy_dcr", "utility_tstr", "fairness_gap"]
    results = {}

    for metric in metrics:
        results[metric] = {}
        for method in df["method"].unique():
            values = df[df["method"] == method][metric].dropna()
            if len(values) >= 3:
                stat, p = stats.shapiro(values)
                results[metric][method] = {"statistic": stat, "p_value": p, "normal": p > 0.05}

    return results


def levene_test(df: pd.DataFrame) -> dict:
    """
    Check homogeneity of variances using Levene's test.
    """
    metrics = ["fidelity_auc", "privacy_dcr", "utility_tstr", "fairness_gap"]
    results = {}

    for metric in metrics:
        groups = [
            df[df["method"] == m][metric].dropna().values
            for m in df["method"].unique()
            if len(df[df["method"] == m][metric].dropna()) > 0
        ]
        if len(groups) >= 2:
            stat, p = stats.levene(*groups)
            results[metric] = {"statistic": stat, "p_value": p, "equal_variance": p > 0.05}

    return results


def omnibus_tests(df: pd.DataFrame) -> dict:
    """
    Perform omnibus tests (ANOVA and Kruskal-Wallis) for each metric.

    Returns dict with test statistics, p-values, and effect sizes.
    """
    metrics = ["fidelity_auc", "privacy_dcr", "utility_tstr", "fairness_gap"]
    results = {}

    for metric in metrics:
        groups = []
        method_names = []
        for method in df["method"].unique():
            values = df[df["method"] == method][metric].dropna().values
            if len(values) > 0:
                groups.append(values)
                method_names.append(method)

        if len(groups) < 2:
            continue

        # One-way ANOVA
        f_stat, anova_p = stats.f_oneway(*groups)

        # Kruskal-Wallis (non-parametric alternative)
        h_stat, kw_p = stats.kruskal(*groups)

        # Effect size: eta-squared for ANOVA
        all_values = np.concatenate(groups)
        grand_mean = all_values.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = sum((v - grand_mean) ** 2 for v in all_values)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        # Effect size interpretation
        if eta_squared < 0.01:
            effect_interp = "negligible"
        elif eta_squared < 0.06:
            effect_interp = "small"
        elif eta_squared < 0.14:
            effect_interp = "medium"
        else:
            effect_interp = "large"

        results[metric] = {
            "anova": {"F": f_stat, "p_value": anova_p},
            "kruskal_wallis": {"H": h_stat, "p_value": kw_p},
            "eta_squared": eta_squared,
            "effect_interpretation": effect_interp,
            "methods": method_names,
            "n_groups": len(groups),
        }

    return results


def pairwise_comparisons(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Perform pairwise comparisons with Bonferroni correction.

    Uses t-tests and Mann-Whitney U tests for robustness.
    """
    metrics = ["fidelity_auc", "privacy_dcr", "utility_tstr", "fairness_gap"]
    results = {}

    methods = df["method"].unique()
    n_comparisons = len(list(combinations(methods, 2)))
    adjusted_alpha = alpha / n_comparisons  # Bonferroni correction

    for metric in metrics:
        results[metric] = {
            "comparisons": [],
            "n_comparisons": n_comparisons,
            "adjusted_alpha": adjusted_alpha,
        }

        for m1, m2 in combinations(methods, 2):
            v1 = df[df["method"] == m1][metric].dropna().values
            v2 = df[df["method"] == m2][metric].dropna().values

            if len(v1) < 2 or len(v2) < 2:
                continue

            # Independent t-test
            t_stat, t_p = stats.ttest_ind(v1, v2)

            # Mann-Whitney U (non-parametric)
            u_stat, mw_p = stats.mannwhitneyu(v1, v2, alternative="two-sided")

            # Cohen's d effect size
            pooled_std = np.sqrt(
                ((len(v1) - 1) * v1.std(ddof=1) ** 2 + (len(v2) - 1) * v2.std(ddof=1) ** 2)
                / (len(v1) + len(v2) - 2)
            )
            cohens_d = (v1.mean() - v2.mean()) / pooled_std if pooled_std > 0 else 0

            # Effect size interpretation
            abs_d = abs(cohens_d)
            if abs_d < 0.2:
                d_interp = "negligible"
            elif abs_d < 0.5:
                d_interp = "small"
            elif abs_d < 0.8:
                d_interp = "medium"
            else:
                d_interp = "large"

            results[metric]["comparisons"].append(
                {
                    "method_1": m1,
                    "method_2": m2,
                    "mean_diff": v1.mean() - v2.mean(),
                    "t_test": {"t": t_stat, "p": t_p, "significant": t_p < adjusted_alpha},
                    "mann_whitney": {"U": u_stat, "p": mw_p, "significant": mw_p < adjusted_alpha},
                    "cohens_d": cohens_d,
                    "effect_interpretation": d_interp,
                }
            )

    return results


def correlation_analysis(df: pd.DataFrame) -> dict:
    """
    Analyze correlations between metrics to identify trade-offs.

    Returns Pearson and Spearman correlations with interpretation.
    This validates the measure independence assumption required for
    the additive multiattribute value model.
    """
    metrics = ["fidelity_auc", "privacy_dcr", "utility_tstr", "fairness_gap"]

    # Use method means for correlation analysis
    method_means = df.groupby("method")[metrics].mean()

    results = {
        "pearson": {},
        "spearman": {},
        "method_means": method_means.to_dict(),
        "independence_validation": {},
    }

    # Expected correlations from pre-registration
    expected_correlations = {
        "fidelity_auc_vs_utility_tstr": (0.4, 0.8),  # moderate positive
        "fidelity_auc_vs_privacy_dcr": (-0.6, -0.2),  # moderate negative
        "utility_tstr_vs_privacy_dcr": (-0.7, -0.3),  # moderate negative
        "fidelity_auc_vs_fairness_gap": (-0.1, 0.5),  # weak positive
    }

    for m1, m2 in combinations(metrics, 2):
        v1 = method_means[m1].values
        v2 = method_means[m2].values

        # Pearson correlation
        r, p = stats.pearsonr(v1, v2)
        results["pearson"][f"{m1}_vs_{m2}"] = {"r": r, "p_value": p}

        # Spearman correlation
        rho, p = stats.spearmanr(v1, v2)
        results["spearman"][f"{m1}_vs_{m2}"] = {"rho": rho, "p_value": p}

        # Check if correlation matches pre-registered expectation
        pair_key = f"{m1}_vs_{m2}"
        if pair_key in expected_correlations:
            exp_low, exp_high = expected_correlations[pair_key]
            within_expected = exp_low <= rho <= exp_high
            results["independence_validation"][pair_key] = {
                "observed_rho": rho,
                "expected_range": (exp_low, exp_high),
                "within_expected": within_expected,
            }

    # Check for high correlations that would violate independence
    high_correlation_pairs = []
    for pair, vals in results["spearman"].items():
        if abs(vals["rho"]) > 0.85:
            high_correlation_pairs.append(
                {
                    "pair": pair,
                    "rho": vals["rho"],
                    "warning": "High correlation may violate preferential independence assumption",
                }
            )

    results["high_correlation_warnings"] = high_correlation_pairs
    results["independence_valid"] = len(high_correlation_pairs) == 0

    return results


def data_quality_check(df: pd.DataFrame) -> dict:
    """
    Check for data quality issues.
    """
    issues = []

    # Check for constant values within method (suggests bug)
    for method in df["method"].unique():
        method_data = df[df["method"] == method]
        for col in ["fidelity_auc", "privacy_dcr", "utility_tstr", "fairness_gap"]:
            values = method_data[col].dropna()
            if len(values) > 1 and values.std() == 0:
                issues.append(
                    {
                        "type": "constant_values",
                        "method": method,
                        "metric": col,
                        "message": f"{method} has identical {col} values across all replicates",
                    }
                )

    # Check for missing values
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            issues.append(
                {
                    "type": "missing_values",
                    "column": col,
                    "count": missing,
                    "message": f"{col} has {missing} missing values",
                }
            )

    return {"issues": issues, "n_issues": len(issues)}


def generate_report(df: pd.DataFrame) -> str:
    """
    Generate a comprehensive statistical analysis report.
    """
    report = []
    report.append("=" * 70)
    report.append("STATISTICAL ANALYSIS REPORT")
    report.append("Synthetic Data Generation Method Comparison")
    report.append("=" * 70)
    report.append("")

    # Data overview
    report.append("1. DATA OVERVIEW")
    report.append("-" * 40)
    report.append(f"Total observations: {len(df)}")
    report.append(f"Methods: {', '.join(df['method'].unique())}")
    report.append(f"Replicates per method: {df.groupby('method').size().unique()[0]}")
    report.append("")

    # Data quality
    quality = data_quality_check(df)
    if quality["n_issues"] > 0:
        report.append("⚠️  DATA QUALITY WARNINGS")
        report.append("-" * 40)
        for issue in quality["issues"]:
            report.append(f"  - {issue['message']}")
        report.append("")

    # Descriptive statistics
    report.append("2. DESCRIPTIVE STATISTICS")
    report.append("-" * 40)
    desc = descriptive_statistics(df)

    for metric in desc["metric"].unique():
        report.append(f"\n{metric.upper()}:")
        metric_data = desc[desc["metric"] == metric].sort_values("mean", ascending=False)
        report.append(f"{'Method':<22} {'Mean':>8} {'Std':>8} {'95% CI':>20}")
        report.append("-" * 60)
        for _, row in metric_data.iterrows():
            ci = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
            report.append(f"{row['method']:<22} {row['mean']:>8.4f} {row['std']:>8.4f} {ci:>20}")

    report.append("")

    # Omnibus tests
    report.append("3. OMNIBUS TESTS (Overall Method Differences)")
    report.append("-" * 40)
    omnibus = omnibus_tests(df)

    for metric, results in omnibus.items():
        report.append(f"\n{metric.upper()}:")
        report.append(
            f"  ANOVA: F={results['anova']['F']:.3f}, p={results['anova']['p_value']:.4f}"
        )
        report.append(
            f"  Kruskal-Wallis: H={results['kruskal_wallis']['H']:.3f}, p={results['kruskal_wallis']['p_value']:.4f}"
        )
        report.append(
            f"  Effect size (η²): {results['eta_squared']:.4f} ({results['effect_interpretation']})"
        )

        if results["anova"]["p_value"] < 0.05:
            report.append("  → Significant differences exist between methods")
        else:
            report.append("  → No significant differences detected")

    report.append("")

    # Pairwise comparisons
    report.append("4. PAIRWISE COMPARISONS (Bonferroni corrected)")
    report.append("-" * 40)
    pairwise = pairwise_comparisons(df)

    for metric, results in pairwise.items():
        report.append(f"\n{metric.upper()} (α_adj = {results['adjusted_alpha']:.4f}):")
        sig_comparisons = [c for c in results["comparisons"] if c["t_test"]["significant"]]

        if sig_comparisons:
            report.append("  Significant differences:")
            for c in sig_comparisons:
                direction = ">" if c["mean_diff"] > 0 else "<"
                report.append(
                    f"    {c['method_1']} {direction} {c['method_2']}: "
                    f"d={c['cohens_d']:.3f} ({c['effect_interpretation']}), "
                    f"p={c['t_test']['p']:.4f}"
                )
        else:
            report.append("  No significant pairwise differences after correction")

    report.append("")

    # Correlation analysis
    report.append("5. METRIC CORRELATIONS (Trade-off Analysis)")
    report.append("-" * 40)
    corr = correlation_analysis(df)

    report.append("Spearman correlations between method-level means:")
    for pair, vals in corr["spearman"].items():
        interp = (
            "strong" if abs(vals["rho"]) > 0.7 else "moderate" if abs(vals["rho"]) > 0.4 else "weak"
        )
        direction = "positive" if vals["rho"] > 0 else "negative"
        report.append(f"  {pair}: ρ={vals['rho']:.3f} ({interp} {direction})")

    report.append("")

    # Summary
    report.append("6. KEY FINDINGS SUMMARY")
    report.append("-" * 40)

    # Find best methods per metric
    desc_pivot = desc.pivot(index="method", columns="metric", values="mean")

    report.append("Best performing method by metric:")
    report.append(
        f"  • Fidelity AUC (lower=better, closer to 0.5): {desc_pivot['fidelity_auc'].idxmin()}"
    )
    report.append(f"  • Privacy DCR (higher=better): {desc_pivot['privacy_dcr'].idxmax()}")
    report.append(f"  • Utility TSTR (higher=better): {desc_pivot['utility_tstr'].idxmax()}")
    report.append(f"  • Fairness Gap (lower=better): {desc_pivot['fairness_gap'].idxmin()}")

    report.append("")
    report.append("=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)

    return "\n".join(report)


def run_analysis(filepath: str | Path) -> tuple[str, dict]:
    """
    Run complete statistical analysis and return report and raw results.
    """
    df = load_results(filepath)

    results = {
        "descriptive": descriptive_statistics(df),
        "normality": check_normality(df),
        "levene": levene_test(df),
        "omnibus": omnibus_tests(df),
        "pairwise": pairwise_comparisons(df),
        "correlations": correlation_analysis(df),
        "data_quality": data_quality_check(df),
    }

    report = generate_report(df)

    return report, results


if __name__ == "__main__":
    import sys

    filepath = sys.argv[1] if len(sys.argv) > 1 else "results/experiments/all_results_complete.csv"
    report, _ = run_analysis(filepath)
    print(report)
