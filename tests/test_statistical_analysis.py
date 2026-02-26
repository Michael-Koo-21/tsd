"""Tests for tsd.analysis.statistical_analysis."""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from tsd.analysis.statistical_analysis import (
    correlation_analysis,
    data_quality_check,
    descriptive_statistics,
    generate_report,
    middle_tier_pairwise_tests,
    omnibus_tests,
    pairwise_comparisons,
    replicate_level_correlations,
)


class TestDescriptiveStatistics:
    def test_returns_dataframe(self, sample_results):
        result = descriptive_statistics(sample_results)
        assert isinstance(result, pd.DataFrame)
        assert "method" in result.columns
        assert "metric" in result.columns
        assert "mean" in result.columns

    def test_all_methods_present(self, sample_results):
        result = descriptive_statistics(sample_results)
        methods = result["method"].unique()
        assert len(methods) == 5

    def test_four_metrics(self, sample_results):
        result = descriptive_statistics(sample_results)
        metrics = result["metric"].unique()
        assert len(metrics) == 4


class TestOmnibusTests:
    def test_returns_dict(self, sample_results):
        result = omnibus_tests(sample_results)
        assert isinstance(result, dict)
        assert "fidelity_auc" in result

    def test_contains_anova_and_kruskal(self, sample_results):
        result = omnibus_tests(sample_results)
        for metric, data in result.items():
            assert "anova" in data
            assert "kruskal_wallis" in data
            assert "eta_squared" in data

    def test_effect_size_nonnegative(self, sample_results):
        result = omnibus_tests(sample_results)
        for metric, data in result.items():
            assert data["eta_squared"] >= 0


class TestPairwiseComparisons:
    def test_returns_dict(self, sample_results):
        result = pairwise_comparisons(sample_results)
        assert isinstance(result, dict)
        assert "fidelity_auc" in result

    def test_comparisons_present(self, sample_results):
        result = pairwise_comparisons(sample_results)
        for metric, data in result.items():
            assert "comparisons" in data
            # 5 methods -> C(5,2) = 10 pairwise comparisons
            assert len(data["comparisons"]) == 10

    def test_bonferroni_correction(self, sample_results):
        result = pairwise_comparisons(sample_results, alpha=0.05)
        for metric, data in result.items():
            assert data["adjusted_alpha"] < 0.05


class TestCorrelationAnalysis:
    def test_returns_dict(self, sample_results):
        result = correlation_analysis(sample_results)
        assert isinstance(result, dict)
        assert "pearson" in result
        assert "spearman" in result

    def test_correlation_range(self, sample_results):
        result = correlation_analysis(sample_results)
        for pair, vals in result["spearman"].items():
            assert -1 <= vals["rho"] <= 1


class TestDataQualityCheck:
    def test_returns_dict(self, sample_results):
        result = data_quality_check(sample_results)
        assert "issues" in result
        assert "n_issues" in result

    def test_no_issues_in_sample(self, sample_results):
        result = data_quality_check(sample_results)
        # sample_results has random data, so no constant values or missing
        assert result["n_issues"] == 0

    def test_detects_constant_values(self):
        df = pd.DataFrame(
            {
                "method": ["a"] * 3 + ["b"] * 3,
                "fidelity_auc": [0.5, 0.5, 0.5, 0.6, 0.7, 0.8],
                "privacy_dcr": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
                "utility_tstr": [0.9, 0.9, 0.9, 0.8, 0.8, 0.8],
                "fairness_gap": [0.05, 0.06, 0.07, 0.05, 0.06, 0.07],
            }
        )
        result = data_quality_check(df)
        assert result["n_issues"] >= 1
        types = [i["type"] for i in result["issues"]]
        assert "constant_values" in types


class TestMiddleTierPairwiseTests:
    def test_returns_dict(self, sample_results):
        result = middle_tier_pairwise_tests(sample_results)
        assert isinstance(result, dict)
        assert "methods" in result
        assert "summary" in result

    def test_default_methods(self, sample_results):
        result = middle_tier_pairwise_tests(sample_results)
        assert result["methods"] == ["ctgan", "dpbn", "great"]

    def test_custom_methods(self, sample_results):
        result = middle_tier_pairwise_tests(sample_results, methods=["ctgan", "dpbn"])
        assert result["methods"] == ["ctgan", "dpbn"]
        assert result["n_comparisons"] == 1

    def test_comparisons_per_metric(self, sample_results):
        result = middle_tier_pairwise_tests(sample_results)
        # 3 methods -> C(3,2) = 3 pairwise comparisons
        for metric in ["fidelity_auc", "privacy_dcr", "utility_tstr", "fairness_gap"]:
            assert len(result[metric]["comparisons"]) == 3

    def test_bonferroni_correction(self, sample_results):
        result = middle_tier_pairwise_tests(sample_results)
        # 3 comparisons, alpha=0.05 -> adjusted = 0.05/3
        assert result["adjusted_alpha"] < 0.05

    def test_summary_has_interpretation(self, sample_results):
        result = middle_tier_pairwise_tests(sample_results)
        assert "interpretation" in result["summary"]
        assert "any_distinguishable" in result["summary"]


class TestDescriptiveStatisticsTDistribution:
    """Verify the CI uses t-distribution, not z=1.96."""

    def test_ci_uses_t_distribution(self, sample_results):
        result = descriptive_statistics(sample_results)
        # Pick one row to verify
        row = result.iloc[0]
        n = int(row["n"])
        mean = row["mean"]
        std = row["std"]
        se = std / np.sqrt(n)
        t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
        expected_lower = mean - t_crit * se
        expected_upper = mean + t_crit * se
        assert row["ci_lower"] == expected_lower
        assert row["ci_upper"] == expected_upper

    def test_ci_wider_than_z_based(self, sample_results):
        result = descriptive_statistics(sample_results)
        for _, row in result.iterrows():
            n = int(row["n"])
            se = row["std"] / np.sqrt(n)
            z_width = 2 * 1.96 * se
            actual_width = row["ci_upper"] - row["ci_lower"]
            # t-based CI should be wider than z-based for small n
            if se > 0:
                assert actual_width > z_width


class TestReplicateLevelCorrelations:
    def test_returns_dict(self, sample_results):
        result = replicate_level_correlations(sample_results)
        assert isinstance(result, dict)
        assert "spearman" in result
        assert "pearson" in result
        assert "n_replicates" in result

    def test_n_replicates_matches(self, sample_results):
        result = replicate_level_correlations(sample_results)
        assert result["n_replicates"] == len(sample_results)

    def test_correlation_range(self, sample_results):
        result = replicate_level_correlations(sample_results)
        for pair, vals in result["spearman"].items():
            assert -1 <= vals["rho"] <= 1
            assert vals["p_value"] >= 0
            assert vals["n"] > 0

    def test_more_observations_than_method_means(self, sample_results):
        result = replicate_level_correlations(sample_results)
        for pair, vals in result["spearman"].items():
            # 5 methods x 3 replicates = 15 > 5 method means
            assert vals["n"] == 15


class TestGenerateReport:
    def test_returns_string(self, sample_results):
        report = generate_report(sample_results)
        assert isinstance(report, str)

    def test_contains_all_sections(self, sample_results):
        report = generate_report(sample_results)
        assert "DATA OVERVIEW" in report
        assert "DESCRIPTIVE STATISTICS" in report
        assert "OMNIBUS TESTS" in report
        assert "PAIRWISE COMPARISONS" in report
        assert "METRIC CORRELATIONS" in report
        assert "REPLICATE-LEVEL CORRELATIONS" in report
        assert "KEY FINDINGS" in report

    def test_contains_method_count(self, sample_results):
        report = generate_report(sample_results)
        assert "Total observations: 15" in report
