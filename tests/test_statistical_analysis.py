"""Tests for tsd.analysis.statistical_analysis."""

import pandas as pd
import pytest

from tsd.analysis.statistical_analysis import (
    correlation_analysis,
    data_quality_check,
    descriptive_statistics,
    generate_report,
    omnibus_tests,
    pairwise_comparisons,
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
        df = pd.DataFrame({
            "method": ["a"] * 3 + ["b"] * 3,
            "fidelity_auc": [0.5, 0.5, 0.5, 0.6, 0.7, 0.8],
            "privacy_dcr": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
            "utility_tstr": [0.9, 0.9, 0.9, 0.8, 0.8, 0.8],
            "fairness_gap": [0.05, 0.06, 0.07, 0.05, 0.06, 0.07],
        })
        result = data_quality_check(df)
        assert result["n_issues"] >= 1
        types = [i["type"] for i in result["issues"]]
        assert "constant_values" in types


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
        assert "KEY FINDINGS" in report

    def test_contains_method_count(self, sample_results):
        report = generate_report(sample_results)
        assert "Total observations: 15" in report
