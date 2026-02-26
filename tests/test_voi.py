"""Tests for tsd.analysis.voi_analysis."""

import pandas as pd
import pytest

from tsd.analysis.voi_analysis import (
    compute_method_value,
    format_results_tables,
    get_method_performance,
    normalize_to_value,
    run_voi_analysis,
    strategy_s1_random,
    strategy_s2_heuristic,
    strategy_s4_benchmark,
)


class TestGetMethodPerformance:
    def test_returns_means_and_stds(self, sample_results):
        means, stds = get_method_performance(sample_results)
        assert isinstance(means, pd.DataFrame)
        assert isinstance(stds, pd.DataFrame)
        assert len(means) == 5
        assert len(stds) == 5

    def test_means_in_expected_range(self, sample_results):
        means, _ = get_method_performance(sample_results)
        assert (means["fidelity_auc"] >= 0).all()
        assert (means["fidelity_auc"] <= 1).all()


class TestNormalizeToValue:
    def test_output_range_zero_one(self, sample_results):
        means, _ = get_method_performance(sample_results)
        values = normalize_to_value(means)
        for col in values.columns:
            assert values[col].min() >= -1e-9
            assert values[col].max() <= 1.0 + 1e-9

    def test_constant_column_gets_half(self):
        df = pd.DataFrame(
            {
                "fidelity_auc": [0.7, 0.7],
                "privacy_dcr": [0.1, 0.1],
                "utility_tstr": [0.9, 0.9],
                "fairness_gap": [0.05, 0.05],
            },
            index=["a", "b"],
        )
        values = normalize_to_value(df)
        assert (values == 0.5).all().all()


class TestComputeMethodValue:
    def test_returns_series(self, sample_results):
        means, _ = get_method_performance(sample_results)
        values = normalize_to_value(means)
        weights = {
            "fidelity": 0.25,
            "privacy": 0.25,
            "utility": 0.25,
            "fairness": 0.15,
            "efficiency": 0.10,
        }
        result = compute_method_value(values, weights)
        assert isinstance(result, pd.Series)
        assert len(result) == 5

    def test_values_nonnegative(self, sample_results):
        means, _ = get_method_performance(sample_results)
        values = normalize_to_value(means)
        weights = {
            "fidelity": 0.25,
            "privacy": 0.25,
            "utility": 0.25,
            "fairness": 0.15,
            "efficiency": 0.10,
        }
        result = compute_method_value(values, weights)
        assert (result >= 0).all()


class TestStrategies:
    def test_s1_random_is_mean(self):
        vals = pd.Series([0.3, 0.5, 0.7], index=["a", "b", "c"])
        assert strategy_s1_random(vals) == pytest.approx(0.5)

    def test_s2_heuristic_privacy_first(self):
        vals = pd.Series(
            {"dpbn": 0.8, "ctgan": 0.6, "great": 0.5, "synthpop": 0.4, "independent_marginals": 0.3}
        )
        result = strategy_s2_heuristic(vals, "privacy_first")
        assert result == 0.8

    def test_s2_heuristic_unknown_archetype(self):
        vals = pd.Series(
            {"dpbn": 0.8, "ctgan": 0.6, "great": 0.5, "synthpop": 0.4, "independent_marginals": 0.3}
        )
        result = strategy_s2_heuristic(vals, "unknown")
        assert result == 0.4  # falls back to synthpop

    def test_s4_benchmark_is_max(self):
        vals = pd.Series([0.3, 0.9, 0.5], index=["a", "b", "c"])
        assert strategy_s4_benchmark(vals) == 0.9


class TestRunVOIAnalysis:
    def test_returns_list_of_results(self, sample_results):
        results = run_voi_analysis(sample_results, n_simulations=100, seed=42)
        assert len(results) == 3  # 3 archetypes
        for r in results:
            assert r.s4_benchmark >= r.s1_random

    def test_oracle_ge_s4(self, sample_results):
        """EVPI requires Oracle >= S4 by Jensen's inequality.

        With fixed-anchor normalization (linear in θ), this is guaranteed
        mathematically. Small tolerance accounts for Monte Carlo noise.
        """
        results = run_voi_analysis(sample_results, n_simulations=5000, seed=42)
        for r in results:
            assert r.oracle >= r.s4_benchmark - 0.02, (
                f"{r.archetype}: Oracle ({r.oracle:.4f}) < S4 ({r.s4_benchmark:.4f})"
            )

    def test_voi_decomposition_sums(self, sample_results):
        results = run_voi_analysis(sample_results, n_simulations=100, seed=42)
        for r in results:
            expected_total = r.s4_benchmark - r.s1_random
            assert r.total_improvement == pytest.approx(expected_total, abs=1e-9)


class TestFormatResultsTables:
    def test_returns_two_strings(self, sample_results):
        results = run_voi_analysis(sample_results, n_simulations=100, seed=42)
        strategy_table, voi_table = format_results_tables(results)
        assert isinstance(strategy_table, str)
        assert isinstance(voi_table, str)
        assert "S1 (Random)" in strategy_table
        assert "Value of Prioritization" in voi_table
