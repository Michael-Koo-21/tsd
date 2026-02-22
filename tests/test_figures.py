"""Tests for tsd.analysis.figures (value functions and data loading)."""

import pytest

from tsd.analysis.figures import (
    calculate_weighted_score,
    load_and_compute_values,
    v_efficiency,
    v_fairness,
    v_fidelity,
    v_privacy,
    v_utility,
)


class TestValueFunctions:
    def test_v_fidelity_ideal(self):
        # AUC = 0.5 is ideal
        assert v_fidelity(0.5) == pytest.approx(1.0)

    def test_v_fidelity_worst(self):
        # AUC = 0.88 (x_worst) should give 0
        assert v_fidelity(0.88) == pytest.approx(0.0, abs=0.01)

    def test_v_fidelity_clamped(self):
        assert v_fidelity(0.95) == 0.0  # beyond worst, clamped
        assert v_fidelity(0.4) == 1.0  # beyond ideal, clamped

    def test_v_privacy_best(self):
        assert v_privacy(0.14) == pytest.approx(1.0)

    def test_v_privacy_worst(self):
        assert v_privacy(0.007) == pytest.approx(0.0)

    def test_v_privacy_clamped(self):
        assert v_privacy(0.0) == 0.0
        assert v_privacy(0.2) == 1.0

    def test_v_utility_ideal(self):
        assert v_utility(1.0) == pytest.approx(1.0)

    def test_v_utility_worst(self):
        assert v_utility(0.037) == pytest.approx(0.0)

    def test_v_utility_clamped(self):
        assert v_utility(0.0) == 0.0
        assert v_utility(1.1) == 1.0

    def test_v_fairness_ideal(self):
        # gap = 0 is ideal
        assert v_fairness(0.0) == pytest.approx(1.0)

    def test_v_fairness_worst(self):
        assert v_fairness(0.08) == pytest.approx(0.0)

    def test_v_fairness_clamped(self):
        assert v_fairness(0.1) == 0.0
        assert v_fairness(-0.01) == 1.0

    def test_v_efficiency_fast(self):
        assert v_efficiency(0.3) == 1.0  # faster than x_best

    def test_v_efficiency_slow(self):
        result = v_efficiency(90.0)
        assert 0 < result < 1

    def test_v_efficiency_very_slow(self):
        assert v_efficiency(480) == pytest.approx(0.0, abs=0.01)

    def test_v_efficiency_clamped(self):
        assert v_efficiency(1000) == 0.0


class TestLoadAndComputeValues:
    def test_returns_dict(self, tmp_path, sample_results):
        f = tmp_path / "results.csv"
        sample_results.to_csv(f, index=False)
        result = load_and_compute_values(f)
        assert isinstance(result, dict)
        assert len(result) == 5

    def test_value_keys(self, tmp_path, sample_results):
        f = tmp_path / "results.csv"
        sample_results.to_csv(f, index=False)
        result = load_and_compute_values(f)
        for method, scores in result.items():
            assert set(scores.keys()) == {"fidelity", "privacy", "utility", "fairness", "efficiency"}

    def test_values_in_range(self, tmp_path, sample_results):
        f = tmp_path / "results.csv"
        sample_results.to_csv(f, index=False)
        result = load_and_compute_values(f)
        for method, scores in result.items():
            for attr, val in scores.items():
                assert 0.0 <= val <= 1.0


class TestCalculateWeightedScore:
    def test_basic_calculation(self):
        value_scores = {
            "test_method": {
                "fidelity": 0.5,
                "privacy": 0.5,
                "utility": 0.5,
                "fairness": 0.5,
                "efficiency": 0.5,
            }
        }
        weights = {"fidelity": 0.2, "privacy": 0.2, "utility": 0.2,
                   "fairness": 0.2, "efficiency": 0.2}
        result = calculate_weighted_score(value_scores, "test_method", weights)
        assert result == pytest.approx(0.5)

    def test_weighted_sum(self):
        value_scores = {
            "m": {"fidelity": 1.0, "privacy": 0.0, "utility": 0.0,
                  "fairness": 0.0, "efficiency": 0.0}
        }
        weights = {"fidelity": 0.5, "privacy": 0.1, "utility": 0.2,
                   "fairness": 0.1, "efficiency": 0.1}
        result = calculate_weighted_score(value_scores, "m", weights)
        assert result == pytest.approx(0.5)
