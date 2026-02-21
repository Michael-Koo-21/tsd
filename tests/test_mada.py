"""Tests for the MADA framework: normalize, weight, rank, sensitivity."""

import pandas as pd
import pytest

from tsd.analysis.mada_framework import (
    add_efficiency_data,
    calculate_weighted_scores,
    get_method_scores,
    normalize_scores,
    rank_methods,
    sensitivity_analysis,
)
from tsd.constants import METRIC_DIRECTION, PROFILES


class TestNormalize:
    def test_output_range_zero_one(self, sample_results):
        scores = get_method_scores(sample_results)
        normalized = normalize_scores(scores)
        for col in normalized.columns:
            if col in METRIC_DIRECTION:
                assert normalized[col].min() >= -1e-9
                assert normalized[col].max() <= 1.0 + 1e-9

    def test_direction_inversion(self, sample_results):
        scores = get_method_scores(sample_results)
        normalized = normalize_scores(scores)
        # For inverted metrics, the original worst should map to 0
        for metric, higher_is_better in METRIC_DIRECTION.items():
            if metric not in scores.columns:
                continue
            if not higher_is_better:
                worst_method = scores[metric].idxmax()  # highest raw = worst
                assert normalized.loc[worst_method, metric] == pytest.approx(0.0, abs=1e-9)


class TestWeightedScores:
    def test_weights_must_sum_to_one(self, sample_results):
        scores = get_method_scores(sample_results)
        normalized = normalize_scores(scores)
        bad_weights = {
            "fidelity": 0.5,
            "privacy": 0.5,
            "utility": 0.5,
            "fairness": 0.1,
            "efficiency": 0.1,
        }
        with pytest.raises(ValueError, match="sum to 1"):
            calculate_weighted_scores(normalized, bad_weights)

    def test_scores_in_valid_range(self, sample_results):
        scores = get_method_scores(sample_results)
        normalized = normalize_scores(scores)
        weights = PROFILES["balanced"].weights
        weighted = calculate_weighted_scores(normalized, weights)
        assert (weighted >= 0).all()
        assert (weighted <= 1.0 + 1e-9).all()


class TestRanking:
    def test_rankings_consistent(self, sample_results):
        scores = get_method_scores(sample_results)
        normalized = normalize_scores(scores)
        weights = PROFILES["balanced"].weights
        weighted = calculate_weighted_scores(normalized, weights)
        ranking = rank_methods(weighted)
        # Should have one row per method, ranks 1-5
        assert set(ranking["rank"].values) == {1, 2, 3, 4, 5}


class TestSensitivity:
    def test_returns_dataframe(self, sample_results):
        scores = get_method_scores(sample_results)
        normalized = normalize_scores(scores)
        weights = PROFILES["balanced"].weights
        result = sensitivity_analysis(normalized, weights, "privacy", steps=5)
        assert isinstance(result, pd.DataFrame)
        assert "weight" in result.columns
        assert "method" in result.columns
        assert "score" in result.columns


class TestEfficiency:
    def test_adds_efficiency_column(self, sample_results):
        df = add_efficiency_data(sample_results)
        assert "efficiency_time" in df.columns
