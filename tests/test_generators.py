"""Smoke tests for generators (only IndependentMarginals — others need heavy deps)."""

import pandas as pd

from tsd.generators.independent_marginals import generate_independent_marginals


class TestIndependentMarginals:
    def test_output_shape(self, small_train_test):
        df_train, _ = small_train_test
        n = 50
        result = generate_independent_marginals(df_train, n, random_state=42)
        assert len(result) == n
        assert list(result.columns) == list(df_train.columns)

    def test_categorical_values_preserved(self, small_train_test):
        df_train, _ = small_train_test
        result = generate_independent_marginals(df_train, 100, random_state=42)
        for col in ["sex", "race", "education"]:
            assert set(result[col]).issubset(set(df_train[col]))

    def test_deterministic_with_seed(self, small_train_test):
        df_train, _ = small_train_test
        r1 = generate_independent_marginals(df_train, 50, random_state=42)
        r2 = generate_independent_marginals(df_train, 50, random_state=42)
        pd.testing.assert_frame_equal(r1, r2)
