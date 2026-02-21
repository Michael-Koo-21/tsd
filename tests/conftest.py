"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_dataset():
    """200-row DataFrame with generic columns (no ACS dependency)."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=n),
            "income": rng.exponential(40000, size=n).astype(int),
            "sex": rng.choice([1, 2], size=n),
            "race": rng.choice([1, 2, 3, 4, 5], size=n),
            "education": rng.choice([1, 2, 3, 4, 5], size=n),
            "target": rng.choice([0, 1], size=n),
        }
    )


@pytest.fixture
def small_train_test(small_dataset):
    """70/30 split of small_dataset."""
    n_train = int(len(small_dataset) * 0.7)
    return small_dataset.iloc[:n_train].copy(), small_dataset.iloc[n_train:].copy()


@pytest.fixture
def small_synthetic(small_train_test):
    """IndependentMarginals output from small_train_test."""
    from tsd.generators.independent_marginals import generate_independent_marginals

    df_train, _ = small_train_test
    return generate_independent_marginals(df_train, len(df_train), random_state=42)


@pytest.fixture
def sample_results():
    """DataFrame mimicking all_results.csv with 5 methods x 3 replicates."""
    rng = np.random.default_rng(99)
    rows = []
    methods = ["independent_marginals", "ctgan", "dpbn", "synthpop", "great"]
    for method in methods:
        for rep in range(1, 4):
            rows.append(
                {
                    "method": method,
                    "replicate": rep,
                    "seed": rep * 100,
                    "fidelity_auc": rng.uniform(0.5, 0.9),
                    "privacy_dcr": rng.uniform(0.01, 0.15),
                    "utility_tstr": rng.uniform(0.3, 1.0),
                    "fairness_gap": rng.uniform(0.02, 0.08),
                }
            )
    return pd.DataFrame(rows)
