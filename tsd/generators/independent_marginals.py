"""
Independent Marginals Generator

Baseline method that generates synthetic data by sampling from independent
marginal distributions. This method captures univariate distributions but
ignores all correlations between variables.

Based on study_design_spec.md Section 3.1: "Samples independently from each
variable's empirical distribution. Captures univariate statistics perfectly
but destroys all relationships."
"""

from typing import Optional

import numpy as np
import pandas as pd


class IndependentMarginalsGenerator:
    """
    Generate synthetic data by independently sampling from marginal distributions.

    This is the simplest baseline method - it:
    1. Learns the empirical distribution of each variable independently
    2. Generates synthetic data by sampling each variable independently
    3. Preserves univariate statistics but destroys all correlations

    Attributes:
        marginals: Dictionary storing empirical distributions for each variable
    """

    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the generator.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.marginals = {}
        self.variable_types = {}
        self.rng = np.random.RandomState(random_state)

    def fit(self, df: pd.DataFrame) -> "IndependentMarginalsGenerator":
        """
        Learn the marginal distribution of each variable.

        Args:
            df: Training data

        Returns:
            self (for method chaining)
        """
        print("Fitting Independent Marginals generator...")

        for col in df.columns:
            # Determine variable type
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if it's actually categorical (small number of unique values)
                n_unique = df[col].nunique()
                if n_unique <= 20:  # Treat as categorical
                    self.variable_types[col] = "categorical"
                    # Store value counts
                    value_counts = df[col].value_counts(normalize=True, dropna=False)
                    self.marginals[col] = {
                        "values": value_counts.index.tolist(),
                        "probabilities": value_counts.values.tolist(),
                    }
                else:
                    self.variable_types[col] = "continuous"
                    # Store all observed values for empirical sampling
                    self.marginals[col] = df[col].dropna().values.copy()
            else:
                self.variable_types[col] = "categorical"
                value_counts = df[col].value_counts(normalize=True, dropna=False)
                self.marginals[col] = {
                    "values": value_counts.index.tolist(),
                    "probabilities": value_counts.values.tolist(),
                }

        print(f"✓ Learned marginal distributions for {len(df.columns)} variables")
        print(
            f"  Categorical: {sum(1 for t in self.variable_types.values() if t == 'categorical')}"
        )
        print(f"  Continuous: {sum(1 for t in self.variable_types.values() if t == 'continuous')}")

        return self

    def sample(self, n: int) -> pd.DataFrame:
        """
        Generate synthetic data by independently sampling from marginals.

        Args:
            n: Number of synthetic records to generate

        Returns:
            DataFrame with synthetic data (same schema as training data)
        """
        print(f"Generating {n:,} synthetic records...")

        synthetic_data = {}

        for col in self.marginals.keys():
            if self.variable_types[col] == "categorical":
                # Sample from categorical distribution
                values = self.marginals[col]["values"]
                probs = self.marginals[col]["probabilities"]
                synthetic_data[col] = self.rng.choice(values, size=n, p=probs)
            else:
                # Sample from empirical distribution (with replacement)
                synthetic_data[col] = self.rng.choice(self.marginals[col], size=n, replace=True)

        df_synthetic = pd.DataFrame(synthetic_data)

        # Ensure column order matches training data
        df_synthetic = df_synthetic[list(self.marginals.keys())]

        print(f"✓ Generated {len(df_synthetic):,} synthetic records")

        return df_synthetic

    def fit_sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Fit and generate synthetic data in one call.

        Args:
            df: Training data
            n: Number of synthetic records to generate

        Returns:
            DataFrame with synthetic data
        """
        self.fit(df)
        return self.sample(n)


def generate_independent_marginals(
    df_train: pd.DataFrame, n_samples: int, random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Convenience function to generate synthetic data using Independent Marginals.

    Args:
        df_train: Training data
        n_samples: Number of synthetic records to generate
        random_state: Random seed

    Returns:
        DataFrame with synthetic data
    """
    generator = IndependentMarginalsGenerator(random_state=random_state)
    return generator.fit_sample(df_train, n_samples)
