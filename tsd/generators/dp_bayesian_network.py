"""
DP Bayesian Network Generator

Privacy-preserving method using differential privacy with Bayesian networks.

Based on study_design_spec.md Section 3.3: "PrivBayes-inspired implementation
with epsilon=1.0. Provides formal differential privacy guarantee while
maintaining utility through Bayesian network structure learning."

Uses DataSynthesizer package with PrivBayes algorithm.

Privacy Guarantee:
- Provides (ε, 0)-differential privacy (pure DP, not approximate)
- Default ε=1.0 is a "moderate privacy" setting commonly used in literature
- Lower ε = stronger privacy but lower utility
- Typical ranges: ε=0.1 (strong), ε=1.0 (moderate), ε=10.0 (weak)

Bayesian Network Parameters:
- degree_of_bayesian_network (k): Maximum parents per node in the BN structure
- k=2 (default) allows at most 2-way dependencies to be captured
- Higher k = more complex relationships but requires more noise for same ε

Reference: Zhang, J., Cormode, G., Procopiuc, C. M., Srivastava, D., & Xiao, X. (2017).
PrivBayes: Private Data Release via Bayesian Networks. TODS 2017.
"""

import multiprocessing
import os
import tempfile
from typing import Optional

import pandas as pd
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator


class DPBayesianNetworkGenerator:
    """
    Generate synthetic data using DP Bayesian Networks (PrivBayes algorithm).

    This method:
    1. Learns a Bayesian network structure from data with differential privacy
    2. Estimates conditional distributions with noise added for privacy
    3. Generates synthetic data by sampling from the learned network

    Attributes:
        epsilon: Privacy budget (epsilon parameter for differential privacy)
        degree_of_bayesian_network: Maximum degree of Bayesian network (complexity/utility tradeoff)
        describer: DataSynthesizer's DataDescriber
        generator: DataSynthesizer's DataGenerator
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        degree_of_bayesian_network: int = 2,
        category_threshold: int = 20,
        random_state: Optional[int] = None,
    ):
        """
        Initialize DP Bayesian Network generator.

        Args:
            epsilon: Privacy budget (epsilon for differential privacy). Lower = more private.
                    Study uses epsilon=1.0 as specified in research_review_responses.md
            degree_of_bayesian_network: Maximum number of parents for each node.
                    Higher = more complex relationships but more noise needed
            category_threshold: Maximum number of distinct values for a column to be
                    treated as categorical. Columns with more unique values are treated
                    as numerical. Default=20 ensures all recoded variables are categorical.
            random_state: Random seed
        """
        self.epsilon = epsilon
        self.degree_of_bayesian_network = degree_of_bayesian_network
        self.category_threshold = category_threshold
        self.random_state = random_state

        self.describer = None
        self.generator = None
        self.description_file = None
        self.column_order = None

        # Create temporary directory for DataSynthesizer files
        self.temp_dir = tempfile.mkdtemp(prefix="dp_bn_")

    def fit(self, df: pd.DataFrame) -> "DPBayesianNetworkGenerator":
        """
        Learn DP Bayesian network from data.

        Args:
            df: Training data

        Returns:
            self (for method chaining)
        """
        # Fix for Python 3.13 on macOS: DataSynthesizer uses multiprocessing internally
        try:
            multiprocessing.set_start_method("fork", force=True)
        except RuntimeError:
            pass  # Already set

        print("Training DP Bayesian Network generator...")
        print(f"  Epsilon (privacy budget): {self.epsilon}")
        print(f"  Max degree: {self.degree_of_bayesian_network}")
        print(f"  Training samples: {len(df):,}")

        # Store column order
        self.column_order = df.columns.tolist()

        # DataSynthesizer has issues with NaN values - fill them with a placeholder
        # and convert all columns to appropriate types
        df_clean = df.copy()

        # Convert float columns to int if they're actually categorical (like ESR)
        for col in df_clean.columns:
            if df_clean[col].dtype == "float64":
                # Check if all non-null values are integers
                non_null = df_clean[col].dropna()
                if len(non_null) > 0 and all(non_null == non_null.astype(int)):
                    # Fill NaN with -1 (placeholder for missing)
                    df_clean[col] = df_clean[col].fillna(-1).astype(int)

        # Save data to temporary CSV (DataSynthesizer requires file input)
        data_file = os.path.join(self.temp_dir, "train_data.csv")
        df_clean.to_csv(data_file, index=False)

        # Description file path
        self.description_file = os.path.join(self.temp_dir, "description.json")

        # Initialize DataDescriber with configurable category threshold
        # category_threshold determines when integer columns are treated as categorical
        self.describer = DataDescriber(category_threshold=self.category_threshold)

        # Describe dataset with differential privacy
        # This learns the Bayesian network structure with DP guarantees
        self.describer.describe_dataset_in_correlated_attribute_mode(
            dataset_file=data_file,
            epsilon=self.epsilon,
            k=self.degree_of_bayesian_network,
            seed=self.random_state if self.random_state is not None else 0,
        )

        # Save description
        self.describer.save_dataset_description_to_file(self.description_file)

        # Generator will be initialized during sample()

        print("✓ DP Bayesian Network training complete")
        print(f"  Learned network with {len(df.columns)} nodes")

        return self

    def sample(self, n: int) -> pd.DataFrame:
        """
        Generate synthetic data from learned DP Bayesian network.

        Args:
            n: Number of synthetic records to generate

        Returns:
            DataFrame with synthetic data
        """
        if self.description_file is None:
            raise ValueError("Model not trained. Call fit() first.")

        print(f"Generating {n:,} synthetic records with DP Bayesian Network...")

        # Generate synthetic data
        synthetic_file = os.path.join(self.temp_dir, "synthetic_data.csv")

        # Initialize generator with desired sample size
        self.generator = DataGenerator()
        self.generator.generate_dataset_in_correlated_attribute_mode(
            n=n,
            description_file=self.description_file,
            seed=self.random_state if self.random_state is not None else 0,
        )

        # Save to file
        self.generator.save_synthetic_data(synthetic_file)

        # Load synthetic data
        df_synthetic = pd.read_csv(synthetic_file)

        # Ensure column order matches training data
        df_synthetic = df_synthetic[self.column_order]

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

    def __del__(self):
        """Clean up temporary files."""
        import shutil

        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def generate_dp_bayesian_network(
    df_train: pd.DataFrame,
    n_samples: int,
    epsilon: float = 1.0,
    degree_of_bayesian_network: int = 2,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convenience function to generate synthetic data using DP Bayesian Networks.

    Args:
        df_train: Training data
        n_samples: Number of synthetic records to generate
        epsilon: Privacy budget (epsilon for differential privacy)
        degree_of_bayesian_network: Maximum degree of network
        random_state: Random seed

    Returns:
        DataFrame with synthetic data
    """
    generator = DPBayesianNetworkGenerator(
        epsilon=epsilon,
        degree_of_bayesian_network=degree_of_bayesian_network,
        random_state=random_state,
    )
    return generator.fit_sample(df_train, n_samples)


if __name__ == "__main__":
    # Test with preprocessed ACS data
    from tsd.preprocessing.load_data import preprocess_acs_data

    print("=" * 60)
    print("Testing DP Bayesian Network Generator")
    print("=" * 60)

    # Load and preprocess data (using small sample for testing)
    df_train, df_test = preprocess_acs_data(sample_size=1000)

    # Generate synthetic data
    df_synthetic = generate_dp_bayesian_network(
        df_train=df_train,
        n_samples=len(df_train),
        epsilon=1.0,
        degree_of_bayesian_network=2,
        random_state=42,
    )

    print("\nSynthetic data preview:")
    print(df_synthetic.head())

    print("\nVariable distributions comparison:")
    for col in df_train.columns:
        print(f"\n{col}:")
        print(f"  Real:      {df_train[col].describe()['mean']:.2f} mean")
        print(f"  Synthetic: {df_synthetic[col].describe()['mean']:.2f} mean")

    print("\nSchema validation:")
    print(f"  Columns match: {set(df_train.columns) == set(df_synthetic.columns)}")
    print(f"  Shape: Train {df_train.shape}, Synthetic {df_synthetic.shape}")

    print("\nPrivacy guarantee:")
    print(f"  Epsilon: {1.0} (formal differential privacy)")
