"""
CTGAN Generator

Deep learning method using Conditional Tabular GAN (CTGAN) for synthetic data generation.

Based on study_design_spec.md Section 3.2: "Conditional Tabular GAN with mode-
specific normalization. Strong baseline for capturing complex relationships."

Reference: Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019).
Modeling Tabular Data using Conditional GAN. NeurIPS 2019.
"""

import pandas as pd
import numpy as np
import torch
import random
from typing import Optional, List
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import warnings


class CTGANGenerator:
    """
    Generate synthetic data using CTGAN.

    CTGAN is a GAN-based method specifically designed for tabular data with:
    - Mode-specific normalization for continuous variables
    - Conditional generation for handling imbalanced categorical variables
    - Training-by-sampling to focus on rare categories

    Attributes:
        model: SDV's CTGANSynthesizer instance
        metadata: Table metadata (variable types, constraints)
    """

    def __init__(
        self,
        epochs: int = 300,
        batch_size: int = 500,
        generator_dim: tuple = (256, 256),
        discriminator_dim: tuple = (256, 256),
        verbose: bool = False,
        random_state: Optional[int] = None
    ):
        """
        Initialize CTGAN generator.

        Args:
            epochs: Number of training epochs (default 300)
            batch_size: Training batch size (default 500)
            generator_dim: Generator network dimensions (default (256, 256))
            discriminator_dim: Discriminator network dimensions (default (256, 256))
            verbose: Whether to print training progress
            random_state: Random seed
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.verbose = verbose
        self.random_state = random_state

        self.model = None
        self.metadata = None
        self.column_order = None

    def _create_metadata(self, df: pd.DataFrame) -> SingleTableMetadata:
        """
        Create metadata for the table.

        Automatically detects variable types and creates appropriate metadata.

        Args:
            df: Training data

        Returns:
            SingleTableMetadata object
        """
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        # Override detection for known categorical variables
        # (SDV may auto-detect some integers as continuous)
        categorical_cols = ['SEX', 'RAC1P_RECODED', 'SCHL_RECODED', 'MAR', 'POBP_RECODED']

        for col in categorical_cols:
            if col in df.columns:
                metadata.update_column(col, sdtype='categorical')

        # ESR has missing values (NaN for some people), mark as categorical
        if 'ESR' in df.columns:
            metadata.update_column('ESR', sdtype='categorical')

        return metadata

    def _set_random_seeds(self):
        """
        Set all random seeds for reproducibility.
        
        CTGAN uses PyTorch internally, so we must set seeds for:
        - Python's random module
        - NumPy's random generator
        - PyTorch's random generators (CPU and GPU)
        """
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_state)
                torch.cuda.manual_seed_all(self.random_state)
                # For deterministic behavior (may slow down training)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def fit(self, df: pd.DataFrame) -> 'CTGANGenerator':
        """
        Train CTGAN on the data.

        Args:
            df: Training data

        Returns:
            self (for method chaining)
        """
        # Set random seeds BEFORE any model operations for reproducibility
        self._set_random_seeds()
        
        print(f"Training CTGAN generator...")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Random state: {self.random_state}")
        print(f"  Training samples: {len(df):,}")

        # Store column order
        self.column_order = df.columns.tolist()

        # Create metadata
        self.metadata = self._create_metadata(df)

        # Initialize CTGAN model
        self.model = CTGANSynthesizer(
            metadata=self.metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            verbose=self.verbose
        )

        # Train the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress SDV warnings
            self.model.fit(df)

        print(f"✓ CTGAN training complete")

        return self

    def sample(self, n: int) -> pd.DataFrame:
        """
        Generate synthetic data.

        Args:
            n: Number of synthetic records to generate

        Returns:
            DataFrame with synthetic data
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Set seeds again for reproducible sampling (use offset to differ from training)
        if self.random_state is not None:
            sample_seed = self.random_state + 1000
            random.seed(sample_seed)
            np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(sample_seed)

        print(f"Generating {n:,} synthetic records with CTGAN...")

        # Generate synthetic data
        df_synthetic = self.model.sample(num_rows=n)

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


def generate_ctgan(
    df_train: pd.DataFrame,
    n_samples: int,
    epochs: int = 300,
    batch_size: int = 500,
    verbose: bool = False,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Convenience function to generate synthetic data using CTGAN.

    Args:
        df_train: Training data
        n_samples: Number of synthetic records to generate
        epochs: Number of training epochs
        batch_size: Training batch size
        verbose: Whether to print training progress
        random_state: Random seed

    Returns:
        DataFrame with synthetic data
    """
    generator = CTGANGenerator(
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        random_state=random_state
    )
    return generator.fit_sample(df_train, n_samples)


if __name__ == "__main__":
    # Test with preprocessed ACS data
    import sys
    sys.path.append('.')
    from src.preprocessing.load_data import preprocess_acs_data

    print("="*60)
    print("Testing CTGAN Generator")
    print("="*60)

    # Load and preprocess data (using small sample for testing)
    df_train, df_test = preprocess_acs_data(sample_size=1000)

    # Generate synthetic data
    df_synthetic = generate_ctgan(
        df_train=df_train,
        n_samples=len(df_train),
        epochs=10,  # Reduced for testing
        batch_size=100,
        verbose=True,
        random_state=42
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
