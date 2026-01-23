"""
GReaT (Generation of Realistic Tabular data) Generator.

Wraps the be_great library with proper random seeding for reproducibility.
Requires GPU for reasonable performance - consider using Colab for training.

References:
    - Paper: https://arxiv.org/abs/2210.06280
    - Library: https://github.com/kathrinse/be_great
"""

import pandas as pd
import numpy as np
import torch
import random
import warnings
from typing import Optional

warnings.filterwarnings('ignore')


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class GReaTGenerator:
    """
    GReaT synthetic data generator with proper seeding.

    Uses a pre-trained language model (GPT-2 variants) fine-tuned on tabular data
    to generate realistic synthetic records.

    Parameters
    ----------
    llm : str
        HuggingFace model checkpoint. Options:
        - 'distilgpt2' (fastest, ~82M params)
        - 'gpt2' (medium, ~124M params)
        - 'gpt2-medium' (~355M params)
        - 'gpt2-large' (~774M params)
    epochs : int
        Number of fine-tuning epochs (default: 100)
    batch_size : int
        Training batch size (default: 32)
    random_state : int, optional
        Random seed for reproducibility

    Example
    -------
    >>> gen = GReaTGenerator(llm='distilgpt2', epochs=50, random_state=42)
    >>> gen.fit(df_train)
    >>> df_synth = gen.generate(n_samples=1000)
    """

    def __init__(
        self,
        llm: str = 'distilgpt2',
        epochs: int = 100,
        batch_size: int = 32,
        random_state: Optional[int] = None,
        experiment_dir: str = 'great_checkpoints',
    ):
        self.llm = llm
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.experiment_dir = experiment_dir
        self.model = None
        self._columns = None

    def fit(self, df: pd.DataFrame) -> 'GReaTGenerator':
        """
        Fit the GReaT model on training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data

        Returns
        -------
        self
        """
        try:
            from be_great import GReaT
        except ImportError:
            raise ImportError(
                "be_great not installed. Install with: pip install be-great"
            )

        # Set seeds before training
        if self.random_state is not None:
            set_all_seeds(self.random_state)

        self._columns = df.columns.tolist()

        # Create model with seed passed to TrainingArguments
        train_kwargs = {}
        if self.random_state is not None:
            train_kwargs['seed'] = self.random_state
            train_kwargs['data_seed'] = self.random_state

        self.model = GReaT(
            llm=self.llm,
            experiment_dir=self.experiment_dir,
            epochs=self.epochs,
            batch_size=self.batch_size,
            **train_kwargs
        )

        # Fit model
        self.model.fit(df)

        return self

    def generate(
        self,
        n_samples: int,
        temperature: float = 0.7,
        k: int = 100,
        device: str = 'cuda',
    ) -> pd.DataFrame:
        """
        Generate synthetic data samples.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        temperature : float
            Sampling temperature (higher = more diverse, lower = more conservative)
        k : int
            Sampling batch size
        device : str
            Device for generation ('cuda' or 'cpu')

        Returns
        -------
        pd.DataFrame
            Generated synthetic data
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Set seeds before generation for reproducibility
        if self.random_state is not None:
            set_all_seeds(self.random_state + 1000)  # Offset to differ from training

        # Check device availability
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU")
            device = 'cpu'

        df_synth = self.model.sample(
            n_samples=n_samples,
            temperature=temperature,
            k=k,
            device=device,
            drop_nan=True,
        )

        # Ensure column order matches training data
        if self._columns is not None:
            # Only keep columns that exist
            available_cols = [c for c in self._columns if c in df_synth.columns]
            df_synth = df_synth[available_cols]

        return df_synth


def generate_great(
    df: pd.DataFrame,
    n_samples: int,
    llm: str = 'distilgpt2',
    epochs: int = 100,
    batch_size: int = 32,
    random_state: Optional[int] = None,
    device: str = 'cuda',
) -> pd.DataFrame:
    """
    Convenience function to generate synthetic data using GReaT.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    n_samples : int
        Number of synthetic samples to generate
    llm : str
        Language model to use (default: 'distilgpt2')
    epochs : int
        Training epochs (default: 100)
    batch_size : int
        Training batch size (default: 32)
    random_state : int, optional
        Random seed for reproducibility
    device : str
        Device for generation (default: 'cuda')

    Returns
    -------
    pd.DataFrame
        Generated synthetic data

    Example
    -------
    >>> df_synth = generate_great(df_train, n_samples=1000, random_state=42)
    """
    gen = GReaTGenerator(
        llm=llm,
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
    )
    gen.fit(df)
    return gen.generate(n_samples=n_samples, device=device)


if __name__ == "__main__":
    # Quick test
    print("GReaT Generator Module")
    print("=" * 40)

    # Check if be_great is available
    try:
        from be_great import GReaT
        print("✓ be_great library available")
    except ImportError:
        print("✗ be_great not installed")
        print("  Install with: pip install be-great")

    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("✗ CUDA not available (CPU only - will be slow)")
