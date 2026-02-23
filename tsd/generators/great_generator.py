"""
GReaT (Generation of Realistic Tabular data) Generator.

Wraps the be_great library with proper random seeding for reproducibility.
Supports CUDA (NVIDIA), MPS (Apple Silicon M1/M2/M3), and CPU backends.

References:
    - Paper: https://arxiv.org/abs/2210.06280
    - Library: https://github.com/kathrinse/be_great
"""

import random
from typing import Optional

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None


def _require_torch():
    """Raise a helpful error if torch is not installed."""
    if torch is None:
        raise ImportError("PyTorch is required for GReaT. Install with: pip install -e '.[great]'")


def get_best_device() -> str:
    """
    Detect the best available device for PyTorch.

    Returns:
        str: 'cuda' for NVIDIA GPU, 'mps' for Apple Silicon, 'cpu' otherwise
    """
    _require_torch()
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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
        Number of fine-tuning epochs (default: 15)
    batch_size : int
        Training batch size (default: 64)
    random_state : int, optional
        Random seed for reproducibility

    Example
    -------
    >>> gen = GReaTGenerator(llm='distilgpt2', epochs=15, random_state=42)
    >>> gen.fit(df_train)
    >>> df_synth = gen.generate(n_samples=1000)
    """

    def __init__(
        self,
        llm: str = "distilgpt2",
        epochs: int = 15,
        batch_size: int = 64,
        random_state: Optional[int] = None,
        experiment_dir: str = "great_checkpoints",
    ):
        self.llm = llm
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.experiment_dir = experiment_dir
        self.model = None
        self._columns = None

    def fit(self, df: pd.DataFrame) -> "GReaTGenerator":
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
        except ImportError as e:
            raise ImportError("be_great not installed. Install with: pip install be-great") from e

        # Set seeds before training
        if self.random_state is not None:
            set_all_seeds(self.random_state)

        self._columns = df.columns.tolist()

        # Create model with seed passed to TrainingArguments
        train_kwargs = {}
        if self.random_state is not None:
            train_kwargs["seed"] = self.random_state
            train_kwargs["data_seed"] = self.random_state

        self.model = GReaT(
            llm=self.llm,
            experiment_dir=self.experiment_dir,
            epochs=self.epochs,
            batch_size=self.batch_size,
            **train_kwargs,
        )

        # Fit model
        self.model.fit(df)

        return self

    def generate(
        self,
        n_samples: int,
        temperature: float = 0.7,
        k: int = 100,
        device: str = "auto",
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
            Device for generation ('auto', 'cuda', 'mps', or 'cpu')
            'auto' will select the best available device

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

        # Auto-detect best device
        if device == "auto":
            device = get_best_device()
            print(f"  Using device: {device}")

        # Check device availability and fallback
        _require_torch()
        if device == "cuda" and not torch.cuda.is_available():
            print("  Warning: CUDA not available, falling back to CPU")
            device = "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            print("  Warning: MPS not available, falling back to CPU")
            device = "cpu"

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
    llm: str = "distilgpt2",
    epochs: int = 15,
    batch_size: int = 64,
    random_state: Optional[int] = None,
    device: str = "auto",
    k: int = 100,
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
        Training epochs (default: 15)
    batch_size : int
        Training batch size (default: 64)
    random_state : int, optional
        Random seed for reproducibility
    device : str
        Device for generation ('auto', 'cuda', 'mps', or 'cpu')
        'auto' will auto-detect the best available device
    k : int
        Generation batch size (default: 100, higher = faster generation)

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
    return gen.generate(n_samples=n_samples, device=device, k=k)
