"""
Dataset configuration system for the TSD framework.

Allows users to define their own dataset schema so the pipeline
is not hardcoded to ACS PUMS columns.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    """Configuration describing a dataset for the TSD pipeline."""

    name: str
    data_path: str
    target_variable: str  # e.g., "HIGH_INCOME"
    subgroup_column: str  # e.g., "RACETH" for fairness
    categorical_columns: list[str]
    continuous_columns: list[str]
    target_binarize: dict | None = None  # {"source_column": "PINCP", "threshold": 50000}
    weight_column: str | None = None
    train_fraction: float = 0.7
    sample_size: int | None = None
    random_state: int = 42
    preprocessing: str | None = None  # Named preprocessing pipeline, e.g., "acs_pums"

    def validate(self) -> None:
        """Raise ValueError if the config is internally inconsistent."""
        if not self.categorical_columns and not self.continuous_columns:
            raise ValueError("Must specify at least one categorical or continuous column.")
        if self.target_binarize is not None:
            required = {"source_column", "threshold"}
            missing = required - set(self.target_binarize.keys())
            if missing:
                raise ValueError(f"target_binarize missing keys: {missing}")
        if not 0 < self.train_fraction < 1:
            raise ValueError(f"train_fraction must be in (0, 1), got {self.train_fraction}")

    def validate_against_data(self, df) -> None:
        """Check that configured columns exist in the DataFrame.

        Call this after loading the CSV on the generic (non-ACS) path.
        Raises ValueError with a helpful message listing available columns.
        """
        available = set(df.columns)

        def _check(col_name: str, field: str):
            if col_name not in available:
                raise ValueError(
                    f"Column '{col_name}' specified in {field} not found in CSV.\n"
                    f"Available columns: {sorted(available)}"
                )

        _check(self.target_variable, "target_variable")
        _check(self.subgroup_column, "subgroup_column")

        for col in self.categorical_columns:
            _check(col, "categorical_columns")
        for col in self.continuous_columns:
            _check(col, "continuous_columns")

        if self.target_binarize:
            _check(self.target_binarize["source_column"], "target_binarize.source_column")

        if self.weight_column:
            _check(self.weight_column, "weight_column")


def load_config(path: str | Path) -> DatasetConfig:
    """Load a DatasetConfig from a YAML file."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    config = DatasetConfig(**raw)
    config.validate()
    return config


def default_config_path() -> Path:
    """Return the path to the bundled ACS PUMS config."""
    return Path(__file__).parent / "configs" / "acs_pums.yaml"
