"""Tests for config loading and validation."""

import pytest

from tsd.config import DatasetConfig, default_config_path, load_config


class TestDefaultConfig:
    def test_loads_successfully(self):
        config = load_config(default_config_path())
        assert config.name == "ACS PUMS California"

    def test_has_required_fields(self):
        config = load_config(default_config_path())
        assert config.target_variable == "HIGH_INCOME"
        assert config.subgroup_column == "RACETH"
        assert len(config.categorical_columns) > 0
        assert len(config.continuous_columns) > 0

    def test_categorical_columns_are_strings(self):
        config = load_config(default_config_path())
        for col in config.categorical_columns:
            assert isinstance(col, str)


class TestValidation:
    def test_missing_columns_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            DatasetConfig(
                name="empty",
                data_path="x.csv",
                target_variable="y",
                subgroup_column="z",
                categorical_columns=[],
                continuous_columns=[],
            ).validate()

    def test_bad_train_fraction_raises(self):
        with pytest.raises(ValueError, match="train_fraction"):
            DatasetConfig(
                name="bad",
                data_path="x.csv",
                target_variable="y",
                subgroup_column="z",
                categorical_columns=["a"],
                continuous_columns=["b"],
                train_fraction=1.5,
            ).validate()

    def test_bad_binarize_raises(self):
        with pytest.raises(ValueError, match="target_binarize"):
            DatasetConfig(
                name="bad",
                data_path="x.csv",
                target_variable="y",
                subgroup_column="z",
                categorical_columns=["a"],
                continuous_columns=["b"],
                target_binarize={"source_column": "x"},  # missing threshold
            ).validate()
