"""Tests for the data loading pipeline."""

import pandas as pd

from tsd.preprocessing.load_data import (
    recode_education,
    recode_race_ethnicity,
    train_test_split,
)


class TestRecodeRaceEthnicity:
    def test_hispanic_override(self):
        rac1p = pd.Series([1, 2, 6, 1])  # White, Black, Asian, White
        hisp = pd.Series([1, 1, 1, 2])  # non-Hisp, non-Hisp, non-Hisp, Hispanic
        result = recode_race_ethnicity(rac1p, hisp)
        assert result.iloc[0] == 1  # White non-Hispanic
        assert result.iloc[1] == 2  # Black non-Hispanic
        assert result.iloc[2] == 3  # Asian non-Hispanic
        assert result.iloc[3] == 4  # Hispanic (overrides White)

    def test_other_race(self):
        rac1p = pd.Series([3, 7, 9])  # AIAN, NHPI, Two+
        hisp = pd.Series([1, 1, 1])
        result = recode_race_ethnicity(rac1p, hisp)
        assert (result == 5).all()  # All → Other


class TestRecodeEducation:
    def test_five_levels(self):
        schl = pd.Series([5, 16, 19, 21, 23])
        result = recode_education(schl)
        assert list(result) == [1, 2, 3, 4, 5]


class TestTrainTestSplit:
    def test_split_ratio(self):
        df = pd.DataFrame({"x": range(100)})
        train, test = train_test_split(df, train_size=0.7, random_state=42)
        assert len(train) == 70
        assert len(test) == 30

    def test_no_overlap(self):
        df = pd.DataFrame({"x": range(100)})
        train, test = train_test_split(df, train_size=0.7, random_state=42)
        assert set(train["x"]).isdisjoint(set(test["x"]))
