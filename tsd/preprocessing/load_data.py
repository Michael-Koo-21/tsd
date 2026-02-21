"""
Load and preprocess ACS PUMS California data.

Preprocessing steps:
1. Filter to adults (AGEP >= 18)
2. Filter to valid income values (PINCP not null)
3. Select and recode variables:
   - RACETH: Race/ethnicity with Hispanic override (5 categories)
   - SCHL_RECODED: Education collapsed to 5 levels
   - POBP_RECODED: Birthplace to 7 geographic regions
4. Sample to target size
5. Train/test split: 70%/30%

Final dataset: 50,000 total (35,000 train, 15,000 test)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tsd.config import DatasetConfig


def load_raw_acs_pums(data_path: str = "data/raw/psam_p06.csv") -> pd.DataFrame:
    """
    Load raw ACS PUMS California data.

    Returns:
        DataFrame with all raw variables
    """
    print(f"Loading raw ACS PUMS data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df):,} records with {len(df.columns)} columns")
    return df


def filter_adults(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to adults only (AGEP >= 18).

    Args:
        df: Raw ACS PUMS data

    Returns:
        DataFrame filtered to adults
    """
    print("Filtering to adults (AGEP >= 18)...")
    df_adults = df[df["AGEP"] >= 18].copy()
    print(f"✓ {len(df_adults):,} adult records ({len(df_adults)/len(df)*100:.1f}%)")
    return df_adults


def filter_valid_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to records with valid (non-missing) income values.

    Args:
        df: DataFrame with PINCP column

    Returns:
        DataFrame filtered to valid income records
    """
    print("Filtering to valid income values (PINCP not null)...")
    n_before = len(df)
    df_valid = df[df["PINCP"].notna()].copy()
    n_dropped = n_before - len(df_valid)
    print(
        f"✓ {len(df_valid):,} records with valid income ({n_dropped:,} dropped, {n_dropped/n_before*100:.1f}%)"
    )
    return df_valid


def recode_race_ethnicity(rac1p_series: pd.Series, hisp_series: pd.Series) -> pd.Series:
    """
    Recode RAC1P (race) with HISP (Hispanic origin) override.

    This follows standard demographic practice where Hispanic ethnicity
    takes precedence over race, as Hispanic individuals can be of any race.
    This is important for fairness analysis since Hispanic is a protected class.

    ACS PUMS RAC1P codes:
    1 = White alone
    2 = Black or African American alone
    3 = American Indian alone
    4 = Alaska Native alone
    5 = American Indian and Alaska Native tribes specified
    6 = Asian alone
    7 = Native Hawaiian and Other Pacific Islander alone
    8 = Some Other Race alone
    9 = Two or More Races

    ACS PUMS HISP codes:
    1 = Not Hispanic
    2-24 = Hispanic (various origins)

    Recoding with Hispanic override:
    1 = White (non-Hispanic)
    2 = Black (non-Hispanic)
    3 = Asian (non-Hispanic)
    4 = Hispanic (any race) - OVERRIDE
    5 = Other (non-Hispanic: AIAN, NHPI, Some Other Race, Two or More Races)
    """

    def recode_value(rac1p, hisp):
        # Hispanic override: if Hispanic (HISP > 1), classify as Hispanic regardless of race
        if hisp > 1:
            return 4  # Hispanic
        # Non-Hispanic: classify by race
        elif rac1p == 1:
            return 1  # White (non-Hispanic)
        elif rac1p == 2:
            return 2  # Black (non-Hispanic)
        elif rac1p == 6:
            return 3  # Asian (non-Hispanic)
        else:
            return 5  # Other (non-Hispanic): AIAN, NHPI, Some Other Race, Two or More Races

    return pd.Series(
        [recode_value(r, h) for r, h in zip(rac1p_series, hisp_series)], index=rac1p_series.index
    )


def recode_education(schl_series: pd.Series) -> pd.Series:
    """
    Recode SCHL (educational attainment) into broader categories.

    ACS PUMS SCHL codes (1-24):
    1-15: Less than high school
    16-17: High school graduate or GED
    18-20: Some college, no degree
    21: Bachelor's degree
    22-24: Graduate/Professional degree

    Recoding:
    1 = Less than high school
    2 = High school graduate or GED
    3 = Some college, no degree
    4 = Bachelor's degree
    5 = Graduate or professional degree
    """

    def recode_value(x):
        if pd.isna(x):
            return np.nan
        elif x <= 15:
            return 1  # Less than HS
        elif x <= 17:
            return 2  # HS/GED
        elif x <= 20:
            return 3  # Some college
        elif x == 21:
            return 4  # Bachelor's
        elif x <= 24:
            return 5  # Graduate/Professional
        else:
            return 1  # Default to Less than HS

    return schl_series.apply(recode_value)


def recode_birthplace(pobp_series: pd.Series) -> pd.Series:
    """
    Recode POBP (place of birth) into broader geographic regions.

    ACS PUMS POBP codes:
    1-56: US states/territories (1-56)
    6: California specifically
    60-99: US territories
    100-199: Americas (excluding US)
    200-299: Europe
    300-399: Asia
    400-499: Africa
    500-599: Oceania

    Recoding:
    1 = California
    2 = Other US state
    3 = Mexico (303)
    4 = Other Americas
    5 = Europe
    6 = Asia
    7 = Other (Africa, Oceania, unknown)
    """

    def recode_value(x):
        if x == 6:
            return 1  # California
        elif 1 <= x <= 56:
            return 2  # Other US state
        elif x == 303:
            return 3  # Mexico
        elif 200 <= x <= 299:
            return 5  # Europe
        elif 300 <= x <= 399:
            return 6  # Asia
        elif 100 <= x <= 199:
            return 4  # Other Americas
        else:
            return 7  # Other (Africa, Oceania, territories)

    return pobp_series.apply(recode_value)


def select_and_recode_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the 8 predictors + 1 target variable and apply recoding.

    Variables:
    - AGEP: Age (continuous)
    - PINCP: Total person's income (continuous)
    - SEX: Sex (2 categories)
    - RACETH: Race/Ethnicity with Hispanic override (5 categories)
    - SCHL_RECODED: Educational attainment recoded (5 levels)
    - MAR: Marital status (5 categories)
    - ESR: Employment status (6 categories)
    - POBP_RECODED: Place of birth recoded (7 categories)
    - PWGTP: Person weight (used for sampling, not modeling)
    """
    print("Selecting and recoding variables...")

    # Select base variables (including HISP for race/ethnicity recoding)
    df_selected = df[
        ["AGEP", "PINCP", "SEX", "MAR", "ESR", "PWGTP", "RAC1P", "HISP", "SCHL", "POBP"]
    ].copy()

    # Apply recoding with Hispanic override for race/ethnicity
    df_selected["RACETH"] = recode_race_ethnicity(df_selected["RAC1P"], df_selected["HISP"])
    df_selected["SCHL_RECODED"] = recode_education(df_selected["SCHL"])
    df_selected["POBP_RECODED"] = recode_birthplace(df_selected["POBP"])

    # Drop original versions
    df_selected = df_selected.drop(columns=["RAC1P", "HISP", "SCHL", "POBP"])

    # Reorder columns: predictors + weight
    final_cols = [
        "AGEP",
        "PINCP",
        "SEX",
        "RACETH",
        "SCHL_RECODED",
        "MAR",
        "ESR",
        "POBP_RECODED",
        "PWGTP",
    ]
    df_selected = df_selected[final_cols]

    print("✓ Selected and recoded 9 variables (8 predictors + 1 weight)")
    print(
        f"  RAC1P+HISP → RACETH: {df_selected['RACETH'].nunique()} categories (with Hispanic override)"
    )
    print(f"  SCHL → SCHL_RECODED: {df_selected['SCHL_RECODED'].nunique()} categories")
    print(f"  POBP → POBP_RECODED: {df_selected['POBP_RECODED'].nunique()} categories")

    return df_selected


def sample_data(df: pd.DataFrame, n: int = 50000, random_state: int = 42) -> pd.DataFrame:
    """
    Sample N records from the full dataset.

    Args:
        df: Full adult data
        n: Sample size (default 50,000)
        random_state: Random seed

    Returns:
        Sampled DataFrame
    """
    print(f"Sampling {n:,} records...")
    if len(df) <= n:
        print(f"⚠ Dataset has only {len(df):,} records, using all")
        return df.copy()

    df_sample = df.sample(n=n, random_state=random_state)
    print(f"✓ Sampled {len(df_sample):,} records")
    return df_sample


def train_test_split(
    df: pd.DataFrame, train_size: float = 0.7, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train (70%) and test (30%) sets.

    Args:
        df: Full dataset
        train_size: Fraction for training (default 0.7)
        random_state: Random seed

    Returns:
        (df_train, df_test)
    """
    print(f"Splitting data: {train_size*100:.0f}% train, {(1-train_size)*100:.0f}% test...")

    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * train_size)

    df_train = df_shuffled.iloc[:split_idx].copy()
    df_test = df_shuffled.iloc[split_idx:].copy()

    print(f"✓ Train: {len(df_train):,} records")
    print(f"✓ Test: {len(df_test):,} records")

    return df_train, df_test


def preprocess_acs_data(
    data_path: str = "data/raw/psam_p06.csv",
    sample_size: int = 50000,
    train_size: float = 0.7,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline.

    Args:
        data_path: Path to raw ACS PUMS CSV
        sample_size: Number of records to sample
        train_size: Fraction for training
        random_state: Random seed

    Returns:
        (df_train, df_test)
    """
    print("=" * 60)
    print("ACS PUMS Data Preprocessing")
    print("=" * 60)

    # Load raw data
    df_raw = load_raw_acs_pums(data_path)

    # Filter to adults
    df_adults = filter_adults(df_raw)

    # Filter to valid income values
    df_valid = filter_valid_income(df_adults)

    # Select and recode variables
    df_selected = select_and_recode_variables(df_valid)

    # Sample
    df_sample = sample_data(df_selected, n=sample_size, random_state=random_state)

    # Train/test split
    df_train, df_test = train_test_split(
        df_sample, train_size=train_size, random_state=random_state
    )

    print("=" * 60)
    print("✓ Preprocessing complete")
    print(f"  Train: {len(df_train):,} records × {len(df_train.columns)} variables")
    print(f"  Test:  {len(df_test):,} records × {len(df_test.columns)} variables")
    print("=" * 60)

    return df_train, df_test


def load_and_preprocess(config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generic entry point: load CSV, optionally preprocess, split.

    For ACS PUMS (``config.preprocessing == "acs_pums"``), delegates to
    :func:`preprocess_acs_data`.  For other datasets, loads the CSV directly,
    samples, and splits.

    Returns:
        (df_train, df_test)
    """
    if config.preprocessing == "acs_pums":
        return preprocess_acs_data(
            data_path=config.data_path,
            sample_size=config.sample_size or 50000,
            train_size=config.train_fraction,
            random_state=config.random_state,
        )

    # Generic path: load CSV, sample, split
    df = pd.read_csv(config.data_path)
    if config.sample_size and len(df) > config.sample_size:
        df = df.sample(n=config.sample_size, random_state=config.random_state)
    return train_test_split(df, train_size=config.train_fraction, random_state=config.random_state)


if __name__ == "__main__":
    # Test the preprocessing pipeline
    df_train, df_test = preprocess_acs_data()

    print("\nTrain data preview:")
    print(df_train.head())

    print("\nTrain data info:")
    print(df_train.info())
