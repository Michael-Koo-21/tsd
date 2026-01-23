"""
Load and preprocess ACS PUMS California data.

Based on study_design_spec.md Section 2.2:
- Universe: Adults (AGEP >= 18)
- Variables: 8 predictors + 1 target
- Train/test split: 70%/30%
- Sample size: 50,000 total (35,000 train, 15,000 test)
"""

import pandas as pd
import numpy as np
from pathlib import Path


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
    print(f"Filtering to adults (AGEP >= 18)...")
    df_adults = df[df['AGEP'] >= 18].copy()
    print(f"✓ {len(df_adults):,} adult records ({len(df_adults)/len(df)*100:.1f}%)")
    return df_adults


def recode_race(rac1p_series: pd.Series) -> pd.Series:
    """
    Recode RAC1P (race) into broader categories.

    ACS PUMS RAC1P codes:
    1 = White alone
    2 = Black or African American alone
    3 = American Indian alone
    4 = Alaska Native alone
    5 = American Indian and Alaska Native tribes specified; or American Indian or Alaska Native, not specified and no other races
    6 = Asian alone
    7 = Native Hawaiian and Other Pacific Islander alone
    8 = Some Other Race alone
    9 = Two or More Races

    Recoding:
    1 = White
    2 = Black
    3 = Asian
    4 = Other single race (AIAN, NHPI, Other)
    5 = Two or more races
    """
    def recode_value(x):
        if x == 1:
            return 1  # White
        elif x == 2:
            return 2  # Black
        elif x == 6:
            return 3  # Asian
        elif x in [3, 4, 5, 7, 8]:
            return 4  # Other single race
        elif x == 9:
            return 5  # Two or more races
        else:
            return 4  # Default to Other

    return rac1p_series.apply(recode_value)


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

    Variables (from research_review_responses.md Decision #1):
    - AGEP: Age
    - PINCP: Total person's income
    - SEX: Sex
    - RAC1P_RECODED: Race (recoded)
    - SCHL_RECODED: Educational attainment (recoded)
    - MAR: Marital status
    - ESR: Employment status
    - POBP_RECODED: Place of birth (recoded)
    - PWGTP: Target variable (person weight)
    """
    print("Selecting and recoding variables...")

    # Select base variables
    df_selected = df[['AGEP', 'PINCP', 'SEX', 'MAR', 'ESR', 'PWGTP', 'RAC1P', 'SCHL', 'POBP']].copy()

    # Apply recoding
    df_selected['RAC1P_RECODED'] = recode_race(df_selected['RAC1P'])
    df_selected['SCHL_RECODED'] = recode_education(df_selected['SCHL'])
    df_selected['POBP_RECODED'] = recode_birthplace(df_selected['POBP'])

    # Drop original versions
    df_selected = df_selected.drop(columns=['RAC1P', 'SCHL', 'POBP'])

    # Reorder columns: predictors + target
    final_cols = [
        'AGEP', 'PINCP', 'SEX', 'RAC1P_RECODED', 'SCHL_RECODED',
        'MAR', 'ESR', 'POBP_RECODED', 'PWGTP'
    ]
    df_selected = df_selected[final_cols]

    print(f"✓ Selected and recoded 9 variables (8 predictors + 1 target)")
    print(f"  RAC1P: {df['RAC1P'].nunique()} → {df_selected['RAC1P_RECODED'].nunique()} categories")
    print(f"  SCHL:  {df['SCHL'].nunique()} → {df_selected['SCHL_RECODED'].nunique()} categories")
    print(f"  POBP:  {df['POBP'].nunique()} → {df_selected['POBP_RECODED'].nunique()} categories")

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
    df: pd.DataFrame,
    train_size: float = 0.7,
    random_state: int = 42
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
    random_state: int = 42
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
    print("="*60)
    print("ACS PUMS Data Preprocessing")
    print("="*60)

    # Load raw data
    df_raw = load_raw_acs_pums(data_path)

    # Filter to adults
    df_adults = filter_adults(df_raw)

    # Select and recode variables
    df_selected = select_and_recode_variables(df_adults)

    # Sample
    df_sample = sample_data(df_selected, n=sample_size, random_state=random_state)

    # Train/test split
    df_train, df_test = train_test_split(df_sample, train_size=train_size, random_state=random_state)

    print("="*60)
    print(f"✓ Preprocessing complete")
    print(f"  Train: {len(df_train):,} records × {len(df_train.columns)} variables")
    print(f"  Test:  {len(df_test):,} records × {len(df_test.columns)} variables")
    print("="*60)

    return df_train, df_test


if __name__ == "__main__":
    # Test the preprocessing pipeline
    df_train, df_test = preprocess_acs_data()

    print("\nTrain data preview:")
    print(df_train.head())

    print("\nTrain data info:")
    print(df_train.info())
