"""
TSTR (Train on Synthetic, Test on Real) Utility Measure

Measures utility by training a classifier on synthetic data and evaluating on real test data.
Compares performance to a classifier trained on real training data (gold standard).

F1 Ratio = 1.0 means synthetic data achieves parity with real data.
F1 Ratio < 1.0 means synthetic data has lower utility.

Based on study_design_spec.md Section 4.2.3: "Train gradient boosting classifier
on synthetic data, evaluate on real test set, compare to real-trained baseline."

Classifier Hyperparameters (Fixed, No Tuning):
- Model: GradientBoostingClassifier (sklearn)
- n_estimators: 100 (number of boosting stages)
- max_depth: 5 (maximum depth of individual trees)
- learning_rate: 0.1 (default)
- No hyperparameter tuning performed per study design to ensure fair comparison
  across synthetic data methods. These are standard defaults that provide
  reasonable performance without overfitting.

Reference:
- Dankar, F. K., & Ibrahim, M. (2021). Fake it till you make it: Guidelines for
  effective synthetic data generation. Applied Sciences, 11(5), 2158.
- Hernandez, M., Epelde, G., Alberdi, A., Cilla, R., & Rankin, D. (2022).
  Synthetic data generation for tabular health records: A systematic review.
  Neurocomputing, 493, 28-45.
"""

import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def tstr_utility(
    df_real_train: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    df_real_test: pd.DataFrame,
    target_col: str,
    n_estimators: int = 100,
    max_depth: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Compute TSTR (Train on Synthetic, Test on Real) utility measure.

    Uses GradientBoostingClassifier with fixed hyperparameters (no tuning)
    to ensure fair comparison across synthetic data generation methods.

    Args:
        df_real_train: Real training data
        df_synthetic: Synthetic data
        df_real_test: Real test data (for evaluation)
        target_col: Name of target variable
        n_estimators: Number of boosting stages (default: 100, no tuning)
        max_depth: Maximum depth of trees (default: 5, no tuning)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with:
            - f1_real: F1 score of model trained on real data
            - f1_synthetic: F1 score of model trained on synthetic data
            - f1_ratio: Ratio f1_synthetic / f1_real (1.0 = parity)
            - auc_real: AUC of model trained on real data
            - auc_synthetic: AUC of model trained on synthetic data
            - auc_ratio: Ratio auc_synthetic / auc_real
            - utility_score: Transformed score (0-1, higher = better)
    """
    # Separate features and target
    X_real_train = df_real_train.drop(target_col, axis=1)
    y_real_train = df_real_train[target_col]

    X_synthetic = df_synthetic.drop(target_col, axis=1)
    y_synthetic = df_synthetic[target_col]

    X_test = df_real_test.drop(target_col, axis=1)
    y_test = df_real_test[target_col]

    # Identify categorical and continuous columns
    categorical_cols = X_real_train.select_dtypes(include=["object", "category"]).columns.tolist()
    continuous_cols = X_real_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

    # Create preprocessing pipeline
    if len(categorical_cols) > 0 and len(continuous_cols) > 0:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_cols,
                ),
                ("num", StandardScaler(), continuous_cols),
            ]
        )
    elif len(categorical_cols) > 0:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_cols,
                )
            ]
        )
    else:
        preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), continuous_cols)])

    # Fit preprocessor on real training data
    X_real_train_enc = preprocessor.fit_transform(X_real_train)
    X_synthetic_enc = preprocessor.transform(X_synthetic)
    X_test_enc = preprocessor.transform(X_test)

    # Train classifier on REAL data (gold standard)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_real = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        clf_real.fit(X_real_train_enc, y_real_train)

    # Predict on test set
    y_pred_real = clf_real.predict(X_test_enc)
    y_pred_proba_real = clf_real.predict_proba(X_test_enc)[:, 1]

    f1_real = f1_score(y_test, y_pred_real)
    auc_real = roc_auc_score(y_test, y_pred_proba_real)

    # Train classifier on SYNTHETIC data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_synthetic = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        clf_synthetic.fit(X_synthetic_enc, y_synthetic)

    # Predict on test set
    y_pred_synthetic = clf_synthetic.predict(X_test_enc)
    y_pred_proba_synthetic = clf_synthetic.predict_proba(X_test_enc)[:, 1]

    f1_synthetic = f1_score(y_test, y_pred_synthetic)
    auc_synthetic = roc_auc_score(y_test, y_pred_proba_synthetic)

    # Compute ratios
    f1_ratio = f1_synthetic / f1_real if f1_real > 0 else 0.0
    auc_ratio = auc_synthetic / auc_real if auc_real > 0 else 0.0

    # Utility score: F1 ratio capped at 1.0 (can't do better than real)
    utility_score = min(1.0, f1_ratio)

    return {
        "f1_real": f1_real,
        "f1_synthetic": f1_synthetic,
        "f1_ratio": f1_ratio,
        "auc_real": auc_real,
        "auc_synthetic": auc_synthetic,
        "auc_ratio": auc_ratio,
        "utility_score": utility_score,
    }


def tstr_utility_per_subgroup(
    df_real_train: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    df_real_test: pd.DataFrame,
    target_col: str,
    subgroup_col: str,
    min_subgroup_size: int = 100,
    n_estimators: int = 100,
    max_depth: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Compute TSTR utility separately for each subgroup.

    Used for fairness measure (subgroup utility gaps).

    Args:
        df_real_train: Real training data
        df_synthetic: Synthetic data
        df_real_test: Real test data
        target_col: Target variable name
        subgroup_col: Subgroup variable (e.g., 'RACETH')
        min_subgroup_size: Minimum test set size to include subgroup
        n_estimators: Number of boosting stages
        max_depth: Maximum depth
        random_state: Random seed

    Returns:
        Dictionary with:
            - overall_f1_ratio: Overall F1 ratio (all data)
            - subgroup_f1_ratios: Dict of {subgroup_value: f1_ratio}
            - subgroup_sizes: Dict of {subgroup_value: test_set_size}
            - excluded_subgroups: List of subgroups excluded due to small size
    """
    # Compute overall utility
    overall_result = tstr_utility(
        df_real_train=df_real_train,
        df_synthetic=df_synthetic,
        df_real_test=df_real_test,
        target_col=target_col,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )

    # Get unique subgroups in test set
    subgroups = df_real_test[subgroup_col].unique()

    subgroup_f1_ratios = {}
    subgroup_sizes = {}
    excluded_subgroups = []

    # Train models once on full data
    X_real_train = df_real_train.drop(target_col, axis=1)
    y_real_train = df_real_train[target_col]

    X_synthetic = df_synthetic.drop(target_col, axis=1)
    y_synthetic = df_synthetic[target_col]

    # Identify column types
    categorical_cols = X_real_train.select_dtypes(include=["object", "category"]).columns.tolist()
    continuous_cols = X_real_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

    # Remove subgroup_col from features if it's in X
    if subgroup_col in categorical_cols:
        categorical_cols = [c for c in categorical_cols if c != subgroup_col]
    if subgroup_col in continuous_cols:
        continuous_cols = [c for c in continuous_cols if c != subgroup_col]

    # Create preprocessor
    if len(categorical_cols) > 0 and len(continuous_cols) > 0:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_cols,
                ),
                ("num", StandardScaler(), continuous_cols),
            ]
        )
    elif len(categorical_cols) > 0:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_cols,
                )
            ]
        )
    else:
        preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), continuous_cols)])

    # Fit preprocessor and train models
    X_real_train_enc = preprocessor.fit_transform(X_real_train)
    X_synthetic_enc = preprocessor.transform(X_synthetic)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Train on real
        clf_real = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        clf_real.fit(X_real_train_enc, y_real_train)

        # Train on synthetic
        clf_synthetic = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        clf_synthetic.fit(X_synthetic_enc, y_synthetic)

    # Evaluate on each subgroup
    for subgroup in subgroups:
        # Filter test set to this subgroup
        df_test_subgroup = df_real_test[df_real_test[subgroup_col] == subgroup]
        subgroup_size = len(df_test_subgroup)

        if subgroup_size < min_subgroup_size:
            excluded_subgroups.append(subgroup)
            continue

        X_test_subgroup = df_test_subgroup.drop(target_col, axis=1)
        y_test_subgroup = df_test_subgroup[target_col]

        # Transform test data
        X_test_subgroup_enc = preprocessor.transform(X_test_subgroup)

        # Predict with both models
        y_pred_real = clf_real.predict(X_test_subgroup_enc)
        y_pred_synthetic = clf_synthetic.predict(X_test_subgroup_enc)

        # Compute F1 scores
        f1_real = f1_score(y_test_subgroup, y_pred_real, zero_division=0)
        f1_synthetic = f1_score(y_test_subgroup, y_pred_synthetic, zero_division=0)

        # Compute ratio
        f1_ratio = f1_synthetic / f1_real if f1_real > 0 else 0.0

        subgroup_f1_ratios[str(subgroup)] = f1_ratio
        subgroup_sizes[str(subgroup)] = subgroup_size

    return {
        "overall_f1_ratio": overall_result["f1_ratio"],
        "subgroup_f1_ratios": subgroup_f1_ratios,
        "subgroup_sizes": subgroup_sizes,
        "excluded_subgroups": excluded_subgroups,
    }


if __name__ == "__main__":
    # Test with preprocessed ACS data
    from tsd.generators.ctgan_generator import generate_ctgan
    from tsd.generators.independent_marginals import generate_independent_marginals
    from tsd.preprocessing.load_data import preprocess_acs_data

    print("=" * 70)
    print("Testing TSTR Utility Measure")
    print("=" * 70)

    # Load data
    print("\nLoading and preprocessing data...")
    df_train, df_test = preprocess_acs_data(sample_size=2000, random_state=42)

    print("\nData sizes:")
    print(f"  Train: {len(df_train):,} records")
    print(f"  Test: {len(df_test):,} records")
    print(
        f"  Target distribution (test): {df_test['HIGH_INCOME'].value_counts(normalize=True).to_dict()}"
    )

    # Generate synthetic data
    print("\nGenerating synthetic data...")

    print("  1. Independent Marginals...")
    df_synth_indep = generate_independent_marginals(
        df_train=df_train, n_samples=len(df_train), random_state=42
    )

    print("  2. CTGAN...")
    df_synth_ctgan = generate_ctgan(
        df_train=df_train, n_samples=len(df_train), epochs=50, verbose=False, random_state=42
    )

    # Compute TSTR utility
    print("\n" + "=" * 70)
    print("TSTR Utility Results")
    print("=" * 70)

    print("\n>>> Independent Marginals")
    result_indep = tstr_utility(
        df_real_train=df_train,
        df_synthetic=df_synth_indep,
        df_real_test=df_test,
        target_col="HIGH_INCOME",
        random_state=42,
    )

    print(f"  Real model F1: {result_indep['f1_real']:.4f}")
    print(f"  Synthetic model F1: {result_indep['f1_synthetic']:.4f}")
    print(f"  F1 Ratio: {result_indep['f1_ratio']:.4f}")
    print(f"  Utility Score: {result_indep['utility_score']:.4f}")
    print(f"  AUC Ratio: {result_indep['auc_ratio']:.4f}")

    print("\n>>> CTGAN")
    result_ctgan = tstr_utility(
        df_real_train=df_train,
        df_synthetic=df_synth_ctgan,
        df_real_test=df_test,
        target_col="HIGH_INCOME",
        random_state=42,
    )

    print(f"  Real model F1: {result_ctgan['f1_real']:.4f}")
    print(f"  Synthetic model F1: {result_ctgan['f1_synthetic']:.4f}")
    print(f"  F1 Ratio: {result_ctgan['f1_ratio']:.4f}")
    print(f"  Utility Score: {result_ctgan['utility_score']:.4f}")
    print(f"  AUC Ratio: {result_ctgan['auc_ratio']:.4f}")

    # Test subgroup utility
    print("\n" + "=" * 70)
    print("Per-Subgroup Utility")
    print("=" * 70)

    print("\n>>> CTGAN - By Race")
    subgroup_result = tstr_utility_per_subgroup(
        df_real_train=df_train,
        df_synthetic=df_synth_ctgan,
        df_real_test=df_test,
        target_col="HIGH_INCOME",
        subgroup_col="RACETH",
        min_subgroup_size=50,
        random_state=42,
    )

    print(f"\nOverall F1 Ratio: {subgroup_result['overall_f1_ratio']:.4f}")
    print("\nSubgroup F1 Ratios:")
    for subgroup, ratio in subgroup_result["subgroup_f1_ratios"].items():
        size = subgroup_result["subgroup_sizes"][subgroup]
        print(f"  {subgroup}: {ratio:.4f} (n={size})")

    if subgroup_result["excluded_subgroups"]:
        print(f"\nExcluded subgroups (too small): {subgroup_result['excluded_subgroups']}")

    # Interpretation
    print("\n" + "=" * 70)
    print("Interpretation")
    print("=" * 70)

    print("\nF1 Ratio Interpretation:")
    print("  - 1.0 = Synthetic data achieves parity with real data")
    print("  - 0.8-1.0 = Good utility (minor degradation)")
    print("  - 0.6-0.8 = Moderate utility (noticeable degradation)")
    print("  - <0.6 = Poor utility (significant degradation)")

    print("\nComparison:")
    if result_ctgan["f1_ratio"] > result_indep["f1_ratio"]:
        diff = result_ctgan["f1_ratio"] - result_indep["f1_ratio"]
        print(f"  ✓ CTGAN has better utility (+{diff:.3f})")
    else:
        diff = result_indep["f1_ratio"] - result_ctgan["f1_ratio"]
        print(f"  ✓ Independent Marginals has better utility (+{diff:.3f})")
