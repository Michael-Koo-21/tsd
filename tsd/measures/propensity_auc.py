"""
Propensity AUC Measure

Measures how well a classifier can distinguish between real and synthetic data.
Lower AUC is better - indicates synthetic data is more similar to real data.

Based on study_design_spec.md Section 4.1.1: "Train gradient boosting classifier
to distinguish real (label=1) from synthetic (label=0). AUC close to 0.5 indicates
indistinguishability; close to 1.0 indicates easily detectable synthetic data."

Reference: Snoke, J., Raab, G. M., Nowok, B., Dibben, C., & Slavkovic, A. (2018).
General and specific utility measures for synthetic data. JRSS-A, 181(3), 663-688.

METHODOLOGY NOTE (Updated):
Per Snoke et al. 2018 and SDMetrics industry standard, propensity score evaluation
for adaptive models (like Gradient Boosting) MUST use cross-validation to prevent
overfitting bias. Training and evaluating on the same data inflates AUC estimates.

The correct approach uses Stratified K-Fold cross-validation (k=3 per SDMetrics):
1. Combine real + synthetic data with labels
2. Split into k folds using stratification
3. For each fold: train on k-1 folds, predict on held-out fold
4. Average the k AUC scores
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def propensity_auc(
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int = 3,
    n_splits: int = 3,
    random_state: int = 42,
) -> dict:
    """
    Compute propensity AUC - how well a classifier can distinguish real from synthetic.

    Uses cross-validation per Snoke et al. (2018) and SDMetrics methodology to avoid
    overfitting bias that occurs when training and evaluating on the same data.

    Args:
        df_real: Real data
        df_synthetic: Synthetic data
        n_estimators: Number of boosting stages (default 100)
        max_depth: Maximum depth of trees (default 3)
        n_splits: Number of cross-validation folds (default 3, per SDMetrics)
        random_state: Random seed

    Returns:
        Dictionary with:
            - auc: Cross-validated AUC score (0.5 = indistinguishable, 1.0 = easily distinguishable)
            - auc_fidelity: Transformed fidelity score (1.0 = perfect fidelity, 0.0 = poor)
            - fold_aucs: List of AUC scores from each fold
            - feature_importance: Dictionary of feature importances (from full model)
    """
    # Combine data and create labels
    df_real_labeled = df_real.copy()
    df_real_labeled["is_synthetic"] = 0

    df_synthetic_labeled = df_synthetic.copy()
    df_synthetic_labeled["is_synthetic"] = 1

    df_combined = pd.concat([df_real_labeled, df_synthetic_labeled], ignore_index=True)

    # Separate features and target
    X = df_combined.drop("is_synthetic", axis=1)
    y = df_combined["is_synthetic"]

    # Encode categorical features
    X_encoded = pd.DataFrame()
    feature_names = []

    for col in X.columns:
        if X[col].dtype in ["object", "category"]:
            # One-hot encode categorical
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
            feature_names.extend(dummies.columns.tolist())
        else:
            # Keep numeric as-is
            X_encoded[col] = X[col]
            feature_names.append(col)

    # Fill any remaining NaNs
    X_encoded = X_encoded.fillna(X_encoded.median())

    # Convert to numpy arrays
    X_array = X_encoded.to_numpy()
    y_array = y.to_numpy()

    # Use Stratified K-Fold cross-validation (per SDMetrics methodology)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_aucs = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for train_idx, test_idx in kf.split(X_array, y_array):
            clf = GradientBoostingClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
            )
            clf.fit(X_array[train_idx], y_array[train_idx])
            y_pred_proba = clf.predict_proba(X_array[test_idx])[:, 1]
            fold_auc = roc_auc_score(y_array[test_idx], y_pred_proba)
            fold_aucs.append(fold_auc)

    # Average AUC across folds
    auc = np.mean(fold_aucs)

    # Transform AUC to fidelity score (1 - 2*|AUC - 0.5|)
    # This gives 1.0 when AUC=0.5 (perfect) and 0.0 when AUC=1.0 (worst)
    auc_fidelity = 1.0 - 2.0 * abs(auc - 0.5)

    # Train full model for feature importances (informational only)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_full = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        clf_full.fit(X_array, y_array)

    feature_importance = dict(zip(feature_names, clf_full.feature_importances_))

    # Sort by importance
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    return {
        "auc": auc,
        "auc_fidelity": auc_fidelity,
        "fold_aucs": fold_aucs,
        "feature_importance": feature_importance,
    }


def propensity_auc_no_cv(
    df_real: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int = 3,
    random_state: int = 42,
) -> dict:
    """
    Compute propensity AUC WITHOUT cross-validation (legacy method).

    WARNING: This method trains and evaluates on the same data, which can lead
    to overfitting and biased (overly optimistic) AUC estimates. Use propensity_auc()
    with cross-validation instead for proper methodology per Snoke et al. (2018).

    This function is retained for backwards compatibility and comparison purposes.

    Args:
        df_real: Real data
        df_synthetic: Synthetic data
        n_estimators: Number of boosting stages (default 100)
        max_depth: Maximum depth of trees (default 3)
        random_state: Random seed

    Returns:
        Dictionary with:
            - auc: AUC score (BIASED - trained on same data as evaluation)
            - auc_fidelity: Transformed fidelity score
            - feature_importance: Dictionary of feature importances
    """
    # Combine data and create labels
    df_real_labeled = df_real.copy()
    df_real_labeled["is_synthetic"] = 0

    df_synthetic_labeled = df_synthetic.copy()
    df_synthetic_labeled["is_synthetic"] = 1

    df_combined = pd.concat([df_real_labeled, df_synthetic_labeled], ignore_index=True)

    # Separate features and target
    X = df_combined.drop("is_synthetic", axis=1)
    y = df_combined["is_synthetic"]

    # Encode categorical features
    X_encoded = pd.DataFrame()
    feature_names = []

    for col in X.columns:
        if X[col].dtype in ["object", "category"]:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
            feature_names.extend(dummies.columns.tolist())
        else:
            X_encoded[col] = X[col]
            feature_names.append(col)

    X_encoded = X_encoded.fillna(X_encoded.median())

    # Train propensity classifier (NO CV - trains and evaluates on same data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        clf.fit(X_encoded, y)

    y_pred_proba = clf.predict_proba(X_encoded)[:, 1]
    auc = roc_auc_score(y, y_pred_proba)
    auc_fidelity = 1.0 - 2.0 * abs(auc - 0.5)

    feature_importance = dict(zip(feature_names, clf.feature_importances_))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    return {"auc": auc, "auc_fidelity": auc_fidelity, "feature_importance": feature_importance}


def propensity_auc_holdout(
    df_real_train: pd.DataFrame,
    df_real_holdout: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int = 3,
    random_state: int = 42,
) -> dict:
    """
    Compute propensity AUC using train/holdout split.

    Trains on real_train vs synthetic, evaluates on real_holdout vs synthetic.
    This avoids overfitting issues.

    Args:
        df_real_train: Real training data
        df_real_holdout: Real holdout data (for evaluation)
        df_synthetic: Synthetic data
        n_estimators: Number of boosting stages
        max_depth: Maximum depth of trees
        random_state: Random seed

    Returns:
        Dictionary with auc, auc_fidelity, feature_importance
    """
    # Train set: real_train + half of synthetic
    n_synth_train = len(df_synthetic) // 2
    df_synth_train = df_synthetic.iloc[:n_synth_train]
    df_synth_holdout = df_synthetic.iloc[n_synth_train:]

    # Combine train data
    df_real_train_labeled = df_real_train.copy()
    df_real_train_labeled["is_synthetic"] = 0

    df_synth_train_labeled = df_synth_train.copy()
    df_synth_train_labeled["is_synthetic"] = 1

    df_train = pd.concat([df_real_train_labeled, df_synth_train_labeled], ignore_index=True)

    # Combine holdout data
    df_real_holdout_labeled = df_real_holdout.copy()
    df_real_holdout_labeled["is_synthetic"] = 0

    df_synth_holdout_labeled = df_synth_holdout.copy()
    df_synth_holdout_labeled["is_synthetic"] = 1

    df_holdout = pd.concat([df_real_holdout_labeled, df_synth_holdout_labeled], ignore_index=True)

    # Separate features and target
    X_train = df_train.drop("is_synthetic", axis=1)
    y_train = df_train["is_synthetic"]

    X_holdout = df_holdout.drop("is_synthetic", axis=1)
    y_holdout = df_holdout["is_synthetic"]

    # Encode categorical features
    X_train_encoded = pd.DataFrame()
    feature_names = []

    for col in X_train.columns:
        if X_train[col].dtype in ["object", "category"]:
            dummies_train = pd.get_dummies(X_train[col], prefix=col, drop_first=True)
            X_train_encoded = pd.concat([X_train_encoded, dummies_train], axis=1)
            feature_names.extend(dummies_train.columns.tolist())
        else:
            X_train_encoded[col] = X_train[col]
            feature_names.append(col)

    # Encode holdout with same features
    X_holdout_encoded = pd.DataFrame()
    for col in X_holdout.columns:
        if X_holdout[col].dtype in ["object", "category"]:
            dummies_holdout = pd.get_dummies(X_holdout[col], prefix=col, drop_first=True)
            X_holdout_encoded = pd.concat([X_holdout_encoded, dummies_holdout], axis=1)
        else:
            X_holdout_encoded[col] = X_holdout[col]

    # Align columns
    X_holdout_encoded = X_holdout_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

    # Fill NaNs
    X_train_encoded = X_train_encoded.fillna(X_train_encoded.median())
    X_holdout_encoded = X_holdout_encoded.fillna(X_train_encoded.median())

    # Train classifier
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        clf.fit(X_train_encoded, y_train)

    # Predict on holdout
    y_pred_proba = clf.predict_proba(X_holdout_encoded)[:, 1]

    # Compute AUC on holdout
    auc = roc_auc_score(y_holdout, y_pred_proba)

    # Transform to fidelity score
    auc_fidelity = 1.0 - 2.0 * abs(auc - 0.5)

    # Feature importances
    feature_importance = dict(zip(feature_names, clf.feature_importances_))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    return {"auc": auc, "auc_fidelity": auc_fidelity, "feature_importance": feature_importance}


if __name__ == "__main__":
    # Test with preprocessed ACS data
    from tsd.generators.independent_marginals import generate_independent_marginals
    from tsd.preprocessing.load_data import preprocess_acs_data

    print("=" * 60)
    print("Testing Propensity AUC Measure (Cross-Validated)")
    print("=" * 60)

    # Load data
    df_train, df_test = preprocess_acs_data(sample_size=1000)

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    df_synthetic = generate_independent_marginals(
        df_train=df_train, n_samples=len(df_train), random_state=42
    )

    # Compute propensity AUC with cross-validation (CORRECT method)
    print("\nComputing propensity AUC (with 3-fold CV - correct method)...")
    result_cv = propensity_auc(df_train, df_synthetic, random_state=42)

    print("\nCross-Validated Results:")
    print(f"  AUC (mean of folds): {result_cv['auc']:.4f}")
    print(f"  AUC per fold: {[f'{x:.4f}' for x in result_cv['fold_aucs']]}")
    print(f"  AUC Fidelity: {result_cv['auc_fidelity']:.4f}")

    # Compare with non-CV method to show the bias
    print("\n" + "-" * 60)
    print("Comparing with NO cross-validation (biased method)...")
    result_no_cv = propensity_auc_no_cv(df_train, df_synthetic, random_state=42)

    print(f"\nNon-CV AUC (biased): {result_no_cv['auc']:.4f}")
    print(f"CV AUC (unbiased):   {result_cv['auc']:.4f}")
    print(f"Bias (overestimate): {result_no_cv['auc'] - result_cv['auc']:.4f}")

    print("\nInterpretation:")
    print("  - AUC = 0.5 means perfect indistinguishability")
    print("  - AUC = 1.0 means easily distinguishable")
    print("  - Fidelity = 1.0 means perfect fidelity (AUC=0.5)")
    print("  - Fidelity = 0.0 means poor fidelity (AUC=1.0)")
    print("\n  The non-CV method typically overestimates AUC due to overfitting.")

    print("\nTop 5 most important features for discrimination:")
    for i, (feat, imp) in enumerate(list(result_cv["feature_importance"].items())[:5], 1):
        print(f"  {i}. {feat}: {imp:.4f}")

    # Test holdout version
    print("\n" + "=" * 60)
    print("Testing Holdout Version")
    print("=" * 60)

    result_holdout = propensity_auc_holdout(
        df_real_train=df_train, df_real_holdout=df_test, df_synthetic=df_synthetic, random_state=42
    )

    print("\nHoldout Results:")
    print(f"  AUC: {result_holdout['auc']:.4f}")
    print(f"  AUC Fidelity: {result_holdout['auc_fidelity']:.4f}")
