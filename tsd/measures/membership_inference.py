"""
Membership Inference Privacy Measure

Measures privacy as resistance to membership inference attacks.
Higher attack success rate = worse privacy (more memorization of training data).

Based on research_review_responses.md Decision #2: "Use membership inference
attack success rate as primary privacy measure. This measures whether an
attacker can identify if a synthetic record was derived from a specific
training record."

Reference: Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017).
Membership Inference Attacks Against Machine Learning Models. S&P 2017.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


def membership_inference_privacy(
    df_real_train: pd.DataFrame,
    df_real_test: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    random_state: int = 42,
    min_attack_auc: float = 0.6,
) -> dict:
    """
    Measure privacy as resistance to membership inference attacks.

    The attack:
    1. Train a classifier to distinguish training members (df_real_train) from
       non-members (df_real_test) using the synthetic data as proxy
    2. For each synthetic record, predict if it came from a training member
    3. High attack success rate = synthetic data reveals training membership

    IMPORTANT: This measure is only valid if the attack classifier achieves
    AUC > min_attack_auc on real data. If the attack itself doesn't work,
    the measure cannot meaningfully evaluate synthetic data privacy.

    Args:
        df_real_train: Real training data (members)
        df_real_test: Real test data (non-members)
        df_synthetic: Synthetic data generated from training data
        random_state: Random seed
        min_attack_auc: Minimum AUC required for attack to be considered valid (default: 0.6)

    Returns:
        Dictionary with:
            - attack_success_rate: Fraction of synthetic records classified as members
            - attack_auc: AUC of membership classifier on real data
            - attack_valid: Boolean - whether attack is strong enough to be meaningful
            - mean_membership_confidence: Average membership probability
            - privacy_score: Transformed score (1.0 = perfect privacy, 0.0 = no privacy)
            - warning: String message if attack is too weak
    """
    # Label real data: train=1 (members), test=0 (non-members)
    df_train_labeled = df_real_train.copy()
    df_train_labeled["is_member"] = 1

    df_test_labeled = df_real_test.copy()
    df_test_labeled["is_member"] = 0

    # Combine labeled real data
    df_real_combined = pd.concat([df_train_labeled, df_test_labeled], ignore_index=True)

    # Separate features and labels
    X_real = df_real_combined.drop("is_member", axis=1)
    y_real = df_real_combined["is_member"]

    # Encode categorical features
    X_real_encoded = pd.DataFrame()
    feature_names = []

    for col in X_real.columns:
        if X_real[col].dtype in ["object", "category"]:
            dummies = pd.get_dummies(X_real[col], prefix=col, drop_first=True)
            X_real_encoded = pd.concat([X_real_encoded, dummies], axis=1)
            feature_names.extend(dummies.columns.tolist())
        else:
            X_real_encoded[col] = X_real[col]
            feature_names.append(col)

    # Fill NaNs
    X_real_encoded = X_real_encoded.fillna(X_real_encoded.median())

    # Train attack classifier
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        attack_model = LogisticRegression(max_iter=1000, random_state=random_state, solver="lbfgs")
        attack_model.fit(X_real_encoded, y_real)

    # Encode synthetic data with same features
    X_synth = df_synthetic.copy()
    X_synth_encoded = pd.DataFrame()

    for col in X_synth.columns:
        if X_synth[col].dtype in ["object", "category"]:
            dummies = pd.get_dummies(X_synth[col], prefix=col, drop_first=True)
            X_synth_encoded = pd.concat([X_synth_encoded, dummies], axis=1)
        else:
            X_synth_encoded[col] = X_synth[col]

    # Align columns with training data
    X_synth_encoded = X_synth_encoded.reindex(columns=X_real_encoded.columns, fill_value=0)
    X_synth_encoded = X_synth_encoded.fillna(X_real_encoded.median())

    # Predict membership for synthetic records
    membership_probs = attack_model.predict_proba(X_synth_encoded)[:, 1]
    membership_predictions = (membership_probs > 0.5).astype(int)

    # Attack success rate: How often does attacker predict "member"
    attack_success_rate = membership_predictions.mean()

    # Mean membership confidence
    mean_membership_confidence = membership_probs.mean()

    # AUC on real data (to assess attack quality) — use cross-validation
    # to avoid inflated training-set AUC that would be identical across runs
    cv_model = LogisticRegression(max_iter=1000, random_state=random_state, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_real_pred_proba = cross_val_predict(
            cv_model, X_real_encoded, y_real, cv=5, method="predict_proba"
        )[:, 1]
    attack_auc = roc_auc_score(y_real, y_real_pred_proba)

    # Validate attack quality
    attack_valid = attack_auc >= min_attack_auc
    warning = None

    if not attack_valid:
        warning = (
            f"⚠️  Membership inference attack is too weak (AUC={attack_auc:.3f} < {min_attack_auc}). "
            f"This means the attack classifier cannot reliably distinguish training members from "
            f"non-members in real data. Results may not be meaningful. "
            f"Consider using DCR (Distance to Closest Record) as alternative privacy measure."
        )
        warnings.warn(warning, stacklevel=2)

    # Privacy score: Transform attack success rate to privacy score
    # Perfect privacy: attack_success_rate = 0.5 (random guessing) → privacy = 1.0
    # No privacy: attack_success_rate = 1.0 (always correct) → privacy = 0.0
    # Formula: privacy = 1 - 2*|attack_success_rate - 0.5|
    privacy_score = 1.0 - 2.0 * abs(attack_success_rate - 0.5)

    # If attack is invalid, set privacy_score to NaN to signal unreliable measure
    if not attack_valid:
        privacy_score = np.nan

    return {
        "attack_success_rate": attack_success_rate,
        "attack_auc": attack_auc,
        "attack_valid": attack_valid,
        "mean_membership_confidence": mean_membership_confidence,
        "privacy_score": privacy_score,
        "warning": warning,
    }


def membership_inference_privacy_per_record(
    df_real_train: pd.DataFrame,
    df_real_test: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute membership inference scores per synthetic record.

    Returns a DataFrame with each synthetic record and its membership probability.

    Args:
        df_real_train: Real training data (members)
        df_real_test: Real test data (non-members)
        df_synthetic: Synthetic data

    Returns:
        DataFrame with synthetic records and columns:
            - membership_prob: Probability of being from training set
            - is_likely_member: Binary flag (prob > 0.5)
    """
    # Same encoding and training as above
    df_train_labeled = df_real_train.copy()
    df_train_labeled["is_member"] = 1

    df_test_labeled = df_real_test.copy()
    df_test_labeled["is_member"] = 0

    df_real_combined = pd.concat([df_train_labeled, df_test_labeled], ignore_index=True)

    X_real = df_real_combined.drop("is_member", axis=1)
    y_real = df_real_combined["is_member"]

    # Encode
    X_real_encoded = pd.DataFrame()
    for col in X_real.columns:
        if X_real[col].dtype in ["object", "category"]:
            dummies = pd.get_dummies(X_real[col], prefix=col, drop_first=True)
            X_real_encoded = pd.concat([X_real_encoded, dummies], axis=1)
        else:
            X_real_encoded[col] = X_real[col]

    X_real_encoded = X_real_encoded.fillna(X_real_encoded.median())

    # Train
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        attack_model = LogisticRegression(max_iter=1000, random_state=random_state, solver="lbfgs")
        attack_model.fit(X_real_encoded, y_real)

    # Encode synthetic
    X_synth = df_synthetic.copy()
    X_synth_encoded = pd.DataFrame()

    for col in X_synth.columns:
        if X_synth[col].dtype in ["object", "category"]:
            dummies = pd.get_dummies(X_synth[col], prefix=col, drop_first=True)
            X_synth_encoded = pd.concat([X_synth_encoded, dummies], axis=1)
        else:
            X_synth_encoded[col] = X_synth[col]

    X_synth_encoded = X_synth_encoded.reindex(columns=X_real_encoded.columns, fill_value=0)
    X_synth_encoded = X_synth_encoded.fillna(X_real_encoded.median())

    # Predict
    membership_probs = attack_model.predict_proba(X_synth_encoded)[:, 1]

    # Create result DataFrame
    df_result = df_synthetic.copy()
    df_result["membership_prob"] = membership_probs
    df_result["is_likely_member"] = (membership_probs > 0.5).astype(int)

    return df_result
