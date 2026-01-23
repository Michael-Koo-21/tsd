"""
Distance to Closest Record (DCR) Privacy Measure

PRIMARY PRIVACY MEASURE for this study.

Measures privacy as the distribution of distances from each synthetic record
to its nearest neighbor in the training data. Low DCR indicates potential
memorization or disclosure risk.

The 5th percentile DCR is used: "What fraction of synthetic records are
suspiciously close to training records?"

Methodology Notes:
- Features are standardized (z-score normalization) before distance computation
  to ensure all variables contribute equally regardless of scale
- Categorical variables are one-hot encoded
- The "typical distance" baseline is computed from a 1000-record sample of 
  training data to establish what distance is expected by chance
- Privacy score is normalized relative to this typical distance

Interpretation:
- Higher DCR = better privacy (synthetic records are far from training records)
- DCR 5th percentile > 1.0: Good privacy (worst 5% still reasonably distant)
- DCR 5th percentile < 0.3: Privacy concern (some records very close to training)

Reference:
- Taub, J., Elliot, M., Pampaka, M., & Smith, D. (2018). Differential
  Correct Attribution Probability for Synthetic Data. Privacy in Statistical
  Databases, 122-137.
- El Emam, K., Mosquera, L., & Hoptroff, R. (2020). Practical Synthetic
  Data Generation. O'Reilly Media.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import warnings


def dcr_privacy(
    df_real_train: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    metric: str = 'euclidean',
    percentile: int = 5,
    normalize: bool = True,
    random_state: int = 42
) -> dict:
    """
    Compute Distance to Closest Record (DCR) privacy measure.

    For each synthetic record, computes the distance to its nearest neighbor
    in the training data. Returns the specified percentile of these distances.

    Args:
        df_real_train: Real training data
        df_synthetic: Synthetic data
        metric: Distance metric ('euclidean', 'manhattan', 'cosine')
        percentile: Which percentile to report (default: 5th = worst 5%)
        normalize: Whether to standardize features before computing distance
        random_state: Random seed

    Returns:
        Dictionary with:
            - dcr_percentile: The specified percentile of DCR values
            - dcr_mean: Mean DCR across all synthetic records
            - dcr_median: Median DCR
            - dcr_min: Minimum DCR (closest match)
            - num_close_matches: Number of synthetic records with DCR < threshold
            - privacy_score: Transformed score (0=worst, 1=best)
    """
    # Encode categorical features
    df_train_encoded = _encode_dataframe(df_real_train)
    df_synth_encoded = _encode_dataframe(df_synthetic)

    # Align columns
    common_cols = df_train_encoded.columns.intersection(df_synth_encoded.columns)
    df_train_encoded = df_train_encoded[common_cols]
    df_synth_encoded = df_synth_encoded[common_cols]

    # Handle missing values
    df_train_encoded = df_train_encoded.fillna(df_train_encoded.median())
    df_synth_encoded = df_synth_encoded.fillna(df_train_encoded.median())

    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(df_train_encoded)
        X_synth = scaler.transform(df_synth_encoded)
    else:
        X_train = df_train_encoded.values
        X_synth = df_synth_encoded.values

    # Compute pairwise distances (memory efficient: batch processing)
    n_synth = X_synth.shape[0]
    batch_size = min(1000, n_synth)  # Process in batches to avoid memory issues

    min_distances = []

    for i in range(0, n_synth, batch_size):
        batch_end = min(i + batch_size, n_synth)
        X_synth_batch = X_synth[i:batch_end]

        # Compute distances from this batch to all training records
        distances = cdist(X_synth_batch, X_train, metric=metric)

        # For each synthetic record, find minimum distance to training record
        batch_min_distances = distances.min(axis=1)
        min_distances.extend(batch_min_distances)

    min_distances = np.array(min_distances)

    # Compute statistics
    dcr_percentile = np.percentile(min_distances, percentile)
    dcr_mean = np.mean(min_distances)
    dcr_median = np.median(min_distances)
    dcr_min = np.min(min_distances)

    # Count "suspiciously close" matches (within 1 standard deviation of minimum)
    threshold = dcr_min + np.std(min_distances)
    num_close_matches = np.sum(min_distances < threshold)
    pct_close_matches = 100 * num_close_matches / len(min_distances)

    # Transform to privacy score (0 = worst privacy, 1 = best privacy)
    # Higher DCR = better privacy (records are further from training data)
    # We normalize by the median distance in training data (typical "random" distance)

    # Compute typical distance in training data itself
    # Sample 1000 pairs to estimate typical distance
    np.random.seed(random_state)
    n_train = X_train.shape[0]
    if n_train > 1000:
        sample_indices = np.random.choice(n_train, size=1000, replace=False)
        X_train_sample = X_train[sample_indices]
    else:
        X_train_sample = X_train

    # Compute pairwise distances within training data
    train_distances = cdist(X_train_sample, X_train_sample, metric=metric)
    # Exclude diagonal (distance to self = 0)
    train_distances = train_distances[np.triu_indices_from(train_distances, k=1)]
    typical_distance = np.median(train_distances)

    # Privacy score: ratio of DCR percentile to typical distance
    # If DCR percentile = typical distance → score = 0.5 (neutral)
    # If DCR percentile = 0 → score = 0 (worst - exact copies)
    # If DCR percentile = 2*typical → score = 1.0 (best)
    privacy_score = min(1.0, dcr_percentile / (2 * typical_distance))

    return {
        'dcr_percentile': dcr_percentile,
        'dcr_mean': dcr_mean,
        'dcr_median': dcr_median,
        'dcr_min': dcr_min,
        'num_close_matches': int(num_close_matches),
        'pct_close_matches': pct_close_matches,
        'typical_distance': typical_distance,
        'privacy_score': privacy_score,
        'percentile_used': percentile,
        'metric_used': metric
    }


def dcr_privacy_per_record(
    df_real_train: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    metric: str = 'euclidean',
    normalize: bool = True
) -> pd.DataFrame:
    """
    Compute DCR for each synthetic record.

    Returns a DataFrame with each synthetic record and its DCR value.

    Args:
        df_real_train: Real training data
        df_synthetic: Synthetic data
        metric: Distance metric
        normalize: Whether to standardize features

    Returns:
        DataFrame with synthetic records and column 'dcr' (distance to closest)
    """
    # Encode
    df_train_encoded = _encode_dataframe(df_real_train)
    df_synth_encoded = _encode_dataframe(df_synthetic)

    # Align
    common_cols = df_train_encoded.columns.intersection(df_synth_encoded.columns)
    df_train_encoded = df_train_encoded[common_cols]
    df_synth_encoded = df_synth_encoded[common_cols]

    # Handle missing
    df_train_encoded = df_train_encoded.fillna(df_train_encoded.median())
    df_synth_encoded = df_synth_encoded.fillna(df_train_encoded.median())

    # Normalize
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(df_train_encoded)
        X_synth = scaler.transform(df_synth_encoded)
    else:
        X_train = df_train_encoded.values
        X_synth = df_synth_encoded.values

    # Compute distances
    distances = cdist(X_synth, X_train, metric=metric)
    min_distances = distances.min(axis=1)
    closest_indices = distances.argmin(axis=1)

    # Create result
    df_result = df_synthetic.copy()
    df_result['dcr'] = min_distances
    df_result['closest_train_index'] = closest_indices

    return df_result


def _encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features using one-hot encoding.

    Args:
        df: Input dataframe

    Returns:
        Encoded dataframe with all numeric features
    """
    df_encoded = pd.DataFrame()

    for col in df.columns:
        if df[col].dtype in ['object', 'category']:
            # One-hot encode categorical
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
        else:
            # Keep numeric as-is
            df_encoded[col] = df[col]

    return df_encoded


def compare_dcr_distributions(
    df_real_train: pd.DataFrame,
    synthetic_datasets: dict,
    metric: str = 'euclidean',
    percentile: int = 5
) -> pd.DataFrame:
    """
    Compare DCR distributions across multiple synthetic datasets.

    Args:
        df_real_train: Real training data
        synthetic_datasets: Dict of {method_name: df_synthetic}
        metric: Distance metric
        percentile: Which percentile to report

    Returns:
        DataFrame comparing DCR statistics across methods
    """
    results = []

    for method_name, df_synth in synthetic_datasets.items():
        result = dcr_privacy(
            df_real_train=df_real_train,
            df_synthetic=df_synth,
            metric=metric,
            percentile=percentile
        )
        result['method'] = method_name
        results.append(result)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test with preprocessed ACS data
    import sys
    sys.path.append('.')
    from src.preprocessing.load_data import preprocess_acs_data
    from src.generators.independent_marginals import generate_independent_marginals
    from src.generators.ctgan_generator import generate_ctgan

    print("="*70)
    print("Testing DCR Privacy Measure")
    print("="*70)

    # Load data
    print("\nLoading and preprocessing data...")
    df_train, df_test = preprocess_acs_data(sample_size=2000, random_state=42)

    print(f"\nData sizes:")
    print(f"  Train: {len(df_train):,} records")
    print(f"  Test: {len(df_test):,} records")

    # Generate synthetic data with two methods
    print("\nGenerating synthetic data...")

    print("  1. Independent Marginals...")
    df_synth_indep = generate_independent_marginals(
        df_train=df_train,
        n_samples=len(df_train),
        random_state=42
    )

    print("  2. CTGAN...")
    df_synth_ctgan = generate_ctgan(
        df_train=df_train,
        n_samples=len(df_train),
        epochs=30,
        verbose=False,
        random_state=42
    )

    # Compute DCR for both methods
    print("\n" + "="*70)
    print("DCR Privacy Results")
    print("="*70)

    print("\n>>> Independent Marginals")
    result_indep = dcr_privacy(
        df_real_train=df_train,
        df_synthetic=df_synth_indep,
        percentile=5
    )

    print(f"  DCR 5th Percentile: {result_indep['dcr_percentile']:.4f}")
    print(f"  DCR Mean: {result_indep['dcr_mean']:.4f}")
    print(f"  DCR Median: {result_indep['dcr_median']:.4f}")
    print(f"  DCR Min: {result_indep['dcr_min']:.4f}")
    print(f"  Close Matches (<threshold): {result_indep['num_close_matches']} ({result_indep['pct_close_matches']:.1f}%)")
    print(f"  Typical Distance (in training): {result_indep['typical_distance']:.4f}")
    print(f"  Privacy Score: {result_indep['privacy_score']:.4f}")

    print("\n>>> CTGAN")
    result_ctgan = dcr_privacy(
        df_real_train=df_train,
        df_synthetic=df_synth_ctgan,
        percentile=5
    )

    print(f"  DCR 5th Percentile: {result_ctgan['dcr_percentile']:.4f}")
    print(f"  DCR Mean: {result_ctgan['dcr_mean']:.4f}")
    print(f"  DCR Median: {result_ctgan['dcr_median']:.4f}")
    print(f"  DCR Min: {result_ctgan['dcr_min']:.4f}")
    print(f"  Close Matches (<threshold): {result_ctgan['num_close_matches']} ({result_ctgan['pct_close_matches']:.1f}%)")
    print(f"  Typical Distance (in training): {result_ctgan['typical_distance']:.4f}")
    print(f"  Privacy Score: {result_ctgan['privacy_score']:.4f}")

    # Interpretation
    print("\n" + "="*70)
    print("Interpretation")
    print("="*70)

    print("\nDCR Privacy Score Interpretation:")
    print("  - 1.0 = Perfect privacy (synthetic records far from training)")
    print("  - 0.5 = Neutral (typical distance)")
    print("  - 0.0 = No privacy (exact copies of training records)")

    print("\nComparison:")
    if result_indep['privacy_score'] > result_ctgan['privacy_score']:
        diff = result_indep['privacy_score'] - result_ctgan['privacy_score']
        print(f"  ✓ Independent Marginals has better privacy (+{diff:.3f})")
    elif result_ctgan['privacy_score'] > result_indep['privacy_score']:
        diff = result_ctgan['privacy_score'] - result_indep['privacy_score']
        print(f"  ✓ CTGAN has better privacy (+{diff:.3f})")
    else:
        print(f"  = Both methods have similar privacy")

    # Compare distributions
    print("\n" + "="*70)
    print("Method Comparison")
    print("="*70)

    comparison = compare_dcr_distributions(
        df_real_train=df_train,
        synthetic_datasets={
            'Independent Marginals': df_synth_indep,
            'CTGAN': df_synth_ctgan
        },
        percentile=5
    )

    print("\n" + comparison[['method', 'dcr_percentile', 'dcr_mean', 'dcr_median',
                              'privacy_score']].to_string(index=False))
