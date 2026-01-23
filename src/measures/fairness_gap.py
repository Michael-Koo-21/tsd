"""
Fairness Measure: Maximum Subgroup Utility Gap

Measures fairness by computing utility (TSTR F1 ratio) separately for each demographic
subgroup and reporting the maximum gap between any subgroup and the overall utility.

Lower gap = more fair (all subgroups achieve similar utility)
Higher gap = less fair (some subgroups disadvantaged)

Based on study_design_spec.md Section 4.2.4: "Compute TSTR F1 ratio separately for
each demographic subgroup. Report the maximum absolute difference between any subgroup's
ratio and the overall ratio."

Reference:
- Xu, R., Baracaldo, N., Zhou, Y., Anwar, A., & Ludwig, H. (2021). HyFed: A
  hybrid federated framework for privacy-preserving machine learning. In
  Proceedings of the 2021 ACM Workshop on Information Hiding and Multimedia Security.
"""

import pandas as pd
import numpy as np
from typing import Optional
from src.measures.tstr_utility import tstr_utility_per_subgroup


def fairness_gap(
    df_real_train: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    df_real_test: pd.DataFrame,
    target_col: str = 'HIGH_INCOME',
    subgroup_col: str = 'RAC1P_RECODED',
    min_subgroup_size: int = 100,
    n_estimators: int = 100,
    max_depth: int = 5,
    random_state: int = 42
) -> dict:
    """
    Compute maximum subgroup utility gap as fairness measure.

    Args:
        df_real_train: Real training data
        df_synthetic: Synthetic data
        df_real_test: Real test data
        target_col: Target variable name
        subgroup_col: Subgroup variable (e.g., 'RAC1P_RECODED' for race)
        min_subgroup_size: Minimum test set size to include subgroup
        n_estimators: Number of boosting stages
        max_depth: Maximum tree depth
        random_state: Random seed

    Returns:
        Dictionary with:
            - max_gap: Maximum absolute gap between any subgroup and overall utility
            - mean_gap: Mean absolute gap across all subgroups
            - subgroup_gaps: Dict of {subgroup: gap}
            - worst_subgroup: Subgroup with largest gap
            - overall_utility: Overall F1 ratio
            - fairness_score: Transformed score (1.0 = perfect fairness, 0.0 = worst)
    """
    # Compute per-subgroup utilities
    subgroup_result = tstr_utility_per_subgroup(
        df_real_train=df_real_train,
        df_synthetic=df_synthetic,
        df_real_test=df_real_test,
        target_col=target_col,
        subgroup_col=subgroup_col,
        min_subgroup_size=min_subgroup_size,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    overall_utility = subgroup_result['overall_f1_ratio']
    subgroup_utilities = subgroup_result['subgroup_f1_ratios']

    # Compute gaps (absolute difference from overall)
    subgroup_gaps = {}
    gaps = []

    for subgroup, utility in subgroup_utilities.items():
        gap = abs(utility - overall_utility)
        subgroup_gaps[subgroup] = gap
        gaps.append(gap)

    # Compute statistics
    max_gap = max(gaps) if gaps else 0.0
    mean_gap = np.mean(gaps) if gaps else 0.0

    # Identify worst subgroup
    if gaps:
        worst_subgroup = max(subgroup_gaps, key=subgroup_gaps.get)
    else:
        worst_subgroup = None

    # Fairness score: 1.0 - normalized max gap
    # We need to normalize by a reasonable worst-case gap
    # Assume worst-case gap = 0.5 (subgroup has half the utility of overall)
    worst_case_gap = 0.5
    fairness_score = max(0.0, 1.0 - (max_gap / worst_case_gap))

    return {
        'max_gap': max_gap,
        'mean_gap': mean_gap,
        'subgroup_gaps': subgroup_gaps,
        'worst_subgroup': worst_subgroup,
        'overall_utility': overall_utility,
        'subgroup_utilities': subgroup_utilities,
        'subgroup_sizes': subgroup_result['subgroup_sizes'],
        'excluded_subgroups': subgroup_result['excluded_subgroups'],
        'fairness_score': fairness_score
    }


def compare_fairness_across_methods(
    df_real_train: pd.DataFrame,
    synthetic_datasets: dict,
    df_real_test: pd.DataFrame,
    target_col: str = 'HIGH_INCOME',
    subgroup_col: str = 'RAC1P_RECODED',
    min_subgroup_size: int = 100
) -> pd.DataFrame:
    """
    Compare fairness gaps across multiple synthetic datasets.

    Args:
        df_real_train: Real training data
        synthetic_datasets: Dict of {method_name: df_synthetic}
        df_real_test: Real test data
        target_col: Target variable name
        subgroup_col: Subgroup variable
        min_subgroup_size: Minimum subgroup size

    Returns:
        DataFrame comparing fairness metrics across methods
    """
    results = []

    for method_name, df_synth in synthetic_datasets.items():
        result = fairness_gap(
            df_real_train=df_real_train,
            df_synthetic=df_synth,
            df_real_test=df_real_test,
            target_col=target_col,
            subgroup_col=subgroup_col,
            min_subgroup_size=min_subgroup_size
        )

        results.append({
            'method': method_name,
            'max_gap': result['max_gap'],
            'mean_gap': result['mean_gap'],
            'worst_subgroup': result['worst_subgroup'],
            'overall_utility': result['overall_utility'],
            'fairness_score': result['fairness_score']
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test with preprocessed ACS data
    import sys
    sys.path.append('.')
    from src.preprocessing.load_data import preprocess_acs_data
    from src.generators.independent_marginals import generate_independent_marginals
    from src.generators.ctgan_generator import generate_ctgan

    print("="*70)
    print("Testing Fairness Gap Measure")
    print("="*70)

    # Load data
    print("\nLoading and preprocessing data...")
    df_train, df_test = preprocess_acs_data(sample_size=2000, random_state=42)

    print(f"\nData sizes:")
    print(f"  Train: {len(df_train):,} records")
    print(f"  Test: {len(df_test):,} records")

    print(f"\nSubgroup distribution (test set):")
    subgroup_counts = df_test['RAC1P_RECODED'].value_counts()
    for subgroup, count in subgroup_counts.items():
        pct = 100 * count / len(df_test)
        print(f"  {subgroup}: {count} ({pct:.1f}%)")

    # Generate synthetic data
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
        epochs=50,
        verbose=False,
        random_state=42
    )

    # Compute fairness gaps
    print("\n" + "="*70)
    print("Fairness Gap Results")
    print("="*70)

    print("\n>>> Independent Marginals")
    result_indep = fairness_gap(
        df_real_train=df_train,
        df_synthetic=df_synth_indep,
        df_real_test=df_test,
        target_col='HIGH_INCOME',
        subgroup_col='RAC1P_RECODED',
        min_subgroup_size=50,
        random_state=42
    )

    print(f"  Overall Utility (F1 ratio): {result_indep['overall_utility']:.4f}")
    print(f"  Max Subgroup Gap: {result_indep['max_gap']:.4f}")
    print(f"  Mean Subgroup Gap: {result_indep['mean_gap']:.4f}")
    print(f"  Worst Subgroup: {result_indep['worst_subgroup']}")
    print(f"  Fairness Score: {result_indep['fairness_score']:.4f}")

    print(f"\n  Subgroup-specific gaps:")
    for subgroup, gap in result_indep['subgroup_gaps'].items():
        utility = result_indep['subgroup_utilities'][subgroup]
        size = result_indep['subgroup_sizes'][subgroup]
        print(f"    {subgroup}: gap={gap:.4f}, utility={utility:.4f} (n={size})")

    print("\n>>> CTGAN")
    result_ctgan = fairness_gap(
        df_real_train=df_train,
        df_synthetic=df_synth_ctgan,
        df_real_test=df_test,
        target_col='HIGH_INCOME',
        subgroup_col='RAC1P_RECODED',
        min_subgroup_size=50,
        random_state=42
    )

    print(f"  Overall Utility (F1 ratio): {result_ctgan['overall_utility']:.4f}")
    print(f"  Max Subgroup Gap: {result_ctgan['max_gap']:.4f}")
    print(f"  Mean Subgroup Gap: {result_ctgan['mean_gap']:.4f}")
    print(f"  Worst Subgroup: {result_ctgan['worst_subgroup']}")
    print(f"  Fairness Score: {result_ctgan['fairness_score']:.4f}")

    print(f"\n  Subgroup-specific gaps:")
    for subgroup, gap in result_ctgan['subgroup_gaps'].items():
        utility = result_ctgan['subgroup_utilities'][subgroup]
        size = result_ctgan['subgroup_sizes'][subgroup]
        print(f"    {subgroup}: gap={gap:.4f}, utility={utility:.4f} (n={size})")

    # Compare methods
    print("\n" + "="*70)
    print("Method Comparison")
    print("="*70)

    comparison = compare_fairness_across_methods(
        df_real_train=df_train,
        synthetic_datasets={
            'Independent Marginals': df_synth_indep,
            'CTGAN': df_synth_ctgan
        },
        df_real_test=df_test,
        target_col='HIGH_INCOME',
        subgroup_col='RAC1P_RECODED',
        min_subgroup_size=50
    )

    print("\n" + comparison.to_string(index=False))

    # Interpretation
    print("\n" + "="*70)
    print("Interpretation")
    print("="*70)

    print("\nFairness Gap Interpretation:")
    print("  - 0.00 = Perfect fairness (all subgroups have equal utility)")
    print("  - 0.05 = Minor fairness issue")
    print("  - 0.10 = Moderate fairness issue")
    print("  - 0.20+ = Significant fairness issue")

    print("\nComparison:")
    if result_ctgan['max_gap'] < result_indep['max_gap']:
        diff = result_indep['max_gap'] - result_ctgan['max_gap']
        print(f"  ✓ CTGAN is more fair (-{diff:.3f} gap)")
    elif result_indep['max_gap'] < result_ctgan['max_gap']:
        diff = result_ctgan['max_gap'] - result_indep['max_gap']
        print(f"  ✓ Independent Marginals is more fair (-{diff:.3f} gap)")
    else:
        print(f"  = Both methods have similar fairness")
