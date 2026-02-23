"""
Fairness Measure: Maximum Subgroup Utility Gap

Measures fairness by computing utility (TSTR F1 ratio) separately for each demographic
subgroup and reporting the maximum gap between any subgroup and the overall utility.

Lower gap = more fair (all subgroups achieve similar utility)
Higher gap = less fair (some subgroups disadvantaged)

Reference:
- Xu, R., Baracaldo, N., Zhou, Y., Anwar, A., & Ludwig, H. (2021). HyFed: A
  hybrid federated framework for privacy-preserving machine learning. In
  Proceedings of the 2021 ACM Workshop on Information Hiding and Multimedia Security.
"""

import numpy as np
import pandas as pd

from tsd.measures.tstr_utility import tstr_utility_per_subgroup


def fairness_gap(
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
    Compute maximum subgroup utility gap as fairness measure.

    Args:
        df_real_train: Real training data
        df_synthetic: Synthetic data
        df_real_test: Real test data
        target_col: Target variable name
        subgroup_col: Subgroup variable (e.g., 'RACETH' for race)
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
        random_state=random_state,
    )

    overall_utility = subgroup_result["overall_f1_ratio"]
    subgroup_utilities = subgroup_result["subgroup_f1_ratios"]

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
        "max_gap": max_gap,
        "mean_gap": mean_gap,
        "subgroup_gaps": subgroup_gaps,
        "worst_subgroup": worst_subgroup,
        "overall_utility": overall_utility,
        "subgroup_utilities": subgroup_utilities,
        "subgroup_sizes": subgroup_result["subgroup_sizes"],
        "excluded_subgroups": subgroup_result["excluded_subgroups"],
        "fairness_score": fairness_score,
    }


def compare_fairness_across_methods(
    df_real_train: pd.DataFrame,
    synthetic_datasets: dict,
    df_real_test: pd.DataFrame,
    target_col: str,
    subgroup_col: str,
    min_subgroup_size: int = 100,
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
            min_subgroup_size=min_subgroup_size,
        )

        results.append(
            {
                "method": method_name,
                "max_gap": result["max_gap"],
                "mean_gap": result["mean_gap"],
                "worst_subgroup": result["worst_subgroup"],
                "overall_utility": result["overall_utility"],
                "fairness_score": result["fairness_score"],
            }
        )

    return pd.DataFrame(results)
