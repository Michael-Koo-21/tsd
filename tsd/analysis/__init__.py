"""
Analysis utilities for experiments.

Includes:
- Statistical analysis (ANOVA, pairwise comparisons, effect sizes)
- Results aggregation
- Visualization utilities (coming soon)
"""

from tsd.constants import PROFILES, UserProfile

from .mada_framework import (
    calculate_weighted_scores,
    demo_all_profiles,
    generate_recommendation,
    get_method_scores,
    normalize_scores,
    plot_comparison,
    rank_methods,
    sensitivity_analysis,
)
from .statistical_analysis import (
    correlation_analysis,
    descriptive_statistics,
    generate_report,
    load_results,
    omnibus_tests,
    pairwise_comparisons,
    run_analysis,
)
__all__ = [
    # Statistical analysis
    "load_results",
    "descriptive_statistics",
    "omnibus_tests",
    "pairwise_comparisons",
    "correlation_analysis",
    "generate_report",
    "run_analysis",
    # MADA Framework
    "PROFILES",
    "UserProfile",
    "get_method_scores",
    "normalize_scores",
    "calculate_weighted_scores",
    "rank_methods",
    "sensitivity_analysis",
    "generate_recommendation",
    "plot_comparison",
    "demo_all_profiles",
]
