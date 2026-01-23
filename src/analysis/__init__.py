"""
Analysis utilities for experiments.

Includes:
- Statistical analysis (ANOVA, pairwise comparisons, effect sizes)
- Results aggregation
- Visualization utilities (coming soon)
"""

from .statistical_analysis import (
    load_results,
    descriptive_statistics,
    omnibus_tests,
    pairwise_comparisons,
    correlation_analysis,
    generate_report,
    run_analysis,
)

from .visualizations import (
    bar_chart_comparison,
    tradeoff_scatter_plots,
    box_plots,
    correlation_heatmap,
    radar_chart,
    pareto_frontier,
    generate_all_visualizations,
)

from .mada_framework import (
    PROFILES,
    UserProfile,
    get_method_scores,
    normalize_scores,
    calculate_weighted_scores,
    rank_methods,
    sensitivity_analysis,
    generate_recommendation,
    plot_comparison,
    demo_all_profiles,
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
    # Visualizations
    "bar_chart_comparison",
    "tradeoff_scatter_plots",
    "box_plots",
    "correlation_heatmap",
    "radar_chart",
    "pareto_frontier",
    "generate_all_visualizations",
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
