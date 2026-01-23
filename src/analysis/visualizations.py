"""
Visualization utilities for synthetic data generation experiment results.

Creates:
- Bar charts comparing methods across metrics
- Trade-off scatter plots
- Box plots showing variance
- Correlation heatmaps
- Radar/spider charts for multi-metric comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'independent_marginals': '#4C72B0',
    'ctgan': '#55A868',
    'dpbn': '#C44E52',
    'synthpop': '#8172B3',
    'great': '#CCB974',
}
METHOD_LABELS = {
    'independent_marginals': 'Indep. Marginals',
    'ctgan': 'CTGAN',
    'dpbn': 'DP-BN (PrivBayes)',
    'synthpop': 'Synthpop',
    'great': 'GReaT',
}
METRIC_LABELS = {
    'fidelity_auc': 'Fidelity\n(Propensity AUC)',
    'privacy_dcr': 'Privacy\n(DCR)',
    'utility_tstr': 'Utility\n(TSTR)',
    'fairness_gap': 'Fairness\n(Gap ↓)',
}


def load_results(filepath: str | Path) -> pd.DataFrame:
    """Load experiment results from CSV."""
    return pd.read_csv(filepath)


def bar_chart_comparison(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Create grouped bar chart comparing all methods across all metrics.
    """
    metrics = ['fidelity_auc', 'privacy_dcr', 'utility_tstr', 'fairness_gap']
    methods = list(COLORS.keys())

    # Calculate means and stds
    summary = df.groupby('method')[metrics].agg(['mean', 'std'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        means = [summary.loc[m, (metric, 'mean')] if m in summary.index else 0 for m in methods]
        stds = [summary.loc[m, (metric, 'std')] if m in summary.index else 0 for m in methods]
        colors = [COLORS[m] for m in methods]
        labels = [METHOD_LABELS[m] for m in methods]

        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('')
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(METRIC_LABELS[metric].replace('\n', ' '), fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.annotate(f'{mean:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

        # Highlight best
        if metric == 'fairness_gap':
            best_idx = np.argmin(means)
        else:
            best_idx = np.argmax(means)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

    plt.suptitle('Synthetic Data Generation Methods Comparison\n(Error bars = 1 SD, gold border = best)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / 'method_comparison_bars.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def tradeoff_scatter_plots(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Create scatter plots showing trade-offs between metrics.
    """
    summary = df.groupby('method').agg({
        'fidelity_auc': ['mean', 'std'],
        'privacy_dcr': ['mean', 'std'],
        'utility_tstr': ['mean', 'std'],
        'fairness_gap': ['mean', 'std'],
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    tradeoffs = [
        ('fidelity_auc', 'utility_tstr', 'Fidelity vs Utility Trade-off'),
        ('privacy_dcr', 'utility_tstr', 'Privacy vs Utility Trade-off'),
        ('fidelity_auc', 'privacy_dcr', 'Fidelity vs Privacy (Aligned)'),
    ]

    for ax, (x_metric, y_metric, title) in zip(axes, tradeoffs):
        for method in COLORS.keys():
            if method not in summary.index:
                continue

            x_mean = summary.loc[method, (x_metric, 'mean')]
            y_mean = summary.loc[method, (y_metric, 'mean')]
            x_std = summary.loc[method, (x_metric, 'std')]
            y_std = summary.loc[method, (y_metric, 'std')]

            ax.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std,
                       fmt='o', markersize=12, color=COLORS[method],
                       ecolor=COLORS[method], elinewidth=2, capsize=5,
                       label=METHOD_LABELS[method], markeredgecolor='black', markeredgewidth=1)

        ax.set_xlabel(METRIC_LABELS[x_metric].replace('\n', ' '), fontsize=11)
        ax.set_ylabel(METRIC_LABELS[y_metric].replace('\n', ' '), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Trade-off Analysis Between Metrics\n(Error bars = 1 SD)',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()

    output_path = output_dir / 'tradeoff_scatter.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def box_plots(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Create box plots showing distribution of results across replicates.
    """
    metrics = ['fidelity_auc', 'privacy_dcr', 'utility_tstr', 'fairness_gap']
    methods = list(COLORS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        data = [df[df['method'] == m][metric].dropna().values for m in methods]
        labels = [METHOD_LABELS[m] for m in methods]

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

        for patch, method in zip(bp['boxes'], methods):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.7)

        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(METRIC_LABELS[metric].replace('\n', ' '), fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # Add individual points
        for i, (method, d) in enumerate(zip(methods, data)):
            x = np.random.normal(i + 1, 0.04, size=len(d))
            ax.scatter(x, d, alpha=0.6, color='black', s=20, zorder=3)

    plt.suptitle('Distribution of Results Across Replicates\n(Box = IQR, whiskers = 1.5×IQR, dots = individual runs)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / 'replicate_boxplots.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Create heatmap showing correlations between metrics.
    """
    metrics = ['fidelity_auc', 'privacy_dcr', 'utility_tstr', 'fairness_gap']

    # Use method-level means
    method_means = df.groupby('method')[metrics].mean()

    # Calculate Spearman correlation
    from scipy.stats import spearmanr

    n = len(metrics)
    corr_matrix = np.zeros((n, n))

    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                rho, _ = spearmanr(method_means[m1], method_means[m2])
                corr_matrix[i, j] = rho

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Spearman ρ', fontsize=11)

    # Set ticks
    labels = [METRIC_LABELS[m].replace('\n', ' ') for m in metrics]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Add correlation values
    for i in range(n):
        for j in range(n):
            color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                   ha='center', va='center', color=color, fontsize=12, fontweight='bold')

    ax.set_title('Metric Correlations (Spearman)\nBased on Method-Level Means',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'correlation_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def radar_chart(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Create radar/spider chart for multi-dimensional comparison.
    """
    metrics = ['fidelity_auc', 'privacy_dcr', 'utility_tstr', 'fairness_gap']
    methods = list(COLORS.keys())

    # Calculate means and normalize to 0-1 scale
    summary = df.groupby('method')[metrics].mean()

    # Normalize (min-max scaling per metric)
    normalized = summary.copy()
    for metric in metrics:
        min_val = summary[metric].min()
        max_val = summary[metric].max()
        if max_val > min_val:
            normalized[metric] = (summary[metric] - min_val) / (max_val - min_val)
        else:
            normalized[metric] = 0.5

    # Invert fairness_gap (lower is better)
    normalized['fairness_gap'] = 1 - normalized['fairness_gap']

    # Set up radar chart
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for method in methods:
        if method not in normalized.index:
            continue

        values = normalized.loc[method].tolist()
        values += values[:1]  # Complete the loop

        ax.plot(angles, values, 'o-', linewidth=2, color=COLORS[method],
                label=METHOD_LABELS[method], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=COLORS[method])

    # Set labels
    labels = ['Fidelity', 'Privacy', 'Utility', 'Fairness\n(inverted)']
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    ax.set_title('Multi-Dimensional Method Comparison\n(Normalized scores, larger = better)',
                fontsize=14, fontweight='bold', y=1.08)

    output_path = output_dir / 'radar_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def pareto_frontier(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Create Pareto frontier plot for fidelity vs utility trade-off.
    """
    summary = df.groupby('method').agg({
        'fidelity_auc': 'mean',
        'utility_tstr': 'mean',
    })

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all points
    for method in COLORS.keys():
        if method not in summary.index:
            continue

        x = summary.loc[method, 'fidelity_auc']
        y = summary.loc[method, 'utility_tstr']

        ax.scatter(x, y, s=200, color=COLORS[method], edgecolor='black',
                  linewidth=2, label=METHOD_LABELS[method], zorder=3)

        # Add method label
        offset = (10, 10) if method != 'synthpop' else (-60, 10)
        ax.annotate(METHOD_LABELS[method], (x, y), xytext=offset,
                   textcoords='offset points', fontsize=10, fontweight='bold')

    # Calculate and draw Pareto frontier
    points = [(summary.loc[m, 'fidelity_auc'], summary.loc[m, 'utility_tstr'], m)
              for m in summary.index]

    # Find Pareto optimal points (maximize both)
    pareto_points = []
    for p in points:
        is_dominated = False
        for q in points:
            if q[0] > p[0] and q[1] > p[1]:
                is_dominated = True
                break
            if q[0] >= p[0] and q[1] > p[1]:
                is_dominated = True
                break
            if q[0] > p[0] and q[1] >= p[1]:
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append(p)

    # Sort by x and draw frontier
    pareto_points.sort(key=lambda x: x[0])
    if len(pareto_points) > 1:
        px = [p[0] for p in pareto_points]
        py = [p[1] for p in pareto_points]
        ax.plot(px, py, 'k--', linewidth=2, alpha=0.5, label='Pareto Frontier')

    # Mark Pareto optimal points
    for p in pareto_points:
        ax.scatter(p[0], p[1], s=300, facecolors='none', edgecolors='gold',
                  linewidth=3, zorder=4)

    ax.set_xlabel('Fidelity (Propensity AUC) →', fontsize=12)
    ax.set_ylabel('Utility (TSTR) →', fontsize=12)
    ax.set_title('Pareto Frontier: Fidelity vs Utility\n(Gold circles = Pareto optimal, dashed = frontier)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add quadrant annotations
    ax.axhline(y=0.7, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0.85, color='gray', linestyle=':', alpha=0.5)

    output_path = output_dir / 'pareto_frontier.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def generate_all_visualizations(filepath: str | Path, output_dir: str | Path = None) -> dict:
    """
    Generate all visualizations and return paths.
    """
    df = load_results(filepath)

    if output_dir is None:
        output_dir = Path(filepath).parent / 'figures'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    print("Generating bar chart comparison...")
    results['bar_chart'] = bar_chart_comparison(df, output_dir)

    print("Generating trade-off scatter plots...")
    results['tradeoff_scatter'] = tradeoff_scatter_plots(df, output_dir)

    print("Generating box plots...")
    results['box_plots'] = box_plots(df, output_dir)

    print("Generating correlation heatmap...")
    results['correlation_heatmap'] = correlation_heatmap(df, output_dir)

    print("Generating radar chart...")
    results['radar_chart'] = radar_chart(df, output_dir)

    print("Generating Pareto frontier...")
    results['pareto_frontier'] = pareto_frontier(df, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}")

    return results


if __name__ == "__main__":
    import sys

    filepath = sys.argv[1] if len(sys.argv) > 1 else "results/experiments/all_results_complete.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    results = generate_all_visualizations(filepath, output_dir)

    print("\nGenerated files:")
    for name, path in results.items():
        print(f"  - {name}: {path}")
