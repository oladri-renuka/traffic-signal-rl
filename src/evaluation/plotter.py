"""Plotting utilities for evaluation results."""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_learning_curves(
    results: Dict,
    output_dir: str = 'experiments/results',
    show: bool = False
) -> Path:
    """
    Plot learning curves comparing all methods.

    Args:
        results: Dict with results organized by baseline
        output_dir: Output directory for plots
        show: If True, display plot

    Returns:
        Path to saved plot
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {'fixed_timing': 'red', 'actuated': 'orange', 'independent_rl': 'blue'}
    markers = {'fixed_timing': 'o', 'actuated': 's', 'independent_rl': '^'}

    for baseline_name, baseline_data in results.items():
        if baseline_name not in results:
            continue

        all_wait_times = []
        for run_data in baseline_data['runs'].values():
            wait_times = [ep.get('avg_wait_time', 0) for ep in run_data['episodes']]
            all_wait_times.append(wait_times)

        # Compute mean and std across runs
        all_wait_times = np.array(all_wait_times)
        mean_wait = np.mean(all_wait_times, axis=0)
        std_wait = np.std(all_wait_times, axis=0)

        episodes = np.arange(1, len(mean_wait) + 1)
        ax.plot(episodes, mean_wait, label=baseline_name, marker=markers.get(baseline_name),
                color=colors.get(baseline_name), linewidth=2, markersize=4)
        ax.fill_between(episodes, mean_wait - std_wait, mean_wait + std_wait, alpha=0.2,
                         color=colors.get(baseline_name))

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Waiting Time (seconds)', fontsize=12)
    ax.set_title('Learning Curves: Average Waiting Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plot_path = output_path / 'learning_curves.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Learning curves plot saved to {plot_path}")

    if show:
        plt.show()
    plt.close()

    return plot_path


def plot_co2_comparison(
    results: Dict,
    output_dir: str = 'experiments/results',
    show: bool = False
) -> Path:
    """
    Plot CO₂ emissions comparison across methods.

    Args:
        results: Dict with results organized by baseline
        output_dir: Output directory for plots
        show: If True, display plot

    Returns:
        Path to saved plot
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = []
    co2_means = []
    co2_stds = []

    for baseline_name, baseline_data in results.items():
        all_co2 = []
        for run_data in baseline_data['runs'].values():
            co2_values = [ep.get('co2_kg', 0) for ep in run_data['episodes']]
            all_co2.extend(co2_values)

        methods.append(baseline_name)
        co2_means.append(np.mean(all_co2))
        co2_stds.append(np.std(all_co2))

    # Bar plot with error bars
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, co2_means, yerr=co2_stds, capsize=10, alpha=0.7,
                   color=['red', 'orange', 'blue'])

    # Highlight best method
    best_idx = np.argmin(co2_means)
    bars[best_idx].set_color('green')

    ax.set_ylabel('CO₂ Emissions (kg)', fontsize=12)
    ax.set_title('CO₂ Emissions Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(co2_means, co2_stds)):
        ax.text(i, mean + std + 10, f'{mean:.1f}', ha='center', fontsize=10)

    plot_path = output_path / 'co2_comparison.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    logger.info(f"CO₂ comparison plot saved to {plot_path}")

    if show:
        plt.show()
    plt.close()

    return plot_path


def plot_waiting_time_distribution(
    results: Dict,
    output_dir: str = 'experiments/results',
    show: bool = False
) -> Path:
    """
    Plot waiting time distribution as box plots.

    Args:
        results: Dict with results organized by baseline
        output_dir: Output directory for plots
        show: If True, display plot

    Returns:
        Path to saved plot
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot = []
    labels = []

    for baseline_name, baseline_data in results.items():
        all_wait_times = []
        for run_data in baseline_data['runs'].values():
            wait_times = [ep.get('avg_wait_time', 0) for ep in run_data['episodes']]
            all_wait_times.extend(wait_times)

        data_to_plot.append(all_wait_times)
        labels.append(baseline_name)

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

    # Color the boxes
    colors = ['red', 'orange', 'blue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Average Waiting Time (seconds)', fontsize=12)
    ax.set_title('Waiting Time Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plot_path = output_path / 'waiting_time_distribution.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Waiting time distribution plot saved to {plot_path}")

    if show:
        plt.show()
    plt.close()

    return plot_path


def plot_all_results(results_file: str = 'experiments/results/baseline_evaluation.json',
                     output_dir: str = 'experiments/results') -> List[Path]:
    """
    Generate all plots from saved results file.

    Args:
        results_file: Path to JSON results file
        output_dir: Output directory for plots

    Returns:
        List of paths to generated plots
    """
    with open(results_file, 'r') as f:
        results = json.load(f)

    plots = []

    # Generate plots
    plots.append(plot_learning_curves(results, output_dir))
    plots.append(plot_co2_comparison(results, output_dir))
    plots.append(plot_waiting_time_distribution(results, output_dir))

    logger.info(f"Generated {len(plots)} plots")

    return plots
