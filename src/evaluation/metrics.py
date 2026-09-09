"""Metrics calculation for traffic signal RL evaluation."""

import numpy as np
from typing import List, Dict, Tuple
from scipy import stats

from src.utils.sumo_config import EPA_CO2_PER_MINUTE_KG


def calculate_co2_emissions(idle_minutes: float) -> float:
    """
    Calculate CO₂ emissions from idle vehicle time using EPA factor.

    Formula: idle_minutes × 2.4g CO₂/min ÷ 1000 = kg CO₂

    Args:
        idle_minutes: Total vehicle-minutes of idling

    Returns:
        CO₂ emissions in kg
    """
    # EPA idle emission: 2.4 grams per minute
    # Convert to kg: 2.4 / 1000 = 0.0024
    return idle_minutes * EPA_CO2_PER_MINUTE_KG


def compare_vs_baseline(coordinated_idle_min: float, baseline_idle_min: float) -> Dict[str, float]:
    """
    Compare coordinated RL vs baseline in terms of CO₂ savings.

    Args:
        coordinated_idle_min: Idle minutes for coordinated RL
        baseline_idle_min: Idle minutes for baseline

    Returns:
        Dict with co2_saved_kg and percentage_improvement
    """
    co2_coordinated = calculate_co2_emissions(coordinated_idle_min)
    co2_baseline = calculate_co2_emissions(baseline_idle_min)

    co2_saved = co2_baseline - co2_coordinated
    percentage_saved = (co2_saved / co2_baseline * 100) if co2_baseline > 0 else 0

    return {
        'co2_coordinated_kg': co2_coordinated,
        'co2_baseline_kg': co2_baseline,
        'co2_saved_kg': co2_saved,
        'percentage_improvement': percentage_saved,
    }


def aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple episodes/runs.

    Args:
        results: List of episode result dicts

    Returns:
        Aggregated metrics dict with mean, std, etc.
    """
    if not results:
        return {}

    wait_times = [r.get('avg_wait_time', 0) for r in results if 'avg_wait_time' in r]
    co2_values = [r.get('co2_kg', 0) for r in results if 'co2_kg' in r]
    rewards = [r.get('avg_reward', 0) for r in results if 'avg_reward' in r]

    return {
        'avg_wait_time_mean': np.mean(wait_times) if wait_times else 0,
        'avg_wait_time_std': np.std(wait_times) if wait_times else 0,
        'avg_wait_time_min': np.min(wait_times) if wait_times else 0,
        'avg_wait_time_max': np.max(wait_times) if wait_times else 0,
        'co2_mean_kg': np.mean(co2_values) if co2_values else 0,
        'co2_std_kg': np.std(co2_values) if co2_values else 0,
        'avg_reward_mean': np.mean(rewards) if rewards else 0,
        'avg_reward_std': np.std(rewards) if rewards else 0,
        'num_episodes': len(results),
    }


def statistical_significance_test(
    coordinated_results: List[Dict],
    baseline_results: List[Dict],
    metric: str = 'avg_wait_time'
) -> Dict[str, float]:
    """
    Perform paired t-test between coordinated RL and baseline.

    Args:
        coordinated_results: List of coordinated RL episode results
        baseline_results: List of baseline episode results
        metric: Metric to compare (e.g., 'avg_wait_time', 'co2_kg')

    Returns:
        Dict with t-stat, p-value, and Cohen's d effect size
    """
    coordinated_values = [r.get(metric, 0) for r in coordinated_results]
    baseline_values = [r.get(metric, 0) for r in baseline_results]

    if not coordinated_values or not baseline_values:
        return {'error': 'Insufficient data for t-test'}

    # Paired t-test (assumes equal length samples)
    # For different lengths, use independent samples t-test
    if len(coordinated_values) == len(baseline_values):
        t_stat, p_value = stats.ttest_rel(coordinated_values, baseline_values)
    else:
        t_stat, p_value = stats.ttest_ind(coordinated_values, baseline_values)

    # Cohen's d effect size
    mean_diff = np.mean(coordinated_values) - np.mean(baseline_values)
    pooled_std = np.sqrt(
        (np.std(coordinated_values)**2 + np.std(baseline_values)**2) / 2
    )
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'is_significant_at_0_05': p_value < 0.05,
        'mean_improvement': float(mean_diff),
    }


def format_results_for_display(
    method_name: str,
    aggregated_metrics: Dict[str, float],
    sig_test_results: Dict[str, float] = None
) -> str:
    """
    Format results for console display.

    Args:
        method_name: Name of the method (e.g., 'Fixed Timing')
        aggregated_metrics: Aggregated metrics dict
        sig_test_results: Statistical significance test results

    Returns:
        Formatted string for display
    """
    lines = [
        f"\n{'='*60}",
        f"Method: {method_name}",
        f"{'='*60}",
        f"Episodes run: {aggregated_metrics.get('num_episodes', 0)}",
        f"Avg wait time: {aggregated_metrics.get('avg_wait_time_mean', 0):.2f} ± {aggregated_metrics.get('avg_wait_time_std', 0):.2f} seconds",
        f"CO₂ emissions: {aggregated_metrics.get('co2_mean_kg', 0):.2f} ± {aggregated_metrics.get('co2_std_kg', 0):.2f} kg",
        f"Avg reward: {aggregated_metrics.get('avg_reward_mean', 0):.4f} ± {aggregated_metrics.get('avg_reward_std', 0):.4f}",
    ]

    if sig_test_results and 'p_value' in sig_test_results:
        p_val = sig_test_results['p_value']
        is_sig = sig_test_results.get('is_significant_at_0_05', False)
        lines.append(f"Statistical significance (vs baseline): p={p_val:.4f} {'✓ SIGNIFICANT' if is_sig else '✗ Not significant'}")
        lines.append(f"Cohen's d effect size: {sig_test_results.get('cohens_d', 0):.4f}")

    lines.append(f"{'='*60}")
    return '\n'.join(lines)
