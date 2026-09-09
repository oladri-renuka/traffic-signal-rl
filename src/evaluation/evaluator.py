"""Multi-run evaluator for comparing all baselines and coordinated RL."""

import json
from pathlib import Path
from typing import Dict, List

from src.utils.logger import get_logger, setup_logger
from src.evaluation.metrics import (
    aggregate_metrics,
    statistical_significance_test,
    format_results_for_display,
    calculate_co2_emissions,
)
from src.baselines.fixed_timing import run_fixed_timing_baseline
from src.baselines.actuated_control import run_actuated_baseline
from src.baselines.independent_rl import run_independent_rl_baseline

logger = get_logger(__name__)


class BaselineEvaluator:
    """Evaluator for running and comparing all baselines."""

    def __init__(self, results_dir: str = 'experiments/results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def run_all_baselines(self, num_runs: int = 3, episodes_per_run: int = 10, seeds: List[int] = None) -> Dict:
        """
        Run all baselines multiple times with different seeds.

        Args:
            num_runs: Number of independent runs per baseline
            episodes_per_run: Number of episodes per run
            seeds: List of random seeds (default: [42, 123, 456])

        Returns:
            Dict with all results organized by baseline and seed
        """
        if seeds is None:
            seeds = [42, 123, 456][:num_runs]

        baselines = {
            'fixed_timing': run_fixed_timing_baseline,
            'actuated': run_actuated_baseline,
            'independent_rl': run_independent_rl_baseline,
        }

        all_results = {}

        for baseline_name, baseline_func in baselines.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"Running {baseline_name} baseline...")
            logger.info(f"{'='*70}")

            baseline_results = {
                'name': baseline_name,
                'runs': {}
            }

            for run_idx, seed in enumerate(seeds):
                logger.info(f"\nRun {run_idx + 1}/{num_runs} (seed={seed})")
                run_results = baseline_func(num_episodes=episodes_per_run, seed=seed)

                baseline_results['runs'][f'run_{run_idx}'] = {
                    'seed': seed,
                    'episodes': run_results,
                }

            all_results[baseline_name] = baseline_results

        self.results = all_results
        return all_results

    def aggregate_results(self) -> Dict:
        """
        Aggregate results across all runs and baselines.

        Returns:
            Dict with aggregated metrics per baseline
        """
        aggregated = {}

        for baseline_name, baseline_data in self.results.items():
            all_episodes = []

            # Collect all episodes from all runs
            for run_data in baseline_data['runs'].values():
                all_episodes.extend(run_data['episodes'])

            # Aggregate metrics
            aggregated[baseline_name] = aggregate_metrics(all_episodes)
            aggregated[baseline_name]['all_episode_results'] = all_episodes

        return aggregated

    def compute_statistical_tests(self, aggregated: Dict, baseline_name: str = 'fixed_timing') -> Dict:
        """
        Compute statistical significance tests comparing all methods vs a baseline.

        Args:
            aggregated: Aggregated metrics dict
            baseline_name: Name of baseline to compare against

        Returns:
            Dict with statistical test results
        """
        if baseline_name not in aggregated:
            logger.warning(f"Baseline {baseline_name} not found")
            return {}

        baseline_results = aggregated[baseline_name].get('all_episode_results', [])
        test_results = {}

        for method_name, method_data in aggregated.items():
            if method_name == baseline_name:
                continue

            method_results = method_data.get('all_episode_results', [])

            # Test on waiting time
            sig_test = statistical_significance_test(
                method_results, baseline_results, metric='avg_wait_time'
            )
            test_results[f'{method_name}_vs_{baseline_name}'] = sig_test

        return test_results

    def print_summary(self, aggregated: Dict, sig_tests: Dict = None) -> None:
        """Print summary of results to console."""
        logger.info(f"\n{'='*70}")
        logger.info("EVALUATION SUMMARY")
        logger.info(f"{'='*70}\n")

        for method_name in sorted(aggregated.keys()):
            metrics = aggregated[method_name]
            print(format_results_for_display(method_name, metrics))

        # Print CO₂ savings prominently
        logger.info(f"\n{'='*70}")
        logger.info("CO₂ SAVINGS SUMMARY")
        logger.info(f"{'='*70}\n")

        fixed_co2 = aggregated.get('fixed_timing', {}).get('co2_mean_kg', 0)

        for method_name in sorted(aggregated.keys()):
            co2 = aggregated[method_name].get('co2_mean_kg', 0)
            saved = fixed_co2 - co2
            pct = (saved / fixed_co2 * 100) if fixed_co2 > 0 else 0

            logger.info(f"{method_name:20s}: {co2:8.2f} kg CO₂ (saved: {saved:+8.2f} kg / {pct:+6.1f}%)")

        # Print statistical significance tests
        if sig_tests:
            logger.info(f"\n{'='*70}")
            logger.info("STATISTICAL SIGNIFICANCE TESTS")
            logger.info(f"{'='*70}\n")

            for test_name, test_result in sig_tests.items():
                if 'p_value' in test_result:
                    p_val = test_result['p_value']
                    is_sig = test_result.get('is_significant_at_0_05', False)
                    logger.info(f"{test_name}: p={p_val:.4f} {'✓ SIGNIFICANT' if is_sig else '✗ Not significant'}")

    def save_results(self, filename: str = 'baseline_evaluation.json') -> Path:
        """
        Save all results to JSON file.

        Args:
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.results_dir / filename

        # Prepare results for JSON serialization
        results_to_save = {}
        for baseline_name, baseline_data in self.results.items():
            results_to_save[baseline_name] = {
                'name': baseline_data['name'],
                'runs': {}
            }

            for run_name, run_data in baseline_data['runs'].items():
                results_to_save[baseline_name]['runs'][run_name] = {
                    'seed': run_data['seed'],
                    'num_episodes': len(run_data['episodes']),
                    'episodes': [
                        {
                            'steps': ep.get('episode_steps', 0),
                            'avg_reward': float(ep.get('avg_reward', 0)),
                            'avg_wait_time': float(ep.get('avg_wait_time', 0)),
                            'co2_kg': float(ep.get('co2_kg', 0)),
                        }
                        for ep in run_data['episodes']
                    ]
                }

        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        logger.info(f"Results saved to {output_path}")
        return output_path


def run_baseline_evaluation(
    num_runs: int = 3,
    episodes_per_run: int = 10
) -> None:
    """
    Run complete baseline evaluation with statistical tests.

    Args:
        num_runs: Number of independent runs per baseline
        episodes_per_run: Episodes per run
    """
    setup_logger('src', log_file='experiments/results/evaluation.log')

    evaluator = BaselineEvaluator()

    # Run all baselines
    logger.info("Starting baseline evaluation...")
    evaluator.run_all_baselines(num_runs=num_runs, episodes_per_run=episodes_per_run)

    # Aggregate results
    aggregated = evaluator.aggregate_results()

    # Statistical tests
    sig_tests = evaluator.compute_statistical_tests(aggregated, baseline_name='fixed_timing')

    # Print summary
    evaluator.print_summary(aggregated, sig_tests)

    # Save results
    evaluator.save_results()

    logger.info("Evaluation complete!")
