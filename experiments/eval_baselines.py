#!/usr/bin/env python3
"""
Evaluate all baseline methods with statistical significance testing.

Usage:
    python experiments/eval_baselines.py --num-runs 3 --num-episodes 100
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluator import BaselineEvaluator
from src.evaluation.plotter import plot_all_results
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate baseline methods for traffic signal control'
    )
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of independent runs per baseline')
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='Episodes per run')
    parser.add_argument('--results-dir', type=str, default='experiments/results',
                        help='Results directory')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')

    args = parser.parse_args()

    # Setup
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logger('src', log_file=str(results_dir / 'baselines_evaluation.log'))

    print(f"\nBaseline Evaluation Configuration:")
    print(f"  Runs per baseline: {args.num_runs}")
    print(f"  Episodes per run: {args.num_episodes}")
    print(f"  Results Dir: {results_dir}\n")

    # Run evaluation
    print("Starting baseline evaluation...\n")
    evaluator = BaselineEvaluator(results_dir=str(results_dir))
    evaluator.run_all_baselines(
        num_runs=args.num_runs,
        episodes_per_run=args.num_episodes
    )

    # Aggregate results
    print("\nAggregating results...\n")
    aggregated = evaluator.aggregate_results()

    # Statistical tests
    print("Computing statistical significance tests...\n")
    sig_tests = evaluator.compute_statistical_tests(aggregated, baseline_name='fixed_timing')

    # Print summary
    evaluator.print_summary(aggregated, sig_tests)

    # Save results
    results_file = evaluator.save_results('baseline_evaluation.json')

    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...\n")
        try:
            plots = plot_all_results(results_file=str(results_file), output_dir=str(results_dir))
            print(f"Generated {len(plots)} plots:")
            for plot_path in plots:
                print(f"  - {plot_path}")
        except Exception as e:
            print(f"Warning: Failed to generate plots: {e}")

    print(f"\n✓ Evaluation complete! Results saved to {results_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
