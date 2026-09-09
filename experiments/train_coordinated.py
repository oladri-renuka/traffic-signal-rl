#!/usr/bin/env python3
"""
Train coordinated multi-agent RL on traffic signal control.

Usage:
    python experiments/train_coordinated.py --seed 42 --episodes 1000 --eval-interval 100
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import run_training
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description='Train coordinated multi-agent RL for traffic signal control'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--eval-interval', type=int, default=100, help='Evaluation interval')
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers')
    parser.add_argument('--results-dir', type=str, default='experiments/results',
                        help='Results directory')

    args = parser.parse_args()

    # Setup logging
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logger('src', log_file=str(results_dir / f'train_seed{args.seed}.log'))

    print(f"Training Configuration:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Seed: {args.seed}")
    print(f"  Eval Interval: {args.eval_interval}")
    print(f"  Workers: {args.workers}")
    print(f"  Results Dir: {results_dir}\n")

    # Run training
    results = run_training(
        num_episodes=args.episodes,
        seed=args.seed,
        num_workers=args.workers,
        eval_interval=args.eval_interval
    )

    # Print results
    if 'error' in results:
        print(f"ERROR: {results['error']}")
        return 1
    else:
        print(f"\nTraining Complete!")
        print(f"Final model saved: {results.get('final_model_path', 'N/A')}")
        print(f"Results saved: {results.get('results_file', 'N/A')}")
        if 'final_eval' in results:
            final_eval = results['final_eval']
            print(f"Final evaluation:")
            print(f"  Avg wait time: {final_eval.get('avg_wait_time', 0):.2f}s")
            print(f"  Avg CO₂: {final_eval.get('avg_co2_kg', 0):.2f}kg")
        return 0


if __name__ == '__main__':
    sys.exit(main())
