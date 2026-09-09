#!/usr/bin/env python3
"""
Train coordinated multi-agent RL on traffic signal control.

Usage:
    python experiments/train_coordinated.py --seed 42 --episodes 100 --eval-interval 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import run_training
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description='Train coordinated multi-agent RL for traffic signal control'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--eval-interval', type=int, default=50, help='Evaluation interval')
    parser.add_argument('--results-dir', type=str, default='experiments/results',
                        help='Results directory')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logger('src', log_file=str(results_dir / f'train_seed{args.seed}.log'))

    print(f"\n{'='*60}")
    print(f"Multi-Agent RL Training: Traffic Signal Control")
    print(f"{'='*60}")
    print(f"Episodes: {args.episodes}")
    print(f"Seed: {args.seed}")
    print(f"Eval Interval: {args.eval_interval}")
    print(f"Results Dir: {results_dir}\n")

    results = run_training(
        num_episodes=args.episodes,
        seed=args.seed,
        eval_interval=args.eval_interval
    )

    if 'error' in results:
        print(f"✗ ERROR: {results['error']}")
        return 1
    else:
        print(f"\n{'='*60}")
        print(f"✓ Training Complete!")
        print(f"{'='*60}")
        print(f"Model: {results.get('final_model_path', 'N/A')}")
        print(f"Results: {results.get('results_file', 'N/A')}")
        if 'final_eval' in results:
            eval_result = results['final_eval']
            print(f"\nFinal Evaluation ({eval_result.get('num_eval_episodes', 0)} episodes):")
            print(f"  Avg wait time: {eval_result.get('avg_wait_time', 0):.2f}s")
            print(f"  Avg CO₂: {eval_result.get('avg_co2_kg', 0):.2f}kg")
        return 0


if __name__ == '__main__':
    sys.exit(main())
