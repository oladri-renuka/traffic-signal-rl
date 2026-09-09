"""RLlib multi-agent PPO training loop."""

import json
import os
from pathlib import Path
from typing import Dict, Any

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune import CLIReporter

from src.utils.logger import get_logger, setup_logger
from src.environment.sumo_env import SUMOTrafficEnv
from src.training.config import get_ppo_config, get_model_config

logger = get_logger(__name__)


class TrafficSignalTrainer:
    """Trainer for multi-agent RL on traffic signal control."""

    def __init__(self, results_dir: str = 'experiments/results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.trainer = None
        self.training_results = []

    def train(
        self,
        num_episodes: int = 1000,
        eval_interval: int = 100,
        seed: int = 42,
        num_workers: int = 2,
        checkpoint_freq: int = 50
    ) -> Dict[str, Any]:
        """
        Train multi-agent PPO on traffic signal control.

        Args:
            num_episodes: Total training episodes
            eval_interval: Evaluation interval (episodes)
            seed: Random seed
            num_workers: Number of parallel workers
            checkpoint_freq: Save checkpoint every N episodes

        Returns:
            Training results dict
        """
        setup_logger('src', log_file=str(self.results_dir / 'training.log'))

        logger.info(f"Starting training: {num_episodes} episodes, seed={seed}")

        # Initialize ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        try:
            # Create environment
            def env_creator(config):
                return SUMOTrafficEnv(gui=False)

            # Get config
            config = get_ppo_config(env_creator, num_workers=num_workers)

            # Create trainer
            self.trainer = PPO(config=config)

            # Training loop
            episode_count = 0
            eval_count = 0

            while episode_count < num_episodes:
                # Train
                result = self.trainer.train()

                # Track results
                self.training_results.append({
                    'episode': episode_count,
                    'reward': result.get('episode_reward_mean', 0),
                    'policy_loss': result.get('info', {}).get('learner', {}).get('default_policy', {}).get('policy_loss', 0),
                })

                episode_count = result.get('episodes_total', episode_count)

                # Evaluate
                if eval_count % eval_interval == 0:
                    eval_result = self._evaluate(5)  # 5 eval episodes
                    logger.info(
                        f"Episode {episode_count}/{num_episodes}: "
                        f"reward={result.get('episode_reward_mean', 0):.2f}, "
                        f"eval_wait_time={eval_result.get('avg_wait_time', 0):.2f}s"
                    )

                # Checkpoint
                if eval_count % checkpoint_freq == 0 and eval_count > 0:
                    checkpoint_path = self.trainer.save(str(self.results_dir / 'checkpoints'))
                    logger.info(f"Checkpoint saved: {checkpoint_path}")

                eval_count += 1

            # Final evaluation
            final_eval = self._evaluate(10)
            logger.info(f"Final evaluation: avg_wait={final_eval.get('avg_wait_time', 0):.2f}s")

            # Save final model
            model_path = self.trainer.save(str(self.results_dir / 'final_model'))
            logger.info(f"Final model saved: {model_path}")

            # Save results
            self._save_results(seed)

            return {
                'final_model_path': model_path,
                'results_file': str(self.results_dir / f'training_results_seed{seed}.json'),
                'final_eval': final_eval,
            }

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return {'error': str(e)}

        finally:
            if self.trainer:
                self.trainer.stop()
            ray.shutdown()

    def _evaluate(self, num_episodes: int) -> Dict[str, float]:
        """
        Evaluate current policy on test episodes.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            Evaluation metrics dict
        """
        total_wait = 0
        total_co2 = 0

        for _ in range(num_episodes):
            env = SUMOTrafficEnv(gui=False)
            obs, _ = env.reset()

            episode_done = False
            while not episode_done:
                actions = {}
                for agent in env.agents:
                    # Get action from trained policy
                    policy = self.trainer.get_policy('traffic_light')
                    action, _, _ = policy.compute_single_action(obs[agent])
                    actions[agent] = action

                obs, _, dones, _, _ = env.step(actions)
                episode_done = dones['__all__']

            stats = env.get_episode_stats()
            total_wait += stats.get('avg_wait', 0)
            total_co2 += env.traci_manager.get_idle_co2()
            env.close()

        return {
            'avg_wait_time': total_wait / num_episodes,
            'avg_co2_kg': total_co2 / num_episodes,
            'num_eval_episodes': num_episodes,
        }

    def _save_results(self, seed: int) -> None:
        """Save training results to JSON."""
        results_file = self.results_dir / f'training_results_seed{seed}.json'

        with open(results_file, 'w') as f:
            json.dump(self.training_results, f, indent=2)

        logger.info(f"Training results saved to {results_file}")


def run_training(
    num_episodes: int = 1000,
    seed: int = 42,
    num_workers: int = 2,
    eval_interval: int = 100
) -> Dict[str, Any]:
    """
    Run training with given parameters.

    Args:
        num_episodes: Total training episodes
        seed: Random seed
        num_workers: Number of parallel workers
        eval_interval: Evaluation interval

    Returns:
        Training results
    """
    trainer = TrafficSignalTrainer()
    return trainer.train(
        num_episodes=num_episodes,
        seed=seed,
        num_workers=num_workers,
        eval_interval=eval_interval
    )
