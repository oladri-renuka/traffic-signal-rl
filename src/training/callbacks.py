"""Custom RLlib callbacks for training monitoring and evaluation."""

from typing import Dict
import numpy as np

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrafficSignalCallback(DefaultCallbacks):
    """Custom callback for traffic signal RL training."""

    def on_train_result(self, algorithm: Algorithm, result: Dict, **info) -> None:
        """Called after each training iteration."""
        episode = result.get('episodes_total', 0)
        episode_reward_mean = result.get('episode_reward_mean', 0)
        policy_loss = result.get('info', {}).get('learner', {}).get('default_policy', {}).get('policy_loss', 0)

        if episode % 100 == 0:
            logger.info(
                f"Episode {episode}: "
                f"reward={episode_reward_mean:.4f}, "
                f"policy_loss={policy_loss:.4f}"
            )

    def on_evaluate_end(self, algorithm: Algorithm, evaluation_metrics: Dict, **info) -> None:
        """Called after evaluation phase."""
        # Extract evaluation metrics
        if 'evaluation' in evaluation_metrics:
            eval_reward = evaluation_metrics['evaluation'].get('episode_reward_mean', 0)
            logger.info(f"Evaluation reward: {eval_reward:.4f}")
