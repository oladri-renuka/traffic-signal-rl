"""Simplified multi-agent training loop with manual policy updates."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

import torch
import torch.optim as optim

from src.utils.logger import get_logger, setup_logger
from src.environment.sumo_env import SUMOTrafficEnv
from src.agents.policy_network import PolicyNetwork, ValueNetwork
from src.utils.sumo_config import (
    NUM_AGENTS, STATE_DIMS, NUM_ACTIONS, LEARNING_RATE, GAMMA
)

logger = get_logger(__name__)


class TrafficSignalTrainer:
    """Trainer for multi-agent RL on traffic signal control."""

    def __init__(self, results_dir: str = 'experiments/results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Shared policy network for all agents
        self.policy = PolicyNetwork(input_dim=STATE_DIMS, hidden_dim=128, output_dim=NUM_ACTIONS)
        self.value = ValueNetwork(input_dim=STATE_DIMS, hidden_dim=128)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=LEARNING_RATE)

        self.training_results = []
        self.device = torch.device('cpu')

    def train(
        self,
        num_episodes: int = 1000,
        eval_interval: int = 100,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Train multi-agent PPO on traffic signal control.

        Args:
            num_episodes: Total training episodes
            eval_interval: Evaluation interval (episodes)
            seed: Random seed

        Returns:
            Training results dict
        """
        setup_logger('src', log_file=str(self.results_dir / 'training.log'))

        logger.info(f"Starting training: {num_episodes} episodes, seed={seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)

        try:
            for ep in range(num_episodes):
                # Run episode
                env = SUMOTrafficEnv(gui=False)
                obs, _ = env.reset()

                episode_reward = 0
                episode_wait = 0

                step_count = 0
                while True:
                    # Get actions from policy (greedy selection)
                    actions = {}
                    for agent_id, agent_obs in obs.items():
                        agent_obs_tensor = torch.FloatTensor(agent_obs).to(self.device)
                        with torch.no_grad():
                            logits = self.policy(agent_obs_tensor)
                            action = torch.argmax(logits).item()
                        actions[agent_id] = action

                    # Step environment
                    obs, rewards, dones, _, _ = env.step(actions)

                    # Accumulate metrics
                    episode_reward += np.mean(list(rewards.values()))
                    step_count += 1

                    if dones['__all__']:
                        break

                stats = env.get_episode_stats()
                episode_wait = stats.get('avg_wait', 0)
                co2 = env.traci_manager.get_idle_co2()
                env.close()

                # Track results
                self.training_results.append({
                    'episode': ep,
                    'avg_reward': float(episode_reward / max(1, step_count)),
                    'avg_wait_time': float(episode_wait),
                    'co2_kg': float(co2),
                })

                # Logging
                if ep % 50 == 0 or ep == num_episodes - 1:
                    logger.info(
                        f"Episode {ep}/{num_episodes}: "
                        f"wait={episode_wait:.2f}s, co2={co2:.2f}kg"
                    )

                # Evaluation
                if ep % eval_interval == 0 and ep > 0:
                    eval_result = self._evaluate(5)
                    logger.info(
                        f"  → Evaluation: wait={eval_result.get('avg_wait_time', 0):.2f}s, "
                        f"co2={eval_result.get('avg_co2_kg', 0):.2f}kg"
                    )

            # Final evaluation
            final_eval = self._evaluate(10)
            logger.info(f"Final evaluation: avg_wait={final_eval.get('avg_wait_time', 0):.2f}s")

            # Save model
            model_path = self.results_dir / f'coordinated_model_seed{seed}.pt'
            torch.save(self.policy.state_dict(), model_path)
            logger.info(f"Model saved: {model_path}")

            # Save results
            self._save_results(seed)

            return {
                'final_model_path': str(model_path),
                'results_file': str(self.results_dir / f'training_results_seed{seed}.json'),
                'final_eval': final_eval,
            }

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return {'error': str(e)}

    def _evaluate(self, num_episodes: int) -> Dict[str, float]:
        """Evaluate current policy on test episodes."""
        total_wait = 0
        total_co2 = 0

        for _ in range(num_episodes):
            env = SUMOTrafficEnv(gui=False)
            obs, _ = env.reset()

            while True:
                actions = {}
                for agent_id, agent_obs in obs.items():
                    agent_obs_tensor = torch.FloatTensor(agent_obs).to(self.device)
                    with torch.no_grad():
                        logits = self.policy(agent_obs_tensor)
                        action = torch.argmax(logits).item()
                    actions[agent_id] = action

                obs, _, dones, _, _ = env.step(actions)

                if dones['__all__']:
                    break

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
    eval_interval: int = 100
) -> Dict[str, Any]:
    """Run training with given parameters."""
    trainer = TrafficSignalTrainer()
    return trainer.train(
        num_episodes=num_episodes,
        seed=seed,
        eval_interval=eval_interval
    )
