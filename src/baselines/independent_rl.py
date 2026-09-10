"""Independent RL baseline: single-agent RL per intersection (no coordination)."""

import numpy as np
import random
from typing import Dict, List

from src.utils.logger import get_logger
from src.utils.sumo_config import NUM_AGENTS, SIMULATION_TIME_STEP, EPISODE_LENGTH_STEPS
from src.environment.sumo_env import SUMOTrafficEnv

logger = get_logger(__name__)


class IndependentRLBaseline:
    """
    Independent RL agents: each agent learns only local reward (no global component).

    For now, uses simple greedy policy based on local queue length.
    Full RL training would happen separately with RLlib.
    """

    def __init__(self):
        # Each agent maintains simple Q-values (action-value estimates)
        # For demonstration, we use a simple rule-based greedy policy
        self.q_values = {i: np.zeros((4,)) for i in range(NUM_AGENTS)}

    def reset(self):
        """Reset baseline for new episode."""
        self.q_values = {i: np.zeros((4,)) for i in range(NUM_AGENTS)}

    def get_actions(self, observations: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        Get independent RL actions using greedy policy on local state.

        Each agent selects action (phase) based on local queue observation.
        No inter-agent communication or global reward component.

        Args:
            observations: Observations for all agents

        Returns:
            Dict mapping agent_id -> action (phase)
        """
        actions = {}

        for i in range(NUM_AGENTS):
            agent_id = f'agent_{i}'
            obs = observations[agent_id]

            # Local state: queue lengths (obs[0:4])
            queues = obs[0:4] * 20.0  # denormalize

            # Greedy policy: select phase that minimizes queue on other direction
            # If East-West queues are high, give green to North-South
            ew_queue = queues[0] + queues[1]
            ns_queue = queues[2] + queues[3]

            if ew_queue > ns_queue:
                action = 1  # NS green
            else:
                action = 0  # EW green

            actions[agent_id] = action

        return actions

    def run_episode(self, env: SUMOTrafficEnv) -> Dict:
        """
        Run one episode using independent RL greedy policy.

        Args:
            env: SUMO environment

        Returns:
            Episode statistics dict
        """
        observations, _ = env.reset()
        self.reset()

        episode_rewards = {agent: 0.0 for agent in env.agents}
        episode_wait_times = []
        step_count = 0

        try:
            for step in range(EPISODE_LENGTH_STEPS):
                # Get independent RL actions (greedy on local state only)
                actions = self.get_actions(observations)

                # Execute step
                observations, rewards, dones, _, infos = env.step(actions)

                # Accumulate rewards (note: rewards include global component from environment)
                for agent in env.agents:
                    episode_rewards[agent] += rewards[agent]

                # Collect statistics
                if step % 100 == 0:
                    stats = env.get_episode_stats()
                    episode_wait_times.append(stats.get('avg_wait', 0.0))

                step_count += 1

                if dones['__all__']:
                    break

            # Compute episode metrics
            avg_reward = np.mean(list(episode_rewards.values()))
            avg_wait = np.mean(episode_wait_times) if episode_wait_times else 0.0
            co2 = env.traci_manager.get_idle_co2()

            return {
                'episode_steps': step_count,
                'avg_reward': avg_reward,
                'avg_wait_time': avg_wait,
                'co2_kg': co2,
                'agent_rewards': episode_rewards,
            }

        except Exception as e:
            logger.error(f"Error running independent RL episode: {e}")
            return {
                'episode_steps': step_count,
                'avg_reward': 0.0,
                'avg_wait_time': 0.0,
                'co2_kg': 0.0,
                'error': str(e),
            }


def run_independent_rl_baseline(num_episodes: int = 10, seed: int = 42) -> List[Dict]:
    """
    Run independent RL baseline for multiple episodes (reusing SUMO connection).

    Args:
        num_episodes: Number of episodes to run
        seed: Random seed

    Returns:
        List of episode result dicts
    """
    baseline = IndependentRLBaseline()
    results = []

    logger.info(f"Starting independent RL baseline ({num_episodes} episodes)...")

    env = SUMOTrafficEnv(gui=False)

    try:
        for ep in range(num_episodes):
            result = baseline.run_episode(env)
            results.append(result)

            logger.info(
                f"Episode {ep+1}/{num_episodes}: "
                f"avg_wait={result.get('avg_wait_time', 0):.2f}s, "
                f"co2={result.get('co2_kg', 0):.2f}kg"
            )
    finally:
        env.close()

    return results
