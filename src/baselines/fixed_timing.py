"""Fixed timing baseline: deterministic 30s/30s green/red cycles."""

import numpy as np
from typing import Dict, List

from src.utils.logger import get_logger
from src.utils.sumo_config import NUM_AGENTS, SIMULATION_TIME_STEP, EPISODE_LENGTH_STEPS
from src.environment.sumo_env import SUMOTrafficEnv

logger = get_logger(__name__)


class FixedTimingBaseline:
    """Fixed 30s/30s timing pattern (standard traffic signal)."""

    def __init__(self):
        self.green_duration = 30  # seconds
        self.red_duration = 30  # seconds
        self.cycle_duration = self.green_duration + self.red_duration
        self.last_switch_time = {}

    def reset(self):
        """Reset baseline for new episode."""
        self.last_switch_time = {i: 0 for i in range(NUM_AGENTS)}

    def get_actions(self, elapsed_time: float) -> Dict[str, int]:
        """
        Get fixed phase actions based on elapsed time.

        Phase pattern: 0 (EW green) for 30s, then 1 (NS green) for 30s, repeat.

        Args:
            elapsed_time: Simulation time in seconds

        Returns:
            Dict mapping agent_id -> phase (0 or 1)
        """
        actions = {}
        for i in range(NUM_AGENTS):
            # Alternating phases with fixed duration
            cycle_pos = elapsed_time % self.cycle_duration
            if cycle_pos < self.green_duration:
                phase = 0  # EW green
            else:
                phase = 1  # NS green

            actions[f'agent_{i}'] = phase

        return actions

    def run_episode(self, env: SUMOTrafficEnv) -> Dict:
        """
        Run one episode using fixed timing policy.

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
                # Get fixed actions
                elapsed_time = step * SIMULATION_TIME_STEP
                actions = self.get_actions(elapsed_time)

                # Execute step
                observations, rewards, dones, _, infos = env.step(actions)

                # Accumulate rewards
                for agent in env.agents:
                    episode_rewards[agent] += rewards[agent]

                # Collect statistics
                if step % 100 == 0:  # Sample every 100 steps
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
            logger.error(f"Error running fixed timing episode: {e}")
            return {
                'episode_steps': step_count,
                'avg_reward': 0.0,
                'avg_wait_time': 0.0,
                'co2_kg': 0.0,
                'error': str(e),
            }


def run_fixed_timing_baseline(num_episodes: int = 10, seed: int = 42) -> List[Dict]:
    """
    Run fixed timing baseline for multiple episodes (reusing SUMO connection).

    Args:
        num_episodes: Number of episodes to run
        seed: Random seed (for consistency)

    Returns:
        List of episode result dicts
    """
    baseline = FixedTimingBaseline()
    results = []

    logger.info(f"Starting fixed timing baseline ({num_episodes} episodes)...")

    # Create environment once, reuse across episodes
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
