"""Actuated control baseline: queue-responsive traffic signal timing."""

import numpy as np
from typing import Dict, List

from src.utils.logger import get_logger
from src.utils.sumo_config import NUM_AGENTS, SIMULATION_TIME_STEP, EPISODE_LENGTH_STEPS
from src.environment.sumo_env import SUMOTrafficEnv

logger = get_logger(__name__)


class ActuatedControlBaseline:
    """Simple actuated control: extend green if queue detected."""

    def __init__(self, queue_threshold: int = 5, min_green: int = 10, max_green: int = 60):
        """
        Initialize actuated control.

        Args:
            queue_threshold: Queue length threshold to trigger phase extension
            min_green: Minimum green phase duration (seconds)
            max_green: Maximum green phase duration (seconds)
        """
        self.queue_threshold = queue_threshold
        self.min_green = min_green
        self.max_green = max_green
        self.current_phase = {}
        self.phase_start_time = {}

    def reset(self):
        """Reset baseline for new episode."""
        self.current_phase = {i: 0 for i in range(NUM_AGENTS)}
        self.phase_start_time = {i: 0 for i in range(NUM_AGENTS)}

    def get_actions(self, elapsed_time: float, observations: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        Get actuated control actions based on observations.

        Simple rule: if queue > threshold, keep green; otherwise switch after min_green.

        Args:
            elapsed_time: Simulation time in seconds
            observations: Observations for all agents

        Returns:
            Dict mapping agent_id -> phase
        """
        actions = {}

        for i in range(NUM_AGENTS):
            agent_id = f'agent_{i}'
            obs = observations[agent_id]

            # Get queue length for current phase (state indices 0-3)
            queues = obs[0:4] * 20.0  # denormalize: max 20 vehicles
            current_phase = self.current_phase[i]
            time_in_phase = elapsed_time - self.phase_start_time[i]

            # Determine if we should switch phase
            should_switch = False

            if time_in_phase >= self.max_green:
                # Force switch at max_green
                should_switch = True
            elif time_in_phase >= self.min_green:
                # Check queue: if queue is low, switch
                # Assume current phase serves queues at indices [0,1] or [2,3]
                if current_phase == 0:  # EW green
                    current_queue = (queues[0] + queues[1]) / 2
                else:  # NS green
                    current_queue = (queues[2] + queues[3]) / 2

                if current_queue < self.queue_threshold:
                    should_switch = True

            if should_switch:
                self.current_phase[i] = (self.current_phase[i] + 1) % 2
                self.phase_start_time[i] = elapsed_time

            actions[agent_id] = self.current_phase[i]

        return actions

    def run_episode(self, env: SUMOTrafficEnv) -> Dict:
        """
        Run one episode using actuated control.

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
                # Get actuated actions
                elapsed_time = step * SIMULATION_TIME_STEP
                actions = self.get_actions(elapsed_time, observations)

                # Execute step
                observations, rewards, dones, _, infos = env.step(actions)

                # Accumulate rewards
                for agent in env.agents:
                    episode_rewards[agent] += rewards[agent]

                # Collect statistics
                if step % 100 == 0:
                    stats = env.get_episode_stats()
                    episode_wait_times.append(stats.get('avg_wait', 0.0))

                step_count += 1

                if dones['__all__']:
                    break

            env.close()

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
            logger.error(f"Error running actuated control episode: {e}")
            env.close()
            return {
                'episode_steps': step_count,
                'avg_reward': 0.0,
                'avg_wait_time': 0.0,
                'co2_kg': 0.0,
                'error': str(e),
            }


def run_actuated_baseline(num_episodes: int = 10, seed: int = 42) -> List[Dict]:
    """
    Run actuated control baseline for multiple episodes.

    Args:
        num_episodes: Number of episodes to run
        seed: Random seed

    Returns:
        List of episode result dicts
    """
    baseline = ActuatedControlBaseline()
    results = []

    logger.info(f"Starting actuated control baseline ({num_episodes} episodes)...")

    for ep in range(num_episodes):
        env = SUMOTrafficEnv(gui=False)
        result = baseline.run_episode(env)
        results.append(result)

        logger.info(
            f"Episode {ep+1}/{num_episodes}: "
            f"avg_wait={result.get('avg_wait_time', 0):.2f}s, "
            f"co2={result.get('co2_kg', 0):.2f}kg"
        )

    return results
