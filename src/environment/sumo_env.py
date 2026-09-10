"""PettingZoo multi-agent environment wrapper for SUMO traffic simulation."""

import numpy as np
from typing import Dict, Any

try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces

try:
    from pettingzoo import ParallelEnv
except ImportError:
    ParallelEnv = object

from src.utils.logger import get_logger
from src.utils.sumo_config import (
    NUM_AGENTS, GRID_SIZE, EPISODE_LENGTH_STEPS,
    STATE_DIMS, NUM_ACTIONS,
    LOCAL_REWARD_WEIGHT, GLOBAL_REWARD_WEIGHT, LONG_RED_PENALTY
)
from src.environment.sumo_utils import TraCIManager

logger = get_logger(__name__)


class SUMOTrafficEnv(ParallelEnv):
    """
    PettingZoo parallel environment for multi-agent traffic signal control.

    Agents: 16 traffic light controllers (one per intersection in 4x4 grid)
    State: 26-dimensional continuous vector per agent
    Action: discrete phase selection (0-3)
    Reward: local + global waiting time minimization + pedestrian safety
    """

    metadata = {'render_modes': []}

    def __init__(self, gui=False, render_mode=None):
        """
        Initialize environment.

        Args:
            gui: If True, use SUMO GUI
            render_mode: Currently unused, for compatibility
        """
        self.gui = gui
        self.render_mode = render_mode

        # Initialize TraCI manager
        self.traci_manager = TraCIManager()

        # Agent IDs
        self.agents = [f'agent_{i}' for i in range(NUM_AGENTS)]
        self.possible_agents = self.agents.copy()

        # Spaces
        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(STATE_DIMS,), dtype=np.float32)
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(NUM_ACTIONS)
            for agent in self.agents
        }

        # Episode state
        self.step_count = 0
        self.episode_count = 0

        # Previous state for reward calculation
        self.last_wait_times = {agent: 0.0 for agent in self.agents}

    def reset(self, seed=None):
        """
        Reset environment for new episode without restarting SUMO.

        Args:
            seed: Random seed (currently unused, handled by SUMO)

        Returns:
            Observations dict and info dict
        """
        try:
            # Connect on first reset only
            if not self.traci_manager.is_connected():
                self.traci_manager.connect(gui=self.gui, verbose=False)

            # Clear all vehicles and reset state (but keep SUMO running)
            self.traci_manager.clear_vehicles()
            self.traci_manager.reset_idle_accumulator()

            # Reset episode counters
            self.step_count = 0
            self.episode_count += 1
            self.last_wait_times = {agent: 0.0 for agent in self.agents}

            # Get initial observations
            observations = self._get_observations()

            logger.info(f"Episode {self.episode_count} reset. Initial observations retrieved.")

            return observations, {}

        except Exception as e:
            logger.error(f"Error during reset: {e}")
            raise

    def step(self, actions: Dict[str, int]):
        """
        Execute one environment step.

        Args:
            actions: Dict mapping agent_id -> action (phase 0-3)

        Returns:
            observations, rewards, dones, truncateds, infos
        """
        if not self.traci_manager.is_connected():
            raise RuntimeError("Environment not initialized. Call reset() first.")

        try:
            # Add vehicles continuously (keeps simulation alive)
            self.traci_manager.add_vehicles_continuously(self.step_count)

            # Set phases for all agents
            for agent_id, action in actions.items():
                agent_idx = int(agent_id.split('_')[1])
                phase = int(action) % NUM_ACTIONS
                self.traci_manager.set_agent_phase(agent_idx, phase)

            # Run SUMO simulation step
            self.traci_manager.simulate_step()

            # Get new observations
            observations = self._get_observations()

            # Calculate rewards
            rewards = self._calculate_rewards(observations)

            # Check if episode done
            done = self.step_count >= EPISODE_LENGTH_STEPS - 1
            dones = {agent: done for agent in self.agents}
            dones['__all__'] = done

            # Truncated (not used in this environment)
            truncateds = {agent: False for agent in self.agents}
            truncateds['__all__'] = False

            # Info
            infos = {agent: {} for agent in self.agents}

            self.step_count += 1

            return observations, rewards, dones, truncateds, infos

        except Exception as e:
            logger.error(f"Error during step: {e}")
            raise

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents."""
        try:
            observations = {}
            for i, agent in enumerate(self.agents):
                obs = self.traci_manager.get_agent_state(i)
                observations[agent] = obs
            return observations
        except Exception as e:
            logger.error(f"Error getting observations: {e}")
            return {agent: np.zeros(STATE_DIMS, dtype=np.float32) for agent in self.agents}

    def _calculate_rewards(self, observations: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate rewards for all agents.

        Reward = LOCAL_REWARD_WEIGHT * local_reward + GLOBAL_REWARD_WEIGHT * global_reward
        local_reward = -avg_wait_time_at_intersection
        global_reward = -avg_wait_time_network_wide
        """
        try:
            rewards = {}
            stats = self.traci_manager.get_network_stats()

            global_avg_wait = stats.get('avg_wait', 0.0)
            global_reward = -global_avg_wait

            for i, agent in enumerate(self.agents):
                # Local reward: waiting time at this intersection (estimated from observation)
                # State indices 4-7 are normalized waiting times
                local_wait = observations[agent][4:8].mean() * 120.0  # denormalize
                local_reward = -local_wait

                # Combined reward
                agent_reward = (
                    LOCAL_REWARD_WEIGHT * local_reward +
                    GLOBAL_REWARD_WEIGHT * global_reward
                )

                # Pedestrian safety penalty: penalize long red phases
                # Phase index is at state[8]
                phase = int(observations[agent][8] * 4)
                elapsed = observations[agent][9] * 120.0
                if elapsed > 60:  # Red phase > 60 seconds
                    agent_reward -= LONG_RED_PENALTY

                rewards[agent] = float(agent_reward)

            return rewards

        except Exception as e:
            logger.error(f"Error calculating rewards: {e}")
            return {agent: 0.0 for agent in self.agents}

    def close(self):
        """Close environment and disconnect from SUMO."""
        self.traci_manager.disconnect()

    def observation_space(self, agent: str):
        """Get observation space for agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Get action space for agent."""
        return self.action_spaces[agent]

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics for current episode."""
        return {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            **self.traci_manager.get_network_stats()
        }
