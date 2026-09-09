"""Traffic signal agent configuration."""

import numpy as np

try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces

from src.utils.sumo_config import STATE_DIMS, NUM_ACTIONS, STATE_MIN, STATE_MAX


class AgentConfig:
    """Configuration for traffic signal control agents."""

    def __init__(self):
        # Observation space: 26-dimensional continuous, normalized to [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(STATE_DIMS,),
            dtype=np.float32
        )

        # Action space: discrete, 4 phases
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # State bounds for normalization
        self.state_min = np.array(STATE_MIN, dtype=np.float32)
        self.state_max = np.array(STATE_MAX, dtype=np.float32)

    def get_observation_space(self):
        """Get observation space."""
        return self.observation_space

    def get_action_space(self):
        """Get action space."""
        return self.action_space
