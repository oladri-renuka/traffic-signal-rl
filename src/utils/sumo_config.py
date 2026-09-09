"""Configuration constants for SUMO integration."""

import os
from pathlib import Path

# SUMO paths
SUMO_HOME = os.getenv('SUMO_HOME', '/usr/share/sumo')
SUMO_BINARY = os.path.join(SUMO_HOME, 'bin', 'sumo')
SUMO_GUI = os.path.join(SUMO_HOME, 'bin', 'sumo-gui')

# Network configuration
NETWORK_DIR = Path(__file__).parent.parent.parent / 'sumo' / 'network'
NETWORK_FILE = NETWORK_DIR / 'grid_4x4.net.xml'
ROUTE_FILE = NETWORK_DIR / 'grid_4x4.rou.xml'
SUMO_CONFIG_FILE = NETWORK_DIR / 'grid_4x4.sumocfg'

# Grid dimensions
GRID_SIZE = 4
NUM_AGENTS = GRID_SIZE * GRID_SIZE  # 16 agents

# Simulation parameters
SIMULATION_TIME_STEP = 0.1  # seconds
EPISODE_LENGTH_SECONDS = 3600  # 1 hour
EPISODE_LENGTH_STEPS = int(EPISODE_LENGTH_SECONDS / SIMULATION_TIME_STEP)

# State space
STATE_DIMS = 26  # [queue_0-3, wait_0-3, phase, elapsed, neighbor_queues_0-15]
STATE_MIN = [0.0] * STATE_DIMS
STATE_MAX = [1.0] * STATE_DIMS

# Action space
NUM_ACTIONS = 4  # 4 phase transitions per intersection

# EPA emissions factor
EPA_CO2_PER_MINUTE = 2.4  # grams per minute of idling
EPA_CO2_PER_MINUTE_KG = EPA_CO2_PER_MINUTE / 1000.0  # kg per minute

# Reward parameters
LOCAL_REWARD_WEIGHT = 0.5
GLOBAL_REWARD_WEIGHT = 0.5
LONG_RED_PENALTY = 0.1

# Training parameters
FIXED_SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
GAMMA = 0.99
ENTROPY_COEFF = 0.01

# Evaluation parameters
EVAL_INTERVAL = 100
NUM_EVAL_EPISODES = 10
NUM_BASELINE_RUNS = 3

# Random seed for traffic generation
TRAFFIC_SEED = 42
