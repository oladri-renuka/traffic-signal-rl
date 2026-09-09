"""Training configuration constants."""

from src.utils.sumo_config import (
    NUM_AGENTS, STATE_DIMS, NUM_ACTIONS,
    LEARNING_RATE, BATCH_SIZE, GAMMA, ENTROPY_COEFF
)

# Network architecture
POLICY_HIDDEN_DIMS = [128, 128]
VALUE_HIDDEN_DIMS = [128, 128]

# Training hyperparameters
PPO_CLIP_PARAM = 0.2
PPO_ENTROPY_COEFF = ENTROPY_COEFF
PPO_GAMMA = GAMMA
PPO_LR = LEARNING_RATE

# Batch and episode settings
BATCH_SIZE = BATCH_SIZE
EPISODE_LENGTH = 3600  # seconds (1 hour)
