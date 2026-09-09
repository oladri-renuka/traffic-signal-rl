"""PyTorch neural network policy for traffic signal agents."""

import torch
import torch.nn as nn
import numpy as np

from src.utils.sumo_config import STATE_DIMS, NUM_ACTIONS


class PolicyNetwork(nn.Module):
    """Small MLP policy network for traffic signal control."""

    def __init__(self, input_dim=STATE_DIMS, hidden_dim=128, output_dim=NUM_ACTIONS):
        """
        Initialize policy network.

        Args:
            input_dim: Input state dimension (26)
            hidden_dim: Hidden layer dimension (128)
            output_dim: Output action dimension (4)
        """
        super(PolicyNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # MLP: input -> hidden -> hidden -> output
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()

        # Initialize weights using He initialization
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        Forward pass through policy network.

        Args:
            x: Input state tensor (batch_size, STATE_DIMS) or (STATE_DIMS,)

        Returns:
            Action logits (batch_size, NUM_ACTIONS) or (NUM_ACTIONS,)
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def get_action_probs(self, x):
        """Get action probabilities (softmax of logits)."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def get_action_log_probs(self, x):
        """Get action log probabilities."""
        logits = self.forward(x)
        return torch.log_softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """Small MLP value network for critic (advantage estimation)."""

    def __init__(self, input_dim=STATE_DIMS, hidden_dim=128):
        """
        Initialize value network.

        Args:
            input_dim: Input state dimension (26)
            hidden_dim: Hidden layer dimension (128)
        """
        super(ValueNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

        # Initialize weights using He initialization
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        Forward pass through value network.

        Args:
            x: Input state tensor

        Returns:
            Value estimate (scalar or batch)
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        return value
