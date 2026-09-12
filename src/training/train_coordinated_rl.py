"""Train coordinated multi-agent RL model with shared policy."""

import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from src.utils.logger import get_logger
from src.utils.sumo_config import NUM_AGENTS, STATE_DIMS, NUM_ACTIONS
from src.environment.sumo_env import SUMOTrafficEnv

logger = get_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SharedPolicyNetwork(nn.Module):
    """Shared neural network policy for all agents."""

    def __init__(self, state_dim=STATE_DIMS, action_dim=NUM_ACTIONS, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.network(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value


class CoordinatedRLTrainer:
    """Trainer for coordinated multi-agent RL using shared policy."""

    def __init__(self, num_episodes=1000, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95):
        self.num_episodes = num_episodes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Initialize environment
        self.env = SUMOTrafficEnv(gui=False)

        # Initialize shared policy network
        self.policy = SharedPolicyNetwork().to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Metrics
        self.metrics = defaultdict(list)
        self.start_time = datetime.now()

    def select_actions(self, observations):
        """Select actions for all agents using shared policy."""
        actions = {}
        with torch.no_grad():
            for agent_id, obs in observations.items():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                action_logits, _ = self.policy(obs_tensor)
                action = torch.argmax(action_logits, dim=1).item()
                actions[agent_id] = action
        return actions

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0

        values = list(values) + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return advantages

    def train_episode(self, episode_num):
        """Train for one episode."""
        observations, _ = self.env.reset()

        episode_rewards_per_step = []  # rewards per timestep (average across agents)
        episode_values = []
        episode_dones = []
        episode_obs_list = []

        step = 0
        while True:
            # Get actions for all agents
            actions = self.select_actions(observations)

            # Step environment
            next_observations, rewards, dones, _, infos = self.env.step(actions)

            # Collect trajectory data - use first agent's obs for policy (all get same input distribution)
            first_agent = self.env.agents[0]
            obs = observations[first_agent]
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                _, value = self.policy(obs_tensor)

            episode_obs_list.append(obs)
            episode_values.append(value.item())

            # Average reward across all agents
            avg_reward = np.mean([rewards[agent_id] for agent_id in self.env.agents])
            episode_rewards_per_step.append(avg_reward)
            episode_dones.append(dones.get("__all__", False))

            observations = next_observations
            step += 1

            if dones.get("__all__", False):
                break

        # Compute final value
        with torch.no_grad():
            first_agent = self.env.agents[0]
            final_obs = torch.FloatTensor(observations[first_agent]).unsqueeze(0).to(DEVICE)
            _, final_value = self.policy(final_obs)

        # Compute advantages
        advantages = self.compute_gae(episode_rewards_per_step, episode_values, episode_dones, final_value.item())

        # Update policy (simplified PPO update)
        if len(episode_obs_list) > 0:
            batch_obs = torch.FloatTensor(np.array(episode_obs_list)).to(DEVICE)
            batch_advantages = torch.FloatTensor(np.array(advantages)).to(DEVICE)
            batch_returns = batch_advantages + torch.FloatTensor(np.array(episode_values)).to(DEVICE)

            # Normalize advantages
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

            # Compute loss
            action_logits, values = self.policy(batch_obs)
            value_loss = nn.MSELoss()(values.squeeze(), batch_returns)

            entropy = -(action_logits.softmax(dim=1) * action_logits.log_softmax(dim=1)).sum(dim=1).mean()

            loss = value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Log metrics
        avg_episode_reward = np.mean(episode_rewards_per_step)
        stats = self.env.get_episode_stats()

        self.metrics['episode'].append(episode_num)
        self.metrics['avg_reward'].append(avg_episode_reward)
        self.metrics['avg_wait'].append(stats.get('avg_wait', 0.0))
        self.metrics['co2_kg'].append(stats.get('co2_kg', 0.0))

        if (episode_num + 1) % 50 == 0:
            logger.info(
                f"Episode {episode_num + 1}/{self.num_episodes}: "
                f"avg_reward={avg_episode_reward:.4f}, "
                f"avg_wait={stats.get('avg_wait', 0.0):.2f}s, "
                f"co2={stats.get('co2_kg', 0.0):.2f}kg"
            )

        return avg_episode_reward

    def train(self):
        """Train for all episodes."""
        logger.info(f"Starting coordinated RL training for {self.num_episodes} episodes on {DEVICE}")

        for episode in range(self.num_episodes):
            self.train_episode(episode)

        self.env.close()
        logger.info("Training complete!")
        return self.metrics

    def save_results(self, output_dir="experiments/results"):
        """Save trained model and metrics."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = os.path.join(output_dir, "coordinated_rl_model.pt")
        torch.save(self.policy.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

        # Save metrics
        metrics_path = os.path.join(output_dir, "rl_training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

        # Save summary
        summary = {
            "total_episodes": self.num_episodes,
            "total_time": str(datetime.now() - self.start_time),
            "final_avg_reward": float(np.mean(self.metrics['avg_reward'][-100:])) if len(self.metrics['avg_reward']) > 0 else 0.0,
            "best_avg_reward": float(np.max(self.metrics['avg_reward'])) if len(self.metrics['avg_reward']) > 0 else 0.0,
            "device": str(DEVICE),
        }

        summary_path = os.path.join(output_dir, "rl_training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")

        return model_path, metrics_path


if __name__ == "__main__":
    trainer = CoordinatedRLTrainer(num_episodes=1000)
    metrics = trainer.train()
    trainer.save_results()
