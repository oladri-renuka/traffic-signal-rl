"""RLlib configuration for multi-agent PPO training."""

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import Policy

from src.agents.policy_network import PolicyNetwork
from src.utils.sumo_config import (
    NUM_AGENTS, STATE_DIMS, NUM_ACTIONS,
    LEARNING_RATE, BATCH_SIZE, GAMMA, ENTROPY_COEFF
)


def get_ppo_config(env_creator, num_workers: int = 2):
    """
    Get RLlib PPO configuration for multi-agent traffic signal control.

    Args:
        env_creator: Environment creation function
        num_workers: Number of rollout workers

    Returns:
        PPOConfig instance
    """
    config = (
        PPOConfig()
        .environment(env_creator=env_creator, env_config={})
        .framework("torch")
        .rollouts(num_rollout_workers=num_workers)
        .training(
            # PPO hyperparameters
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            entropy_coeff=ENTROPY_COEFF,
            clip_param=0.2,
            sgd_minibatch_size=64,
            train_batch_size=BATCH_SIZE,
            num_sgd_iter=20,
            lr_schedule=None,
            # Value function
            use_critic=True,
            use_gae=True,
            lambda_=0.95,
            vf_clip_param=10.0,
            entropy_coeff_schedule=None,
            grad_clip=None,
        )
        .resources(num_gpus=0)  # Set to 1 if GPU available
        .api_stack(enable_rl_module_and_learner=False)
        .multi_agent(
            policies={
                'traffic_light': (
                    None,  # Policy class (auto-generated)
                    None,  # Observation space (from env)
                    None,  # Action space (from env)
                    {}     # Config dict
                )
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: 'traffic_light',
        )
        .debugging(log_level="INFO")
    )

    return config


def get_model_config():
    """
    Get model configuration for neural network policy.

    Returns:
        Dict with model config
    """
    return {
        'fcnet_hiddens': [128, 128],
        'fcnet_activation': 'relu',
        'use_lstm': False,
        'max_seq_len': 20,
        'framestack': 1,
        'vf_share_layers': False,
    }
