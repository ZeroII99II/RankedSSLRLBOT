import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from src.training.policy import create_ssl_policy, create_ssl_critic


def test_policy_and_critic_output_shapes():
    config = {"hidden_sizes": [107], "use_attention": False}
    policy = create_ssl_policy(config)
    critic = create_ssl_critic(config)

    obs = torch.zeros(2, 107)
    policy_out = policy(obs)
    assert policy_out["continuous_actions"].shape == (2, 5)
    assert policy_out["discrete_actions"].shape == (2, 3)

    value = critic(obs)
    assert value.shape == (2, 1)
