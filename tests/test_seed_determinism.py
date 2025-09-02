import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training.train import PPOTrainer


class DummyEnv:
    def __init__(self):
        self.action_space = type("as", (), {"sample": lambda self: 0})()

    def reset(self):
        return np.zeros(107, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(107, dtype=np.float32)
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info


def minimal_config(seed: int):
    return {
        'device': {'auto_detect': False, 'device': 'cpu', 'cuda': False},
        'policy': {'obs_dim': 107, 'continuous_actions': 5, 'discrete_actions': 3},
        'ppo': {
            'actor_lr': 1e-3,
            'critic_lr': 1e-3,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'n_epochs': 1,
            'steps_per_update': 1,
            'mini_batches': 1,
            'clip_ratio': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
        },
        'env': {
            'team_size': 1,
            'tick_skip': 1,
            'use_injector': False,
            'self_play': False,
            'spawn_opponents': False,
        },
        'training': {'seed': seed},
    }


class DummyPolicy(torch.nn.Module):
    def sample_actions(self, obs):
        return {
            'continuous_actions': torch.randn(obs.shape[0], 5),
            'discrete_actions': torch.randn(obs.shape[0], 3),
        }

    def log_prob(self, obs, actions):
        return torch.zeros(obs.shape[0])


class DummyCritic(torch.nn.Module):
    def forward(self, obs):
        return torch.zeros(obs.shape[0], 1)


def test_seed_reproducibility(monkeypatch):
    seed = 123

    # Patch trainer dependencies
    monkeypatch.setattr(PPOTrainer, '_load_config', lambda self, path: minimal_config(seed))
    monkeypatch.setattr(PPOTrainer, '_create_environment', lambda self: DummyEnv())
    monkeypatch.setattr('src.training.train.create_ssl_policy', lambda cfg: DummyPolicy())
    monkeypatch.setattr('src.training.train.create_ssl_critic', lambda cfg: DummyCritic())

    trainer_a = PPOTrainer('cfg', 'curr', seed=seed)
    trainer_b = PPOTrainer('cfg', 'curr', seed=seed)

    obs = torch.zeros(1, 107)
    actions_a = [trainer_a.policy.sample_actions(obs) for _ in range(5)]
    actions_b = [trainer_b.policy.sample_actions(obs) for _ in range(5)]

    for a, b in zip(actions_a, actions_b):
        assert np.allclose(a['continuous_actions'].numpy(), b['continuous_actions'].numpy())
        assert np.allclose(a['discrete_actions'].numpy(), b['discrete_actions'].numpy())

