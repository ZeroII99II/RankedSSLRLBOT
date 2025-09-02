import sys
from pathlib import Path

import numpy as np
import torch
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Stub out rlgym modules used by trainer
rlgym_mod = types.ModuleType("rlgym")
rlgym_mod.make = lambda *args, **kwargs: None

utils_mod = types.ModuleType("rlgym.utils")
action_parsers_mod = types.ModuleType("rlgym.utils.action_parsers")

class DefaultAction:
    def __init__(self, *args, **kwargs):
        pass

action_parsers_mod.DefaultAction = DefaultAction

terminal_conditions_mod = types.ModuleType("rlgym.utils.terminal_conditions")
common_conditions = types.SimpleNamespace(
    TimeoutCondition=lambda *args, **kwargs: None,
    GoalScoredCondition=lambda *args, **kwargs: None,
)
terminal_conditions_mod.common_conditions = common_conditions

api_mod = types.ModuleType("rlgym.api")
config_mod = types.ModuleType("rlgym.api.config")

class ObsBuilder:
    pass

class RewardFunction:
    pass

config_mod.ObsBuilder = ObsBuilder
config_mod.RewardFunction = RewardFunction
api_mod.config = config_mod

rocket_mod = types.ModuleType("rlgym.rocket_league")
common_values_mod = types.ModuleType("rlgym.rocket_league.common_values")
common_values_mod.CAR_MAX_BOOST = 100
common_values_mod.BOOST_LOCATIONS = []
common_values_mod.CEILING_Z = 2044
common_values_mod.BALL_RADIUS = 92.75
common_values_mod.CAR_MAX_SPEED = 2300
common_values_mod.__getattr__ = lambda name: 0
common_values_mod.__file__ = "stub"
rocket_mod.common_values = common_values_mod
api_rl_mod = types.ModuleType("rlgym.rocket_league.api")

class GameState:
    pass

api_rl_mod.GameState = GameState
rocket_mod.api = api_rl_mod

utils_mod.action_parsers = action_parsers_mod
utils_mod.terminal_conditions = terminal_conditions_mod
rlgym_mod.utils = utils_mod
rlgym_mod.api = api_mod
rlgym_mod.rocket_league = rocket_mod

sys.modules.setdefault("rlgym", rlgym_mod)
sys.modules.setdefault("rlgym.utils", utils_mod)
sys.modules.setdefault("rlgym.utils.action_parsers", action_parsers_mod)
sys.modules.setdefault("rlgym.utils.terminal_conditions", terminal_conditions_mod)
sys.modules.setdefault("rlgym.api", api_mod)
sys.modules.setdefault("rlgym.api.config", config_mod)
sys.modules.setdefault("rlgym.rocket_league", rocket_mod)
sys.modules.setdefault("rlgym.rocket_league.common_values", common_values_mod)
sys.modules.setdefault("rlgym.rocket_league.api", api_rl_mod)

# Stub gym/gymnasium modules
gym_mod = types.ModuleType("gym")
class Space:
    pass
class Box:
    def __init__(self, *args, **kwargs):
        pass
spaces_mod = types.ModuleType("gym.spaces")
spaces_mod.Box = Box
gym_mod.Space = Space
gym_mod.spaces = spaces_mod
sys.modules.setdefault("gym", gym_mod)
sys.modules.setdefault("gymnasium", gym_mod)
sys.modules.setdefault("gym.spaces", spaces_mod)

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
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def sample_actions(self, obs, generator=None):
        return {
            'continuous_actions': torch.randn(obs.shape[0], 5, generator=generator),
            'discrete_actions': torch.randn(obs.shape[0], 3, generator=generator),
        }

    def log_prob(self, obs, actions):
        return torch.zeros(obs.shape[0])


class DummyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

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
    actions_a = [
        trainer_a.policy.sample_actions(obs, generator=trainer_a.torch_rng)
        for _ in range(5)
    ]
    actions_b = [
        trainer_b.policy.sample_actions(obs, generator=trainer_b.torch_rng)
        for _ in range(5)
    ]

    for a, b in zip(actions_a, actions_b):
        assert np.allclose(a['continuous_actions'].numpy(), b['continuous_actions'].numpy())
        assert np.allclose(a['discrete_actions'].numpy(), b['discrete_actions'].numpy())

