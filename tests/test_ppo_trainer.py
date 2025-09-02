import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import types

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

utils_mod.action_parsers = action_parsers_mod
utils_mod.terminal_conditions = terminal_conditions_mod
rlgym_mod.utils = utils_mod

sys.modules.setdefault("rlgym", rlgym_mod)
sys.modules.setdefault("rlgym.utils", utils_mod)
sys.modules.setdefault("rlgym.utils.action_parsers", action_parsers_mod)
sys.modules.setdefault("rlgym.utils.terminal_conditions", terminal_conditions_mod)

from src.training.train import PPOTrainer


class DummyEnv:
    def __init__(self):
        self.step_count = 0

    def reset(self):
        return np.zeros(107, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        obs = np.zeros(107, dtype=np.float32)
        reward = 0.0
        done = True
        info = {}
        return obs, reward, done, info


class DummyCurriculum:
    def __init__(self, *args, **kwargs):
        pass

    class Phase:
        name = "test"

    def get_current_phase(self):
        return self.Phase()


def minimal_config():
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
    }


def test_collect_one_step(monkeypatch):
    import torch

    class DummyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(107, 8)

        def sample_actions(self, obs):
            logits = self.linear(obs)
            return {
                'continuous_actions': torch.tanh(logits[:, :5]),
                'discrete_actions': torch.sigmoid(logits[:, 5:])
            }

        def log_prob(self, obs, actions):
            return torch.zeros(obs.shape[0])

        def entropy(self, obs):
            return torch.zeros(obs.shape[0])

    class DummyCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(107, 1)

        def forward(self, obs):
            return self.linear(obs)

    # Patch curriculum, environment, and networks
    monkeypatch.setattr('src.training.train.CurriculumManager', DummyCurriculum)
    monkeypatch.setattr(PPOTrainer, '_load_config', lambda self, path: minimal_config())
    monkeypatch.setattr(PPOTrainer, '_create_environment', lambda self: DummyEnv())
    monkeypatch.setattr('src.training.train.create_ssl_policy', lambda cfg: DummyPolicy())
    monkeypatch.setattr('src.training.train.create_ssl_critic', lambda cfg: DummyCritic())

    trainer = PPOTrainer('cfg', 'curr')

    rollouts = trainer._collect_rollouts(1)
    assert rollouts['observations'].shape == (1, 107)
    assert rollouts['actions']['continuous_actions'].shape == (1, 5)
    assert rollouts['actions']['discrete_actions'].shape == (1, 3)

    metrics = trainer._update_policy(rollouts)
    for key in ('policy_loss', 'value_loss', 'entropy_loss'):
        assert key in metrics
        assert np.isfinite(metrics[key])
