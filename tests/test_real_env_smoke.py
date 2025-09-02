import sys, types
from pathlib import Path

import sys
from pathlib import Path
import types
import numpy as np
import torch

# Ensure repo root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Minimal stubs for rlgym utils required by the trainer
rlgym_utils = types.ModuleType("rlgym.utils")
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

rlgym_utils.action_parsers = action_parsers_mod
rlgym_utils.terminal_conditions = terminal_conditions_mod

sys.modules.setdefault("rlgym.utils", rlgym_utils)
sys.modules.setdefault("rlgym.utils.action_parsers", action_parsers_mod)
sys.modules.setdefault("rlgym.utils.terminal_conditions", terminal_conditions_mod)

import src.training.state_setters as state_setters_mod
state_setters_mod.SSLStateSetter = object

from src.training.env_factory import make_env
from src.training.train import PPOTrainer


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
            'steps_per_update': 4,
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


class DummyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(107, 8)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def sample_actions(self, obs, generator=None):
        logits = self.linear(obs)
        noise = torch.randn(logits.shape, generator=generator, device=logits.device)
        logits = logits + noise
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
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, obs):
        return self.linear(obs)


# Helper to configure trainer with RLMatchEnv and dummy networks

def setup_trainer(monkeypatch, seed=0):
    monkeypatch.setattr('src.training.train.CurriculumManager', DummyCurriculum)
    monkeypatch.setattr(PPOTrainer, '_load_config', lambda self, path: minimal_config())
    monkeypatch.setattr(
        PPOTrainer,
        '_create_environment',
        lambda self: make_env(seed=self.seed)(),
    )
    monkeypatch.setattr(PPOTrainer, '_convert_actions_to_env',
                        lambda self, a: {
                            'cont': a['continuous_actions'][0].cpu().numpy(),
                            'disc': a['discrete_actions'][0].cpu().numpy(),
                        })
    monkeypatch.setattr('src.training.train.create_ssl_policy', lambda cfg: DummyPolicy())
    monkeypatch.setattr('src.training.train.create_ssl_critic', lambda cfg: DummyCritic())
    return PPOTrainer('cfg', 'curr', seed=seed)


def test_real_env_rollout_losses(monkeypatch):
    trainer = setup_trainer(monkeypatch, seed=123)
    rollouts = trainer._collect_rollouts(4)
    losses = trainer._update_policy(rollouts)
    assert np.isfinite(losses['policy_loss'])
    assert np.isfinite(losses['value_loss'])


def test_private_rng_determinism(monkeypatch):
    trainer1 = setup_trainer(monkeypatch, seed=999)
    trainer2 = setup_trainer(monkeypatch, seed=999)
    actions1 = trainer1._collect_rollouts(4)['actions']
    actions2 = trainer2._collect_rollouts(4)['actions']
    assert torch.equal(actions1['continuous_actions'], actions2['continuous_actions'])
    assert torch.equal(actions1['discrete_actions'], actions2['discrete_actions'])

    sample1 = trainer1.env.action_space.sample()
    sample2 = trainer2.env.action_space.sample()
    assert np.allclose(sample1['cont'], sample2['cont'])
    assert np.array_equal(sample1['disc'], sample2['disc'])