import sys, types, numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import src
import src.training  # ensure training package is loaded before stubbing

import src  # ensure package exists before stubbing submodules

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

config_mod.ObsBuilder = ObsBuilder
class RewardFunction:
    pass

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

# Stub project modules expected by PPOTrainer
state_setters_mod = types.ModuleType("src.training.state_setters")


class SSLStateSetter:
    pass


state_setters_mod.SSLStateSetter = SSLStateSetter
sys.modules.setdefault("src.training.state_setters", state_setters_mod)

# Constants for action dimensions
CONT_DIM = 5
DISC_DIM = 3

# Stub gymnasium to satisfy gym_compat when unavailable
try:
    import gymnasium  # type: ignore
except Exception:
    gym_mod = types.ModuleType("gymnasium")
    gym_mod.Space = object
    gym_mod.spaces = types.SimpleNamespace(Box=object)
    sys.modules.setdefault("gymnasium", gym_mod)

from src.training.train import PPOTrainer

# Stub tensorboard to avoid heavy dependency
tensorboard_mod = types.ModuleType("torch.utils.tensorboard")

class SummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass

tensorboard_mod.SummaryWriter = SummaryWriter
sys.modules.setdefault("torch.utils.tensorboard", tensorboard_mod)

# Stub gym modules required by gym_compat
gym_mod = types.ModuleType("gym")
class Space:
    pass

class Box(Space):
    def __init__(self, low, high, shape, dtype=None):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

gym_mod.Space = Space
gym_mod.spaces = types.SimpleNamespace(Box=Box)
sys.modules.setdefault("gym", gym_mod)
sys.modules.setdefault("gymnasium", gym_mod)

from src.training.env_factory import CONT_DIM, DISC_DIM

import src.training.state_setters as state_setters_mod
state_setters_mod.SSLStateSetter = object


class DummyEnv:
    def __init__(self):
        self.step_count = 0
        self.action_space = types.SimpleNamespace(seed=lambda *args, **kwargs: None)

        self.action_space = type('AS', (), {'seed': lambda self, val: None})()

    def reset(self):
        return np.zeros(107, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        obs = np.zeros(107, dtype=np.float32)
        reward = 0.0
        done = True
        info = {}
        return obs, reward, done, info


class NonTerminatingEnv:
    def __init__(self):
        self.step_count = 0
        self.action_space = types.SimpleNamespace(seed=lambda *args, **kwargs: None)

        self.action_space = type('AS', (), {'seed': lambda self, val: None})()

    def reset(self):
        return np.zeros(107, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        obs = np.zeros(107, dtype=np.float32)
        reward = 0.0
        done = False
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
            'steps_per_update': 2,
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

        def sample_actions(self, obs, generator=None):
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
    rollouts = trainer._collect_rollouts(2)
    steps_per_update = trainer.config['ppo']['steps_per_update']
    assert rollouts['observations'].shape == (steps_per_update, 107)
    assert rollouts['actions']['continuous_actions'].shape == (
        steps_per_update,
        CONT_DIM,
    )
    assert rollouts['actions']['discrete_actions'].shape == (
        steps_per_update,
        DISC_DIM,
    )

    # _compute_advantages expects float done flags
    rollouts['dones'] = rollouts['dones'].float()

    # Update policy and verify returned loss metrics are finite
    losses = trainer._update_policy(rollouts)
    for key in ('policy_loss', 'value_loss', 'entropy_loss'):
        assert key in losses
        assert np.isfinite(losses[key])


def test_collect_rollout_without_done(monkeypatch):
    import torch

    class DummyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(107, 8)

        def sample_actions(self, obs, generator=None):
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
    monkeypatch.setattr(PPOTrainer, '_create_environment', lambda self: NonTerminatingEnv())
    monkeypatch.setattr('src.training.train.create_ssl_policy', lambda cfg: DummyPolicy())
    monkeypatch.setattr('src.training.train.create_ssl_critic', lambda cfg: DummyCritic())

    trainer = PPOTrainer('cfg', 'curr')
    rollouts = trainer._collect_rollouts(2)

    assert rollouts['episode_rewards'] == [0.0]
    assert rollouts['episode_lengths'] == [2]


def test_single_step_collection(monkeypatch):
    import torch

    class DummyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(107, 8)

        def sample_actions(self, obs, generator=None):
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

    monkeypatch.setattr('src.training.train.CurriculumManager', DummyCurriculum)
    monkeypatch.setattr(PPOTrainer, '_load_config', lambda self, path: minimal_config())
    monkeypatch.setattr(PPOTrainer, '_create_environment', lambda self: DummyEnv())
    monkeypatch.setattr('src.training.train.create_ssl_policy', lambda cfg: DummyPolicy())
    monkeypatch.setattr('src.training.train.create_ssl_critic', lambda cfg: DummyCritic())

    trainer = PPOTrainer('cfg', 'curr')
    rollouts = trainer._collect_rollouts(1)

    assert rollouts['observations'].shape == (1, 107)
    assert rollouts['actions']['continuous_actions'].shape == (1, CONT_DIM)
    assert rollouts['actions']['discrete_actions'].shape == (1, DISC_DIM)
