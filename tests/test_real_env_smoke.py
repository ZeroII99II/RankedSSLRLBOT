import sys, types
from pathlib import Path

import numpy as np
import torch

# Ensure repo root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Stub minimal rlgym modules so PPOTrainer can be imported without dependency
# ---------------------------------------------------------------------------
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

api_mod = types.ModuleType("rlgym.api")
config_mod = types.ModuleType("rlgym.api.config")
class ObsBuilder:
    pass
config_mod.ObsBuilder = ObsBuilder
class RewardFunction:
    pass
config_mod.RewardFunction = RewardFunction
api_mod.config = config_mod
sys.modules.setdefault("rlgym.api", api_mod)
sys.modules.setdefault("rlgym.api.config", config_mod)


rocket_mod = types.ModuleType("rlgym.rocket_league")
common_values_mod = types.ModuleType("rlgym.rocket_league.common_values")
common_values_mod.CAR_MAX_SPEED = 2300
common_values_mod.BALL_MAX_SPEED = 6000
common_values_mod.CEILING_Z = 2044
common_values_mod.BALL_RADIUS = 92.75
common_values_mod.SIDE_WALL_X = 4096
common_values_mod.BACK_WALL_Y = 5120
common_values_mod.CAR_MAX_ANG_VEL = 5.5
common_values_mod.BOOST_LOCATIONS = np.zeros((1, 3))
common_values_mod.BLUE_GOAL_BACK = np.zeros(3)
common_values_mod.BLUE_GOAL_CENTER = np.zeros(3)
common_values_mod.ORANGE_GOAL_BACK = np.zeros(3)
common_values_mod.ORANGE_GOAL_CENTER = np.zeros(3)
common_values_mod.GOAL_HEIGHT = 0
common_values_mod.ORANGE_TEAM = 1
rocket_mod.common_values = common_values_mod
sys.modules.setdefault("rlgym.rocket_league", rocket_mod)
sys.modules.setdefault("rlgym.rocket_league.common_values", common_values_mod)
api_rl_mod = types.ModuleType("rlgym.rocket_league.api")
class GameState:
    pass
api_rl_mod.GameState = GameState
sys.modules.setdefault("rlgym.rocket_league.api", api_rl_mod)
# ---------------------------------------------------------------------------
# Stub project modules expected by PPOTrainer
env_factory_mod = types.ModuleType("src.training.env_factory")


CONT_DIM = 5
DISC_DIM = 3


class ActionSpace:
    def __init__(self):
        self.rng = np.random.default_rng()

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def sample(self):
        return {
            'cont': self.rng.uniform(-1, 1, CONT_DIM).astype(np.float32),
            'disc': self.rng.integers(0, 2, DISC_DIM, dtype=np.int8),
        }


class RLMatchEnv:
    def __init__(self, seed=0, num_players_per_team=1):
        self.action_space = ActionSpace()
        self.seed = seed
        self.num_players_per_team = num_players_per_team

    def reset(self):
        return np.zeros(107, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(107, dtype=np.float32), 0.0, True, {}


env_factory_mod.RLMatchEnv = RLMatchEnv
env_factory_mod.CONT_DIM = CONT_DIM
env_factory_mod.DISC_DIM = DISC_DIM
sys.modules.setdefault("src.training.env_factory", env_factory_mod)

state_setters_mod = types.ModuleType("src.training.state_setters")


class SSLStateSetter:
    pass


state_setters_mod.SSLStateSetter = SSLStateSetter
sys.modules.setdefault("src.training.state_setters", state_setters_mod)

from src.training.env_factory import RLMatchEnv, CONT_DIM, DISC_DIM
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
        lambda self: RLMatchEnv(seed=self.seed, num_players_per_team=1),
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
    steps_per_update = trainer.config['ppo']['steps_per_update']
    assert rollouts['actions']['continuous_actions'].shape == (
        steps_per_update,
        CONT_DIM,
    )
    assert rollouts['actions']['discrete_actions'].shape == (
        steps_per_update,
        DISC_DIM,
    )
    losses = trainer._update_policy(rollouts)
    assert np.isfinite(losses['policy_loss'])
    assert np.isfinite(losses['value_loss'])


def test_private_rng_determinism(monkeypatch):
    trainer1 = setup_trainer(monkeypatch, seed=999)
    trainer2 = setup_trainer(monkeypatch, seed=999)
    actions1 = trainer1._collect_rollouts(4)['actions']
    actions2 = trainer2._collect_rollouts(4)['actions']
    steps_per_update = trainer1.config['ppo']['steps_per_update']
    assert actions1['continuous_actions'].shape == (
        steps_per_update,
        CONT_DIM,
    )
    assert actions1['discrete_actions'].shape == (
        steps_per_update,
        DISC_DIM,
    )
    assert torch.equal(actions1['continuous_actions'], actions2['continuous_actions'])
    assert torch.equal(actions1['discrete_actions'], actions2['discrete_actions'])

    sample1 = trainer1.env.action_space.sample()
    sample2 = trainer2.env.action_space.sample()
    assert np.allclose(sample1['cont'], sample2['cont'])
    assert np.array_equal(sample1['disc'], sample2['disc'])
