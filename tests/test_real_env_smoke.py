import sys
import types
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

# ---------------------------------------------------------------------------
# Stub minimal rlgym modules so PPOTrainer and env can be imported
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

rlgym_utils.action_parsers = action_parsers_mod
rlgym_utils.terminal_conditions = terminal_conditions_mod

sys.modules.setdefault("rlgym.utils", rlgym_utils)
sys.modules.setdefault("rlgym.utils.action_parsers", action_parsers_mod)
sys.modules.setdefault("rlgym.utils.terminal_conditions", terminal_conditions_mod)

import src.training.state_setters as state_setters_mod
state_setters_mod.SSLStateSetter = object

from src.training.env_factory import make_env

api_cfg_mod = types.ModuleType("rlgym.api.config")
class ObsBuilder: ...
class RewardFunction: ...
api_cfg_mod.ObsBuilder = ObsBuilder
api_cfg_mod.RewardFunction = RewardFunction

sys.modules["rlgym"] = rlgym_mod
sys.modules["rlgym.utils"] = utils_mod
sys.modules["rlgym.utils.action_parsers"] = action_parsers_mod
sys.modules["rlgym.utils.terminal_conditions"] = terminal_conditions_mod
sys.modules["rlgym.api"] = types.ModuleType("rlgym.api")
sys.modules["rlgym.api.config"] = api_cfg_mod

common_values_mod = types.ModuleType("rlgym.rocket_league.common_values")
common_values_mod.BOOST_LOCATIONS = np.zeros((1, 3), dtype=np.float32)
common_values_mod.TICKS_PER_SECOND = 60
common_values_mod.BALL_RADIUS = 92.75
common_values_mod.SIDE_WALL_X = 4096
common_values_mod.BACK_WALL_Y = 5120
common_values_mod.CEILING_Z = 2044
common_values_mod.CAR_MAX_SPEED = 2300
common_values_mod.BALL_MAX_SPEED = 6000
common_values_mod.CAR_MAX_ANG_VEL = 5.5
common_values_mod.BLUE_GOAL_BACK = np.zeros(3)
common_values_mod.BLUE_GOAL_CENTER = np.zeros(3)
common_values_mod.ORANGE_GOAL_BACK = np.zeros(3)
common_values_mod.ORANGE_GOAL_CENTER = np.zeros(3)
common_values_mod.GOAL_HEIGHT = 0
common_values_mod.ORANGE_TEAM = 1


class PhysicsObject:
    def __init__(self):
        self.position = np.zeros(3, dtype=np.float32)
        self.linear_velocity = np.zeros(3, dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        self.euler_angles = np.zeros(3, dtype=np.float32)

    @property
    def forward(self):
        return np.array([1, 0, 0], dtype=np.float32)

    @property
    def up(self):
        return np.array([0, 0, 1], dtype=np.float32)

    @property
    def pitch(self):
        return 0.0

    @property
    def yaw(self):
        return 0.0

    @property
    def roll(self):
        return 0.0


class Car:
    def __init__(self):
        self.team_num = 0
        self.ball_touches = 0
        self.boost_amount = 0.0
        self.on_ground = True
        self.has_flip = True
        self.has_jumped = False
        self.is_demoed = False
        self.physics = PhysicsObject()


class GameConfig:
    pass


class GameState:
    def __init__(self):
        self.ball = PhysicsObject()
        self.cars = {}
        self.boost_pad_timers = np.zeros(len(common_values_mod.BOOST_LOCATIONS), dtype=np.float32)
        self.tick_count = 0
        self.goal_scored = False
        self.config = GameConfig()


class RocketSimEngine:
    def __init__(self, rlbot_delay: bool = False):
        self.agents = [0]
        self.state = GameState()

    def create_base_state(self) -> GameState:
        return GameState()

    def set_state(self, state: GameState, info):
        self.state = state

    def step(self, actions, info) -> GameState:
        return self.state


class GoalCondition:
    def reset(self, agents, state, info):
        pass

    def is_done(self, agents, state, info):
        return {agents[0]: False}


class TimeoutCondition:
    def __init__(self, *_):
        pass

    def reset(self, agents, state, info):
        pass

    def is_done(self, agents, state, info):
        return {agents[0]: False}


sys.modules["rlgym.rocket_league.api"] = types.ModuleType("rlgym.rocket_league.api")
sys.modules["rlgym.rocket_league.api"].GameState = GameState
sys.modules["rlgym.rocket_league.api"].Car = Car
sys.modules["rlgym.rocket_league.api"].PhysicsObject = PhysicsObject
sys.modules["rlgym.rocket_league.api"].GameConfig = GameConfig
sys.modules["rlgym.rocket_league.common_values"] = common_values_mod
sys.modules["rlgym.rocket_league.sim.rocketsim_engine"] = types.ModuleType(
    "rlgym.rocket_league.sim.rocketsim_engine"
)
sys.modules["rlgym.rocket_league.sim.rocketsim_engine"].RocketSimEngine = RocketSimEngine
sys.modules["rlgym.rocket_league.done_conditions.goal_condition"] = types.ModuleType(
    "rlgym.rocket_league.done_conditions.goal_condition"
)
sys.modules["rlgym.rocket_league.done_conditions.goal_condition"].GoalCondition = GoalCondition
sys.modules["rlgym.rocket_league.done_conditions.timeout_condition"] = types.ModuleType(
    "rlgym.rocket_league.done_conditions.timeout_condition"
)
sys.modules["rlgym.rocket_league.done_conditions.timeout_condition"].TimeoutCondition = TimeoutCondition


from src.training.env_factory import RL2v2Env
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
        "device": {"auto_detect": False, "device": "cpu", "cuda": False},
        "policy": {"obs_dim": 107, "continuous_actions": 5, "discrete_actions": 3},
        "ppo": {
            "actor_lr": 1e-3,
            "critic_lr": 1e-3,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "n_epochs": 1,
            "steps_per_update": 4,
            "mini_batches": 1,
            "clip_ratio": 0.2,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
        },
        "env": {
            "team_size": 1,
            "tick_skip": 1,
            "use_injector": False,
            "self_play": False,
            "spawn_opponents": False,
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
            "continuous_actions": torch.tanh(logits[:, :5]),
            "discrete_actions": torch.sigmoid(logits[:, 5:]),
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


def setup_trainer(monkeypatch, seed=0):
    monkeypatch.setattr("src.training.train.CurriculumManager", DummyCurriculum)
    monkeypatch.setattr(PPOTrainer, "_load_config", lambda self, path: minimal_config())
    monkeypatch.setattr(
        PPOTrainer,
        '_create_environment',
        lambda self: make_env(seed=self.seed)(),
        "_create_environment",
        lambda self: RL2v2Env(seed=self.seed),
    )
    monkeypatch.setattr(
        PPOTrainer,
        "_convert_actions_to_env",
        lambda self, a: {
            "cont": np.array(
                a["continuous_actions"][0].detach().cpu().tolist(), dtype=np.float32
            ),
            "disc": np.array(
                a["discrete_actions"][0].detach().cpu().tolist(), dtype=np.float32
            ),
        },
    )
    monkeypatch.setattr("src.training.train.create_ssl_policy", lambda cfg: DummyPolicy())
    monkeypatch.setattr("src.training.train.create_ssl_critic", lambda cfg: DummyCritic())
    return PPOTrainer("cfg", "curr", seed=seed)


def test_real_env_rollout_losses(monkeypatch):
    trainer = setup_trainer(monkeypatch, seed=123)
    rollouts = trainer._collect_rollouts(4)
    losses = trainer._update_policy(rollouts)
    assert np.isfinite(losses["policy_loss"])
    assert np.isfinite(losses["value_loss"])


def test_private_rng_determinism(monkeypatch):
    trainer1 = setup_trainer(monkeypatch, seed=999)
    trainer2 = setup_trainer(monkeypatch, seed=999)
    actions1 = trainer1._collect_rollouts(4)["actions"]
    actions2 = trainer2._collect_rollouts(4)["actions"]
    assert torch.equal(actions1["continuous_actions"], actions2["continuous_actions"])
    assert torch.equal(actions1["discrete_actions"], actions2["discrete_actions"])

    sample1 = trainer1.env.action_space.sample()
    sample2 = trainer2.env.action_space.sample()
    assert np.allclose(sample1["cont"], sample2["cont"])
    assert np.array_equal(sample1["disc"], sample2["disc"])

