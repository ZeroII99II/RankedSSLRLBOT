import sys
import types
from pathlib import Path
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stub minimal rlgym modules expected by env_factory

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training.env_factory import make_env, CONT_DIM, DISC_DIM
from rlgym.api.config import ObsBuilder, RewardFunction


def test_reset_returns_obs_vec():
    env = make_env()()


# ---------------------------------------------------------------------------
# Stub minimal rlgym modules required by env_factory
# ---------------------------------------------------------------------------
rlgym_mod = types.ModuleType("rlgym")



api_cfg_mod = types.ModuleType("rlgym.api.config")
class ObsBuilder: ...


class RewardFunction: ...
api_cfg_mod.ObsBuilder = ObsBuilder
api_cfg_mod.RewardFunction = RewardFunction
sys.modules["rlgym.api"] = types.ModuleType("rlgym.api")
sys.modules["rlgym.api.config"] = api_cfg_mod


config_mod.ObsBuilder = ObsBuilder
config_mod.RewardFunction = RewardFunction
api_mod.config = config_mod
rlgym_mod.api = api_mod

rocket_mod = types.ModuleType("rlgym.rocket_league")
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
common_values_mod.TICKS_PER_SECOND = 60
rocket_mod.common_values = common_values_mod



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


# API types -----------------------------------------------------------------
api_rl_mod = types.ModuleType("rlgym.rocket_league.api")


class GameState: ...


class Car: ...


class PhysicsObject: ...


class GameConfig: ...


api_rl_mod.GameState = GameState
api_rl_mod.Car = Car
api_rl_mod.PhysicsObject = PhysicsObject
api_rl_mod.GameConfig = GameConfig

# Engine --------------------------------------------------------------------
sim_mod = types.ModuleType("rlgym.rocket_league.sim.rocketsim_engine")


class RocketSimEngine:
    def __init__(self, rlbot_delay: bool = False):
        self.agents = [0]
        self.state = None

    def create_base_state(self):
        gs = GameState()
        gs.ball = PhysicsObject()
        gs.cars = {}
        gs.boost_pad_timers = np.zeros(
            len(common_values_mod.BOOST_LOCATIONS), dtype=np.float32
        )
        gs.config = GameConfig()
        gs.tick_count = 0
        gs.goal_scored = False
        return gs

    def set_state(self, state, info):
        self.state = state

    def step(self, actions, info):
        self.state.tick_count += 1
        return self.state


sim_mod.RocketSimEngine = RocketSimEngine

# Done conditions ------------------------------------------------------------
goal_mod = types.ModuleType("rlgym.rocket_league.done_conditions.goal_condition")


class GoalCondition:
    def reset(self, agents, state, info):
        pass

    def is_done(self, agents, state, info):
        return {a: False for a in agents}


goal_mod.GoalCondition = GoalCondition

timeout_mod = types.ModuleType(
    "rlgym.rocket_league.done_conditions.timeout_condition"
)


class TimeoutCondition:
    def __init__(self, timeout: int = 500):
        self.timeout = timeout
        self.count = 0

    def reset(self, agents, state, info):
        self.count = 0

    def is_done(self, agents, state, info):
        self.count += 1
        done = self.count >= self.timeout
        return {a: done for a in agents}


timeout_mod.TimeoutCondition = TimeoutCondition

# Register stub modules ------------------------------------------------------
sys.modules["rlgym"] = rlgym_mod
sys.modules["rlgym.rocket_league.api"] = types.ModuleType("rlgym.rocket_league.api")
sys.modules["rlgym.rocket_league.api"].GameState = GameState
sys.modules["rlgym.rocket_league.api"].Car = Car
sys.modules["rlgym.rocket_league.api"].PhysicsObject = PhysicsObject
sys.modules["rlgym.rocket_league.api"].GameConfig = GameConfig
sys.modules["rlgym.rocket_league.common_values"] = common_values_mod
sys.modules["rlgym.rocket_league.api"] = api_rl_mod
sys.modules["rlgym.rocket_league.sim.rocketsim_engine"] = sim_mod
sys.modules["rlgym.rocket_league.done_conditions.goal_condition"] = goal_mod
sys.modules[
    "rlgym.rocket_league.done_conditions.timeout_condition"
] = timeout_mod


# ---------------------------------------------------------------------------
# Import environment factory with stubs in place

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

# ---------------------------------------------------------------------------
# Import environment factory
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))
import importlib

import src.training.observers as observers
import src.training.rewards as rewards

importlib.reload(observers)
importlib.reload(rewards)
env_factory = importlib.reload(importlib.import_module("src.training.env_factory"))

RL2v2Env = env_factory.RL2v2Env
CONT_DIM = env_factory.CONT_DIM
DISC_DIM = env_factory.DISC_DIM

from src.rlbot_integration.observation_adapter import OBS_SIZE
from rlgym.api.config import ObsBuilder, RewardFunction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def env():
    return RL2v2Env()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_reset_returns_obs_vec(env):
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_SIZE,)
    assert issubclass(obs.dtype.type, np.floating)
    assert isinstance(env._obs_builder, ObsBuilder)
    assert isinstance(env._reward_fn, RewardFunction)


def test_step_produces_finite_reward(env):

env_factory = importlib.reload(importlib.import_module("src.training.env_factory"))
RL2v2Env = env_factory.RL2v2Env
CONT_DIM = env_factory.CONT_DIM
DISC_DIM = env_factory.DISC_DIM


def test_reset_returns_obs_vec():
    env = RL2v2Env()
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape


def test_step_produces_float_reward():
    env = make_env()()

    env = RL2v2Env()
    env.reset()
    action = {
        "cont": np.zeros(CONT_DIM, dtype=np.float32),
        "disc": np.zeros(DISC_DIM, dtype=np.float32),
    }
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_SIZE,)
    assert np.isfinite(reward)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)

