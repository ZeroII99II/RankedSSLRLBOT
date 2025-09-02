import sys
import types
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stub minimal rlgym modules expected by env_factory
# ---------------------------------------------------------------------------
rlgym_mod = types.ModuleType("rlgym")
api_mod = types.ModuleType("rlgym.api")
config_mod = types.ModuleType("rlgym.api.config")


class ObsBuilder: ...


class RewardFunction: ...


config_mod.ObsBuilder = ObsBuilder
config_mod.RewardFunction = RewardFunction
api_mod.config = config_mod
rlgym_mod.api = api_mod

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
common_values_mod.BLUE_GOAL_BACK = np.array([0, -5000, 0])
common_values_mod.BLUE_GOAL_CENTER = np.array([0, -5120, 0])
common_values_mod.ORANGE_GOAL_BACK = np.array([0, 5000, 0])
common_values_mod.ORANGE_GOAL_CENTER = np.array([0, 5120, 0])
common_values_mod.GOAL_HEIGHT = 642
common_values_mod.ORANGE_TEAM = 1
common_values_mod.TICKS_PER_SECOND = 60
rocket_mod.common_values = common_values_mod

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
sys.modules["rlgym.api"] = api_mod
sys.modules["rlgym.api.config"] = config_mod
sys.modules["rlgym.rocket_league"] = rocket_mod
sys.modules["rlgym.rocket_league.common_values"] = common_values_mod
sys.modules["rlgym.rocket_league.api"] = api_rl_mod
sys.modules["rlgym.rocket_league.sim.rocketsim_engine"] = sim_mod
sys.modules["rlgym.rocket_league.done_conditions.goal_condition"] = goal_mod
sys.modules[
    "rlgym.rocket_league.done_conditions.timeout_condition"
] = timeout_mod


# ---------------------------------------------------------------------------
# Import environment factory with stubs in place
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

