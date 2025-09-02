import sys
import types
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Stub rlgym modules for env_factory
# ---------------------------------------------------------------------------
rlgym_mod = types.ModuleType("rlgym")

api_cfg_mod = types.ModuleType("rlgym.api.config")
class ObsBuilder: ...
class RewardFunction: ...
api_cfg_mod.ObsBuilder = ObsBuilder
api_cfg_mod.RewardFunction = RewardFunction
sys.modules["rlgym.api"] = types.ModuleType("rlgym.api")
sys.modules["rlgym.api.config"] = api_cfg_mod

gym_mod = types.ModuleType("gymnasium")

class Space:
    pass

class Box(Space):
    def __init__(self, *args, shape=None, **kwargs):
        self.shape = shape

class MultiBinary(Space):
    def __init__(self, n, **kwargs):
        self.n = n

class Dict(dict):
    def __init__(self, spaces):
        super().__init__(spaces)

gym_mod.Env = object
gym_mod.Space = Space
gym_mod.spaces = types.SimpleNamespace(Box=Box, Dict=Dict, MultiBinary=MultiBinary)
gym_mod.utils = types.SimpleNamespace(seeding=types.SimpleNamespace(np_random=lambda seed: (np.random.default_rng(seed), seed)))
sys.modules["gymnasium"] = gym_mod

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


class Match:
    def __init__(self, *_, **__):
        pass

    def reset(self):
        return np.zeros(OBS_SIZE, dtype=np.float32)

    def step(self, action):
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}


sys.modules["rlgym"] = rlgym_mod
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
sys.modules["rlgym.rocket_league.match"] = types.ModuleType(
    "rlgym.rocket_league.match"
)
sys.modules["rlgym.rocket_league.match"].Match = Match

# ---------------------------------------------------------------------------
# Import environment factory
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))
import importlib
import src.training.observers as observers_mod
import src.training.rewards as rewards_mod
importlib.reload(observers_mod)
importlib.reload(rewards_mod)
env_factory = importlib.reload(importlib.import_module("src.training.env_factory"))
make_env = env_factory.make_env
CONT_DIM = env_factory.CONT_DIM
DISC_DIM = env_factory.DISC_DIM

from src.rlbot_integration.observation_adapter import OBS_SIZE
from rlgym.api.config import ObsBuilder, RewardFunction  # noqa: F401 - stubs for import side effects


def test_make_env_observation_size():
    env = make_env()()
    obs, _ = env.reset()
    assert obs.shape == (OBS_SIZE,)


def test_step_reward_not_nan():
    env = make_env()()
    env.reset()
    action = {
        "cont": np.zeros(CONT_DIM, dtype=np.float32),
        "disc": np.zeros(DISC_DIM, dtype=np.float32),
    }
    _, reward, _, _, _ = env.step(action)
    assert np.isfinite(reward)

