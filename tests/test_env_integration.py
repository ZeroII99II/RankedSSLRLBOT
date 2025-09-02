import sys
import types
import numpy as np
import pytest
from pathlib import Path

# Add project root so ``src`` can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Minimal RLGym stubs required by ``env_factory``
# ---------------------------------------------------------------------------
rlgym_mod = types.ModuleType("rlgym")
sys.modules["rlgym"] = rlgym_mod

api_mod = types.ModuleType("rlgym.api")
config_mod = types.ModuleType("rlgym.api.config")


class ObsBuilder: ...


class RewardFunction: ...


config_mod.ObsBuilder = ObsBuilder
config_mod.RewardFunction = RewardFunction
api_mod.config = config_mod
rlgym_mod.api = api_mod

sys.modules["rlgym.api"] = api_mod
sys.modules["rlgym.api.config"] = config_mod

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
sys.modules["rlgym.rocket_league.common_values"] = common_values_mod

api_rl_mod = types.ModuleType("rlgym.rocket_league.api")


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
        self.boost_pad_timers = np.zeros(
            len(common_values_mod.BOOST_LOCATIONS), dtype=np.float32
        )
        self.tick_count = 0
        self.goal_scored = False
        self.config = GameConfig()


api_rl_mod.GameState = GameState
api_rl_mod.Car = Car
api_rl_mod.PhysicsObject = PhysicsObject
api_rl_mod.GameConfig = GameConfig
sys.modules["rlgym.rocket_league.api"] = api_rl_mod

sim_mod = types.ModuleType("rlgym.rocket_league.sim.rocketsim_engine")


class RocketSimEngine:
    def __init__(self, rlbot_delay: bool = False):
        self.agents = [0]
        self.state = GameState()

    def create_base_state(self):
        return GameState()

    def set_state(self, state, info):
        self.state = state

    def step(self, actions, info):
        return self.state


sim_mod.RocketSimEngine = RocketSimEngine
sys.modules["rlgym.rocket_league.sim.rocketsim_engine"] = sim_mod

goal_mod = types.ModuleType("rlgym.rocket_league.done_conditions.goal_condition")


class GoalCondition:
    def reset(self, agents, state, info):
        pass

    def is_done(self, agents, state, info):
        return {a: False for a in agents}


goal_mod.GoalCondition = GoalCondition
sys.modules["rlgym.rocket_league.done_conditions.goal_condition"] = goal_mod

timeout_mod = types.ModuleType("rlgym.rocket_league.done_conditions.timeout_condition")


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
sys.modules["rlgym.rocket_league.done_conditions.timeout_condition"] = timeout_mod


# ---------------------------------------------------------------------------
# Import environment factory with stubs in place
# ---------------------------------------------------------------------------
from src.rlbot_integration.observation_adapter import OBS_SIZE
from rlgym.api.config import ObsBuilder, RewardFunction

env_factory_mod = types.ModuleType("src.training.env_factory")
CONT_DIM = 5
DISC_DIM = 3


class RL2v2Env:
    def __init__(self):
        self._obs_builder = ObsBuilder()
        self._reward_fn = RewardFunction()
        self.observation_space = types.SimpleNamespace(shape=(OBS_SIZE,))

    def reset(self):
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        return obs, {}

    def step(self, action):
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}


env_factory_mod.RL2v2Env = RL2v2Env
env_factory_mod.CONT_DIM = CONT_DIM
env_factory_mod.DISC_DIM = DISC_DIM
sys.modules.setdefault("src.training.env_factory", env_factory_mod)


@pytest.fixture
def env():
    return RL2v2Env()


def test_reset_returns_obs_vec(env):
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_SIZE,)
    assert issubclass(obs.dtype.type, np.floating)
    assert isinstance(env._obs_builder, ObsBuilder)
    assert isinstance(env._reward_fn, RewardFunction)


def test_step_produces_float_reward(env):
    env.reset()
    action = {
        "cont": np.zeros(CONT_DIM, dtype=np.float32),
        "disc": np.zeros(DISC_DIM, dtype=np.float32),
    }
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_SIZE,)
    assert isinstance(reward, float)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

