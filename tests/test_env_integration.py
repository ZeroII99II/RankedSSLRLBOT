import sys, types
from pathlib import Path
import numpy as np

# Stub minimal rlgym modules expected by env_factory
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
rocket_mod.common_values = common_values_mod

sys.modules["rlgym"] = rlgym_mod
sys.modules["rlgym.api"] = api_mod
sys.modules["rlgym.api.config"] = config_mod
sys.modules["rlgym.rocket_league"] = rocket_mod
sys.modules["rlgym.rocket_league.common_values"] = common_values_mod
api_rl_mod = types.ModuleType("rlgym.rocket_league.api")
class GameState: ...
api_rl_mod.GameState = GameState
sys.modules["rlgym.rocket_league.api"] = api_rl_mod

sys.path.append(str(Path(__file__).resolve().parents[1]))
import importlib
import src.training.observers as observers
import src.training.rewards as rewards
importlib.reload(observers)
importlib.reload(rewards)
env_factory = importlib.reload(importlib.import_module("src.training.env_factory"))
RLMatchEnv = env_factory.RLMatchEnv
CONT_DIM = env_factory.CONT_DIM
DISC_DIM = env_factory.DISC_DIM
from rlgym.api.config import ObsBuilder, RewardFunction


def test_reset_returns_obs_vec():
    env = RLMatchEnv()
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert isinstance(env._obs_builder, ObsBuilder)
    assert isinstance(env._reward_fn, RewardFunction)


def test_step_produces_float_reward():
    env = RLMatchEnv()
    env.reset()
    action = {
        "cont": np.zeros(CONT_DIM, dtype=np.float32),
        "disc": np.zeros(DISC_DIM, dtype=np.float32),
    }
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert np.isfinite(reward)
