import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training.env_factory import make_env, CONT_DIM, DISC_DIM
from src.rlbot_integration.observation_adapter import OBS_SIZE
from rlgym.api.config import ObsBuilder, RewardFunction


def test_make_env_observation_size():
    env = make_env()()
    obs, _ = env.reset()
    assert obs.shape == (OBS_SIZE,)
    assert isinstance(env._obs_builder, ObsBuilder)
    assert isinstance(env._reward_fn, RewardFunction)


def test_step_reward_not_nan():
    env = make_env()()
    env.reset()
    action = {
        "cont": np.zeros(CONT_DIM, dtype=np.float32),
        "disc": np.zeros(DISC_DIM, dtype=np.float32),
    }
    _, reward, _, _, _ = env.step(action)
    assert np.isfinite(reward)


def test_env_eventually_terminates():
    env = make_env()()
    env.reset()
    action = {"cont": np.zeros(CONT_DIM, dtype=np.float32), "disc": np.zeros(DISC_DIM, dtype=np.float32)}
    terminated = truncated = False
    for _ in range(1000):
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    assert terminated or truncated
