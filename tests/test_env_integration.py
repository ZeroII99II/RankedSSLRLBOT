import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.training.env_factory import RL2v2Env, CONT_DIM, DISC_DIM
from src.rlbot_integration.observation_adapter import OBS_SIZE

def test_reset_returns_obs_vec():
    env = RL2v2Env()
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_SIZE,)


def test_step_produces_float_reward():
    env = RL2v2Env()
    env.reset()
    action = {
        "cont": np.zeros(CONT_DIM, dtype=np.float32),
        "disc": np.zeros(DISC_DIM, dtype=np.float32),
    }
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, np.ndarray) and obs.shape == (OBS_SIZE,)
    assert isinstance(reward, float)
    assert np.isfinite(reward)
