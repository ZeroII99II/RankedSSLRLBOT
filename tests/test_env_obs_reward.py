import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.training.env_factory import make_env, CONT_DIM, DISC_DIM
from src.rlbot_integration.observation_adapter import OBS_SIZE


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
    assert not np.isnan(reward)
