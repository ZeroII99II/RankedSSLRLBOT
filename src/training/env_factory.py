from __future__ import annotations
from typing import Dict, Any, Callable
import numpy as np
from src.utils.gym_compat import gym

# Your RLBot obs adapter defines the canonical size:
from src.rlbot_integration.observation_adapter import OBS_SIZE  # must be 107

# Action schema:
#   cont: [steer, throttle, pitch, yaw, roll] in [-1, 1]
#   disc: [jump, boost, handbrake] as {0,1}
CONT_DIM = 5
DISC_DIM = 3

class RL2v2Env(gym.Env):
    """
    Gymnasium Env wrapping your RLGym 2.0 match (2v2 self-play).
    You MUST fill TODOs to call your actual RLGym v2 session/match.
    """
    metadata = {"render_modes": []}

    def __init__(self, seed: int = 42):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Dict({
            "cont": gym.spaces.Box(low=-1.0, high=1.0, shape=(CONT_DIM,), dtype=np.float32),
            "disc": gym.spaces.MultiBinary(DISC_DIM)
        })
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._prev_action = np.zeros(CONT_DIM + DISC_DIM, dtype=np.float32)

        # TODO: initialize your RLGym v2 match/session here (e.g., rocketsim + builders)
        # self._session = build_match_v2(...)  # you provide; keep a handle to step/reset

    # Gymnasium API:
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        # TODO: call your state setter / match reset for a 2v2 kickoff
        # state = self._session.reset()
        # obs_vec = your_obs_builder.build_obs(state, self._prev_action)
        obs_vec = np.zeros((OBS_SIZE,), dtype=np.float32)  # REPLACE with real obs
        info: Dict[str, Any] = {}
        self._prev_action[:] = 0
        return obs_vec, info

    def step(self, action: Dict[str, np.ndarray]):
        # Unpack actions
        a_cont = np.clip(action["cont"].astype(np.float32), -1.0, 1.0)
        a_disc = action["disc"].astype(np.float32).clip(0, 1)

        # TODO: apply actions to the two controlled cars for one tick in your RLGym v2 session.
        # state = self._session.step(a_cont, a_disc)

        # Build observation and reward
        # obs_vec = your_obs_builder.build_obs(state, self._prev_action)
        obs_vec = np.zeros((OBS_SIZE,), dtype=np.float32)  # REPLACE with real obs

        # TODO: compute reward via your reward system (2v2 teamplay rewards recommended)
        reward = 0.0

        # TODO: decide termination from your terminal conditions (goal scored, time, mercy, etc.)
        terminated = False  # True if episode ended naturally
        truncated = False   # True if time-limit or external cutoff
        info: Dict[str, Any] = {}

        self._prev_action[:CONT_DIM] = a_cont
        self._prev_action[CONT_DIM:] = a_disc
        return obs_vec, float(reward), bool(terminated), bool(truncated), info

def make_env(seed: int = 42) -> Callable[[], RL2v2Env]:
    def _thunk():
        return RL2v2Env(seed=seed)
    return _thunk
