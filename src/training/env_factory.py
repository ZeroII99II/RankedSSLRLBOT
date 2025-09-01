"""Gymnasium environment factory wrapping a RLGym v2 session.

This implementation provides a very small stub environment used in tests and
as a placeholder for the real Rocket League environment.  It exposes the
observation and action spaces expected by the training pipeline:

- Observation: 107-dimensional float vector.
- Action: Dict with continuous values ``cont`` (5 dims in [-1, 1]) and
  discrete logits ``disc`` (3 dims).
"""
from __future__ import annotations

import numpy as np
from gymnasium import Env, spaces


class StubRLGymEnv(Env):
    """Minimal stub matching the interface of the actual RLGym environment."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(107,), dtype=np.float32
        )
        self.action_space = spaces.Dict(
            {
                "cont": spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32),
                "disc": spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return np.zeros(107, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(107, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


def make_env() -> Env:
    """Return an environment instance.

    In production this should construct the real RLGym v2 environment, but for
    tests we return a lightweight stub with the correct spaces.
    """
    return StubRLGymEnv()
