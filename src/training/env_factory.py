from __future__ import annotations

"""Environment factory providing a minimal Rocket League style gym env.

This module wires the project's SSL observation builder and reward function
into a lightweight environment.  The environment does **not** rely on the
heavy RocketSim engine used in production; instead it uses simple physics
based on the compatibility dataclasses in ``src.compat.rlgym_v2_compat``.

The goal of this environment is to expose the observation and reward
implementations for testing and training loops without requiring the full
simulation stack.
"""

from typing import Callable, Dict, Any
from pathlib import Path

import numpy as np

try:  # pragma: no cover - fallback if PyYAML is not installed
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from src.utils.gym_compat import gym
from src.rlbot_integration.observation_adapter import OBS_SIZE
from src.training.observers import SSLObsBuilder
from src.training.rewards import SSLRewardFunction
from src.compat.rlgym_v2_compat.game_state import (
    GameState,
    PlayerData,
    CarData,
    BallData,
    BoostPad,
)
from src.compat.rlgym_v2_compat import common_values
from src.training.state_setters.scenarios import SCENARIOS


# Action schema: continuous and discrete controls
CONT_DIM = 5
DISC_DIM = 3


class RL2v2Env(gym.Env):
    """Small 2v2 Rocket League environment using project builders."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 42):
        super().__init__()

        # Observation and action spaces mirror the production setup.
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = gym.spaces.Dict(
            {
                "cont": gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(CONT_DIM,), dtype=np.float32
                ),
                "disc": gym.spaces.MultiBinary(DISC_DIM),
            }
        )

        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Project observation builder and reward function
        self._obs_builder = SSLObsBuilder()
        self._reward_fn = SSLRewardFunction()

        self._state: GameState | None = None
        self._prev_action = np.zeros(CONT_DIM + DISC_DIM, dtype=np.float32)

        # Scenario configuration
        self._scenario_funcs = SCENARIOS
        cfg_path = Path(__file__).resolve().parents[2] / "configs" / "scenario_weights.yaml"
        self._scenario_weights = self._load_scenario_weights(cfg_path)

    # ------------------------------------------------------------------
    # State helpers
    def _load_scenario_weights(self, path: Path) -> Dict[str, float]:
        """Read scenario weights from YAML configuration."""
        if yaml is None or not path.is_file():
            return {}
        try:  # pragma: no cover - simple config loader
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return {}
        return {str(k): float(v) for k, v in data.items() if k in self._scenario_funcs}

    def _random_state(self) -> GameState:
        """Create a scenario-driven 2v2 game state."""
        names = list(self._scenario_funcs.keys())
        weights = np.array([self._scenario_weights.get(n, 1.0) for n in names], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()

        choice = self.np_random.choice(names, p=weights)
        scenario_fn = self._scenario_funcs[choice]
        return scenario_fn(self.np_random)

    # ------------------------------------------------------------------
    # Gym API
    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self._state = self._random_state()
        self._obs_builder.reset(self._state)
        self._reward_fn.reset(self._state)
        self._prev_action.fill(0.0)

        obs = self._obs_builder.build_obs(
            self._state.players[0], self._state, self._prev_action
        )
        return obs.astype(np.float32), {}

    def step(self, action: Dict[str, np.ndarray]):
        if self._state is None:
            raise RuntimeError("Environment must be reset before stepping")

        a_cont = np.clip(action["cont"].astype(np.float32), -1.0, 1.0)
        a_disc = action["disc"].astype(np.float32).clip(0, 1)
        self._prev_action = np.concatenate([a_cont, a_disc])

        # Basic kinematics for controlled player (index 0)
        car = self._state.players[0].car_data
        car.set_lin_vel(
            a_cont[0] * common_values.CAR_MAX_SPEED,
            a_cont[1] * common_values.CAR_MAX_SPEED,
            a_cont[2] * common_values.CAR_MAX_SPEED,
        )
        car.position += car.linear_velocity * 0.016  # small time step

        # Drift ball slightly to keep observations dynamic
        self._state.ball.position += self._state.ball.linear_velocity * 0.016

        obs = self._obs_builder.build_obs(
            self._state.players[0], self._state, self._prev_action
        )
        reward = self._reward_fn.get_reward(
            self._state.players[0], self._state, self._prev_action
        )
        return obs.astype(np.float32), float(reward), False, False, {}


def make_env(seed: int = 42) -> Callable[[], RL2v2Env]:
    """Return a thunk that creates a seeded ``RL2v2Env`` instance."""

    def _thunk() -> RL2v2Env:
        return RL2v2Env(seed=seed)

    return _thunk

