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

import numpy as np

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


# Action schema: continuous and discrete controls
CONT_DIM = 5
DISC_DIM = 3


class RL2v2Env(gym.Env):
    """Small 2v2 Rocket League environment using project builders."""

    # ``gym.Env`` expects ``metadata['render_modes']`` to advertise the
    # available render modes.  We support a single RGB array mode which is
    # used by the training script when ``--render`` is supplied.
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

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

    # ------------------------------------------------------------------
    # State helpers
    def _random_state(self) -> GameState:
        """Create a randomly-initialised 2v2 game state."""
        players = []
        for i in range(4):
            team = 0 if i < 2 else 1
            car = CarData(team_num=team)
            car.set_pos(*self.np_random.uniform(-1000, 1000, size=3))
            car.set_lin_vel(*self.np_random.uniform(-500, 500, size=3))
            car.set_ang_vel(*self.np_random.uniform(-5, 5, size=3))
            car.set_rot(*self.np_random.uniform(-np.pi, np.pi, size=3))
            player = PlayerData(
                car_data=car,
                team_num=team,
                boost_amount=float(self.np_random.uniform(0, 100)),
            )
            players.append(player)

        ball = BallData()
        ball.set_pos(*self.np_random.uniform(-1000, 1000, size=3))
        ball.set_lin_vel(*self.np_random.uniform(-500, 500, size=3))

        pads = [BoostPad(position=loc.astype(np.float32)) for loc in common_values.BOOST_LOCATIONS]

        return GameState(ball=ball, players=players, boost_pads=pads)

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

    # ------------------------------------------------------------------
    # Rendering
    def render(self, mode: str = "rgb_array"):
        """Render a simple top-down view of the field.

        Parameters
        ----------
        mode:
            Only ``"rgb_array"`` is supported.  The function returns a
            ``(H, W, 3)`` ``uint8`` array representing the frame.
        """

        if mode != "rgb_array":
            raise NotImplementedError(f"Unsupported render mode: {mode}")
        if self._state is None:
            raise RuntimeError("Environment must be reset before rendering")

        width, height = 320, 240
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (40, 160, 40)  # simple green field

        def _project(pos: np.ndarray) -> tuple[int, int]:
            # Map world coordinates roughly spanning [-3000, 3000] in X and
            # [-4000, 4000] in Y to pixel coordinates.
            x = int((np.clip(pos[0], -3000, 3000) + 3000) / 6000 * width)
            y = int((np.clip(pos[1], -4000, 4000) + 4000) / 8000 * height)
            return x, height - y - 1  # origin at bottom left

        # Draw players
        for player in self._state.players:
            px, py = _project(player.car_data.position)
            color = (255, 0, 0) if player.team_num == 0 else (0, 0, 255)
            x0, x1 = max(0, px - 2), min(width, px + 3)
            y0, y1 = max(0, py - 2), min(height, py + 3)
            frame[y0:y1, x0:x1] = color

        # Draw ball
        bx, by = _project(self._state.ball.position)
        x0, x1 = max(0, bx - 2), min(width, bx + 3)
        y0, y1 = max(0, by - 2), min(height, by + 3)
        frame[y0:y1, x0:x1] = (255, 255, 255)

        return frame


def make_env(seed: int = 42) -> Callable[[], RL2v2Env]:
    """Return a thunk that creates a seeded ``RL2v2Env`` instance."""

    def _thunk() -> RL2v2Env:
        return RL2v2Env(seed=seed)

    return _thunk

