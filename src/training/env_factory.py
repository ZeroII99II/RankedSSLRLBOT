from __future__ import annotations

"""Environment factory providing a minimal Rocket League style gym env.

This module wires the project's SSL observation builder and reward function
into a lightweight environment.  It now relies on the real RLGym v2
simulation classes and RocketSim based transition engine instead of the
lightweight compatibility dataclasses that previously lived under
``src.compat``.  Only a tiny wrapper is kept to expose the attributes that
our observation builder and reward function expect.

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

# True RLGym v2 classes and helpers
from rlgym.rocket_league.api import GameState, Car, PhysicsObject, GameConfig
from rlgym.rocket_league.common_values import (
    BOOST_LOCATIONS,
    TICKS_PER_SECOND,
)
from rlgym.rocket_league.sim.rocketsim_engine import RocketSimEngine
from rlgym.rocket_league.done_conditions.goal_condition import GoalCondition
from rlgym.rocket_league.done_conditions.timeout_condition import TimeoutCondition

from src.training.state_setters.scenarios import SCENARIOS


# Action schema: continuous and discrete controls
CONT_DIM = 5
DISC_DIM = 3


class CarDataWrapper:
    """Expose the minimal car interface expected by our builders."""

    def __init__(self, car: Car):
        self._car = car

    # Positions / velocities -------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self._car.physics.position

    @property
    def linear_velocity(self) -> np.ndarray:
        return self._car.physics.linear_velocity

    @property
    def angular_velocity(self) -> np.ndarray:
        return self._car.physics.angular_velocity

    # Orientation helpers ----------------------------------------------------
    def forward(self) -> np.ndarray:
        return self._car.physics.forward

    def up(self) -> np.ndarray:
        return self._car.physics.up

    # Pitch/Yaw/Roll ---------------------------------------------------------
    @property
    def pitch(self) -> float:
        return float(self._car.physics.pitch)

    @property
    def yaw(self) -> float:
        return float(self._car.physics.yaw)

    @property
    def roll(self) -> float:
        return float(self._car.physics.roll)


class PlayerWrapper:
    """Minimal player wrapper backed by a RLGym ``Car`` instance."""

    def __init__(self, car: Car):
        self.car_data = CarDataWrapper(car)
        self.team_num = car.team_num
        self.boost_amount = float(car.boost_amount)
        self.on_ground = car.on_ground
        self.has_flip = car.has_flip
        # ``has_jump`` in legacy dataclasses corresponds to ``not has_jumped``
        self.has_jump = not car.has_jumped
        self.ball_touched = car.ball_touches > 0
        self.is_demoed = car.is_demoed
        self.match_demolishes = 0


class BallDataWrapper:
    def __init__(self, phys: PhysicsObject):
        self.position = phys.position
        self.linear_velocity = phys.linear_velocity


class BoostPadWrapper:
    def __init__(self, pos: np.ndarray, timer: float):
        self.position = pos.astype(np.float32)
        self.is_active = timer <= 0


class GameStateWrapper:
    """Compatibility view over the RLGym ``GameState``."""

    def __init__(self, state: GameState):
        self.ball = BallDataWrapper(state.ball)
        # Ensure deterministic ordering of players
        self.players = [PlayerWrapper(state.cars[i]) for i in sorted(state.cars.keys())]
        self.boost_pads = [
            BoostPadWrapper(np.array(loc), state.boost_pad_timers[i])
            for i, loc in enumerate(BOOST_LOCATIONS)
        ]
        # Dummy scoreboard / timing information
        self.blue_score = 0
        self.orange_score = 0
        self.game_seconds_remaining = max(
            0.0, 300.0 - state.tick_count / TICKS_PER_SECOND
        )
        self.is_overtime = False
        self.is_kickoff_pause = False


class RL2v2Env(gym.Env):
    """Small 2v2 Rocket League environment using project builders."""

    # ``gym.Env`` expects ``metadata['render_modes']`` to advertise the
    # available render modes.  We support a single RGB array mode which is
    # used by the training script when ``--render`` is supplied.
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, seed: int = 42, render: bool = False, num_players_per_team: int = 2):

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(self, seed: int = 42, render: bool = False):
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

        # RLGym engine and done conditions
        self._engine = RocketSimEngine(rlbot_delay=False)
        self._termination_cond = GoalCondition()
        # Short timeout keeps unit tests fast while still exercising truncation logic
        self._truncation_cond = TimeoutCondition(5.0)

        self._state: GameState | None = None
        self._prev_action = np.zeros(CONT_DIM + DISC_DIM, dtype=np.float32)
        self._render_enabled = render

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
        """Create a ``GameState`` from the configured scenarios.

        The environment samples a scenario based on the configured weights. If
        the selected scenario already returns a :class:`GameState` instance it is
        used directly.  Otherwise a fallback random state is generated using the
        internal RocketSim engine.
        """

        names = list(self._scenario_funcs.keys())
        weights = np.array(
            [self._scenario_weights.get(n, 1.0) for n in names], dtype=float
        )
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()

        choice = self.np_random.choice(names, p=weights)
        scenario_fn = self._scenario_funcs[choice]
        state = scenario_fn(self.np_random)
        if isinstance(state, GameState):
            return state

        """Create a scenario-driven 2v2 game state."""

        # Fallback: create a random state using the real engine
        gs = self._engine.create_base_state()
        gs.tick_count = 0
        gs.goal_scored = False

        # Random ball
        ball = PhysicsObject()
        ball.position = self.np_random.uniform(-1000, 1000, size=3).astype(np.float32)
        ball.linear_velocity = self.np_random.uniform(-500, 500, size=3).astype(
            np.float32
        )
        ball.angular_velocity = self.np_random.uniform(-5, 5, size=3).astype(
            np.float32
        )
        ball.euler_angles = self.np_random.uniform(-np.pi, np.pi, size=3).astype(
            np.float32
        )
        gs.ball = ball

        # Random cars

        # Initialise four cars (two per team) with default physics
        gs.cars = {}
        total_players = 2 * self._num_players_per_team
        for i in range(total_players):
            team = 0 if i < self._num_players_per_team else 1
            car = Car()
            car.team_num = team
            car.ball_touches = 0
            car.boost_amount = float(self.np_random.uniform(0, 100))
            car.on_ground = True
            car.has_flip = True
            car.has_jumped = False
            car.is_holding_jump = False
            car.jump_time = 0.0
            car.has_flipped = False
            car.has_double_jumped = False
            car.air_time_since_jump = 0.0
            car.flip_time = 0.0
            car.flip_torque = np.zeros(3, dtype=np.float32)
            car.is_autoflipping = False
            car.autoflip_timer = 0.0
            car.autoflip_direction = 1.0
            phys = PhysicsObject()
            phys.position = self.np_random.uniform(-1000, 1000, size=3).astype(
                np.float32
            )
            phys.linear_velocity = self.np_random.uniform(-500, 500, size=3).astype(
                np.float32
            )
            phys.angular_velocity = self.np_random.uniform(-5, 5, size=3).astype(
                np.float32
            )
            phys.euler_angles = self.np_random.uniform(-np.pi, np.pi, size=3).astype(
                np.float32
            )
            car.physics = phys

            car.is_demoed = False
            car.physics = PhysicsObject()
            gs.cars[i] = car

        gs.boost_pad_timers = np.zeros(len(BOOST_LOCATIONS), dtype=np.float32)
        gs.config = GameConfig()

        return gs

        names = list(self._scenario_funcs.keys())
        weights = np.array([self._scenario_weights.get(n, 1.0) for n in names], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()

        choice = self.np_random.choice(names, p=weights)
        scenario_fn = self._scenario_funcs[choice]
        return scenario_fn(self.np_random, gs)

    # ------------------------------------------------------------------
    # Gym API
    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self._state = self._random_state()
        self._engine.set_state(self._state, {})

        wrapper = GameStateWrapper(self._state)
        self._obs_builder.reset(wrapper)
        self._reward_fn.reset(wrapper)
        self._termination_cond.reset(self._engine.agents, self._state, {})
        self._truncation_cond.reset(self._engine.agents, self._state, {})
        self._prev_action.fill(0.0)

        obs = self._obs_builder.build_obs(
            wrapper.players[0], wrapper, self._prev_action
        )
        return obs.astype(np.float32), {}

    def step(self, action: Dict[str, np.ndarray]):
        if self._state is None:
            raise RuntimeError("Environment must be reset before stepping")

        a_cont = np.clip(action["cont"].astype(np.float32), -1.0, 1.0)
        a_disc = action["disc"].astype(np.float32).clip(0, 1)
        self._prev_action = np.concatenate([a_cont, a_disc])

        engine_action = np.concatenate([a_cont, a_disc]).reshape(1, -1)
        actions = {
            aid: np.zeros_like(engine_action) for aid in self._engine.agents
        }
        actions[self._engine.agents[0]] = engine_action

        self._state = self._engine.step(actions, {})
        wrapper = GameStateWrapper(self._state)

        obs = self._obs_builder.build_obs(
            wrapper.players[0], wrapper, self._prev_action
        )
        reward = self._reward_fn.get_reward(
            wrapper.players[0], wrapper, self._prev_action
        )

        terminated = self._termination_cond.is_done(
            self._engine.agents, self._state, {}
        )[self._engine.agents[0]]
        truncated = self._truncation_cond.is_done(
            self._engine.agents, self._state, {}
        )[self._engine.agents[0]]

        return (
            obs.astype(np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            {},
        )

    # ------------------------------------------------------------------
    # Rendering
    def render(self, mode: str = "rgb_array"):
        if not self._render_enabled:
            raise RuntimeError("Rendering disabled; initialise with render=True")
        if mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode: {mode}")
        if self._state is None:
            return np.zeros((256, 256, 3), dtype=np.uint8)

        size = 256
        frame = np.zeros((size, size, 3), dtype=np.uint8)

        def to_px(vec):
            x = int((vec[0] + 2000) / 4000 * (size - 1))
            y = int((vec[1] + 2000) / 4000 * (size - 1))
            return x, y

        bx, by = to_px(self._state.ball.position)
        frame[by, bx] = (255, 255, 255)
        for car in self._state.cars.values():
            px, py = to_px(car.physics.position)
            color = (0, 0, 255) if car.team_num == 0 else (255, 0, 0)
            frame[py, px] = color

        if mode == "human":
            try:
                import cv2  # type: ignore

                cv2.imshow("RL2v2Env", frame)
                cv2.waitKey(1)
            except Exception:
                pass
        return frame



def make_env(seed: int = 42, render: bool = False) -> Callable[[], RL2v2Env]:
    """Return a thunk that creates a seeded ``RL2v2Env`` instance."""

    def _thunk() -> RL2v2Env:
        return RL2v2Env(seed=seed, render=render)

    return _thunk

