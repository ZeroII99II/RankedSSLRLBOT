from __future__ import annotations

from typing import Dict, Any, Callable, List

import numpy as np

from src.compat.rlgym_v2_compat.common_values import BOOST_LOCATIONS, CAR_MAX_SPEED
from src.utils.gym_compat import gym

# Modern RLGym v2 components used by the environment
from .observers import ModernObsBuilder
from .rewards import ModernRewardSystem
from .state_setters import ModernStateSetter

# Your RLBot obs adapter defines the canonical size:
from src.rlbot_integration.observation_adapter import OBS_SIZE  # must be 107

# Action schema:
#   cont: [steer, throttle, pitch, yaw, roll] in [-1, 1]
#   disc: [jump, boost, handbrake] as {0,1}
CONT_DIM = 5
DISC_DIM = 3


class _Ball:
    """Simple physics object representing the ball."""

    def __init__(self):
        self.position = np.zeros(3, dtype=np.float32)
        self.linear_velocity = np.zeros(3, dtype=np.float32)

    def set_pos(self, x: float, y: float, z: float):
        self.position[:] = (x, y, z)

    def set_lin_vel(self, x: float, y: float, z: float):
        self.linear_velocity[:] = (x, y, z)


class _Car:
    """Minimal car model with orientation helpers."""

    def __init__(self, team: int):
        self.team_num = team
        self.position = np.zeros(3, dtype=np.float32)
        self.linear_velocity = np.zeros(3, dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0

    def set_pos(self, x: float, y: float, z: float):
        self.position[:] = (x, y, z)

    def set_lin_vel(self, x: float, y: float, z: float):
        self.linear_velocity[:] = (x, y, z)

    def set_ang_vel(self, x: float, y: float, z: float):
        self.angular_velocity[:] = (x, y, z)

    def set_rot(self, pitch: float, yaw: float, roll: float):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

    # Orientation helpers used by observation builder
    def forward(self) -> np.ndarray:
        cp = np.cos(self.pitch)
        sp = np.sin(self.pitch)
        cy = np.cos(self.yaw)
        sy = np.sin(self.yaw)
        return np.array([cp * cy, cp * sy, sp], dtype=np.float32)

    def up(self) -> np.ndarray:
        cp = np.cos(self.pitch)
        sp = np.sin(self.pitch)
        cy = np.cos(self.yaw)
        sy = np.sin(self.yaw)
        cr = np.cos(self.roll)
        sr = np.sin(self.roll)
        return np.array([
            -sr * sy + cr * sp * cy,
            sr * cy + cr * sp * sy,
            cr * cp,
        ], dtype=np.float32)


class _Player:
    """Lightweight player wrapper used for obs/reward builders."""

    def __init__(self, car: _Car):
        self.car_data = car
        self.team_num = car.team_num
        self.boost_amount = 0.0
        self.on_ground = True
        self.has_flip = True
        self.has_jump = True
        self.ball_touched = False
        self.is_demoed = False
        self.match_demolishes = 0


class _BoostPad:
    def __init__(self, position: np.ndarray):
        self.position = position
        self.is_active = True


class _GameState:
    """Container storing dynamic game information."""

    def __init__(self, ball: _Ball, players: List[_Player], boost_pads: List[_BoostPad]):
        self.ball = ball
        self.players = players
        self.blue_score = 0
        self.orange_score = 0
        self.game_seconds_remaining = 300.0
        self.is_overtime = False
        self.is_kickoff_pause = False
        self.boost_pads = boost_pads

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

        # Core RLGymv2-style components
        self._obs_builder = ModernObsBuilder()
        self._reward_fn = ModernRewardSystem()
        self._state_setter = ModernStateSetter()

        # Create underlying physics objects shared between state wrapper and game state
        cars = [_Car(0), _Car(0), _Car(1), _Car(1)]
        players = [_Player(c) for c in cars]
        boost_pads = [_BoostPad(pos.copy()) for pos in BOOST_LOCATIONS]
        self._ball = _Ball()

        # Wrapper used by the state setter to configure scenarios
        self._state_wrapper = type("StateWrapper", (), {})()
        self._state_wrapper.cars = cars
        self._state_wrapper.ball = self._ball

        # Game state object passed to builders/rewards
        self._state = _GameState(self._ball, players, boost_pads)
        # Two controlled players (first two on blue team)
        self._controlled: List[_Player] = players[:2]

    # Gymnasium API:
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        # Apply state setter to configure kickoff / scenario. Some of the
        # placeholder state setters used in tests lack full implementations,
        # so we guard against missing methods.
        try:
            self._state_setter.reset(self._state_wrapper)
        except AttributeError:
            pass

        # Builders expect the game state to be up to date after state setter
        self._obs_builder.reset(self._state)
        self._reward_fn.reset(self._state)

        obs_vec = self._obs_builder.build_obs(self._controlled[0], self._state, self._prev_action)
        info: Dict[str, Any] = {}
        self._prev_action[:] = 0
        return obs_vec.astype(np.float32), info

    def step(self, action: Dict[str, np.ndarray]):
        # Unpack actions
        a_cont = np.clip(action["cont"].astype(np.float32), -1.0, 1.0)
        a_disc = action["disc"].astype(np.float32).clip(0, 1)
        # Apply actions to controlled cars
        for player in self._controlled:
            car = player.car_data
            # Orientation changes
            car.set_rot(
                car.pitch + a_cont[2] * 0.02,
                car.yaw + a_cont[3] * 0.02,
                car.roll + a_cont[4] * 0.02,
            )
            # Simple throttle-based velocity in forward direction
            forward = car.forward()
            speed = a_cont[1] * CAR_MAX_SPEED * 0.1
            car.set_lin_vel(*(forward * speed))
            car.set_pos(*(car.position + car.linear_velocity * 0.016))

            # Discrete actions
            if a_disc[0]:  # jump
                player.on_ground = False
                player.has_jump = False
            else:
                player.on_ground = True
            if a_disc[1]:  # boost
                player.boost_amount = min(100.0, player.boost_amount + 10.0)
            else:
                player.boost_amount = max(0.0, player.boost_amount - 0.5)

        # Advance simple game timer
        self._state.game_seconds_remaining = max(
            0.0, self._state.game_seconds_remaining - 0.016
        )

        # Build observation and reward
        obs_vec = self._obs_builder.build_obs(self._controlled[0], self._state, self._prev_action)
        reward = 0.0
        for player in self._controlled:
            reward += self._reward_fn.get_reward(player, self._state, self._prev_action)
        reward /= len(self._controlled)

        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        self._prev_action[:CONT_DIM] = a_cont
        self._prev_action[CONT_DIM:] = a_disc
        return obs_vec.astype(np.float32), float(reward), bool(terminated), bool(truncated), info

def make_env(seed: int = 42) -> Callable[[], RL2v2Env]:
    def _thunk():
        return RL2v2Env(seed=seed)
    return _thunk
