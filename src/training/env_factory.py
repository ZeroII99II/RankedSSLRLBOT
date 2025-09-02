from __future__ import annotations

from typing import Dict, Any, Callable, List

import numpy as np

from src.compat.rlgym_v2_compat.common_values import BOOST_LOCATIONS, CAR_MAX_SPEED
from src.utils.gym_compat import gym

# Import real RLGym v2 data structures
from rlgym.rocket_league.api import GameState, Car, PhysicsObject
from rlgym.rocket_league.api.game_config import GameConfig
from rlgym.rocket_league.math import euler_to_rotation

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

from dataclasses import dataclass


@dataclass
class BoostPad:
    position: np.ndarray
    is_active: bool = True


class BallObject(PhysicsObject):
    __slots__ = PhysicsObject.__slots__

    def __init__(self):
        super().__init__()
        self.position = np.zeros(3, dtype=np.float32)
        self.linear_velocity = np.zeros(3, dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        self.rotation_mtx = np.eye(3, dtype=np.float32)

    def set_pos(self, x: float, y: float, z: float):
        self.position[:] = (x, y, z)

    def set_lin_vel(self, x: float, y: float, z: float):
        self.linear_velocity[:] = (x, y, z)


class CarObject(Car):
    __slots__ = Car.__slots__

    def __init__(self, team: int):
        super().__init__()
        self.team_num = team
        self.hitbox_type = 0
        self.ball_touches = 0
        self.bump_victim_id = None
        self.demo_respawn_timer = 0.0
        self.wheels_with_contact = (True, True, True, True)
        self.supersonic_time = 0.0
        self.boost_amount = 0.0
        self.boost_active_time = 0.0
        self.handbrake = 0.0
        self.is_jumping = False
        self.has_jumped = False
        self.is_holding_jump = False
        self.jump_time = 0.0
        self.has_flipped = False
        self.has_double_jumped = False
        self.air_time_since_jump = 0.0
        self.flip_time = 0.0
        self.flip_torque = np.zeros(3, dtype=np.float32)
        self.is_autoflipping = False
        self.autoflip_timer = 0.0
        self.autoflip_direction = 0.0
        self.physics = PhysicsObject()
        self.physics.position = np.zeros(3, dtype=np.float32)
        self.physics.linear_velocity = np.zeros(3, dtype=np.float32)
        self.physics.angular_velocity = np.zeros(3, dtype=np.float32)
        self.physics.rotation_mtx = np.eye(3, dtype=np.float32)
        self._inverted_physics = None

    def set_pos(self, x: float, y: float, z: float):
        self.physics.position[:] = (x, y, z)

    def set_lin_vel(self, x: float, y: float, z: float):
        self.physics.linear_velocity[:] = (x, y, z)

    def set_ang_vel(self, x: float, y: float, z: float):
        self.physics.angular_velocity[:] = (x, y, z)

    def set_rot(self, pitch: float, yaw: float, roll: float):
        self.physics.euler_angles = np.array([pitch, yaw, roll], dtype=np.float32)
        self.physics.rotation_mtx = euler_to_rotation(self.physics.euler_angles)

    def forward(self) -> np.ndarray:
        return self.physics.forward

    def up(self) -> np.ndarray:
        return self.physics.up

    @property
    def pitch(self) -> float:
        return float(self.physics.pitch)

    @property
    def yaw(self) -> float:
        return float(self.physics.yaw)

    @property
    def roll(self) -> float:
        return float(self.physics.roll)

    # Alias used by state setters
    @property
    def boost(self) -> float:
        return float(self.boost_amount)

    @boost.setter
    def boost(self, val: float):
        self.boost_amount = float(val)

    # Convenience aliases used by existing code
    @property
    def position(self) -> np.ndarray:
        return self.physics.position

    @property
    def linear_velocity(self) -> np.ndarray:
        return self.physics.linear_velocity

    @property
    def angular_velocity(self) -> np.ndarray:
        return self.physics.angular_velocity


class Player:
    """Lightweight player wrapper used for obs/reward builders."""

    def __init__(self, car: CarObject):
        self.car_data = car
        self.team_num = car.team_num
        self.on_ground = True
        self.has_flip = True
        self.has_jump = True
        self.ball_touched = False
        self.is_demoed = False
        self.match_demolishes = 0

    @property
    def boost_amount(self) -> float:
        return float(self.car_data.boost_amount)

    @boost_amount.setter
    def boost_amount(self, val: float):
        self.car_data.boost_amount = float(val)


class RLGameState(GameState):
    __slots__ = GameState.__slots__ + (
        "players",
        "boost_pads",
        "blue_score",
        "orange_score",
        "game_seconds_remaining",
        "is_overtime",
        "is_kickoff_pause",
    )

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
        cars = [CarObject(0), CarObject(0), CarObject(1), CarObject(1)]
        players = [Player(c) for c in cars]
        boost_pads = [BoostPad(pos.copy()) for pos in BOOST_LOCATIONS]
        self._ball = BallObject()

        # Wrapper used by the state setter to configure scenarios
        self._state_wrapper = type("StateWrapper", (), {})()
        self._state_wrapper.cars = cars
        self._state_wrapper.ball = self._ball

        # Game state object passed to builders/rewards
        self._state = RLGameState()
        self._state.tick_count = 0
        self._state.goal_scored = False
        self._state.config = GameConfig()
        self._state.cars = {i: c for i, c in enumerate(cars)}
        self._state.ball = self._ball
        self._state._inverted_ball = None
        self._state.boost_pad_timers = np.zeros(len(BOOST_LOCATIONS), dtype=np.float32)
        self._state._inverted_boost_pad_timers = None
        self._state.blue_score = 0
        self._state.orange_score = 0
        self._state.game_seconds_remaining = 300.0
        self._state.is_overtime = False
        self._state.is_kickoff_pause = False
        self._state.boost_pads = boost_pads
        self._state.players = players

        # Two controlled players (first two on blue team)
        self._controlled: List[Player] = players[:2]

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
