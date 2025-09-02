"""Scenario helpers for constructing :class:`GameState` instances.

These functions provide lightweight starting states for different training
situations.  Each helper accepts an ``np.random.Generator`` to ensure
deterministic sampling when seeded.  The resulting :class:`GameState` objects
use the real RLGym v2 data structures.

situations.  Each helper accepts an ``np.random.Generator`` along with a base
``GameState`` and mutates it in-place to represent the scenario before
returning it.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from rlgym.rocket_league.api import GameState, Car, PhysicsObject, GameConfig
from rlgym.rocket_league.common_values import (
    BOOST_LOCATIONS,
    BALL_RADIUS,
    SIDE_WALL_X,
    BACK_WALL_Y,
)


def _empty_state() -> GameState:
    """Create a blank ``GameState`` with default configuration."""
    gs = GameState()
    gs.tick_count = 0
    gs.goal_scored = False
    cfg = GameConfig()
    cfg.gravity = 1
    cfg.boost_consumption = 1
    cfg.dodge_deadzone = 0.5
    gs.config = cfg
    gs.cars = {}
    gs.boost_pad_timers = np.zeros(len(BOOST_LOCATIONS), dtype=np.float32)
    gs.ball = PhysicsObject()
    return gs


def _car(
    rng: np.random.Generator,
    team: int,
    position: np.ndarray,
    lin_vel: np.ndarray | None = None,
    ang_vel: np.ndarray | None = None,
    rot: np.ndarray | None = None,
    boost: float = 100.0,
) -> Car:
    """Construct a minimal ``Car`` with the provided parameters."""
    car = Car()
    car.team_num = team
    car.hitbox_type = 0
    car.ball_touches = 0
    car.bump_victim_id = None
    car.demo_respawn_timer = 0.0
    car.wheels_with_contact = (True, True, True, True)
    car.supersonic_time = 0.0
    car.boost_amount = boost
    car.boost_active_time = 0.0
    car.handbrake = 0.0
    car.is_jumping = False
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
    phys.position = position.astype(np.float32)
    phys.linear_velocity = (
        lin_vel.astype(np.float32) if lin_vel is not None else np.zeros(3, dtype=np.float32)
    )
    phys.angular_velocity = (
        ang_vel.astype(np.float32) if ang_vel is not None else np.zeros(3, dtype=np.float32)
    )
    phys.euler_angles = (
        rot.astype(np.float32) if rot is not None else np.zeros(3, dtype=np.float32)
    )
    car.physics = phys
    return car

from rlgym.rocket_league.api import GameState, PhysicsObject
from rlgym.rocket_league.common_values import (
    BALL_RADIUS,
    SIDE_WALL_X,
    BACK_WALL_Y,
    BOOST_LOCATIONS,
)


def _zero_phys() -> PhysicsObject:
    phys = PhysicsObject()
    phys.position = np.zeros(3, dtype=np.float32)
    phys.linear_velocity = np.zeros(3, dtype=np.float32)
    phys.angular_velocity = np.zeros(3, dtype=np.float32)
    phys.euler_angles = np.zeros(3, dtype=np.float32)
    return phys


def kickoff_state(rng: np.random.Generator, gs: GameState) -> GameState:
    """Ball at centre field with cars in standard kickoff positions."""
    gs = _empty_state()


 
    kickoff_positions = [
        (-2048, -2560, BALL_RADIUS),
        (2048, -2560, BALL_RADIUS),
        (-2048, 2560, BALL_RADIUS),
        (2048, 2560, BALL_RADIUS),
    ]
    for i, pos in enumerate(kickoff_positions):
        team = 0 if i < 2 else 1
        gs.cars[i] = _car(rng, team, np.array(pos, dtype=np.float32))

    ball = PhysicsObject()
    ball.position = np.array([0, 0, BALL_RADIUS], dtype=np.float32)
    ball.linear_velocity = np.zeros(3, dtype=np.float32)
    ball.angular_velocity = np.zeros(3, dtype=np.float32)
    ball.euler_angles = np.zeros(3, dtype=np.float32)
    gs.ball = ball

    for i, (x, y, z) in enumerate(kickoff_positions):
        car = gs.cars[i]
        car.physics = _zero_phys()
        car.physics.position = np.array([x, y, z], dtype=np.float32)
        car.team_num = 0 if i < 2 else 1
        car.boost_amount = 100.0

    ball = _zero_phys()
    ball.position = np.array([0, 0, BALL_RADIUS], dtype=np.float32)
    gs.ball = ball
    gs.boost_pad_timers = np.zeros(len(BOOST_LOCATIONS), dtype=np.float32)
    return gs


def corner_shot_state(rng: np.random.Generator, gs: GameState) -> GameState:
    """Ball positioned in a corner moving toward the opposite goal."""
    gs = _empty_state()
    for i in range(4):
        team = 0 if i < 2 else 1
        pos = rng.uniform(-1000, 1000, size=3)
        gs.cars[i] = _car(rng, team, pos)

    ball = PhysicsObject()


    kickoff_state(rng, gs)
    x = SIDE_WALL_X - 200
    y = BACK_WALL_Y - 200
    x *= -1 if rng.random() > 0.5 else 1
    y *= -1 if rng.random() > 0.5 else 1
    ball.position = np.array([x, y, BALL_RADIUS], dtype=np.float32)
    vel = np.array([-np.sign(x) * 1000, -np.sign(y) * 1000, 0], dtype=np.float32)
    ball.linear_velocity = vel
    ball.angular_velocity = np.zeros(3, dtype=np.float32)
    ball.euler_angles = np.zeros(3, dtype=np.float32)


    ball = _zero_phys()
    ball.position = np.array([x, y, BALL_RADIUS], dtype=np.float32)
    vel = np.array([-np.sign(x) * 1000, -np.sign(y) * 1000, 0], dtype=np.float32)
    ball.linear_velocity = vel
    gs.ball = ball
    return gs


def random_state(rng: np.random.Generator) -> GameState:
    """Fully randomised state."""
    gs = _empty_state()
    for i in range(4):
        team = 0 if i < 2 else 1
        pos = rng.uniform(-1000, 1000, size=3)
        lin = rng.uniform(-500, 500, size=3)
        ang = rng.uniform(-5, 5, size=3)
        rot = rng.uniform(-np.pi, np.pi, size=3)
        boost = float(rng.uniform(0, 100))
        gs.cars[i] = _car(rng, team, pos, lin, ang, rot, boost)

    ball = PhysicsObject()
    ball.position = rng.uniform(-1000, 1000, size=3).astype(np.float32)
    ball.linear_velocity = rng.uniform(-500, 500, size=3).astype(np.float32)
    ball.angular_velocity = rng.uniform(-5, 5, size=3).astype(np.float32)
    ball.euler_angles = rng.uniform(-np.pi, np.pi, size=3).astype(np.float32)
    gs.ball = ball

def random_state(rng: np.random.Generator, gs: GameState) -> GameState:
    """Fully random state similar to the previous implementation."""

    for i, car in gs.cars.items():
        car.physics = _zero_phys()
        car.physics.position = rng.uniform(-1000, 1000, size=3).astype(np.float32)
        car.physics.linear_velocity = rng.uniform(-500, 500, size=3).astype(np.float32)
        car.physics.angular_velocity = rng.uniform(-5, 5, size=3).astype(np.float32)
        car.physics.euler_angles = rng.uniform(-np.pi, np.pi, size=3).astype(np.float32)
        car.team_num = 0 if i < 2 else 1
        car.boost_amount = float(rng.uniform(0, 100))

    ball = _zero_phys()
    ball.position = rng.uniform(-1000, 1000, size=3).astype(np.float32)
    ball.linear_velocity = rng.uniform(-500, 500, size=3).astype(np.float32)
    gs.ball = ball
    gs.boost_pad_timers = np.zeros(len(BOOST_LOCATIONS), dtype=np.float32)
 
    return gs


# Mapping used by the environment to look up scenarios by name
SCENARIOS: Dict[str, Callable[[np.random.Generator, GameState], GameState]] = {
    "kickoff": kickoff_state,
    "corner_shot": corner_shot_state,
    "random": random_state,
}

