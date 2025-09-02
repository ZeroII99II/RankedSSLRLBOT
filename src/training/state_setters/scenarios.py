"""Scenario helpers for constructing ``GameState`` instances.

These functions provide lightweight starting states for different training
situations.  Each helper accepts an ``np.random.Generator`` along with a base
``GameState`` and mutates it in-place to represent the scenario before
returning it.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

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

    kickoff_positions = [
        (-2048, -2560, BALL_RADIUS),
        (2048, -2560, BALL_RADIUS),
        (-2048, 2560, BALL_RADIUS),
        (2048, 2560, BALL_RADIUS),
    ]
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

    kickoff_state(rng, gs)
    x = SIDE_WALL_X - 200
    y = BACK_WALL_Y - 200
    x *= -1 if rng.random() > 0.5 else 1
    y *= -1 if rng.random() > 0.5 else 1

    ball = _zero_phys()
    ball.position = np.array([x, y, BALL_RADIUS], dtype=np.float32)
    vel = np.array([-np.sign(x) * 1000, -np.sign(y) * 1000, 0], dtype=np.float32)
    ball.linear_velocity = vel
    gs.ball = ball
    return gs


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

