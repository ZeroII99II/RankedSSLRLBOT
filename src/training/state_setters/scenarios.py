"""Scenario helpers for constructing ``GameState`` instances.

These functions provide lightweight starting states for different training
situations.  Each helper accepts an ``np.random.Generator`` to ensure
deterministic sampling when seeded.  The resulting ``GameState`` objects are
compatible with the simplified compatibility layer used in tests.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from src.compat.rlgym_v2_compat.game_state import (
    GameState,
    PlayerData,
    CarData,
    BallData,
    BoostPad,
)
from src.compat.rlgym_v2_compat import common_values


def _boost_pads() -> list[BoostPad]:
    """Create boost pad instances from static locations."""
    return [
        BoostPad(position=loc.astype(np.float32))
        for loc in common_values.BOOST_LOCATIONS
    ]


def _basic_players(rng: np.random.Generator) -> list[PlayerData]:
    """Return four players placed randomly on the field."""
    players: list[PlayerData] = []
    for i in range(4):
        team = 0 if i < 2 else 1
        car = CarData(team_num=team)
        car.set_pos(*rng.uniform(-1000, 1000, size=3))
        car.set_lin_vel(0, 0, 0)
        car.set_ang_vel(0, 0, 0)
        car.set_rot(0, 0, 0)
        players.append(PlayerData(car_data=car, team_num=team, boost_amount=100.0))
    return players


def kickoff_state(rng: np.random.Generator) -> GameState:
    """Ball at centre field with cars in standard kickoff positions."""
    players = _basic_players(rng)
    kickoff_positions = [
        (-2048, -2560, common_values.BALL_RADIUS),
        (2048, -2560, common_values.BALL_RADIUS),
        (-2048, 2560, common_values.BALL_RADIUS),
        (2048, 2560, common_values.BALL_RADIUS),
    ]
    for player, (x, y, z) in zip(players, kickoff_positions):
        player.car_data.set_pos(x, y, z)

    ball = BallData()
    ball.set_pos(0, 0, common_values.BALL_RADIUS)
    ball.set_lin_vel(0, 0, 0)
    return GameState(ball=ball, players=players, boost_pads=_boost_pads())


def corner_shot_state(rng: np.random.Generator) -> GameState:
    """Ball positioned in a corner moving toward the opposite goal."""
    players = _basic_players(rng)

    ball = BallData()
    x = common_values.SIDE_WALL_X - 200
    y = common_values.BACK_WALL_Y - 200
    x *= -1 if rng.random() > 0.5 else 1
    y *= -1 if rng.random() > 0.5 else 1
    ball.set_pos(x, y, common_values.BALL_RADIUS)
    vel = np.array([-np.sign(x) * 1000, -np.sign(y) * 1000, 0], dtype=np.float32)
    ball.set_lin_vel(*vel)

    return GameState(ball=ball, players=players, boost_pads=_boost_pads())


def random_state(rng: np.random.Generator) -> GameState:
    """Fully random state similar to the previous implementation."""
    players: list[PlayerData] = []
    for i in range(4):
        team = 0 if i < 2 else 1
        car = CarData(team_num=team)
        car.set_pos(*rng.uniform(-1000, 1000, size=3))
        car.set_lin_vel(*rng.uniform(-500, 500, size=3))
        car.set_ang_vel(*rng.uniform(-5, 5, size=3))
        car.set_rot(*rng.uniform(-np.pi, np.pi, size=3))
        players.append(
            PlayerData(
                car_data=car,
                team_num=team,
                boost_amount=float(rng.uniform(0, 100)),
            )
        )

    ball = BallData()
    ball.set_pos(*rng.uniform(-1000, 1000, size=3))
    ball.set_lin_vel(*rng.uniform(-500, 500, size=3))

    return GameState(ball=ball, players=players, boost_pads=_boost_pads())


# Mapping used by the environment to look up scenarios by name
SCENARIOS: Dict[str, Callable[[np.random.Generator], GameState]] = {
    "kickoff": kickoff_state,
    "corner_shot": corner_shot_state,
    "random": random_state,
}

