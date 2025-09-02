"""Lightweight compatibility layer mimicking minimal pieces of RLGym v2.

Provides base classes used by training code without requiring the full
rlgym package to be installed during testing.  Only functionality needed by
our tests is implemented.
"""
from .base import ObsBuilder, RewardFunction, StateSetter, TerminalCondition
from .game_state import GameState, PlayerData, CarData, BoostPad, BallData
from . import common_values

__all__ = [
    "ObsBuilder",
    "RewardFunction",
    "StateSetter",
    "TerminalCondition",
    "GameState",
    "PlayerData",
    "CarData",
    "BoostPad",
    "BallData",
    "common_values",
]
