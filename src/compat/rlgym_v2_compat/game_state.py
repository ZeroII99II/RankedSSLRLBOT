from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass(eq=False)
class BallData:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def set_pos(self, x: float, y: float, z: float) -> None:
        self.position[:] = (x, y, z)

    def set_lin_vel(self, x: float, y: float, z: float) -> None:
        self.linear_velocity[:] = (x, y, z)


@dataclass(eq=False)
class CarData:
    team_num: int
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0

    def set_pos(self, x: float, y: float, z: float) -> None:
        self.position[:] = (x, y, z)

    def set_lin_vel(self, x: float, y: float, z: float) -> None:
        self.linear_velocity[:] = (x, y, z)

    def set_ang_vel(self, x: float, y: float, z: float) -> None:
        self.angular_velocity[:] = (x, y, z)

    def set_rot(self, pitch: float, yaw: float, roll: float) -> None:
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

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


@dataclass(eq=False)
class PlayerData:
    car_data: CarData
    team_num: int
    boost_amount: float = 0.0
    on_ground: bool = True
    has_flip: bool = True
    has_jump: bool = True
    ball_touched: bool = False
    is_demoed: bool = False
    match_demolishes: int = 0


@dataclass
class BoostPad:
    position: np.ndarray
    is_active: bool = True


@dataclass
class GameState:
    ball: BallData
    players: List[PlayerData]
    boost_pads: List[BoostPad]
    blue_score: int = 0
    orange_score: int = 0
    game_seconds_remaining: float = 300.0
    is_overtime: bool = False
    is_kickoff_pause: bool = False
