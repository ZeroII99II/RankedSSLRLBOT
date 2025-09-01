import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training.observers import ModernObsBuilder
from src.rlbot_integration.observation_adapter import build_observation


# --- Stubs for RLGym-style objects -------------------------------------------------
class CarData:
    def __init__(self):
        self.position = np.zeros(3)
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

    def forward(self):
        return np.array([1.0, 0.0, 0.0])

    def up(self):
        return np.array([0.0, 1.0, 0.0])


class Player:
    def __init__(self):
        self.car_data = CarData()
        self.team_num = 0
        self.boost_amount = 0.0
        self.on_ground = True
        self.has_flip = True
        self.has_jump = True
        self.ball_touched = False


class Ball:
    def __init__(self):
        self.position = np.zeros(3)
        self.linear_velocity = np.zeros(3)


class GameState:
    def __init__(self):
        self.players = [Player()]
        self.ball = Ball()
        self.blue_score = 0
        self.orange_score = 0
        self.game_seconds_remaining = 300.0
        self.is_overtime = False
        self.is_kickoff_pause = False
        self.boost_pads = []


# --- RLBot packet stubs -----------------------------------------------------------
class Vec3:
    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = x, y, z


class Rotator:
    def __init__(self):
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0


class Physics:
    def __init__(self):
        self.location = Vec3()
        self.velocity = Vec3()
        self.angular_velocity = Vec3()
        self.rotation = Rotator()


class CarPacket:
    def __init__(self):
        self.physics = Physics()
        self.team = 0
        self.boost = 0.0


class BallPacket:
    def __init__(self):
        self.physics = Physics()


class GameInfo:
    def __init__(self):
        self.blue_score = 0
        self.orange_score = 0
        self.seconds_remaining = 300.0
        self.is_overtime = False
        self.is_kickoff_pause = False


class Packet:
    def __init__(self):
        self.game_cars = [CarPacket()]
        self.game_ball = BallPacket()
        self.game_info = GameInfo()


# ----------------------------------------------------------------------------------
def test_observation_alignment():
    builder = ModernObsBuilder()
    state = GameState()
    obs_train = builder.build_obs(state.players[0], state, np.zeros(8))

    packet = Packet()
    obs_rlbot = build_observation(packet)

    assert len(obs_train) == 107
    assert obs_train.shape == obs_rlbot.shape
    # Verify that self-car features (first 20 indices) align exactly
    assert np.allclose(obs_train[:20], obs_rlbot[:20])
