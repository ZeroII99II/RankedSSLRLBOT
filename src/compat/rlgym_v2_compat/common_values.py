"""Subset of RLGym common values needed for tests."""
import numpy as np

# Field and game constants
CEILING_Z = 2044.0
BALL_RADIUS = 92.75
CAR_MAX_SPEED = 2300.0
CAR_MAX_ANG_VEL = 5.5
BALL_MAX_SPEED = 6000.0
GOAL_HEIGHT = 642.775
BLUE_GOAL_CENTER = [0, -5120, 0]
BLUE_GOAL_BACK = [0, -6000, 0]
ORANGE_GOAL_CENTER = [0, 5120, 0]
ORANGE_GOAL_BACK = [0, 6000, 0]
ORANGE_TEAM = 1
SIDE_WALL_X = 4096.0
BACK_WALL_Y = 5120.0

# Simplified boost pad locations matching RLBot adapter
BOOST_LOCATIONS = [
    np.array([0, -4240, 70]), np.array([-1792, -4184, 70]), np.array([1792, -4184, 70]),
    np.array([-3072, -4096, 73]), np.array([3072, -4096, 73]), np.array([-940, -3308, 70]),
    np.array([940, -3308, 70]), np.array([0, 4240, 70]), np.array([-1792, 4184, 70]),
    np.array([1792, 4184, 70]), np.array([-3072, 4096, 73]), np.array([3072, 4096, 73]),
    np.array([-940, 3308, 70]), np.array([940, 3308, 70]), np.array([-2048, -1036, 70]),
    np.array([2048, -1036, 70]), np.array([-2048, 1036, 70]), np.array([2048, 1036, 70])
]
