"""Map policy outputs → RLBot SimpleControllerState."""
from __future__ import annotations
from rlbot.agents.base_agent import SimpleControllerState
import numpy as np

# Threshold for discrete buttons
BTN_T = 0.5


def to_controls(a_cont: np.ndarray, a_disc: np.ndarray) -> SimpleControllerState:
    # a_cont expected shape (5,) in [-1, 1]
    steer, throttle, pitch, yaw, roll = [float(x) for x in a_cont]
    # a_disc expected shape (3,) logits or probs → treat > 0.5 as pressed
    if a_disc.ndim > 0:
        jump = bool(a_disc[0] > BTN_T)
        boost = bool(a_disc[1] > BTN_T)
        handbrake = bool(a_disc[2] > BTN_T)
    else:
        jump = boost = handbrake = False

    ctrl = SimpleControllerState()
    ctrl.steer = max(-1.0, min(1.0, steer))
    ctrl.throttle = max(-1.0, min(1.0, throttle))
    ctrl.pitch = max(-1.0, min(1.0, pitch))
    ctrl.yaw = max(-1.0, min(1.0, yaw))
    ctrl.roll = max(-1.0, min(1.0, roll))
    ctrl.jump = jump
    ctrl.boost = boost
    ctrl.handbrake = handbrake
    return ctrl