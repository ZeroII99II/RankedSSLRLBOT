"""Training-side action transform matching RLBot adapter."""
from __future__ import annotations

import numpy as np

BTN_T = 0.5


def to_rlgym(a_cont: np.ndarray, a_disc: np.ndarray) -> np.ndarray:
    """Convert policy outputs to RLGym action array.

    Args:
        a_cont: Continuous actions in [-1, 1] with order
            [steer, throttle, pitch, yaw, roll].
        a_disc: Discrete action logits/probabilities for
            [jump, boost, handbrake].

    Returns:
        Array ordered as expected by RLGym's DefaultAction:
        [throttle, steer, pitch, yaw, roll, jump, boost, handbrake].
    """
    steer, throttle, pitch, yaw, roll = [float(x) for x in a_cont]
    jump = float(a_disc[0] > BTN_T)
    boost = float(a_disc[1] > BTN_T)
    handbrake = float(a_disc[2] > BTN_T)
    return np.array(
        [throttle, steer, pitch, yaw, roll, jump, boost, handbrake],
        dtype=np.float32,
    )
