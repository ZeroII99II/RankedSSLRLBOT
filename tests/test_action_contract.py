import numpy as np
import sys
from pathlib import Path
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Provide a minimal RLBot SimpleControllerState to satisfy controller_adapter
class SimpleControllerState:
    def __init__(self):
        self.steer = self.throttle = self.pitch = self.yaw = self.roll = 0.0
        self.jump = False
        self.boost = False
        self.handbrake = False


rlbot_mod = types.ModuleType("rlbot")
agents_mod = types.ModuleType("rlbot.agents")
base_agent_mod = types.ModuleType("rlbot.agents.base_agent")
base_agent_mod.SimpleControllerState = SimpleControllerState
sys.modules.setdefault("rlbot", rlbot_mod)
sys.modules.setdefault("rlbot.agents", agents_mod)
sys.modules.setdefault("rlbot.agents.base_agent", base_agent_mod)

from src.training.action_adapter import to_rlgym
from src.rlbot_integration.controller_adapter import to_controls


def ctrl_to_array(ctrl) -> np.ndarray:
    return np.array(
        [
            ctrl.throttle,
            ctrl.steer,
            ctrl.pitch,
            ctrl.yaw,
            ctrl.roll,
            float(ctrl.jump),
            float(ctrl.boost),
            float(ctrl.handbrake),
        ],
        dtype=np.float32,
    )


def test_action_transform():
    rng = np.random.default_rng(0)
    for _ in range(10):
        a_cont = rng.uniform(-1, 1, 5)
        a_disc = rng.uniform(-1, 1, 3)
        env_action = to_rlgym(a_cont, a_disc)
        ctrl = to_controls(a_cont, a_disc)
        ctrl_array = ctrl_to_array(ctrl)
        assert np.allclose(env_action, ctrl_array)
