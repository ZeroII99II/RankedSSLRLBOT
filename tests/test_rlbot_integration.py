import os
import sys
import types
import importlib.util
from pathlib import Path
# Add repo root to PYTHONPATH for src imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Provide minimal 'imp' module for flatbuffers on Python 3.12+
if 'imp' not in sys.modules:
    imp = types.ModuleType('imp')

    def find_module(name, path=None):
        spec = importlib.util.find_spec(name)
        if spec is None:
            raise ImportError(name)
        return None, spec.origin, ('', '', 0)

    imp.find_module = find_module
    sys.modules['imp'] = imp

rlbot_mod = types.ModuleType("rlbot")
agents_mod = types.ModuleType("rlbot.agents")
base_agent_mod = types.ModuleType("rlbot.agents.base_agent")

class SimpleControllerState:
    def __init__(self):
        self.throttle = 0.0
        self.steer = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.jump = False
        self.boost = False
        self.handbrake = False

class BaseAgent:
    def __init__(self, *args, **kwargs):
        self.logger = types.SimpleNamespace(info=lambda *a, **k: None,
                                            warn=lambda *a, **k: None,
                                            error=lambda *a, **k: None)

base_agent_mod.SimpleControllerState = SimpleControllerState
base_agent_mod.BaseAgent = BaseAgent
agents_mod.base_agent = base_agent_mod
rlbot_mod.agents = agents_mod

sys.modules.setdefault("rlbot", rlbot_mod)
sys.modules.setdefault("rlbot.agents", agents_mod)
sys.modules.setdefault("rlbot.agents.base_agent", base_agent_mod)

import torch
from rlbot.agents.base_agent import SimpleControllerState
from src.inference import export as export_mod


class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class Rotator:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll


class Physics:
    def __init__(self):
        self.location = Vec3()
        self.velocity = Vec3()
        self.angular_velocity = Vec3()
        self.rotation = Rotator()


class Player:
    def __init__(self, team=0):
        self.team = team
        self.physics = Physics()
        self.boost = 0
        self.has_wheel_contact = True
        self.has_flip = True
        self.has_jump = True


class Ball:
    def __init__(self):
        self.physics = Physics()


class GameInfo:
    def __init__(self):
        self.blue_score = 0
        self.orange_score = 0
        self.seconds_remaining = 300.0
        self.is_overtime = False
        self.is_kickoff_pause = False


class FakePacket:
    def __init__(self):
        self.game_cars = [Player(team=0)]
        self.game_ball = Ball()
        self.game_info = GameInfo()


from src.training import policy as policy_mod

def _export_policy(tmp_path: Path) -> Path:
    def build_policy(obs_dim, cont, disc):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(obs_dim, cont + disc)
            def forward(self, x):
                logits = self.linear(x)
                a_cont = torch.tanh(logits[:, :cont])
                a_disc = logits[:, cont:]
                return a_cont, a_disc
        return Net()
    policy_mod.build_policy = build_policy

    ckpt = tmp_path / "ckpt.pth"
    torch.save({}, ckpt)  # minimal state dict
    out = tmp_path / "model.ts"
    argv = ["export", "--ckpt", str(ckpt), "--out", str(out)]
    old_argv = sys.argv.copy()
    sys.argv = argv
    try:
        export_mod.main()
    finally:
        sys.argv = old_argv
    return out
def test_rlbot_integration(tmp_path, monkeypatch):
    model_path = _export_policy(tmp_path)
    monkeypatch.setenv('SSL_POLICY_PATH', str(model_path))
    from src.rlbot_integration.bot import SSLBot

    bot = SSLBot('test', 0, 0)
    bot.initialize_agent()
    assert bot.model is not None

    packet = FakePacket()
    controls = bot.get_output(packet)
    assert isinstance(controls, SimpleControllerState)
