import sys, types
from pathlib import Path

import pytest
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))


def test_create_environment(monkeypatch):
    # Create dummy rlgym module hierarchy
    dummy_env = object()

    def dummy_make(**kwargs):
        return dummy_env

    rlgym_mod = types.ModuleType("rlgym")
    rlgym_mod.make = dummy_make

    utils_mod = types.ModuleType("rlgym.utils")
    tc_mod = types.ModuleType("rlgym.utils.terminal_conditions")
    ap_mod = types.ModuleType("rlgym.utils.action_parsers")

    common_conditions = types.SimpleNamespace(
        TimeoutCondition=lambda *a, **k: None,
        GoalScoredCondition=lambda *a, **k: None,
    )
    tc_mod.common_conditions = common_conditions
    ap_mod.DefaultAction = object

    sys.modules["rlgym"] = rlgym_mod
    sys.modules["rlgym.utils"] = utils_mod
    sys.modules["rlgym.utils.terminal_conditions"] = tc_mod
    sys.modules["rlgym.utils.action_parsers"] = ap_mod

    rocket_mod = types.ModuleType("rlgym.rocket_league")
    common_values_mod = types.ModuleType("rlgym.rocket_league.common_values")
    common_values_mod.CAR_MAX_SPEED = 0
    common_values_mod.BALL_MAX_SPEED = 0
    common_values_mod.CEILING_Z = 0
    common_values_mod.BALL_RADIUS = 0
    common_values_mod.SIDE_WALL_X = 0
    common_values_mod.BACK_WALL_Y = 0
    common_values_mod.CAR_MAX_ANG_VEL = 0
    rocket_mod.common_values = common_values_mod
    sys.modules["rlgym.rocket_league"] = rocket_mod
    sys.modules["rlgym.rocket_league.common_values"] = common_values_mod

    # Stub torch modules
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False, get_device_name=lambda *args: "cpu")
    torch_mod.device = object
    torch_mod.Tensor = object
    nn_mod = types.ModuleType("torch.nn")
    class Module:
        pass
    nn_mod.Module = Module
    optim_mod = types.ModuleType("torch.optim")
    optim_mod.Adam = object
    tb_mod = types.ModuleType("torch.utils.tensorboard")
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
    tb_mod.SummaryWriter = SummaryWriter
    torch_mod.nn = nn_mod
    torch_mod.optim = optim_mod
    torch_utils = types.ModuleType("torch.utils")
    torch_utils.tensorboard = tb_mod
    sys.modules.update({
        "torch": torch_mod,
        "torch.nn": nn_mod,
        "torch.optim": optim_mod,
        "torch.utils": torch_utils,
        "torch.utils.tensorboard": tb_mod,
    })

    # Stub policy module
    policy_mod = types.ModuleType("src.training.policy")
    class SSLPolicy:
        pass
    class SSLCritic:
        pass
    def create_ssl_policy(cfg):
        return SSLPolicy()
    def create_ssl_critic(cfg):
        return SSLCritic()
    policy_mod.SSLPolicy = SSLPolicy
    policy_mod.SSLCritic = SSLCritic
    policy_mod.create_ssl_policy = create_ssl_policy
    policy_mod.create_ssl_critic = create_ssl_critic
    sys.modules["src.training.policy"] = policy_mod

    # Stub rich library modules
    rich_mod = types.ModuleType("rich")
    console_mod = types.ModuleType("rich.console")
    console_mod.Console = object
    progress_mod = types.ModuleType("rich.progress")
    progress_mod.Progress = object
    progress_mod.SpinnerColumn = object
    progress_mod.TextColumn = object
    progress_mod.BarColumn = object
    progress_mod.TimeElapsedColumn = object
    table_mod = types.ModuleType("rich.table")
    table_mod.Table = object
    panel_mod = types.ModuleType("rich.panel")
    panel_mod.Panel = object

    sys.modules.update({
        "rich": rich_mod,
        "rich.console": console_mod,
        "rich.progress": progress_mod,
        "rich.table": table_mod,
        "rich.panel": panel_mod,
    })

    # Minimal gymnasium stub
    gym_mod = types.ModuleType("gymnasium")
    gym_mod.Space = type("Space", (), {})
    gym_mod.spaces = types.SimpleNamespace(Box=object)
    sys.modules["gymnasium"] = gym_mod

    import src.training.state_setters as state_setters_mod
    class DummySetter:
        def __init__(self, *args, **kwargs):
            pass
    state_setters_mod.SSLStateSetter = DummySetter

    if 'src.training.train' in sys.modules:
        del sys.modules['src.training.train']
    from src.training.train import PPOTrainer

    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = {
        "env": {
            "team_size": 1,
            "tick_skip": 8,
            "use_injector": False,
            "self_play": False,
            "spawn_opponents": False,
        }
    }
    trainer.curriculum = types.SimpleNamespace(
        get_current_phase=lambda: types.SimpleNamespace(name="test")
    )
    trainer.action_parser = object()
    trainer.np_rng = np.random.default_rng(0)

    env = trainer._create_environment()
    assert env is dummy_env
