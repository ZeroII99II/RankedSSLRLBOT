import sys, types
from pathlib import Path

import pytest

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

    env = trainer._create_environment()
    assert env is dummy_env
