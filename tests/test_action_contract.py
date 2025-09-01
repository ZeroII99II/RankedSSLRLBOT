def test_action_shapes():
    import sys, types
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    class SimpleControllerState:
        pass

    rlbot_mod = types.ModuleType("rlbot")
    agents_mod = types.ModuleType("rlbot.agents")
    base_agent_mod = types.ModuleType("rlbot.agents.base_agent")
    base_agent_mod.SimpleControllerState = SimpleControllerState
    sys.modules.setdefault("rlbot", rlbot_mod)
    sys.modules.setdefault("rlbot.agents", agents_mod)
    sys.modules.setdefault("rlbot.agents.base_agent", base_agent_mod)

    from src.rlbot_integration.controller_adapter import CONT_DIM, DISC_DIM

    assert CONT_DIM == 5 and DISC_DIM == 3
