def test_obs_size():
    import sys
    from pathlib import Path
    import types

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    gym_stub = types.ModuleType("gymnasium")
    gym_stub.Space = type("Space", (), {})
    gym_stub.spaces = types.SimpleNamespace(Box=gym_stub.Space)
    sys.modules.setdefault("gymnasium", gym_stub)
    sys.modules.setdefault("gym", gym_stub)
    rlgym = types.ModuleType("rlgym")
    api_module = types.ModuleType("rlgym.api")
    config_module = types.ModuleType("rlgym.api.config")
    config_module.ObsBuilder = type("ObsBuilder", (), {})
    api_module.config = config_module
    rocket_league_module = types.ModuleType("rlgym.rocket_league")
    common_values = types.ModuleType("rlgym.rocket_league.common_values")
    common_values.BOOST_LOCATIONS = []
    common_values.CEILING_Z = 0
    common_values.BALL_RADIUS = 0
    common_values.CAR_MAX_SPEED = 0
    rl_api_module = types.ModuleType("rlgym.rocket_league.api")
    rl_api_module.GameState = type("GameState", (), {})
    rocket_league_module.common_values = common_values
    rocket_league_module.api = rl_api_module
    rlgym.api = api_module
    rlgym.rocket_league = rocket_league_module
    sys.modules.setdefault("rlgym", rlgym)
    sys.modules.setdefault("rlgym.api", api_module)
    sys.modules.setdefault("rlgym.api.config", config_module)
    sys.modules.setdefault("rlgym.rocket_league", rocket_league_module)
    sys.modules.setdefault("rlgym.rocket_league.common_values", common_values)
    sys.modules.setdefault("rlgym.rocket_league.api", rl_api_module)

    from ModernObsBuilder import OBS_SIZE as MODERN_OBS
    from src.rlbot_integration.observation_adapter import OBS_SIZE as RLBOT_OBS

    assert MODERN_OBS == 107
    assert RLBOT_OBS == 107
    assert MODERN_OBS == RLBOT_OBS
