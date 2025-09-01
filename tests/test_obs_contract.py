def test_obs_size():
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.rlbot_integration.observation_adapter import OBS_SIZE as RLBOT_OBS

    assert RLBOT_OBS == 107
