import yaml


def test_curriculum_yaml_structure():
    """Ensure each curriculum phase has all required fields."""
    with open("configs/curriculum.yaml", "r") as f:
        data = yaml.safe_load(f)

    phases = data.get("phases")
    assert phases is not None, "Missing 'phases' section"
    assert set(phases.keys()) == {"bronze", "gold"}

    required_fields = {
        "description",
        "reward_weights",
        "scenario_weights",
        "opponent_mix",
        "progression_gates",
        "min_training_steps",
        "max_training_steps",
        "eval_metrics",
    }

    for name, cfg in phases.items():
        missing = required_fields - cfg.keys()
        assert not missing, f"{name} missing fields: {missing}"

