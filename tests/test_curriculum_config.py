import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training.curriculum import CurriculumManager


def test_curriculum_config_fields():
    manager = CurriculumManager("configs/curriculum.yaml")

    # Ensure phases are loaded as expected mapping
    assert set(manager.phases.keys()) == {"bronze", "gold"}

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

    # Verify each phase includes all required fields
    for phase_name, phase in manager.phases.items():
        phase_dict = phase.__dict__
        missing = required_fields - phase_dict.keys()
        assert not missing, f"{phase_name} missing fields: {missing}"

