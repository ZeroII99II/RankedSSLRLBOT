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

        # progression_gates should be a flat mapping without nested structures
        gates = phase.progression_gates
        assert isinstance(gates, dict)
        assert "min_games" in gates
        assert all(not isinstance(v, dict) for v in gates.values()), "progression_gates must be flat"


def test_can_progress_when_thresholds_met():
    manager = CurriculumManager("configs/curriculum.yaml")

    # Focus on bronze phase for a simple check
    manager.current_phase = "bronze"
    bronze = manager.get_current_phase()

    # Satisfy both min_training_steps and min_games gates
    manager.training_steps = bronze.min_training_steps
    manager.games_played = bronze.progression_gates["min_games"]

    # Provide evaluation metrics that meet or exceed thresholds
    eval_metrics = {
        "on_target_pct": bronze.progression_gates["on_target_pct"],
        "recoveries_per_min": bronze.progression_gates["recoveries_per_min"],
    }

    assert manager.can_progress(eval_metrics), "Bronze phase should allow progression when thresholds are met"

