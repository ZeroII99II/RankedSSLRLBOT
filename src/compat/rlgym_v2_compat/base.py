"""Minimal stand‑ins for RLGym v2 base classes."""
from __future__ import annotations
from typing import Any

import numpy as np


class ObsBuilder:
    """Base observation builder interface."""

    def reset(self, initial_state: Any) -> None:
        pass

    def build_obs(self, player: Any, state: Any, previous_action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_obs_space(self) -> Any:
        raise NotImplementedError


class RewardFunction:
    """Base reward function interface."""

    def reset(self, initial_state: Any) -> None:
        pass

    def get_reward(self, player: Any, state: Any, previous_action: np.ndarray) -> float:
        raise NotImplementedError


class StateSetter:
    """Base state setter interface."""

    def reset(self, initial_state: Any) -> None:
        pass

    def set_state(self, state_wrapper: Any) -> None:
        raise NotImplementedError


class TerminalCondition:
    """Base terminal condition interface."""

    def reset(self, initial_state: Any) -> None:
        pass

    def is_terminal(self, state: Any) -> bool:
        raise NotImplementedError
