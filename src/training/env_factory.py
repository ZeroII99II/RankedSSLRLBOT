from __future__ import annotations

"""Factory for constructing a minimal RLGym session."""

from typing import Callable, Dict

import numpy as np

from src.utils.gym_compat import gym
from src.rlbot_integration.observation_adapter import OBS_SIZE
from src.training.observers import ModernObsBuilder
from src.training.rewards import ModernRewardSystem
from src.training.state_setters import ModernStateSetter

from rlgym.rocket_league.match import Match
from rlgym.rocket_league.sim.rocketsim_engine import RocketSimEngine
from rlgym.rocket_league.done_conditions.goal_condition import GoalCondition
from rlgym.rocket_league.done_conditions.timeout_condition import TimeoutCondition

# Action schema: continuous and discrete controls
CONT_DIM = 5
DISC_DIM = 3


class RLGymSession(gym.Env):
    """Thin wrapper around :class:`rlgym.rocket_league.match.Match`."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self._obs_builder = ModernObsBuilder()
        self._reward_fn = ModernRewardSystem()
        self._state_setter = ModernStateSetter()
        self._match = Match(
            reward_function=self._reward_fn,
            obs_builder=self._obs_builder,
            state_setter=self._state_setter,
            simulator=RocketSimEngine(rlbot_delay=False),
            terminal_conditions=[GoalCondition()],
            truncation_conditions=[TimeoutCondition(5.0)],
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = gym.spaces.Dict(
            {
                "cont": gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(CONT_DIM,), dtype=np.float32
                ),
                "disc": gym.spaces.MultiBinary(DISC_DIM),
            }
        )

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        """Reset the underlying match."""
        obs = self._match.reset()
        return obs, {}

    def step(self, action: Dict[str, np.ndarray]):
        """Step the underlying match."""
        obs, reward, terminated, truncated, info = self._match.step(action)
        return obs, float(reward), bool(terminated), bool(truncated), info


def make_env() -> Callable[[], RLGymSession]:
    """Return a constructor for :class:`RLGymSession`."""

    def _init() -> RLGymSession:
        return RLGymSession()

    return _init
