"""Environment factory for training agents.

This module exposes a thin Gym-compatible wrapper around the new
`rlgym.rocket_league` API and a helper :func:`make_env` for constructing
seeded instances of that environment.  The observation space is fixed at
107 dimensions and the action space is split into 5 continuous controls and
3 discrete binary actions as expected by the rest of the project.
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Environment description
# ---------------------------------------------------------------------------
# Action schema: 5 continuous + 3 discrete controls
CONT_DIM = 5
DISC_DIM = 3


class RLMatchEnv(gym.Env):
    """Minimal Rocket League environment using :class:`rlgym`.

    Parameters
    ----------
    seed:
        Random seed forwarded to the underlying action space.
    render:
        Whether to enable rendering (currently unused but kept for parity with
        ``RL2v2Env`` from the real project).
    num_players_per_team:
        Number of players per team.  Stored for reference so training code can
        adapt behaviour based on team size.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int = 0,
        render: bool = False,
        num_players_per_team: int = 1,
    ) -> None:
        super().__init__()

        self.num_players_per_team = num_players_per_team

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
        self.action_space.seed(seed)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        """Reset the underlying match and return the initial observation."""
        if seed is not None:
            self.action_space.seed(seed)
        obs = self._match.reset()
        return obs, {}

    def step(self, action: Dict[str, np.ndarray]):
        """Step the underlying match."""
        obs, reward, terminated, truncated, info = self._match.step(action)
        return obs, float(reward), bool(terminated), bool(truncated), info


# Some tests import ``RL2v2Env`` by name.  The real project uses a class with
# that name which is functionally equivalent to ``RLMatchEnv`` here, so we
# provide it as an alias for compatibility.
RL2v2Env = RLMatchEnv


def make_env(
    seed: int = 42, render: bool = False, team_size: int = 2
) -> Callable[[], RLMatchEnv]:
    """Return a thunk creating a seeded :class:`RLMatchEnv` instance.

    Parameters
    ----------
    seed:
        Random seed for the environment and its action space.
    render:
        Flag indicating whether rendering should be enabled.  Currently kept
        for API compatibility and forwarded to ``RLMatchEnv``.
    team_size:
        Number of players per team; forwarded to ``RLMatchEnv`` so training
        scripts can easily switch between 1v1 and 2v2 setups.
    """

    def _thunk() -> RLMatchEnv:
        return RLMatchEnv(
            seed=seed, render=render, num_players_per_team=team_size
        )

    return _thunk


__all__ = [
    "RLMatchEnv",
    "RL2v2Env",
    "make_env",
    "CONT_DIM",
    "DISC_DIM",
]

