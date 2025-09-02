from __future__ import annotations

from typing import Dict, Any, Callable, List

import numpy as np
from rlgym.api import RLGym
from rlgym.api.config import (
    ActionParser as APIActionParser,
    TransitionEngine as APITransitionEngine,
    RewardFunction as APIRewardFunction,
    ObsBuilder as APIObsBuilder,
)
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators.fixed_team_size_mutator import (
    FixedTeamSizeMutator,
)
from rlgym.rocket_league.state_mutators.kickoff_mutator import KickoffMutator
from rlgym.rocket_league.state_mutators.mutator_sequence import MutatorSequence

from src.utils.gym_compat import gym
from src.rlbot_integration.observation_adapter import OBS_SIZE
from .action_adapter import to_rlgym


# Action schema: continuous and discrete controls
CONT_DIM = 5
DISC_DIM = 3


class SimpleActionParser(
    APIActionParser[int, Dict[str, np.ndarray], np.ndarray, GameState, gym.spaces.Dict]
):
    """Convert environment actions to RocketSim controls."""

    def __init__(self, action_space: gym.spaces.Dict):
        self._action_space = action_space

    def get_action_space(self, agent: int) -> gym.spaces.Dict:
        return self._action_space

    def parse_actions(
        self,
        actions: Dict[int, Dict[str, np.ndarray]],
        state: GameState,
        shared_info: Dict[str, Any],
    ) -> Dict[int, np.ndarray]:
        engine_actions: Dict[int, np.ndarray] = {}
        for agent, act in actions.items():
            a_cont = np.clip(act["cont"].astype(np.float32), -1.0, 1.0)
            a_disc = act["disc"].astype(np.float32).clip(0, 1)
            engine_actions[agent] = to_rlgym(a_cont, a_disc)[None, :]
        return engine_actions

    def reset(
        self, agents: List[int], initial_state: GameState, shared_info: Dict[str, Any]
    ) -> None:
        pass


class DummyObsBuilder(
    APIObsBuilder[int, np.ndarray, GameState, gym.spaces.Box]
):
    """Observation builder returning zero vectors of fixed size."""

    def __init__(self):
        self._space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )

    def get_obs_space(self, agent: int) -> gym.spaces.Box:
        return self._space

    def reset(
        self, agents: List[int], initial_state: GameState, shared_info: Dict[str, Any]
    ) -> None:
        pass

    def build_obs(
        self, agents: List[int], state: GameState, shared_info: Dict[str, Any]
    ) -> Dict[int, np.ndarray]:
        return {a: np.zeros(OBS_SIZE, dtype=np.float32) for a in agents}


class DummyReward(APIRewardFunction[int, GameState, float]):
    """Simple reward function producing zeros."""

    def reset(
        self, agents: List[int], initial_state: GameState, shared_info: Dict[str, Any]
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[int],
        state: GameState,
        is_terminated: Dict[int, bool],
        is_truncated: Dict[int, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[int, float]:
        return {a: 0.0 for a in agents}


class RL2v2Env(gym.Env):
    """Gymnasium environment backed by an RLGym 2.0 RocketSim engine."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 42):
        super().__init__()
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
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self._obs_builder = DummyObsBuilder()
        self._reward_fn = DummyReward()
        self._state_mutator = MutatorSequence(
            FixedTeamSizeMutator(blue_size=2, orange_size=2), KickoffMutator()
        )
        self._action_parser = SimpleActionParser(self.action_space)
        self._engine: APITransitionEngine[int, GameState, np.ndarray] = RocketSimEngine()

        self._match = RLGym(
            state_mutator=self._state_mutator,
            obs_builder=self._obs_builder,
            action_parser=self._action_parser,
            reward_fn=self._reward_fn,
            transition_engine=self._engine,
        )

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        obs_dict = self._match.reset()
        first_agent = self._engine.agents[0]
        obs_vec = obs_dict[first_agent]
        return obs_vec.astype(np.float32), {}

    def step(self, action: Dict[str, np.ndarray]):
        a_cont = np.clip(action["cont"].astype(np.float32), -1.0, 1.0)
        a_disc = action["disc"].astype(np.float32).clip(0, 1)
        rl_action = {
            agent: {"cont": a_cont, "disc": a_disc}
            for agent in self._engine.agents
        }

        obs, rewards, terminated, truncated = self._match.step(rl_action)

        obs_vec = obs[self._engine.agents[0]]
        reward = float(np.mean(list(rewards.values())))
        done = any(terminated.values())
        trunc = any(truncated.values())
        return obs_vec.astype(np.float32), reward, bool(done), bool(trunc), {}


def make_env(seed: int = 42) -> Callable[[], RL2v2Env]:
    def _thunk() -> RL2v2Env:
        return RL2v2Env(seed=seed)

    return _thunk

