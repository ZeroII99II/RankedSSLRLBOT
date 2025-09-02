from __future__ import annotations

from typing import Dict, Any, Callable, List

import numpy as np
from rlgym.api import RLGym
from rlgym.api.config import (
    ActionParser as APIActionParser,
    TransitionEngine as APITransitionEngine,
    StateMutator as APIStateMutator,
    RewardFunction as APIRewardFunction,
    ObsBuilder as APIObsBuilder,
)

from src.compat.rlgym_v2_compat.common_values import BOOST_LOCATIONS, CAR_MAX_SPEED
from src.compat.rlgym_v2_compat import GameState, PlayerData, CarData, BoostPad, BallData
from src.utils.gym_compat import gym

from .observers import ModernObsBuilder
from .rewards import ModernRewardSystem
from .state_setters import ModernStateSetter
from .action_adapter import to_rlgym
from src.rlbot_integration.observation_adapter import OBS_SIZE  # must be 107

# Action schema:
#   cont: [steer, throttle, pitch, yaw, roll] in [-1, 1]
#   disc: [jump, boost, handbrake] as {0,1}
CONT_DIM = 5
DISC_DIM = 3


class SimpleActionParser(APIActionParser[int, Dict[str, np.ndarray], np.ndarray, GameState, gym.spaces.Dict]):
    """Action parser converting env dict actions to RLGym's engine format."""

    def __init__(self, action_space: gym.spaces.Dict):
        self._action_space = action_space

    def get_action_space(self, agent: int) -> gym.spaces.Dict:
        return self._action_space

    def reset(self, agents: List[int], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def parse_actions(
        self, actions: Dict[int, Dict[str, np.ndarray]], state: GameState, shared_info: Dict[str, Any]
    ) -> Dict[int, np.ndarray]:
        engine_actions: Dict[int, np.ndarray] = {}
        for agent, act in actions.items():
            a_cont = np.clip(act["cont"].astype(np.float32), -1.0, 1.0)
            a_disc = act["disc"].astype(np.float32).clip(0, 1)
            engine_actions[agent] = to_rlgym(a_cont, a_disc)
        return engine_actions


class SimplePhysicsEngine(APITransitionEngine[int, GameState, np.ndarray]):
    """Minimal transition engine advancing a tiny GameState."""

    def __init__(self):
        self._agents = [0, 1]
        self._config: Dict[str, Any] = {}
        self._state = self.create_base_state()

    # Required interface -------------------------------------------------
    @property
    def agents(self) -> List[int]:
        return self._agents

    @property
    def max_num_agents(self) -> int:
        return len(self._state.players)

    @property
    def state(self) -> GameState:
        return self._state

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @config.setter
    def config(self, value: Dict[str, Any]) -> None:
        self._config = value

    # Core logic ---------------------------------------------------------
    def step(self, actions: Dict[int, np.ndarray], shared_info: Dict[str, Any]) -> GameState:
        for agent, act in actions.items():
            player = self._state.players[agent]
            car = player.car_data
            throttle, steer, pitch, yaw, roll, jump, boost, handbrake = act

            # Orientation update
            car.set_rot(
                car.pitch + pitch * 0.02,
                car.yaw + yaw * 0.02,
                car.roll + roll * 0.02,
            )
            # Simple velocity/position update along forward vector
            forward = car.forward()
            speed = throttle * CAR_MAX_SPEED * 0.1
            car.set_lin_vel(*(forward * speed))
            car.set_pos(*(car.position + car.linear_velocity * 0.016))

            # Discrete mechanics
            if jump:
                player.on_ground = False
                player.has_jump = False
            else:
                player.on_ground = True
            if boost:
                player.boost_amount = min(100.0, player.boost_amount + 10.0)
            else:
                player.boost_amount = max(0.0, player.boost_amount - 0.5)

        # Advance timer
        self._state.game_seconds_remaining = max(
            0.0, self._state.game_seconds_remaining - 0.016
        )
        return self._state

    def create_base_state(self) -> GameState:
        ball = BallData()
        cars = [CarData(0), CarData(0), CarData(1), CarData(1)]
        players = [PlayerData(c, c.team_num) for c in cars]
        boost_pads = [BoostPad(pos.copy()) for pos in BOOST_LOCATIONS]
        return GameState(ball, players, boost_pads)

    def set_state(self, desired_state: GameState, shared_info: Dict[str, Any]) -> GameState:
        self._state = desired_state
        return self._state

    def close(self) -> None:
        pass


class StateSetterMutator(APIStateMutator[GameState]):
    """Adapter turning a ModernStateSetter into a StateMutator."""

    def __init__(self, setter: ModernStateSetter):
        self._setter = setter

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        wrapper = type("StateWrapper", (), {})()
        wrapper.cars = [p.car_data for p in state.players]
        wrapper.ball = state.ball
        self._setter.reset(wrapper)


class RewardAdapter(APIRewardFunction[int, GameState, float]):
    """Adapter so ModernRewardSystem matches the RLGym reward API."""

    def __init__(self, reward: ModernRewardSystem):
        self._reward = reward
        self.prev_actions: Dict[int, np.ndarray] = {}

    def reset(self, agents: List[int], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self._reward.reset(initial_state)
        self.prev_actions = {a: np.zeros(CONT_DIM + DISC_DIM, dtype=np.float32) for a in agents}

    def get_rewards(
        self,
        agents: List[int],
        state: GameState,
        is_terminated: Dict[int, bool],
        is_truncated: Dict[int, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[int, float]:
        rewards = {}
        for a in agents:
            player = state.players[a]
            rewards[a] = self._reward.get_reward(player, state, self.prev_actions[a])
        return rewards


class ObsBuilderAdapter(APIObsBuilder[int, np.ndarray, GameState, gym.spaces.Box]):
    """Adapter so ModernObsBuilder conforms to the RLGym obs API."""

    def __init__(self, builder: ModernObsBuilder):
        self._builder = builder
        self.prev_actions: Dict[int, np.ndarray] = {}
        self._obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)

    def get_obs_space(self, agent: int) -> gym.spaces.Box:
        return self._obs_space

    def reset(self, agents: List[int], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self._builder.reset(initial_state)
        self.prev_actions = {a: np.zeros(CONT_DIM + DISC_DIM, dtype=np.float32) for a in agents}

    def build_obs(
        self, agents: List[int], state: GameState, shared_info: Dict[str, Any]
    ) -> Dict[int, np.ndarray]:
        obs: Dict[int, np.ndarray] = {}
        for a in agents:
            player = state.players[a]
            obs[a] = self._builder.build_obs(player, state, self.prev_actions[a]).astype(np.float32)
        return obs


class RL2v2Env(gym.Env):
    """Gymnasium environment backed by an RLGym 2.0 session."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 42):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Dict(
            {
                "cont": gym.spaces.Box(low=-1.0, high=1.0, shape=(CONT_DIM,), dtype=np.float32),
                "disc": gym.spaces.MultiBinary(DISC_DIM),
            }
        )
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Core components
        self._obs_adapter = ObsBuilderAdapter(ModernObsBuilder())
        self._reward_adapter = RewardAdapter(ModernRewardSystem())
        self._state_mutator = StateSetterMutator(ModernStateSetter())
        self._action_parser = SimpleActionParser(self.action_space)
        self._engine = SimplePhysicsEngine()

        self._match = RLGym(
            state_mutator=self._state_mutator,
            obs_builder=self._obs_adapter,
            action_parser=self._action_parser,
            reward_fn=self._reward_adapter,
            transition_engine=self._engine,
        )

    # Gymnasium API ------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        obs_dict = self._match.reset()
        obs_vec = obs_dict[self._engine.agents[0]]
        info: Dict[str, Any] = {}
        return obs_vec.astype(np.float32), info

    def step(self, action: Dict[str, np.ndarray]):
        a_cont = np.clip(action["cont"].astype(np.float32), -1.0, 1.0)
        a_disc = action["disc"].astype(np.float32).clip(0, 1)
        rl_action = {a: {"cont": a_cont, "disc": a_disc} for a in self._engine.agents}

        # Step match
        obs, rewards, terminated, truncated = self._match.step(rl_action)

        # Current action in engine format for next step's prev_action
        engine_act = to_rlgym(a_cont, a_disc)
        for a in self._engine.agents:
            self._obs_adapter.prev_actions[a] = engine_act
            self._reward_adapter.prev_actions[a] = engine_act

        obs_vec = obs[self._engine.agents[0]]
        reward = float(np.mean([rewards[a] for a in self._engine.agents]))
        done = any(terminated.values())
        trunc = any(truncated.values())
        info: Dict[str, Any] = {}
        return obs_vec.astype(np.float32), reward, bool(done), bool(trunc), info


def make_env(seed: int = 42) -> Callable[[], RL2v2Env]:
    def _thunk() -> RL2v2Env:
        return RL2v2Env(seed=seed)

    return _thunk
