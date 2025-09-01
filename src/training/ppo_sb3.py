"""Stable-Baselines3 PPO helper for v2 trainer."""
from __future__ import annotations

from typing import Optional

from stable_baselines3 import PPO


def make_ppo(env, tensorboard_log: Optional[str] = None) -> PPO:
    """Build a PPO agent with the architecture used for training.

    The network uses an MLP extractor with layers 1024 → 1024 → 512 for both the
    policy and value networks.  Action space is expected to be a Dict with
    continuous (5) and discrete (3) components.
    """

    policy_kwargs = dict(
        net_arch=[dict(pi=[1024, 1024, 512], vf=[1024, 1024, 512])]
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
    )
    return model
