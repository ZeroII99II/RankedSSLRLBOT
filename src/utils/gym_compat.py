"""Compatibility helpers for transitioning from Gym to Gymnasium APIs."""
from __future__ import annotations

try:  # Prefer gymnasium when available
    import gymnasium as gym  # type: ignore
except ImportError:  # Fall back to classic gym
    import gym  # type: ignore


def reset_env(env, *args, **kwargs):
    """Reset an environment and return only the observation.

    Handles the API difference where Gymnasium's ``reset`` returns ``(obs, info)``.
    Additional positional and keyword arguments are forwarded to ``env.reset``.
    """
    result = env.reset(*args, **kwargs)
    if isinstance(result, tuple) and len(result) == 2:
        obs, _info = result
        return obs
    return result


def step_env(env, action):
    """Step an environment and return ``(obs, reward, done, info)``.

    Gymnasium splits termination and truncation; this helper recombines them into
    a single ``done`` flag for compatibility with older Gym code. If the
    environment already follows the old API, the values are returned unchanged.
    """
    result = env.step(action)
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
        return obs, reward, done, info
    return result

__all__ = ["gym", "reset_env", "step_env"]
