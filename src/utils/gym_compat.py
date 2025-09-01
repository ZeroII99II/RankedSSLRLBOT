from __future__ import annotations

try:
    import gymnasium as gym
except Exception:  # fallback if some dep still needs old gym
    import gym  # type: ignore
    # Warning: old gym is unmaintained; prefer gymnasium
    pass

def reset_env(env, **kwargs):
    out = env.reset(**kwargs)
    if isinstance(out, tuple) and len(out) == 2:  # Gymnasium
        obs, info = out
    else:  # legacy gym
        obs, info = out, {}
    return obs, info

def step_env(env, action):
    out = env.step(action)
    # Gymnasium: (obs, reward, terminated, truncated, info)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:  # legacy gym
        obs, reward, done, info = out
    return obs, reward, done, info

__all__ = ["gym", "reset_env", "step_env"]
