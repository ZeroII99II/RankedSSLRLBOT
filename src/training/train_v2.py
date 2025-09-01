"""High level training entrypoint for the SB3-based v2 trainer."""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.vec_env import SubprocVecEnv

from .env_factory import make_env
from .ppo_sb3 import make_ppo


def build_vec_env(n_envs: int):
    return SubprocVecEnv([make_env for _ in range(n_envs)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Training steps")
    parser.add_argument("--ckpt_dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--tensorboard", type=str, default=None)
    args = parser.parse_args()

    vec_env = build_vec_env(args.envs)
    model = make_ppo(vec_env, tensorboard_log=args.tensorboard)
    model.learn(total_timesteps=args.steps)

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.ckpt_dir / "best_sb3.zip")


if __name__ == "__main__":
    main()
