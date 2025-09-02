from __future__ import annotations
import argparse, os
import torch
from stable_baselines3 import PPO
from src.training.env_factory import make_env
from src.training.ppo_sb3 import make_vec, make_policy_kwargs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_dir", type=str, default="models/checkpoints")
    ap.add_argument("--tensorboard", type=str, default="runs/ssl_v2")
    ap.add_argument("--switch_every", type=int, default=100_000,
                    help="Timesteps between team size switches")
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    team_sizes = [1, 2]
    team_idx = 0
    env_fns = [make_env(seed=args.seed + i, team_size=team_sizes[team_idx]) for i in range(args.envs)]
    vec = make_vec(env_fns)
    policy_kwargs = make_policy_kwargs()

    model = PPO(
        "MlpPolicy",
        vec,
        n_steps=max(2048 // args.envs, 128),
        batch_size=4096,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        n_epochs=6,
        tensorboard_log=args.tensorboard,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    steps_done = 0
    while steps_done < args.steps:
        learn_steps = min(args.switch_every, args.steps - steps_done)
        model.learn(total_timesteps=learn_steps, reset_num_timesteps=False)
        steps_done += learn_steps
        team_idx = (team_idx + 1) % len(team_sizes)
        if steps_done < args.steps:
            vec.close()
            env_fns = [
                make_env(seed=args.seed + i, team_size=team_sizes[team_idx])
                for i in range(args.envs)
            ]
            vec = make_vec(env_fns)
            model.set_env(vec)

    model.save(os.path.join(args.ckpt_dir, "best_sb3.zip"))

if __name__ == "__main__":
    main()
