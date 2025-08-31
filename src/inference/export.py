"""
Export best checkpoint to TorchScript for RLBot inference.

Expected contract:
- There exists a build_policy(obs_dim, n_cont, n_disc) -> nn.Module in src.training.policy
- Checkpoint at --ckpt contains state_dict compatible with that policy
- Outputs: models/exported/ssl_policy.ts
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import torch

# Defaults (adjust if your repo paths differ)
DEFAULT_CKPT = Path("models/checkpoints/best.pt")
DEFAULT_OUT = Path("models/exported/ssl_policy.ts")

# Observation / action sizes must match training
OBS_DIM = 1 + 1  # placeholder; will be overridden below if found
N_CONT = 5       # steer, throttle, pitch, yaw, roll
N_DISC = 3       # jump, boost, handbrake


def _infer_obs_dim_from_repo() -> int:
    """Try to discover OBS_SIZE from your training code to avoid drift."""
    try:
        # If your ModernObsBuilder defines OBS_SIZE, import and read it here
        from ModernObsBuilder import OBS_SIZE  # type: ignore
        if isinstance(OBS_SIZE, int) and OBS_SIZE > 0:
            return OBS_SIZE
    except Exception:
        pass
    # Fallback: environment variable override or last-resort constant
    env_val = os.getenv("SSL_OBS_SIZE")
    if env_val and env_val.isdigit():
        return int(env_val)
    # Final fallback — you MUST set this to your real obs size
    return 128


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--obs_dim", type=int, default=None)
    parser.add_argument("--n_cont", type=int, default=N_CONT)
    parser.add_argument("--n_disc", type=int, default=N_DISC)
    args = parser.parse_args()

    obs_dim = args.obs_dim or _infer_obs_dim_from_repo()
    n_cont = args.n_cont
    n_disc = args.n_disc

    # Lazy import to avoid heavy deps on import
    from src.training.policy import build_policy  # your implementation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[export] obs_dim={obs_dim}, n_cont={n_cont}, n_disc={n_disc}, device={device}")

    policy = build_policy(obs_dim, n_cont, n_disc).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)

    # Flexible loaders: accept either direct state_dict or nested
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()

    # Include normalization constants if your pipeline uses them
    if hasattr(policy, "set_normalization"):
        try:
            policy.set_normalization(ckpt.get("obs_norm"), ckpt.get("act_norm"))
        except Exception:
            pass

    # Script
    example = torch.zeros(1, obs_dim, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(policy, example)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.out))
    print(f"[export] Saved TorchScript → {args.out}")


if __name__ == "__main__":
    main()