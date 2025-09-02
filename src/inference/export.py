from __future__ import annotations
import argparse
from pathlib import Path
import torch
import yaml

DEFAULT_OUT = "models/exported/ssl_policy.ts"
# Fallback dimensions when configuration does not specify them
OBS_DIM_DEFAULT = 107
CONT_DIM = 5
DISC_DIM = 3
DEFAULT_CFG = Path(__file__).resolve().parents[2] / "configs/ppo_ssl.yaml"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sb3", action="store_true", help="export from SB3 PPO zip")
    ap.add_argument("--ckpt", type=str, required=True, help="path to checkpoint")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT, help="output TorchScript path")
    ap.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="training config to derive policy architecture (default: configs/ppo_ssl.yaml)",
    )
    args = ap.parse_args()

    # Load policy configuration
    cfg_path = Path(args.cfg) if args.cfg else DEFAULT_CFG
    policy_cfg = {}
    if cfg_path.is_file():
        with cfg_path.open() as f:
            policy_cfg = yaml.safe_load(f).get("policy", {})

    obs_dim = policy_cfg.get("obs_dim", OBS_DIM_DEFAULT)
    cont_dim = policy_cfg.get("continuous_actions", CONT_DIM)
    disc_dim = policy_cfg.get("discrete_actions", DISC_DIM)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.sb3:
        from stable_baselines3 import PPO

        model = PPO.load(args.ckpt, device=device)
        policy = model.policy

        class SB3Wrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.features_extractor = policy.features_extractor
                self.mlp_extractor = policy.mlp_extractor
                self.action_net = policy.action_net

            def forward(self, obs: torch.Tensor):
                x = self.features_extractor(obs)
                if self.mlp_extractor is not None:
                    x, _ = self.mlp_extractor(x)
                logits = self.action_net(x)
                a_cont = torch.tanh(logits[:, :cont_dim])
                a_disc_logits = logits[:, cont_dim:cont_dim+disc_dim]
                return a_cont, a_disc_logits

        module = SB3Wrapper(policy).to(device)
        example = torch.zeros(1, obs_dim, device=device)
        scripted = torch.jit.trace(module, example)
    else:
        from src.training.policy import create_ssl_policy

        # Create SSL policy from configuration
        ssl_policy = create_ssl_policy(policy_cfg).to(device)
        ckpt = torch.load(args.ckpt, map_location=device)
        state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        ssl_policy.load_state_dict(state_dict, strict=False)
        ssl_policy.eval()

        class PolicyWrapper(torch.nn.Module):
            """TorchScript-compatible wrapper for SSLPolicy outputs."""

            def __init__(self, policy: torch.nn.Module) -> None:
                super().__init__()
                self.policy = policy

            def forward(self, obs: torch.Tensor):
                out = self.policy(obs)
                if isinstance(out, tuple):
                    return out
                return out["continuous_actions"], out["discrete_actions"]

        example = torch.zeros(1, obs_dim, device=device)
        scripted = torch.jit.trace(PolicyWrapper(ssl_policy), example)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(out_path.as_posix())

if __name__ == "__main__":
    main()
