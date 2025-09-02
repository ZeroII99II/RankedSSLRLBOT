def test_default_policy_export(tmp_path):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    import torch
    from src.training.policy import create_ssl_policy

    policy = create_ssl_policy({
        "obs_dim": 107,
        "continuous_actions": 5,
        "discrete_actions": 3,
        # Simplified network ensures tracing succeeds without full training setup
        "hidden_sizes": [],
        "use_attention": False,
    })
    policy.eval()

    class PolicyWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            out = self.policy(obs)
            return out["continuous_actions"], out["discrete_actions"]

    example = torch.zeros(1, 107)
    scripted = torch.jit.trace(PolicyWrapper(policy), example)
    out_file = tmp_path / "policy.ts"
    scripted.save(out_file.as_posix())
    assert out_file.exists()

    loaded = torch.jit.load(out_file.as_posix())
    out_cont, out_disc = loaded(torch.zeros(1, 107))
    assert out_cont.shape == (1, 5)
    assert out_disc.shape == (1, 3)

