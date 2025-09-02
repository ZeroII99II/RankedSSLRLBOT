import sys
from pathlib import Path
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from src.inference import export


def test_export_default_path(tmp_path, monkeypatch):
    # Use temporary working directory
    monkeypatch.chdir(tmp_path)

    # Create dummy policy builder
    from src.training import policy as policy_module

    class TinyPolicy(torch.nn.Module):
        def __init__(self, obs_dim, cont, disc):
            super().__init__()
            self.fc = torch.nn.Linear(obs_dim, cont + disc)
            self.cont = cont
            self.disc = disc
        def forward(self, x):
            logits = self.fc(x)
            return torch.tanh(logits[:, : self.cont]), logits[:, self.cont:]

    def build_policy(obs_dim, cont, disc):
        return TinyPolicy(obs_dim, cont, disc)

    policy_module.build_policy = build_policy

    # Save checkpoint for tiny model
    ckpt_path = tmp_path / "ckpt.pt"
    tiny_policy = policy_module.create_ssl_policy({
        "obs_dim": export.OBS_DIM_DEFAULT,
        "continuous_actions": export.CONT_DIM,
        "discrete_actions": export.DISC_DIM,
    })
    torch.save(tiny_policy.state_dict(), ckpt_path)

    # Run export script with explicit output path and load the result
    out_file = tmp_path / "tiny.ts"
    monkeypatch.setattr(
        sys,
        "argv",
        ["export.py", "--ckpt", str(ckpt_path), "--out", out_file.as_posix()],
    )
    export.main()

    assert out_file.is_file()

    scripted = torch.jit.load(out_file.as_posix())
    out_cont, out_disc = scripted(torch.zeros(1, export.OBS_DIM_DEFAULT))
    assert out_cont.shape == (1, export.CONT_DIM)
    assert out_disc.shape == (1, export.DISC_DIM)
