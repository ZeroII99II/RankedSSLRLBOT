import sys
from pathlib import Path
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from src.inference import export


def test_export_default_path(tmp_path, monkeypatch):
    # Use temporary working directory
    monkeypatch.chdir(tmp_path)

    # Create dummy build_policy
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
    torch.save(build_policy(export.OBS_DIM_DEFAULT, export.CONT_DIM, export.DISC_DIM).state_dict(), ckpt_path)

    # Run export script with default output path
    monkeypatch.setattr(sys, "argv", ["export.py", "--ckpt", str(ckpt_path)])
    export.main()

    out_path = tmp_path / export.DEFAULT_OUT
    assert out_path.is_file()

    scripted = torch.jit.load(out_path.as_posix())
    out_cont, out_disc = scripted(torch.zeros(1, export.OBS_DIM_DEFAULT))
    assert out_cont.shape == (1, export.CONT_DIM)
    assert out_disc.shape == (1, export.DISC_DIM)
