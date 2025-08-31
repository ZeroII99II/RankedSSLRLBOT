"""
RLBot bot that loads a TorchScript model and runs inference live.
Ensure the observation matches training exactly.
"""
from __future__ import annotations
import os
import time
from pathlib import Path

import numpy as np
import torch
from rlbot.agents.base_agent import BaseAgent

from .observation_adapter import build_observation, OBS_SIZE
from .controller_adapter import to_controls

# Where to find exported policy (override via env SSL_POLICY_PATH)
POLICY_PATH = Path(os.getenv("SSL_POLICY_PATH", "models/exported/ssl_policy.ts"))


class SSLBot(BaseAgent):
    def initialize_agent(self):
        self.model = None
        self._last_mtime = 0.0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model(force=True)

    def _load_model(self, force: bool = False):
        try:
            if not POLICY_PATH.exists():
                if force:
                    self.logger.warn(f"Policy not found at {POLICY_PATH}, running fallback controller.")
                return
            mtime = POLICY_PATH.stat().st_mtime
            if force or mtime > self._last_mtime:
                self.model = torch.jit.load(str(POLICY_PATH), map_location=self._device)
                self.model.eval()
                self._last_mtime = mtime
                self.logger.info(f"Loaded TorchScript policy: {POLICY_PATH}")
        except Exception as e:
            self.logger.error(f"Failed to load policy: {e}")
            self.model = None

    def get_output(self, packet):
        # Hot-reload if file updated
        self._load_model()

        # Build observation from packet (BaseAgent sets self.index used by adapter)
        try:
            obs_np = build_observation(packet)
        except Exception as e:
            self.logger.error(f"Obs build failed: {e}")
            obs_np = np.zeros(OBS_SIZE, dtype=np.float32)

        if self.model is None:
            # Simple fallback: drive forward a bit
            a_cont = np.array([0.0, 0.3, 0.0, 0.0, 0.0], dtype=np.float32)
            a_disc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            return to_controls(a_cont, a_disc)

        # Forward pass
        with torch.no_grad():
            obs = torch.from_numpy(obs_np).to(self._device).unsqueeze(0)
            out = self.model(obs)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                a_cont, a_disc = out
            else:
                # If model outputs only continuous, create zeros for buttons
                a_cont, a_disc = out, torch.zeros((1, 3), device=self._device)
            a_cont = a_cont.squeeze(0).detach().cpu().numpy()
            a_disc = torch.sigmoid(a_disc).squeeze(0).detach().cpu().numpy()
        return to_controls(a_cont, a_disc)