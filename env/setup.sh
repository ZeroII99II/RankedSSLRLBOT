#!/usr/bin/env bash
set -euo pipefail

# --- find a Python 3.9 interpreter ---
PY_BIN=""
for CAND in python3.9 python3 python; do
  if command -v "$CAND" >/dev/null 2>&1; then
    VER=$("$CAND" - <<'PY'
import sys
print(".".join(map(str, sys.version_info[:2])))
PY
)
    if [ "$VER" = "3.9" ]; then PY_BIN="$CAND"; break; fi
  fi
done

if [ -z "${PY_BIN}" ]; then
  echo "ERROR: Python 3.9 not found. Install it or use pyenv: 'pyenv install 3.9.18'." >&2
  exit 1
fi

# --- venv & pip ---
$PY_BIN -m venv .venv
. .venv/bin/activate
python -m pip install -U pip wheel

# --- Torch: CPU by default; enable CUDA by setting TORCH_CUDA=1 ---
if [ "${TORCH_CUDA:-0}" = "1" ]; then
  # CUDA 12.1 wheels
  pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
else
  pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
fi

# --- rest of requirements ---
pip install -r env/requirements.txt

# --- smoke ---
python - <<'PY'
import torch, gymnasium, sys
try:
    import rlbot
    rlb = getattr(rlbot, "__version__", "dev")
except Exception:
    rlb = "import-failed"
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gymnasium", gymnasium.__version__)
print("rlbot", rlb)
print("python", sys.version)
PY
