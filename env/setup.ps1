$ErrorActionPreference="Stop"

# Use Python 3.9
py -3.9 -m venv .\.venv
.\.venv\Scripts\python -m pip install --upgrade pip wheel

# Try CUDA Torch if NVIDIA GPU detected
$gpu = (Get-WmiObject win32_VideoController | Select-Object -Expand Name) -match "NVIDIA"
if ($gpu) {
  .\.venv\Scripts\pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
} else {
  .\.venv\Scripts\pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
}

# Rest of deps
.\.venv\Scripts\pip install -r .\env\requirements.txt

# Smoke
.\.venv\Scripts\python - << 'PY'
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
