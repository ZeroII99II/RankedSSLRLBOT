$ErrorActionPreference="Stop"
# Create venv (Python 3.9 for RLGym compatibility)
py -3.9 -m venv .\.venv
.\.venv\Scripts\python -m pip install --upgrade pip wheel

# Try CUDA torch if NVIDIA GPU detected
$gpu = (Get-WmiObject win32_VideoController | Select-Object -Expand Name) -match "NVIDIA"
if ($gpu) {
  .\.venv\Scripts\python -m pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
}

.\.venv\Scripts\python -m pip install -r .\env\requirements.txt

# Quick import check
.\.venv\Scripts\python -c "import torch, rlgym, rlbot; print('OK:', torch.__version__)"