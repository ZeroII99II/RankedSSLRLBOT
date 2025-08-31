# SSL Bot Environment Setup Script for Windows PowerShell
# Creates Python 3.10 venv and installs dependencies with GPU detection

Write-Host "Setting up SSL Bot environment..." -ForegroundColor Green

# Check if Python 3.10+ is available
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "Python (\d+)\.(\d+)") {
    $major = [int]$matches[1]
    $minor = [int]$matches[2]
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
        Write-Host "Error: Python 3.10+ required. Found: $pythonVersion" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Error: Python not found. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

Write-Host "Python version: $pythonVersion" -ForegroundColor Green

# Create virtual environment
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists. Removing..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}

Write-Host "Creating virtual environment..." -ForegroundColor Green
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# Detect GPU and install appropriate PyTorch
Write-Host "Detecting GPU..." -ForegroundColor Green
try {
    $gpuInfo = nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>$null
    if ($gpuInfo) {
        Write-Host "GPU detected: $gpuInfo" -ForegroundColor Green
        Write-Host "Installing CUDA-enabled PyTorch..." -ForegroundColor Green
        pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
    } else {
        Write-Host "No GPU detected, installing CPU-only PyTorch..." -ForegroundColor Yellow
        pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
    }
} catch {
    Write-Host "GPU detection failed, installing CPU-only PyTorch..." -ForegroundColor Yellow
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
}

# Install other requirements
Write-Host "Installing other requirements..." -ForegroundColor Green
pip install -r requirements.txt

# Verify installations
Write-Host "Verifying installations..." -ForegroundColor Green
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import rlgym; print('RLGym imported successfully')"
python -c "import rlbot; print('RLBot imported successfully')"

Write-Host "Environment setup complete!" -ForegroundColor Green
Write-Host "To activate the environment in the future, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
