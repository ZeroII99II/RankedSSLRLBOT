@echo off
REM SSL Bot Training Script for Windows
REM Runs the SSL bot training with PPO and curriculum learning

echo Starting SSL Bot Training...
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Please run env\setup.ps1 first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if required files exist
if not exist "configs\ppo_ssl.yaml" (
    echo Configuration file not found: configs\ppo_ssl.yaml
    pause
    exit /b 1
)

if not exist "configs\curriculum.yaml" (
    echo Curriculum file not found: configs\curriculum.yaml
    pause
    exit /b 1
)

REM Create models directory if it doesn't exist
if not exist "models\checkpoints" mkdir models\checkpoints
if not exist "models\exported" mkdir models\exported

REM Run training
echo Starting training...
echo Configuration: configs\ppo_ssl.yaml
echo Curriculum: configs\curriculum.yaml
echo.

python -m src.training.train --cfg configs\ppo_ssl.yaml --curr configs\curriculum.yaml

REM Check if training completed successfully
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Training completed successfully!
    echo Check models\checkpoints\ for saved models.
) else (
    echo.
    echo Training failed with error code %ERRORLEVEL%
)

echo.
pause
