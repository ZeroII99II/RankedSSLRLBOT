@echo off
REM SSL Bot RLBot Local Match Script for Windows
REM Runs a local RLBot match with SSL Bot vs Nexto/Necto

echo Starting SSL Bot RLBot Local Match...
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
if not exist "models\exported\ssl_policy.ts" (
    echo SSL Bot policy not found: models\exported\ssl_policy.ts
    echo Please train and export a model first.
    pause
    exit /b 1
)

if not exist "configs\rlbot_match.toml" (
    echo RLBot match configuration not found: configs\rlbot_match.toml
    pause
    exit /b 1
)

REM Check if RLBot is installed
python -c "import rlbot" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo RLBot not found. Please install it first.
    echo Run: pip install rlbot
    pause
    exit /b 1
)

REM Create logs directory
if not exist "logs" mkdir logs

REM Run RLBot match
echo Starting RLBot match...
echo Configuration: configs\rlbot_match.toml
echo SSL Bot policy: models\exported\ssl_policy.ts
echo.

REM Set environment variables for RLBot
set RLBOT_CONFIG_FILE=configs\rlbot_match.toml
set RLBOT_LOG_LEVEL=INFO

REM Run RLBot
python -m rlbot.main

REM Check if match completed successfully
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Match completed successfully!
    echo Check logs\ for match logs and replays.
) else (
    echo.
    echo Match failed with error code %ERRORLEVEL%
)

echo.
pause
