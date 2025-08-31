#!/bin/bash
# SSL Bot RLBot Local Match Script for Linux/macOS
# Runs a local RLBot match with SSL Bot vs Nexto/Necto

set -e

echo "Starting SSL Bot RLBot Local Match..."
echo

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "Virtual environment not found. Please run env/setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if required files exist
if [ ! -f "models/exported/ssl_policy.ts" ]; then
    echo "SSL Bot policy not found: models/exported/ssl_policy.ts"
    echo "Please train and export a model first."
    exit 1
fi

if [ ! -f "configs/rlbot_match.toml" ]; then
    echo "RLBot match configuration not found: configs/rlbot_match.toml"
    exit 1
fi

# Check if RLBot is installed
python -c "import rlbot" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "RLBot not found. Please install it first."
    echo "Run: pip install rlbot"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Run RLBot match
echo "Starting RLBot match..."
echo "Configuration: configs/rlbot_match.toml"
echo "SSL Bot policy: models/exported/ssl_policy.ts"
echo

# Set environment variables for RLBot
export RLBOT_CONFIG_FILE="configs/rlbot_match.toml"
export RLBOT_LOG_LEVEL="INFO"

# Run RLBot
python -m rlbot.main

# Check if match completed successfully
if [ $? -eq 0 ]; then
    echo
    echo "Match completed successfully!"
    echo "Check logs/ for match logs and replays."
else
    echo
    echo "Match failed with error code $?"
    exit 1
fi
