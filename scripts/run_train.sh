#!/bin/bash
# SSL Bot Training Script for Linux/macOS
# Runs the SSL bot training with PPO and curriculum learning

set -e

echo "Starting SSL Bot Training..."
echo

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "Virtual environment not found. Please run env/setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if required files exist
if [ ! -f "configs/ppo_ssl.yaml" ]; then
    echo "Configuration file not found: configs/ppo_ssl.yaml"
    exit 1
fi

if [ ! -f "configs/curriculum.yaml" ]; then
    echo "Curriculum file not found: configs/curriculum.yaml"
    exit 1
fi

# Create models directory if it doesn't exist
mkdir -p models/checkpoints
mkdir -p models/exported

# Run training
echo "Starting training..."
echo "Configuration: configs/ppo_ssl.yaml"
echo "Curriculum: configs/curriculum.yaml"
echo

python -m src.training.train --cfg configs/ppo_ssl.yaml --curr configs/curriculum.yaml

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo
    echo "Training completed successfully!"
    echo "Check models/checkpoints/ for saved models."
else
    echo
    echo "Training failed with error code $?"
    exit 1
fi
