# SSL Bot - Necto→SSL Pipeline

A clean, reproducible reinforcement learning pipeline for training SSL-level Rocket League bots using curriculum learning from Bronze to SSL.

## 🚀 Quick Start

### 1) Environment
```powershell
powershell -ExecutionPolicy Bypass -File .\env\setup.ps1
```

### 2) Train


Use `--seed` to make training runs deterministic.

> Use `--seed` to make runs deterministic. If omitted, the value from `training.seed` in the config is used.

### 3) Export TorchScript

```bash
python -m src.inference.export --ckpt models/checkpoints/best.pt --out models/exported/ssl_policy.ts
```
The exporter reads `configs/ppo_ssl.yaml` to rebuild the model. Use `--cfg` to supply a different training configuration.

### 4) Watch in Rocket League (RLBot)

```bash
scripts/run_rlbot_local.bat  # or .sh on Linux/Mac
```

> Set `SSL_POLICY_PATH` env var to point RLBot to a different model if needed.

### 5) Run Tests

```bash
# Run the PPO trainer smoke tests
pytest tests/test_real_env_smoke.py

# RLBot integration test
pytest tests/test_rlbot_integration.py

# Run the full suite
pytest
```

## Training (RLGym 2.0 + SB3)

```bash
python src/training/train_v2.py --envs 8 --steps 1000000
python src/inference/export.py --sb3 --ckpt models/checkpoints/best_sb3.zip --out models/exported/ssl_policy.ts --cfg configs/ppo_ssl.yaml
```

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Curriculum Learning](#curriculum-learning)
- [Installation](#installation)
- [Training](#training)
- [Export & Inference](#export--inference)
- [RLBot Integration](#rlbot-integration)
- [Streaming](#streaming)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

SSL Bot is a reinforcement learning pipeline that trains Rocket League bots to achieve SSL-level performance through curriculum learning. The system progresses through 8 phases from Bronze to SSL, with each phase focusing on specific mechanics and scenarios.

### Key Features

- **Curriculum Learning**: Bronze → Silver → Gold → Platinum → Diamond → Champion → GC → SSL
- **SSL-Level Mechanics**: Aerial interceptions, wall reads, backboard plays, double taps, flip resets
- **Fast Training**: RLGym + RocketSim for high-speed simulation
- **Self-Play**: Adaptive opponent generation for robust learning
- **RLBot Integration**: Live inference in Rocket League client
- **TorchScript Export**: Optimized models for production use

## 🏗️ Architecture

### Project Structure

```
ssl-bot/
├── configs/                 # Configuration files
│   ├── ppo_ssl.yaml        # PPO training configuration
│   ├── curriculum.yaml     # Curriculum phase definitions
│   └── rlbot_match.toml    # RLBot match configuration
├── models/                  # Model storage
│   ├── checkpoints/        # Training checkpoints
│   └── exported/           # TorchScript models
├── src/                     # Source code
│   ├── training/           # Training components
│   │   ├── observers.py    # SSL observation builder
│   │   ├── rewards.py      # Curriculum-aware rewards
│   │   ├── state_setters.py # Scenario sampling
│   │   ├── curriculum.py   # Curriculum manager
│   │   ├── policy.py       # MLP + attention policy
│   │   └── train.py        # PPO training loop
│   ├── inference/          # Model export
│   │   └── export.py       # TorchScript export
│   └── rlbot_integration/  # RLBot components
│       ├── bot.py          # RLBot agent
│       ├── observation_adapter.py
│       └── controller_adapter.py
├── scripts/                 # Launch scripts
│   ├── run_train.bat/sh    # Training script
│   └── run_rlbot_local.bat/sh # RLBot match script
├── env/                     # Environment setup
│   ├── requirements.txt    # Dependencies
│   ├── setup.ps1          # Windows setup
│   └── setup.sh           # Linux/macOS setup
└── Necto.bot.toml         # RLBot configuration
```

### Core Components

#### 1. Observation System (`observers.py`)
- **Static shape**: 107-dimensional observation vector
- **SSL-focused features**: Aerial mechanics, wall reads, backboard plays
- **Normalized inputs**: All features scaled to [-1, 1] range

#### 2. Reward System (`rewards.py`)
- **Curriculum-aware**: Reward weights adjust based on training phase
- **SSL mechanics**: Aerial interceptions, wall reads, double taps, flip resets
- **Penalty system**: Own-goal risk, panic jumps, bad touches

#### 3. State Setters (`state_setters.py`)
- **Scenario sampling**: Probabilistic scenario generation
- **Phase-specific**: Scenarios weighted by curriculum phase
- **SSL scenarios**: Aerial ladders, wall reads, backboard clears, ceiling pinches

#### 4. Policy Network (`policy.py`)
- **Architecture**: MLP + multi-head attention
- **Parameters**: ~1-3M parameters
- **Outputs**: Continuous actions (5) + discrete actions (3)

#### 5. Curriculum System (`curriculum.py`)
- **YAML-driven**: Phase definitions in configuration files
- **Progression gates**: Metrics-based phase advancement
- **Adaptive**: Reward and scenario weights per phase

## 📚 Curriculum Learning

The curriculum progresses through 8 phases, each with specific focus areas:

### Phase Progression

| Phase | Focus | Key Mechanics | Scenarios |
|-------|-------|---------------|-----------|
| **Bronze** | Basic movement, touches, recoveries | Ground play, basic aerials | Kickoffs, ground play, simple aerials |
| **Silver** | Improved movement, basic aerials | Wall bounces, aerial basics | Wall bounces, aerial intercepts |
| **Gold** | Boost management, positioning | Wall reads, boost efficiency | Wall reads, backboard clears |
| **Platinum** | Advanced positioning | Backboard plays, aerial control | Backboard clears, aerial intercepts |
| **Diamond** | Power shots, dribbles | Advanced aerials, double taps | Double tap setups, power shots |
| **Champion** | Advanced aerials, flip resets | Complex aerials, flip resets | Flip reset setups, complex aerials |
| **GC** | Expert aerial mechanics | Ceiling plays, expert aerials | Ceiling pinches, expert aerials |
| **SSL** | All SSL-level mechanics | Complete SSL skill set | All advanced scenarios |

### Progression Criteria

Each phase has specific progression gates:
- **Win rate**: Minimum win rate against opponents
- **Goal rate**: Goals per episode
- **Mechanic success**: Success rate for phase-specific mechanics
- **Training steps**: Minimum/maximum training steps

## 🛠️ Installation

### System Requirements

- **OS**: Windows 10/11 (primary), Linux/macOS (secondary)
- **Python**: 3.10 or higher
- **GPU**: CUDA-capable GPU recommended (RTX 3060+)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ free space

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ssl-bot.git
   cd ssl-bot
   ```

2. **Run setup script:**
   ```bash
   # Windows PowerShell
   .\env\setup.ps1
   
   # Linux/macOS
   bash env/setup.sh
   ```

3. **Verify installation:**
   ```bash
   # Activate environment
   # Windows
   .\venv\Scripts\Activate.ps1
   
   # Linux/macOS
   source venv/bin/activate
   
   # Test imports
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import rlgym; print('RLGym: OK')"
   python -c "import rlbot; print('RLBot: OK')"
   ```

### Dependencies

Core dependencies (automatically installed):
- `torch==2.2.2` (CUDA-enabled if GPU detected)
- `rlgym==2.0.0`
- `rlgym-ppo==1.3.0`
- `rocketsim==4.5.0`
- `rlbot==1.74.1`
- `pydantic==2.7.4`
- `omegaconf==2.3.0`
- `rich==13.7.1`

## 🏋️ Training

### Quick Start Training

```bash
# Windows
.\scripts\run_train.bat

# Linux/macOS
bash scripts/run_train.sh
```

### Manual Training

```bash
# Activate environment
# Windows
.\venv\Scripts\Activate.ps1

# Linux/macOS
source venv/bin/activate

# Run training
python -m src.training.train --cfg configs/ppo_ssl.yaml --curr configs/curriculum.yaml
```

### Training Configuration

Key training parameters in `configs/ppo_ssl.yaml`:

```yaml
ppo:
  steps_per_update: 32768
  mini_batches: 16
  n_epochs: 4
  actor_lr: 3e-4
  critic_lr: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2

policy:
  hidden_sizes: [1024, 1024, 512]
  use_attention: true
  num_heads: 8
  head_dim: 64
```

### Monitoring Training

Training progress is logged to:
- **TensorBoard**: `runs/ssl_bot_training/`
- **Console**: Rich progress bars and metrics
- **Checkpoints**: `models/checkpoints/`

View training progress:
```bash
# Start TensorBoard
tensorboard --logdir runs/ssl_bot_training/

# Open browser to http://localhost:6006
```

### Training Phases

The bot automatically progresses through curriculum phases based on performance:

1. **Bronze** (100K-500K steps): Basic mechanics
2. **Silver** (200K-800K steps): Improved movement
3. **Gold** (300K-1M steps): Boost management
4. **Platinum** (400K-1.2M steps): Advanced positioning
5. **Diamond** (500K-1.5M steps): Power shots
6. **Champion** (600K-1.8M steps): Advanced aerials
7. **GC** (700K-2M steps): Expert mechanics
8. **SSL** (800K-2.5M steps): Complete SSL skill set

## 📤 Export & Inference

### Export Trained Model

```bash
# Export best checkpoint
python -m src.inference.export --ckpt models/checkpoints/best.pt --out models/exported/ssl_policy.ts

# Export specific checkpoint
python -m src.inference.export --ckpt models/checkpoints/checkpoint_1000000.pt --out models/exported/ssl_policy.ts

# Test exported model
python -m src.inference.export --ckpt models/checkpoints/best.pt --out models/exported/ssl_policy.ts --test
```

The script reads `configs/ppo_ssl.yaml` by default to match the training setup. Use `--cfg` to specify a different config file.

### Export Options

```bash
# Use torch.jit.script instead of torch.jit.trace
python -m src.inference.export --ckpt models/checkpoints/best.pt --out models/exported/ssl_policy.ts --use-script

# Load custom configuration
python -m src.inference.export --ckpt models/checkpoints/best.pt --out models/exported/ssl_policy.ts --cfg configs/ppo_ssl.yaml

# Load observation statistics
python -m src.inference.export --ckpt models/checkpoints/best.pt --out models/exported/ssl_policy.ts --obs-stats obs_stats.npy
```

## 🤖 RLBot Integration

### Setup RLBot

1. **Install RLBot:**
   ```bash
   pip install rlbot
   ```

2. **Configure Rocket League:**
   - Set FPS to 120
   - Disable VSync
   - Enable RLBot in settings

### Run Local Match

```bash
# Windows
.\scripts\run_rlbot_local.bat

# Linux/macOS
bash scripts/run_rlbot_local.sh
```

### Manual RLBot Setup

```bash
# Set environment variables
export RLBOT_CONFIG_FILE="configs/rlbot_match.toml"
export RLBOT_LOG_LEVEL="INFO"

# Run RLBot
python -m rlbot.main
```

### Match Configuration

Edit `configs/rlbot_match.toml` to customize matches:

```toml
[match]
name = "SSL Bot vs Nexto"
map = "DFH Stadium"
game_mode = "Soccer"
team_size = 1
max_score = 3

[blue_team]
[[blue_team.bots]]
name = "SSLBot"
type = "python"
path = "src/rlbot_integration/bot.py"

[orange_team]
[[orange_team.bots]]
name = "Nexto"
type = "python"
path = "rlbot_support/Nexto/bot.py"
```

### Bot Configuration

Edit `Necto.bot.toml` to customize bot appearance and settings:

```toml
[Bot]
name = "SSL Bot"
description = "SSL-level Rocket League bot"

[Bot Loadout]
car_id = 23  # Octane
team_color_id = 60
wheels_id = 1565
boost_id = 35

[Bot Settings]
curriculum_phase = "ssl"
use_gpu = false
max_inference_time = 0.01
```

## Streaming

See [docs/streaming.md](docs/streaming.md) for how to launch the trainer with `--render`, visualise the latest policy in Rocket League and broadcast the window to Twitch.

## ⚙️ Configuration

### Training Configuration (`configs/ppo_ssl.yaml`)

```yaml
# Environment settings
env:
  team_size: 3
  tick_skip: 8
  self_play: true

# PPO hyperparameters
ppo:
  steps_per_update: 32768
  mini_batches: 16
  actor_lr: 3e-4
  critic_lr: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2

# Policy architecture
policy:
  hidden_sizes: [1024, 1024, 512]
  use_attention: true
  num_heads: 8
  head_dim: 64
  dropout: 0.1
```

### Curriculum Configuration (`configs/curriculum.yaml`)

```yaml
phases:
  bronze:
    description: "Basic movement, touches, recoveries"
    reward_weights:
      ball_to_goal_velocity: 0.3
      touch_quality: 0.5
      recovery: 1.0
    scenario_weights:
      kickoff: 0.3
      ground_play: 0.4
      simple_aerial: 0.1
    progression_gates:
      min_win_rate: 0.6
      min_goal_rate: 0.3
    min_training_steps: 100000
    max_training_steps: 500000
```

### RLBot Configuration (`configs/rlbot_match.toml`)

```toml
[match]
name = "SSL Bot vs Nexto"
map = "DFH Stadium"
team_size = 1
max_score = 3

[blue_team]
[[blue_team.bots]]
name = "SSLBot"
type = "python"
path = "src/rlbot_integration/bot.py"

[orange_team]
[[orange_team.bots]]
name = "Nexto"
type = "python"
path = "rlbot_support/Nexto/bot.py"
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Environment Setup Issues

**Problem**: Virtual environment creation fails
```bash
# Solution: Use Python 3.10+
python --version  # Should be 3.10+
python -m venv venv
```

**Problem**: CUDA not detected
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA PyTorch manually
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

#### 2. Training Issues

**Problem**: Training crashes with memory error
```yaml
# Reduce batch size in configs/ppo_ssl.yaml
ppo:
  steps_per_update: 16384  # Reduce from 32768
  mini_batches: 8          # Reduce from 16
```

**Problem**: Training is too slow
```yaml
# Enable mixed precision
device:
  mixed_precision: true

# Reduce model size
policy:
  hidden_sizes: [512, 512, 256]  # Smaller model
```

**Problem**: Bot not progressing through curriculum
```yaml
# Check progression gates in configs/curriculum.yaml
# Ensure evaluation metrics are being met
# Consider adjusting thresholds
```

#### 3. Export Issues

**Problem**: TorchScript export fails
```bash
# Try different export method
python -m src.inference.export --ckpt models/checkpoints/best.pt --out models/exported/ssl_policy.ts --use-script

# Check model compatibility
python -c "import torch; print(torch.__version__)"
```

**Problem**: Exported model doesn't work
```bash
# Run smoke test
python -m src.inference.export --ckpt models/checkpoints/best.pt --out models/exported/ssl_policy.ts --test

# Check model file
ls -la models/exported/ssl_policy.ts
```

#### 4. RLBot Issues

**Problem**: RLBot can't find bot
```bash
# Check file paths in Necto.bot.toml
# Ensure policy file exists
ls -la models/exported/ssl_policy.ts
```

**Problem**: Bot performs poorly in RLBot
```bash
# Check inference time
# Ensure 120 FPS in Rocket League
# Disable VSync
# Check observation normalization
```

**Problem**: RLBot crashes
```bash
# Check logs
tail -f logs/rlbot.log

# Verify dependencies
pip list | grep rlbot
```

### Performance Optimization

#### Training Performance

1. **GPU Memory Optimization:**
   ```yaml
   # Reduce batch size
   ppo:
     steps_per_update: 16384
     mini_batches: 8
   
   # Enable mixed precision
   device:
     mixed_precision: true
   ```

2. **CPU Optimization:**
   ```yaml
   # Increase workers
   optimization:
     num_workers: 8
     prefetch_factor: 4
   ```

3. **Model Optimization:**
   ```yaml
   # Smaller model
   policy:
     hidden_sizes: [512, 512, 256]
     use_attention: false
   ```

#### Inference Performance

1. **RLBot Optimization:**
   ```toml
   [Bot Settings]
   use_gpu = false  # Use CPU for inference
   max_inference_time = 0.01  # 10ms max
   ```

2. **Model Optimization:**
   ```bash
   # Use torch.jit.script for better optimization
   python -m src.inference.export --ckpt models/checkpoints/best.pt --out models/exported/ssl_policy.ts --use-script
   ```

### Debug Mode

Enable debug mode for detailed logging:

```yaml
# In configs/ppo_ssl.yaml
training:
  seed: 0  # Random seed for reproducibility
  log_frequency: 10  # More frequent logging
  tensorboard: true
  wandb: true  # Enable Weights & Biases
```

```toml
# In Necto.bot.toml
[Bot Settings]
debug_mode = true
log_level = "DEBUG"
```

### Getting Help

1. **Check logs:**
   ```bash
   # Training logs
   tail -f runs/ssl_bot_training/events.out.tfevents.*
   
   # RLBot logs
   tail -f logs/rlbot.log
   ```

2. **Verify installation:**
   ```bash
   # Test all components
   python -c "import torch, rlgym, rlbot; print('All imports successful')"
   ```

3. **Check system requirements:**
   ```bash
   # GPU info
   nvidia-smi
   
   # Python version
   python --version
   
   # Available memory
   free -h  # Linux
   # or check Task Manager on Windows
   ```

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create development environment:**
   ```bash
   git clone https://github.com/your-username/ssl-bot.git
   cd ssl-bot
   .\env\setup.ps1  # or bash env/setup.sh
   ```

3. **Install development dependencies:**
   ```bash
   pip install black ruff mypy pytest
   ```

4. **Run tests:**
   ```bash
   pytest tests/
   ```

5. **Format code:**
   ```bash
   black src/
   ruff check src/
   mypy src/
   ```

### Code Style

- **Python**: Black formatting, line length 100
- **Linting**: Ruff with E, F, I, UP, B, W, N rules
- **Type checking**: MyPy with strict settings
- **Documentation**: Google-style docstrings

### Pull Request Process

1. **Create feature branch**
2. **Implement changes**
3. **Add tests**
4. **Update documentation**
5. **Submit pull request**

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_policy.py

# Run with coverage
pytest --cov=src tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **RLGym**: Rocket League gym environment
- **RocketSim**: High-performance simulation
- **RLBot**: Bot framework for Rocket League
- **Necto**: Original bot implementation
- **Rocket League Community**: For inspiration and feedback

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/ssl-bot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/ssl-bot/discussions)
- **Discord**: [RLBot Discord](https://discord.gg/rlbot)

---

**⚠️ Important Notice**: This bot is for educational and research purposes. Do not use automated bots in online matchmaking - they are intended for local/private matches only. Always respect Rocket League's Terms of Service.
