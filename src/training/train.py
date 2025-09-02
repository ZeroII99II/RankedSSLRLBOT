"""
SSL Bot Training Script with PPO and Curriculum Learning.
Implements vectorized environments using RocketSim for fast training.
"""

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

# RLGym imports
from rlgym.utils.action_parsers import DefaultAction

# Local imports
from .observers import SSLObsBuilder
from .rewards import SSLRewardFunction
from .state_setters import SSLStateSetter
from .curriculum import CurriculumManager
from .policy import SSLPolicy, SSLCritic, create_ssl_policy, create_ssl_critic
from src.utils.gym_compat import gym, reset_env, step_env


class PPOTrainer:
    """
    PPO trainer for SSL bot with curriculum learning and self-play.
    """

    def __init__(self, config_path: str, curriculum_path: str, seed: Optional[int] = None):
        self.console = Console()
        self.config = self._load_config(config_path)

        training_cfg = self.config.setdefault('training', {})
        if seed is not None:
            training_cfg['seed'] = seed
        self.seed = training_cfg.get('seed', 42)

        self.curriculum = CurriculumManager(curriculum_path)

        # Setup device
        self.device = self._setup_device()

        # Initialize RNGs
        self.py_rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        self.torch_rng = torch.Generator(device=self.device).manual_seed(self.seed)
        
        # Initialize models
        self.policy = create_ssl_policy(self.config['policy']).to(self.device)
        self.critic = create_ssl_critic(self.config['policy']).to(self.device)
        
        # Setup optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config['ppo']['actor_lr']
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config['ppo']['critic_lr']
        )
        
        # Setup environment
        self.action_parser = DefaultAction()
        self.env = self._create_environment()
        
        # Training state
        self.training_steps = 0
        self.episode_count = 0
        self.best_eval_score = float('-inf')
        
        # Setup logging
        self.writer = SummaryWriter(log_dir='runs/ssl_bot_training')
        
        # Setup checkpointing
        self.checkpoint_dir = Path('models/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.console.print(f"[green]SSL Bot Trainer initialized[/green]")
        self.console.print(f"Device: {self.device}")
        self.console.print(f"Current phase: {self.curriculum.get_current_phase().name}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config['device']['auto_detect']:
            if torch.cuda.is_available() and self.config['device']['cuda']:
                device = torch.device('cuda')
                self.console.print(f"[green]Using CUDA: {torch.cuda.get_device_name()}[/green]")
            else:
                device = torch.device('cpu')
                self.console.print("[yellow]Using CPU[/yellow]")
        else:
            device = torch.device(self.config['device'].get('device', 'cpu'))
        
        return device
    
    def _create_environment(self):
        """Create RLGym environment with SSL components."""
        import rlgym
        from rlgym.utils.terminal_conditions import common_conditions

        # Create observation builder
        obs_builder = SSLObsBuilder(
            n_players=self.config['env']['team_size'] * 2,
            tick_skip=self.config['env']['tick_skip']
        )
        
        # Create reward function
        reward_fn = SSLRewardFunction(
            curriculum_phase=self.curriculum.get_current_phase().name
        )
        
        # Create state setter
        state_setter = SSLStateSetter(
            curriculum_phase=self.curriculum.get_current_phase().name,
            rng=self.np_rng,
        )
        
        # Create environment
        env = rlgym.make(
            use_injector=self.config['env']['use_injector'],
            self_play=self.config['env']['self_play'],
            team_size=self.config['env']['team_size'],
            tick_skip=self.config['env']['tick_skip'],
            spawn_opponents=self.config['env']['spawn_opponents'],
            seed=getattr(self, 'seed', None),
            obs_builder=obs_builder,
            reward_fn=reward_fn,
            state_setter=state_setter,
            terminal_conditions=[
                common_conditions.TimeoutCondition(300),  # 5 minutes
                common_conditions.GoalScoredCondition()
            ],
            action_parser=self.action_parser
        )
        
        return env
    
    def _collect_rollouts(self, num_steps: int) -> Dict[str, torch.Tensor]:
        """Collect rollouts for PPO training."""
        obs_buffer = []
        action_buffer = {
            'continuous_actions': [],
            'discrete_actions': []
        }
        reward_buffer = []
        value_buffer = []
        log_prob_buffer = []
        done_buffer = []

        try:
            obs, _info = reset_env(
                self.env, seed=int(self.np_rng.integers(0, 2**32))
            )
        except TypeError:
            obs, _info = reset_env(self.env)
        self.env.action_space.seed(int(self.np_rng.integers(0, 2**32)))
        episode_rewards = []
        episode_lengths = []
        episode_reward = 0.0
        episode_len = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Collecting rollouts...", total=num_steps)
            
            for step in range(num_steps):
                # Convert observations to tensor with batch dimension
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                # Get action from policy
                with torch.no_grad():
                    action_outputs = self.policy.sample_actions(
                        obs_tensor, generator=self.torch_rng
                    )
                    value = self.critic(obs_tensor)
                    log_prob = self.policy.log_prob(obs_tensor, action_outputs)
                
                # Convert actions to environment format
                actions = self._convert_actions_to_env(action_outputs)

                # Step environment
                next_obs, reward, done, info = step_env(self.env, actions)
                
                # Store experience
                obs_buffer.append(obs_tensor.cpu())
                action_buffer['continuous_actions'].append(
                    action_outputs['continuous_actions'].cpu()
                )
                action_buffer['discrete_actions'].append(
                    action_outputs['discrete_actions'].cpu()
                )
                reward_buffer.append(torch.tensor([reward], dtype=torch.float32))
                value_buffer.append(value.cpu())
                log_prob_buffer.append(log_prob.cpu())
                done_buffer.append(torch.tensor([done], dtype=torch.bool))

                obs = next_obs

                # Track episode statistics
                episode_reward += reward
                episode_len += 1
                if done:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_len)
                    episode_reward = 0.0
                    episode_len = 0
                    try:
                        obs, _info = reset_env(
                            self.env, seed=int(self.np_rng.integers(0, 2**32))
                        )
                    except TypeError:
                        obs, _info = reset_env(self.env)
                    self.env.action_space.seed(int(self.np_rng.integers(0, 2**32)))

                progress.update(task, advance=1)

        # If rollout finished without episode termination, record the ongoing episode
        if episode_len > 0:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_len)

        # Convert buffers to tensors
        actions = {
            'continuous_actions': torch.cat(action_buffer['continuous_actions']),
            'discrete_actions': torch.cat(action_buffer['discrete_actions']),
        }

        rollouts = {
            'observations': torch.cat(obs_buffer),
            'actions': actions,
            'rewards': torch.cat(reward_buffer),
            'values': torch.cat(value_buffer),
            'log_probs': torch.cat(log_prob_buffer),
            'dones': torch.cat(done_buffer),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
        }
        
        return rollouts
    
    def _convert_actions_to_env(self, action_outputs: Dict[str, torch.Tensor]) -> np.ndarray:
        """Convert policy actions to environment format."""
        continuous_actions = action_outputs['continuous_actions'].cpu().numpy()
        discrete_actions = action_outputs['discrete_actions'].cpu().numpy()

        # Combine continuous and discrete actions for single environment
        action = np.concatenate([
            continuous_actions[0],  # throttle, steer, pitch, yaw, roll
            discrete_actions[0]     # jump, boost, handbrake
        ])

        return action
    
    def _compute_advantages(self, rollouts: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages and returns using GAE."""
        rewards = rollouts['rewards']
        values = rollouts['values']
        dones = rollouts['dones'].float()
        
        gamma = self.config['ppo']['gamma']
        gae_lambda = self.config['ppo']['gae_lambda']
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute advantages using GAE
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        # Compute returns
        returns = advantages + values
        
        return advantages, returns
    
    def _update_policy(self, rollouts: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy using PPO."""
        advantages, returns = self._compute_advantages(rollouts)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        
        # Prepare data
        obs = rollouts['observations'].to(self.device)
        actions = {k: v.to(self.device) for k, v in rollouts['actions'].items()}
        old_log_probs = rollouts['log_probs'].to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # PPO update
        n_epochs = self.config['ppo']['n_epochs']
        mini_batch_size = self.config['ppo']['steps_per_update'] // self.config['ppo']['mini_batches']
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for epoch in range(n_epochs):
            # Create mini-batches
            indices = torch.randperm(len(obs), generator=self.torch_rng)
            
            for start_idx in range(0, len(obs), mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, len(obs))
                batch_indices = indices[start_idx:end_idx]
                
                batch_obs = obs[batch_indices]
                batch_actions = {k: v[batch_indices] for k, v in actions.items()}
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Policy update
                new_log_probs = self.policy.log_prob(batch_obs, batch_actions)
                entropy = self.policy.entropy(batch_obs)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config['ppo']['clip_ratio'], 
                                  1 + self.config['ppo']['clip_ratio']) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value update
                values = self.critic(batch_obs).squeeze()
                value_loss = nn.MSELoss()(values, batch_returns)
                
                # Total loss
                entropy_loss = -entropy.mean()
                total_loss = (policy_loss + 
                            self.config['ppo']['value_loss_coef'] * value_loss +
                            self.config['ppo']['entropy_coef'] * entropy_loss)
                
                # Update policy
                self.policy_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 
                                             self.config['ppo']['max_grad_norm'])
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 
                                             self.config['ppo']['max_grad_norm'])
                self.policy_optimizer.step()
                self.critic_optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses)
        }
    
    def _evaluate(self, num_episodes: int = 20) -> Dict[str, float]:
        """Evaluate current policy."""
        eval_rewards = []
        eval_lengths = []
        
        obs, _info = reset_env(
            self.env, seed=int(self.np_rng.integers(0, 2**32))
        )

        for episode in range(num_episodes):
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action_outputs = self.policy.sample_actions(
                        obs_tensor, generator=self.torch_rng
                    )

                actions = self._convert_actions_to_env(action_outputs)
                obs, reward, done, info = step_env(self.env, actions)

                episode_reward += reward
                episode_length += 1
                if done:
                    obs, _info = reset_env(
                        self.env, seed=int(self.np_rng.integers(0, 2**32))
                    )

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        return {
            'eval_reward': np.mean(eval_rewards),
            'eval_length': np.mean(eval_lengths),
            'eval_reward_std': np.std(eval_rewards)
        }
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'training_steps': self.training_steps,
            'episode_count': self.episode_count,
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'curriculum_phase': self.curriculum.get_current_phase().name,
            'best_eval_score': self.best_eval_score
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_{self.training_steps}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            self.console.print(f"[green]New best model saved![/green]")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.training_steps = checkpoint['training_steps']
        self.episode_count = checkpoint['episode_count']
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.best_eval_score = checkpoint['best_eval_score']
        
        # Load curriculum state
        if 'curriculum_phase' in checkpoint:
            self.curriculum.current_phase = checkpoint['curriculum_phase']
        
        self.console.print(f"[green]Loaded checkpoint from {checkpoint_path}[/green]")
    
    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to tensorboard."""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
    
    def _display_progress(self, metrics: Dict[str, float]):
        """Display training progress."""
        table = Table(title="SSL Bot Training Progress")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in metrics.items():
            table.add_row(key, f"{value:.4f}")
        
        self.console.print(table)
    
    def train(self, max_steps: int = 1000000, resume_from: Optional[str] = None):
        """Main training loop."""
        if resume_from:
            self._load_checkpoint(resume_from)
        
        self.console.print(f"[green]Starting SSL Bot training for {max_steps} steps[/green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Training...", total=max_steps)
            
            while self.training_steps < max_steps:
                # Collect rollouts
                rollouts = self._collect_rollouts(self.config['ppo']['steps_per_update'])
                
                # Update policy
                update_metrics = self._update_policy(rollouts)
                
                # Update training steps
                self.training_steps += self.config['ppo']['steps_per_update']
                self.curriculum.update_training_steps(self.config['ppo']['steps_per_update'])
                self.episode_count += len(rollouts['episode_rewards'])
                self.curriculum.update_games(len(rollouts['episode_rewards']))
                
                # Log metrics
                metrics = {
                    'training/episode_reward': np.mean(rollouts['episode_rewards']),
                    'training/episode_length': np.mean(rollouts['episode_lengths']),
                    'training/policy_loss': update_metrics['policy_loss'],
                    'training/value_loss': update_metrics['value_loss'],
                    'training/entropy_loss': update_metrics['entropy_loss'],
                    'training/training_steps': self.training_steps,
                    'training/episode_count': self.episode_count
                }
                
                self._log_metrics(metrics, self.training_steps)
                
                # Evaluation
                if self.training_steps % self.config['training']['eval_frequency'] == 0:
                    eval_metrics = self._evaluate(self.config['training']['eval_episodes'])
                    
                    eval_log_metrics = {
                        'eval/eval_reward': eval_metrics['eval_reward'],
                        'eval/eval_length': eval_metrics['eval_length'],
                        'eval/eval_reward_std': eval_metrics['eval_reward_std']
                    }
                    
                    self._log_metrics(eval_log_metrics, self.training_steps)
                    
                    # Check for best model
                    if eval_metrics['eval_reward'] > self.best_eval_score:
                        self.best_eval_score = eval_metrics['eval_reward']
                        self._save_checkpoint(is_best=True)
                    
                    # Display progress
                    self._display_progress({**metrics, **eval_log_metrics})
                
                # Save checkpoint
                if self.training_steps % self.config['training']['save_frequency'] == 0:
                    self._save_checkpoint()
                
                # Curriculum progression
                if (self.config['curriculum']['auto_progress'] and 
                    self.training_steps % self.config['curriculum']['progress_check_frequency'] == 0):
                    
                    # Get evaluation metrics for curriculum
                    eval_metrics = self._evaluate(self.config['training']['eval_episodes'])
                    
                    if self.curriculum.should_progress(eval_metrics):
                        self.curriculum.progress_to_next_phase()
                        
                        # Update environment components
                        self.env.reward_fn.set_curriculum_phase(self.curriculum.get_current_phase().name)
                        self.env.state_setter.set_curriculum_phase(self.curriculum.get_current_phase().name)
                        
                        self.console.print(f"[green]Progressed to {self.curriculum.get_current_phase().name} phase[/green]")
                
                progress.update(task, completed=self.training_steps)
        
        self.console.print("[green]Training completed![/green]")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train SSL Bot with PPO and Curriculum Learning')
    parser.add_argument('--cfg', type=str, default='configs/ppo_ssl.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--curr', type=str, default='configs/curriculum.yaml',
                       help='Path to curriculum configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--max-steps', type=int, default=1000000,
                       help='Maximum training steps')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--log-dir', type=str, default='runs/ssl_bot_training',
                       help='Directory for logging')
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints',
                       help='Directory for saving checkpoints')
    parser.add_argument('--dry_run', type=int, default=0,
                       help='If > 0, run this many env steps and exit')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Validate config files exist
    if not os.path.exists(args.cfg):
        raise FileNotFoundError(f"Config file not found: {args.cfg}")
    if not os.path.exists(args.curr):
        raise FileNotFoundError(f"Curriculum file not found: {args.curr}")
    
    # Create trainer
    trainer = PPOTrainer(args.cfg, args.curr, seed=args.seed)
    
    # Override device if specified
    if args.device != 'auto':
        trainer.device = torch.device(args.device)
        trainer.policy = trainer.policy.to(trainer.device)
        trainer.critic = trainer.critic.to(trainer.device)
    
    # Override directories if specified
    if args.log_dir != 'runs/ssl_bot_training':
        trainer.writer = SummaryWriter(log_dir=args.log_dir)
    if args.checkpoint_dir != 'models/checkpoints':
        trainer.checkpoint_dir = Path(args.checkpoint_dir)
        trainer.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dry_run:
        obs, _info = reset_env(
            trainer.env, seed=int(trainer.np_rng.integers(0, 2**32))
        )
        for _ in range(args.dry_run):
            trainer.env.action_space.seed(int(trainer.np_rng.integers(0, 2**32)))
            action = trainer.env.action_space.sample()
            obs, _, done, _ = step_env(trainer.env, action)
            if done:
                obs, _info = reset_env(
                    trainer.env, seed=int(trainer.np_rng.integers(0, 2**32))
                )
        return

    # Start training
    trainer.train(max_steps=args.max_steps, resume_from=args.resume)


if __name__ == "__main__":
    main()
