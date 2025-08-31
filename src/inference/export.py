"""
TorchScript export functionality for SSL Bot policy.
Exports trained policy to optimized TorchScript for RLBot integration.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.jit
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Local imports
from ..training.policy import SSLPolicy, create_ssl_policy


class SSLPolicyWrapper(torch.nn.Module):
    """
    Wrapper for SSL policy that includes input normalization and output processing.
    Designed for TorchScript export and RLBot integration.
    """
    
    def __init__(self, policy: SSLPolicy, obs_mean: Optional[torch.Tensor] = None, 
                 obs_std: Optional[torch.Tensor] = None):
        super().__init__()
        self.policy = policy
        self.obs_mean = obs_mean or torch.zeros(107)
        self.obs_std = obs_std or torch.ones(107)
        
        # Register buffers for TorchScript compatibility
        self.register_buffer('obs_mean_buffer', self.obs_mean)
        self.register_buffer('obs_std_buffer', self.obs_std)
    
    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with input normalization.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            
        Returns:
            Dictionary containing:
            - continuous_actions: Continuous action logits (batch_size, 5)
            - discrete_actions: Discrete action logits (batch_size, 3)
        """
        # Normalize observations
        obs_normalized = (obs - self.obs_mean_buffer) / (self.obs_std_buffer + 1e-8)
        
        # Get policy outputs
        outputs = self.policy(obs_normalized)
        
        return {
            'continuous_actions': outputs['continuous_actions'],
            'discrete_actions': outputs['discrete_actions']
        }
    
    def get_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get actions in RLBot format.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            
        Returns:
            Actions tensor of shape (batch_size, 8) in RLBot format:
            [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        """
        outputs = self.forward(obs)
        
        continuous_actions = outputs['continuous_actions']
        discrete_actions = torch.sigmoid(outputs['discrete_actions'])
        
        # Combine actions
        actions = torch.cat([continuous_actions, discrete_actions], dim=-1)
        
        return actions


class TorchScriptExporter:
    """
    Exports SSL policy to TorchScript for RLBot integration.
    """
    
    def __init__(self):
        self.console = Console()
    
    def export_policy(
        self,
        checkpoint_path: str,
        output_path: str,
        config: Optional[Dict[str, Any]] = None,
        obs_stats: Optional[Dict[str, np.ndarray]] = None,
        use_trace: bool = True
    ) -> str:
        """
        Export policy to TorchScript.
        
        Args:
            checkpoint_path: Path to model checkpoint
            output_path: Path to save TorchScript model
            config: Policy configuration
            obs_stats: Observation statistics for normalization
            use_trace: Whether to use torch.jit.trace (True) or torch.jit.script (False)
            
        Returns:
            Path to exported model
        """
        self.console.print(f"[green]Exporting policy from {checkpoint_path}[/green]")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create policy
        if config is None:
            config = {
                'obs_dim': 107,
                'hidden_sizes': [1024, 1024, 512],
                'continuous_actions': 5,
                'discrete_actions': 3,
                'use_attention': True,
                'num_heads': 8,
                'head_dim': 64,
                'dropout': 0.1
            }
        
        policy = create_ssl_policy(config)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy.eval()
        
        # Setup observation statistics
        if obs_stats is None:
            obs_mean = torch.zeros(107)
            obs_std = torch.ones(107)
        else:
            obs_mean = torch.FloatTensor(obs_stats['mean'])
            obs_std = torch.FloatTensor(obs_stats['std'])
        
        # Create wrapper
        wrapper = SSLPolicyWrapper(policy, obs_mean, obs_std)
        wrapper.eval()
        
        # Export to TorchScript
        if use_trace:
            self.console.print("[yellow]Using torch.jit.trace for export[/yellow]")
            exported_model = self._trace_export(wrapper)
        else:
            self.console.print("[yellow]Using torch.jit.script for export[/yellow]")
            exported_model = self._script_export(wrapper)
        
        # Save model
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.jit.save(exported_model, str(output_path))
        
        self.console.print(f"[green]Model exported to {output_path}[/green]")
        
        # Run smoke test
        self._smoke_test(exported_model, output_path)
        
        return str(output_path)
    
    def _trace_export(self, wrapper: SSLPolicyWrapper) -> torch.jit.ScriptModule:
        """Export using torch.jit.trace."""
        # Create dummy input
        dummy_obs = torch.randn(1, 107)
        
        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(wrapper, dummy_obs)
        
        return traced_model
    
    def _script_export(self, wrapper: SSLPolicyWrapper) -> torch.jit.ScriptModule:
        """Export using torch.jit.script."""
        # Script the model
        scripted_model = torch.jit.script(wrapper)
        
        return scripted_model
    
    def _smoke_test(self, model: torch.jit.ScriptModule, model_path: str):
        """Run smoke test on exported model."""
        self.console.print("[yellow]Running smoke test...[/yellow]")
        
        try:
            # Test with random input
            test_obs = torch.randn(1, 107)
            
            with torch.no_grad():
                outputs = model(test_obs)
            
            # Check output shapes
            assert 'continuous_actions' in outputs
            assert 'discrete_actions' in outputs
            assert outputs['continuous_actions'].shape == (1, 5)
            assert outputs['discrete_actions'].shape == (1, 3)
            
            # Test get_actions method
            actions = model.get_actions(test_obs)
            assert actions.shape == (1, 8)
            
            # Test with batch input
            batch_obs = torch.randn(4, 107)
            batch_outputs = model(batch_obs)
            assert batch_outputs['continuous_actions'].shape == (4, 5)
            assert batch_outputs['discrete_actions'].shape == (4, 3)
            
            self.console.print("[green]Smoke test passed![/green]")
            
        except Exception as e:
            self.console.print(f"[red]Smoke test failed: {e}[/red]")
            raise
    
    def load_and_test(self, model_path: str) -> torch.jit.ScriptModule:
        """Load and test exported model."""
        self.console.print(f"[yellow]Loading model from {model_path}[/yellow]")
        
        model = torch.jit.load(model_path)
        model.eval()
        
        # Test loading
        test_obs = torch.randn(1, 107)
        with torch.no_grad():
            outputs = model(test_obs)
        
        self.console.print("[green]Model loaded successfully![/green]")
        self.console.print(f"Output keys: {list(outputs.keys())}")
        self.console.print(f"Continuous actions shape: {outputs['continuous_actions'].shape}")
        self.console.print(f"Discrete actions shape: {outputs['discrete_actions'].shape}")
        
        return model


def compute_obs_stats(rollout_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute observation statistics from rollout data.
    
    Args:
        rollout_data: Dictionary containing observation data
        
    Returns:
        Dictionary containing mean and std of observations
    """
    observations = rollout_data['observations']
    
    mean = np.mean(observations, axis=0)
    std = np.std(observations, axis=0)
    
    # Avoid division by zero
    std = np.where(std < 1e-8, 1.0, std)
    
    return {
        'mean': mean,
        'std': std
    }


def main():
    parser = argparse.ArgumentParser(description='Export SSL Bot policy to TorchScript')
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--out', type=str, required=True,
                       help='Path to save TorchScript model')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to policy configuration file')
    parser.add_argument('--obs-stats', type=str, default=None,
                       help='Path to observation statistics file')
    parser.add_argument('--use-script', action='store_true',
                       help='Use torch.jit.script instead of torch.jit.trace')
    parser.add_argument('--test', action='store_true',
                       help='Test exported model')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Load observation statistics
    obs_stats = None
    if args.obs_stats:
        obs_stats = np.load(args.obs_stats)
    
    # Create exporter
    exporter = TorchScriptExporter()
    
    # Export model
    output_path = exporter.export_policy(
        checkpoint_path=args.ckpt,
        output_path=args.out,
        config=config,
        obs_stats=obs_stats,
        use_trace=not args.use_script
    )
    
    # Test model if requested
    if args.test:
        exporter.load_and_test(output_path)


if __name__ == "__main__":
    main()
