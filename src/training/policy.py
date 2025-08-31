"""
SSL-focused policy network with MLP + attention architecture.
Designed for SSL-level mechanics with ~1-3M parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for processing car/ball/opponent relationships."""
    
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask of shape (batch_size, seq_len)
        Returns:
            output: Attended features of shape (batch_size, seq_len, embed_dim)
            attention_weights: Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Output projection and residual connection
        output = self.out_proj(attended)
        output = self.layer_norm(output + residual)
        
        return output, attention_weights.mean(dim=1)  # Average across heads


class MLPBlock(nn.Module):
    """MLP block with LayerNorm and SiLU activation."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = F.silu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x + residual


class SSLPolicy(nn.Module):
    """
    SSL-focused policy network with MLP + attention architecture.
    
    Architecture:
    - Input: observation vector (107 dims)
    - MLP backbone: 1024 → 1024 → 512
    - Optional attention over car/ball/opponent sub-tokens
    - Output heads: continuous actions (5) + discrete actions (3)
    """
    
    def __init__(
        self,
        obs_dim: int = 107,
        hidden_sizes: List[int] = [1024, 1024, 512],
        continuous_actions: int = 5,
        discrete_actions: int = 3,
        use_attention: bool = True,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.1,
        activation: str = "silu"
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.continuous_actions = continuous_actions
        self.discrete_actions = discrete_actions
        self.use_attention = use_attention
        
        # Input normalization
        self.obs_norm = nn.LayerNorm(obs_dim)
        
        # MLP backbone
        self.mlp_layers = nn.ModuleList()
        prev_dim = obs_dim
        
        for hidden_dim in hidden_sizes:
            self.mlp_layers.append(MLPBlock(prev_dim, hidden_dim, hidden_dim, dropout))
            prev_dim = hidden_dim
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = MultiHeadAttention(
                embed_dim=prev_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout
            )
            self.attention_proj = nn.Linear(obs_dim, prev_dim)
        
        # Output heads
        self.continuous_head = nn.Sequential(
            nn.Linear(prev_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, continuous_actions),
            nn.Tanh()  # Actions in [-1, 1]
        )
        
        self.discrete_head = nn.Sequential(
            nn.Linear(prev_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, discrete_actions)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            
        Returns:
            Dictionary containing:
            - continuous_actions: Continuous action logits (batch_size, continuous_actions)
            - discrete_actions: Discrete action logits (batch_size, discrete_actions)
            - attention_weights: Optional attention weights (batch_size, seq_len, seq_len)
        """
        batch_size = obs.shape[0]
        
        # Normalize observations
        x = self.obs_norm(obs)
        
        # MLP backbone
        for mlp_layer in self.mlp_layers:
            x = mlp_layer(x)
        
        # Optional attention mechanism
        attention_weights = None
        if self.use_attention:
            # Create sub-tokens from observation
            # This is a simplified approach - in practice, you might want to
            # structure the observation differently to better utilize attention
            x_attn = self.attention_proj(obs)
            x_attn = x_attn.unsqueeze(1)  # Add sequence dimension
            
            # Apply attention
            x_attn, attention_weights = self.attention(x_attn)
            x = x_attn.squeeze(1)  # Remove sequence dimension
        
        # Output heads
        continuous_actions = self.continuous_head(x)
        discrete_actions = self.discrete_head(x)
        
        return {
            'continuous_actions': continuous_actions,
            'discrete_actions': discrete_actions,
            'attention_weights': attention_weights
        }
    
    def get_action_distribution(self, obs: torch.Tensor) -> Dict[str, Any]:
        """
        Get action distributions for sampling.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Dictionary containing action distributions
        """
        outputs = self.forward(obs)
        
        # Continuous actions (Gaussian)
        continuous_mean = outputs['continuous_actions']
        continuous_std = torch.ones_like(continuous_mean) * 0.1  # Fixed std for simplicity
        continuous_dist = torch.distributions.Normal(continuous_mean, continuous_std)
        
        # Discrete actions (Bernoulli)
        discrete_logits = outputs['discrete_actions']
        discrete_dist = torch.distributions.Bernoulli(logits=discrete_logits)
        
        return {
            'continuous_dist': continuous_dist,
            'discrete_dist': discrete_dist,
            'continuous_mean': continuous_mean,
            'discrete_logits': discrete_logits
        }
    
    def sample_actions(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Sample actions from the policy.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Dictionary containing sampled actions
        """
        dists = self.get_action_distribution(obs)
        
        continuous_actions = dists['continuous_dist'].sample()
        discrete_actions = dists['discrete_dist'].sample()
        
        return {
            'continuous_actions': continuous_actions,
            'discrete_actions': discrete_actions
        }
    
    def log_prob(self, obs: torch.Tensor, actions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute log probability of actions.
        
        Args:
            obs: Observation tensor
            actions: Dictionary containing actions
            
        Returns:
            Log probability tensor
        """
        dists = self.get_action_distribution(obs)
        
        continuous_log_prob = dists['continuous_dist'].log_prob(actions['continuous_actions']).sum(dim=-1)
        discrete_log_prob = dists['discrete_dist'].log_prob(actions['discrete_actions']).sum(dim=-1)
        
        return continuous_log_prob + discrete_log_prob
    
    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of action distributions.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Entropy tensor
        """
        dists = self.get_action_distribution(obs)
        
        continuous_entropy = dists['continuous_dist'].entropy().sum(dim=-1)
        discrete_entropy = dists['discrete_dist'].entropy().sum(dim=-1)
        
        return continuous_entropy + discrete_entropy


class SSLCritic(nn.Module):
    """
    SSL-focused value network (critic).
    
    Architecture:
    - Input: observation vector (107 dims)
    - MLP backbone: 1024 → 1024 → 512
    - Output: scalar value
    """
    
    def __init__(
        self,
        obs_dim: int = 107,
        hidden_sizes: List[int] = [1024, 1024, 512],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        
        # Input normalization
        self.obs_norm = nn.LayerNorm(obs_dim)
        
        # MLP backbone
        self.mlp_layers = nn.ModuleList()
        prev_dim = obs_dim
        
        for hidden_dim in hidden_sizes:
            self.mlp_layers.append(MLPBlock(prev_dim, hidden_dim, hidden_dim, dropout))
            prev_dim = hidden_dim
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the value network.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            
        Returns:
            Value tensor of shape (batch_size, 1)
        """
        # Normalize observations
        x = self.obs_norm(obs)
        
        # MLP backbone
        for mlp_layer in self.mlp_layers:
            x = mlp_layer(x)
        
        # Value head
        value = self.value_head(x)
        
        return value


def create_ssl_policy(config: Dict[str, Any]) -> SSLPolicy:
    """Create SSL policy from configuration."""
    return SSLPolicy(
        obs_dim=config.get('obs_dim', 107),
        hidden_sizes=config.get('hidden_sizes', [1024, 1024, 512]),
        continuous_actions=config.get('continuous_actions', 5),
        discrete_actions=config.get('discrete_actions', 3),
        use_attention=config.get('use_attention', True),
        num_heads=config.get('num_heads', 8),
        head_dim=config.get('head_dim', 64),
        dropout=config.get('dropout', 0.1),
        activation=config.get('activation', 'silu')
    )


def create_ssl_critic(config: Dict[str, Any]) -> SSLCritic:
    """Create SSL critic from configuration."""
    return SSLCritic(
        obs_dim=config.get('obs_dim', 107),
        hidden_sizes=config.get('hidden_sizes', [1024, 1024, 512]),
        dropout=config.get('dropout', 0.1)
    )


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the policy network
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
    critic = create_ssl_critic(config)
    
    print(f"Policy parameters: {count_parameters(policy):,}")
    print(f"Critic parameters: {count_parameters(critic):,}")
    print(f"Total parameters: {count_parameters(policy) + count_parameters(critic):,}")
    
    # Test forward pass
    batch_size = 4
    obs = torch.randn(batch_size, 107)
    
    # Policy forward pass
    policy_outputs = policy(obs)
    print(f"Policy outputs keys: {policy_outputs.keys()}")
    print(f"Continuous actions shape: {policy_outputs['continuous_actions'].shape}")
    print(f"Discrete actions shape: {policy_outputs['discrete_actions'].shape}")
    
    # Critic forward pass
    value = critic(obs)
    print(f"Value shape: {value.shape}")
    
    # Action sampling
    actions = policy.sample_actions(obs)
    print(f"Sampled actions keys: {actions.keys()}")
    print(f"Sampled continuous actions shape: {actions['continuous_actions'].shape}")
    print(f"Sampled discrete actions shape: {actions['discrete_actions'].shape}")
