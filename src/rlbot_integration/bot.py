"""
SSL Bot for RLBot integration.
Runs the trained TorchScript policy in live Rocket League matches.
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlgym_compat import GameState

# Local imports
from .observation_adapter import RLBotObservationAdapter
from .controller_adapter import RLBotControllerAdapter


class SSLBot(BaseAgent):
    """
    SSL Bot for RLBot integration.
    Uses trained TorchScript policy for live inference.
    """
    
    def __init__(self, name: str, team: int, index: int, policy_path: str = None):
        super().__init__(name, team, index)
        
        self.policy_path = policy_path or "models/exported/ssl_policy.ts"
        self.policy = None
        self.obs_adapter = None
        self.controller_adapter = None
        
        # Timing
        self.tick_skip = 8
        self.ticks = 0
        self.prev_time = 0
        
        # State tracking
        self.game_state = None
        self.last_action = None
        
        # Performance tracking
        self.inference_times = []
        self.total_inferences = 0
        
        print(f"SSL Bot initialized - Index: {index}")
        print(f"Policy path: {self.policy_path}")
        print("Remember to run at 120fps with vsync off for best performance!")
    
    def initialize_agent(self):
        """Initialize the bot when the game starts."""
        try:
            # Load TorchScript policy
            if not os.path.exists(self.policy_path):
                raise FileNotFoundError(f"Policy file not found: {self.policy_path}")
            
            self.policy = torch.jit.load(self.policy_path, map_location='cpu')
            self.policy.eval()
            print(f"Loaded policy from {self.policy_path}")
            
            # Initialize adapters
            field_info = self.get_field_info()
            self.obs_adapter = RLBotObservationAdapter(field_info)
            self.controller_adapter = RLBotControllerAdapter()
            
            # Initialize game state
            self.game_state = GameState(field_info)
            self.ticks = self.tick_skip  # Take action on first tick
            self.prev_time = 0
            
            print("SSL Bot ready!")
            
        except Exception as e:
            print(f"Error initializing SSL Bot: {e}")
            raise
    
    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """Get bot output for the current game tick."""
        try:
            # Calculate time delta
            cur_time = packet.game_info.seconds_elapsed
            delta = cur_time - self.prev_time
            self.prev_time = cur_time
            
            # Calculate ticks elapsed
            ticks_elapsed = round(delta * 120)
            self.ticks += ticks_elapsed
            
            # Update game state
            self.obs_adapter.update_game_state(packet, ticks_elapsed)
            
            # Reset on new round
            if not packet.game_info.is_round_active:
                self.obs_adapter.reset()
                self.controller_adapter.reset_button_states()
                self.ticks = self.tick_skip
            
            # Take action if it's time
            if self.ticks >= self.tick_skip and len(packet.game_cars) > self.index:
                start_time = time.time()
                
                # Get observation
                obs = self.obs_adapter.get_observation(self.index)
                
                # Get action from policy
                action = self._get_policy_action(obs)
                
                # Convert to controller state
                controller = self.controller_adapter.actions_to_controller(
                    action['continuous_actions'],
                    action['discrete_actions']
                )
                
                # Track performance
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                self.total_inferences += 1
                
                # Keep only recent inference times
                if len(self.inference_times) > 100:
                    self.inference_times = self.inference_times[-100:]
                
                self.last_action = action
                self.ticks = 0
                
                return controller
            
            # Return last action or neutral
            if self.last_action is not None:
                return self.controller_adapter.actions_to_controller(
                    self.last_action['continuous_actions'],
                    self.last_action['discrete_actions']
                )
            else:
                return SimpleControllerState()
                
        except Exception as e:
            print(f"Error in get_output: {e}")
            return SimpleControllerState()
    
    def _get_policy_action(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get action from policy.
        
        Args:
            obs: Observation array
            
        Returns:
            Dictionary containing continuous and discrete actions
        """
        # Convert to tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dimension
        
        # Get action from policy
        with torch.no_grad():
            outputs = self.policy(obs_tensor)
        
        # Convert back to numpy
        continuous_actions = outputs['continuous_actions'].squeeze(0).numpy()
        discrete_actions = outputs['discrete_actions'].squeeze(0).numpy()
        
        return {
            'continuous_actions': continuous_actions,
            'discrete_actions': discrete_actions
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'total_inferences': self.total_inferences
        }
    
    def print_performance_stats(self):
        """Print performance statistics."""
        stats = self.get_performance_stats()
        if stats:
            print("SSL Bot Performance Stats:")
            print(f"  Average inference time: {stats['avg_inference_time']*1000:.2f}ms")
            print(f"  Max inference time: {stats['max_inference_time']*1000:.2f}ms")
            print(f"  Min inference time: {stats['min_inference_time']*1000:.2f}ms")
            print(f"  Total inferences: {stats['total_inferences']}")


# RLBot entry point
def create_agent(name: str, team: int, index: int) -> SSLBot:
    """Create SSL Bot instance for RLBot."""
    return SSLBot(name, team, index)


# For testing
if __name__ == "__main__":
    # Test the bot
    bot = SSLBot("SSLBot", 0, 0)
    print("SSL Bot created successfully!")
    
    # Test performance stats
    bot.print_performance_stats()
