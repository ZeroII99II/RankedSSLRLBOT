"""
Controller adapter for RLBot integration.
Converts policy actions to RLBot SimpleControllerState.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from rlbot.agents.base_agent import SimpleControllerState


class RLBotControllerAdapter:
    """
    Adapter to convert policy actions to RLBot controller state.
    Handles action clamping and button hysteresis.
    """
    
    def __init__(self, button_threshold: float = 0.5, button_hysteresis: float = 0.1):
        self.button_threshold = button_threshold
        self.button_hysteresis = button_hysteresis
        
        # Button state tracking for hysteresis
        self.button_states = {
            'jump': False,
            'boost': False,
            'handbrake': False
        }
    
    def actions_to_controller(
        self, 
        continuous_actions: np.ndarray, 
        discrete_actions: np.ndarray
    ) -> SimpleControllerState:
        """
        Convert policy actions to RLBot controller state.
        
        Args:
            continuous_actions: Array of shape (5,) containing [throttle, steer, pitch, yaw, roll]
            discrete_actions: Array of shape (3,) containing [jump, boost, handbrake] probabilities
            
        Returns:
            SimpleControllerState for RLBot
        """
        # Clamp continuous actions to valid ranges
        throttle = np.clip(continuous_actions[0], -1.0, 1.0)
        steer = np.clip(continuous_actions[1], -1.0, 1.0)
        pitch = np.clip(continuous_actions[2], -1.0, 1.0)
        yaw = np.clip(continuous_actions[3], -1.0, 1.0)
        roll = np.clip(continuous_actions[4], -1.0, 1.0)
        
        # Convert discrete actions to buttons with hysteresis
        jump = self._button_with_hysteresis(discrete_actions[0], 'jump')
        boost = self._button_with_hysteresis(discrete_actions[1], 'boost')
        handbrake = self._button_with_hysteresis(discrete_actions[2], 'handbrake')
        
        # Create controller state
        controller = SimpleControllerState()
        controller.throttle = float(throttle)
        controller.steer = float(steer)
        controller.pitch = float(pitch)
        controller.yaw = float(yaw)
        controller.roll = float(roll)
        controller.jump = jump
        controller.boost = boost
        controller.handbrake = handbrake
        
        return controller
    
    def _button_with_hysteresis(self, probability: float, button_name: str) -> bool:
        """
        Convert probability to button state with hysteresis to prevent rapid switching.
        
        Args:
            probability: Button activation probability
            button_name: Name of the button for state tracking
            
        Returns:
            Boolean button state
        """
        current_state = self.button_states[button_name]
        
        if current_state:
            # Button is currently pressed
            # Only release if probability drops below threshold - hysteresis
            if probability < (self.button_threshold - self.button_hysteresis):
                self.button_states[button_name] = False
        else:
            # Button is currently not pressed
            # Only press if probability exceeds threshold
            if probability > self.button_threshold:
                self.button_states[button_name] = True
        
        return self.button_states[button_name]
    
    def reset_button_states(self):
        """Reset all button states."""
        for button in self.button_states:
            self.button_states[button] = False
    
    def get_button_states(self) -> Dict[str, bool]:
        """Get current button states."""
        return self.button_states.copy()
