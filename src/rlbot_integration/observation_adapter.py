"""
Observation adapter for RLBot integration.
Converts RLBot GameTickPacket to SSL observation format.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlgym_compat import GameState

# Local imports
from ..training.observers import SSLObsBuilder


class RLBotObservationAdapter:
    """
    Adapter to convert RLBot GameTickPacket to SSL observation format.
    Mirrors the training observation builder for consistency.
    """
    
    def __init__(self, field_info=None):
        self.field_info = field_info
        self.obs_builder = SSLObsBuilder(n_players=6, tick_skip=8)
        self.game_state = None
        
    def update_game_state(self, packet: GameTickPacket, ticks_elapsed: int):
        """Update internal game state from packet."""
        if self.game_state is None:
            self.game_state = GameState(self.field_info)
        
        self.game_state.decode(packet, ticks_elapsed)
    
    def get_observation(self, player_index: int) -> np.ndarray:
        """
        Get observation for a specific player.
        
        Args:
            player_index: Index of the player to get observation for
            
        Returns:
            Observation array of shape (107,)
        """
        if self.game_state is None:
            raise ValueError("Game state not initialized. Call update_game_state first.")
        
        # Get player data
        player = self.game_state.players[player_index]
        
        # Build observation using the same logic as training
        obs = self.obs_builder.build_obs(player, self.game_state, np.zeros(8))
        
        return obs
    
    def get_all_observations(self) -> List[np.ndarray]:
        """
        Get observations for all players.
        
        Returns:
            List of observation arrays, one per player
        """
        if self.game_state is None:
            raise ValueError("Game state not initialized. Call update_game_state first.")
        
        observations = []
        for i in range(len(self.game_state.players)):
            obs = self.get_observation(i)
            observations.append(obs)
        
        return observations
    
    def reset(self):
        """Reset observation builder state."""
        if self.game_state is not None:
            self.obs_builder.reset(self.game_state)
