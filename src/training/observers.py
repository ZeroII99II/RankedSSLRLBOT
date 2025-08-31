"""
SSL-focused observation builder for high-level Rocket League mechanics.
Builds vectorized observations aligned to SSL-level plays including:
- Aerial interceptions, wall reads, backboard attacks/clears
- Double taps, fast aerials, flip resets
- Advanced positioning and boost management
"""

import numpy as np
from typing import Any, Dict, List, Tuple
from gym import Space
from gym.spaces import Box
from rlgym.utils import ObsBuilder
from rlgym.utils.common_values import BOOST_LOCATIONS, CEILING_Z, BALL_RADIUS, CAR_MAX_SPEED
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.math import cosine_similarity


class SSLObsBuilder(ObsBuilder):
    """
    SSL-focused observation builder with static shape for consistent training.
    
    Observation shape: (obs_dim,) where obs_dim = 107
    - Self car features: 20
    - Ball features: 8  
    - Top-2 allies: 20
    - Top-2 opponents: 20
    - Geometry features: 15
    - Game context: 8
    - Boost pad availability: 16
    """
    
    def __init__(self, n_players: int = 6, tick_skip: int = 8):
        super().__init__()
        self.n_players = n_players
        self.tick_skip = tick_skip
        self.boost_locations = np.array(BOOST_LOCATIONS)
        
        # Normalization constants
        self.pos_scale = 2300.0  # Field dimensions
        self.vel_scale = CAR_MAX_SPEED
        self.ang_vel_scale = 5.5
        self.boost_scale = 100.0
        
        # Static observation dimension
        self.obs_dim = 107
        
    def reset(self, initial_state: GameState):
        """Reset observation builder state."""
        pass
        
    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        """Build observation vector for a single player."""
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        idx = 0
        
        # 1. Self car features (20)
        obs[idx:idx+3] = player.car_data.position / self.pos_scale
        idx += 3
        obs[idx:idx+3] = player.car_data.linear_velocity / self.vel_scale
        idx += 3
        obs[idx:idx+3] = player.car_data.angular_velocity / self.ang_vel_scale
        idx += 3
        obs[idx] = player.boost_amount / self.boost_scale
        idx += 1
        obs[idx] = float(player.on_ground)
        idx += 1
        obs[idx] = float(player.has_flip)
        idx += 1
        obs[idx] = float(player.has_jump)
        idx += 1
        
        # Car orientation (quaternion to euler-like features)
        forward = player.car_data.forward()
        up = player.car_data.up()
        obs[idx] = forward[2]  # pitch component
        idx += 1
        obs[idx] = np.arctan2(forward[1], forward[0]) / np.pi  # yaw
        idx += 1
        obs[idx] = np.arctan2(up[2], np.sqrt(up[0]**2 + up[1]**2)) / np.pi  # roll
        idx += 1
        
        # Last jump time (approximate from has_jump)
        obs[idx] = 0.0 if player.has_jump else 1.0
        idx += 1
        
        # 2. Ball features (8)
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        obs[idx:idx+3] = ball_pos / self.pos_scale
        idx += 3
        obs[idx:idx+3] = ball_vel / self.vel_scale
        idx += 3
        
        # Time to ground (approximate)
        if ball_vel[2] < 0 and ball_pos[2] > BALL_RADIUS:
            time_to_ground = ball_pos[2] / abs(ball_vel[2])
        else:
            time_to_ground = 10.0  # Large value if ball going up
        obs[idx] = np.clip(time_to_ground / 5.0, 0, 1)
        idx += 1
        
        # Height bucket (for aerial mechanics)
        height_bucket = np.clip(ball_pos[2] / CEILING_Z, 0, 1)
        obs[idx] = height_bucket
        idx += 1
        
        # 3. Top-2 allies (20)
        allies = [p for p in state.players if p.team_num == player.team_num and p != player]
        allies = sorted(allies, key=lambda p: np.linalg.norm(p.car_data.position - ball_pos))[:2]
        
        for ally in allies:
            rel_pos = ally.car_data.position - player.car_data.position
            rel_vel = ally.car_data.linear_velocity - player.car_data.linear_velocity
            obs[idx:idx+3] = rel_pos / self.pos_scale
            idx += 3
            obs[idx:idx+3] = rel_vel / self.vel_scale
            idx += 3
            obs[idx] = np.linalg.norm(rel_pos) / self.pos_scale
            idx += 1
            obs[idx] = ally.boost_amount / self.boost_scale
            idx += 1
            obs[idx] = float(ally.on_ground)
            idx += 1
            obs[idx] = float(ally.has_flip)
            idx += 1
        # Pad with zeros if fewer than 2 allies
        idx += (2 - len(allies)) * 10
        
        # 4. Top-2 opponents (20)
        opponents = [p for p in state.players if p.team_num != player.team_num]
        opponents = sorted(opponents, key=lambda p: np.linalg.norm(p.car_data.position - ball_pos))[:2]
        
        for opponent in opponents:
            rel_pos = opponent.car_data.position - player.car_data.position
            rel_vel = opponent.car_data.linear_velocity - player.car_data.linear_velocity
            obs[idx:idx+3] = rel_pos / self.pos_scale
            idx += 3
            obs[idx:idx+3] = rel_vel / self.vel_scale
            idx += 3
            obs[idx] = np.linalg.norm(rel_pos) / self.pos_scale
            idx += 1
            obs[idx] = opponent.boost_amount / self.boost_scale
            idx += 1
            obs[idx] = float(opponent.on_ground)
            idx += 1
            obs[idx] = float(opponent.has_flip)
            idx += 1
        # Pad with zeros if fewer than 2 opponents
        idx += (2 - len(opponents)) * 10
        
        # 5. Geometry features (15)
        # Wall/ceiling normals and distances
        car_pos = player.car_data.position
        
        # Distance to side walls
        obs[idx] = np.clip((4096 - abs(car_pos[0])) / 4096, 0, 1)
        idx += 1
        obs[idx] = np.clip((5120 - abs(car_pos[1])) / 5120, 0, 1)
        idx += 1
        obs[idx] = np.clip((CEILING_Z - car_pos[2]) / CEILING_Z, 0, 1)
        idx += 1
        
        # Backboard distance and angle
        backboard_dist = np.linalg.norm(car_pos[:2] - np.array([0, 5120]))
        obs[idx] = np.clip(backboard_dist / 5120, 0, 1)
        idx += 1
        
        # Corner proximity
        corner_dist = min(
            np.linalg.norm(car_pos[:2] - np.array([4096, 5120])),
            np.linalg.norm(car_pos[:2] - np.array([-4096, 5120])),
            np.linalg.norm(car_pos[:2] - np.array([4096, -5120])),
            np.linalg.norm(car_pos[:2] - np.array([-4096, -5120]))
        )
        obs[idx] = np.clip(corner_dist / 6000, 0, 1)
        idx += 1
        
        # Ball-wall relationships
        ball_to_wall_x = (4096 - abs(ball_pos[0])) / 4096
        ball_to_wall_y = (5120 - abs(ball_pos[1])) / 5120
        ball_to_ceiling = (CEILING_Z - ball_pos[2]) / CEILING_Z
        obs[idx:idx+3] = [ball_to_wall_x, ball_to_wall_y, ball_to_ceiling]
        idx += 3
        
        # Aerial opportunity indicators
        ball_height = ball_pos[2] / CEILING_Z
        car_height = car_pos[2] / CEILING_Z
        obs[idx] = ball_height
        idx += 1
        obs[idx] = car_height
        idx += 1
        obs[idx] = float(ball_height > 0.3)  # High ball indicator
        idx += 1
        obs[idx] = float(car_height > 0.2)  # Aerial position indicator
        idx += 1
        
        # Wall read opportunity
        wall_read_angle = self._calculate_wall_read_angle(player, ball_pos, ball_vel)
        obs[idx] = wall_read_angle
        idx += 1
        
        # 6. Game context (8)
        obs[idx] = state.blue_score - state.orange_score
        idx += 1
        obs[idx] = np.clip(state.game_seconds_remaining / 300.0, 0, 1)  # 5 min max
        idx += 1
        obs[idx] = float(state.is_overtime)
        idx += 1
        obs[idx] = float(state.is_kickoff_pause)
        idx += 1
        
        # Team advantage indicators
        blue_boost = sum(p.boost_amount for p in state.players if p.team_num == 0)
        orange_boost = sum(p.boost_amount for p in state.players if p.team_num == 1)
        boost_diff = (blue_boost - orange_boost) / (self.n_players * self.boost_scale)
        if player.team_num == 1:
            boost_diff *= -1
        obs[idx] = np.clip(boost_diff, -1, 1)
        idx += 1
        
        # Ball possession (approximate)
        ball_owner = self._get_ball_owner(state.players, ball_pos)
        obs[idx] = 1.0 if ball_owner == player.team_num else -1.0 if ball_owner is not None else 0.0
        idx += 1
        
        # Pressure indicator
        pressure = self._calculate_pressure(player, state.players, ball_pos)
        obs[idx] = pressure
        idx += 1
        
        # 7. Boost pad availability (16) - coarse representation
        # Group boost pads into regions for efficiency
        boost_regions = self._get_boost_regions()
        for region in boost_regions:
            region_available = any(
                self._is_boost_available(pad_pos, state.boost_pads)
                for pad_pos in region
            )
            obs[idx] = float(region_available)
            idx += 1
            
        return obs
    
    def _calculate_wall_read_angle(self, player: PlayerData, ball_pos: np.ndarray, ball_vel: np.ndarray) -> float:
        """Calculate angle for wall read opportunity."""
        car_pos = player.car_data.position
        
        # Predict ball trajectory to wall
        if abs(ball_vel[0]) > 0.1:  # Moving toward side wall
            time_to_wall = (4096 - abs(ball_pos[0])) / abs(ball_vel[0])
            if time_to_wall > 0 and time_to_wall < 3.0:
                predicted_pos = ball_pos + ball_vel * time_to_wall
                car_to_predicted = predicted_pos - car_pos
                if np.linalg.norm(car_to_predicted) > 0:
                    angle = cosine_similarity(car_to_predicted, player.car_data.forward())
                    return np.clip(angle, -1, 1)
        return 0.0
    
    def _get_ball_owner(self, players: List[PlayerData], ball_pos: np.ndarray) -> int:
        """Determine which team has ball possession."""
        closest_player = min(players, key=lambda p: np.linalg.norm(p.car_data.position - ball_pos))
        if np.linalg.norm(closest_player.car_data.position - ball_pos) < 200:
            return closest_player.team_num
        return None
    
    def _calculate_pressure(self, player: PlayerData, players: List[PlayerData], ball_pos: np.ndarray) -> float:
        """Calculate pressure level on the ball."""
        opponents = [p for p in players if p.team_num != player.team_num]
        if not opponents:
            return 0.0
            
        # Distance-weighted pressure
        total_pressure = 0.0
        for opponent in opponents:
            dist = np.linalg.norm(opponent.car_data.position - ball_pos)
            if dist < 1000:  # Within pressure range
                pressure = (1000 - dist) / 1000
                total_pressure += pressure
                
        return np.clip(total_pressure, 0, 1)
    
    def _get_boost_regions(self) -> List[List[np.ndarray]]:
        """Group boost pads into regions for coarse representation."""
        regions = []
        # Front/back regions
        regions.append([pos for pos in self.boost_locations if pos[1] > 2000])
        regions.append([pos for pos in self.boost_locations if pos[1] < -2000])
        # Left/right regions  
        regions.append([pos for pos in self.boost_locations if pos[0] > 2000])
        regions.append([pos for pos in self.boost_locations if pos[0] < -2000])
        # Center region
        regions.append([pos for pos in self.boost_locations if abs(pos[0]) < 2000 and abs(pos[1]) < 2000])
        # Corner regions
        regions.append([pos for pos in self.boost_locations if pos[0] > 2000 and pos[1] > 2000])
        regions.append([pos for pos in self.boost_locations if pos[0] < -2000 and pos[1] > 2000])
        regions.append([pos for pos in self.boost_locations if pos[0] > 2000 and pos[1] < -2000])
        regions.append([pos for pos in self.boost_locations if pos[0] < -2000 and pos[1] < -2000])
        
        return regions[:16]  # Limit to 16 regions
    
    def _is_boost_available(self, pad_pos: np.ndarray, boost_pads: List) -> bool:
        """Check if a boost pad is available."""
        for pad in boost_pads:
            if np.allclose(pad.position, pad_pos, atol=10):
                return pad.is_active
        return False
    
    def get_obs_space(self) -> Space:
        """Return observation space."""
        return Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
