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
    
    LOCKED OBSERVATION SPECIFICATION:
    ================================
    Total observation dimension: 107 (OBS_SIZE)
    
    Feature breakdown:
    - Self car features (20): pos(3) + vel(3) + ang_vel(3) + boost(1) + on_ground(1) + 
      has_flip(1) + has_jump(1) + pitch(1) + yaw(1) + roll(1) + last_jump_time(1) + 
      car_height(1) + aerial_indicator(1) + wall_read_angle(1) + pressure(1) + 
      boost_efficiency(1) + recovery_state(1) + demo_timer(1) + demo_immunity(1)
    
    - Ball features (8): pos(3) + vel(3) + time_to_ground(1) + height_bucket(1)
    
    - Top-2 allies (20): For each ally: rel_pos(3) + rel_vel(3) + distance(1) + 
      boost(1) + on_ground(1) + has_flip(1) [padded to 2 allies]
    
    - Top-2 opponents (20): For each opponent: rel_pos(3) + rel_vel(3) + distance(1) + 
      boost(1) + on_ground(1) + has_flip(1) [padded to 2 opponents]
    
    - Geometry features (15): wall_distances(3) + backboard_dist(1) + corner_dist(1) + 
      ball_wall_relationships(3) + aerial_opportunities(2) + wall_read_opportunity(1) + 
      ceiling_proximity(1) + field_position(1) + goal_angle(1) + shot_angle(1)
    
    - Game context (8): score_diff(1) + time_remaining(1) + overtime(1) + kickoff(1) + 
      boost_advantage(1) + ball_possession(1) + pressure(1) + game_phase(1)
    
    - Boost pad availability (16): Regional boost availability (16 regions)
    
    Normalization: All values normalized to approximately [-1, 1] range
    """
    
    # LOCKED: Do not change this value without updating all dependent code
    OBS_SIZE = 107
    
    def __init__(self, n_players: int = 6, tick_skip: int = 8):
        super().__init__()
        self.n_players = n_players
        self.tick_skip = tick_skip
        self.boost_locations = np.array(BOOST_LOCATIONS)
        
        # Normalization constants - LOCKED for consistency
        self.pos_scale = 2300.0  # Field dimensions
        self.vel_scale = CAR_MAX_SPEED
        self.ang_vel_scale = 5.5
        self.boost_scale = 100.0
        
        # Static observation dimension - LOCKED
        self.obs_dim = self.OBS_SIZE
        
    def reset(self, initial_state: GameState):
        """Reset observation builder state."""
        pass
        
    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        """Build observation vector for a single player."""
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        idx = 0
        
        # 1. Self car features (20) - LOCKED SPEC
        # Position (3)
        obs[idx:idx+3] = np.clip(player.car_data.position / self.pos_scale, -1, 1)
        idx += 3
        
        # Velocity (3)
        obs[idx:idx+3] = np.clip(player.car_data.linear_velocity / self.vel_scale, -1, 1)
        idx += 3
        
        # Angular velocity (3)
        obs[idx:idx+3] = np.clip(player.car_data.angular_velocity / self.ang_vel_scale, -1, 1)
        idx += 3
        
        # Boost amount (1)
        obs[idx] = np.clip(player.boost_amount / self.boost_scale, 0, 1)
        idx += 1
        
        # On ground (1)
        obs[idx] = float(player.on_ground)
        idx += 1
        
        # Has flip (1)
        obs[idx] = float(player.has_flip)
        idx += 1
        
        # Has jump (1)
        obs[idx] = float(player.has_jump)
        idx += 1
        
        # Car orientation - pitch (1)
        forward = player.car_data.forward()
        obs[idx] = np.clip(forward[2], -1, 1)
        idx += 1
        
        # Car orientation - yaw (1)
        obs[idx] = np.clip(np.arctan2(forward[1], forward[0]) / np.pi, -1, 1)
        idx += 1
        
        # Car orientation - roll (1)
        up = player.car_data.up()
        obs[idx] = np.clip(np.arctan2(up[2], np.sqrt(up[0]**2 + up[1]**2)) / np.pi, -1, 1)
        idx += 1
        
        # Last jump time (1)
        obs[idx] = 0.0 if player.has_jump else 1.0
        idx += 1
        
        # Car height (1)
        car_height = player.car_data.position[2] / CEILING_Z
        obs[idx] = np.clip(car_height, 0, 1)
        idx += 1
        
        # Aerial indicator (1)
        obs[idx] = float(car_height > 0.2)
        idx += 1
        
        # Wall read angle (1)
        wall_read_angle = self._calculate_wall_read_angle(player, state.ball.position, state.ball.linear_velocity)
        obs[idx] = np.clip(wall_read_angle, -1, 1)
        idx += 1
        
        # Pressure (1)
        pressure = self._calculate_pressure(player, state.players, state.ball.position)
        obs[idx] = np.clip(pressure, 0, 1)
        idx += 1
        
        # Boost efficiency (1) - approximate from velocity vs boost usage
        speed = np.linalg.norm(player.car_data.linear_velocity)
        boost_efficiency = speed / (player.boost_amount + 1e-6)
        obs[idx] = np.clip(boost_efficiency / 10.0, 0, 1)
        idx += 1
        
        # Recovery state (1)
        recovery_state = 1.0 if player.on_ground and player.has_flip else 0.0
        obs[idx] = recovery_state
        idx += 1
        
        # Demo timer (1) - approximate
        obs[idx] = 0.0  # Placeholder - would need demo state tracking
        idx += 1
        
        # Demo immunity (1) - approximate
        obs[idx] = 0.0  # Placeholder - would need demo state tracking
        idx += 1
        
        # 2. Ball features (8) - LOCKED SPEC
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        
        # Ball position (3)
        obs[idx:idx+3] = np.clip(ball_pos / self.pos_scale, -1, 1)
        idx += 3
        
        # Ball velocity (3)
        obs[idx:idx+3] = np.clip(ball_vel / self.vel_scale, -1, 1)
        idx += 3
        
        # Time to ground (1)
        if ball_vel[2] < 0 and ball_pos[2] > BALL_RADIUS:
            time_to_ground = ball_pos[2] / abs(ball_vel[2])
        else:
            time_to_ground = 10.0  # Large value if ball going up
        obs[idx] = np.clip(time_to_ground / 5.0, 0, 1)
        idx += 1
        
        # Height bucket (1)
        height_bucket = np.clip(ball_pos[2] / CEILING_Z, 0, 1)
        obs[idx] = height_bucket
        idx += 1
        
        # 3. Top-2 allies (20) - LOCKED SPEC
        allies = [p for p in state.players if p.team_num == player.team_num and p != player]
        allies = sorted(allies, key=lambda p: np.linalg.norm(p.car_data.position - ball_pos))[:2]
        
        for ally in allies:
            rel_pos = ally.car_data.position - player.car_data.position
            rel_vel = ally.car_data.linear_velocity - player.car_data.linear_velocity
            # Relative position (3)
            obs[idx:idx+3] = np.clip(rel_pos / self.pos_scale, -1, 1)
            idx += 3
            # Relative velocity (3)
            obs[idx:idx+3] = np.clip(rel_vel / self.vel_scale, -1, 1)
            idx += 3
            # Distance (1)
            obs[idx] = np.clip(np.linalg.norm(rel_pos) / self.pos_scale, 0, 1)
            idx += 1
            # Boost (1)
            obs[idx] = np.clip(ally.boost_amount / self.boost_scale, 0, 1)
            idx += 1
            # On ground (1)
            obs[idx] = float(ally.on_ground)
            idx += 1
            # Has flip (1)
            obs[idx] = float(ally.has_flip)
            idx += 1
        
        # Pad with zeros if fewer than 2 allies (10 features per ally)
        idx += (2 - len(allies)) * 10
        
        # 4. Top-2 opponents (20) - LOCKED SPEC
        opponents = [p for p in state.players if p.team_num != player.team_num]
        opponents = sorted(opponents, key=lambda p: np.linalg.norm(p.car_data.position - ball_pos))[:2]
        
        for opponent in opponents:
            rel_pos = opponent.car_data.position - player.car_data.position
            rel_vel = opponent.car_data.linear_velocity - player.car_data.linear_velocity
            # Relative position (3)
            obs[idx:idx+3] = np.clip(rel_pos / self.pos_scale, -1, 1)
            idx += 3
            # Relative velocity (3)
            obs[idx:idx+3] = np.clip(rel_vel / self.vel_scale, -1, 1)
            idx += 3
            # Distance (1)
            obs[idx] = np.clip(np.linalg.norm(rel_pos) / self.pos_scale, 0, 1)
            idx += 1
            # Boost (1)
            obs[idx] = np.clip(opponent.boost_amount / self.boost_scale, 0, 1)
            idx += 1
            # On ground (1)
            obs[idx] = float(opponent.on_ground)
            idx += 1
            # Has flip (1)
            obs[idx] = float(opponent.has_flip)
            idx += 1
        
        # Pad with zeros if fewer than 2 opponents (10 features per opponent)
        idx += (2 - len(opponents)) * 10
        
        # 5. Geometry features (15) - LOCKED SPEC
        car_pos = player.car_data.position
        
        # Wall distances (3)
        obs[idx] = np.clip((4096 - abs(car_pos[0])) / 4096, 0, 1)  # Side walls
        idx += 1
        obs[idx] = np.clip((5120 - abs(car_pos[1])) / 5120, 0, 1)  # Front/back walls
        idx += 1
        obs[idx] = np.clip((CEILING_Z - car_pos[2]) / CEILING_Z, 0, 1)  # Ceiling
        idx += 1
        
        # Backboard distance (1)
        backboard_dist = np.linalg.norm(car_pos[:2] - np.array([0, 5120]))
        obs[idx] = np.clip(backboard_dist / 5120, 0, 1)
        idx += 1
        
        # Corner proximity (1)
        corner_dist = min(
            np.linalg.norm(car_pos[:2] - np.array([4096, 5120])),
            np.linalg.norm(car_pos[:2] - np.array([-4096, 5120])),
            np.linalg.norm(car_pos[:2] - np.array([4096, -5120])),
            np.linalg.norm(car_pos[:2] - np.array([-4096, -5120]))
        )
        obs[idx] = np.clip(corner_dist / 6000, 0, 1)
        idx += 1
        
        # Ball-wall relationships (3)
        ball_to_wall_x = (4096 - abs(ball_pos[0])) / 4096
        ball_to_wall_y = (5120 - abs(ball_pos[1])) / 5120
        ball_to_ceiling = (CEILING_Z - ball_pos[2]) / CEILING_Z
        obs[idx:idx+3] = [ball_to_wall_x, ball_to_wall_y, ball_to_ceiling]
        idx += 3
        
        # Aerial opportunities (2)
        ball_height = ball_pos[2] / CEILING_Z
        obs[idx] = float(ball_height > 0.3)  # High ball indicator
        idx += 1
        obs[idx] = float(car_height > 0.2)  # Aerial position indicator
        idx += 1
        
        # Wall read opportunity (1)
        wall_read_angle = self._calculate_wall_read_angle(player, ball_pos, ball_vel)
        obs[idx] = np.clip(wall_read_angle, -1, 1)
        idx += 1
        
        # Ceiling proximity (1)
        obs[idx] = np.clip((CEILING_Z - car_pos[2]) / CEILING_Z, 0, 1)
        idx += 1
        
        # Field position (1) - normalized position on field
        field_pos = np.linalg.norm(car_pos[:2]) / 6000
        obs[idx] = np.clip(field_pos, 0, 1)
        idx += 1
        
        # Goal angle (1) - angle to goal
        goal_pos = np.array([0, 5120 if player.team_num == 0 else -5120, 0])
        goal_vec = goal_pos - car_pos
        goal_angle = np.arccos(np.clip(np.dot(goal_vec[:2], player.car_data.forward()[:2]) / 
                                      (np.linalg.norm(goal_vec[:2]) + 1e-6), -1, 1))
        obs[idx] = np.clip(goal_angle / np.pi, 0, 1)
        idx += 1
        
        # Shot angle (1) - angle from ball to goal
        ball_to_goal = goal_pos - ball_pos
        shot_angle = np.arccos(np.clip(np.dot(ball_to_goal[:2], ball_vel[:2]) / 
                                      (np.linalg.norm(ball_to_goal[:2]) * np.linalg.norm(ball_vel[:2]) + 1e-6), -1, 1))
        obs[idx] = np.clip(shot_angle / np.pi, 0, 1)
        idx += 1
        
        # 6. Game context (8) - LOCKED SPEC
        # Score difference (1)
        score_diff = state.blue_score - state.orange_score
        if player.team_num == 1:
            score_diff *= -1
        obs[idx] = np.clip(score_diff / 10.0, -1, 1)
        idx += 1
        
        # Time remaining (1)
        obs[idx] = np.clip(state.game_seconds_remaining / 300.0, 0, 1)  # 5 min max
        idx += 1
        
        # Overtime (1)
        obs[idx] = float(state.is_overtime)
        idx += 1
        
        # Kickoff (1)
        obs[idx] = float(state.is_kickoff_pause)
        idx += 1
        
        # Boost advantage (1)
        blue_boost = sum(p.boost_amount for p in state.players if p.team_num == 0)
        orange_boost = sum(p.boost_amount for p in state.players if p.team_num == 1)
        boost_diff = (blue_boost - orange_boost) / (self.n_players * self.boost_scale)
        if player.team_num == 1:
            boost_diff *= -1
        obs[idx] = np.clip(boost_diff, -1, 1)
        idx += 1
        
        # Ball possession (1)
        ball_owner = self._get_ball_owner(state.players, ball_pos)
        obs[idx] = 1.0 if ball_owner == player.team_num else -1.0 if ball_owner is not None else 0.0
        idx += 1
        
        # Pressure (1)
        pressure = self._calculate_pressure(player, state.players, ball_pos)
        obs[idx] = np.clip(pressure, 0, 1)
        idx += 1
        
        # Game phase (1) - early/mid/late game
        game_phase = 1.0 - (state.game_seconds_remaining / 300.0)
        obs[idx] = np.clip(game_phase, 0, 1)
        idx += 1
        
        # 7. Boost pad availability (16) - LOCKED SPEC
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
