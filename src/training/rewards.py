"""
SSL-focused reward system with curriculum-based weighting.
Supports rank-aware rewards from Bronze to SSL with different emphasis
on basic mechanics vs advanced aerial/wall plays.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from rlgym.api.config import RewardFunction
from rlgym.rocket_league.common_values import (
    BALL_MAX_SPEED,
    CAR_MAX_SPEED,
    CEILING_Z,
    BALL_RADIUS,
    BLUE_GOAL_BACK,
    BLUE_GOAL_CENTER,
    ORANGE_GOAL_BACK,
    ORANGE_GOAL_CENTER,
    GOAL_HEIGHT,
    CAR_MAX_ANG_VEL,
    ORANGE_TEAM,
)
from rlgym.rocket_league.api import GameState


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class ModernRewardSystem(RewardFunction):
    """
    SSL-focused reward function with curriculum-based weighting.
    
    Supports different reward emphasis based on training phase:
    - Bronze/Silver: Basic movement, touches, recoveries
    - Gold/Plat: Boost management, positioning, basic aerials
    - Diamond/Champ: Advanced positioning, power shots, dribbles
    - GC/SSL: Aerial mechanics, wall reads, double taps, flip resets
    """
    
    def __init__(
        self,
        curriculum_phase: str = "bronze",
        team_spirit: float = 0.6,
        # Universal rewards
        goal_w: float = 10.0,
        ball_to_goal_velocity_w: float = 2.0,
        touch_quality_w: float = 1.5,
        recovery_w: float = 1.0,
        boost_economy_w: float = 0.8,
        demo_evade_w: float = 0.5,
        demo_opportunity_w: float = 0.5,
        # Defense rewards
        shadowing_w: float = 0.3,
        save_quality_w: float = 2.0,
        clear_quality_w: float = 1.5,
        # Advanced aerial rewards
        aerial_intercept_w: float = 1.0,
        backboard_read_w: float = 1.5,
        double_tap_setup_w: float = 2.0,
        flip_reset_w: float = 3.0,
        fast_aerial_w: float = 1.0,
        # Penalties
        own_goal_risk_w: float = -2.0,
        panic_jump_w: float = -0.5,
        bad_touch_w: float = -0.3,
        idle_w: float = -0.1
    ):
        super().__init__()
        
        self.curriculum_phase = curriculum_phase
        self.team_spirit = team_spirit
        
        # Store weights
        self.weights = {
            'goal': goal_w,
            'ball_to_goal_velocity': ball_to_goal_velocity_w,
            'touch_quality': touch_quality_w,
            'recovery': recovery_w,
            'boost_economy': boost_economy_w,
            'demo_evade': demo_evade_w,
            'demo_opportunity': demo_opportunity_w,
            'shadowing': shadowing_w,
            'save_quality': save_quality_w,
            'clear_quality': clear_quality_w,
            'aerial_intercept': aerial_intercept_w,
            'backboard_read': backboard_read_w,
            'double_tap_setup': double_tap_setup_w,
            'flip_reset': flip_reset_w,
            'fast_aerial': fast_aerial_w,
            'own_goal_risk': own_goal_risk_w,
            'panic_jump': panic_jump_w,
            'bad_touch': bad_touch_w,
            'idle': idle_w
        }
        
        # Goal positions
        self.blue_goal = (np.array(BLUE_GOAL_BACK) + np.array(BLUE_GOAL_CENTER)) / 2
        self.orange_goal = (np.array(ORANGE_GOAL_BACK) + np.array(ORANGE_GOAL_CENTER)) / 2
        
        # State tracking
        self.last_state: Optional[GameState] = None
        self.last_ball_vel: Optional[np.ndarray] = None
        self.last_car_positions: Optional[Dict[int, np.ndarray]] = None
        self.last_boost_amounts: Optional[Dict[int, float]] = None
        self.touch_history: Optional[Dict[int, List[float]]] = None
        
    def set_curriculum_phase(self, phase: str):
        """Update curriculum phase and adjust reward weights."""
        self.curriculum_phase = phase
        self._update_weights_for_phase()
        
    def _update_weights_for_phase(self):
        """Adjust reward weights based on curriculum phase."""
        phase_multipliers = {
            'bronze': {
                'ball_to_goal_velocity': 0.3,
                'touch_quality': 0.5,
                'recovery': 1.0,
                'boost_economy': 0.5,
                'aerial_intercept': 0.1,
                'backboard_read': 0.1,
                'double_tap_setup': 0.1,
                'flip_reset': 0.1,
                'fast_aerial': 0.1
            },
            'silver': {
                'ball_to_goal_velocity': 0.5,
                'touch_quality': 0.7,
                'recovery': 1.0,
                'boost_economy': 0.7,
                'aerial_intercept': 0.2,
                'backboard_read': 0.2,
                'double_tap_setup': 0.1,
                'flip_reset': 0.1,
                'fast_aerial': 0.2
            },
            'gold': {
                'ball_to_goal_velocity': 0.7,
                'touch_quality': 1.0,
                'recovery': 1.0,
                'boost_economy': 1.0,
                'aerial_intercept': 0.4,
                'backboard_read': 0.3,
                'double_tap_setup': 0.2,
                'flip_reset': 0.1,
                'fast_aerial': 0.4
            },
            'platinum': {
                'ball_to_goal_velocity': 0.8,
                'touch_quality': 1.0,
                'recovery': 1.0,
                'boost_economy': 1.0,
                'aerial_intercept': 0.6,
                'backboard_read': 0.5,
                'double_tap_setup': 0.3,
                'flip_reset': 0.2,
                'fast_aerial': 0.6
            },
            'diamond': {
                'ball_to_goal_velocity': 1.0,
                'touch_quality': 1.0,
                'recovery': 1.0,
                'boost_economy': 1.0,
                'aerial_intercept': 0.8,
                'backboard_read': 0.7,
                'double_tap_setup': 0.5,
                'flip_reset': 0.3,
                'fast_aerial': 0.8
            },
            'champion': {
                'ball_to_goal_velocity': 1.0,
                'touch_quality': 1.0,
                'recovery': 1.0,
                'boost_economy': 1.0,
                'aerial_intercept': 1.0,
                'backboard_read': 0.9,
                'double_tap_setup': 0.7,
                'flip_reset': 0.5,
                'fast_aerial': 1.0
            },
            'gc': {
                'ball_to_goal_velocity': 1.0,
                'touch_quality': 1.0,
                'recovery': 1.0,
                'boost_economy': 1.0,
                'aerial_intercept': 1.0,
                'backboard_read': 1.0,
                'double_tap_setup': 0.9,
                'flip_reset': 0.7,
                'fast_aerial': 1.0
            },
            'ssl': {
                'ball_to_goal_velocity': 1.0,
                'touch_quality': 1.0,
                'recovery': 1.0,
                'boost_economy': 1.0,
                'aerial_intercept': 1.0,
                'backboard_read': 1.0,
                'double_tap_setup': 1.0,
                'flip_reset': 1.0,
                'fast_aerial': 1.0
            }
        }
        
        if self.curriculum_phase in phase_multipliers:
            for reward_type, multiplier in phase_multipliers[self.curriculum_phase].items():
                if reward_type in self.weights:
                    self.weights[reward_type] *= multiplier
    
    def reset(self, initial_state: GameState):
        """Reset reward function state."""
        self.last_state = initial_state
        self.last_ball_vel = initial_state.ball.linear_velocity.copy()
        self.last_car_positions = {
            i: p.car_data.position.copy() for i, p in enumerate(initial_state.players)
        }
        self.last_boost_amounts = {
            i: p.boost_amount for i, p in enumerate(initial_state.players)
        }
        self.touch_history = {i: [] for i in range(len(initial_state.players))}
    
    def get_reward(self, player: Any, state: GameState, previous_action: np.ndarray) -> float:
        """Calculate reward for a single player."""
        if self.last_state is None:
            return 0.0
            
        player_idx = self._get_player_index(player, state)
        if player_idx is None:
            return 0.0
            
        reward = 0.0
        
        # Universal rewards
        reward += self._ball_to_goal_velocity_reward(player, state)
        reward += self._touch_quality_reward(player, state, player_idx)
        reward += self._recovery_reward(player, state, player_idx)
        reward += self._boost_economy_reward(player, state, player_idx)
        reward += self._demo_rewards(player, state, player_idx)
        
        # Defense rewards
        reward += self._shadowing_reward(player, state)
        reward += self._save_quality_reward(player, state)
        reward += self._clear_quality_reward(player, state)
        
        # Advanced aerial rewards (weighted by curriculum phase)
        reward += self._aerial_intercept_reward(player, state)
        reward += self._backboard_read_reward(player, state)
        reward += self._double_tap_setup_reward(player, state)
        reward += self._flip_reset_reward(player, state)
        reward += self._fast_aerial_reward(player, state)
        
        # Penalties
        reward += self._own_goal_risk_penalty(player, state)
        reward += self._panic_jump_penalty(player, state, previous_action)
        reward += self._bad_touch_penalty(player, state)
        reward += self._idle_penalty(player, state, player_idx)
        
        # Update state tracking
        self._update_state_tracking(player, state, player_idx)
        
        return float(reward)
    
    def _get_player_index(self, player: Any, state: GameState) -> Optional[int]:
        """Get player index in state.players list."""
        for i, p in enumerate(state.players):
            if (p.car_data.position == player.car_data.position).all():
                return i
        return None
    
    def _ball_to_goal_velocity_reward(self, player: Any, state: GameState) -> float:
        """Reward for ball velocity toward opponent goal."""
        ball_vel = state.ball.linear_velocity
        ball_pos = state.ball.position
        
        # Determine target goal
        if player.team_num == 0:  # Blue team
            target_goal = self.orange_goal
        else:  # Orange team
            target_goal = self.blue_goal
            
        # Calculate velocity toward goal
        to_goal = target_goal - ball_pos
        if np.linalg.norm(to_goal) > 0:
            to_goal_norm = to_goal / np.linalg.norm(to_goal)
            goal_velocity = np.dot(ball_vel, to_goal_norm)
            
            # Scale by distance (closer to goal = more important)
            distance_factor = np.exp(-np.linalg.norm(to_goal) / 2000)
            return self.weights['ball_to_goal_velocity'] * goal_velocity / BALL_MAX_SPEED * distance_factor
            
        return 0.0
    
    def _touch_quality_reward(self, player: Any, state: GameState, player_idx: int) -> float:
        """Reward for high-quality ball touches."""
        if not player.ball_touched:
            return 0.0
            
        # Calculate touch quality based on ball speed change
        if self.last_state is not None:
            ball_speed_change = np.linalg.norm(state.ball.linear_velocity) - np.linalg.norm(self.last_state.ball.linear_velocity)
            
            # Reward positive speed changes toward goal
            ball_vel = state.ball.linear_velocity
            ball_pos = state.ball.position
            
            if player.team_num == 0:
                target_goal = self.orange_goal
            else:
                target_goal = self.blue_goal
                
            to_goal = target_goal - ball_pos
            if np.linalg.norm(to_goal) > 0:
                to_goal_norm = to_goal / np.linalg.norm(to_goal)
                goal_direction_velocity = np.dot(ball_vel, to_goal_norm)
                
                # Reward touches that increase ball speed toward goal
                if goal_direction_velocity > 0:
                    touch_quality = ball_speed_change * goal_direction_velocity / BALL_MAX_SPEED
                    return self.weights['touch_quality'] * touch_quality
                    
        return 0.0
    
    def _recovery_reward(self, player: Any, state: GameState, player_idx: int) -> float:
        """Reward for quick recoveries (upright, wheels on ground)."""
        if self.last_state is None:
            return 0.0
            
        last_player = self.last_state.players[player_idx]
        
        # Reward getting upright quickly
        upright_reward = 0.0
        if not last_player.on_ground and player.on_ground:
            # Time spent in air (approximate)
            air_time = 0.1  # Assume 0.1s per step
            upright_reward = np.exp(-air_time)  # Exponential decay
            
        # Reward wheels on ground
        wheels_reward = 0.0
        if player.on_ground and not last_player.on_ground:
            wheels_reward = 0.5
            
        return self.weights['recovery'] * (upright_reward + wheels_reward)
    
    def _boost_economy_reward(self, player: Any, state: GameState, player_idx: int) -> float:
        """Reward for efficient boost usage."""
        if self.last_state is None:
            return 0.0
            
        last_boost = self.last_boost_amounts.get(player_idx, 0)
        current_boost = player.boost_amount
        
        boost_change = current_boost - last_boost
        
        # Reward boost collection
        if boost_change > 0:
            return self.weights['boost_economy'] * boost_change / 100.0
            
        # Penalize boost waste (but less harshly)
        elif boost_change < 0:
            # Check if boost was used effectively
            car_speed = np.linalg.norm(player.car_data.linear_velocity)
            if car_speed > CAR_MAX_SPEED * 0.8:  # High speed = good boost usage
                return 0.0
            else:
                return self.weights['boost_economy'] * boost_change / 200.0
                
        return 0.0
    
    def _demo_rewards(self, player: Any, state: GameState, player_idx: int) -> float:
        """Reward for demo evasion and opportunities."""
        if self.last_state is None:
            return 0.0
            
        last_player = self.last_state.players[player_idx]
        reward = 0.0
        
        # Demo evasion reward
        if last_player.is_demoed and not player.is_demoed:
            reward += self.weights['demo_evade'] * 0.5
            
        # Demo opportunity reward
        if player.match_demolishes > last_player.match_demolishes:
            reward += self.weights['demo_opportunity'] * 0.5
            
        return reward
    
    def _shadowing_reward(self, player: Any, state: GameState) -> float:
        """Reward for good defensive shadowing."""
        ball_pos = state.ball.position
        car_pos = player.car_data.position
        
        # Find closest opponent to ball
        opponents = [p for p in state.players if p.team_num != player.team_num]
        if not opponents:
            return 0.0
            
        closest_opponent = min(opponents, key=lambda p: np.linalg.norm(p.car_data.position - ball_pos))
        opp_pos = closest_opponent.car_data.position
        
        # Calculate shadowing line (between opponent and goal)
        if player.team_num == 0:  # Blue team defending
            goal_pos = self.blue_goal
        else:  # Orange team defending
            goal_pos = self.orange_goal
            
        # Distance to shadowing line
        shadowing_line = goal_pos - opp_pos
        if np.linalg.norm(shadowing_line) > 0:
            shadowing_line_norm = shadowing_line / np.linalg.norm(shadowing_line)
            car_to_line = car_pos - opp_pos
            distance_to_line = np.linalg.norm(car_to_line - np.dot(car_to_line, shadowing_line_norm) * shadowing_line_norm)
            
            # Reward being close to shadowing line
            shadowing_quality = np.exp(-distance_to_line / 500)
            return self.weights['shadowing'] * shadowing_quality
            
        return 0.0
    
    def _save_quality_reward(self, player: Any, state: GameState) -> float:
        """Reward for high-quality saves."""
        # This would need more complex logic to detect save situations
        # For now, return 0
        return 0.0
    
    def _clear_quality_reward(self, player: Any, state: GameState) -> float:
        """Reward for high-quality clears."""
        # This would need more complex logic to detect clear situations
        # For now, return 0
        return 0.0
    
    def _aerial_intercept_reward(self, player: Any, state: GameState) -> float:
        """Reward for successful aerial interceptions."""
        if not player.ball_touched or player.on_ground:
            return 0.0
            
        # Check if this was an aerial touch
        car_height = player.car_data.position[2]
        ball_height = state.ball.position[2]
        
        if car_height > BALL_RADIUS * 2 and ball_height > BALL_RADIUS * 2:
            # Aerial touch - reward based on height and ball control
            height_factor = min(car_height / CEILING_Z, 1.0)
            ball_control = np.linalg.norm(state.ball.linear_velocity) / BALL_MAX_SPEED
            
            return self.weights['aerial_intercept'] * height_factor * ball_control
            
        return 0.0
    
    def _backboard_read_reward(self, player: Any, state: GameState) -> float:
        """Reward for successful backboard reads."""
        if not player.ball_touched:
            return 0.0
            
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        
        # Check if ball is near backboard
        backboard_distance = abs(ball_pos[1]) - 5120
        if backboard_distance < 200 and abs(ball_vel[1]) > 500:
            # Ball is near backboard with significant velocity
            car_pos = player.car_data.position
            car_to_ball = ball_pos - car_pos
            
            # Reward if car is positioned well for backboard read
            if np.linalg.norm(car_to_ball) < 500 and car_pos[2] > BALL_RADIUS:
                return self.weights['backboard_read'] * 0.5
                
        return 0.0
    
    def _double_tap_setup_reward(self, player: Any, state: GameState) -> float:
        """Reward for setting up double taps."""
        if not player.ball_touched:
            return 0.0
            
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        
        # Check if ball is set up for double tap (high, moving toward backboard)
        if (ball_pos[2] > CEILING_Z * 0.3 and 
            abs(ball_vel[1]) > 800 and 
            abs(ball_pos[1]) > 3000):
            
            car_pos = player.car_data.position
            car_vel = player.car_data.linear_velocity
            
            # Reward if car is following ball trajectory
            if (np.linalg.norm(car_vel) > CAR_MAX_SPEED * 0.5 and
                car_pos[2] > BALL_RADIUS):
                return self.weights['double_tap_setup'] * 0.3
                
        return 0.0
    
    def _flip_reset_reward(self, player: Any, state: GameState) -> float:
        """Reward for successful flip resets."""
        if (player.ball_touched and 
            player.has_flip and 
            not self.last_state.players[self._get_player_index(player, state)].has_flip):
            
            car_pos = player.car_data.position
            ball_pos = state.ball.position
            
            # Check if this was a flip reset (car under ball, high up)
            if (car_pos[2] > CEILING_Z * 0.2 and 
                np.linalg.norm(car_pos - ball_pos) < BALL_RADIUS * 2 and
                ball_pos[2] - car_pos[2] > BALL_RADIUS):
                
                return self.weights['flip_reset'] * 1.0
                
        return 0.0
    
    def _fast_aerial_reward(self, player: Any, state: GameState) -> float:
        """Reward for fast aerials."""
        if not player.ball_touched or player.on_ground:
            return 0.0
            
        car_vel = player.car_data.linear_velocity
        car_pos = player.car_data.position
        ball_pos = state.ball.position
        
        # Check if this was a fast aerial (high speed, upward movement)
        if (car_pos[2] > BALL_RADIUS and 
            car_vel[2] > 0 and 
            np.linalg.norm(car_vel) > CAR_MAX_SPEED * 0.7):
            
            # Reward based on speed and height
            speed_factor = np.linalg.norm(car_vel) / CAR_MAX_SPEED
            height_factor = min(car_pos[2] / CEILING_Z, 1.0)
            
            return self.weights['fast_aerial'] * speed_factor * height_factor
            
        return 0.0
    
    def _own_goal_risk_penalty(self, player: Any, state: GameState) -> float:
        """Penalty for actions that risk own goals."""
        ball_vel = state.ball.linear_velocity
        ball_pos = state.ball.position
        
        # Determine own goal
        if player.team_num == 0:  # Blue team
            own_goal = self.blue_goal
        else:  # Orange team
            own_goal = self.orange_goal
            
        # Check if ball is moving toward own goal
        to_own_goal = own_goal - ball_pos
        if np.linalg.norm(to_own_goal) > 0:
            to_own_goal_norm = to_own_goal / np.linalg.norm(to_own_goal)
            own_goal_velocity = np.dot(ball_vel, to_own_goal_norm)
            
            if own_goal_velocity > 0:  # Moving toward own goal
                distance_factor = np.exp(-np.linalg.norm(to_own_goal) / 1000)
                return self.weights['own_goal_risk'] * own_goal_velocity / BALL_MAX_SPEED * distance_factor
                
        return 0.0
    
    def _panic_jump_penalty(self, player: Any, state: GameState, previous_action: np.ndarray) -> float:
        """Penalty for panic jumps (jumping when not necessary)."""
        if len(previous_action) < 6 or previous_action[5] <= 0:  # No jump
            return 0.0
            
        # Check if jump was unnecessary
        ball_pos = state.ball.position
        car_pos = player.car_data.position
        
        # Penalize jumping when ball is far away or on ground
        ball_distance = np.linalg.norm(ball_pos - car_pos)
        if ball_distance > 1000 or ball_pos[2] < BALL_RADIUS * 2:
            return self.weights['panic_jump'] * -0.1
            
        return 0.0
    
    def _bad_touch_penalty(self, player: Any, state: GameState) -> float:
        """Penalty for bad touches (hitting ball into own half)."""
        if not player.ball_touched:
            return 0.0
            
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        
        # Check if ball is moving toward own half
        if player.team_num == 0:  # Blue team
            if ball_pos[1] > 0 and ball_vel[1] > 0:  # Moving toward blue half
                return self.weights['bad_touch'] * -0.2
        else:  # Orange team
            if ball_pos[1] < 0 and ball_vel[1] < 0:  # Moving toward orange half
                return self.weights['bad_touch'] * -0.2
                
        return 0.0
    
    def _idle_penalty(self, player: Any, state: GameState, player_idx: int) -> float:
        """Penalty for being idle (not moving)."""
        if self.last_state is None:
            return 0.0
            
        last_pos = self.last_car_positions.get(player_idx, player.car_data.position)
        current_pos = player.car_data.position
        
        # Check if car moved significantly
        movement = np.linalg.norm(current_pos - last_pos)
        if movement < 10:  # Very little movement
            return self.weights['idle'] * -0.1
            
        return 0.0
    
    def _update_state_tracking(self, player: Any, state: GameState, player_idx: int):
        """Update state tracking variables."""
        self.last_state = state
        self.last_ball_vel = state.ball.linear_velocity.copy()
        self.last_car_positions[player_idx] = player.car_data.position.copy()
        self.last_boost_amounts[player_idx] = player.boost_amount

# Backwards compatibility
SSLRewardFunction = ModernRewardSystem

