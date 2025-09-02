"""
SSL-focused state setters for scenario sampling.
Provides probabilistic scenario generation for different training phases
with emphasis on SSL-level mechanics like aerials, wall reads, and backboard plays.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from rlgym.rocket_league.common_values import (
    CAR_MAX_SPEED,
    BALL_MAX_SPEED,
    CEILING_Z,
    BALL_RADIUS,
    SIDE_WALL_X,
    BACK_WALL_Y,
    CAR_MAX_ANG_VEL,
)

def rand_vec3(rng: np.random.Generator, max_magnitude: float = 1.0) -> np.ndarray:
    """Generate a random 3D vector with components in [-max_magnitude, max_magnitude]."""
    return rng.uniform(-max_magnitude, max_magnitude, 3)

=======

class DefaultState:
    """Placeholder for compatibility during testing."""
    def reset(self, state_wrapper: StateWrapper):
        # Center the ball and stop motion
        state_wrapper.ball.set_pos(0, 0, BALL_RADIUS)
        state_wrapper.ball.set_lin_vel(0, 0, 0)
        # Place all cars at origin on ground
        for car in state_wrapper.cars:
            car.set_pos(0, 0, BALL_RADIUS)
            car.set_lin_vel(0, 0, 0)
            car.set_ang_vel(0, 0, 0)
            car.set_rot(0, 0, 0)


class StateWrapper:
    """Minimal stand-in used only for type hints in tests."""
    def __init__(self):
        self.cars = []
        self.ball = type("Ball", (), {"position": np.zeros(3)})()


class ModernStateSetter:
    """
    SSL-focused state setter with curriculum-based scenario sampling.
    
    Scenarios are weighted based on training phase:
    - Bronze/Silver: Basic kickoffs, ground play, simple aerials
    - Gold/Plat: More aerial scenarios, wall bounces
    - Diamond/Champ: Advanced aerials, backboard plays
    - GC/SSL: Complex aerial mechanics, double taps, flip resets
    """
    
    def __init__(
        self,
        curriculum_phase: str = "bronze",
        scenario_weights: Optional[Dict[str, float]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__()
        self.curriculum_phase = curriculum_phase
        self.scenario_weights = scenario_weights or self._get_default_weights()
        self.rng = rng or np.random.default_rng()

    def _rand_vec3(self, scale: float) -> np.ndarray:
        return self.rng.uniform(-scale, scale, 3)
        
    def _get_default_weights(self) -> Dict[str, float]:
        """Get default scenario weights based on curriculum phase."""
        weights = {
            'bronze': {
                'kickoff': 0.3,
                'ground_play': 0.4,
                'simple_aerial': 0.1,
                'defensive_clear': 0.1,
                'boost_collection': 0.1
            },
            'silver': {
                'kickoff': 0.25,
                'ground_play': 0.3,
                'simple_aerial': 0.2,
                'wall_bounce': 0.1,
                'defensive_clear': 0.1,
                'boost_collection': 0.05
            },
            'gold': {
                'kickoff': 0.2,
                'ground_play': 0.25,
                'simple_aerial': 0.2,
                'wall_bounce': 0.15,
                'backboard_clear': 0.1,
                'defensive_clear': 0.05,
                'boost_collection': 0.05
            },
            'platinum': {
                'kickoff': 0.15,
                'ground_play': 0.2,
                'simple_aerial': 0.2,
                'wall_bounce': 0.15,
                'backboard_clear': 0.15,
                'aerial_intercept': 0.1,
                'defensive_clear': 0.05
            },
            'diamond': {
                'kickoff': 0.1,
                'ground_play': 0.15,
                'simple_aerial': 0.15,
                'wall_bounce': 0.15,
                'backboard_clear': 0.15,
                'aerial_intercept': 0.15,
                'double_tap_setup': 0.1,
                'defensive_clear': 0.05
            },
            'champion': {
                'kickoff': 0.08,
                'ground_play': 0.12,
                'simple_aerial': 0.12,
                'wall_bounce': 0.15,
                'backboard_clear': 0.15,
                'aerial_intercept': 0.15,
                'double_tap_setup': 0.15,
                'flip_reset_setup': 0.08
            },
            'gc': {
                'kickoff': 0.05,
                'ground_play': 0.1,
                'simple_aerial': 0.1,
                'wall_bounce': 0.15,
                'backboard_clear': 0.15,
                'aerial_intercept': 0.15,
                'double_tap_setup': 0.15,
                'flip_reset_setup': 0.1,
                'ceiling_pinch': 0.05
            },
            'ssl': {
                'kickoff': 0.05,
                'ground_play': 0.08,
                'simple_aerial': 0.08,
                'wall_bounce': 0.12,
                'backboard_clear': 0.15,
                'aerial_intercept': 0.15,
                'double_tap_setup': 0.15,
                'flip_reset_setup': 0.12,
                'ceiling_pinch': 0.1
            }
        }
        
        return weights.get(self.curriculum_phase, weights['bronze'])
    
    def set_curriculum_phase(self, phase: str):
        """Update curriculum phase and scenario weights."""
        self.curriculum_phase = phase
        self.scenario_weights = self._get_default_weights()
    
    def reset(self, state_wrapper: StateWrapper):
        """Reset state with a randomly selected scenario."""
        scenarios = list(self.scenario_weights.keys())
        weights = list(self.scenario_weights.values())
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(scenarios)] * len(scenarios)
        
        scenario = self.rng.choice(scenarios, p=weights)

        scenario = self.rng.choice(scenarios, p=weights)
        self._last_scenario = scenario
        
        # Execute selected scenario
        if scenario == 'kickoff':
            self._kickoff_scenario(state_wrapper)
        elif scenario == 'ground_play':
            self._ground_play_scenario(state_wrapper)
        elif scenario == 'simple_aerial':
            self._simple_aerial_scenario(state_wrapper)
        elif scenario == 'wall_bounce':
            self._wall_bounce_scenario(state_wrapper)
        elif scenario == 'backboard_clear':
            self._backboard_clear_scenario(state_wrapper)
        elif scenario == 'aerial_intercept':
            self._aerial_intercept_scenario(state_wrapper)
        elif scenario == 'double_tap_setup':
            self._double_tap_setup_scenario(state_wrapper)
        elif scenario == 'flip_reset_setup':
            self._flip_reset_setup_scenario(state_wrapper)
        elif scenario == 'ceiling_pinch':
            self._ceiling_pinch_scenario(state_wrapper)
        elif scenario == 'defensive_clear':
            self._defensive_clear_scenario(state_wrapper)
        elif scenario == 'boost_collection':
            self._boost_collection_scenario(state_wrapper)
        else:
            # Fallback to default state
            DefaultState().reset(state_wrapper)
    
    def _kickoff_scenario(self, state_wrapper: StateWrapper):
        """Standard kickoff scenario."""
        DefaultState().reset(state_wrapper)
    
    def _ground_play_scenario(self, state_wrapper: StateWrapper):
        """Ground-based play scenario."""
        # Ball on ground with moderate speed
        ball_x = self.rng.uniform(-3000, 3000)
        ball_y = self.rng.uniform(-4000, 4000)
        state_wrapper.ball.set_pos(x=ball_x, y=ball_y, z=BALL_RADIUS)
        
        # Ball velocity toward one of the goals
        goal_direction = 1 if self.rng.random() > 0.5 else -1
        ball_vel = rand_vec3(self.rng, self.rng.uniform(500, 1500))

        ball_vel = self._rand_vec3(self.rng.uniform(500, 1500))
        ball_vel[1] = goal_direction * abs(ball_vel[1])
        state_wrapper.ball.set_lin_vel(*ball_vel)
        
        # Place cars near ball
        self._place_cars_near_ball(state_wrapper, max_distance=800)
    
    def _simple_aerial_scenario(self, state_wrapper: StateWrapper):
        """Simple aerial scenario with ball at moderate height."""
        # Ball in air
        ball_x = self.rng.uniform(-2000, 2000)
        ball_y = self.rng.uniform(-3000, 3000)
        ball_z = self.rng.uniform(200, 800)
        state_wrapper.ball.set_pos(x=ball_x, y=ball_y, z=ball_z)
        
        # Ball moving horizontally
        ball_vel = rand_vec3(self.rng, self.rng.uniform(300, 1000))

        ball_vel = self._rand_vec3(self.rng.uniform(300, 1000))
        ball_vel[2] = self.rng.uniform(-200, 200)  # Small vertical component
        state_wrapper.ball.set_lin_vel(*ball_vel)
        
        # Place cars on ground, some with boost
        self._place_cars_near_ball(state_wrapper, max_distance=1200)
        self._give_boost_to_cars(state_wrapper, boost_prob=0.7)
    
    def _wall_bounce_scenario(self, state_wrapper: StateWrapper):
        """Wall bounce scenario for wall read training."""
        # Ball near wall
        wall_side = self.rng.choice(['left', 'right', 'back'])
        if wall_side == 'left':
            ball_x = self.rng.uniform(3500, 4000)
            ball_y = self.rng.uniform(-4000, 4000)
            ball_vel = rand_vec3(self.rng, self.rng.uniform(800, 1500))

            ball_vel = self._rand_vec3(self.rng.uniform(800, 1500))
            ball_vel[0] = -abs(ball_vel[0])  # Moving toward wall
        elif wall_side == 'right':
            ball_x = self.rng.uniform(-4000, -3500)
            ball_y = self.rng.uniform(-4000, 4000)
            ball_vel = rand_vec3(self.rng, self.rng.uniform(800, 1500))
            ball_vel = self._rand_vec3(self.rng.uniform(800, 1500))
 
            ball_vel[0] = abs(ball_vel[0])  # Moving toward wall
        else:  # back
            ball_x = self.rng.uniform(-3000, 3000)
            ball_y = self.rng.uniform(4500, 5000) if self.rng.random() > 0.5 else self.rng.uniform(-5000, -4500)
            ball_vel = rand_vec3(self.rng, self.rng.uniform(800, 1500))

            ball_vel = self._rand_vec3(self.rng.uniform(800, 1500))
            ball_vel[1] = -abs(ball_vel[1]) if ball_y > 0 else abs(ball_vel[1])
        
        ball_z = self.rng.uniform(100, 600)
        state_wrapper.ball.set_pos(x=ball_x, y=ball_y, z=ball_z)
        state_wrapper.ball.set_lin_vel(*ball_vel)
        
        # Place cars to intercept wall bounce
        self._place_cars_for_wall_intercept(state_wrapper, wall_side)
    
    def _backboard_clear_scenario(self, state_wrapper: StateWrapper):
        """Backboard clear scenario."""
        # Ball near backboard
        ball_x = self.rng.uniform(-1000, 1000)
        ball_y = self.rng.uniform(4500, 5000) if self.rng.random() > 0.5 else self.rng.uniform(-5000, -4500)
        ball_z = self.rng.uniform(200, 800)
        state_wrapper.ball.set_pos(x=ball_x, y=ball_y, z=ball_z)
        
        # Ball moving toward backboard
        ball_vel = rand_vec3(self.rng, self.rng.uniform(600, 1200))

        ball_vel = self._rand_vec3(self.rng.uniform(600, 1200))
        ball_vel[1] = -abs(ball_vel[1]) if ball_y > 0 else abs(ball_vel[1])
        state_wrapper.ball.set_lin_vel(*ball_vel)
        
        # Place cars for backboard clear
        self._place_cars_for_backboard_clear(state_wrapper)
    
    def _aerial_intercept_scenario(self, state_wrapper: StateWrapper):
        """Aerial intercept scenario."""
        # Ball high in air
        ball_x = self.rng.uniform(-2000, 2000)
        ball_y = self.rng.uniform(-3000, 3000)
        ball_z = self.rng.uniform(600, 1200)
        state_wrapper.ball.set_pos(x=ball_x, y=ball_y, z=ball_z)
        
        # Ball moving with significant horizontal velocity
        ball_vel = rand_vec3(self.rng, self.rng.uniform(800, 1500))

        ball_vel = self._rand_vec3(self.rng.uniform(800, 1500))
        ball_vel[2] = self.rng.uniform(-300, 100)  # Falling
        state_wrapper.ball.set_lin_vel(*ball_vel)
        
        # Place cars on ground with boost
        self._place_cars_near_ball(state_wrapper, max_distance=1500)
        self._give_boost_to_cars(state_wrapper, boost_prob=0.8)
    
    def _double_tap_setup_scenario(self, state_wrapper: StateWrapper):
        """Double tap setup scenario."""
        # Ball high, moving toward backboard
        ball_x = self.rng.uniform(-1500, 1500)
        ball_y = self.rng.uniform(3000, 4500) if self.rng.random() > 0.5 else self.rng.uniform(-4500, -3000)
        ball_z = self.rng.uniform(800, 1400)
        state_wrapper.ball.set_pos(x=ball_x, y=ball_y, z=ball_z)
        
        # Ball moving toward backboard with good speed
        ball_vel = rand_vec3(self.rng, self.rng.uniform(1000, 1800))
        ball_vel = self._rand_vec3(self.rng.uniform(1000, 1800))
        ball_vel[1] = -abs(ball_vel[1]) if ball_y > 0 else abs(ball_vel[1])
        ball_vel[2] = self.rng.uniform(-200, 200)
        state_wrapper.ball.set_lin_vel(*ball_vel)
        
        # Place cars for double tap setup
        self._place_cars_for_double_tap(state_wrapper)
    
    def _flip_reset_setup_scenario(self, state_wrapper: StateWrapper):
        """Flip reset setup scenario."""
        # Ball high, moving toward ceiling
        ball_x = self.rng.uniform(-1000, 1000)
        ball_y = self.rng.uniform(-2000, 2000)
        ball_z = self.rng.uniform(1000, 1800)
        state_wrapper.ball.set_pos(x=ball_x, y=ball_y, z=ball_z)
        
        # Ball moving upward
        ball_vel = rand_vec3(self.rng, self.rng.uniform(400, 800))

        ball_vel = self._rand_vec3(self.rng.uniform(400, 800))
        ball_vel[2] = self.rng.uniform(200, 600)  # Upward
        state_wrapper.ball.set_lin_vel(*ball_vel)
        
        # Place cars with boost for flip reset
        self._place_cars_near_ball(state_wrapper, max_distance=1000)
        self._give_boost_to_cars(state_wrapper, boost_prob=0.9)
    
    def _ceiling_pinch_scenario(self, state_wrapper: StateWrapper):
        """Ceiling pinch scenario."""
        # Ball near ceiling
        ball_x = self.rng.uniform(-2000, 2000)
        ball_y = self.rng.uniform(-3000, 3000)
        ball_z = self.rng.uniform(CEILING_Z - 200, CEILING_Z - 50)
        state_wrapper.ball.set_pos(x=ball_x, y=ball_y, z=ball_z)
        
        # Ball moving along ceiling
        ball_vel = rand_vec3(self.rng, self.rng.uniform(600, 1200))

        ball_vel = self._rand_vec3(self.rng.uniform(600, 1200))
        ball_vel[2] = self.rng.uniform(-100, 100)  # Small vertical component
        state_wrapper.ball.set_lin_vel(*ball_vel)
        
        # Place cars for ceiling play
        self._place_cars_for_ceiling_play(state_wrapper)
    
    def _defensive_clear_scenario(self, state_wrapper: StateWrapper):
        """Defensive clear scenario."""
        # Ball in defensive half
        ball_x = self.rng.uniform(-3000, 3000)
        ball_y = self.rng.uniform(-5000, -2000) if self.rng.random() > 0.5 else self.rng.uniform(2000, 5000)
        ball_z = self.rng.uniform(100, 600)
        state_wrapper.ball.set_pos(x=ball_x, y=ball_y, z=ball_z)
        
        # Ball moving toward goal
        goal_direction = 1 if ball_y < 0 else -1
        ball_vel = rand_vec3(self.rng, self.rng.uniform(800, 1500))
        ball_vel = self._rand_vec3(self.rng.uniform(800, 1500))
        ball_vel[1] = goal_direction * abs(ball_vel[1])
        state_wrapper.ball.set_lin_vel(*ball_vel)
        
        # Place cars for defensive clear
        self._place_cars_for_defensive_clear(state_wrapper)
    
    def _boost_collection_scenario(self, state_wrapper: StateWrapper):
        """Boost collection scenario."""
        # Ball in center
        state_wrapper.ball.set_pos(x=0, y=0, z=BALL_RADIUS)
        state_wrapper.ball.set_lin_vel(0, 0, 0)
        
        # Place cars with low boost
        for i, car in enumerate(state_wrapper.cars):
            # Place cars away from ball
            car_x = self.rng.uniform(-3000, 3000)
            car_y = self.rng.uniform(-4000, 4000)
            car.set_pos(x=car_x, y=car_y, z=BALL_RADIUS)
            car.set_lin_vel(0, 0, 0)
            car.set_rot(pitch=0, yaw=0, roll=0)
            car.set_ang_vel(0, 0, 0)
            car.boost = self.rng.uniform(0, 30)  # Low boost
    
    def _place_cars_near_ball(self, state_wrapper: StateWrapper, max_distance: float = 1000):
        """Place cars near the ball."""
        ball_pos = state_wrapper.ball.position
        
        for i, car in enumerate(state_wrapper.cars):
            # Place car within max_distance of ball
            angle = self.rng.uniform(0, 2 * np.pi)
            distance = self.rng.uniform(200, max_distance)
            
            car_x = ball_pos[0] + distance * np.cos(angle)
            car_y = ball_pos[1] + distance * np.sin(angle)
            car_z = BALL_RADIUS
            
            # Keep car in bounds
            car_x = np.clip(car_x, -SIDE_WALL_X + 100, SIDE_WALL_X - 100)
            car_y = np.clip(car_y, -BACK_WALL_Y + 100, BACK_WALL_Y - 100)
            
            car.set_pos(x=car_x, y=car_y, z=car_z)
            car.set_lin_vel(0, 0, 0)
            car.set_rot(pitch=0, yaw=self.rng.uniform(-np.pi, np.pi), roll=0)
            car.set_ang_vel(0, 0, 0)
            car.boost = self.rng.uniform(0, 100)
    
    def _place_cars_for_wall_intercept(self, state_wrapper: StateWrapper, wall_side: str):
        """Place cars for wall intercept."""
        ball_pos = state_wrapper.ball.position
        
        for i, car in enumerate(state_wrapper.cars):
            if wall_side == 'left':
                # Place cars to intercept left wall bounce
                car_x = self.rng.uniform(2000, 3500)
                car_y = ball_pos[1] + self.rng.uniform(-800, 800)
            elif wall_side == 'right':
                # Place cars to intercept right wall bounce
                car_x = self.rng.uniform(-3500, -2000)
                car_y = ball_pos[1] + self.rng.uniform(-800, 800)
            else:  # back
                # Place cars to intercept back wall bounce
                car_x = ball_pos[0] + self.rng.uniform(-800, 800)
                car_y = self.rng.uniform(3000, 4500) if ball_pos[1] > 0 else self.rng.uniform(-4500, -3000)
            
            car_z = BALL_RADIUS
            car.set_pos(x=car_x, y=car_y, z=car_z)
            car.set_lin_vel(0, 0, 0)
            car.set_rot(pitch=0, yaw=self.rng.uniform(-np.pi, np.pi), roll=0)
            car.set_ang_vel(0, 0, 0)
            car.boost = self.rng.uniform(50, 100)
    
    def _place_cars_for_backboard_clear(self, state_wrapper: StateWrapper):
        """Place cars for backboard clear."""
        ball_pos = state_wrapper.ball.position
        
        for i, car in enumerate(state_wrapper.cars):
            # Place cars in front of backboard
            car_x = ball_pos[0] + self.rng.uniform(-1000, 1000)
            car_y = ball_pos[1] + self.rng.uniform(-800, 800)
            car_z = BALL_RADIUS
            
            car.set_pos(x=car_x, y=car_y, z=car_z)
            car.set_lin_vel(0, 0, 0)
            car.set_rot(pitch=0, yaw=self.rng.uniform(-np.pi, np.pi), roll=0)
            car.set_ang_vel(0, 0, 0)
            car.boost = self.rng.uniform(60, 100)
    
    def _place_cars_for_double_tap(self, state_wrapper: StateWrapper):
        """Place cars for double tap setup."""
        ball_pos = state_wrapper.ball.position
        
        for i, car in enumerate(state_wrapper.cars):
            # Place cars to follow ball trajectory
            car_x = ball_pos[0] + self.rng.uniform(-1200, 1200)
            car_y = ball_pos[1] + self.rng.uniform(-1000, 1000)
            car_z = BALL_RADIUS
            
            car.set_pos(x=car_x, y=car_y, z=car_z)
            car.set_lin_vel(0, 0, 0)
            car.set_rot(pitch=0, yaw=self.rng.uniform(-np.pi, np.pi), roll=0)
            car.set_ang_vel(0, 0, 0)
            car.boost = self.rng.uniform(70, 100)
    
    def _place_cars_for_ceiling_play(self, state_wrapper: StateWrapper):
        """Place cars for ceiling play."""
        ball_pos = state_wrapper.ball.position
        
        for i, car in enumerate(state_wrapper.cars):
            # Place cars below ball
            car_x = ball_pos[0] + self.rng.uniform(-1000, 1000)
            car_y = ball_pos[1] + self.rng.uniform(-1000, 1000)
            car_z = self.rng.uniform(BALL_RADIUS, ball_pos[2] - 200)
            
            car.set_pos(x=car_x, y=car_y, z=car_z)
            car.set_lin_vel(0, 0, 0)
            car.set_rot(pitch=0, yaw=self.rng.uniform(-np.pi, np.pi), roll=0)
            car.set_ang_vel(0, 0, 0)
            car.boost = self.rng.uniform(80, 100)
    
    def _place_cars_for_defensive_clear(self, state_wrapper: StateWrapper):
        """Place cars for defensive clear."""
        ball_pos = state_wrapper.ball.position
        
        for i, car in enumerate(state_wrapper.cars):
            # Place cars in defensive positions
            car_x = ball_pos[0] + self.rng.uniform(-1500, 1500)
            car_y = ball_pos[1] + self.rng.uniform(-1200, 1200)
            car_z = BALL_RADIUS
            
            car.set_pos(x=car_x, y=car_y, z=car_z)
            car.set_lin_vel(0, 0, 0)
            car.set_rot(pitch=0, yaw=self.rng.uniform(-np.pi, np.pi), roll=0)
            car.set_ang_vel(0, 0, 0)
            car.boost = self.rng.uniform(40, 100)
    
    def _give_boost_to_cars(self, state_wrapper: StateWrapper, boost_prob: float = 0.5):
        """Give boost to cars with given probability."""
        for car in state_wrapper.cars:
            if self.rng.random() < boost_prob:
                car.boost = self.rng.uniform(50, 100)
            else:
                car.boost = self.rng.uniform(0, 50)

# Backwards compatibility
SSLStateSetter = ModernStateSetter

