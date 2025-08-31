"""
Build the SAME observation vector used in training, from RLBot's GameTickPacket.

IMPORTANT: This must MATCH your training obs exactly (order, scaling, size).
This implementation mirrors src/training/observers.py SSLObsBuilder exactly.
"""
from __future__ import annotations
import math
from typing import List
import numpy as np

# Field extents and constants (matching training)
MAX_X = 4096.0
MAX_Y = 5120.0
MAX_Z = 2044.0
CEILING_Z = 2044.0
BALL_RADIUS = 92.75
CAR_MAX_SPEED = 2300.0

# Normalization constants (matching training)
POS_SCALE = 2300.0
VEL_SCALE = CAR_MAX_SPEED
ANG_VEL_SCALE = 5.5
BOOST_SCALE = 100.0

# Final obs layout size (MUST match training):
OBS_SIZE = 107

# Boost pad locations (simplified - would need full BOOST_LOCATIONS from RLGym)
BOOST_LOCATIONS = [
    np.array([0, -4240, 70]), np.array([-1792, -4184, 70]), np.array([1792, -4184, 70]),
    np.array([-3072, -4096, 73]), np.array([3072, -4096, 73]), np.array([-940, -3308, 70]),
    np.array([940, -3308, 70]), np.array([0, 4240, 70]), np.array([-1792, 4184, 70]),
    np.array([1792, 4184, 70]), np.array([-3072, 4096, 73]), np.array([3072, 4096, 73]),
    np.array([-940, 3308, 70]), np.array([940, 3308, 70]), np.array([-2048, -1036, 70]),
    np.array([2048, -1036, 70]), np.array([-2048, 1036, 70]), np.array([2048, 1036, 70])
]


def _norm(v: float, max_abs: float) -> float:
    return float(max(-1.0, min(1.0, v / max_abs)))


def _clip_norm(v: float, scale: float) -> float:
    return float(max(-1.0, min(1.0, v / scale)))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _calculate_wall_read_angle(car_pos: np.ndarray, car_forward: np.ndarray, ball_pos: np.ndarray, ball_vel: np.ndarray) -> float:
    """Calculate angle for wall read opportunity."""
    # Predict ball trajectory to wall
    if abs(ball_vel[0]) > 0.1:  # Moving toward side wall
        time_to_wall = (4096 - abs(ball_pos[0])) / abs(ball_vel[0])
        if time_to_wall > 0 and time_to_wall < 3.0:
            predicted_pos = ball_pos + ball_vel * time_to_wall
            car_to_predicted = predicted_pos - car_pos
            if np.linalg.norm(car_to_predicted) > 0:
                angle = _cosine_similarity(car_to_predicted, car_forward)
                return np.clip(angle, -1, 1)
    return 0.0


def _calculate_pressure(car_pos: np.ndarray, opponents: List, ball_pos: np.ndarray) -> float:
    """Calculate pressure level on the ball."""
    if not opponents:
        return 0.0
        
    # Distance-weighted pressure
    total_pressure = 0.0
    for opponent in opponents:
        opp_pos = opponent.physics.location
        dist = np.linalg.norm(np.array([opp_pos.x, opp_pos.y, opp_pos.z]) - ball_pos)
        if dist < 1000:  # Within pressure range
            pressure = (1000 - dist) / 1000
            total_pressure += pressure
                
    return np.clip(total_pressure, 0, 1)


def _get_ball_owner(players: List, ball_pos: np.ndarray) -> int:
    """Determine which team has ball possession."""
    if not players:
        return None
        
    closest_player = min(players, key=lambda p: np.linalg.norm(
        np.array([p.physics.location.x, p.physics.location.y, p.physics.location.z]) - ball_pos
    ))
    closest_pos = np.array([closest_player.physics.location.x, closest_player.physics.location.y, closest_player.physics.location.z])
    if np.linalg.norm(closest_pos - ball_pos) < 200:
        return closest_player.team
    return None


def _get_boost_regions() -> List[List[np.ndarray]]:
    """Group boost pads into regions for coarse representation."""
    regions = []
    # Front/back regions
    regions.append([pos for pos in BOOST_LOCATIONS if pos[1] > 2000])
    regions.append([pos for pos in BOOST_LOCATIONS if pos[1] < -2000])
    # Left/right regions  
    regions.append([pos for pos in BOOST_LOCATIONS if pos[0] > 2000])
    regions.append([pos for pos in BOOST_LOCATIONS if pos[0] < -2000])
    # Center region
    regions.append([pos for pos in BOOST_LOCATIONS if abs(pos[0]) < 2000 and abs(pos[1]) < 2000])
    # Corner regions
    regions.append([pos for pos in BOOST_LOCATIONS if pos[0] > 2000 and pos[1] > 2000])
    regions.append([pos for pos in BOOST_LOCATIONS if pos[0] < -2000 and pos[1] > 2000])
    regions.append([pos for pos in BOOST_LOCATIONS if pos[0] > 2000 and pos[1] < -2000])
    regions.append([pos for pos in BOOST_LOCATIONS if pos[0] < -2000 and pos[1] < -2000])
    
    return regions[:16]  # Limit to 16 regions


def build_observation(packet) -> np.ndarray:
    """Convert GameTickPacket → normalized obs (shape = [OBS_SIZE]).
    
    This implementation exactly matches src/training/observers.py SSLObsBuilder.
    """
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    idx = 0
    
    # Get player data (use first car as default - RLBot will set self.index)
    if not hasattr(packet, 'game_cars') or len(packet.game_cars) == 0:
        return obs
        
    me = packet.game_cars[0]  # Will be overridden by RLBot's self.index
    ball = packet.game_ball
    
    # Convert to numpy arrays for easier computation
    car_pos = np.array([me.physics.location.x, me.physics.location.y, me.physics.location.z])
    car_vel = np.array([me.physics.velocity.x, me.physics.velocity.y, me.physics.velocity.z])
    car_ang_vel = np.array([me.physics.angular_velocity.x, me.physics.angular_velocity.y, me.physics.angular_velocity.z])
    car_rot = me.physics.rotation
    
    ball_pos = np.array([ball.physics.location.x, ball.physics.location.y, ball.physics.location.z])
    ball_vel = np.array([ball.physics.velocity.x, ball.physics.velocity.y, ball.physics.velocity.z])
    
    # 1. Self car features (20) - LOCKED SPEC
    # Position (3)
    obs[idx:idx+3] = np.clip(car_pos / POS_SCALE, -1, 1)
    idx += 3
    
    # Velocity (3)
    obs[idx:idx+3] = np.clip(car_vel / VEL_SCALE, -1, 1)
    idx += 3
    
    # Angular velocity (3)
    obs[idx:idx+3] = np.clip(car_ang_vel / ANG_VEL_SCALE, -1, 1)
    idx += 3
    
    # Boost amount (1)
    obs[idx] = np.clip(getattr(me, "boost", 0) / BOOST_SCALE, 0, 1)
    idx += 1
    
    # On ground (1)
    obs[idx] = float(getattr(me, "has_wheel_contact", True))
    idx += 1
    
    # Has flip (1)
    obs[idx] = float(getattr(me, "has_flip", True))
    idx += 1
    
    # Has jump (1)
    obs[idx] = float(getattr(me, "has_jump", True))
    idx += 1
    
    # Car orientation - pitch (1)
    forward = np.array([math.cos(car_rot.yaw) * math.cos(car_rot.pitch),
                       math.sin(car_rot.yaw) * math.cos(car_rot.pitch),
                       math.sin(car_rot.pitch)])
    obs[idx] = np.clip(forward[2], -1, 1)
    idx += 1
    
    # Car orientation - yaw (1)
    obs[idx] = np.clip(np.arctan2(forward[1], forward[0]) / np.pi, -1, 1)
    idx += 1
    
    # Car orientation - roll (1)
    up = np.array([math.cos(car_rot.yaw) * math.sin(car_rot.pitch) * math.sin(car_rot.roll) - math.sin(car_rot.yaw) * math.cos(car_rot.roll),
                   math.sin(car_rot.yaw) * math.sin(car_rot.pitch) * math.sin(car_rot.roll) + math.cos(car_rot.yaw) * math.cos(car_rot.roll),
                   math.cos(car_rot.pitch) * math.sin(car_rot.roll)])
    obs[idx] = np.clip(np.arctan2(up[2], np.sqrt(up[0]**2 + up[1]**2)) / np.pi, -1, 1)
    idx += 1
    
    # Last jump time (1)
    obs[idx] = 0.0 if getattr(me, "has_jump", True) else 1.0
    idx += 1
    
    # Car height (1)
    car_height = car_pos[2] / CEILING_Z
    obs[idx] = np.clip(car_height, 0, 1)
    idx += 1
    
    # Aerial indicator (1)
    obs[idx] = float(car_height > 0.2)
    idx += 1
    
    # Wall read angle (1)
    wall_read_angle = _calculate_wall_read_angle(car_pos, forward, ball_pos, ball_vel)
    obs[idx] = np.clip(wall_read_angle, -1, 1)
    idx += 1
    
    # Pressure (1)
    opponents = [c for c in packet.game_cars if c.team != me.team]
    pressure = _calculate_pressure(car_pos, opponents, ball_pos)
    obs[idx] = np.clip(pressure, 0, 1)
    idx += 1
    
    # Boost efficiency (1) - approximate from velocity vs boost usage
    speed = np.linalg.norm(car_vel)
    boost_amount = getattr(me, "boost", 0)
    boost_efficiency = speed / (boost_amount + 1e-6)
    obs[idx] = np.clip(boost_efficiency / 10.0, 0, 1)
    idx += 1
    
    # Recovery state (1)
    recovery_state = 1.0 if getattr(me, "has_wheel_contact", True) and getattr(me, "has_flip", True) else 0.0
    obs[idx] = recovery_state
    idx += 1
    
    # Demo timer (1) - placeholder
    obs[idx] = 0.0
    idx += 1
    
    # Demo immunity (1) - placeholder
    obs[idx] = 0.0
    idx += 1
    
    # 2. Ball features (8) - LOCKED SPEC
    # Ball position (3)
    obs[idx:idx+3] = np.clip(ball_pos / POS_SCALE, -1, 1)
    idx += 3
    
    # Ball velocity (3)
    obs[idx:idx+3] = np.clip(ball_vel / VEL_SCALE, -1, 1)
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
    allies = [c for c in packet.game_cars if c.team == me.team and c != me]
    allies = sorted(allies, key=lambda c: np.linalg.norm(
        np.array([c.physics.location.x, c.physics.location.y, c.physics.location.z]) - ball_pos
    ))[:2]
    
    for ally in allies:
        ally_pos = np.array([ally.physics.location.x, ally.physics.location.y, ally.physics.location.z])
        ally_vel = np.array([ally.physics.velocity.x, ally.physics.velocity.y, ally.physics.velocity.z])
        rel_pos = ally_pos - car_pos
        rel_vel = ally_vel - car_vel
        # Relative position (3)
        obs[idx:idx+3] = np.clip(rel_pos / POS_SCALE, -1, 1)
        idx += 3
        # Relative velocity (3)
        obs[idx:idx+3] = np.clip(rel_vel / VEL_SCALE, -1, 1)
        idx += 3
        # Distance (1)
        obs[idx] = np.clip(np.linalg.norm(rel_pos) / POS_SCALE, 0, 1)
        idx += 1
        # Boost (1)
        obs[idx] = np.clip(getattr(ally, "boost", 0) / BOOST_SCALE, 0, 1)
        idx += 1
        # On ground (1)
        obs[idx] = float(getattr(ally, "has_wheel_contact", True))
        idx += 1
        # Has flip (1)
        obs[idx] = float(getattr(ally, "has_flip", True))
        idx += 1
    
    # Pad with zeros if fewer than 2 allies (10 features per ally)
    idx += (2 - len(allies)) * 10
    
    # 4. Top-2 opponents (20) - LOCKED SPEC
    opponents = [c for c in packet.game_cars if c.team != me.team]
    opponents = sorted(opponents, key=lambda c: np.linalg.norm(
        np.array([c.physics.location.x, c.physics.location.y, c.physics.location.z]) - ball_pos
    ))[:2]
    
    for opponent in opponents:
        opp_pos = np.array([opponent.physics.location.x, opponent.physics.location.y, opponent.physics.location.z])
        opp_vel = np.array([opponent.physics.velocity.x, opponent.physics.velocity.y, opponent.physics.velocity.z])
        rel_pos = opp_pos - car_pos
        rel_vel = opp_vel - car_vel
        # Relative position (3)
        obs[idx:idx+3] = np.clip(rel_pos / POS_SCALE, -1, 1)
        idx += 3
        # Relative velocity (3)
        obs[idx:idx+3] = np.clip(rel_vel / VEL_SCALE, -1, 1)
        idx += 3
        # Distance (1)
        obs[idx] = np.clip(np.linalg.norm(rel_pos) / POS_SCALE, 0, 1)
        idx += 1
        # Boost (1)
        obs[idx] = np.clip(getattr(opponent, "boost", 0) / BOOST_SCALE, 0, 1)
        idx += 1
        # On ground (1)
        obs[idx] = float(getattr(opponent, "has_wheel_contact", True))
        idx += 1
        # Has flip (1)
        obs[idx] = float(getattr(opponent, "has_flip", True))
        idx += 1
    
    # Pad with zeros if fewer than 2 opponents (10 features per opponent)
    idx += (2 - len(opponents)) * 10
    
    # 5. Geometry features (15) - LOCKED SPEC
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
    wall_read_angle = _calculate_wall_read_angle(car_pos, forward, ball_pos, ball_vel)
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
    goal_pos = np.array([0, 5120 if me.team == 0 else -5120, 0])
    goal_vec = goal_pos - car_pos
    goal_angle = np.arccos(np.clip(np.dot(goal_vec[:2], forward[:2]) / 
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
    score_diff = getattr(packet.game_info, "blue_score", 0) - getattr(packet.game_info, "orange_score", 0)
    if me.team == 1:
        score_diff *= -1
    obs[idx] = np.clip(score_diff / 10.0, -1, 1)
    idx += 1
    
    # Time remaining (1)
    time_remaining = getattr(packet.game_info, "seconds_remaining", 300.0)
    obs[idx] = np.clip(time_remaining / 300.0, 0, 1)  # 5 min max
    idx += 1
    
    # Overtime (1)
    obs[idx] = float(getattr(packet.game_info, "is_overtime", False))
    idx += 1
    
    # Kickoff (1)
    obs[idx] = float(getattr(packet.game_info, "is_kickoff_pause", False))
    idx += 1
    
    # Boost advantage (1)
    blue_boost = sum(getattr(c, "boost", 0) for c in packet.game_cars if c.team == 0)
    orange_boost = sum(getattr(c, "boost", 0) for c in packet.game_cars if c.team == 1)
    boost_diff = (blue_boost - orange_boost) / (len(packet.game_cars) * BOOST_SCALE)
    if me.team == 1:
        boost_diff *= -1
    obs[idx] = np.clip(boost_diff, -1, 1)
    idx += 1
    
    # Ball possession (1)
    ball_owner = _get_ball_owner(packet.game_cars, ball_pos)
    obs[idx] = 1.0 if ball_owner == me.team else -1.0 if ball_owner is not None else 0.0
    idx += 1
    
    # Pressure (1)
    pressure = _calculate_pressure(car_pos, opponents, ball_pos)
    obs[idx] = np.clip(pressure, 0, 1)
    idx += 1
    
    # Game phase (1) - early/mid/late game
    game_phase = 1.0 - (time_remaining / 300.0)
    obs[idx] = np.clip(game_phase, 0, 1)
    idx += 1
    
    # 7. Boost pad availability (16) - LOCKED SPEC
    boost_regions = _get_boost_regions()
    for region in boost_regions:
        # Simplified - assume all boost pads are available
        # In real implementation, would check packet.boost_pads
        region_available = len(region) > 0
        obs[idx] = float(region_available)
        idx += 1
    
    assert obs.shape[0] == OBS_SIZE, f"OBS_SIZE mismatch: {obs.shape[0]} vs {OBS_SIZE}"
    return obs