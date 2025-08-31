"""
Build the SAME observation vector used in training, from RLBot's GameTickPacket.

IMPORTANT: This must MATCH your training obs exactly (order, scaling, size).
Adjust constants / fields to mirror ModernObsBuilder.
"""
from __future__ import annotations
import math
from typing import List
import numpy as np

# Field extents (approx Rocket League map units)
MAX_X = 4096.0
MAX_Y = 5120.0
MAX_Z = 2044.0
MAX_SPEED = 2300.0
MAX_ANG = 5.5  # ~rad/s

TOPK_OPPONENTS = 2

# Final obs layout size (MUST match training):
OBS_SIZE = (
    3 + 3 + 3 + 6 + 2 +  # car: pos, vel, ang vel, rot(sin/cos for 3 angles), boost, on_ground
    3 + 3 + 2 +          # ball: pos, vel, [height_bucket, t_to_ground_norm]
    TOPK_OPPONENTS * (3 + 3 + 1 + 1) +  # each opp: rel pos, rel vel, goal_dist_norm, threat_angle_norm
    4                                   # context: time_left_norm, score_diff_norm, backboard_dist_norm, corner_prox_norm
)


def _norm(v: float, max_abs: float) -> float:
    return float(max(-1.0, min(1.0, v / max_abs)))


def _pack_rot(pitch: float, yaw: float, roll: float) -> List[float]:
    return [math.sin(pitch), math.cos(pitch), math.sin(yaw), math.cos(yaw), math.sin(roll), math.cos(roll)]


def _time_to_ground(z: float, vz: float, g: float = -650.0) -> float:
    # Basic ballistic t where z + vz t + 0.5 g t^2 = 0 (take smallest positive root)
    a = 0.5 * g
    b = vz
    c = z
    disc = b*b - 4*a*c
    if disc < 0:
        return 0.0
    t1 = (-b - math.sqrt(disc)) / (2*a)
    t2 = (-b + math.sqrt(disc)) / (2*a)
    t = min(t for t in (t1, t2) if t >= 0) if any(t >= 0 for t in (t1, t2)) else 0.0
    return t


def build_observation(packet) -> np.ndarray:
    """Convert GameTickPacket → normalized obs (shape = [OBS_SIZE])."""
    # Player index 0 by default; adjust if team control differs
    me = packet.game_cars[self.index]  # RLBot BaseAgent sets self.index
    ball = packet.game_ball

    # Car state
    cpos = me.physics.location
    cvel = me.physics.velocity
    cang = me.physics.angular_velocity
    crot = me.physics.rotation
    boost = getattr(me, "boost", 0) / 100.0
    on_ground = 1.0 if getattr(me, "has_wheel_contact", True) else 0.0

    car_vec = [
        _norm(cpos.x, MAX_X), _norm(cpos.y, MAX_Y), _norm(cpos.z, MAX_Z),
        _norm(cvel.x, MAX_SPEED), _norm(cvel.y, MAX_SPEED), _norm(cvel.z, MAX_SPEED),
        _norm(cang.x, MAX_ANG), _norm(cang.y, MAX_ANG), _norm(cang.z, MAX_ANG),
        *_pack_rot(crot.pitch, crot.yaw, crot.roll),
        boost, on_ground,
    ]

    # Ball state
    bpos = ball.physics.location
    bvel = ball.physics.velocity
    t_ground = _time_to_ground(bpos.z, bvel.z)
    height_bucket = min(1.0, bpos.z / MAX_Z)
    ball_vec = [
        _norm(bpos.x, MAX_X), _norm(bpos.y, MAX_Y), _norm(bpos.z, MAX_Z),
        _norm(bvel.x, MAX_SPEED), _norm(bvel.y, MAX_SPEED), _norm(bvel.z, MAX_SPEED),
        height_bucket, min(1.0, t_ground / 3.0),  # cap at 3s
    ]

    # Opponents (top-k by proximity)
    opps = [c for i, c in enumerate(packet.game_cars) if i != self.index and c.team != me.team]
    # Sort by distance to me
    def _dist2(car):
        p = car.physics.location
        dx, dy, dz = p.x - cpos.x, p.y - cpos.y, p.z - cpos.z
        return dx*dx + dy*dy + dz*dz
    opps.sort(key=_dist2)

    opp_vec = []
    for k in range(TOPK_OPPONENTS):
        if k < len(opps):
            o = opps[k]
            op = o.physics.location
            ov = o.physics.velocity
            relp = (op.x - cpos.x, op.y - cpos.y, op.z - cpos.z)
            relv = (ov.x - cvel.x, ov.y - cvel.y, ov.z - cvel.z)
            goal_dist = abs(op.y) / MAX_Y  # coarse proxy to a goal line
            threat_angle = 0.0  # placeholder, fill with bearing if you compute it in training
            opp_vec.extend([
                _norm(relp[0], MAX_X), _norm(relp[1], MAX_Y), _norm(relp[2], MAX_Z),
                _norm(relv[0], MAX_SPEED), _norm(relv[1], MAX_SPEED), _norm(relv[2], MAX_SPEED),
                min(1.0, goal_dist), float(threat_angle),
            ])
        else:
            opp_vec.extend([0.0] * (3 + 3 + 1 + 1))

    # Simple context features (replace with your training-time features)
    game_time_left = getattr(packet.game_info, "seconds_remaining", 300.0)
    score_diff = 0.0  # fill if you include score in obs
    backboard_dist = min(1.0, abs((5120.0 - abs(cpos.y))) / 5120.0)
    corner_prox = 0.0
    ctx = [min(1.0, game_time_left / 300.0), _norm(score_diff, 10.0), backboard_dist, corner_prox]

    obs = np.array(car_vec + ball_vec + opp_vec + ctx, dtype=np.float32)
    assert obs.shape[0] == OBS_SIZE, f"OBS_SIZE mismatch: {obs.shape[0]} vs {OBS_SIZE}"
    return obs