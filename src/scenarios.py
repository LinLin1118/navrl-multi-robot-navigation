"""Scenario generation and low level geometry for the multi-robot navigation demo.

Everything here is plain NumPy so it can be tested without torch, Isaac or a GPU.
The functions are pure: given the same random generator they always return the
same field.

Coordinate frame: the arena is a square centred on the origin, the robots fly at
a constant height, so all geometry is 2D.
"""

from __future__ import annotations

import numpy as np

# --- default settings of the experiments -----------------------------------
MAP_HALF = 18.0
DT = 0.1                 # one policy decision per 0.1 s of simulated time
STEPS = 560              # 56 s per episode
GOAL_RADIUS = 0.6        # a robot counts as arrived inside this radius
ROBOT_RADIUS = 0.25      # used both for contact tests and for the safety filter
N_ROBOTS = 6

START_MIN_SEP = 4.5      # starts must be at least this far apart
GOAL_MIN_SEP = 4.0       # goals must be at least this far apart
GOAL_MIN_DIST = 12.0     # every goal must be at least this far from its own start

STATIC_RADIUS = (0.6, 1.1)
DYN_SPEED = (0.5, 1.5)
DYN_RADIUS = (0.34, 0.5)
DYN_TURN_EVERY = 40      # steps between random heading changes


def make_robots(rng, n=N_ROBOTS):
    """Sample start and goal positions with the separation rules of the paper setup."""
    starts, goals = [], []
    guard = 0
    while len(starts) < n and guard < 100000:
        guard += 1
        p = rng.uniform([-15, -15], [15, 15])
        if any(np.hypot(*(p - s)) < START_MIN_SEP for s in starts):
            continue
        starts.append(p)

    guard = 0
    while len(goals) < n and guard < 100000:
        guard += 1
        p = rng.uniform([-15, -15], [15, 15])
        if any(np.hypot(*(p - g)) < GOAL_MIN_SEP for g in goals):
            continue
        if np.hypot(*(p - starts[len(goals)])) < GOAL_MIN_DIST:
            continue
        goals.append(p)

    if len(starts) < n or len(goals) < n:
        raise RuntimeError("could not place all robots with the requested separations")

    vels = [np.zeros(2) for _ in range(n)]
    dirs = [goals[i] - starts[i] for i in range(n)]
    return starts, goals, vels, dirs


def make_static(n, starts, goals, rng, *, keep_clear=2.0, arena=(-14.0, 14.0, -12.5, 12.5)):
    """Sample static cylinders that do not overlap each other or any start/goal."""
    obs = []
    tries = 0
    while len(obs) < n and tries < 20000:
        tries += 1
        x = rng.uniform(arena[0], arena[1])
        y = rng.uniform(arena[2], arena[3])
        r = rng.uniform(*STATIC_RADIUS)
        if any(np.hypot(x - o[0], y - o[1]) < r + o[2] + 2.0 for o in obs):
            continue
        if any(np.hypot(x - p[0], y - p[1]) < r + keep_clear for p in starts + goals):
            continue
        obs.append((x, y, r))
    return obs


def make_dynamic(n, rng):
    """Sample moving obstacles with a random position, heading and speed."""
    pos, vel = [], []
    for _ in range(n):
        for _ in range(10000):
            x = rng.uniform(-15, 15)
            y = rng.uniform(-13, 13)
            if all(np.hypot(x - p[0], y - p[1]) > 1.6 for p in pos):
                break
        pos.append(np.array([x, y]))
        ang = rng.uniform(0, 2 * np.pi)
        sp = rng.uniform(*DYN_SPEED)
        vel.append(np.array([np.cos(ang), np.sin(ang)]) * sp)
    return pos, vel


def radii(n, rng, low=DYN_RADIUS[0], high=DYN_RADIUS[1]):
    return [rng.uniform(low, high) for _ in range(n)]


def safety_filter(pos, vel, static, dyn_pos, dyn_r, *, dt=DT, margin=0.35, bound=17.5):
    """Reject a step that would end inside an obstacle.

    If the candidate step is free, the velocity is returned unchanged. Otherwise
    the robot brakes and slides along the tangent of the closest blocking object,
    which is enough to keep it out of the obstacle without any planning.
    """
    nxt = pos + vel * dt
    if not np.all(np.abs(nxt) < bound):
        return np.zeros(2)

    blockers = []
    for (ox, oy, orr) in static:
        if np.hypot(nxt[0] - ox, nxt[1] - oy) < orr + margin:
            blockers.append(np.array([ox, oy]))
    for p, r in zip(dyn_pos, dyn_r):
        if np.hypot(nxt[0] - p[0], nxt[1] - p[1]) < r + margin:
            blockers.append(np.array(p))

    if not blockers:
        return vel

    centre = min(blockers, key=lambda q: np.hypot(*(q - pos)))
    away = pos - centre
    norm = np.linalg.norm(away) + 1e-9
    away = away / norm
    tangent = np.array([-away[1], away[0]])
    if np.dot(tangent, vel) < 0:
        tangent = -tangent
    return 0.45 * vel + 0.9 * tangent


def step_dynamic(pos, vel, rng, t, *, dt=DT, bound=15.0):
    """Advance the moving obstacles, bouncing them off the arena border."""
    for i in range(len(pos)):
        if t % DYN_TURN_EVERY == 0:
            ang = rng.uniform(0, 2 * np.pi)
            sp = rng.uniform(*DYN_SPEED)
            vel[i] = np.array([np.cos(ang), np.sin(ang)]) * sp
        pos[i] = pos[i] + vel[i] * dt
        for k in (0, 1):
            if pos[i][k] > bound:
                pos[i][k] = bound
                vel[i][k] = -abs(vel[i][k])
            if pos[i][k] < -bound:
                pos[i][k] = -bound
                vel[i][k] = abs(vel[i][k])


def clearance_to_obstacles(point, static, dyn_pos, dyn_r):
    """Smallest distance from `point` to any obstacle surface.

    Negative values mean the point is inside an obstacle. If there is nothing to
    measure against, infinity is returned.
    """
    values = [np.hypot(point[0] - o[0], point[1] - o[1]) - o[2] for o in static]
    values += [np.hypot(point[0] - p[0], point[1] - p[1]) - r for p, r in zip(dyn_pos, dyn_r)]
    return float(min(values)) if values else float("inf")
