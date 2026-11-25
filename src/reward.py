"""The reward equation the released NavRL checkpoint was trained with.

This is a transcription of the reward block in the upstream training environment,
``isaac-training/training/scripts/env.py``, lines 546-577 of the NavRL repository at
commit ``3725bcc2e7c1be4ecf1455d922299ae85042603a`` (see ``ATTRIBUTION.md``).

Nothing in this repository trains a policy: the checkpoint is used for inference
only. This module exists so that the equation the policy was shaped by is present as
executable code with tests, instead of only being described in prose.

The reward is a sum of six terms, evaluated once per training step:

.. math::

    r_t = \\underbrace{v_t \\cdot \\hat{g}_t}_{\\text{goal progress}}
        + \\underbrace{1}_{\\text{step bonus}}
        + \\underbrace{\\frac{1}{W H}\\sum_{w,h}
            \\log \\mathrm{clamp}(r_{max} - \\lambda_{t,w,h},\\, \\epsilon,\\, r_{max})}
            _{\\text{static safety}}
        + \\underbrace{\\frac{1}{K}\\sum_{k=1}^{K}
            \\log \\mathrm{clamp}(\\delta_{t,k},\\, \\epsilon,\\, r_{max})}
            _{\\text{dynamic safety, only if } K > 0}
        - \\underbrace{0.1 \\, \\lVert v_t - v_{t-1} \\rVert}_{\\text{smoothness}}
        - \\underbrace{8 \\left[ (z_t - z_{hi})_+^2 + (z_{lo} - z_t)_+^2 \\right]}
            _{\\text{height penalty}}

where

* ``v_t`` is the world-frame velocity command, ``\\hat{g}_t`` the unit vector from the
  robot to its goal, so the first term rewards motion towards the goal and only that
  (a robot flying sideways earns nothing from it);
* ``\\lambda_{t,w,h}`` is the LiDAR *gap* image: the upstream code stores
  ``r_max - distance``, not the distance itself (``self.lidar_scan = self.lidar_range -
  ...``, env.py line 437). ``r_{max} - \\lambda`` therefore recovers the distance to the
  nearest static surface in that beam, and the term is ``log`` of that distance. Passing
  the raw measured distance would flip the sign of the term;
* ``\\delta_{t,k}`` is the distance from the robot to the surface of the ``k``-th of the
  ``K = 5`` nearest moving obstacles, clamped to the sensor range;
* ``z_{lo}, z_{hi}`` are per-episode bounds: ``z_{lo} = \\min(z_{start}, z_{goal}) - 0.2``
  and ``z_{hi} = \\max(z_{start}, z_{goal}) + 0.2`` (env.py lines 416-417 and 565-566).
  They are *not* the fixed terminal bounds 0.2 m / 4 m — with start and goal both at 2 m
  the penalty-free band is 1.8-2.2 m, so a robot at 3 m is already penalised even though
  it has not left the terminal band;
* the final step adds nothing: the ``-50`` collision term is commented out upstream
  (env.py line 580), so a bad episode cannot be bought back later. Note that reaching the
  goal does *not* by itself terminate the episode — upstream ``terminated`` is only
  ``below_bound | above_bound | collision`` and ``reach_goal`` goes to the statistics
  only; the episode ends on a collision, leaving the altitude band, or the step limit.

Two shapes are worth noting. Both safety terms are logarithmic rather than linear,
so the reward saturates at the sensor range and a robot gains little by hovering far
from everything. The smoothness term penalises acceleration, not speed: fast but
steady motion stays cheap, while a jittery command stream is expensive.

Every function below is pure NumPy and is exercised by ``tests/test_reward.py``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "GOAL_PROGRESS",
    "STEP_BONUS",
    "STATIC_WEIGHT",
    "DYNAMIC_WEIGHT",
    "SMOOTH_WEIGHT",
    "HEIGHT_WEIGHT",
    "goal_progress_term",
    "static_safety_term",
    "dynamic_safety_term",
    "smoothness_penalty",
    "height_penalty",
    "total_reward",
]

#: Weights as they appear in the upstream sum (lines 575-577). The safety terms carry
#: a weight of 1.0, the step bonus is a constant 1.0, smoothness is scaled by 0.1 and
#: the height penalty by 8.0.
GOAL_PROGRESS = 1.0
STEP_BONUS = 1.0
STATIC_WEIGHT = 1.0
DYNAMIC_WEIGHT = 1.0
SMOOTH_WEIGHT = 0.1
HEIGHT_WEIGHT = 8.0

EPS = 1e-6


def goal_progress_term(velocity, goal_rel_pos):
    """``v . g_hat``: velocity projected on the unit vector to the goal.

    ``velocity`` and ``goal_rel_pos`` are ``(..., 3)`` arrays in the world frame.
    A zero-length goal vector yields a zero term instead of a division by zero.
    """
    velocity = np.asarray(velocity, dtype=float)
    goal_rel_pos = np.asarray(goal_rel_pos, dtype=float)
    dist = np.linalg.norm(goal_rel_pos, axis=-1, keepdims=True)
    direction = goal_rel_pos / np.maximum(dist, EPS)
    return np.sum(velocity * direction, axis=-1)


def static_safety_term(lidar_scan, lidar_range=4.0):
    """``mean log(clamp(lidar_range - scan, eps, lidar_range))`` over the gap image.

    ``lidar_scan`` is the **gap image** the upstream environment stores, i.e.
    ``lidar_range - distance`` (env.py line 437), not the measured distance. The term is
    therefore ``log`` of the true distance, averaged over every beam: close surfaces give
    a small reward, open space saturates at ``log(lidar_range)``. Clamping at ``eps``
    keeps ``log`` finite when a beam reads the full range; the upper clamp at
    ``lidar_range`` prevents any single beam from dominating.

    Passing the raw measured distance instead of the gap image would invert the term, so
    do not "fix" the subtraction below.
    """
    scan = np.asarray(lidar_scan, dtype=float)
    gap = np.clip(lidar_range - scan, EPS, lidar_range)
    return np.log(gap).mean()


def dynamic_safety_term(obstacle_distances, lidar_range=4.0):
    """``mean log(clamp(delta, eps, lidar_range))`` over the K nearest moving obstacles.

    ``obstacle_distances`` is the distance from the robot to each obstacle *surface*
    (centre distance minus radius), already reduced to the K nearest. Returns ``0.0``
    for an empty array, which is what the upstream code does when a scene has no
    dynamic obstacles: the term is simply omitted from the sum.
    """
    d = np.asarray(obstacle_distances, dtype=float).ravel()
    if d.size == 0:
        return 0.0
    return np.log(np.clip(d, EPS, lidar_range)).mean()


def smoothness_penalty(velocity, prev_velocity):
    """``||v_t - v_{t-1}||``, penalising acceleration rather than speed."""
    v = np.asarray(velocity, dtype=float)
    p = np.asarray(prev_velocity, dtype=float)
    return np.linalg.norm(v - p, axis=-1)


def height_penalty(z, z_lo, z_hi):
    """``(z - z_hi)_+^2 + (z_lo - z)_+^2``, zero inside the band.

    ``z_lo`` and ``z_hi`` are the per-episode bounds ``min(z_start, z_goal) - 0.2`` and
    ``max(z_start, z_goal) + 0.2``; they have no default on purpose, because the fixed
    terminal bounds (0.2 m and 4 m) are a *different* pair and using them here would
    silently widen the penalty-free band by metres.
    """
    z = np.asarray(z, dtype=float)
    return np.maximum(z - z_hi, 0.0) ** 2 + np.maximum(z_lo - z, 0.0) ** 2


def total_reward(velocity, prev_velocity, goal_rel_pos, z, lidar_scan, z_lo, z_hi,
                 obstacle_distances=None, lidar_range=4.0):
    """Sum of the six terms, exactly as the upstream reward is assembled.

    ``lidar_scan`` is the gap image (see :func:`static_safety_term`) and ``z_lo`` /
    ``z_hi`` are the per-episode altitude bounds (see :func:`height_penalty`).
    """
    r = (GOAL_PROGRESS * goal_progress_term(velocity, goal_rel_pos)
         + STEP_BONUS
         + STATIC_WEIGHT * static_safety_term(lidar_scan, lidar_range)
         + DYNAMIC_WEIGHT * dynamic_safety_term(
             [] if obstacle_distances is None else obstacle_distances, lidar_range)
         - SMOOTH_WEIGHT * smoothness_penalty(velocity, prev_velocity)
         - HEIGHT_WEIGHT * height_penalty(z, z_lo, z_hi))
    return r
