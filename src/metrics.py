"""Metric helpers for the multi-robot navigation runs.

A metric here is either per robot (arrival time, closest approach) or per run
(success rate, number of contacts). Keeping them in one module means the same
definitions are used by the runner, the tests and the report.

Contact is judged against the robot radius from the navigation setup: a robot
whose hull comes within 0.25 m of an obstacle surface is counted as a near miss,
and a negative clearance means the hull actually overlapped the obstacle.
"""

from __future__ import annotations

ROBOT_RADIUS = 0.25


def classify_contact(clearance, radius=ROBOT_RADIUS):
    """Turn a signed clearance into 'clear', 'near_miss' or 'penetration'."""
    if clearance < 0.0:
        return "penetration"
    if clearance < radius:
        return "near_miss"
    return "clear"


def summarise(arrivals, min_clearances, radius=ROBOT_RADIUS):
    """Aggregate one run.

    Parameters
    ----------
    arrivals : list of float or None
        Arrival time of each robot, ``None`` if it never reached the goal.
    min_clearances : list of float
        Closest approach to any obstacle surface during the episode.
    """
    if len(arrivals) != len(min_clearances):
        raise ValueError("arrivals and min_clearances must have the same length")
    if not arrivals:
        raise ValueError("an empty run has no metrics")

    n = len(arrivals)
    reached = [a for a in arrivals if a is not None]
    classes = [classify_contact(c, radius) for c in min_clearances]

    return {
        "robots": n,
        "reached": len(reached),
        "success_rate": round(100.0 * len(reached) / n, 1),
        "near_miss": sum(1 for c in classes if c == "near_miss"),
        "penetration": sum(1 for c in classes if c == "penetration"),
        "mean_arrival": round(float(sum(reached) / len(reached)), 2) if reached else None,
        "arrivals": [round(float(a), 2) if a is not None else None for a in arrivals],
        "mean_min_clearance": round(float(sum(min_clearances) / n), 3),
        "worst_min_clearance": round(float(min(min_clearances)), 3),
        "per_robot_clearance": [round(float(c), 3) for c in min_clearances],
    }
