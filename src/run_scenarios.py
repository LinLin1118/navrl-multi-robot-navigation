"""Run the three multi-robot scenes with the pretrained NavRL policy.

Example
-------
    python src/run_scenarios.py --out runs/ --no-frames

The script never trains anything. It loads the checkpoint that ships with the
repository, asks the policy for a velocity command for every robot at every
0.1 s step, and records arrival times and closest approaches.

Running this needs torch, tensordict and torchrl (see requirements.txt). The
geometry and the metrics live in `scenarios.py` and `metrics.py`, which are pure
NumPy and can be tested without any of that.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "navrl_pretrained"))

import metrics  # noqa: E402
import scenarios as sc  # noqa: E402

SCENES = [
    # label, n_dynamic, n_static, seed
    ("Static (15)", 0, 15, 202),
    ("Dynamic (15)", 15, 0, 101),
    ("Mixed (15)", 8, 7, 303),
]


def run_scene(label, n_dynamic, n_static, outdir, seed, *, steps=sc.STEPS,
              n_robots=sc.N_ROBOTS, frames=True):
    """Simulate one scene and return its metrics as a dict."""
    import torch
    from agent import Agent  # noqa: E402  (vendored copy of quick-demos)

    # plotting is optional so the script also runs on a headless machine
    if frames:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm
    else:
        plt = None

    rng = np.random.default_rng(seed)
    if frames:
        os.makedirs(outdir, exist_ok=True)
        for name in os.listdir(outdir):
            os.remove(os.path.join(outdir, name))

    starts, goals, vels, dirs = sc.make_robots(rng, n_robots)
    positions = [p.copy() for p in starts]
    static = sc.make_static(n_static, starts, goals, rng) if n_static else []
    dyn_pos, dyn_vel = sc.make_dynamic(n_dynamic, rng) if n_dynamic else ([], [])
    dyn_r = sc.radii(n_dynamic, rng) if n_dynamic else []

    trajectories = [[p.copy()] for p in positions]
    arrived = [False] * n_robots
    arrival_time = [None] * n_robots
    min_clearance = [float("inf")] * n_robots

    agent = Agent(device=torch.device("cpu"))

    if frames:
        cols = [cm.tab10(i) for i in range(10)]
        fig, ax = plt.subplots(figsize=(9, 7.2), dpi=100)
        ax.set_xlim(-sc.MAP_HALF, sc.MAP_HALF)
        ax.set_ylim(-sc.MAP_HALF, sc.MAP_HALF)
        ax.set_aspect("equal")
        ax.grid(True, color="#e8e8e8", linewidth=0.6)
        ax.set_xlabel("x position (m)")
        ax.set_ylabel("y position (m)")
        header = fig.text(0.09, 0.955, "", fontsize=9, fontweight="bold")
        for (ox, oy, orr) in static:
            ax.add_patch(plt.Circle((ox, oy), orr, color="#b0b0b0", alpha=0.85, zorder=2))
        dyn_dots = [plt.Circle((dyn_pos[i][0], dyn_pos[i][1]), dyn_r[i],
                               color="#f2c14e", alpha=0.92, zorder=2)
                    for i in range(n_dynamic)]
        for c in dyn_dots:
            ax.add_patch(c)

    for step in range(steps):
        t = step * sc.DT
        if n_dynamic:
            sc.step_dynamic(dyn_pos, dyn_vel, rng, step)
            if frames:
                for i, c in enumerate(dyn_dots):
                    c.set_center((dyn_pos[i][0], dyn_pos[i][1]))

        for i in range(n_robots):
            if arrived[i]:
                continue
            goal = goals[i]
            if np.linalg.norm(goal - positions[i]) < sc.GOAL_RADIUS:
                arrived[i] = True
                arrival_time[i] = t
                continue

            state = _robot_state(positions[i], goal, vels[i])
            lidar = _lidar(positions[i], static, float(dirs[i][1]), float(dirs[i][0]))
            target_dir = _target_dir(dirs[i])
            empty = np.zeros((1, 0, 10), dtype=np.float32)
            vel = agent.plan(state, lidar, empty, target_dir)
            if vel.shape[0] < 2:
                continue
            vel = sc.safety_filter(positions[i], vel, static, dyn_pos, dyn_r)
            positions[i] = positions[i] + vel * sc.DT
            vels[i] = vel
            trajectories[i].append(positions[i].copy())

        for i in range(n_robots):
            min_clearance[i] = min(min_clearance[i],
                                   sc.clearance_to_obstacles(positions[i], static, dyn_pos, dyn_r))

        if frames:
            header.set_text("Multi-Robot Navigation (6 robots) + %s - step %d - Time: %.1fs"
                            % (label, step, t))
            fig.savefig(os.path.join(outdir, "f_%04d.png" % step), dpi=100)

    if frames:
        plt.close(fig)

    result = metrics.summarise(arrival_time, min_clearance)
    result.update({"scenario": label, "static": n_static, "dynamic": n_dynamic,
                   "seed": seed, "steps": steps, "sim_time": round(steps * sc.DT, 1)})
    return result


def _robot_state(position, goal, velocity):
    """8-dim ego state: relative goal, distance, velocity (matches quick-demos)."""
    import torch
    rel = goal - position
    dist = np.linalg.norm(rel)
    vec = np.array([rel[0] / dist, rel[1] / dist, velocity[0], velocity[1],
                    dist, 0.0, velocity[0], velocity[1]], dtype=np.float32)
    return torch.from_numpy(vec).unsqueeze(0)


def _target_dir(direction):
    import torch
    vec = np.array([direction[0], direction[1], 0.0], dtype=np.float32)
    norm = np.linalg.norm(vec) + 1e-9
    return torch.from_numpy(vec / norm).unsqueeze(0).unsqueeze(0)


def _lidar(position, static, ax, ay):
    """36x4 range image from the analytic ray cast in the vendored utils."""
    import torch
    from utils import get_ray_cast
    scan, _, _ = get_ray_cast(position, static, max_range=4.0, hres_deg=10.0,
                              vfov_angles_deg=[-10.0, 0.0, 10.0, 20.0],
                              start_angle_deg=float(np.degrees(np.arctan2(ay, ax))),
                              device=torch.device("cpu"))
    return scan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs", help="where frames are written")
    parser.add_argument("--steps", type=int, default=sc.STEPS)
    parser.add_argument("--no-frames", action="store_true", help="skip rendering")
    parser.add_argument("--scene", default="all", help="scene label, or 'all'")
    args = parser.parse_args(argv)

    results = []
    for label, n_dyn, n_static, seed in SCENES:
        if args.scene != "all" and args.scene != label:
            continue
        outdir = os.path.join(args.out, label.split(" ")[0])
        print("=== %s" % label, flush=True)
        res = run_scene(label, n_dyn, n_static, outdir, seed,
                        steps=args.steps, frames=not args.no_frames)
        print(json.dumps(res, indent=2), flush=True)
        results.append(res)

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    if results:
        with open(os.path.join("results", "summary.csv"), "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["scenario", "static", "dynamic", "reached", "success_rate",
                             "near_miss", "penetration", "mean_arrival",
                             "mean_min_clearance", "worst_min_clearance"])
            for r in results:
                writer.writerow([r["scenario"], r["static"], r["dynamic"], r["reached"],
                                 r["success_rate"], r["near_miss"], r["penetration"],
                                 r["mean_arrival"], r["mean_min_clearance"],
                                 r["worst_min_clearance"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
