import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib import cm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "navrl_pretrained"))
import torch
from utils import get_robot_state, get_ray_cast, get_dyn_obs_state
from agent import Agent

DEVICE = torch.device("cpu")
SEED = 0
np.random.seed(SEED)

MAP = 18.0
DT = 0.1
DT_SIM = 0.02          # 5 policy decisions per rendered frame would be slow; keep 1:1
STEPS = 560
GOAL_R = 0.6
N_ROBOTS = 6
LIDAR_RANGE = 4.0
HRES = 10.0
VFOV = [-10.0, 0.0, 10.0, 20.0]

OUTROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "runs", "frames")

COLORS = [cm.tab10(i) for i in range(10)]
DYN_PALETTE = ["#f2c14e", "#f4a07a", "#c9a7e0", "#8fd3c7", "#eda3c0",
               "#b9d48b", "#9bb7e0", "#f0b7a4", "#a9d6e5", "#d9c27a"]


def make_robots(rng):
    starts, goals = [], []
    while len(starts) < N_ROBOTS:
        p = rng.uniform([-15, -15], [15, 15])
        if any(np.hypot(*(p - s)) < 4.5 for s in starts):
            continue
        starts.append(p)
    while len(goals) < N_ROBOTS:
        p = rng.uniform([-15, -15], [15, 15])
        if any(np.hypot(*(p - g)) < 4.0 for g in goals):
            continue
        if np.hypot(*(p - starts[len(goals)])) < 12.0:
            continue
        goals.append(p)
    vels = [np.zeros(2) for _ in range(N_ROBOTS)]
    dirs = [goals[i] - starts[i] for i in range(N_ROBOTS)]
    return starts, goals, vels, dirs


def make_static(n, starts, goals, rng):
    obs = []
    tries = 0
    while len(obs) < n and tries < 20000:
        tries += 1
        x = rng.uniform(-14, 14)
        y = rng.uniform(-12.5, 12.5)
        r = rng.uniform(0.6, 1.1)
        if any(np.hypot(x - o[0], y - o[1]) < r + o[2] + 2.0 for o in obs):
            continue
        if any(np.hypot(x - p[0], y - p[1]) < r + 2.0 for p in starts + goals):
            continue
        obs.append((x, y, r))
    return obs


def make_dynamic(n, rng):
    pos = []
    vel = []
    for _ in range(n):
        while True:
            x = rng.uniform(-15, 15)
            y = rng.uniform(-13, 13)
            if all(np.hypot(x - p[0], y - p[1]) > 1.6 for p in pos):
                break
        pos.append(np.array([x, y]))
        ang = rng.uniform(0, 2 * np.pi)
        sp = rng.uniform(0.5, 1.5)
        vel.append(np.array([np.cos(ang), np.sin(ang)]) * sp)
    return pos, vel


def radii(n_dyn, rng):
    return [rng.uniform(0.34, 0.5) for _ in range(n_dyn)]


def safety_filter(pos, vel, static, dyn_pos, dyn_r, margin=0.35):
    """Refuse a step that would end inside an obstacle: brake and slide instead."""
    nxt = pos + vel * DT
    if not (np.all(np.abs(nxt) < 17.5)):
        return np.zeros(2)
    hit = []
    for (ox, oy, orr) in static:
        if np.hypot(nxt[0] - ox, nxt[1] - oy) < orr + margin:
            hit.append(np.array([ox, oy]))
    for p, r in zip(dyn_pos, dyn_r):
        if np.hypot(nxt[0] - p[0], nxt[1] - p[1]) < r + margin:
            hit.append(np.array(p))
    if not hit:
        return vel
    # steer around the nearest blocking object
    c = min(hit, key=lambda q: np.hypot(*(q - pos)))
    away = pos - c
    n = np.linalg.norm(away) + 1e-9
    away /= n
    tangent = np.array([-away[1], away[0]])
    if np.dot(tangent, vel) < 0:
        tangent = -tangent
    return 0.45 * vel + 0.9 * tangent


def step_dynamic(pos, vel, rng, t):
    for i in range(len(pos)):
        if (t % 40) == 0:
            ang = rng.uniform(0, 2 * np.pi)
            sp = rng.uniform(0.5, 1.5)
            vel[i] = np.array([np.cos(ang), np.sin(ang)]) * sp
        pos[i] = pos[i] + vel[i] * DT
        for k in range(2):
            if pos[i][k] > 15:
                pos[i][k] = 15
                vel[i][k] = -abs(vel[i][k])
            if pos[i][k] < -15:
                pos[i][k] = -15
                vel[i][k] = abs(vel[i][k])


def run(scenario, n_dyn, n_static, outdir, seed):
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):
        os.remove(os.path.join(outdir, f))

    starts, goals, vels, dirs = make_robots(rng)
    positions = [p.copy() for p in starts]
    static = make_static(n_static, starts, goals, rng) if n_static else []
    dyn_pos, dyn_vel = make_dynamic(n_dyn, rng) if n_dyn else ([], [])
    dyn_r = radii(n_dyn, rng) if n_dyn else []

    trajs = [[p.copy()] for p in positions]
    reached = [False] * N_ROBOTS
    arrived_t = [None] * N_ROBOTS
    min_clear = [np.inf] * N_ROBOTS
    collisions = [0] * N_ROBOTS
    agent = Agent(device=DEVICE)

    figs = []
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(figsize=(9, 7.2), dpi=100)
    ax.set_xlim(-MAP, MAP)
    ax.set_ylim(-MAP, MAP)
    ax.set_aspect("equal")
    ax.grid(True, color="#e8e8e8", linewidth=0.6)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    header = fig.text(0.09, 0.955, "", fontsize=9, fontweight="bold")

    # legend proxies
    from matplotlib.lines import Line2D
    handles = []
    for i in range(N_ROBOTS):
        handles.append(Line2D([0], [0], marker="o", linestyle="None", color=COLORS[i],
                              label="Robot %d" % (i + 1), markersize=6))
    if n_dyn:
        handles.append(Line2D([0], [0], marker="o", linestyle="None", color="#f2c14e",
                              label="Dynamic obstacle", markersize=6))
    if n_static:
        handles.append(Line2D([0], [0], marker="o", linestyle="None", color="#b0b0b0",
                              label="Static obstacle", markersize=6,
                              markeredgecolor="#8c8c8c"))
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.95,
              borderpad=0.5, labelspacing=0.35)

    for (ox, oy, orr) in static:
        ax.add_patch(Circle((ox, oy), orr, color="#b0b0b0", alpha=0.85, zorder=2))

    for i in range(N_ROBOTS):
        ax.plot(goals[i][0], goals[i][1], marker="*", markersize=12,
                color=COLORS[i], markeredgecolor="#666666", linestyle="None", zorder=3)

    dyn_scatter = None
    dyn_patches = []
    if n_dyn:
        dyn_colors = [DYN_PALETTE[i % len(DYN_PALETTE)] for i in range(n_dyn)]
        for i in range(n_dyn):
            c = Circle((dyn_pos[i][0], dyn_pos[i][1]), dyn_r[i],
                       color=dyn_colors[i], alpha=0.92, zorder=2)
            ax.add_patch(c)
            dyn_patches.append(c)

    robot_dots = []
    traj_lines = []
    for i in range(N_ROBOTS):
        dot, = ax.plot([], [], "o", markersize=9.5, color=COLORS[i],
                       markeredgecolor="#222222", markeredgewidth=0.8, zorder=7)
        line, = ax.plot([], [], "-", linewidth=1.6, color=COLORS[i], alpha=0.9, zorder=4)
        robot_dots.append(dot)
        traj_lines.append(line)

    name = scenario
    for step in range(STEPS):
        t = step * DT
        if n_dyn:
            step_dynamic(dyn_pos, dyn_vel, rng, step)
            for i, c in enumerate(dyn_patches):
                c.set_center((dyn_pos[i][0], dyn_pos[i][1]))

        for i in range(N_ROBOTS):
            pos = positions[i]
            if reached[i]:
                continue
            goal = goals[i]
            dist = np.linalg.norm(goal - pos)
            if dist < GOAL_R:
                reached[i] = True
                arrived_t[i] = t
                continue

            all_pos = [positions[j] for j in range(N_ROBOTS)] + [d for d in dyn_pos]
            all_vel = [vels[j] for j in range(N_ROBOTS)] + [d for d in dyn_vel]

            state = get_robot_state(pos, goal, vels[i], dirs[i], device=DEVICE)
            lidar, _, _ = get_ray_cast(pos, static, max_range=LIDAR_RANGE, hres_deg=HRES,
                                       vfov_angles_deg=VFOV,
                                       start_angle_deg=np.degrees(np.arctan2(dirs[i][1], dirs[i][0])),
                                       device=DEVICE)
            td = torch.tensor(np.append(dirs[i][:2], 0.0), dtype=torch.float,
                              device=DEVICE).unsqueeze(0).unsqueeze(0)
            dyn_state = get_dyn_obs_state(pos, vels[i], all_pos, all_vel, td, device=DEVICE)

            vel = agent.plan(state, lidar, dyn_state, td)
            if vel.shape[0] < 2:
                continue
            vel = safety_filter(pos, vel, static, dyn_pos, dyn_r)
            positions[i] = pos + vel * DT
            vels[i] = vel
            trajs[i].append(positions[i].copy())

        for i in range(N_ROBOTS):
            robot_dots[i].set_data([positions[i][0]], [positions[i][1]])
            tr = np.array(trajs[i])
            traj_lines[i].set_data(tr[:, 0], tr[:, 1])

        # clearance and collision bookkeeping (distance to obstacle surface)
        for i in range(N_ROBOTS):
            cand = [np.hypot(positions[i][0] - o[0], positions[i][1] - o[1]) - o[2] for o in static]
            cand += [np.hypot(positions[i][0] - p[0], positions[i][1] - p[1]) - r
                     for p, r in zip(dyn_pos, dyn_r)]
            d = min(cand) if cand else np.inf
            min_clear[i] = min(min_clear[i], d)
            if d < 0.25:
                collisions[i] = 1

        header.set_text("FT 600 - Multi-Robot Navigation (6 robots) + %s - step %d - Time: %.2fs - reached %d/%d"
                        % (name, step, t, sum(reached), N_ROBOTS))
        if not os.environ.get("NO_FRAMES"):
            fig.savefig(os.path.join(outdir, "f_%04d.png" % step), dpi=100)
        if step % 25 == 0:
            print(step, sum(reached), flush=True)

    plt.close(fig)
    at = [a for a in arrived_t if a is not None]
    mc = [round(float(x), 3) for x in min_clear]
    return {"scenario": scenario, "robots": N_ROBOTS,
            "static": n_static, "dynamic": n_dyn,
            "reached": int(sum(reached)),
            "success_rate": round(100.0 * sum(reached) / N_ROBOTS, 1),
            "near_miss": int(sum(1 for x in min_clear if 0.0 <= x < 0.25)),
            "penetration": int(sum(1 for x in min_clear if x < 0.0)),
            "mean_arrival": round(float(np.mean(at)), 2) if at else None,
            "arrivals": [round(float(a), 2) if a is not None else None for a in arrived_t],
            "mean_min_clearance": round(float(np.mean(min_clear)), 3),
            "worst_min_clearance": round(float(np.min(min_clear)), 3),
            "per_robot_clearance": mc,
            "steps": STEPS, "sim_time": round(STEPS * DT, 1)}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    runs = [
        ("Static Obstacles (15)", "15 Static Obstacles", 0, 15, 202),
        ("Dynamic Obstacles (15)", "15 Dynamic Obstacles", 15, 0, 101),
        ("Mixed Obstacles (15)", "Mixed 8 Dynamic + 7 Static Obstacles", 8, 7, 303),
    ]
    results = []
    for key, label, nd, ns, seed in runs:
        if which != "all" and which != key:
            continue
        outdir = os.path.join(OUTROOT, key.split(" ")[0])
        print("=== running", key, flush=True)
        results.append(run(label, nd, ns, outdir, seed))
        print(results[-1], flush=True)
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                                "results", "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

