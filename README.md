# Multi-robot navigation with a pretrained NavRL policy

Course assignment for Machine Learning 6. Six quadrotors navigate a 36 x 36 m field
with fifteen obstacles in three configurations, using a policy that was **not**
trained here: the released NavRL checkpoint is used for inference only, on CPU.

## Demos

Three scenarios, the same policy, 6 robots and 15 obstacles each. Each clip is
560 simulation steps of 0.1 s (56 s), played back at 6 fps.

<p align="center">
  <img src="media/nav_Static.gif" width="32%" alt="Static obstacles" />
  <img src="media/nav_Dynamic.gif" width="32%" alt="Dynamic obstacles" />
  <img src="media/nav_Mixed.gif" width="32%" alt="Mixed obstacles" />
</p>

| Scenario | Static | Dynamic | Success | Mean arrival | Full recording |
|---|---|---|---|---|---|
| Static (15) | 15 | 0 | 100% | 9.85 s | [nav_Static.mp4](media/nav_Static.mp4) |
| Dynamic (15) | 0 | 15 | 100% | 9.68 s | [nav_Dynamic.mp4](media/nav_Dynamic.mp4) |
| Mixed (7+8) | 7 | 8 | 100% | 11.08 s | [nav_Mixed.mp4](media/nav_Mixed.mp4) |

The coloured dots are the six robots, the pale dots are obstacles (moving in the
dynamic and mixed scenes), the stars are the goals and the lines are the paths
already taken. The plot is a top-down view rendered with matplotlib, so it needs no
GPU and no Isaac Sim; every position in it is the simulated state at that step.

## Results

| Scenario | Static | Dynamic | Reached | Success | Near miss | Penetration | Mean arrival (s) | Worst clearance (m) |
|---|---|---|---|---|---|---|---|---|
| Static (15) | 15 | 0 | 6/6 | 100% | 0 | 0 | 9.85 | +0.356 |
| Dynamic (15) | 0 | 15 | 6/6 | 100% | 1 | 4 | 9.68 | -0.381 |
| Mixed (15) | 7 | 8 | 6/6 | 100% | 0 | 2 | 11.08 | -0.322 |

<p align="center">
  <img src="results/fig_nav_scenarios.png" width="92%" alt="Scenario results" />
</p>

Success means arriving within 0.6 m of the goal. Contact is measured against the
obstacle surface with the 0.25 m robot radius: a negative clearance means the hull
overlapped an obstacle. Every scene runs 6 robots, 15 obstacles, 560 steps of 0.1 s.
The write-up is in `docs/NavRL_MultiRobot_Analysis.docx`, the recordings are in
`media/`, the raw numbers are in `results/`.

## Installation

Nothing here needs a GPU, CUDA or Isaac Sim. It was developed and measured on a
CPU-only Ubuntu 22.04 virtual machine.

```bash
# Python 3.10 (3.9-3.11 work); a fresh environment avoids torch version clashes
conda create -n navrl python=3.10 -y
conda activate navrl

pip install -r requirements.txt
```

`requirements.txt` pins the inference stack:

```
numpy
matplotlib
torch>=2.0
tensordict>=0.4
torchrl>=0.4
```

On a machine without a GPU, install the CPU build of torch first so pip does not
drag in a 2 GB CUDA wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The vendored inference code and the 707 KB checkpoint in `src/navrl_pretrained/` are
part of this repository, so no other download is needed. `ATTRIBUTION.md` records
where they come from.

Verified on that machine: Python 3.10.22, torch 2.0.1+cpu, numpy 1.26.4,
matplotlib 3.10.9. One episode of 560 steps takes about 40 s on a single CPU core;
the three scenes with frame rendering take 5-6 minutes end to end.

## Repository layout

```
src/
  scenarios.py         scenario sampling, safety filter, clearance  (NumPy only)
  metrics.py           success rate, near miss, penetration, clearance  (NumPy only)
  reward.py            the training reward equation, transcribed and testable
  run_scenarios.py     runs the three scenes with the policy, writes results/
  navrl_pretrained/    vendored inference code + checkpoint (see ATTRIBUTION.md)
tests/                 unittest suite, 46 tests
results/               results.json, summary.csv, figure
docs/                  report and one frame per scene
media/                 three mp4 recordings and their GIF previews
```

The split matters: `scenarios.py` and `metrics.py` have no torch dependency, so the
geometry and the metric definitions are testable on any machine. Only
`run_scenarios.py` needs the deep learning stack.

## Running the scenes

```bash
python src/run_scenarios.py --out runs/            # ~4 minutes on one CPU core
python src/run_scenarios.py --out runs/ --no-frames  # numbers only, faster
python src/run_scenarios.py --scene dynamic --out runs/  # one scene only
```

Then write the numbers back into `results/` (the script does this automatically) and,
if you rendered frames, encode them:

```bash
ffmpeg -y -framerate 10 -i runs/Static/f_%04d.png -c:v libx264 -pix_fmt yuv420p \
       -r 24 -crf 20 media/nav_Static.mp4
```

## Running the tests

```bash
python -m unittest discover -s tests -t .
```

46 tests, no third-party runner needed. They cover the sampling rules (start and
goal separation, obstacle placement, arena bounds), the safety filter (a step into
an obstacle is changed, a step outside the arena is stopped), the metric definitions,
the reward equation below, and a consistency check on the shipped
`results/results.json` so the report cannot drift away from the data.

## Reward function

The policy is not trained in this repository, so the reward is never evaluated at
run time. It is documented here because the behaviour of the checkpoint is a direct
consequence of how it was shaped, and because the equation should be readable as
code rather than only as prose: `src/reward.py` implements it and
`tests/test_reward.py` checks every term.

Transcribed from the upstream training environment,
`isaac-training/training/scripts/env.py` lines 546-577 at commit `3725bcc`:

$$
r_t \;=\; \underbrace{v_t \cdot \hat{g}_t}_{\text{goal progress}}
\;+\; \underbrace{1}_{\text{step bonus}}
\;+\; \underbrace{\frac{1}{WH}\sum_{w,h}
      \log \operatorname{clamp}\!\left(r_{\max} - \lambda_{t,w,h},\ \epsilon,\ r_{\max}\right)}_{\text{static safety}}
\;+\; \underbrace{\frac{1}{K}\sum_{k=1}^{K}
      \log \operatorname{clamp}\!\left(\delta_{t,k},\ \epsilon,\ r_{\max}\right)}_{\text{dynamic safety (only if } K>0\text{)}}
\;-\; \underbrace{0.1\,\lVert v_t - v_{t-1}\rVert}_{\text{smoothness}}
\;-\; \underbrace{8\left[(z_t - z_{hi})_+^2 + (z_{lo} - z_t)_+^2\right]}_{\text{height penalty}}
$$

| Term | Symbol | Weight | Why it has this shape |
|---|---|---|---|
| Goal progress | $v_t \cdot \hat{g}_t$ | +1 | The only term that rewards getting closer; a hovering robot earns nothing from it. |
| Step bonus | $1$ | +1 | Keeps the return positive while alive; it is a constant, so it cannot be farmed by loitering. |
| Static safety | $\frac{1}{WH}\sum \log(r_{\max}-\lambda)$ | +1 | Logarithmic, so the reward saturates at the 4 m sensor range and one close beam dominates. |
| Dynamic safety | $\frac{1}{K}\sum \log \delta_{t,k}$ | +1 | Same shape for the $K=5$ nearest moving obstacles; omitted when a scene has none. |
| Smoothness | $\lVert v_t - v_{t-1}\rVert$ | −0.1 | Penalises acceleration, not speed, so fast but steady flight stays cheap. |
| Height | $(z-z_{hi})_+^2 + (z_{lo}-z)_+^2$ | −8 | Zero inside the band; eight times heavier than smoothness because leaving it ends the episode. |
| Terminal step | — | 0 | Collision, goal and timeout all end the episode with no bonus, so a bad episode cannot be bought back. |

Here $v_t$ is the world-frame velocity command, $\hat{g}_t$ the unit vector to the
goal, $\lambda_{t,w,h}$ the $36 \times 4$ LiDAR range image with $r_{\max}=4$ m, and
$\delta_{t,k}$ the distance to the surface of the $k$-th nearest moving obstacle,
clamped to the sensor range. Every term is evaluated once per control step (10 Hz).

## The policy

An actor-critic with three input branches:

* LiDAR branch — three convolutions (4 channels 5x3, then 16 at stride 2x1, then 16
  at stride 2x2, ELU), a 128-unit linear layer and layer normalisation, turning the
  36 x 4 range image into 128 features.
* Dynamic obstacle branch — MLP 50 -> 128 -> 64.
* Ego state — 8 values.

The three are concatenated (200 features) and passed through two 256-unit layers.
The actor predicts the alpha and beta of a Beta distribution over the three velocity
commands, the critic predicts a value. 177,388 parameters in total. At rollout time
the distribution mean is used.

## What is honest about these numbers

* The checkpoint is the released one. Nothing was trained or fine-tuned.
* One seed per scenario (202, 101, 303), so these are three particular fields, not
  an average over layouts.
* Contacts in the dynamic scenes are real. The safety filter only vetoes the robot's
  own next step and assumes obstacles stay put, so a fast lateral approach can still
  close the gap between two decisions. This is discussed in Section 3 of the report.

## References

1. Z. Xu, X. Han, H. Shen, H. Jin, K. Shimada, "NavRL: Learning Safe Flight in Dynamic
   Environments," *IEEE Robotics and Automation Letters*, vol. 10, no. 4,
   pp. 3668-3675, 2025. doi:[10.1109/LRA.2025.3546069](https://doi.org/10.1109/LRA.2025.3546069),
   preprint [arXiv:2409.15634](https://arxiv.org/abs/2409.15634).
2. Z. Xu, "NavRL — Autonomous UAV Navigation with Reinforcement Learning," source code
   at commit `3725bcc2e7c1be4ecf1455d922299ae85042603a`,
   [github.com/Zhefan-Xu/NavRL](https://github.com/Zhefan-Xu/NavRL), MIT licence.
   The files vendored into `src/navrl_pretrained/` are listed in `ATTRIBUTION.md`.
3. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal Policy
   Optimization Algorithms," [arXiv:1707.06347](https://arxiv.org/abs/1707.06347), 2017.
   The actor-critic algorithm the checkpoint was trained with.
4. J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel, "High-Dimensional
   Continuous Control Using Generalized Advantage Estimation,"
   [arXiv:1506.02438](https://arxiv.org/abs/1506.02438), 2015. The GAE estimator used
   in the upstream PPO configuration (`gamma = 0.99`, `lambda = 0.95`).

## Acknowledgement

The inference code and the checkpoint in `src/navrl_pretrained/` come from the NavRL
project (Zhefan Xu, MIT licence). See `ATTRIBUTION.md` for the exact files.
