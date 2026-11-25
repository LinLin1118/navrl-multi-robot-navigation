# Attribution

`src/navrl_pretrained/` is not my code. It is a copy of files taken from the NavRL
project so the demo can run without installing that repository:

* Upstream: NavRL — Autonomous UAV Navigation with Reinforcement Learning
  (CERLAB, Zhefan Xu), <https://github.com/Zhefan-Xu/NavRL>
* Commit used: `3725bcc2e7c1be4ecf1455d922299ae85042603a`
* Licence: MIT, Copyright (c) 2025 Zhefan Xu (full text in `LICENSE-navrl`)

Copied unchanged, from the upstream `quick-demos/` folder:

| File | What it is |
|---|---|
| `agent.py` | builds the network, loads the checkpoint, exposes `plan()` |
| `ppo.py` | the PPO actor-critic definition |
| `utils.py` | ray casting, robot state and dynamic obstacle encoding |
| `env.py` | the training env reward block, kept for reference |
| `ckpts/navrl_checkpoint.pt` | released checkpoint, 177,388 parameters |

Everything else is mine: `scenarios.py`, `metrics.py`, `reward.py`, `run_scenes.py`, the test
suite, the results and the report.

If you reuse this repository, keep this file and the upstream MIT licence with it.
