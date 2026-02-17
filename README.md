# Cube-Reinforcer

`Cube-Reinforcer` is a research-style project about solving the **2x2 Rubik’s Cube** with a custom simulator and a NumPy-only REINFORCE pipeline.

The repository combines:
- a mathematically correct 2x2 cube engine with 12 discrete actions,
- a GUI + HTTP simulator API for control and visualization,
- a policy-gradient training stack for RL experiments.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Why This Problem Is Hard](#why-this-problem-is-hard)
3. [2x2 Rubik Cube Facts](#2x2-rubik-cube-facts)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Running Tests](#running-tests)
7. [Simulator](#simulator)
8. [Training (REINFORCE, NumPy)](#training-reinforce-numpy)
9. [Evaluation](#evaluation)
10. [Experiments](#experiments)
11. [Current Limits and Notes](#current-limits-and-notes)

---

## Project Overview
The project is focused on RL for a compact but non-trivial combinatorial domain:
- **Environment**: true 2x2 Rubik cube dynamics.
- **Action space**: 12 actions (`U/D/L/R/F/B` with `+/-` quarter-turns).
- **State**: one-hot stickers (`24 x 6`), with additional one-hot history of the last 4 actions.
- **Policy**: two-layer network implemented manually in NumPy.
- **Algorithm**: REINFORCE with discounted returns (no learned value network).

Main goals:
- produce a deterministic, testable simulator,
- provide online control via HTTP for RL loops,
- train and evaluate policies with checkpoint resume.

---

## Why This Problem Is Hard
Even for a 2x2 cube, the task is difficult for RL because:
- transitions are deterministic but highly non-linear in sticker space,
- rewards are sparse if only solved/not solved is used,
- local action effects are deceptive (easy to undo progress),
- representation and exploration strategy strongly affect sample efficiency.

From an engineering perspective, complexity comes from:
- exact move permutations,
- orientation-invariant solved checks,
- synchronized GUI + HTTP + animation,
- stable training with parallel environments and custom gradients.

---

## 2x2 Rubik Cube Facts
- The 2x2 cube (Pocket Cube) has **3,674,160 reachable states**.
- Every state is solvable in at most **11 face turns** in the standard optimal metric for 2x2.
- Unlike 3x3, 2x2 has only corner cubies (no fixed centers/edges), but orientation/permutation constraints still make the space highly structured.

---

## Repository Structure
```text
Cube-Reinforcer/
├── rubik_sim/                 # Simulator engine, GUI, HTTP server
│   ├── actions.py
│   ├── engine.py
│   ├── gui.py
│   ├── server.py
│   ├── state_codec.py
│   └── cli.py
├── rubik_rl/                  # RL components
│   ├── policy.py
│   ├── trainer.py
│   ├── infer.py
│   ├── checkpoint.py
│   ├── reward.py
│   └── client.py
├── scripts/
│   ├── train_reinforce.py
│   └── infer_policy.py
└── tests/
```

---

## Installation
The project uses `uv` for environment and dependency management.

1. Install dependencies:
```bash
cd Cube-Reinforcer
uv sync
```

2. Run modules with project environment:
```bash
uv run python -m rubik_sim.cli --help
uv run python -m rubik_rl.trainer --help
```

---

## Running Tests
Run full test suite:
```bash
cd Cube-Reinforcer
uv run pytest -q
```

The tests cover:
- engine move correctness and inverse properties,
- solved-check logic (including global orientation invariance),
- API behavior,
- policy gradient math (finite-difference check),
- checkpoint save/load,
- train/infer smoke integration.

---

## Simulator
### Purpose
The simulator provides:
- exact 2x2 dynamics for RL,
- one-hot state API for agents,
- GUI for interactive inspection and animated evaluation.

### Full Functionality
- True 2x2 cube transitions via precomputed sticker permutations.
- `cube_size` hyperparameter exposed in CLI (`v1` supports only `2`).
- Scramble with optional seed.
- Scramble logic excludes immediate inverse of the previous scramble action.
- Orientation-invariant solved check.
- Headless HTTP server mode.
- GUI mode with:
  - 3D-style rendering,
  - mouse camera rotation,
  - face rotation animation,
  - `Scramble`, `Reset`, `Eval ON/OFF`,
  - `Anti-repeat x5` checkbox in eval mode.

### GUI Controls
- Mouse drag: rotate camera.
- Keyboard turns:
  - `U/J` -> `U+/U-`
  - `D/C` -> `D+/D-`
  - `L/K` -> `L+/L-`
  - `R/E` -> `R+/R-`
  - `F/G` -> `F+/F-`
  - `B/N` -> `B+/B-`
- `ESC` or `Q`: quit.

### State and Observation Format
- API state format: one-hot matrix `shape = (24, 6)`.
- Internally, color IDs are also supported as flat length-24 vectors.
- RL observation:
  - cube one-hot: `24 * 6 = 144`,
  - action history one-hot (last 4 actions): `4 * 12 = 48`,
  - total input size: `192`.

### Action Index Table
| Action | Meaning |
|---|---|
| 0 | U+ |
| 1 | U- |
| 2 | D+ |
| 3 | D- |
| 4 | L+ |
| 5 | L- |
| 6 | R+ |
| 7 | R- |
| 8 | F+ |
| 9 | F- |
| 10 | B+ |
| 11 | B- |

### HTTP API
- `GET /health`
- `GET /state`
- `POST /state`
- `POST /reset`
- `POST /scramble`
- `POST /step`
- `POST /step_animated`
- `GET /solved`

### Simulator CLI Parameters
Command:
```bash
uv run python -m rubik_sim.cli <mode> [options]
```
Modes:
- `gui`
- `headless`

Common options:
- `--host` (default `127.0.0.1`)
- `--port` (default `8000`)
- `--cube-size` (default `2`, currently only `2` supported)
- `--state-json` (JSON string with initial state)
- `--state-file` (path to JSON file with initial state)

Mode options:
- `--scramble-steps` (default `20`)

Examples:
```bash
# GUI + HTTP server
uv run python -m rubik_sim.cli gui --host 127.0.0.1 --port 8000 --scramble-steps 8

# Headless server
uv run python -m rubik_sim.cli headless --host 127.0.0.1 --port 8001 --scramble-steps 8

# Start with explicit state from file
uv run python -m rubik_sim.cli gui --state-file ./state.json
```

### System Sketch (ASCII)
```text
                    +---------------------+
                    |   rubik_rl.trainer  |
                    | (REINFORCE updates) |
                    +----------+----------+
                               |
                               | HTTP (/state, /step, /scramble, /reset)
                               v
 +--------------------+    +---------------------+    +------------------+
 | rubik_rl.infer.py  +--->| rubik_sim.server.py |--->| rubik_sim.engine |
 | policy checkpoint  |    |  (headless or GUI)  |    |  cube dynamics   |
 +--------------------+    +----------+----------+    +--------+---------+
                                      |                        |
                                      | GUI mode               | one-hot state
                                      v                        v
                                 +-----------+         +-------------------+
                                 | rubik_sim |         | rubik_sim.codec   |
                                 |   .gui    |         | state validation  |
                                 +-----------+         +-------------------+
```

---

## Training (REINFORCE, NumPy)
### Algorithm and Notation
Notation follows standard REINFORCE lecture style:
- trajectory: $\tau = (s_0,a_0,r_1,\dots,s_T)$
- policy: $\pi_\theta(a_t\mid s_t)$
- discounted return:
```math
G_t = \sum_{k=0}^{T-t-1}\gamma^k r_{t+k+1}
```

Policy objective:
```math
J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_t G_t \log \pi_\theta(a_t\mid s_t)\right]
```

Gradient estimator:
```math
\nabla_\theta J(\theta)\approx \sum_t G_t \nabla_\theta \log \pi_\theta(a_t\mid s_t)
```

### Network Structure
Current policy in code:
- input $x\in\mathbb{R}^{192}$,
- first affine layer:
```math
  h_{\text{pre}} = xW_1 + b_1,\quad W_1\in\mathbb{R}^{192\times 128}
```
- activation:
  ```math
  h = \text{ELU}(h_{\text{pre}})
  ```
- output affine layer:
  ```math
  z = hW_2 + b_2,\quad W_2\in\mathbb{R}^{128\times 12}
  ```
- action probabilities:
  ```math
  \pi = \text{softmax}(z)
  ```

### Manual Gradient Derivation (Implemented)
For sampled action $a$ and one-hot $e_a$:
$$
\delta = \frac{\partial \log \pi_\theta(a\mid s)}{\partial z} = e_a - \pi
$$
Then:
$$
\frac{\partial \log \pi}{\partial W_2} = h^\top \delta,\qquad
\frac{\partial \log \pi}{\partial b_2} = \delta
$$
$$
g_h = W_2\delta,\qquad
g_{\text{pre}} = g_h \odot \text{ELU}'(h_{\text{pre}})
$$
$$
\frac{\partial \log \pi}{\partial W_1} = x^\top g_{\text{pre}},\qquad
\frac{\partial \log \pi}{\partial b_1} = g_{\text{pre}}
$$

Batch update (average over parallel environments):
$$
\Delta\theta = \frac{1}{B}\sum_{i=1}^{B}\sum_t G_t^{(i)}\nabla_\theta \log \pi_\theta(a_t^{(i)}\mid s_t^{(i)})
$$

### Reward Shaping
Default reward components:
- step reward: `-1`,
- inverse-to-previous-action penalty: `-20`,
- 4 identical actions in a row penalty: `-100`,
- timeout (unsolved at max steps): additional `-100`.

### Training CLI
```bash
uv run python -m rubik_rl.trainer [args]
```

Main arguments:
- `--episodes` (required)
- `--num-envs` (parallel env count)
- `--scramble-steps` (maximum scramble depth; per episode sampled uniformly from `1..max`)
- `--max-episode-steps`
- `--gamma`
- `--lr`
- `--save-every`
- `--checkpoint-dir`
- `--seed`
- `--log-interval`
- `--stats-window`

Important training details:
- **Scramble sampling rule**: for each episode, scramble depth is sampled as
  $$
  s \sim \mathcal{U}\{1,\dots,S_{\max}\},\quad S_{\max}=\texttt{--scramble-steps}
  $$
  so `--scramble-steps` is an upper bound, not a fixed depth.
- **Batch-size / learning-rate behavior**: gradients are **averaged** across parallel environments, not summed:
  $$
  g_{\text{batch}}=\frac{1}{B}\sum_{i=1}^{B} g_i,\qquad
  \theta \leftarrow \theta + \text{lr}\cdot g_{\text{batch}}
  $$
  therefore the update scale is stable when `--num-envs` changes (you do not need to manually divide `lr` by batch size).

Examples:
```bash
# Fast local parallel training (internal process pool envs)
uv run python -m rubik_rl.trainer \
  --num-envs 16 \
  --episodes 200000 \
  --scramble-steps 5 \
  --max-episode-steps 40 \
  --gamma 0.95 \
  --lr 1e-4 \
  --save-every 5000 \
  --checkpoint-dir checkpoints \
  --log-interval 1000 \
  --stats-window 5000

# Train against already running external server (single env only)
uv run python -m rubik_rl.trainer \
  --external-server \
  --host 127.0.0.1 \
  --port 8000 \
  --num-envs 1 \
  --episodes 50000 \
  --scramble-steps 4
```

---

## Evaluation
There are two evaluation flows.

### 1) GUI Eval Button
Run GUI and click `Eval: OFF` -> `Eval: ON`.
- GUI loads latest checkpoint from `checkpoints/`.
- It applies actions with animation until solved.
- Optional checkbox `Anti-repeat x5`: if model proposes the same action 5th time in a row, GUI uses the second-best action by probability.

Start GUI:
```bash
uv run python -m rubik_sim.cli gui --host 127.0.0.1 --port 8000
```

### 2) Inference Script (separate process)
```bash
uv run python -m rubik_rl.infer \
  --host 127.0.0.1 \
  --port 8000 \
  --scramble-steps 6 \
  --max-steps 400 \
  --step-duration-ms 350 \
  --checkpoint-dir checkpoints
```

or:
```bash
uv run python scripts/infer_policy.py --host 127.0.0.1 --port 8000 --scramble-steps 6
```

---

## Experiments
_This section is intentionally prepared as a template._

### 1. Training Curves
![Experiment: solve rate](docs/images/exp_solve_rate.png)
![Experiment: average return](docs/images/exp_avg_return.png)

### 2. Ablations
- [ ] Reward shaping ablation
- [ ] `num-envs` scaling
- [ ] History length ablation
- [ ] Scramble curriculum (`--scramble-steps`)

### 3. Qualitative Evaluation
![Example rollout 1](docs/images/exp_rollout_1.gif)
![Example rollout 2](docs/images/exp_rollout_2.gif)

### 4. Notes
- TODO: add exact hyperparameter tables.
- TODO: attach seed-averaged metrics.
- TODO: compare against no-normalization baseline.

---

## Current Limits and Notes
- `cube_size` is exposed, but v1 supports only `2`.
- `pygame` may print a `pkg_resources` deprecation warning; this is external to project logic.
- For reproducibility, keep checkpoint directory and CLI args saved with runs.
