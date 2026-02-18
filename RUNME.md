# Cube-Reinforcer — Run & Code

How to install, run tests, start the simulator, train, and evaluate. For problem background, math, and experiment results see [README.md](README.md).

---

## Table of Contents
1. [Repository Structure](#repository-structure)
2. [Installation](#installation)
3. [Running Tests](#running-tests)
4. [Simulator](#simulator)
5. [Training](#training)
6. [Evaluation](#evaluation)

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
- policy gradient math / autograd sanity checks,
- checkpoint save/load,
- train/infer smoke integration.

---

## Simulator

### Simulator CLI
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

### HTTP API
- `GET /health`
- `GET /state`
- `POST /state`
- `POST /reset`
- `POST /scramble`
- `POST /step`
- `POST /step_animated`
- `GET /solved`

### State and Observation Reference
- API state format: one-hot matrix `shape = (24, 6)`.
- RL observation: cube one-hot `24*6 = 144`, action history one-hot (last 4 actions) `4*12 = 48`, total input size `192`.

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

## Training

### Training CLI
```bash
uv run python -m rubik_rl.trainer [args]
```

Main arguments:
- `--episodes` (required)
- `--num-envs` (parallel env count)
- `--scramble-steps` (initial fixed scramble depth for curriculum)
- `--max-episode-steps`
- `--gamma`
- `--lr`
- `--save-every`
- `--checkpoint-dir`
- `--seed`
- `--log-interval`
- `--device` (`cpu`, `cuda`, `mps`)
- `--tensorboard-logdir` (base directory; each run creates `run_YYYYMMDD_HHMMSS`)
- `--exp-name` (optional experiment name; if set logs go to `<tensorboard-logdir>/<exp-name>/run_YYYYMMDD_HHMMSS` and name is written to TensorBoard)
- `--torch-env` / `--no-torch-env` (torch-env backend flag; currently torch-env is the intended training backend)

Training mode:
- **No servers are started during training.** The trainer runs batched cube logic in `TorchRubikBatchEnv` directly on the selected device for speed.

Example:
```bash
# Fast local training (local cube engines, no HTTP)
uv run python -m rubik_rl.trainer \
  --num-envs 16 \
  --episodes 200000 \
  --scramble-steps 5 \
  --max-episode-steps 40 \
  --gamma 0.95 \
  --lr 1e-4 \
  --device cuda \
  --exp-name baseline_scr3 \
  --tensorboard-logdir runs/cube_reinforcer \
  --save-every 5000 \
  --checkpoint-dir checkpoints \
  --log-interval 1000
```

---

## Evaluation

### 1) GUI Eval Button
Run GUI and click `Eval: OFF` -> `Eval: ON`.
- GUI loads latest checkpoint from `checkpoints/`.
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

### 3) Offline Checkpoint Evaluation (no GUI / no HTTP)
Runs local batched evaluation for scramble depths `1..20`, each with many random episodes.

Default command (full run):
```bash
uv run python -m rubik_rl.evaluate_checkpoint \
  --checkpoint-dir checkpoints \
  --device cpu \
  --episodes-per-scramble 100000 \
  --scramble-min 1 \
  --scramble-max 20 \
  --max-episode-steps 100 \
  --eval-batch-size 4096 \
  --output-dir eval_reports \
  --output-prefix checkpoint_eval \
  --progress on
```

Wrapper script:
```bash
uv run python scripts/evaluate_checkpoint.py --checkpoint-dir checkpoints
```

Quick smoke run:
```bash
uv run python -m rubik_rl.evaluate_checkpoint \
  --checkpoint-dir checkpoints \
  --episodes-per-scramble 64 \
  --scramble-min 1 \
  --scramble-max 2 \
  --max-episode-steps 10 \
  --eval-batch-size 32 \
  --progress off
```

Outputs:
- PNG plots: `<output-dir>/<output-prefix>_success_rate.png`, `<output-dir>/<output-prefix>_steps_stats.png`
- Machine-readable: `<output-dir>/<output-prefix>_metrics.csv`, `<output-dir>/<output-prefix>_metrics.json`

Reported metrics per scramble depth: `success_rate`; solved-only steps `steps_solved_min/mean/max`; all-episodes steps `steps_all_min/mean/max`.
