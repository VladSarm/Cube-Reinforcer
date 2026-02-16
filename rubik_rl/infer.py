"""Policy inference runner that drives GUI via animated API steps."""

from __future__ import annotations

import argparse
import numpy as np

from tqdm import tqdm

from .checkpoint import CheckpointManager
from .client import RubikAPIClient


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run inference policy for Rubik 2x2")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--scramble-steps", type=int, required=True)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--step-duration-ms", type=int, default=400)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=None)
    return p


def run_inference(args: argparse.Namespace) -> bool:
    ckpt = CheckpointManager(args.checkpoint_dir)
    policy, episode = ckpt.load_latest()
    if policy is None:
        raise RuntimeError("No checkpoints found. Train first.")

    client = RubikAPIClient(host=args.host, port=args.port)

    print(f"inference_init host={args.host}:{args.port} scramble_steps={args.scramble_steps}", flush=True)
    print("api_call POST /reset", flush=True)
    client.reset()
    print(f"api_call POST /scramble steps={args.scramble_steps}", flush=True)
    client.scramble(args.scramble_steps)

    solved = False
    action_history: list[int] = []
    for step_idx in tqdm(range(1, args.max_steps + 1), desc="Inference steps", unit="step"):
        print(f"api_call GET /state step={step_idx}", flush=True)
        state_oh = client.get_state()
        hist_oh = policy.history_one_hot(action_history)
        action, probs = policy.sample_action(state_oh, hist_oh)
        print(f"api_call POST /step_animated action={action} duration_ms={args.step_duration_ms}", flush=True)
        out = client.step_animated(action, duration_ms=args.step_duration_ms)
        action_history = (action_history + [action])[-4:]
        solved = bool(out["solved"])
        print(f"step={step_idx} action={action} p={probs[action]:.4f} solved={solved}", flush=True)
        if solved:
            break

    print(f"inference_done solved={solved} loaded_from_episode={episode}", flush=True)
    return solved


def main() -> None:
    args = build_parser().parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
