"""Offline checkpoint evaluation over scramble depths."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from .checkpoint import CheckpointManager
from .policy import LinearSoftmaxPolicy
from .torch_env import TorchRubikBatchEnv

matplotlib.use("Agg")


@dataclass
class ScrambleMetrics:
    scramble_depth: int
    episodes: int
    solved_count: int
    unsolved_count: int
    success_rate: float
    steps_solved_min: float | None
    steps_solved_mean: float | None
    steps_solved_max: float | None
    steps_all_min: float
    steps_all_mean: float
    steps_all_max: float
    eval_time_sec: float
    episodes_per_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scramble_depth": self.scramble_depth,
            "episodes": self.episodes,
            "solved_count": self.solved_count,
            "unsolved_count": self.unsolved_count,
            "success_rate": self.success_rate,
            "steps_solved_min": self.steps_solved_min,
            "steps_solved_mean": self.steps_solved_mean,
            "steps_solved_max": self.steps_solved_max,
            "steps_all_min": self.steps_all_min,
            "steps_all_mean": self.steps_all_mean,
            "steps_all_max": self.steps_all_max,
            "eval_time_sec": self.eval_time_sec,
            "episodes_per_sec": self.episodes_per_sec,
        }


def _resolve_device(device_name: str) -> torch.device:
    name = device_name.lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested --device cuda, but CUDA is not available")
        return torch.device("cuda")
    if name == "mps":
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            raise ValueError("Requested --device mps, but MPS is not available")
        return torch.device("mps")
    raise ValueError("--device must be one of: cpu, cuda, mps")


def _aggregate_metrics(
    scramble_depth: int,
    solved: np.ndarray,
    steps: np.ndarray,
    eval_time_sec: float,
) -> ScrambleMetrics:
    solved = np.asarray(solved, dtype=bool)
    steps = np.asarray(steps, dtype=np.int64)
    episodes = int(steps.size)
    solved_count = int(solved.sum())
    unsolved_count = int(episodes - solved_count)
    success_rate = float(solved_count / episodes) if episodes > 0 else 0.0

    steps_all_min = float(np.min(steps)) if episodes > 0 else 0.0
    steps_all_mean = float(np.mean(steps)) if episodes > 0 else 0.0
    steps_all_max = float(np.max(steps)) if episodes > 0 else 0.0

    if solved_count > 0:
        solved_steps = steps[solved]
        steps_solved_min = float(np.min(solved_steps))
        steps_solved_mean = float(np.mean(solved_steps))
        steps_solved_max = float(np.max(solved_steps))
    else:
        steps_solved_min = None
        steps_solved_mean = None
        steps_solved_max = None

    eps_per_sec = float(episodes / max(eval_time_sec, 1e-9))
    return ScrambleMetrics(
        scramble_depth=scramble_depth,
        episodes=episodes,
        solved_count=solved_count,
        unsolved_count=unsolved_count,
        success_rate=success_rate,
        steps_solved_min=steps_solved_min,
        steps_solved_mean=steps_solved_mean,
        steps_solved_max=steps_solved_max,
        steps_all_min=steps_all_min,
        steps_all_mean=steps_all_mean,
        steps_all_max=steps_all_max,
        eval_time_sec=float(eval_time_sec),
        episodes_per_sec=eps_per_sec,
    )


def _fmt_opt(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.2f}"


def _print_header() -> None:
    print(
        "scramble | success_rate | solved/total | solved_steps(min/mean/max) | all_steps(min/mean/max) | eps/s",
        flush=True,
    )


def _print_row(m: ScrambleMetrics) -> None:
    solved_steps = f"{_fmt_opt(m.steps_solved_min)}/{_fmt_opt(m.steps_solved_mean)}/{_fmt_opt(m.steps_solved_max)}"
    all_steps = f"{m.steps_all_min:.2f}/{m.steps_all_mean:.2f}/{m.steps_all_max:.2f}"
    print(
        f"{m.scramble_depth:8d} | "
        f"{m.success_rate:11.4f} | "
        f"{m.solved_count:6d}/{m.episodes:<6d} | "
        f"{solved_steps:27s} | "
        f"{all_steps:24s} | "
        f"{m.episodes_per_sec:7.1f}",
        flush=True,
    )


def _plot_metrics(metrics: list[ScrambleMetrics], output_dir: Path, prefix: str) -> tuple[Path, Path]:
    depths = np.array([m.scramble_depth for m in metrics], dtype=np.int64)
    sr = np.array([m.success_rate for m in metrics], dtype=np.float64)

    solved_min = np.array([np.nan if m.steps_solved_min is None else m.steps_solved_min for m in metrics], dtype=np.float64)
    solved_mean = np.array([np.nan if m.steps_solved_mean is None else m.steps_solved_mean for m in metrics], dtype=np.float64)
    solved_max = np.array([np.nan if m.steps_solved_max is None else m.steps_solved_max for m in metrics], dtype=np.float64)

    all_min = np.array([m.steps_all_min for m in metrics], dtype=np.float64)
    all_mean = np.array([m.steps_all_mean for m in metrics], dtype=np.float64)
    all_max = np.array([m.steps_all_max for m in metrics], dtype=np.float64)

    fig1 = plt.figure(figsize=(10, 5))
    ax1 = fig1.add_subplot(111)
    ax1.plot(depths, sr, marker="o", linewidth=2.0)
    ax1.set_title("Checkpoint Evaluation: Success Rate vs Scramble Depth")
    ax1.set_xlabel("Scramble depth")
    ax1.set_ylabel("Success rate")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)
    sr_path = output_dir / f"{prefix}_success_rate.png"
    fig1.tight_layout()
    fig1.savefig(sr_path, dpi=160)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(11, 6))
    ax2 = fig2.add_subplot(111)
    ax2.plot(depths, solved_min, marker="o", linewidth=1.8, label="Solved min")
    ax2.plot(depths, solved_mean, marker="o", linewidth=1.8, label="Solved mean")
    ax2.plot(depths, solved_max, marker="o", linewidth=1.8, label="Solved max")
    ax2.plot(depths, all_min, linestyle="--", alpha=0.7, linewidth=1.5, label="All min")
    ax2.plot(depths, all_mean, linestyle="--", alpha=0.7, linewidth=1.5, label="All mean")
    ax2.plot(depths, all_max, linestyle="--", alpha=0.7, linewidth=1.5, label="All max")
    ax2.set_title("Checkpoint Evaluation: Steps-to-solve Statistics")
    ax2.set_xlabel("Scramble depth")
    ax2.set_ylabel("Steps")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    steps_path = output_dir / f"{prefix}_steps_stats.png"
    fig2.tight_layout()
    fig2.savefig(steps_path, dpi=160)
    plt.close(fig2)

    return sr_path, steps_path


def _save_reports(
    metrics: list[ScrambleMetrics],
    output_dir: Path,
    prefix: str,
    args: argparse.Namespace,
    checkpoint_path: Path,
    loaded_episode: int,
) -> tuple[Path, Path]:
    csv_path = output_dir / f"{prefix}_metrics.csv"
    json_path = output_dir / f"{prefix}_metrics.json"

    fieldnames = [
        "scramble_depth",
        "episodes",
        "solved_count",
        "unsolved_count",
        "success_rate",
        "steps_solved_min",
        "steps_solved_mean",
        "steps_solved_max",
        "steps_all_min",
        "steps_all_mean",
        "steps_all_max",
        "eval_time_sec",
        "episodes_per_sec",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics:
            writer.writerow(m.to_dict())

    payload = {
        "config": {
            "checkpoint_dir": args.checkpoint_dir,
            "checkpoint_path": str(checkpoint_path),
            "loaded_episode": loaded_episode,
            "device": args.device,
            "episodes_per_scramble": int(args.episodes_per_scramble),
            "scramble_min": int(args.scramble_min),
            "scramble_max": int(args.scramble_max),
            "max_episode_steps": int(args.max_episode_steps),
            "eval_batch_size": int(args.eval_batch_size),
            "seed": args.seed,
            "progress": args.progress,
        },
        "metrics": [m.to_dict() for m in metrics],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return csv_path, json_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline checkpoint evaluation on batched torch Rubik environment")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--checkpoint-path", default=None, help="Optional explicit .pt checkpoint path")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--episodes-per-scramble", type=int, default=100000)
    p.add_argument("--scramble-min", type=int, default=1)
    p.add_argument("--scramble-max", type=int, default=20)
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default="eval_reports")
    p.add_argument("--output-prefix", default="checkpoint_eval")
    p.add_argument("--progress", default="on", choices=["on", "off"])
    return p


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.scramble_min < 1 or args.scramble_max < args.scramble_min:
        raise ValueError("Require 1 <= scramble_min <= scramble_max")
    if args.episodes_per_scramble < 1:
        raise ValueError("--episodes-per-scramble must be >= 1")
    if args.max_episode_steps < 1:
        raise ValueError("--max-episode-steps must be >= 1")
    if args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be >= 1")

    device = _resolve_device(args.device)
    ckpt = CheckpointManager(args.checkpoint_dir)
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
        policy, loaded_episode = ckpt.load(checkpoint_path)
    else:
        checkpoint_path = ckpt.latest_path()
        if checkpoint_path is None:
            raise RuntimeError(f"No checkpoint found in '{args.checkpoint_dir}'")
        policy, loaded_episode = ckpt.load(checkpoint_path)
    if policy is None:
        raise RuntimeError("Checkpoint loading returned empty policy")

    policy.to(device)
    policy.eval()
    env = TorchRubikBatchEnv(batch_size=int(args.eval_batch_size), device=device)
    rng = torch.Generator(device=device)
    if args.seed is not None:
        rng.manual_seed(int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "evaluation_init "
        f"checkpoint={checkpoint_path} loaded_episode={loaded_episode} "
        f"device={device.type} episodes_per_scramble={args.episodes_per_scramble} "
        f"scramble_range={args.scramble_min}..{args.scramble_max} max_episode_steps={args.max_episode_steps}",
        flush=True,
    )
    _print_header()

    metrics: list[ScrambleMetrics] = []
    with torch.no_grad():
        for scramble_depth in range(int(args.scramble_min), int(args.scramble_max) + 1):
            t0 = time.perf_counter()
            n = int(args.episodes_per_scramble)
            solved_out = np.zeros((n,), dtype=bool)
            steps_out = np.zeros((n,), dtype=np.int64)
            offset = 0
            n_chunks = (n + int(args.eval_batch_size) - 1) // int(args.eval_batch_size)
            chunk_iter = range(n_chunks)
            if args.progress == "on":
                chunk_iter = tqdm(
                    chunk_iter,
                    desc=f"scramble={scramble_depth}",
                    unit="chunk",
                    mininterval=1.0,
                    leave=False,
                )

            for _ in chunk_iter:
                b = min(int(args.eval_batch_size), n - offset)
                env.reset(b)
                env.scramble(scramble_depth, generator=rng)
                for _step in range(int(args.max_episode_steps)):
                    if bool(env.done.all().item()):
                        break
                    obs = env.build_observation()
                    logits = policy.forward_logits(obs)
                    actions, _ = LinearSoftmaxPolicy.sample_actions_from_logits(logits)
                    env.step(actions)

                solved_np = env.done.detach().cpu().numpy().astype(bool)
                steps_np = env.steps.detach().cpu().numpy().astype(np.int64)
                steps_np = np.where(solved_np, steps_np, int(args.max_episode_steps))
                solved_out[offset : offset + b] = solved_np
                steps_out[offset : offset + b] = steps_np
                offset += b

            elapsed = time.perf_counter() - t0
            m = _aggregate_metrics(scramble_depth, solved_out, steps_out, elapsed)
            metrics.append(m)
            _print_row(m)

    sr_path, steps_path = _plot_metrics(metrics, output_dir, args.output_prefix)
    csv_path, json_path = _save_reports(metrics, output_dir, args.output_prefix, args, checkpoint_path, loaded_episode)

    avg_sr = float(np.mean([m.success_rate for m in metrics]))
    avg_steps_all = float(np.mean([m.steps_all_mean for m in metrics]))
    print(
        "evaluation_summary "
        f"avg_success_rate={avg_sr:.4f} avg_steps_all_mean={avg_steps_all:.2f} "
        f"sr_plot={sr_path} steps_plot={steps_path} csv={csv_path} json={json_path}",
        flush=True,
    )

    return {
        "metrics": metrics,
        "sr_plot": sr_path,
        "steps_plot": steps_path,
        "csv": csv_path,
        "json": json_path,
        "checkpoint_path": checkpoint_path,
        "loaded_episode": loaded_episode,
    }


def main() -> None:
    args = build_parser().parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
