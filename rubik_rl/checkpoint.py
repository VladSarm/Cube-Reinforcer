"""Checkpoint management for PyTorch policy weights."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

from .policy import LinearSoftmaxPolicy


class CheckpointManager:
    FILE_PATTERN = re.compile(r"policy_ep(\d+)\.pt$")

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path_for_episode(self, episode: int) -> Path:
        return self.dir / f"policy_ep{episode:07d}.pt"

    def save(
        self,
        policy: LinearSoftmaxPolicy,
        episode: int,
        optimizer: torch.optim.Optimizer | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        path = self._path_for_episode(episode)
        payload: dict[str, Any] = {
            "episode": int(episode),
            "model_state_dict": policy.state_dict(),
            "metadata": metadata or {},
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(payload, path)
        return path

    def latest_path(self) -> Path | None:
        best_ep = -1
        best_path: Path | None = None
        for p in self.dir.glob("policy_ep*.pt"):
            m = self.FILE_PATTERN.search(p.name)
            if not m:
                continue
            ep = int(m.group(1))
            if ep > best_ep:
                best_ep = ep
                best_path = p
        return best_path

    def load_latest(self) -> tuple[LinearSoftmaxPolicy | None, int]:
        path = self.latest_path()
        if path is None:
            return None, 0
        try:
            return self.load(path)
        except Exception as exc:
            print(f"checkpoint_warning incompatible checkpoint '{path}': {exc}; using random init", flush=True)
            return None, 0

    def load(self, path: str | Path) -> tuple[LinearSoftmaxPolicy, int]:
        path = Path(path)
        if path.suffix != ".pt":
            raise ValueError(f"Unsupported checkpoint format: {path.suffix}. Only .pt is supported")

        data = torch.load(path, map_location="cpu")
        if not isinstance(data, dict) or "model_state_dict" not in data:
            raise ValueError(f"Checkpoint {path} is not a valid PyTorch checkpoint")

        state = data["model_state_dict"]
        policy = LinearSoftmaxPolicy.from_state_dict(state)
        episode = int(data.get("episode", 0))
        return policy, episode
