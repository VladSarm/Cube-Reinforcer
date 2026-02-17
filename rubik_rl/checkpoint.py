"""Checkpoint management for PyTorch policy weights."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .policy import LinearSoftmaxPolicy


class CheckpointManager:
    FILE_PATTERN = re.compile(r"policy_ep(\d+)\.(pt|npz)$")

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
        for p in self.dir.glob("policy_ep*.*"):
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
        return self.load(path)

    def _load_legacy_npz(self, path: Path) -> tuple[LinearSoftmaxPolicy, int]:
        data = np.load(path, allow_pickle=True)
        episode = int(np.asarray(data["episode"]).reshape(-1)[0]) if "episode" in data else 0
        policy = LinearSoftmaxPolicy()

        with torch.no_grad():
            if all(k in data for k in ("W1", "b1", "W2", "b2")):
                policy.linear1.weight.copy_(torch.from_numpy(np.asarray(data["W1"], dtype=np.float32).T))
                policy.linear1.bias.copy_(torch.from_numpy(np.asarray(data["b1"], dtype=np.float32)))
                policy.linear2.weight.copy_(torch.from_numpy(np.asarray(data["W2"], dtype=np.float32).T))
                policy.linear2.bias.copy_(torch.from_numpy(np.asarray(data["b2"], dtype=np.float32)))
            elif all(k in data for k in ("W", "b")):
                # Legacy single-layer to two-layer embedding (same trick as before).
                W = np.asarray(data["W"], dtype=np.float32)
                b = np.asarray(data["b"], dtype=np.float32)
                in_dim = min(W.shape[0], policy.INPUT_DIM)
                act_dim = min(W.shape[1], policy.ACTION_DIM)
                policy.linear1.weight.zero_()
                policy.linear1.bias.zero_()
                policy.linear2.weight.zero_()
                policy.linear2.bias.zero_()
                policy.linear1.weight[:act_dim, :in_dim] = torch.from_numpy(W[:in_dim, :act_dim].T)
                bias_shift = 10.0
                policy.linear1.bias[:act_dim] = torch.from_numpy(b[:act_dim] + bias_shift)
                for i in range(act_dim):
                    policy.linear2.weight[i, i] = 1.0
                policy.linear2.bias[:] = -bias_shift
            else:
                raise ValueError(f"Checkpoint {path} does not contain supported policy weights")
        return policy, episode

    def load(self, path: str | Path) -> tuple[LinearSoftmaxPolicy, int]:
        path = Path(path)
        if path.suffix == ".npz":
            return self._load_legacy_npz(path)

        data = torch.load(path, map_location="cpu")
        if not isinstance(data, dict) or "model_state_dict" not in data:
            raise ValueError(f"Checkpoint {path} is not a valid PyTorch checkpoint")

        state = data["model_state_dict"]
        policy = LinearSoftmaxPolicy.from_state_dict(state)
        episode = int(data.get("episode", 0))
        return policy, episode

