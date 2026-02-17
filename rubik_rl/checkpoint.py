"""Checkpoint management for policy weights."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .policy import LinearSoftmaxPolicy
from .policy_sparse import SparseLinearSoftmaxPolicy


def load_sparse_latest(checkpoint_dir: str = "checkpoints_sparse") -> tuple[SparseLinearSoftmaxPolicy | None, int]:
    """Load latest sparse policy (6 actions, 31-dim input) from directory. Returns (policy, episode)."""
    dir_path = Path(checkpoint_dir)
    if not dir_path.is_dir():
        return None, 0
    best_ep = -1
    best_path: Path | None = None
    for p in dir_path.glob("policy_ep*.npz"):
        m = CheckpointManager.FILE_PATTERN.search(p.name)
        if not m:
            continue
        ep = int(m.group(1))
        if ep > best_ep:
            best_ep = ep
            best_path = p
    if best_path is None:
        return None, 0
    data = np.load(best_path, allow_pickle=True)
    episode = int(np.asarray(data["episode"]).reshape(-1)[0])
    policy = SparseLinearSoftmaxPolicy(
        W1=np.asarray(data["W1"], dtype=np.float64),
        b1=np.asarray(data["b1"], dtype=np.float64),
        W2=np.asarray(data["W2"], dtype=np.float64),
        b2=np.asarray(data["b2"], dtype=np.float64),
    )
    if "rng_state_json" in data:
        try:
            policy.rng.bit_generator.state = json.loads(str(np.asarray(data["rng_state_json"]).reshape(-1)[0]))
        except Exception:
            pass
    return policy, episode


class CheckpointManager:
    FILE_PATTERN = re.compile(r"policy_ep(\d+)\.npz$")

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path_for_episode(self, episode: int) -> Path:
        return self.dir / f"policy_ep{episode:07d}.npz"

    def save(self, policy: LinearSoftmaxPolicy, episode: int, metadata: dict | None = None) -> Path:
        path = self._path_for_episode(episode)
        metadata = metadata or {}
        np.savez(
            path,
            W1=policy.W1,
            b1=policy.b1,
            W2=policy.W2,
            b2=policy.b2,
            episode=np.array([episode], dtype=np.int64),
            metadata_json=np.array([json.dumps(metadata)], dtype=object),
            rng_state_json=np.array([json.dumps(policy.rng.bit_generator.state)], dtype=object),
        )
        return path

    def save_sparse(self, policy: SparseLinearSoftmaxPolicy, episode: int, metadata: dict | None = None) -> Path:
        path = self._path_for_episode(episode)
        metadata = metadata or {}
        np.savez(
            path,
            W1=policy.W1,
            b1=policy.b1,
            W2=policy.W2,
            b2=policy.b2,
            episode=np.array([episode], dtype=np.int64),
            metadata_json=np.array([json.dumps(metadata)], dtype=object),
            rng_state_json=np.array([json.dumps(policy.rng.bit_generator.state)], dtype=object),
        )
        return path

    def latest_path(self) -> Path | None:
        best_ep = -1
        best_path: Path | None = None
        for p in self.dir.glob("policy_ep*.npz"):
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

    def load(self, path: str | Path) -> tuple[LinearSoftmaxPolicy, int]:
        data = np.load(Path(path), allow_pickle=True)
        episode = int(np.asarray(data["episode"]).reshape(-1)[0])

        if all(k in data for k in ("W1", "b1", "W2", "b2")):
            policy = LinearSoftmaxPolicy(
                W1=np.asarray(data["W1"], dtype=np.float64),
                b1=np.asarray(data["b1"], dtype=np.float64),
                W2=np.asarray(data["W2"], dtype=np.float64),
                b2=np.asarray(data["b2"], dtype=np.float64),
            )
        elif all(k in data for k in ("W", "b")):
            policy = LinearSoftmaxPolicy(
                W=np.asarray(data["W"], dtype=np.float64),
                b=np.asarray(data["b"], dtype=np.float64),
            )
        else:
            raise ValueError(f"Checkpoint {path} does not contain supported policy weights")

        if "rng_state_json" in data:
            rng_state_json = str(np.asarray(data["rng_state_json"]).reshape(-1)[0])
            try:
                policy.rng.bit_generator.state = json.loads(rng_state_json)
            except Exception:
                pass

        return policy, episode
