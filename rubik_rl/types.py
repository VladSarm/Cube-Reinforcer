"""Shared dataclasses for RL pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    state_one_hot: np.ndarray  # shape (54, 6)
    action_history_one_hot: np.ndarray  # shape (48,)
    action: int
    reward: float


@dataclass
class EpisodeResult:
    episode: int
    steps: int
    solved: bool
    total_return: float


@dataclass
class CheckpointData:
    W: np.ndarray
    b: np.ndarray
    episode: int
    rng_state_json: str
