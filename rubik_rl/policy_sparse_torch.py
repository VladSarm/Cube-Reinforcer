"""PyTorch policy for sparse Rubik 2x2: 7-dim state + 4*6 action history -> 6 actions."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .sparse_state import SPARSE_DIM

ACTION_DIM = 6
HISTORY_LEN = 4
INPUT_DIM = SPARSE_DIM + HISTORY_LEN * ACTION_DIM  # 7 + 24 = 31
HIDDEN_DIM_1 = 512
HIDDEN_DIM_2 = 128


class SparsePolicyTorch(nn.Module):
    """Three-layer policy for sparse env: x(31) -> 512 -> ELU -> 128 -> ELU -> 6."""

    INPUT_DIM = INPUT_DIM
    HIDDEN_DIM = HIDDEN_DIM_1  # kept for compat (first hidden layer size)
    ACTION_DIM = ACTION_DIM

    def __init__(self, hidden_dim: int = HIDDEN_DIM_1, seed: int | None = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.hidden_dim = int(hidden_dim)
        self.linear1 = nn.Linear(self.INPUT_DIM, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, HIDDEN_DIM_2)
        self.linear3 = nn.Linear(HIDDEN_DIM_2, self.ACTION_DIM)
        self.activation = nn.ELU()

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 31] -> logits: [B, 6]."""
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        return self.linear3(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward_logits(x), dim=-1)

    @staticmethod
    def sample_actions_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from [B, 6] logits. Returns (actions[B], log_probs[B])."""
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        return actions, dist.log_prob(actions)

    def action_probs(
        self,
        sparse_state: np.ndarray,
        action_history_one_hot: np.ndarray | None = None,
    ) -> np.ndarray:
        self.eval()
        obs = self._build_obs_numpy(sparse_state, action_history_one_hot)
        device = next(self.parameters()).device
        x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = self.forward(x).squeeze(0).cpu().numpy()
        return probs.astype(np.float64)

    def sample_action(
        self,
        sparse_state: np.ndarray,
        action_history_one_hot: np.ndarray | None = None,
    ) -> tuple[int, np.ndarray]:
        obs = self._build_obs_numpy(sparse_state, action_history_one_hot)
        device = next(self.parameters()).device
        x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
        logits = self.forward_logits(x).squeeze(0)
        dist = torch.distributions.Categorical(logits=logits)
        action = int(dist.sample().item())
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float64)
        return action, probs

    @classmethod
    def history_one_hot(cls, action_history: list[int]) -> np.ndarray:
        """Encode last up to 4 actions (0..5) into a vector of length 24."""
        hist = np.asarray(action_history, dtype=np.int64).reshape(-1)
        if hist.size > HISTORY_LEN:
            hist = hist[-HISTORY_LEN:]
        out = np.zeros((HISTORY_LEN * ACTION_DIM,), dtype=np.float32)
        start_slot = HISTORY_LEN - hist.size
        for i, a in enumerate(hist):
            if 0 <= a < ACTION_DIM:
                out[(start_slot + i) * ACTION_DIM + int(a)] = 1.0
        return out

    @staticmethod
    def _build_obs_numpy(
        sparse_state: np.ndarray,
        action_history_one_hot: np.ndarray | None,
    ) -> np.ndarray:
        s = np.asarray(sparse_state, dtype=np.float32).reshape(-1)
        if action_history_one_hot is None:
            hist = np.zeros((HISTORY_LEN * ACTION_DIM,), dtype=np.float32)
        else:
            hist = np.asarray(action_history_one_hot, dtype=np.float32).reshape(-1)
        return np.concatenate([s, hist], axis=0)

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> "SparsePolicyTorch":
        w1 = state_dict.get("linear1.weight")
        hidden_dim = int(w1.shape[0]) if w1 is not None else cls.HIDDEN_DIM
        policy = cls(hidden_dim=hidden_dim)
        policy.load_state_dict(state_dict)
        return policy
