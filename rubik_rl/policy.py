"""PyTorch policy network for Rubik 2x2 REINFORCE."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


class LinearSoftmaxPolicy(nn.Module):
    """Three-layer policy: x(192) -> 512 -> ELU -> 128 -> ELU -> 12 -> softmax."""

    INPUT_DIM = 24 * 6 + 4 * 12
    HIDDEN_DIM_1 = 512
    HIDDEN_DIM_2 = 128
    ACTION_DIM = 12

    def __init__(
        self,
        hidden_dim_1: int = HIDDEN_DIM_1,
        hidden_dim_2: int = HIDDEN_DIM_2,
        seed: int | None = None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.hidden_dim_1 = int(hidden_dim_1)
        self.hidden_dim_2 = int(hidden_dim_2)
        self.linear1 = nn.Linear(self.INPUT_DIM, self.hidden_dim_1)
        self.linear2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.linear3 = nn.Linear(self.hidden_dim_2, self.ACTION_DIM)
        self.activation = nn.ELU()

    @staticmethod
    def flatten_state_one_hot(state_one_hot: np.ndarray) -> np.ndarray:
        arr = np.asarray(state_one_hot, dtype=np.float32)
        if arr.shape != (24, 6):
            raise ValueError(f"state_one_hot must have shape (24, 6), got {arr.shape}")
        return arr.reshape(-1)

    @staticmethod
    def action_one_hot(action: int) -> np.ndarray:
        if action < 0 or action >= 12:
            raise ValueError("action must be in range 0..11")
        vec = np.zeros((12,), dtype=np.float32)
        vec[action] = 1.0
        return vec

    @classmethod
    def history_one_hot(cls, action_history: list[int]) -> np.ndarray:
        """Encode last up to 4 actions into a concatenated one-hot vector length 48."""
        hist = np.asarray(action_history, dtype=np.int64).reshape(-1)
        if hist.size > 4:
            hist = hist[-4:]
        out = np.zeros((4 * cls.ACTION_DIM,), dtype=np.float32)
        start_slot = 4 - hist.size
        for i, a in enumerate(hist):
            if 0 <= int(a) < cls.ACTION_DIM:
                slot = start_slot + i
                out[slot * cls.ACTION_DIM + int(a)] = 1.0
        return out

    @classmethod
    def build_observation(
        cls,
        state_one_hot: np.ndarray,
        action_history_one_hot: np.ndarray | None = None,
    ) -> np.ndarray:
        s = cls.flatten_state_one_hot(state_one_hot)
        if action_history_one_hot is None:
            hist = np.zeros((4 * cls.ACTION_DIM,), dtype=np.float32)
        else:
            hist = np.asarray(action_history_one_hot, dtype=np.float32).reshape(-1)
            if hist.shape != (4 * cls.ACTION_DIM,):
                raise ValueError(
                    f"action_history_one_hot must have shape ({4 * cls.ACTION_DIM},), got {hist.shape}"
                )
        return np.concatenate([s, hist], axis=0)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.activation(self.linear1(x))
        h2 = self.activation(self.linear2(h1))
        return self.linear3(h2)

    @staticmethod
    def sample_actions_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample batched actions from logits. Returns (actions[B], log_probs[B])."""
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(x)
        return torch.softmax(logits, dim=-1)

    def action_probs(
        self,
        state_one_hot: np.ndarray,
        action_history_one_hot: np.ndarray | None = None,
    ) -> np.ndarray:
        self.eval()
        obs = self.build_observation(state_one_hot, action_history_one_hot)
        device = next(self.parameters()).device
        x = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = self.forward(x).squeeze(0).cpu().numpy()
        return probs.astype(np.float64)

    def sample_action(
        self,
        state_one_hot: np.ndarray,
        action_history_one_hot: np.ndarray | None = None,
        return_log_prob: bool = False,
    ) -> tuple[int, np.ndarray] | tuple[int, np.ndarray, torch.Tensor]:
        obs = self.build_observation(state_one_hot, action_history_one_hot)
        device = next(self.parameters()).device
        x = torch.from_numpy(obs).unsqueeze(0).to(device)
        logits = self.forward_logits(x).squeeze(0)
        dist = torch.distributions.Categorical(logits=logits)
        action_t = dist.sample()
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float64)
        action = int(action_t.item())
        if return_log_prob:
            return action, probs, dist.log_prob(action_t)
        return action, probs

    def get_rng_state_json(self) -> str:
        # Kept for backward compatibility with old checkpoint metadata path.
        return "{}"

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> "LinearSoftmaxPolicy":
        # Infer hidden dims from checkpoint.
        w1 = state_dict.get("linear1.weight")
        w2 = state_dict.get("linear2.weight")
        hidden_dim_1 = int(w1.shape[0]) if w1 is not None else cls.HIDDEN_DIM_1
        hidden_dim_2 = int(w2.shape[0]) if w2 is not None else cls.HIDDEN_DIM_2
        policy = cls(hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2)
        policy.load_state_dict(state_dict)
        return policy
