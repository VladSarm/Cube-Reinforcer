"""Sparse-state policy: 7-dim state (cells a..g) + 4*6 action history -> 6 actions (U/L/B only, H fixed)."""

from __future__ import annotations

import numpy as np

from .sparse_state import SPARSE_DIM

ACTION_DIM = 6
HISTORY_LEN = 4
INPUT_DIM = SPARSE_DIM + HISTORY_LEN * ACTION_DIM  # 7 + 24 = 31
HIDDEN_DIM = 128


class SparseLinearSoftmaxPolicy:
    """Two-layer policy for sparse env: x(31) -> linear -> ELU -> linear(128->6) -> softmax."""

    INPUT_DIM = INPUT_DIM
    HIDDEN_DIM = HIDDEN_DIM
    ACTION_DIM = ACTION_DIM

    def __init__(
        self,
        W1: np.ndarray | None = None,
        b1: np.ndarray | None = None,
        W2: np.ndarray | None = None,
        b2: np.ndarray | None = None,
        seed: int | None = None,
        hidden_dim: int | None = None,
        init_scale: float = 0.01,
    ):
        if hidden_dim is not None:
            self.HIDDEN_DIM = int(hidden_dim)
        else:
            self.HIDDEN_DIM = HIDDEN_DIM
        self._init_scale = float(init_scale)
        self.rng = np.random.default_rng(seed)
        if W1 is None:
            W1 = self.rng.normal(0.0, self._init_scale, (self.INPUT_DIM, self.HIDDEN_DIM)).astype(np.float64)
        if b1 is None:
            b1 = np.zeros((self.HIDDEN_DIM,), dtype=np.float64)
        if W2 is None:
            W2 = self.rng.normal(0.0, self._init_scale, (self.HIDDEN_DIM, self.ACTION_DIM)).astype(np.float64)
        if b2 is None:
            b2 = np.zeros((self.ACTION_DIM,), dtype=np.float64)
        self.W1 = np.asarray(W1, dtype=np.float64)
        self.b1 = np.asarray(b1, dtype=np.float64)
        self.W2 = np.asarray(W2, dtype=np.float64)
        self.b2 = np.asarray(b2, dtype=np.float64)
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        if self.W1.shape != (self.INPUT_DIM, self.HIDDEN_DIM):
            raise ValueError(f"W1 must have shape {(self.INPUT_DIM, self.HIDDEN_DIM)}, got {self.W1.shape}")
        if self.b1.shape != (self.HIDDEN_DIM,):
            raise ValueError(f"b1 must have shape {(self.HIDDEN_DIM,)}, got {self.b1.shape}")
        if self.W2.shape != (self.HIDDEN_DIM, self.ACTION_DIM):
            raise ValueError(f"W2 must have shape {(self.HIDDEN_DIM, self.ACTION_DIM)}, got {self.W2.shape}")
        if self.b2.shape != (self.ACTION_DIM,):
            raise ValueError(f"b2 must have shape {(self.ACTION_DIM,)}, got {self.b2.shape}")

    @staticmethod
    def flatten_sparse_state(sparse_state: np.ndarray) -> np.ndarray:
        arr = np.asarray(sparse_state, dtype=np.float64)
        if arr.shape != (SPARSE_DIM,):
            raise ValueError(f"sparse_state must have shape ({SPARSE_DIM},), got {arr.shape}")
        return arr

    @classmethod
    def history_one_hot(cls, action_history: list[int]) -> np.ndarray:
        """Encode last up to 4 actions (0..5) into vector length 24."""
        hist = np.asarray(action_history, dtype=np.int64).reshape(-1)
        if hist.size > HISTORY_LEN:
            hist = hist[-HISTORY_LEN:]
        out = np.zeros((HISTORY_LEN * cls.ACTION_DIM,), dtype=np.float64)
        start_slot = HISTORY_LEN - hist.size
        for i, a in enumerate(hist):
            if 0 <= a < cls.ACTION_DIM:
                slot = start_slot + i
                out[slot * cls.ACTION_DIM + int(a)] = 1.0
        return out

    @classmethod
    def build_observation(
        cls,
        sparse_state: np.ndarray,
        action_history_one_hot: np.ndarray | None = None,
    ) -> np.ndarray:
        s = cls.flatten_sparse_state(sparse_state)
        if action_history_one_hot is None:
            hist = np.zeros((HISTORY_LEN * cls.ACTION_DIM,), dtype=np.float64)
        else:
            hist = np.asarray(action_history_one_hot, dtype=np.float64).reshape(-1)
            if hist.shape != (HISTORY_LEN * cls.ACTION_DIM,):
                raise ValueError(
                    f"action_history_one_hot must have shape ({HISTORY_LEN * cls.ACTION_DIM},), got {hist.shape}"
                )
        return np.concatenate([s, hist], axis=0)

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        exp = np.exp(z)
        return exp / np.sum(exp)

    @staticmethod
    def elu(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0.0, x, np.expm1(x))

    @staticmethod
    def elu_prime(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0.0, 1.0, np.exp(x))

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h_pre = x @ self.W1 + self.b1
        h = self.elu(h_pre)
        logits = h @ self.W2 + self.b2
        probs = self.softmax(logits)
        return h_pre, h, logits, probs

    def action_probs(
        self,
        sparse_state: np.ndarray,
        action_history_one_hot: np.ndarray | None = None,
    ) -> np.ndarray:
        x = self.build_observation(sparse_state, action_history_one_hot)
        _, _, _, probs = self._forward(x)
        return probs

    def sample_action(
        self,
        sparse_state: np.ndarray,
        action_history_one_hot: np.ndarray | None = None,
    ) -> tuple[int, np.ndarray]:
        probs = self.action_probs(sparse_state, action_history_one_hot)
        action = int(self.rng.choice(self.ACTION_DIM, p=probs))
        return action, probs

    def log_policy_gradients(
        self,
        sparse_state: np.ndarray,
        action_history_one_hot: np.ndarray | None,
        action: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if action < 0 or action >= self.ACTION_DIM:
            raise ValueError(f"action must be in range 0..{self.ACTION_DIM - 1}")
        x = self.build_observation(sparse_state, action_history_one_hot)
        h_pre, h, _, probs = self._forward(x)
        delta = -probs.copy()
        delta[action] += 1.0
        dW2 = np.outer(h, delta)
        db2 = delta
        dh = self.W2 @ delta
        dh_pre = dh * self.elu_prime(h_pre)
        dW1 = np.outer(x, dh_pre)
        db1 = dh_pre
        return dW1, db1, dW2, db2

    def apply_gradients(
        self,
        dW1: np.ndarray,
        db1: np.ndarray,
        dW2: np.ndarray,
        db2: np.ndarray,
        lr: float,
    ) -> None:
        self.W1 += lr * np.asarray(dW1, dtype=np.float64)
        self.b1 += lr * np.asarray(db1, dtype=np.float64)
        self.W2 += lr * np.asarray(dW2, dtype=np.float64)
        self.b2 += lr * np.asarray(db2, dtype=np.float64)
