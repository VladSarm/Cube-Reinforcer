"""Numpy-only MLP policy with manual REINFORCE gradients."""

from __future__ import annotations

import numpy as np


class LinearSoftmaxPolicy:
    """Two-layer policy: x(192) -> linear -> ELU -> linear(128->12) -> softmax."""

    INPUT_DIM = 24 * 6 + 4 * 12
    HIDDEN_DIM = 128
    ACTION_DIM = 12

    def __init__(
        self,
        W1: np.ndarray | None = None,
        b1: np.ndarray | None = None,
        W2: np.ndarray | None = None,
        b2: np.ndarray | None = None,
        W: np.ndarray | None = None,
        b: np.ndarray | None = None,
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        if W1 is None and b1 is None and W2 is None and b2 is None and W is not None and b is not None:
            self._init_from_legacy_linear(W=np.asarray(W, dtype=np.float64), b=np.asarray(b, dtype=np.float64))
        else:
            if W1 is None:
                W1 = self.rng.normal(
                    loc=0.0,
                    scale=0.01,
                    size=(self.INPUT_DIM, self.HIDDEN_DIM),
                ).astype(np.float64)
            if b1 is None:
                b1 = np.zeros((self.HIDDEN_DIM,), dtype=np.float64)
            if W2 is None:
                W2 = self.rng.normal(
                    loc=0.0,
                    scale=0.01,
                    size=(self.HIDDEN_DIM, self.ACTION_DIM),
                ).astype(np.float64)
            if b2 is None:
                b2 = np.zeros((self.ACTION_DIM,), dtype=np.float64)

            self.W1 = np.asarray(W1, dtype=np.float64)
            self.b1 = np.asarray(b1, dtype=np.float64)
            self.W2 = np.asarray(W2, dtype=np.float64)
            self.b2 = np.asarray(b2, dtype=np.float64)
        self._validate_shapes()

    def _init_from_legacy_linear(self, W: np.ndarray, b: np.ndarray) -> None:
        # Backward-compatible load for old single-layer checkpoints.
        # 144: state only, 156: state + 1 previous action, 192: state + 4 previous actions.
        if W.ndim != 2 or W.shape[1] != self.ACTION_DIM:
            raise ValueError(
                "Legacy W must have shape (input_dim, 12), "
                f"got {W.shape}"
            )
        if W.shape[0] < self.INPUT_DIM:
            pad_rows = self.INPUT_DIM - W.shape[0]
            W = np.vstack([W, np.zeros((pad_rows, self.ACTION_DIM), dtype=np.float64)])
        if W.shape[0] != self.INPUT_DIM:
            raise ValueError(f"Legacy W must have input_dim {self.INPUT_DIM}, got {W.shape[0]}")
        if b.shape != (self.ACTION_DIM,):
            raise ValueError(f"Legacy b must have shape ({self.ACTION_DIM},), got {b.shape}")

        self.W1 = np.zeros((self.INPUT_DIM, self.HIDDEN_DIM), dtype=np.float64)
        self.b1 = np.zeros((self.HIDDEN_DIM,), dtype=np.float64)
        self.W2 = np.zeros((self.HIDDEN_DIM, self.ACTION_DIM), dtype=np.float64)
        self.b2 = np.zeros((self.ACTION_DIM,), dtype=np.float64)

        self.W1[:, : self.ACTION_DIM] = W
        bias_shift = 10.0
        self.b1[: self.ACTION_DIM] = b + bias_shift
        for a in range(self.ACTION_DIM):
            self.W2[a, a] = 1.0
        self.b2[:] = -bias_shift

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
    def flatten_state_one_hot(state_one_hot: np.ndarray) -> np.ndarray:
        arr = np.asarray(state_one_hot, dtype=np.float64)
        if arr.shape != (24, 6):
            raise ValueError(f"state_one_hot must have shape (24, 6), got {arr.shape}")
        return arr.reshape(-1)

    @staticmethod
    def action_one_hot(action: int) -> np.ndarray:
        if action < 0 or action >= 12:
            raise ValueError("action must be in range 0..11")
        vec = np.zeros((12,), dtype=np.float64)
        vec[action] = 1.0
        return vec

    @classmethod
    def history_one_hot(cls, action_history: list[int]) -> np.ndarray:
        """Encode last up to 4 actions into a concatenated one-hot vector length 48."""
        hist = np.asarray(action_history, dtype=np.int64).reshape(-1)
        if hist.size > 4:
            hist = hist[-4:]
        out = np.zeros((4 * cls.ACTION_DIM,), dtype=np.float64)
        start_slot = 4 - hist.size
        for i, a in enumerate(hist):
            if a < 0 or a >= cls.ACTION_DIM:
                continue
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
            hist = np.zeros((4 * cls.ACTION_DIM,), dtype=np.float64)
        else:
            hist = np.asarray(action_history_one_hot, dtype=np.float64).reshape(-1)
            if hist.shape != (4 * cls.ACTION_DIM,):
                raise ValueError(
                    f"action_history_one_hot must have shape ({4 * cls.ACTION_DIM},), got {hist.shape}"
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
        state_one_hot: np.ndarray,
        action_history_one_hot: np.ndarray | None = None,
    ) -> np.ndarray:
        x = self.build_observation(state_one_hot, action_history_one_hot)
        _, _, _, probs = self._forward(x)
        return probs

    def sample_action(
        self,
        state_one_hot: np.ndarray,
        action_history_one_hot: np.ndarray | None = None,
    ) -> tuple[int, np.ndarray]:
        probs = self.action_probs(state_one_hot, action_history_one_hot)
        action = int(self.rng.choice(self.ACTION_DIM, p=probs))
        return action, probs

    def reinforce_update(
        self,
        state_one_hot: np.ndarray,
        action_history_one_hot: np.ndarray | None,
        action: int,
        advantage: float,
        lr: float,
    ) -> None:
        if action < 0 or action >= self.ACTION_DIM:
            raise ValueError("action must be in range 0..11")

        x = self.build_observation(state_one_hot, action_history_one_hot)
        h_pre, h, _, probs = self._forward(x)

        delta = -probs
        delta[action] += 1.0  # d log pi(a|s) / d logits

        dW2 = np.outer(h, delta)
        db2 = delta
        dh = self.W2 @ delta
        dh_pre = dh * self.elu_prime(h_pre)
        dW1 = np.outer(x, dh_pre)
        db1 = dh_pre

        self.W1 += lr * advantage * dW1
        self.b1 += lr * advantage * db1
        self.W2 += lr * advantage * dW2
        self.b2 += lr * advantage * db2

    def log_policy_gradients(
        self,
        state_one_hot: np.ndarray,
        action_history_one_hot: np.ndarray | None,
        action: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if action < 0 or action >= self.ACTION_DIM:
            raise ValueError("action must be in range 0..11")
        x = self.build_observation(state_one_hot, action_history_one_hot)
        h_pre, h, _, probs = self._forward(x)
        delta = -probs
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

    def get_rng_state_json(self) -> str:
        return str(self.rng.bit_generator.state)
