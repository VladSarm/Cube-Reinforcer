"""Core 3x3 Rubik simulator engine."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from .actions import ACTION_NAMES, MOVE_PERMUTATIONS, solved_state
from .solved_check import is_solved_orientation_invariant
from .state_codec import StateValidationError, state_to_json_one_hot, validate_state


class RubikEngine:
    """Thread-safe 3x3 simulator engine with 12 actions."""

    def __init__(self, cube_size: int = 3, initial_state: list[int] | np.ndarray | None = None):
        if cube_size != 3:
            raise ValueError("Only cube_size=3 is supported")

        self.cube_size = cube_size
        self._lock = threading.RLock()
        self._rng = np.random.default_rng()

        self._state = solved_state() if initial_state is None else validate_state(initial_state)
        self.step_count = 0
        self.history: list[int] = []

    def get_state(self) -> np.ndarray:
        """Return internal flat color-id state (length 54)."""
        with self._lock:
            return self._state.copy()

    def get_state_one_hot(self) -> list[list[int]]:
        with self._lock:
            return state_to_json_one_hot(self._state)

    def set_state(self, state: list[int] | np.ndarray) -> np.ndarray:
        arr = validate_state(state)
        with self._lock:
            self._state = arr
            self.step_count = 0
            self.history = []
            return self._state.copy()

    def reset(self, state: list[int] | np.ndarray | None = None) -> np.ndarray:
        with self._lock:
            if state is None:
                self._state = solved_state()
            else:
                self._state = validate_state(state)
            self.step_count = 0
            self.history = []
            return self._state.copy()

    def is_solved(self) -> bool:
        with self._lock:
            return is_solved_orientation_invariant(self._state)

    def step(self, action: int) -> np.ndarray:
        if not isinstance(action, int) or action < 0 or action >= len(ACTION_NAMES):
            raise StateValidationError("Action must be an integer in range 0..11")

        with self._lock:
            self._state = self._state[MOVE_PERMUTATIONS[action]]
            self.step_count += 1
            self.history.append(action)
            return self._state.copy()

    def scramble(self, steps: int, seed: int | None = None) -> tuple[np.ndarray, list[int]]:
        if not isinstance(steps, int) or steps < 0:
            raise StateValidationError("Scramble steps must be a non-negative integer")

        with self._lock:
            rng = np.random.default_rng(seed) if seed is not None else self._rng
            action_list: list[int] = []
            prev_action: int | None = None

            for _ in range(steps):
                all_actions = np.arange(len(ACTION_NAMES), dtype=np.int32)
                if prev_action is not None:
                    inverse_action = prev_action ^ 1
                    candidates = all_actions[all_actions != inverse_action]
                else:
                    candidates = all_actions
                action = int(rng.choice(candidates))
                action_list.append(action)
                prev_action = action

            for action in action_list:
                self._state = self._state[MOVE_PERMUTATIONS[action]]
                self.step_count += 1
                self.history.append(action)
            return self._state.copy(), action_list

    def state_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": state_to_json_one_hot(self._state),
                "step_count": self.step_count,
                "scrambled": not is_solved_orientation_invariant(self._state),
            }
